//! A synchronous decode round over independent M5 Fast/Half sessions.
//!
//! Dense Q6/Q8 projections reuse dequantized weight fragments through the
//! F32-weight few-row MMA. Routed experts keep F32 operands and use the native
//! token grid. KDA recurrence, latent caches and sparse selection are always
//! per sequence. The first member lends its admitted packed activation
//! scratch for this call; no state or scratch escapes across requests.

use super::*;
use crate::metal::{GpuTokenSampler, encode_copy_offset_f32};
use crate::sampling::{Sampler, SamplingConfig};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLResource, MTLSize};
use std::time::Instant;

#[path = "batch_projection.rs"]
mod projection;
#[path = "batch_selection.rs"]
mod selection;

#[cfg(feature = "perf-ablations")]
pub fn with_batch_waves<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    projection::with_batch_waves(enabled, f)
}
#[cfg(feature = "perf-ablations")]
pub fn with_batch_attention<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    projection::with_batch_attention(enabled, f)
}
#[cfg(feature = "perf-ablations")]
pub fn with_batch_index<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    projection::with_batch_index(enabled, f)
}
#[cfg(feature = "perf-ablations")]
/// Encode and GPU-span milliseconds from the most recent batched step.
pub fn take_last_step_timing() -> Option<(f64, f64)> {
    projection::take_timing()
}

crate::env_flag!(
    /// Default-on four-output-channel IQ2_S gate/up and IQ3_S down grids for qualified M5 Fast/Half independent decode batches of two to four members. Zero restores the original F32 token grids; single-token, Exact/F32 and other primitive policies are unchanged.
    default_on batch_expert_rows_enabled,
    "QWEN_GLM53_BATCH_EXPERT_ROWS"
);

/// Largest qualified independent decode batch.
pub const GLM5_NEXT_MAX_BATCH_SIZE: usize = 4;

/// Admitted output storage for independent decode batches. It is bound to one
/// loaded weight instance, and can be reused as the active width changes.
///
/// M5 Fast/Half only for B > 1: few-row MMA keeps F32 weights/activations
/// and changes the reduction order. Exact and Fast/F32
/// members are refused before mutation and must use ordinary `forward`.
/// B = 1 always calls ordinary `forward`, including its mode contracts.
#[derive(Clone, Debug, Default)]
pub struct GlmBatchLoopStats {
    pub sampled_tokens: usize,
    pub forwarded_tokens: usize,
    pub emitted_tokens: usize,
    pub encode_cpu_ms: f64,
    pub completion_wait_ms: f64,
    pub gpu_ms: f64,
    pub wall_ms: f64,
    pub stopped_on_eos: usize,
    pub terminal_tokens: Vec<i32>,
}

struct LoopInFlight(Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>);
impl Drop for LoopInFlight {
    fn drop(&mut self) {
        if let Some(command) = self.0.take()
            && command.status() != objc2_metal::MTLCommandBufferStatus::NotEnqueued
        {
            crate::metal::wait_unchecked(&command);
        }
    }
}

pub struct Glm5NextBatch {
    logits: MetalTensor,
    weights_instance: u64,
    max_members: usize,
    expert_rows: bool,
    gpu_samplers: Vec<Option<(SamplingConfig, GpuTokenSampler)>>,
    tokens: Vec<MetalTensor>,
}

fn projection_geometry(dtype: GgmlType, shape: &[u64]) -> bool {
    matches!(dtype, GgmlType::Q6_K | GgmlType::Q8_0)
        && shape.len() == 2
        && shape[0] > 0
        && shape[1] > 0
        && shape[0].is_multiple_of(if dtype == GgmlType::Q8_0 { 64 } else { 256 })
}
fn projection_supported(w: &MetalTensor) -> bool {
    projection_geometry(w.dtype, &w.shape)
}
fn batch_mode(lineage: Option<PackedLineage>, precision: Option<FastPrecision>) -> bool {
    lineage == Some(PackedLineage::Fast) && precision == Some(FastPrecision::Half)
}

impl Glm5NextBatch {
    /// Future CPU readback/result allowance to include in concurrent admissions.
    pub fn cpu_reserve_bytes(&self) -> u64 {
        self.logits.n_bytes() * 2
    }
    /// Metadata-only eligibility; does not allocate or compile a pipeline.
    pub fn supported(ctx: &MetalContext, weights: &Glm5NextWeights) -> bool {
        if !ctx.device_facts().gpu_families.apple10
            || !weights.packed_prefill_admitted
            || !projection_supported(&weights.output)
        {
            return false;
        }
        weights.blocks.iter().all(|b| {
            if b.attention_hc.mix.dtype != GgmlType::Q8_0 || b.ffn_hc.mix.dtype != GgmlType::Q8_0 {
                return false;
            }
            let mixer = match &b.mixer {
                MixerTensors::Kda(k) => [
                    &k.query, &k.key, &k.value, &k.decay_a, &k.decay_b, &k.beta, &k.gate_a,
                    &k.gate_b, &k.output,
                ]
                .into_iter()
                .all(projection_supported),
                MixerTensors::Mla(m) => [
                    &m.query_a,
                    &m.query_b,
                    &m.latent,
                    &m.indexer.key,
                    &m.indexer.pool_gate,
                    &m.output,
                ]
                .into_iter()
                .all(projection_supported),
            };
            mixer
                && match &b.ffn {
                    FfnTensors::Dense(d) => [&d.gate, &d.up, &d.down]
                        .into_iter()
                        .all(projection_supported),
                    FfnTensors::Moe(m) => {
                        m.router.dtype == GgmlType::F32
                            && [&m.shared.gate, &m.shared.up, &m.shared.down]
                                .into_iter()
                                .all(projection_supported)
                            && crate::metal::all_slots_gate_up_native_supported(
                                m.gate_experts.dtype,
                                m.up_experts.dtype,
                            )
                            && crate::metal::all_slots_down_native_supported(m.down_experts.dtype)
                    }
                }
        })
    }

    /// Admits and allocates only B-wide logits; activations borrow the first
    /// member's existing packed scratch. CPU allowance includes readback and
    /// the returned per-member vectors. Refusal leaves every session untouched.
    pub fn new(ctx: &MetalContext, weights: &Glm5NextWeights, max_members: usize) -> Result<Self> {
        if !(1..=GLM5_NEXT_MAX_BATCH_SIZE).contains(&max_members) || !Self::supported(ctx, weights)
        {
            return invalid("independent batching needs supported M5 weights and 1..=4 members");
        }
        let bytes = (weights.config.vocab_size as u64)
            .checked_mul(max_members as u64)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Glm5NextMetalError::Invalid("batch output size overflow".into()))?;
        let priced = device_price(ctx)(bytes)
            .ok_or_else(|| Glm5NextMetalError::Invalid("batch output price overflow".into()))?;
        let _allocation = ctx.begin_allocation_transaction();
        let admission = evaluate_metal_memory_admission_with_cpu_bytes(
            priced,
            bytes * 2,
            memory::DYNAMIC_RESERVE_BYTES,
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
        Ok(Self {
            logits: MetalTensor::zeros_f32(
                ctx,
                vec![weights.config.vocab_size as u64, max_members as u64],
            )?,
            weights_instance: weights.instance,
            max_members,
            expert_rows: batch_expert_rows_enabled(),
            gpu_samplers: (0..max_members).map(|_| None).collect(),
            tokens: (0..max_members)
                .map(|_| MetalTensor::zeros_i32(ctx, vec![1]))
                .collect::<std::result::Result<Vec<_>, MetalError>>()?,
        })
    }

    /// One token per member, in the supplied order. Members may have different
    /// positions. Invalid tokens, capacity, weights, modes and workspace are
    /// checked for the whole group before mutation. After encoding begins, an
    /// error or unwind leaves all members poisoned; there is no partial retry.
    /// Callers observe per-request cancellation outside this synchronous call.
    pub fn step(
        &mut self,
        ctx: &MetalContext,
        sessions: &mut [&mut Glm5NextSession<'_>],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        self.step_inner(ctx, sessions, tokens, true)
    }

    /// Forward the batch and sample every member on the GPU, reading back only
    /// tokens. This is the non-pipelined arm; the one-ahead loop builds on the
    /// same per-member sampler cache and token slots.
    pub fn step_tokens(
        &mut self,
        ctx: &MetalContext,
        sessions: &mut [&mut Glm5NextSession<'_>],
        samplers: &mut [&mut Sampler],
        tokens: &[u32],
    ) -> Result<Vec<u32>> {
        let count = sessions.len();
        if count < 2 || count != samplers.len() || count != tokens.len() || count > self.max_members
        {
            return invalid("invalid independent sampling batch width");
        }
        let vocab = sessions[0].weights.config.vocab_size as usize;
        for (i, sampler) in samplers.iter().enumerate() {
            let config = sampler.config();
            let rebuild = match &self.gpu_samplers[i] {
                Some((cached, _)) => *cached != config,
                None => true,
            };
            if rebuild {
                self.gpu_samplers[i] = Some((config, GpuTokenSampler::new(ctx, vocab, config)?));
            }
        }
        let uniforms: Vec<f32> = samplers.iter().map(|s| s.gpu_uniform(0)).collect();
        let result = (|| -> Result<Vec<u32>> {
            self.step_inner(ctx, sessions, tokens, false)?;
            let command = ctx
                .queue
                .commandBuffer()
                .ok_or_else(|| Glm5NextMetalError::Invalid("no sampling command buffer".into()))?;
            let enc = KernelEncoder::begin(&command);
            for i in 0..count {
                self.gpu_samplers[i]
                    .as_ref()
                    .expect("GPU sampler")
                    .1
                    .encode(
                        ctx,
                        &enc,
                        &row(&self.logits, i, vocab),
                        &self.tokens[i],
                        uniforms[i],
                    )?;
            }
            enc.end();
            command.commit();
            wait_completed(&command)?;
            let mut out = Vec::with_capacity(count);
            for token in self.tokens.iter().take(count) {
                let value = read_i32(token)?[0];
                if value < 0 || value as usize >= vocab {
                    return invalid(format!("GPU sampler selected invalid token {value}"));
                }
                out.push(value as u32);
            }
            Ok(out)
        })();
        match result {
            Ok(out) => {
                for sampler in samplers.iter_mut() {
                    sampler.commit_gpu_draw();
                }
                Ok(out)
            }
            Err(error) => {
                for s in sessions.iter_mut() {
                    s.poisoned = true;
                }
                Err(error)
            }
        }
    }

    fn step_inner(
        &mut self,
        ctx: &MetalContext,
        sessions: &mut [&mut Glm5NextSession<'_>],
        tokens: &[u32],
        read_logits: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let count = sessions.len();
        if count == 0 || count != tokens.len() || count > self.max_members {
            return invalid("invalid independent batch width");
        }
        for (s, t) in sessions.iter().zip(tokens) {
            if s.weights.instance != self.weights_instance {
                return invalid("independent batch weights differ");
            }
            if s.poisoned {
                return Err(Glm5NextMetalError::Poisoned);
            }
            if *t >= s.weights.config.vocab_size || s.position >= s.capacity {
                return invalid("invalid independent batch member token/capacity");
            }
        }
        if count == 1 {
            return Ok(vec![sessions[0].forward(ctx, tokens[0])?]);
        }
        if sessions
            .iter()
            .any(|s| !batch_mode(s.packed_lineage(), s.fast_precision()))
        {
            return invalid("independent batch needs Fast/Half members");
        }
        if sessions[0].packed.as_ref().is_none_or(|p| p.rows < count) {
            return invalid("independent batch activation workspace too small");
        }
        if self.logits.buffer.device().registryID() != ctx.device.registryID() {
            return invalid("independent batch device differs");
        }
        let selection_stride =
            selection::stride(ctx, &sessions.iter().map(|s| &**s).collect::<Vec<_>>());
        for s in sessions.iter_mut() {
            s.poisoned = true;
        }
        for s in sessions.iter() {
            let c = &s.weights.config;
            let visible = s.position + 1;
            if visible >= c.sparse_frontier() as usize {
                let sp = s.sparse.as_ref().ok_or_else(|| {
                    Glm5NextMetalError::Invalid("independent sparse member has no scratch".into())
                })?;
                write_i32(
                    &sp.visible_pools,
                    &[(visible / c.indexer_pool as usize) as i32],
                )?;
                write_i32(&sp.visible_rows, &[visible as i32])?;
                write_i32(&sp.select_status, &vec![-1; c.block_count(MixerKind::Mla)])?;
            }
        }
        let refs: Vec<_> = sessions.iter().map(|s| &**s).collect();
        if selection_stride.is_some() {
            selection::prepare(&refs)?;
        }
        refs[0].encode_independent_batch(
            ctx,
            &refs,
            Some(tokens),
            None,
            &self.logits,
            self.expert_rows,
            selection_stride,
            false,
            true,
        )?;
        let p = refs[0].packed.as_ref().expect("preflighted");
        for (block, r) in p.routes.iter().enumerate() {
            if let Some(r) = r {
                let statuses = read_i32(&rows_view(&r.status, count))?;
                if let Some((row, &status)) = statuses
                    .iter()
                    .enumerate()
                    .find(|(_, s)| **s != ROUTE_STATUS_READY)
                {
                    return Err(Glm5NextMetalError::KernelValidation {
                        stage: "independent route",
                        block,
                        row: Some(row),
                        status,
                    });
                }
            }
        }
        for (row, s) in refs.iter().enumerate() {
            if selection_stride.is_none()
                && s.position + 1 >= s.weights.config.sparse_frontier() as usize
            {
                let statuses = read_i32(&s.sparse.as_ref().expect("preflighted").select_status)?;
                if let Some((index, &status)) = statuses
                    .iter()
                    .enumerate()
                    .find(|(_, s)| **s != crate::metal::SELECT_STATUS_OK)
                {
                    return Err(Glm5NextMetalError::KernelValidation {
                        stage: "independent sparse selection",
                        block: mla_block(&s.weights.config, index),
                        row: Some(row),
                        status,
                    });
                }
            }
        }
        if selection_stride.is_some() {
            selection::validate(&refs)?;
        }
        let vocab = refs[0].weights.config.vocab_size as usize;
        let result = if read_logits {
            let values = (0..count)
                .map(|r| read_f32(&row(&self.logits, r, vocab)))
                .collect::<Result<Vec<_>>>()?;
            if values.iter().flatten().any(|x| !x.is_finite()) {
                return invalid("non-finite independent batch logits");
            }
            values
        } else {
            Vec::new()
        };
        for s in sessions.iter_mut() {
            s.position += 1;
            s.poisoned = false;
        }
        Ok(result)
    }

    fn cache_loop_samplers(
        &mut self,
        ctx: &MetalContext,
        sessions: &[&Glm5NextSession<'_>],
        samplers: &mut [&mut Sampler],
    ) -> Result<usize> {
        let count = sessions.len();
        let vocab = sessions[0].weights.config.vocab_size as usize;
        for (i, sampler) in samplers.iter().enumerate() {
            let config = sampler.config();
            let rebuild = match &self.gpu_samplers[i] {
                Some((cached, _)) => *cached != config,
                None => true,
            };
            if rebuild {
                self.gpu_samplers[i] = Some((config, GpuTokenSampler::new(ctx, vocab, config)?));
            }
        }
        Ok(vocab)
    }

    fn prepare_loop_frame(
        &self,
        ctx: &MetalContext,
        sessions: &mut [&mut Glm5NextSession<'_>],
        positions: &[usize],
        host_tokens: Option<&[u32]>,
        uniforms: &[f32],
        selection_stride: Option<usize>,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let count = sessions.len();
        let vocab = sessions[0].weights.config.vocab_size as usize;
        for (s, &p) in sessions.iter_mut().zip(positions) {
            s.position = p;
        }
        let refs: Vec<_> = sessions.iter().map(|s| &**s).collect();
        let command = refs[0]
            .encode_independent_batch(
                ctx,
                &refs,
                host_tokens,
                if host_tokens.is_none() {
                    Some(&self.tokens[..count])
                } else {
                    None
                },
                &self.logits,
                self.expert_rows,
                selection_stride,
                true,
                false,
            )?
            .expect("uncommitted batch command");
        let enc = KernelEncoder::begin(&command);
        for i in 0..count {
            self.gpu_samplers[i]
                .as_ref()
                .expect("GPU sampler")
                .1
                .encode(
                    ctx,
                    &enc,
                    &row(&self.logits, i, vocab),
                    &self.tokens[i],
                    uniforms[i],
                )?;
        }
        enc.end();
        Ok(command)
    }

    fn validate_loop_frame(
        &self,
        sessions: &[&Glm5NextSession<'_>],
        frame_positions: &[usize],
        selection_stride: Option<usize>,
    ) -> Result<()> {
        let count = sessions.len();
        let p = sessions[0].packed.as_ref().expect("preflighted");
        for (block, r) in p.routes.iter().enumerate() {
            if let Some(r) = r {
                let statuses = read_i32(&rows_view(&r.status, count))?;
                if let Some((row, &status)) = statuses
                    .iter()
                    .enumerate()
                    .find(|(_, s)| **s != ROUTE_STATUS_READY)
                {
                    return Err(Glm5NextMetalError::KernelValidation {
                        stage: "independent route",
                        block,
                        row: Some(row),
                        status,
                    });
                }
            }
        }
        for (row, s) in sessions.iter().enumerate() {
            if selection_stride.is_none()
                && frame_positions[row] + 1 >= s.weights.config.sparse_frontier() as usize
            {
                let statuses = read_i32(&s.sparse.as_ref().expect("preflighted").select_status)?;
                if let Some((index, &status)) = statuses
                    .iter()
                    .enumerate()
                    .find(|(_, s)| **s != crate::metal::SELECT_STATUS_OK)
                {
                    return Err(Glm5NextMetalError::KernelValidation {
                        stage: "independent sparse selection",
                        block: mla_block(&s.weights.config, index),
                        row: Some(row),
                        status,
                    });
                }
            }
        }
        if selection_stride.is_some() {
            selection::validate(sessions)?;
        }
        Ok(())
    }

    /// One-ahead token loop over a fixed independent member set. Frame 0
    /// forwards the caller's pending tokens; later frames read each member's
    /// previous token from GPU storage and append that member's sampler to the
    /// same command. The next command is encoded while the current executes.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_gpu_tokens(
        &mut self,
        ctx: &MetalContext,
        sessions: &mut [&mut Glm5NextSession<'_>],
        samplers: &mut [&mut Sampler],
        pending: &[u32],
        max_tokens: usize,
        eos: &[&[i32]],
        mut emit: impl FnMut(usize, u32) -> bool,
    ) -> Result<GlmBatchLoopStats> {
        let count = sessions.len();
        if count < 2
            || count > self.max_members
            || max_tokens == 0
            || samplers.len() != count
            || pending.len() != count
            || eos.len() != count
        {
            return invalid("invalid GLM batch loop extents");
        }
        for (s, &t) in sessions.iter().zip(pending) {
            if s.weights.instance != self.weights_instance
                || s.poisoned
                || t as usize >= s.weights.config.vocab_size as usize
                || s.position >= s.capacity
                || max_tokens > s.capacity - s.position
            {
                return invalid("GLM batch member is not ready for a token loop");
            }
        }
        if sessions
            .iter()
            .any(|s| !batch_mode(s.packed_lineage(), s.fast_precision()))
        {
            return invalid("independent batch needs Fast/Half members");
        }
        if sessions[0].packed.as_ref().is_none_or(|p| p.rows < count) {
            return invalid("independent batch activation workspace too small");
        }
        if self.logits.buffer.device().registryID() != ctx.device.registryID() {
            return invalid("independent batch device differs");
        }
        let starts: Vec<usize> = sessions.iter().map(|s| s.position).collect();
        let selection_stride =
            selection::stride(ctx, &sessions.iter().map(|s| &**s).collect::<Vec<_>>());
        self.cache_loop_samplers(
            ctx,
            &sessions.iter().map(|s| &**s).collect::<Vec<_>>(),
            samplers,
        )?;
        for s in sessions.iter_mut() {
            s.poisoned = true;
        }
        let _qos = crate::metal::GpuThreadQos::enter();
        let started = Instant::now();
        let mut stats = GlmBatchLoopStats::default();
        let u0: Vec<f32> = samplers.iter().map(|s| s.gpu_uniform(0)).collect();
        let positions0: Vec<usize> = starts.clone();
        let mut current = self.prepare_loop_frame(
            ctx,
            sessions,
            &positions0,
            Some(pending),
            &u0,
            selection_stride,
        )?;
        let mut running = LoopInFlight(Some(current.clone()));
        let t = Instant::now();
        current.commit();
        stats.encode_cpu_ms += t.elapsed().as_secs_f64() * 1e3;
        let mut sampled = 0usize;
        let vocab = sessions[0].weights.config.vocab_size as usize;
        let mut terminal = vec![-1i32; count];
        let mut stop = false;
        loop {
            let next = if sampled + 1 < max_tokens {
                let positions: Vec<usize> = starts.iter().map(|s| s + sampled + 1).collect();
                let uniforms: Vec<f32> = samplers.iter().map(|s| s.gpu_uniform(1)).collect();
                let t = Instant::now();
                let command = self.prepare_loop_frame(
                    ctx,
                    sessions,
                    &positions,
                    None,
                    &uniforms,
                    selection_stride,
                )?;
                stats.encode_cpu_ms += t.elapsed().as_secs_f64() * 1e3;
                Some(command)
            } else {
                None
            };
            let frame_positions: Vec<usize> = starts.iter().map(|s| s + sampled).collect();
            let t = Instant::now();
            wait_completed(&current)?;
            stats.completion_wait_ms += t.elapsed().as_secs_f64() * 1e3;
            let elapsed = (current.GPUEndTime() - current.GPUStartTime()) * 1e3;
            if elapsed.is_finite() && elapsed > 0. {
                stats.gpu_ms += elapsed;
            }
            let mut tokens = Vec::with_capacity(count);
            for token in self.tokens.iter().take(count) {
                let value = read_i32(token)?[0];
                if value < 0 || value as usize >= vocab {
                    return invalid(format!("GPU sampler selected invalid token {value}"));
                }
                tokens.push(value as u32);
            }
            self.validate_loop_frame(
                &sessions.iter().map(|s| &**s).collect::<Vec<_>>(),
                &frame_positions,
                selection_stride,
            )?;
            running.0 = None;
            for i in 0..count {
                let token = tokens[i];
                terminal[i] = token as i32;
                if eos[i].contains(&terminal[i]) {
                    stats.stopped_on_eos += 1;
                    stop = true;
                } else {
                    stats.emitted_tokens += 1;
                    if !emit(i, token) {
                        stop = true;
                    }
                }
            }
            sampled += 1;
            stats.sampled_tokens += count;
            for sampler in samplers.iter_mut() {
                sampler.commit_gpu_draw();
            }
            if stop || sampled >= max_tokens {
                break;
            }
            let mut next = next.expect("a continuing burst prepared a frame");
            let t = Instant::now();
            next.commit();
            stats.encode_cpu_ms += t.elapsed().as_secs_f64() * 1e3;
            running.0 = Some(next.clone());
            current = next;
        }
        for (s, &start) in sessions.iter_mut().zip(&starts) {
            s.position = start + sampled;
            s.poisoned = false;
        }
        stats.forwarded_tokens = sampled * count;
        stats.terminal_tokens = terminal;
        stats.wall_ms = started.elapsed().as_secs_f64() * 1e3;
        Ok(stats)
    }
}

fn matmat(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    arith: Arith,
    stage: Stage,
    w: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    rows: usize,
) -> Result<()> {
    if projection::matmat(ctx, enc, w, x, y, k, m, rows)? {
        return Ok(());
    }
    if matches!(w.dtype, GgmlType::Q6_K | GgmlType::Q8_0)
        && k.is_multiple_of(if w.dtype == GgmlType::Q8_0 { 64 } else { 256 })
    {
        crate::metal::encode_mat_mat_fewrow_gen_f32_weights(ctx, enc, w, x, y, k, m, rows)?;
        return Ok(());
    }
    super::matmat(ctx, enc, arith, stage, w, x, y, k, m, rows)
}
fn row(t: &MetalTensor, r: usize, n: usize) -> MetalTensor {
    t.view_subrange((r * n) as u64, vec![n as u64])
}
fn copy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    dst: &MetalTensor,
    n: usize,
) -> Result<()> {
    encode_copy_offset_f32(ctx, enc, src, 0, dst, n)?;
    Ok(())
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GlmTokenRowsArgs {
    rows: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GlmSparseArgs {
    visible: [u32; 4],
    pool: u32,
    rows: u32,
    blocks: u32,
    stride: u32,
}
fn encode_glm_token_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    tokens: &[MetalTensor],
    dst: &MetalTensor,
    rows: usize,
) -> Result<()> {
    enc.set_pipeline(&ctx.pipeline("kernel_glm_token_rows")?);
    enc.set_bytes(
        5,
        &GlmTokenRowsArgs {
            rows: rows as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        },
    );
    for i in 0..4 {
        let t = &tokens[i.min(tokens.len().saturating_sub(1))];
        enc.set_tensor(i, t);
    }
    enc.set_tensor(4, dst);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 4,
            height: 1,
            depth: 1,
        },
    );
    enc.buffer_barrier();
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn encode_glm_sparse_member(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    pools: &MetalTensor,
    visible: &MetalTensor,
    status: &MetalTensor,
    visible_position: usize,
    pool: usize,
    blocks: usize,
) -> Result<()> {
    enc.set_pipeline(&ctx.pipeline("kernel_glm_sparse_member")?);
    let mut args = GlmSparseArgs {
        visible: [0; 4],
        pool: pool as u32,
        rows: 1,
        blocks: blocks as u32,
        stride: blocks as u32,
    };
    args.visible[0] = visible_position as u32;
    enc.set_bytes(3, &args);
    enc.set_tensor(0, pools);
    enc.set_tensor(1, visible);
    enc.set_tensor(2, status);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: blocks.max(1),
            height: 1,
            depth: 1,
        },
    );
    enc.buffer_barrier();
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn encode_glm_sparse_shared(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    pools: &MetalTensor,
    visible: &MetalTensor,
    status: &MetalTensor,
    visible_positions: &[usize],
    pool: usize,
    blocks: usize,
    rows: usize,
    stride: usize,
) -> Result<()> {
    enc.set_pipeline(&ctx.pipeline("kernel_glm_sparse_shared")?);
    let mut args = GlmSparseArgs {
        visible: [0; 4],
        pool: pool as u32,
        rows: rows as u32,
        blocks: blocks as u32,
        stride: stride as u32,
    };
    for (i, &v) in visible_positions.iter().take(4).enumerate() {
        args.visible[i] = v as u32;
    }
    enc.set_bytes(3, &args);
    enc.set_tensor(0, pools);
    enc.set_tensor(1, visible);
    enc.set_tensor(2, status);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: (rows * blocks).max(rows).max(1),
            height: 1,
            depth: 1,
        },
    );
    enc.buffer_barrier();
    Ok(())
}

impl Glm5NextSession<'_> {
    #[allow(clippy::too_many_arguments)]
    fn encode_independent_batch(
        &self,
        ctx: &MetalContext,
        seqs: &[&Self],
        tokens: Option<&[u32]>,
        device_tokens: Option<&[MetalTensor]>,
        logits: &MetalTensor,
        expert_rows: bool,
        selection_stride: Option<usize>,
        device_round: bool,
        commit: bool,
    ) -> Result<Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>> {
        let w = self.weights;
        let c = &w.config;
        let p = self.packed.as_ref().unwrap();
        let rows = seqs.len();
        let h = c.hidden_size as usize;
        assert!(rows <= p.rows);
        if let Some(tokens) = tokens {
            write_i32(
                &rows_view(&p.token, rows),
                &tokens.iter().map(|t| *t as i32).collect::<Vec<_>>(),
            )?;
        }
        let v = |t: &MetalTensor| rows_view(t, rows);
        #[cfg(any(test, feature = "perf-ablations"))]
        let encode_started = std::time::Instant::now();
        let command = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| Glm5NextMetalError::Invalid("no command buffer".into()))?;
        let waves = projection::waves() && seqs.iter().all(|s| s.stage_recorder.borrow().is_none());
        let mut enc = if waves {
            KernelEncoder::begin_causal(&command)
        } else {
            KernelEncoder::begin(&command)
        };
        if let Some(device_tokens) = device_tokens {
            encode_glm_token_rows(ctx, &enc, device_tokens, &v(&p.token), rows)?;
        }
        if device_round {
            let blocks = c.block_count(MixerKind::Mla);
            let pool = c.indexer_pool as usize;
            if selection_stride.is_some() {
                let sp = p.sparse.as_ref().ok_or_else(|| {
                    Glm5NextMetalError::Invalid("independent sparse batch has no scratch".into())
                })?;
                let visibles: Vec<usize> = seqs.iter().map(|s| s.position + 1).collect();
                encode_glm_sparse_shared(
                    ctx,
                    &enc,
                    &sp.visible_pools,
                    &sp.visible_rows,
                    &sp.select_status,
                    &visibles,
                    pool,
                    blocks,
                    rows,
                    sp.rows,
                )?;
            } else {
                for s in seqs {
                    let visible_after = s.position + 1;
                    if visible_after >= c.sparse_frontier() as usize {
                        let sp = s.sparse.as_ref().ok_or_else(|| {
                            Glm5NextMetalError::Invalid(
                                "independent sparse member has no scratch".into(),
                            )
                        })?;
                        encode_glm_sparse_member(
                            ctx,
                            &enc,
                            &sp.visible_pools,
                            &sp.visible_rows,
                            &sp.select_status,
                            visible_after,
                            pool,
                            blocks,
                        )?;
                    }
                }
            }
        }
        encode_get_rows_f32(
            ctx,
            &enc,
            &w.embedding,
            &v(&p.token),
            &v(&p.embedding),
            rows,
            h,
        )?;
        encode_mhc4_repeat_rows(ctx, &enc, h, rows, &v(&p.embedding), &v(&p.residual[0]))?;
        let mut mla_index = 0;
        for (index, block) in w.blocks.iter().enumerate() {
            let (a, b) = (v(&p.residual[0]), v(&p.residual[1]));
            self.encode_hc_pre_rows(
                ctx,
                &enc,
                rows,
                &a,
                &block.attention_hc,
                &block.attention_norm,
            )?;
            match &block.mixer {
                MixerTensors::Kda(kda) => self.encode_batch_kda(ctx, &enc, seqs, index, kda)?,
                MixerTensors::Mla(mla) => {
                    self.encode_batch_mla(
                        ctx,
                        &mut enc,
                        seqs,
                        index,
                        mla_index,
                        mla,
                        selection_stride,
                    )?;
                    mla_index += 1;
                }
            }
            encode_mhc4_post_rows(
                ctx,
                &enc,
                h,
                rows,
                &v(&p.block_out),
                &a,
                &v(&p.post),
                &v(&p.comb),
                &b,
            )?;
            self.encode_hc_pre_rows(ctx, &enc, rows, &b, &block.ffn_hc, &block.ffn_norm)?;
            match &block.ffn {
                FfnTensors::Dense(dense) => {
                    self.stage(&mut enc, Glm5NextStage::DenseFfn, Some(index))?;
                    let f = c.dense_ffn_size as usize;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::DenseFfn,
                        &dense.gate,
                        &v(&p.normed),
                        &v(&p.dense_gate),
                        h,
                        f,
                        rows,
                    )?;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::DenseFfn,
                        &dense.up,
                        &v(&p.normed),
                        &v(&p.dense_up),
                        h,
                        f,
                        rows,
                    )?;
                    let (g, u) = (flat(&p.dense_gate, f * rows), flat(&p.dense_up, f * rows));
                    encode_clamped_swiglu(ctx, &enc, &g, &u, &g, c.swiglu_clamp)?;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::DenseFfn,
                        &dense.down,
                        &v(&p.dense_gate),
                        &v(&p.block_out),
                        f,
                        h,
                        rows,
                    )?;
                }
                FfnTensors::Moe(moe) => {
                    let route = p.routes[index].as_ref().ok_or_else(|| {
                        Glm5NextMetalError::Invalid(format!(
                            "block {index} has no packed route record"
                        ))
                    })?;
                    let (e, f, k) = (
                        c.expert_count as usize,
                        c.expert_ffn_size as usize,
                        c.expert_used_count as usize,
                    );
                    self.stage(&mut enc, Glm5NextStage::Router, Some(index))?;
                    if router_e8p32_selected(
                        stage_lineage(p.lineage, Stage::Router),
                        h,
                        e,
                        moe.router.dtype,
                        rows,
                        || router_e8p32_supported(ctx),
                    ) {
                        crate::metal::encode_mat_mat_f32_router_e8p32_strict(
                            ctx,
                            &enc,
                            &moe.router,
                            &v(&p.normed),
                            &v(&p.router),
                            h,
                            e,
                            rows,
                        )?;
                        #[cfg(test)]
                        router_prefill::record_substitution();
                    } else {
                        matmat(
                            ctx,
                            &enc,
                            p.arith(),
                            Stage::Router,
                            &moe.router,
                            &v(&p.normed),
                            &v(&p.router),
                            h,
                            e,
                            rows,
                        )?;
                    }
                    let spec = LearnedRoute {
                        experts: e,
                        top_k: k,
                        score: RouteScore::Sigmoid,
                        routed_scale: c.expert_weights_scale,
                    };
                    let (route_ids, route_weights) = (v(&route.ids), v(&route.weights));
                    encode_route_learned_rows(
                        ctx,
                        &enc,
                        &spec,
                        rows,
                        &v(&p.router),
                        &moe.selection_bias,
                        &route_ids,
                        &route_weights,
                        &v(&route.status),
                    )?;
                    self.stage(&mut enc, Glm5NextStage::RoutedExperts, Some(index))?;
                    let grouped = GroupedExperts {
                        gate_bank: &moe.gate_experts,
                        up_bank: &moe.up_experts,
                        down_bank: &moe.down_experts,
                        input: &v(&p.normed),
                        ids: &route_ids,
                        weights: &route_weights,
                        status: &v(&route.status),
                        counts: &p.counts,
                        slots: &flat(&p.slots, e * rows),
                        inner: &p.inner.view_subrange(0, vec![f as u64, (k * rows) as u64]),
                        slot_out: &p
                            .slot_out
                            .view_subrange(0, vec![h as u64, (k * rows) as u64]),
                        output: &v(&p.routed),
                    };
                    let gate_up = if expert_rows {
                        crate::metal::encode_glm_batch_gate_up
                    } else {
                        crate::metal::encode_all_slots_gate_up_swiglu_native_fused_tokens
                    };
                    let down = if expert_rows {
                        crate::metal::encode_glm_batch_down
                    } else {
                        crate::metal::encode_all_slots_down_native_tokens
                    };
                    gate_up(
                        ctx,
                        &enc,
                        grouped.gate_bank,
                        grouped.up_bank,
                        grouped.input,
                        grouped.ids,
                        grouped.status,
                        grouped.inner,
                        (h, f, e, k),
                        rows,
                        c.swiglu_clamp,
                    )?;
                    down(
                        ctx,
                        &enc,
                        grouped.down_bank,
                        grouped.inner,
                        grouped.ids,
                        grouped.status,
                        grouped.slot_out,
                        (f, h, e, k),
                        rows,
                    )?;
                    crate::metal::encode_moe_weighted_sum_packed_f32(
                        ctx,
                        &enc,
                        grouped.slot_out,
                        grouped.weights,
                        grouped.output,
                        h,
                        k,
                        rows,
                    )?;
                    self.stage(&mut enc, Glm5NextStage::SharedExpert, Some(index))?;
                    let sf = c.shared_expert_ffn_size as usize;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::SharedExpert,
                        &moe.shared.gate,
                        &v(&p.normed),
                        &v(&p.shared_gate),
                        h,
                        sf,
                        rows,
                    )?;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::SharedExpert,
                        &moe.shared.up,
                        &v(&p.normed),
                        &v(&p.shared_up),
                        h,
                        sf,
                        rows,
                    )?;
                    let (g, u) = (
                        flat(&p.shared_gate, sf * rows),
                        flat(&p.shared_up, sf * rows),
                    );
                    encode_clamped_swiglu(ctx, &enc, &g, &u, &g, c.swiglu_clamp)?;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::SharedExpert,
                        &moe.shared.down,
                        &v(&p.shared_gate),
                        &v(&p.shared),
                        sf,
                        h,
                        rows,
                    )?;
                    encode_add_f32(
                        ctx,
                        &enc,
                        &flat(&p.routed, h * rows),
                        &flat(&p.shared, h * rows),
                        &flat(&p.block_out, h * rows),
                    )?;
                }
            }
            self.stage(&mut enc, Glm5NextStage::FfnPost, Some(index))?;
            encode_mhc4_post_rows(
                ctx,
                &enc,
                h,
                rows,
                &v(&p.block_out),
                &b,
                &v(&p.post),
                &v(&p.comb),
                &a,
            )?;
            if waves {
                enc.finish_causal_scope(ctx, false)?;
                enc.begin_causal_scope();
            }
        }
        for (r, seq) in seqs.iter().enumerate() {
            let s = &seq.s;
            encode_mhc4_collapse(
                ctx,
                &enc,
                h,
                &p.residual[0].view_subrange((r * h * 4) as u64, vec![h as u64, 4]),
                &s.quarter,
                &s.final_hidden,
            )?;
            encode_rms_norm_mul_f32(
                ctx,
                &enc,
                &s.final_hidden,
                &w.output_norm,
                &s.final_normed,
                c.rms_epsilon,
            )?;
            copy(ctx, &enc, &s.final_normed, &row(&p.normed, r, h), h)?;
        }
        matmat(
            ctx,
            &enc,
            p.arith(),
            Stage::MlaProjection,
            &w.output,
            &v(&p.normed),
            &logits.view_subrange(0, vec![c.vocab_size as u64, rows as u64]),
            h,
            c.vocab_size as usize,
            rows,
        )?;
        if waves {
            enc.finish_causal_scope(ctx, false)?;
        }
        enc.end();
        if commit {
            #[cfg(any(test, feature = "perf-ablations"))]
            let encode_ms = encode_started.elapsed().as_secs_f64() * 1e3;
            command.commit();
            wait_completed(&command)?;
            #[cfg(any(test, feature = "perf-ablations"))]
            projection::record_timing(
                encode_ms,
                (command.GPUEndTime() - command.GPUStartTime()) * 1e3,
            );
            Ok(None)
        } else {
            #[cfg(any(test, feature = "perf-ablations"))]
            let _ = encode_started;
            Ok(Some(command))
        }
    }
    fn encode_batch_kda(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        seqs: &[&Self],
        index: usize,
        kda: &crate::glm5_next::KdaTensors<MetalTensor>,
    ) -> Result<()> {
        let c = &self.weights.config;
        let p = self.packed.as_ref().unwrap();
        let rows = seqs.len();
        let v = |t: &MetalTensor| rows_view(t, rows);
        let (h, width, rank) = (
            c.hidden_size as usize,
            c.kda_width() as usize,
            c.kda_head_dim as usize,
        );
        for (w, y, m) in [
            (&kda.query, &p.q, width),
            (&kda.key, &p.k, width),
            (&kda.value, &p.v, width),
            (&kda.decay_a, &p.rank_a, rank),
            (&kda.beta, &p.raw_beta, c.head_count as usize),
            (&kda.gate_a, &p.rank_b, rank),
        ] {
            matmat(
                ctx,
                enc,
                p.arith(),
                Stage::KdaProjection,
                w,
                &v(&p.normed),
                &v(y),
                h,
                m,
                rows,
            )?;
        }
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::KdaExpand,
            &kda.decay_b,
            &v(&p.rank_a),
            &v(&p.raw_gate),
            rank,
            width,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::KdaExpand,
            &kda.gate_b,
            &v(&p.rank_b),
            &v(&p.output_gate),
            rank,
            width,
            rows,
        )?;
        if projection::features() & projection::KDA != 0 {
            let states = seqs
                .iter()
                .map(|seq| match &seq.layers[index] {
                    LayerState::Kda { conv, state } => Ok((conv, state)),
                    _ => invalid("not KDA"),
                })
                .collect::<Result<Vec<_>>>()?;
            crate::metal::encode_kda_decode_members(
                ctx,
                enc,
                c.head_count as usize,
                &KdaDecode {
                    q: &v(&p.q),
                    k: &v(&p.k),
                    v: &v(&p.v),
                    raw_gate: &v(&p.raw_gate),
                    raw_beta: &v(&p.raw_beta),
                    output_gate: &v(&p.output_gate),
                    q_conv: &kda.query_conv,
                    k_conv: &kda.key_conv,
                    v_conv: &kda.value_conv,
                    neg_exp_a_log: &kda.neg_exp_a_log,
                    dt_bias: &kda.decay_bias,
                    output_norm: &kda.output_norm,
                    conv_state: states[0].0,
                    state: states[0].1,
                    out: &v(&p.kda_out),
                },
                &states,
                c.kda_gate_lower_bound,
                c.rms_epsilon,
            )?;
        } else {
            for (r, seq) in seqs.iter().enumerate() {
                let LayerState::Kda { conv, state } = &seq.layers[index] else {
                    return invalid("not KDA");
                };
                encode_kda_decode(
                    ctx,
                    enc,
                    c.head_count as usize,
                    &KdaDecode {
                        q: &row(&p.q, r, width),
                        k: &row(&p.k, r, width),
                        v: &row(&p.v, r, width),
                        raw_gate: &row(&p.raw_gate, r, width),
                        raw_beta: &row(&p.raw_beta, r, c.head_count as usize),
                        output_gate: &row(&p.output_gate, r, width),
                        q_conv: &kda.query_conv,
                        k_conv: &kda.key_conv,
                        v_conv: &kda.value_conv,
                        neg_exp_a_log: &kda.neg_exp_a_log,
                        dt_bias: &kda.decay_bias,
                        output_norm: &kda.output_norm,
                        conv_state: conv,
                        state,
                        out: &row(&p.kda_out, r, width),
                    },
                    c.kda_gate_lower_bound,
                    c.rms_epsilon,
                )?;
            }
        }
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::KdaProjection,
            &kda.output,
            &v(&p.kda_out),
            &v(&p.block_out),
            width,
            h,
            rows,
        )
    }
    fn encode_batch_mla(
        &self,
        ctx: &MetalContext,
        enc: &mut KernelEncoder,
        seqs: &[&Self],
        block: usize,
        mla_index: usize,
        mla: &crate::glm5_next::MlaTensors<MetalTensor>,
        selection_stride: Option<usize>,
    ) -> Result<()> {
        let c = &self.weights.config;
        let p = self.packed.as_ref().unwrap();
        let rows = seqs.len();
        let v = |t: &MetalTensor| rows_view(t, rows);
        let h = c.hidden_size as usize;
        let (q_rank, kv) = (c.q_lora_rank as usize, c.kv_lora_rank as usize);
        let heads = c.head_count as usize;
        let head_dim = c.mla_key_head_dim as usize;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::MlaProjection,
            &mla.query_a,
            &v(&p.normed),
            &v(&p.query_a),
            h,
            q_rank,
            rows,
        )?;
        encode_rms_norm_mul_rows_f32(
            ctx,
            enc,
            &v(&p.query_a),
            &mla.query_a_norm,
            &v(&p.query_r),
            rows,
            q_rank,
            c.rms_epsilon,
        )?;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::MlaProjection,
            &mla.query_b,
            &v(&p.query_r),
            &v(&p.query),
            q_rank,
            heads * head_dim,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::MlaProjection,
            &mla.latent,
            &v(&p.normed),
            &v(&p.latent_raw),
            h,
            kv,
            rows,
        )?;
        encode_rms_norm_mul_rows_f32(
            ctx,
            enc,
            &v(&p.latent_raw),
            &mla.latent_norm,
            &v(&p.latent),
            rows,
            kv,
            c.rms_epsilon,
        )?;
        let views = projection::features() & projection::MLA != 0;
        // Member-axis attention binds each session's latent cache. Shared
        // geometry is required by the single compressed-capacity argument;
        // unequal cache shapes keep the incumbent per-member route rather
        // than failing after encoding begins.
        let states = seqs
            .iter()
            .map(|seq| match &seq.layers[block] {
                LayerState::Mla {
                    latent,
                    pending,
                    pooled,
                } => Ok((latent, pending, pooled)),
                _ => invalid("not MLA"),
            })
            .collect::<Result<Vec<_>>>()?;
        let latents: Vec<&MetalTensor> = states.iter().map(|state| state.0).collect();
        let batch_attention = selection_stride.is_some()
            && projection::features() & projection::ATTENTION != 0
            && rows >= 2
            && latents
                .windows(2)
                .all(|pair| pair[0].shape == pair[1].shape);
        let batch_index = selection_stride.is_some()
            && projection::features() & projection::INDEX != 0
            && rows >= 2
            && crate::metal::mat_vec_q8_0_lcpp_enabled()
            && mla.indexer.query.dtype == GgmlType::Q8_0
            && mla.indexer.key.dtype == GgmlType::Q8_0
            && mla.indexer.pool_gate.dtype == GgmlType::Q8_0
            && states
                .windows(2)
                .all(|pair| pair[0].2.shape == pair[1].2.shape);
        let (index_dim, ih) = (c.indexer_head_dim as usize, c.indexer_head_count as usize);
        if !views && !batch_index {
            for (w, y) in [
                (&mla.indexer.key, &p.index_key),
                (&mla.indexer.pool_gate, &p.index_gate),
            ] {
                matmat(
                    ctx,
                    enc,
                    p.arith(),
                    Stage::MlaProjection,
                    w,
                    &v(&p.normed),
                    &v(y),
                    h,
                    c.indexer_head_dim as usize,
                    rows,
                )?;
            }
        }
        if batch_index {
            let sp = p.sparse.as_ref().expect("preflighted sparse scratch");
            let query_width = ih * index_dim;
            crate::metal::encode_mat_vec_q8_0_batch_f32(
                ctx,
                enc,
                &mla.indexer.query,
                &v(&p.query_r),
                &v(&sp.index_query),
                q_rank,
                query_width,
                rows,
            )?;
            crate::metal::encode_mat_vec_q8_0_batch_f32(
                ctx,
                enc,
                &mla.indexer.key,
                &v(&p.normed),
                &v(&p.index_key),
                h,
                index_dim,
                rows,
            )?;
            crate::metal::encode_mat_vec_q8_0_batch_f32(
                ctx,
                enc,
                &mla.indexer.pool_gate,
                &v(&p.normed),
                &v(&p.index_gate),
                h,
                index_dim,
                rows,
            )?;
            crate::metal::encode_scatter_offset_f32_to_f16(
                ctx,
                enc,
                &v(&sp.index_query),
                &sp.index_query_f16,
                0,
                query_width * rows,
            )?;
        }
        for (r, seq) in seqs.iter().enumerate() {
            let s = &seq.s;
            let LayerState::Mla {
                latent,
                pending,
                pooled,
            } = &seq.layers[block]
            else {
                return invalid("not MLA");
            };
            if !views {
                for (src, dst, n) in [
                    (&p.query, &s.query, heads * head_dim),
                    (&p.query_r, &s.query_r, q_rank),
                    (&p.normed, &s.normed, h),
                    (&p.latent, &s.latent, kv),
                    (&p.index_key, &s.index_key, c.indexer_head_dim as usize),
                    (&p.index_gate, &s.index_gate, c.indexer_head_dim as usize),
                ] {
                    copy(ctx, enc, &row(src, r, n), dst, n)?;
                }
            }
            let query = if views {
                std::borrow::Cow::Owned(row(&p.query, r, heads * head_dim))
            } else {
                std::borrow::Cow::Borrowed(&s.query)
            };
            let query_r = if views {
                std::borrow::Cow::Owned(row(&p.query_r, r, q_rank))
            } else {
                std::borrow::Cow::Borrowed(&s.query_r)
            };
            let normed = if views {
                std::borrow::Cow::Owned(row(&p.normed, r, h))
            } else {
                std::borrow::Cow::Borrowed(&s.normed)
            };
            let latent_row = if views {
                std::borrow::Cow::Owned(row(&p.latent, r, kv))
            } else {
                std::borrow::Cow::Borrowed(&s.latent)
            };
            if !batch_index {
                encode_scatter_offset_f32_to_f16(
                    ctx,
                    enc,
                    &latent_row,
                    latent,
                    seq.position * kv,
                    kv,
                )?;
            }
            if batch_attention {
                grouped_matvec(
                    ctx,
                    enc,
                    &mla.key_absorb,
                    &query,
                    &row(&p.query_latent, r, kv * heads),
                    head_dim,
                    kv,
                    heads,
                )?;
            } else {
                grouped_matvec(
                    ctx,
                    enc,
                    &mla.key_absorb,
                    &query,
                    &s.query_latent,
                    head_dim,
                    kv,
                    heads,
                )?;
            }
            // Indexer cache maintenance from token zero. The pool this token
            // completes is published before attention: a query at visible length
            // L scores the first L / 4 pools, including its own.
            if batch_index {
                let sp = p.sparse.as_ref().expect("preflighted sparse scratch");
                seq.stage(enc, Glm5NextStage::MlaIndexer, Some(block))?;
                matvec(
                    ctx,
                    enc,
                    &mla.indexer.head_weights,
                    &normed,
                    &row(&sp.index_weights, r, ih),
                    h,
                    ih,
                )?;
            } else {
                seq.stage(enc, Glm5NextStage::MlaIndexer, Some(block))?;
                matvec(
                    ctx,
                    enc,
                    &mla.indexer.key,
                    &normed,
                    &s.index_key,
                    h,
                    index_dim,
                )?;
                matvec(
                    ctx,
                    enc,
                    &mla.indexer.pool_gate,
                    &normed,
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
                    seq.position,
                    c.layer_norm_epsilon,
                )?;
                let scale = 1.0 / (head_dim as f32).sqrt();
                let visible = seq.position + 1;
                if let Some(stride) = selection_stride {
                    let scores = p
                        .sparse
                        .as_ref()
                        .unwrap()
                        .scores
                        .view_subrange((r * stride) as u64, vec![pooled.shape[1], 1]);
                    seq.encode_sparse_scores(
                        ctx, enc, mla, pooled, visible, block, &query_r, &normed, &scores,
                    )?;
                } else if visible >= c.sparse_frontier() as usize {
                    seq.encode_sparse_attention_inputs(
                        ctx, enc, mla, latent, pooled, mla_index, visible, scale, block, &query_r,
                        &normed,
                    )?;
                } else {
                    seq.stage(enc, Glm5NextStage::DenseAttention, Some(block))?;
                    encode_latent_attention(
                        ctx,
                        enc,
                        &s.query_latent,
                        latent,
                        &s.no_sink,
                        &s.output_latent,
                        seq.position,
                        1,
                        scale,
                    )?;
                }
            }
        }
        if batch_index {
            let sp = p.sparse.as_ref().expect("preflighted sparse scratch");
            let query_width = ih * index_dim;
            crate::metal::encode_scale_f32_in_place(
                ctx,
                enc,
                &sp.index_weights
                    .view_subrange(0, vec![ih as u64, rows as u64]),
                1.0 / (query_width as f32).sqrt(),
            )?;
            let offsets = seqs.iter().map(|seq| seq.position * kv).collect::<Vec<_>>();
            crate::metal::encode_scatter_offset_f32_to_f16_members(
                ctx,
                enc,
                &v(&p.latent),
                &latents,
                &offsets,
                kv,
            )?;
            let append_states = states
                .iter()
                .zip(seqs.iter())
                .map(|(state, seq)| (state.1, state.2, seq.position))
                .collect::<Vec<_>>();
            crate::metal::encode_indexer_append_members(
                ctx,
                enc,
                &v(&p.index_key),
                &v(&p.index_gate),
                &mla.indexer.key_norm,
                &mla.indexer.key_norm_bias,
                &mla.indexer.pool_position,
                &append_states,
                c.layer_norm_epsilon,
            )?;
            let pool_capacity = states[0].2.shape[1] as usize;
            let max_pools = seqs
                .iter()
                .map(|seq| (seq.position + 1) / c.indexer_pool as usize)
                .max()
                .unwrap_or(0);
            let pooled = states.iter().map(|state| state.2).collect::<Vec<_>>();
            let scores = sp
                .scores
                .view_subrange(0, vec![pool_capacity as u64, rows as u64]);
            crate::metal::encode_lightning_scores_f16_matrix_members(
                ctx,
                enc,
                &crate::metal::LightningScores {
                    queries: &sp
                        .index_query_f16
                        .view_subrange(0, vec![index_dim as u64, ih as u64, rows as u64]),
                    head_weights: &sp
                        .index_weights
                        .view_subrange(0, vec![ih as u64, rows as u64]),
                    keys: pooled[0],
                    visible_counts: &sp.visible_pools.view_subrange(0, vec![rows as u64]),
                    scores: &scores,
                },
                &pooled,
                ih,
                index_dim,
                pool_capacity,
                max_pools,
                rows,
            )?;
        }
        if let Some(stride) = selection_stride {
            selection::encode(ctx, enc, seqs, mla_index, stride)?;
            if batch_attention {
                let sp = p.sparse.as_ref().expect("preflighted sparse scratch");
                let caches = &latents;
                let units = memory::split_attention_units(c, rows as u64);
                crate::metal::encode_online_selected_attention_split_f16_members(
                    ctx,
                    enc,
                    &crate::metal::SelectedAttention {
                        queries: &p
                            .query_latent
                            .view_subrange(0, vec![(kv * heads) as u64, rows as u64]),
                        raw_cache: caches[0],
                        raw_cache_before_chunk: caches[0],
                        compressed_cache: caches[0],
                        selected_ids: &sp
                            .row_ids
                            .view_subrange(0, vec![c.selection_width() as u64, rows as u64]),
                        selected_counts: &sp.row_counts.view_subrange(0, vec![rows as u64]),
                        visible_counts: &sp.visible_rows.view_subrange(0, vec![rows as u64]),
                        sinks: &seqs[0].s.no_sink,
                        output: &p
                            .output_latent
                            .view_subrange(0, vec![(kv * heads) as u64, rows as u64]),
                    },
                    caches,
                    crate::metal::SelectedAttentionShape {
                        head_count: heads,
                        query_count: rows,
                        query_token_offset: 0,
                        token_count: rows,
                        chunk_start_position: seqs[0].position,
                        window: 0,
                        raw_cache_is_chunk: false,
                        selected_slots: c.selection_width() as usize,
                        compressed_capacity: caches[0].shape[1] as usize,
                        scale: 1.0 / (head_dim as f32).sqrt(),
                        direct: true,
                    },
                    &crate::metal::SelectedAttentionPartials {
                        values: &sp
                            .attention_partials
                            .view_subrange(0, vec![kv as u64, units]),
                        stats: &sp.attention_partial_stats.view_subrange(0, vec![2, units]),
                    },
                )?;
            }
        }
        for (r, seq) in seqs.iter().enumerate() {
            let s = &seq.s;
            if selection_stride.is_some() && !batch_attention {
                let LayerState::Mla { latent, .. } = &seq.layers[block] else {
                    return invalid("not MLA");
                };
                let sp = p.sparse.as_ref().unwrap();
                seq.encode_sparse_selected_attention(
                    ctx,
                    enc,
                    latent,
                    1.0 / (head_dim as f32).sqrt(),
                    block,
                    &sp.row_ids.view_subrange(
                        (r * c.selection_width() as usize) as u64,
                        vec![c.selection_width() as u64, 1],
                    ),
                    &row(&sp.row_counts, r, 1),
                    &row(&sp.visible_rows, r, 1),
                )?;
            }
            let heads_out = if views {
                std::borrow::Cow::Owned(row(&p.heads_out, r, heads * head_dim))
            } else {
                std::borrow::Cow::Borrowed(&s.heads_out)
            };
            seq.stage(enc, Glm5NextStage::MlaOutput, Some(block))?;
            if batch_attention {
                grouped_matvec(
                    ctx,
                    enc,
                    &mla.value_expand,
                    &row(&p.output_latent, r, kv * heads),
                    &heads_out,
                    kv,
                    head_dim,
                    heads,
                )?;
            } else {
                grouped_matvec(
                    ctx,
                    enc,
                    &mla.value_expand,
                    &s.output_latent,
                    &heads_out,
                    kv,
                    head_dim,
                    heads,
                )?;
            }

            if !views {
                copy(
                    ctx,
                    enc,
                    &s.heads_out,
                    &row(&p.heads_out, r, heads * head_dim),
                    heads * head_dim,
                )?;
            }
        }
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::MlaProjection,
            &mla.output,
            &v(&p.heads_out),
            &v(&p.block_out),
            heads * head_dim,
            h,
            rows,
        )
    }
}

#[cfg(test)]
#[path = "tests/batch_decode.rs"]
mod tests;
