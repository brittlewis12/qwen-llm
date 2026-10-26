//! One serial command per token, including device-side Engram injection.
use super::*;
use objc2_metal::MTLSize;

pub(super) mod ced;

#[cfg(test)]
mod tests;

pub(super) struct Workspace {
    attention: attention_gpu::Workspace,
    ffn: ffn_gpu::Workspace,
    x: MetalTensor,
    half: MetalTensor,
    initial_pre: MetalTensor,
    raw: MetalTensor,
    pre: MetalTensor,
    post: MetalTensor,
    comb: MetalTensor,
    collapsed: MetalTensor,
    normalized: MetalTensor,
    constants: Vec<[MetalTensor; 3]>,
    norm: MetalTensor,
    logits: MetalTensor,
    commands: usize,
    trace_copies: usize,
    engrams: Vec<EngramWorkspace>,
}

struct EngramWorkspace {
    layer: usize,
    input: MetalTensor,
    projected: MetalTensor,
    gate: engram_gpu::BoundGate,
}

/// Decode stages sampled under `profile::stages()`: the Engram projections,
/// then per layer HC pre (with any Engram gate), attention, HC post and the
/// FFN, then the head. Each stage is its own sampled encoder; splitting adds
/// encoder overhead, so read stage shares rather than absolute sums. Near the
/// memory ceiling the splits are not cheap: IQ3_XXS (~217 of 240 GiB) took
/// seconds per token with ~36 ms of sampled stage time and no disk I/O, so
/// profile artifacts with headroom (GSQ at ~145 GiB stays within 1%).
const DECODE_LAYER_STAGES: [&str; 4] = [
    "decode_hc_pre",
    "decode_attention",
    "decode_hc_post",
    "decode_ffn",
];

fn decode_stage_count(layers: usize) -> usize {
    DECODE_LAYER_STAGES.len() * layers + 2
}

/// Per-dispatch profile capacity for one decode command (GSQ uses ~1,317).
const DECODE_DISPATCH_PROFILE_CAPACITY: usize = 4096;

/// The decode step's encoder. Without samples every boundary is a no-op and
/// the whole step stays in one serial encoder, exactly as before. A borrowed
/// encoder (callers that own their encoder, such as the CED scalar path) never
/// samples.
pub(super) struct StageEncoder<'a> {
    command: Option<
        &'a objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
    >,
    samples: Option<&'a metal::MetalTimestampSampleBuffer>,
    enc: StageInner<'a>,
    stage: usize,
}

enum StageInner<'a> {
    Owned(KernelEncoder),
    Borrowed(&'a KernelEncoder),
}

impl<'a> StageEncoder<'a> {
    fn begin(
        command: &'a objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        >,
        samples: Option<&'a metal::MetalTimestampSampleBuffer>,
    ) -> Self {
        let enc = match samples {
            Some(samples) => KernelEncoder::begin_sampled(command, samples, 0, 1, false),
            None => KernelEncoder::begin(command),
        };
        Self {
            command: Some(command),
            samples,
            enc: StageInner::Owned(enc),
            stage: 0,
        }
    }

    fn borrowed(enc: &'a KernelEncoder) -> Self {
        Self {
            command: None,
            samples: None,
            enc: StageInner::Borrowed(enc),
            stage: 0,
        }
    }

    pub(super) fn enc(&self) -> &KernelEncoder {
        match &self.enc {
            StageInner::Owned(enc) => enc,
            StageInner::Borrowed(enc) => enc,
        }
    }

    /// Close the current stage and open the next one (sampled builds only).
    fn boundary(&mut self) {
        let (Some(samples), Some(command)) = (self.samples, self.command) else {
            return;
        };
        if let StageInner::Owned(enc) = &mut self.enc {
            enc.finish();
        }
        self.stage += 1;
        self.enc = StageInner::Owned(KernelEncoder::begin_sampled(
            command,
            samples,
            2 * self.stage,
            2 * self.stage + 1,
            false,
        ));
    }

    /// End the current plain encoder and begin another on the same command,
    /// so Metal's resource tracking may overlap independent work across the
    /// boundary. No-op for borrowed or sampled encoders.
    fn split(&mut self) {
        let (None, Some(command)) = (self.samples, self.command) else {
            return;
        };
        if let StageInner::Owned(enc) = &mut self.enc {
            enc.finish();
            self.enc = StageInner::Owned(KernelEncoder::begin(command));
        }
    }

    fn end(self) {
        if let StageInner::Owned(enc) = self.enc {
            enc.end();
        }
    }
}

#[cfg(test)]
thread_local! {
    static FFN_SPLIT_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// The decode FFN encodes its shared expert and its router plus routed
/// experts on separate encoders, which Metal's resource tracking overlaps:
/// GSQ tg64 59.0 -> 61.8 tok/s at d0, 52.7 -> 54.9 at d2048 (GPU 16.67 ->
/// 15.82 ms/token). The same split for attention (query | KV | core) lost
/// (60.3 / 53.8): its chains are too short to pay for two more encoders.
/// `QWEN_V41_FFN_ENCODERS=one` keeps one encoder (A/B).
fn ffn_split_encoders() -> bool {
    #[cfg(test)]
    if let Some(split) = FFN_SPLIT_OVERRIDE.with(|o| o.get()) {
        return split;
    }
    static SPLIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SPLIT.get_or_init(|| std::env::var("QWEN_V41_FFN_ENCODERS").as_deref() != Ok("one"))
}

pub(super) fn chunk_widths(
    model: &Forward<'_>,
    capacity: usize,
    rows: usize,
) -> Result<Vec<usize>> {
    chunk_widths_from_geometry(
        &model.config,
        model.layers.len(),
        &engram_projections(model),
        capacity,
        rows,
    )
}

pub(super) fn engram_projections(model: &Forward<'_>) -> Vec<(usize, usize)> {
    model
        .layers
        .iter()
        .filter_map(|layer| {
            layer
                .engram
                .as_ref()
                .map(|e| (e.projection.input, e.projection.output))
        })
        .collect()
}

pub(super) fn validate_chunk_geometry(c: &Config, capacity: usize, rows: usize) -> Result<()> {
    op::require(
        capacity > 0 && rows > 0 && rows <= capacity,
        "invalid chunk geometry",
    )?;
    op::require(c.candidate_block > 0, "zero candidate block")?;
    for (a, b) in [
        (4, c.hidden),
        (c.heads, c.head_dim),
        (c.groups, c.output_rank),
        (c.index_heads, c.index_dim),
        (c.intermediate, c.topk),
        (c.hidden, c.topk),
        (c.candidate_topk, c.candidate_block),
    ] {
        a.checked_mul(b).ok_or("chunk width overflow")?;
    }
    Ok(())
}

pub(super) fn chunk_widths_from_geometry(
    c: &Config,
    layers: usize,
    engram_projections: &[(usize, usize)],
    capacity: usize,
    rows: usize,
) -> Result<Vec<usize>> {
    validate_chunk_geometry(c, capacity, rows)?;
    let mut widths = vec![4 * c.hidden, 4 * c.hidden, c.hidden, 4];
    widths.extend(prefill::HcRows::widths(c));
    widths.extend(ffn_gpu::Rows::widths(c, layers));
    widths.extend(attention_gpu::Rows::widths(c, capacity));
    widths.extend(
        attention_gpu::Rows::select_widths(c, capacity, rows)?
            .into_iter()
            .filter(|&width| width != 0),
    );
    widths.extend(attention_gpu::Rows::tile_widths(c, rows)?);
    for &(input, output) in engram_projections {
        widths.extend([input, output]);
    }
    Ok(widths)
}

pub(super) fn bytes_with_schedule(
    c: &Config,
    capacity: usize,
    schedule: &super::super::gguf::Schedule,
    engrams: usize,
) -> Result<u64> {
    let attention = attention_gpu::bytes_with_schedule(c, capacity, schedule)?;
    let ffn = ffn_gpu::bytes_for_layers(c, schedule.layers)?;
    let mul = |a: u64, b: u64| {
        a.checked_mul(b)
            .ok_or("GPU step memory overflow".to_string())
    };
    let add = |a: u64, b: u64| {
        a.checked_add(b)
            .ok_or("GPU step memory overflow".to_string())
    };
    let h = c.hidden as u64;
    let layers = schedule.layers as u64;
    let engrams = engrams as u64;
    let checkpoints = schedule.index_source_layers.len() as u64;
    let slack = 16384 + 1024;
    let own = add(add(mul(11, h)?, 64)?, c.vocab as u64)?;
    let constants = mul(layers, add(h, 27)?)?;
    let engram = mul(engrams, add(6144, mul(13, h)?)?)?;
    let trace = mul(layers, add(mul(4, h)?, 4)?)?;
    let checkpoint = mul(
        checkpoints,
        add(
            add(mul(2, h)?, c.index_topk as u64)?,
            capacity.div_ceil(c.candidate_block) as u64,
        )?,
    )?;
    let allocations = add(
        add(
            add(mul(5, layers)?, mul(4, engrams)?)?,
            mul(4, checkpoints)?,
        )?,
        16,
    )?;
    let logical = add(add(add(add(own, constants)?, engram)?, trace)?, checkpoint)?;
    add(
        add(attention, ffn)?,
        add(mul(logical, 4)?, mul(allocations, slack)?)?,
    )
}

fn write(t: &MetalTensor, values: &[f32]) -> Result<()> {
    op::require(f32_slice(t)?.len() == values.len(), "GPU step upload shape")?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            t.buffer
                .contents()
                .as_ptr()
                .cast::<f32>()
                .add((t.offset / 4) as usize),
            values.len(),
        );
    }
    Ok(())
}

impl Workspace {
    pub(super) fn run_chunk(
        &mut self,
        model: &Forward<'_>,
        start: usize,
        tokens: &[u32],
        hashes: &[engram::LayerRows],
        cancel: Option<&AtomicBool>,
        with_logits: bool,
        mut capture: Option<&mut dyn capture::Capture>,
    ) -> Result<Vec<f32>> {
        let ctx = model.ctx;
        let c = &model.config;
        let n = tokens.len();
        let prepare = profile::Wall("prefill_host_prepare", [0; 4], profile::start());
        let mut embeddings = Vec::with_capacity(n * 4 * c.hidden);
        for &token in tokens {
            embeddings.extend(decode_rows(model.embedding, token as usize, 1)?.repeat(4));
        }
        let x = upload(ctx, &embeddings, vec![c.hidden as u64, 4, n as u64])?;
        let half = prefill::allocate(ctx, 4 * c.hidden, n, GgmlType::F32)?
            .view_bytes(0, vec![c.hidden as u64, 4, n as u64]);
        let outputs = prefill::allocate(ctx, c.hidden, n, GgmlType::F32)?;
        let incoming = upload(ctx, &[1.0, 0.0, 0.0, 0.0].repeat(n), vec![4, n as u64])?;
        let hc = prefill::HcRows::new(ctx, c, n)?;
        let ffn = ffn_gpu::Rows::new(ctx, c, n, model.layers.len())?;
        let attention = attention_gpu::Rows::new(ctx, c, self.attention.capacity(), n)?;
        let mut engrams = Vec::new();
        let engram_prepare = profile::Wall("prefill_engram_prepare", [0; 4], profile::start());
        for slot in &self.engrams {
            let p = &model.layers[slot.layer].engram.as_ref().unwrap().projection;
            let values = model.engram_rows.decode_tokens(slot.layer, hashes)?;
            engrams.push((
                slot.layer,
                upload(ctx, &values, vec![p.input as u64, n as u64])?,
                prefill::allocate(ctx, p.output, n, GgmlType::F32)?,
            ));
        }
        drop(engram_prepare);
        let snapshots = match capture.as_ref() {
            Some(sink) => capture::batch_snapshots(model, start, n, &**sink)?,
            None => Vec::new(),
        };
        self.attention.begin_batch(start, n)?;
        self.commands = 0;
        self.trace_copies = 0;
        cancelled(cancel)?;
        drop(prepare);
        let started = profile::start();
        let samples = if profile::stages() {
            Some(
                ctx.timestamp_sample_buffer(model.layers.len() * 12)
                    .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        let command = ctx
            .queue
            .commandBuffer()
            .ok_or("Metal command buffer allocation")?;
        let mut enc = match &samples {
            Some(samples) => KernelEncoder::begin_sampled(&command, samples, 0, 1, false),
            None => KernelEncoder::begin(&command),
        };
        let encoded = (|| -> Result<()> {
            for (id, layer) in model.layers.iter().enumerate() {
                cancelled(cancel)?;
                if let Some((_, input, projected)) =
                    engrams.iter().find(|(layer, _, _)| *layer == id)
                {
                    layer
                        .engram
                        .as_ref()
                        .unwrap()
                        .projection
                        .encode_rows(&enc, input, projected, n)
                        .map_err(|e| e.to_string())?;
                    self.engrams
                        .iter()
                        .find(|slot| slot.layer == id)
                        .unwrap()
                        .gate
                        .encode(ctx, &enc, &x, projected, n)?;
                }
                hc.encode(
                    ctx,
                    &enc,
                    c,
                    &layer.attn_hc.projection,
                    &self.constants[id],
                    &x,
                    &incoming,
                )
                .map_err(|e| e.to_string())?;
                if let Some(samples) = &samples {
                    enc.finish();
                    enc = KernelEncoder::begin_sampled(
                        &command,
                        samples,
                        id * 12 + 2,
                        id * 12 + 3,
                        false,
                    );
                }
                self.attention.encode_rows_prepare(
                    ctx,
                    &enc,
                    id,
                    start,
                    &layer.attention,
                    &hc.normalized,
                    &attention,
                )?;
                if let Some(samples) = &samples {
                    enc.finish();
                    enc = KernelEncoder::begin_sampled(
                        &command,
                        samples,
                        id * 12 + 4,
                        id * 12 + 5,
                        false,
                    );
                }
                self.attention.encode_rows_serial(
                    ctx,
                    &enc,
                    id,
                    start,
                    &layer.attention,
                    &hc.normalized,
                    &attention,
                )?;
                if let Some(samples) = &samples {
                    enc.finish();
                    enc = KernelEncoder::begin_sampled(
                        &command,
                        samples,
                        id * 12 + 6,
                        id * 12 + 7,
                        false,
                    );
                }
                self.attention
                    .encode_rows_tiles(ctx, &enc, id, start, &attention)?;
                if let Some(samples) = &samples {
                    enc.finish();
                    enc = KernelEncoder::begin_sampled(
                        &command,
                        samples,
                        id * 12 + 8,
                        id * 12 + 9,
                        false,
                    );
                }
                self.attention
                    .finish_rows(ctx, &enc, &layer.attention, &attention, &outputs)?;
                for (position, _, snapshot) in snapshots.iter().filter(|(_, layer, _)| *layer == id)
                {
                    let row = position - start;
                    self.attention.capture_row(
                        ctx,
                        &enc,
                        snapshot,
                        &prefill::row(&hc.normalized, c.hidden, row),
                        &prefill::row(&outputs, c.hidden, row),
                        &attention,
                        id,
                        *position,
                        row,
                    )?;
                }
                metal::encode_mhc4_post_rows(
                    ctx, &enc, c.hidden, n, &outputs, &x, &hc.post, &hc.comb, &half,
                )
                .map_err(|e| e.to_string())?;
                if let Some(samples) = &samples {
                    enc.finish();
                    enc = KernelEncoder::begin_sampled(
                        &command,
                        samples,
                        id * 12 + 10,
                        id * 12 + 11,
                        false,
                    );
                }
                self.ffn
                    .encode_rows(ctx, &enc, c, layer, id, &half, &hc.pre, &x, &ffn)
                    .map_err(|e| e.to_string())?;
                // Lagged incoming pre is the previous block's mix for this same token.
                metal::encode_copy_offset_f32(ctx, &enc, &ffn.hc.pre, 0, &incoming, 4 * n)
                    .map_err(|e| e.to_string())?;
                if with_logits && id + 1 == model.layers.len() {
                    let last = prefill::row(&x, 4 * c.hidden, n - 1)
                        .view_bytes(0, vec![c.hidden as u64, 4]);
                    metal::encode_mhc4_collapse(
                        ctx,
                        &enc,
                        c.hidden,
                        &last,
                        &prefill::row(&incoming, 4, n - 1),
                        &self.collapsed,
                    )
                    .map_err(|e| e.to_string())?;
                    metal::encode_rms_norm_mul_rows_f32(
                        ctx,
                        &enc,
                        &self.collapsed,
                        &self.norm,
                        &self.normalized,
                        1,
                        c.hidden,
                        c.norm_eps,
                    )
                    .map_err(|e| e.to_string())?;
                    model
                        .head
                        .encode(&enc, &self.normalized, &self.logits)
                        .map_err(|e| e.to_string())?;
                }
                if let Some(samples) = &samples {
                    if id + 1 < model.layers.len() {
                        enc.finish();
                        enc = KernelEncoder::begin_sampled(
                            &command,
                            samples,
                            (id + 1) * 12,
                            (id + 1) * 12 + 1,
                            false,
                        );
                    }
                }
            }
            cancelled(cancel)?;
            Ok(())
        })();
        enc.end();
        encoded?;
        let encode_ms = started.map(|s| s.elapsed().as_secs_f64() * 1000.0);
        let wait = std::time::Instant::now();
        let result = metal::commit_and_wait(&command).map_err(|e| e.to_string());
        self.commands += 1;
        if let Some(encode_ms) = encode_ms {
            profile::submitted_with_driver_scheduling(
                "prefill_chunk_rows",
                [start, model.layers.len(), n, 0],
                encode_ms,
                wait.elapsed().as_secs_f64() * 1000.0,
                result.is_ok(),
                command.GPUStartTime(),
                command.GPUEndTime(),
                || (command.kernelStartTime(), command.kernelEndTime()),
            );
        }
        result?;
        if let Some(samples) = &samples {
            let times = ctx
                .resolve_timestamp_samples(samples, model.layers.len() * 12)
                .map_err(|e| e.to_string())?;
            for (layer, t) in times.chunks_exact(12).enumerate() {
                profile::gpu_interval_layer("prefill_attention_hc", n, layer, t[0], t[1]);
                profile::gpu_interval_layer("prefill_attention_prepare", n, layer, t[2], t[3]);
                profile::gpu_interval_layer("prefill_attention_serial", n, layer, t[4], t[5]);
                profile::gpu_interval_layer("prefill_attention_tiles", n, layer, t[6], t[7]);
                profile::gpu_interval_layer("prefill_attention_finish_post", n, layer, t[8], t[9]);
                profile::gpu_interval_layer("prefill_ffn", n, layer, t[10], t[11]);
            }
        }
        ffn.validate_status()?;
        let status = self.attention.status();
        let value = unsafe {
            *status
                .buffer
                .contents()
                .as_ptr()
                .cast::<i32>()
                .add((status.offset / 4) as usize)
        };
        op::require(value == 0, &format!("GPU attention status {value}"))?;
        cancelled(cancel)?;
        if let Some(sink) = capture.as_mut() {
            for (position, layer, snapshot) in &snapshots {
                snapshot.deliver(c, *position, *layer, &mut **sink)?;
                cancelled(cancel)?;
            }
        }
        self.attention.validate_snapshot_position(start + n)?;
        if with_logits {
            let values = f32_slice(&self.logits)?;
            op::require(values.iter().all(|v| v.is_finite()), "non-finite logits")?;
            Ok(values.to_vec())
        } else {
            Ok(Vec::new())
        }
    }

    pub(super) fn attention(&self) -> &attention_gpu::Workspace {
        &self.attention
    }

    pub(super) fn attention_mut(&mut self) -> &mut attention_gpu::Workspace {
        &mut self.attention
    }
    pub(super) fn new(
        model: &Forward<'_>,
        capacity: usize,
    ) -> std::result::Result<Self, StateError> {
        let c = &model.config;
        model.schedule.validate().map_err(|e| e.to_string())?;
        model.engram_hash.validate().map_err(|e| format!("{e:?}"))?;
        op::require(
            model.layers.len() == model.schedule.layers
                && model
                    .engram_hash
                    .layers
                    .iter()
                    .all(|hash| hash.layer < model.layers.len()),
            "GPU step schedule/Engram layer extent",
        )?;
        let estimate = model.estimate_request_memory_with_gpu_step(capacity)?;
        let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
            estimate.metal_bytes,
            estimate.cpu_bytes,
            0,
            model.ctx.memory_signals(),
            true,
        );
        if let Some(reason) = admission.refusal() {
            return Err(StateError::MemoryAdmissionDenied(reason));
        }
        op::require(
            model.layers.iter().enumerate().all(|(id, l)| {
                l.engram.is_some() == model.engram_hash.layers.iter().any(|hash| hash.layer == id)
            }),
            "GPU step Engram placement",
        )?;
        let f = |n: usize| -> Result<MetalTensor> {
            let priced = model
                .ctx
                .price_shared_buffer_upper(n as u64 * 4)
                .map_err(|e| e.to_string())?;
            op::require(
                priced.priced_upper_bytes <= n as u64 * 4 + 16384,
                "GPU step allocation page slack",
            )?;
            MetalTensor::zeros_dtype_unstaged(model.ctx, vec![n as u64], GgmlType::F32)
                .map_err(|e| e.to_string())
        };
        let constants = model
            .layers
            .iter()
            .map(|l| {
                Ok([
                    upload(model.ctx, &l.attn_hc.scale, vec![3])?,
                    upload(model.ctx, &l.attn_hc.base, vec![24])?,
                    upload(model.ctx, &l.attn_hc.norm, vec![c.hidden as u64])?,
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            attention: attention_gpu::Workspace::new_with_schedule(
                model.ctx,
                c,
                capacity,
                &model
                    .layers
                    .iter()
                    .map(|l| &l.attention)
                    .collect::<Vec<_>>(),
                &model.schedule,
            )?,
            ffn: ffn_gpu::Workspace::new(model)?,
            x: f(4 * c.hidden)?.view_subrange(0, vec![c.hidden as u64, 4]),
            half: f(4 * c.hidden)?.view_subrange(0, vec![c.hidden as u64, 4]),
            initial_pre: upload(model.ctx, &[1.0, 0.0, 0.0, 0.0], vec![4])?,
            raw: f(24)?,
            pre: f(4)?,
            post: f(4)?,
            comb: f(16)?.view_subrange(0, vec![4, 4]),
            collapsed: f(c.hidden)?,
            normalized: f(c.hidden)?,
            norm: upload(model.ctx, &model.norm, vec![c.hidden as u64])?,
            logits: f(c.vocab)?,
            constants,
            commands: 0,
            trace_copies: 0,
            engrams: model
                .engram_hash
                .layers
                .iter()
                .map(|hash| {
                    let e = model.layers[hash.layer]
                        .engram
                        .as_ref()
                        .ok_or("missing Engram")?;
                    Ok(EngramWorkspace {
                        layer: hash.layer,
                        input: f(6144)?,
                        projected: f(e.projection.output)?,
                        gate: engram_gpu::BoundGate::new(model.ctx, &e.gate, c)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn normalize_scalar_input(
        &self,
        model: &Forward<'_>,
        enc: &KernelEncoder,
        id: usize,
        x: &MetalTensor,
        incoming: &MetalTensor,
    ) -> Result<()> {
        let ctx = model.ctx;
        let c = &model.config;
        let l = &model.layers[id];
        let encode = || -> std::result::Result<(), metal::MetalError> {
            if hc_gpu::fused_supported(&l.attn_hc) {
                return hc_gpu::encode_pre(
                    ctx,
                    enc,
                    c,
                    &l.attn_hc,
                    &self.constants[id],
                    x,
                    hc_gpu::HcPreSite {
                        mix: &self.raw,
                        pre: &self.pre,
                        post: &self.post,
                        combination: &self.comb,
                        incoming_pre: incoming,
                        collapsed: &self.collapsed,
                        normalized: &self.normalized,
                    },
                );
            }
            l.attn_hc.projection.encode(enc, x, &self.raw)?;
            enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_projected_rms")?);
            enc.set_bytes(0, &((4 * c.hidden) as u32));
            enc.set_bytes(1, &c.norm_eps);
            enc.set_tensor(2, x);
            enc.set_tensor(3, &self.raw);
            enc.dispatch(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            metal::encode_mhc4_controls(
                ctx,
                enc,
                c.hc_eps,
                &self.raw,
                &self.constants[id][0],
                &self.constants[id][1],
                &self.pre,
                &self.post,
                &self.comb,
            )?;
            metal::encode_mhc4_collapse(ctx, enc, c.hidden, x, incoming, &self.collapsed)?;
            metal::encode_rms_norm_mul_rows_f32(
                ctx,
                enc,
                &self.collapsed,
                &self.constants[id][2],
                &self.normalized,
                1,
                c.hidden,
                c.norm_eps,
            )
        };
        encode().map_err(|e| e.to_string())
    }

    fn layer(
        &mut self,
        model: &Forward<'_>,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        snapshot: Option<&capture::Snapshot>,
        ced: bool,
    ) -> Result<()> {
        self.layer_staged(
            model,
            &mut StageEncoder::borrowed(enc),
            id,
            position,
            snapshot,
            ced,
        )
    }

    /// `layer` with decode stage boundaries (HC pre | attention | HC post |
    /// FFN); boundaries only split encoders when stages are sampled.
    fn layer_staged(
        &mut self,
        model: &Forward<'_>,
        stages: &mut StageEncoder<'_>,
        id: usize,
        position: usize,
        snapshot: Option<&capture::Snapshot>,
        ced: bool,
    ) -> Result<()> {
        let ctx = model.ctx;
        let c = &model.config;
        let l = &model.layers[id];
        let incoming = if id == 0 {
            &self.initial_pre
        } else {
            self.ffn.next_pre()
        };
        metal::dispatch_profile_label("hc_pre", Some(id));
        self.normalize_scalar_input(model, stages.enc(), id, &self.x, incoming)?;
        stages.boundary();
        let enc = stages.enc();
        let output = if ced {
            self.attention.encode_ced_scalar(
                ctx,
                enc,
                id,
                position,
                &l.attention,
                &self.normalized,
            )?
        } else {
            self.attention
                .encode(ctx, enc, id, position, &l.attention, &self.normalized)?
        };
        stages.boundary();
        let enc = stages.enc();
        metal::dispatch_profile_label("hc_post", Some(id));
        metal::encode_mhc4_post(
            ctx, enc, c.hidden, output, &self.x, &self.post, &self.comb, &self.half,
        )
        .map_err(|e| e.to_string())?;
        if let Some(snapshot) = snapshot {
            self.attention
                .capture(ctx, enc, snapshot, &self.normalized, id, position)?;
        }
        stages.boundary();
        if !ffn_split_encoders() {
            let enc = stages.enc();
            return self
                .ffn
                .encode(ctx, enc, c, l, id, &self.half, &self.pre, &self.x)
                .map_err(|e| e.to_string());
        }
        let err = |e: metal::MetalError| e.to_string();
        self.ffn
            .encode_hc_pre(ctx, stages.enc(), c, l, id, &self.half, &self.pre)
            .map_err(err)?;
        stages.split();
        self.ffn
            .encode_shared(ctx, stages.enc(), c, l, id)
            .map_err(err)?;
        stages.split();
        self.ffn
            .encode_routed_experts(ctx, stages.enc(), c, l, id)
            .map_err(err)?;
        stages.split();
        self.ffn
            .encode_combine(ctx, stages.enc(), c, id, &self.half, &self.x)
            .map_err(err)
    }

    pub(super) fn run(
        &mut self,
        model: &Forward<'_>,
        position: usize,
        token: usize,
        rows: &engram::LayerRows,
        cancel: Option<&AtomicBool>,
        trace: Option<&mut dyn FnMut(LayerTrace<'_>) -> Result<()>>,
        capture: Option<&mut dyn capture::Capture>,
        with_logits: bool,
    ) -> Result<Vec<f32>> {
        self.run_prepared(
            model,
            position,
            token,
            rows,
            cancel,
            trace,
            capture,
            with_logits,
            None,
        )
    }

    pub(super) fn run_prepared(
        &mut self,
        model: &Forward<'_>,
        position: usize,
        token: usize,
        rows: &engram::LayerRows,
        cancel: Option<&AtomicBool>,
        mut trace: Option<&mut dyn FnMut(LayerTrace<'_>) -> Result<()>>,
        mut capture: Option<&mut dyn capture::Capture>,
        with_logits: bool,
        prepared: Option<&[(usize, MetalTensor)]>,
    ) -> Result<Vec<f32>> {
        let c = &model.config;
        let ctx = model.ctx;
        let snapshots = if let Some(sink) = capture.as_ref().filter(|s| s.wants_position(position))
        {
            model
                .schedule
                .index_source_layers
                .iter()
                .copied()
                .map(|id| {
                    Ok((
                        id,
                        capture::Snapshot::new(
                            ctx,
                            c,
                            position,
                            id,
                            sink.is_checkpoint(position),
                            &model.schedule,
                        )?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        self.commands = 0;
        self.trace_copies = 0;
        let embedding = decode_rows(model.embedding, token, 1)?;
        write(&self.x, &embedding.repeat(4))?;
        let engram_rows = profile::Wall(
            "decode_engram_rows",
            [0, self.engrams.len(), 0, 0],
            profile::start(),
        );
        if prepared.is_none() {
            // All layers' rows are advised before any read: uncached rows are
            // SSD round trips that should overlap.
            let layers: Vec<usize> = self.engrams.iter().map(|slot| slot.layer).collect();
            let values = model.engram_rows.decode_layers(&layers, rows)?;
            for (slot, values) in self.engrams.iter().zip(values) {
                write(&slot.input, &values)?;
            }
        }
        drop(engram_rows);
        let captures = if trace.is_some() {
            (0..model.layers.len())
                .map(|_| {
                    Ok((
                        MetalTensor::zeros_dtype_unstaged(
                            ctx,
                            vec![(4 * c.hidden) as u64],
                            GgmlType::F32,
                        )
                        .map_err(|e| e.to_string())?,
                        MetalTensor::zeros_dtype_unstaged(ctx, vec![4], GgmlType::F32)
                            .map_err(|e| e.to_string())?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        {
            let (start, end) = (0, model.layers.len());
            cancelled(cancel)?;
            let started = profile::start();
            let _wall = profile::Wall("gpu_step", [position, start, end, 0], started);
            let command = ctx
                .queue
                .commandBuffer()
                .ok_or("Metal command buffer allocation")?;
            let samples = if profile::stages() {
                Some(
                    ctx.timestamp_sample_buffer(2 * decode_stage_count(end - start))
                        .map_err(|e| e.to_string())?,
                )
            } else {
                None
            };
            // Plain encoders begun under this guard split per dispatch, so it
            // must precede the step encoder; stage sampling excludes it.
            let dispatch_profile = if profile::dispatches() && samples.is_none() {
                Some(
                    metal::dispatch_profile_begin(ctx, DECODE_DISPATCH_PROFILE_CAPACITY)
                        .map_err(|e| e.to_string())?,
                )
            } else {
                None
            };
            metal::dispatch_profile_label("engram_projection", None);
            let mut stages = StageEncoder::begin(&command, samples.as_ref());
            let encoded = (|| -> Result<()> {
                for slot in &self.engrams {
                    if prepared.is_some() {
                        continue;
                    }
                    let p = &model.layers[slot.layer].engram.as_ref().unwrap().projection;
                    p.encode(stages.enc(), &slot.input, &slot.projected)
                        .map_err(|e| e.to_string())?;
                }
                for id in start..end {
                    cancelled(cancel)?;
                    stages.boundary();
                    metal::dispatch_profile_label("hc_pre", Some(id));
                    if let Some(slot) = self.engrams.iter().find(|slot| slot.layer == id) {
                        let projected = match prepared {
                            Some(rows) => {
                                &rows
                                    .iter()
                                    .find(|(layer, _)| *layer == id)
                                    .ok_or("missing prepared Engram projection")?
                                    .1
                            }
                            None => &slot.projected,
                        };
                        slot.gate.encode(ctx, stages.enc(), &self.x, projected, 1)?;
                    }
                    self.layer_staged(
                        model,
                        &mut stages,
                        id,
                        position,
                        snapshots
                            .iter()
                            .find(|(layer, _)| *layer == id)
                            .map(|(_, s)| s),
                        false,
                    )?;
                    if !captures.is_empty() {
                        metal::encode_copy_offset_f32(
                            ctx,
                            stages.enc(),
                            &self.x,
                            0,
                            &captures[id].0,
                            4 * c.hidden,
                        )
                        .map_err(|e| e.to_string())?;
                        metal::encode_copy_offset_f32(
                            ctx,
                            stages.enc(),
                            self.ffn.next_pre(),
                            0,
                            &captures[id].1,
                            4,
                        )
                        .map_err(|e| e.to_string())?;
                        self.trace_copies += 2;
                    }
                }
                stages.boundary();
                metal::dispatch_profile_label("head", None);
                if with_logits {
                    metal::encode_mhc4_collapse(
                        ctx,
                        stages.enc(),
                        c.hidden,
                        &self.x,
                        self.ffn.next_pre(),
                        &self.collapsed,
                    )
                    .map_err(|e| e.to_string())?;
                    metal::encode_rms_norm_mul_rows_f32(
                        ctx,
                        stages.enc(),
                        &self.collapsed,
                        &self.norm,
                        &self.normalized,
                        1,
                        c.hidden,
                        c.norm_eps,
                    )
                    .map_err(|e| e.to_string())?;
                    model
                        .head
                        .encode(stages.enc(), &self.normalized, &self.logits)
                        .map_err(|e| e.to_string())?;
                }
                Ok(())
            })();
            stages.end();
            encoded?;
            let encode_ms = started.map(|s| s.elapsed().as_secs_f64() * 1000.0);
            let wait = std::time::Instant::now();
            let result = metal::commit_and_wait(&command).map_err(|e| e.to_string());
            self.commands += 1;
            if let Some(encode_ms) = encode_ms {
                profile::submitted_with_driver_scheduling(
                    "gpu_step",
                    [position, start, end, 0],
                    encode_ms,
                    wait.elapsed().as_secs_f64() * 1000.0,
                    result.is_ok(),
                    command.GPUStartTime(),
                    command.GPUEndTime(),
                    || (command.kernelStartTime(), command.kernelEndTime()),
                );
            }
            result?;
            if let Some(guard) = dispatch_profile {
                profile::record_dispatches(
                    metal::dispatch_profile_finish(ctx, guard).map_err(|e| e.to_string())?,
                );
            }
            if let Some(samples) = &samples {
                let layers = end - start;
                let times = ctx
                    .resolve_timestamp_samples(samples, 2 * decode_stage_count(layers))
                    .map_err(|e| e.to_string())?;
                let interval = |stage: usize| (times[2 * stage], times[2 * stage + 1]);
                let (s, e) = interval(0);
                profile::gpu_interval_layer("decode_engram_projection", 1, 0, s, e);
                for layer in 0..layers {
                    for (k, name) in DECODE_LAYER_STAGES.iter().enumerate() {
                        let (s, e) = interval(1 + DECODE_LAYER_STAGES.len() * layer + k);
                        profile::gpu_interval_layer(name, 1, start + layer, s, e);
                    }
                }
                let (s, e) = interval(decode_stage_count(layers) - 1);
                profile::gpu_interval_layer("decode_head", 1, 0, s, e);
            }
            cancelled(cancel)?;
            let status = unsafe {
                *self
                    .attention
                    .status()
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<i32>()
            };
            op::require(status == 0, &format!("GPU attention status {status}"))?;
            for id in start..end {
                let status = self.ffn.status(id);
                op::require(
                    status == metal::ROUTE_STATUS_READY,
                    &format!("GPU FFN layer {id} route status {status}"),
                )?;
            }
            if let Some(callback) = trace.as_mut() {
                for id in start..end {
                    callback(LayerTrace {
                        position,
                        layer: id,
                        hidden: f32_slice(&captures[id].0)?,
                        next_pre_mix: f32_slice(&captures[id].1)?,
                    })?;
                    cancelled(cancel)?;
                }
            }
            if let Some(sink) = capture.as_mut() {
                for (id, snapshot) in &snapshots {
                    if (start..end).contains(id) {
                        snapshot.deliver(c, position, *id, *sink)?;
                        cancelled(cancel)?;
                    }
                }
            }
        }
        if !with_logits {
            return Ok(Vec::new());
        }
        let logits = f32_slice(&self.logits)?;
        op::require(logits.iter().all(|v| v.is_finite()), "non-finite logits")?;
        Ok(logits.to_vec())
    }
}
