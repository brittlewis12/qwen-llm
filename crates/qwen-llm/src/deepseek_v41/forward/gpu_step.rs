//! One serial command per token, including device-side Engram injection.
use super::*;
use objc2_metal::MTLSize;

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

pub(super) fn chunk_widths(
    model: &Forward<'_>,
    capacity: usize,
    rows: usize,
) -> Result<Vec<usize>> {
    let c = &model.config;
    let mut widths = vec![4 * c.hidden, 4 * c.hidden, c.hidden, 4];
    widths.extend(prefill::HcRows::widths(c));
    widths.extend(ffn_gpu::Rows::widths(c, model.layers.len()));
    widths.extend(attention_gpu::Rows::widths(c, capacity));
    widths.extend(attention_gpu::Rows::tile_widths(c, rows)?);
    for layer in &model.layers {
        if let Some(e) = &layer.engram {
            widths.extend([e.projection.input, e.projection.output]);
        }
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
        for slot in &self.engrams {
            let p = &model.layers[slot.layer].engram.as_ref().unwrap().projection;
            let mut values = Vec::with_capacity(n * p.input);
            for hash in hashes {
                values.extend(model.engram_rows.decode(
                    slot.layer,
                    hash.get(&slot.layer).ok_or("missing Engram hashes")?,
                )?);
            }
            engrams.push((
                slot.layer,
                upload(ctx, &values, vec![p.input as u64, n as u64])?,
                prefill::allocate(ctx, p.output, n, GgmlType::F32)?,
            ));
        }
        let snapshots = match capture.as_ref() {
            Some(sink) => capture::batch_snapshots(model, start, n, &**sink)?,
            None => Vec::new(),
        };
        self.attention.begin_batch(start, n)?;
        self.commands = 0;
        self.trace_copies = 0;
        cancelled(cancel)?;
        let started = profile::start();
        let samples = if profile::stages() {
            Some(
                ctx.timestamp_sample_buffer(model.layers.len() * 8)
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
                        id * 8 + 2,
                        id * 8 + 3,
                        false,
                    );
                }
                self.attention.encode_rows(
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
                        id * 8 + 4,
                        id * 8 + 5,
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
                        id * 8 + 6,
                        id * 8 + 7,
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
                            (id + 1) * 8,
                            (id + 1) * 8 + 1,
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
            profile::submitted(
                "prefill_chunk_rows",
                [start, model.layers.len(), n, 0],
                encode_ms,
                wait.elapsed().as_secs_f64() * 1000.0,
                result.is_ok(),
                command.GPUStartTime(),
                command.GPUEndTime(),
            );
        }
        result?;
        if let Some(samples) = &samples {
            let times = ctx
                .resolve_timestamp_samples(samples, model.layers.len() * 8)
                .map_err(|e| e.to_string())?;
            for t in times.chunks_exact(8) {
                profile::gpu_interval("prefill_attention_hc", n, t[0], t[1]);
                profile::gpu_interval("prefill_attention_rows", n, t[2], t[3]);
                profile::gpu_interval("prefill_attention_finish_post", n, t[4], t[5]);
                profile::gpu_interval("prefill_ffn", n, t[6], t[7]);
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

    fn layer(
        &mut self,
        model: &Forward<'_>,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        snapshot: Option<&capture::Snapshot>,
    ) -> Result<()> {
        let ctx = model.ctx;
        let c = &model.config;
        let l = &model.layers[id];
        let incoming = if id == 0 {
            &self.initial_pre
        } else {
            self.ffn.next_pre()
        };
        let encode = || -> std::result::Result<(), metal::MetalError> {
            if hc_gpu::fused_supported(&l.attn_hc) {
                return hc_gpu::encode_pre(
                    ctx,
                    enc,
                    c,
                    &l.attn_hc,
                    &self.constants[id],
                    &self.x,
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
            l.attn_hc.projection.encode(enc, &self.x, &self.raw)?;
            enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_projected_rms")?);
            enc.set_bytes(0, &((4 * c.hidden) as u32));
            enc.set_bytes(1, &c.norm_eps);
            enc.set_tensor(2, &self.x);
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
            metal::encode_mhc4_collapse(ctx, enc, c.hidden, &self.x, incoming, &self.collapsed)?;
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
        encode().map_err(|e| e.to_string())?;
        let output =
            self.attention
                .encode(ctx, enc, id, position, &l.attention, &self.normalized)?;
        metal::encode_mhc4_post(
            ctx, enc, c.hidden, output, &self.x, &self.post, &self.comb, &self.half,
        )
        .map_err(|e| e.to_string())?;
        if let Some(snapshot) = snapshot {
            self.attention
                .capture(ctx, enc, snapshot, &self.normalized, id, position)?;
        }
        self.ffn
            .encode(ctx, enc, c, l, id, &self.half, &self.pre, &self.x)
            .map_err(|e| e.to_string())
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
        for slot in &self.engrams {
            if prepared.is_some() {
                continue;
            }
            let id = slot.layer;
            let values = model
                .engram_rows
                .decode(id, rows.get(&id).ok_or("missing Engram hashes")?)?;
            write(&slot.input, &values)?;
        }
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
            let enc = KernelEncoder::begin(&command);
            let encoded = (|| -> Result<()> {
                for slot in &self.engrams {
                    if prepared.is_some() {
                        continue;
                    }
                    let p = &model.layers[slot.layer].engram.as_ref().unwrap().projection;
                    p.encode(&enc, &slot.input, &slot.projected)
                        .map_err(|e| e.to_string())?;
                }
                for id in start..end {
                    cancelled(cancel)?;
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
                        slot.gate.encode(ctx, &enc, &self.x, projected, 1)?;
                    }
                    self.layer(
                        model,
                        &enc,
                        id,
                        position,
                        snapshots
                            .iter()
                            .find(|(layer, _)| *layer == id)
                            .map(|(_, s)| s),
                    )?;
                    if !captures.is_empty() {
                        metal::encode_copy_offset_f32(
                            ctx,
                            &enc,
                            &self.x,
                            0,
                            &captures[id].0,
                            4 * c.hidden,
                        )
                        .map_err(|e| e.to_string())?;
                        metal::encode_copy_offset_f32(
                            ctx,
                            &enc,
                            self.ffn.next_pre(),
                            0,
                            &captures[id].1,
                            4,
                        )
                        .map_err(|e| e.to_string())?;
                        self.trace_copies += 2;
                    }
                }
                if with_logits {
                    metal::encode_mhc4_collapse(
                        ctx,
                        &enc,
                        c.hidden,
                        &self.x,
                        self.ffn.next_pre(),
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
                Ok(())
            })();
            enc.end();
            encoded?;
            let encode_ms = started.map(|s| s.elapsed().as_secs_f64() * 1000.0);
            let wait = std::time::Instant::now();
            let result = metal::commit_and_wait(&command).map_err(|e| e.to_string());
            self.commands += 1;
            if let Some(encode_ms) = encode_ms {
                profile::submitted(
                    "gpu_step",
                    [position, start, end, 0],
                    encode_ms,
                    wait.elapsed().as_secs_f64() * 1000.0,
                    result.is_ok(),
                    command.GPUStartTime(),
                    command.GPUEndTime(),
                );
            }
            result?;
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
