//! Serial command segments separated only by Engram CPU boundaries.
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
    let engram = mul(engrams, add(6144, mul(5, h)?)?)?;
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
            add(mul(5, layers)?, mul(2, engrams)?)?,
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
        mut trace: Option<&mut dyn FnMut(LayerTrace<'_>) -> Result<()>>,
        mut capture: Option<&mut dyn capture::Capture>,
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
        let mut engrams = Vec::new();
        for hash in &model.engram_hash.layers {
            let id = hash.layer;
            let e = model.layers[id].engram.as_ref().ok_or("missing Engram")?;
            let values = model
                .engram_rows
                .decode(id, rows.get(&id).ok_or("missing Engram hashes")?)?;
            engrams.push((
                upload(ctx, &values, vec![values.len() as u64])?,
                MetalTensor::zeros_dtype_unstaged(
                    ctx,
                    vec![e.projection.output as u64],
                    GgmlType::F32,
                )
                .map_err(|e| e.to_string())?,
            ));
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
        let boundaries: Vec<_> = std::iter::once(0)
            .chain(model.engram_hash.layers.iter().map(|hash| hash.layer))
            .chain(std::iter::once(model.layers.len()))
            .collect();
        for (segment, boundary) in boundaries.windows(2).enumerate() {
            let (start, end) = (boundary[0], boundary[1]);
            cancelled(cancel)?;
            if segment > 0 {
                let e = model.layers[start]
                    .engram
                    .as_ref()
                    .ok_or("missing Engram")?;
                let mut x = f32_slice(&self.x)?.to_vec();
                e.gate
                    .inject(&mut x, f32_slice(&engrams[segment - 1].1)?, c.norm_eps)?;
                write(&self.x, &x)?;
            }
            let started = profile::start();
            let _wall = profile::Wall("gpu_step", [position, start, end, 0], started);
            let command = ctx
                .queue
                .commandBuffer()
                .ok_or("Metal command buffer allocation")?;
            let enc = KernelEncoder::begin(&command);
            let encoded = (|| -> Result<()> {
                if segment == 0 {
                    for (slot, hash) in model.engram_hash.layers.iter().enumerate() {
                        let id = hash.layer;
                        let p = &model.layers[id].engram.as_ref().unwrap().projection;
                        p.encode(&enc, &engrams[slot].0, &engrams[slot].1)
                            .map_err(|e| e.to_string())?;
                    }
                }
                for id in start..end {
                    cancelled(cancel)?;
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
                if end == model.layers.len() {
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
        let logits = f32_slice(&self.logits)?;
        op::require(logits.iter().all(|v| v.is_finite()), "non-finite logits")?;
        Ok(logits.to_vec())
    }
}
