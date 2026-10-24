use super::super::ced::{Arithmetic, CedCounters};
use super::super::ced_plan::{CedPlan, chunk_views, row_view};
use super::*;
use std::ops::Range;

#[cfg(test)]
#[path = "ced/tests/reuse.rs"]
mod reuse;

fn maximum_decoder_rows(plan: &CedPlan, arithmetic: Arithmetic) -> Result<Option<Range<usize>>> {
    if arithmetic.scalar() {
        return Ok(None);
    }
    let tail = prefill::original_scalar_tail(plan.prompt_rows);
    let mut maximum: Option<Range<usize>> = None;
    for layer in &plan.decoder {
        let end = tail.unwrap_or(layer.output_rows.end);
        for chunk in chunk_views(layer.output_rows.start..end, prefill::DEFAULT_CHUNK_ROWS)
            .map_err(|e| e.to_string())?
        {
            let range = chunk.absolute_rows;
            if maximum.as_ref().is_none_or(|old| range.len() > old.len()) {
                maximum = Some(range);
            }
        }
    }
    Ok(maximum)
}

fn checked_outer_elements(hidden: usize, rows: usize) -> Result<(u64, u64)> {
    op::require(
        hidden > 0 && (1..=prefill::DEFAULT_CHUNK_ROWS).contains(&rows),
        "CED scratch prefix geometry",
    )?;
    let output = hidden
        .checked_mul(rows)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or("CED scratch prefix overflow")?;
    let half = output.checked_mul(4).ok_or("CED scratch prefix overflow")?;
    half.checked_mul(4)
        .ok_or("CED scratch prefix byte overflow")?;
    Ok((half, output))
}

fn check_outer_prefix(t: &MetalTensor, elements: u64) -> Result<()> {
    op::require(
        t.dtype == GgmlType::F32
            && elements <= t.n_elements()
            && t.offset.is_multiple_of(4)
            && elements
                .checked_mul(4)
                .and_then(|bytes| t.offset.checked_add(bytes))
                .is_some_and(|end| end <= t.buffer.length() as u64),
        "CED scratch prefix backing extent",
    )
}

pub(in super::super) fn decoder_chunk_widths(
    model: &Forward<'_>,
    capacity: usize,
    rows: usize,
) -> Result<Vec<usize>> {
    decoder_chunk_widths_from_geometry(&model.config, model.layers.len(), capacity, rows)
}

pub(in super::super) fn decoder_chunk_widths_from_geometry(
    c: &Config,
    layers: usize,
    capacity: usize,
    rows: usize,
) -> Result<Vec<usize>> {
    validate_chunk_geometry(c, capacity, rows)?;
    let mut widths = vec![4 * c.hidden, c.hidden];
    widths.extend(prefill::HcRows::widths(c));
    widths.extend(ffn_gpu::Rows::widths(c, layers));
    widths.extend(attention_gpu::Rows::widths_with_controls(c, capacity));
    widths.extend(
        attention_gpu::Rows::select_widths(c, capacity, rows)?
            .into_iter()
            .filter(|&width| width != 0),
    );
    widths.extend(attention_gpu::Rows::tile_widths(c, rows)?);
    Ok(widths)
}

pub(in super::super) struct Storage {
    residual: MetalTensor,
    mix: MetalTensor,
    controls: attention_gpu::QueryControls,
    range: Range<usize>,
    hidden: usize,
    allocation_base: u64,
    peak: std::cell::Cell<u64>,
}

impl Storage {
    pub(in super::super) fn new(
        model: &Forward<'_>,
        capacity: usize,
        plan: &CedPlan,
    ) -> std::result::Result<Self, StateError> {
        let allocation_base = model.ctx.current_allocated_size();
        op::require(
            attention_gpu::QueryControls::logical_bytes(
                &model.config,
                capacity,
                plan.candidate_rows.clone(),
            )? == plan
                .memory
                .candidate_bytes
                .checked_add(plan.memory.selected_bytes)
                .ok_or("CED control memory overflow")?,
            "CED retained control ledger differs from allocation geometry",
        )?;
        let hidden = model.config.hidden;
        let width = hidden.checked_mul(4).ok_or("CED residual width overflow")?;
        let residual =
            prefill::allocate(model.ctx, width, plan.boundary_rows.len(), GgmlType::F32)?;
        let mix = prefill::allocate(model.ctx, 4, plan.boundary_rows.len(), GgmlType::F32)?;
        let controls = attention_gpu::QueryControls::new(
            model.ctx,
            &model.config,
            capacity,
            plan.candidate_rows.clone(),
        )?;
        let storage = Self {
            residual,
            mix,
            controls,
            range: plan.boundary_rows.clone(),
            hidden,
            allocation_base,
            peak: std::cell::Cell::new(0),
        };
        storage.observe(model.ctx);
        Ok(storage)
    }

    fn observe(&self, ctx: &MetalContext) {
        // This samples allocation, not admission; do not resample host/task budgets.
        self.peak.set(
            self.peak.get().max(
                ctx.current_allocated_size()
                    .saturating_sub(self.allocation_base),
            ),
        );
    }

    fn input(&self, rows: Range<usize>) -> Result<(MetalTensor, MetalTensor)> {
        let view = row_view(&self.range, rows).map_err(|e| e.to_string())?;
        let n = view.absolute_rows.len();
        let width = self
            .hidden
            .checked_mul(4)
            .ok_or("CED residual width overflow")?;
        Ok((
            self.residual.view_subrange(
                (view.row_offset * width) as u64,
                vec![self.hidden as u64, 4, n as u64],
            ),
            self.mix
                .view_subrange((view.row_offset * 4) as u64, vec![4, n as u64]),
        ))
    }

    fn retain(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        start: usize,
        rows: usize,
        x: &MetalTensor,
        incoming: &MetalTensor,
    ) -> Result<()> {
        let end = start.checked_add(rows).ok_or("CED boundary row overflow")?;
        let range = start.max(self.range.start)..end.min(self.range.end);
        if range.is_empty() {
            return Ok(());
        }
        let n = range.len();
        let first = range.start - start;
        let (out, mix) = self.input(range)?;
        let width = self.hidden * 4;
        let x = x.view_subrange((first * width) as u64, vec![width as u64, n as u64]);
        let incoming = incoming.view_subrange((first * 4) as u64, vec![4, n as u64]);
        metal::encode_copy_offset_f32(ctx, enc, &x, 0, &out, n * width)
            .map_err(|e| e.to_string())?;
        metal::encode_copy_offset_f32(ctx, enc, &incoming, 0, &mix, n * 4)
            .map_err(|e| e.to_string())
    }
}

pub(in super::super) struct Output {
    pub logits: Vec<f32>,
    pub counters: CedCounters,
    pub observed_incremental_metal_bytes: u64,
}

struct BlockRows {
    arithmetic: prefill::RowsArithmetic,
    half: MetalTensor,
    output: MetalTensor,
    hc: prefill::HcRows,
    ffn: ffn_gpu::Rows,
    attention: attention_gpu::Rows,
}

impl BlockRows {
    fn checked_prefix(
        &self,
        c: &Config,
        capacity: usize,
        layers: usize,
        controls: &attention_gpu::QueryControlView,
    ) -> Result<Self> {
        let n = controls.rows();
        let (half, output) = checked_outer_elements(c.hidden, n)?;
        op::require(
            self.arithmetic == prefill::RowsArithmetic::Packed && n <= self.hc.rows,
            "CED scratch prefix requires packed maximum backing",
        )?;
        check_outer_prefix(&self.half, half)?;
        check_outer_prefix(&self.output, output)?;
        let hc = self.hc.checked_prefix(c, n)?;
        let attention = self
            .attention
            .checked_prefix_with_controls(c, capacity, controls)?;
        // FFN clears the compact active status slab only after all other views
        // have checked successfully and the previous unit has completed.
        let ffn = self.ffn.checked_prefix(c, n, layers)?;
        Ok(Self {
            arithmetic: self.arithmetic,
            half: self
                .half
                .view_subrange(0, vec![c.hidden as u64, 4, n as u64]),
            output: self
                .output
                .view_subrange(0, vec![c.hidden as u64, n as u64]),
            hc,
            ffn,
            attention,
        })
    }

    fn new(
        model: &Forward<'_>,
        capacity: usize,
        rows: usize,
        controls: Option<&attention_gpu::QueryControlView>,
    ) -> Result<Self> {
        let _allocation = profile::Wall("ced_block_rows_allocate", [0; 4], profile::start());
        let c = &model.config;
        let arithmetic = if controls.is_some() {
            prefill::RowsArithmetic::Packed
        } else {
            prefill::RowsArithmetic::Ordinary
        };
        Ok(Self {
            arithmetic,
            half: prefill::allocate(model.ctx, 4 * c.hidden, rows, GgmlType::F32)?
                .view_bytes(0, vec![c.hidden as u64, 4, rows as u64]),
            output: prefill::allocate(model.ctx, c.hidden, rows, GgmlType::F32)?,
            hc: prefill::HcRows::new(model.ctx, c, rows)?,
            ffn: ffn_gpu::Rows::with_arithmetic(
                model.ctx,
                c,
                rows,
                model.layers.len(),
                arithmetic,
            )?,
            attention: match controls {
                Some(controls) => {
                    attention_gpu::Rows::with_controls(model.ctx, c, capacity, controls)?
                }
                None => attention_gpu::Rows::new(model.ctx, c, capacity, rows)?,
            },
        })
    }
}

struct DecoderRowsPool {
    maximum: Option<Range<usize>>,
    backing: Option<BlockRows>,
}

struct LocalSeed {
    x: MetalTensor,
    incoming: MetalTensor,
    collapsed: MetalTensor,
    normalized: MetalTensor,
    attention: attention_gpu::SeedRows,
}

impl LocalSeed {
    fn new(model: &Forward<'_>, storage: &Storage, range: Range<usize>) -> Result<Self> {
        let c = &model.config;
        let n = range.len();
        let (x, incoming) = storage.input(range)?;
        let seed = Self {
            x,
            incoming,
            collapsed: prefill::allocate(model.ctx, c.hidden, n, GgmlType::F32)?,
            normalized: prefill::allocate(model.ctx, c.hidden, n, GgmlType::F32)?,
            attention: attention_gpu::SeedRows::new(model.ctx, c, n)?,
        };
        #[cfg(test)]
        reuse::seed_allocated(&seed);
        Ok(seed)
    }
}

impl DecoderRowsPool {
    fn checked_rows(
        &mut self,
        model: &Forward<'_>,
        capacity: usize,
        storage: &Storage,
        controls: &attention_gpu::QueryControlView,
    ) -> Result<BlockRows> {
        #[cfg(test)]
        if reuse::fresh_decoder_rows() {
            return BlockRows::new(model, capacity, controls.rows(), Some(controls));
        }
        let maximum = self
            .maximum
            .as_ref()
            .ok_or("CED packed decoder without maximum")?;
        op::require(
            controls.rows() > 0 && controls.rows() <= maximum.len(),
            "CED decoder exceeds planned maximum",
        )?;
        if self.backing.is_none() {
            let controls = storage
                .controls
                .checked_tile_view(maximum.start, maximum.len())?;
            #[cfg(test)]
            super::super::ced::tests::checkpoint("ced_pool_allocate", None)?;
            self.backing = Some(BlockRows::new(
                model,
                capacity,
                maximum.len(),
                Some(&controls),
            )?);
            #[cfg(test)]
            reuse::pool_born(model.ctx, self.backing.as_ref().unwrap());
        }
        #[cfg(test)]
        super::super::ced::tests::checkpoint("ced_pool_view", None)?;
        self.backing.as_ref().unwrap().checked_prefix(
            &model.config,
            capacity,
            model.layers.len(),
            controls,
        )
    }
}

impl Workspace {
    fn ced_command(
        &mut self,
        model: &Forward<'_>,
        role: &'static str,
        dimensions: [usize; 4],
        storage: &Storage,
        counters: &mut CedCounters,
        encode: impl FnOnce(&mut Self, &KernelEncoder) -> Result<()>,
    ) -> Result<()> {
        let started = profile::start();
        let command = model
            .ctx
            .queue
            .commandBuffer()
            .ok_or("Metal command buffer allocation")?;
        let enc = KernelEncoder::begin(&command);
        let encoded = encode(self, &enc);
        enc.end();
        storage.observe(model.ctx);
        encoded?;
        #[cfg(test)]
        let _completion_fault = super::super::ced::tests::completion_fault(role);
        counters.encoded_commands = counters
            .encoded_commands
            .checked_add(1)
            .ok_or("CED command overflow")?;
        let encode_ms = started.map(|s| s.elapsed().as_secs_f64() * 1000.0);
        let wait = std::time::Instant::now();
        let result = metal::commit_and_wait(&command).map_err(|e| e.to_string());
        storage.observe(model.ctx);
        self.commands += 1;
        if let Some(encode_ms) = encode_ms {
            profile::submitted(
                role,
                dimensions,
                encode_ms,
                wait.elapsed().as_secs_f64() * 1000.0,
                result.is_ok(),
                command.GPUStartTime(),
                command.GPUEndTime(),
            );
        }
        result?;
        counters.completed_commands = counters
            .completed_commands
            .checked_add(1)
            .ok_or("CED command overflow")?;
        Ok(())
    }

    fn ced_block(
        &mut self,
        model: &Forward<'_>,
        enc: &KernelEncoder,
        layer: usize,
        start: usize,
        x: &MetalTensor,
        incoming: &MetalTensor,
        rows: &BlockRows,
    ) -> Result<()> {
        let ctx = model.ctx;
        let c = &model.config;
        let weights = &model.layers[layer];
        rows.attention.reset_grouped_layouts();
        rows.hc
            .encode_with_arithmetic(
                ctx,
                enc,
                c,
                &weights.attn_hc.projection,
                &self.constants[layer],
                x,
                incoming,
                rows.arithmetic,
            )
            .map_err(|e| e.to_string())?;
        self.attention.encode_rows_prepare(
            ctx,
            enc,
            layer,
            start,
            &weights.attention,
            &rows.hc.normalized,
            &rows.attention,
        )?;
        self.attention.encode_rows_serial(
            ctx,
            enc,
            layer,
            start,
            &weights.attention,
            &rows.hc.normalized,
            &rows.attention,
        )?;
        self.attention
            .encode_rows_tiles(ctx, enc, layer, start, &rows.attention)?;
        self.attention
            .finish_rows(ctx, enc, &weights.attention, &rows.attention, &rows.output)?;
        metal::encode_mhc4_post_rows(
            ctx,
            enc,
            c.hidden,
            rows.hc.rows,
            &rows.output,
            x,
            &rows.hc.post,
            &rows.hc.comb,
            &rows.half,
        )
        .map_err(|e| e.to_string())?;
        self.ffn
            .encode_rows(
                ctx,
                enc,
                c,
                weights,
                layer,
                &rows.half,
                &rows.hc.pre,
                x,
                &rows.ffn,
            )
            .map_err(|e| e.to_string())?;
        metal::encode_copy_offset_f32(ctx, enc, &rows.ffn.hc.pre, 0, incoming, 4 * rows.hc.rows)
            .map_err(|e| e.to_string())
    }

    fn ced_head(
        &self,
        model: &Forward<'_>,
        enc: &KernelEncoder,
        x: &MetalTensor,
        incoming: &MetalTensor,
        rows: usize,
    ) -> Result<()> {
        let c = &model.config;
        let last = prefill::row(x, 4 * c.hidden, rows - 1).view_bytes(0, vec![c.hidden as u64, 4]);
        metal::encode_mhc4_collapse(
            model.ctx,
            enc,
            c.hidden,
            &last,
            &prefill::row(incoming, 4, rows - 1),
            &self.collapsed,
        )
        .map_err(|e| e.to_string())?;
        metal::encode_rms_norm_mul_rows_f32(
            model.ctx,
            enc,
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
            .encode(enc, &self.normalized, &self.logits)
            .map_err(|e| e.to_string())
    }

    fn ced_encoder_rows(
        &mut self,
        model: &Forward<'_>,
        plan: &CedPlan,
        start: usize,
        tokens: &[u32],
        hashes: &[engram::LayerRows],
        cancel: Option<&AtomicBool>,
        storage: &mut Storage,
        counters: &mut CedCounters,
    ) -> Result<()> {
        let ctx = model.ctx;
        let c = &model.config;
        let n = tokens.len();
        let prepare = profile::Wall("ced_encoder_host_prepare", [0; 4], profile::start());
        let mut embeddings = Vec::with_capacity(n * 4 * c.hidden);
        for &token in tokens {
            embeddings.extend(decode_rows(model.embedding, token as usize, 1)?.repeat(4));
        }
        let x = upload(ctx, &embeddings, vec![c.hidden as u64, 4, n as u64])?;
        let incoming = upload(ctx, &[1.0, 0.0, 0.0, 0.0].repeat(n), vec![4, n as u64])?;
        let scratch = BlockRows::new(model, self.attention.capacity(), n, None)?;
        let publication = attention_gpu::PublicationRows::new(ctx, c, n)?;
        let mut engrams = Vec::new();
        let engram_prepare = profile::Wall("ced_encoder_engram_prepare", [0; 4], profile::start());
        for slot in &self.engrams {
            op::require(
                plan.encoder_layers.contains(&slot.layer),
                "CED decoder Engram",
            )?;
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
        drop(engram_prepare);
        #[cfg(test)]
        reuse::encoder_allocated(&x, &incoming, &scratch);
        storage.observe(ctx);
        cancelled(cancel)?;
        self.attention.begin_ced_encoder(start, n)?;
        drop(prepare);
        self.ced_command(
            model,
            "ced_encoder_rows",
            [start, n, plan.publication_layer, 0],
            storage,
            counters,
            |this, enc| {
                for id in plan.encoder_layers.clone() {
                    cancelled(cancel)?;
                    if let Some((_, input, projected)) =
                        engrams.iter().find(|(id0, _, _)| *id0 == id)
                    {
                        model.layers[id]
                            .engram
                            .as_ref()
                            .unwrap()
                            .projection
                            .encode_rows(enc, input, projected, n)
                            .map_err(|e| e.to_string())?;
                        this.engrams
                            .iter()
                            .find(|slot| slot.layer == id)
                            .unwrap()
                            .gate
                            .encode(ctx, enc, &x, projected, n)?;
                    }
                    this.ced_block(model, enc, id, start, &x, &incoming, &scratch)?;
                }
                #[cfg(test)]
                super::super::ced::tests::checkpoint("ced_publication", cancel)?;
                scratch
                    .hc
                    .normalize_input(
                        ctx,
                        enc,
                        c,
                        &this.constants[plan.publication_layer][2],
                        &x,
                        &incoming,
                    )
                    .map_err(|e| e.to_string())?;
                this.attention.publish_ced_rows(
                    ctx,
                    enc,
                    plan.publication_layer,
                    start,
                    &model.layers[plan.publication_layer].attention,
                    &scratch.hc.normalized,
                    &publication,
                )?;
                storage.retain(ctx, enc, start, n, &x, &incoming)?;
                cancelled(cancel)
            },
        )?;
        scratch
            .ffn
            .validate_status_layers(plan.encoder_layers.clone())?;
        self.attention.complete_ced_encoder(start, n)?;
        #[cfg(test)]
        super::super::ced::qualification::boundary(start, n, &x, &incoming)?;
        for id in plan.encoder_layers.clone() {
            counters.encoder(id, n, model.schedule.produces_index(id))?;
        }
        counters.publication(plan.publication_layer, n)?;
        #[cfg(test)]
        super::super::ced::tests::checkpoint("ced_encoder_complete", cancel)?;
        cancelled(cancel)
    }

    fn ced_encoder_scalar(
        &mut self,
        model: &Forward<'_>,
        plan: &CedPlan,
        position: usize,
        token: u32,
        hashes: &engram::LayerRows,
        cancel: Option<&AtomicBool>,
        storage: &mut Storage,
        counters: &mut CedCounters,
    ) -> Result<()> {
        let ctx = model.ctx;
        write(
            &self.x,
            &decode_rows(model.embedding, token as usize, 1)?.repeat(4),
        )?;
        for slot in &self.engrams {
            op::require(
                plan.encoder_layers.contains(&slot.layer),
                "CED decoder Engram",
            )?;
            let values = model.engram_rows.decode(
                slot.layer,
                hashes.get(&slot.layer).ok_or("missing Engram hashes")?,
            )?;
            write(&slot.input, &values)?;
        }
        self.attention.begin_ced_encoder(position, 1)?;
        self.ced_command(
            model,
            "ced_encoder_scalar",
            [position, 1, plan.publication_layer, 0],
            storage,
            counters,
            |this, enc| {
                for slot in &this.engrams {
                    model.layers[slot.layer]
                        .engram
                        .as_ref()
                        .unwrap()
                        .projection
                        .encode(enc, &slot.input, &slot.projected)
                        .map_err(|e| e.to_string())?;
                }
                for id in plan.encoder_layers.clone() {
                    cancelled(cancel)?;
                    if let Some(slot) = this.engrams.iter().find(|slot| slot.layer == id) {
                        slot.gate.encode(ctx, enc, &this.x, &slot.projected, 1)?;
                    }
                    this.layer(model, enc, id, position, None, true)?;
                }
                #[cfg(test)]
                super::super::ced::tests::checkpoint("ced_publication", cancel)?;
                this.normalize_scalar_input(
                    model,
                    enc,
                    plan.publication_layer,
                    &this.x,
                    this.ffn.next_pre(),
                )?;
                this.attention.publish_ced_scalar(
                    ctx,
                    enc,
                    plan.publication_layer,
                    position,
                    &model.layers[plan.publication_layer].attention,
                    &this.normalized,
                )?;
                storage.retain(ctx, enc, position, 1, &this.x, this.ffn.next_pre())?;
                cancelled(cancel)
            },
        )?;
        for id in plan.encoder_layers.clone() {
            op::require(
                self.ffn.status(id) == metal::ROUTE_STATUS_READY,
                "CED scalar encoder FFN status",
            )?;
        }
        self.attention.complete_ced_encoder(position, 1)?;
        #[cfg(test)]
        super::super::ced::qualification::boundary(position, 1, &self.x, self.ffn.next_pre())?;
        for id in plan.encoder_layers.clone() {
            counters.encoder(id, 1, model.schedule.produces_index(id))?;
        }
        counters.publication(plan.publication_layer, 1)?;
        #[cfg(test)]
        super::super::ced::tests::checkpoint("ced_encoder_complete", cancel)?;
        storage.observe(ctx);
        cancelled(cancel)
    }

    fn ced_seed(
        &mut self,
        model: &Forward<'_>,
        layer: usize,
        range: Range<usize>,
        arithmetic: Arithmetic,
        cancel: Option<&AtomicBool>,
        storage: &mut Storage,
        counters: &mut CedCounters,
    ) -> Result<()> {
        let ctx = model.ctx;
        let n = range.len();
        let seed = LocalSeed::new(model, storage, range.clone())?;
        storage.observe(ctx);
        cancelled(cancel)?;
        self.attention.begin_ced_seed(layer, range.start, n)?;
        self.ced_command(
            model,
            "ced_local_seed",
            [layer, range.start, n, 0],
            storage,
            counters,
            |this, enc| {
                this.ced_encode_seed(model, enc, layer, &range, arithmetic, &seed)?;
                cancelled(cancel)
            },
        )?;
        self.attention.complete_ced_seed(layer, range.start, n)?;
        counters.seed(layer, n)?;
        #[cfg(test)]
        super::super::ced::tests::checkpoint("ced_seed_complete", cancel)?;
        cancelled(cancel)
    }

    fn ced_encode_seed(
        &mut self,
        model: &Forward<'_>,
        enc: &KernelEncoder,
        layer: usize,
        range: &Range<usize>,
        arithmetic: Arithmetic,
        seed: &LocalSeed,
    ) -> Result<()> {
        let ctx = model.ctx;
        let c = &model.config;
        let n = range.len();
        if arithmetic.scalar() {
            op::require(n == 1, "scalar CED seed width")?;
            self.normalize_scalar_input(
                model,
                enc,
                layer,
                &seed.x.view_bytes(0, vec![c.hidden as u64, 4]),
                &seed.incoming.view_bytes(0, vec![4]),
            )?;
            self.attention.seed_ced_scalar(
                ctx,
                enc,
                layer,
                range.start,
                &model.layers[layer].attention,
                &self.normalized,
            )
        } else {
            metal::encode_mhc4_collapse_rows(
                ctx,
                enc,
                c.hidden,
                n,
                &seed.x,
                &seed.incoming,
                &seed.collapsed,
            )
            .map_err(|e| e.to_string())?;
            metal::encode_rms_norm_mul_rows_f32(
                ctx,
                enc,
                &seed.collapsed,
                &self.constants[layer][2],
                &seed.normalized,
                n,
                c.hidden,
                c.norm_eps,
            )
            .map_err(|e| e.to_string())?;
            self.attention.seed_ced_rows(
                ctx,
                enc,
                layer,
                range.start,
                &model.layers[layer].attention,
                &seed.normalized,
                &seed.attention,
            )
        }
    }

    fn ced_decoder_rows(
        &mut self,
        model: &Forward<'_>,
        layer: usize,
        range: Range<usize>,
        seed_range: Option<Range<usize>>,
        final_logits: bool,
        cancel: Option<&AtomicBool>,
        storage: &mut Storage,
        counters: &mut CedCounters,
        pool: &mut DecoderRowsPool,
    ) -> Result<()> {
        let ctx = model.ctx;
        let n = range.len();
        let seed = seed_range
            .as_ref()
            .map(|range| LocalSeed::new(model, storage, range.clone()))
            .transpose()?;
        let (x, incoming) = storage.input(range.clone())?;
        let controls = storage.controls.checked_tile_view(range.start, n)?;
        let scratch = pool.checked_rows(model, self.attention.capacity(), storage, &controls)?;
        storage.observe(ctx);
        cancelled(cancel)?;
        if let Some(seed) = &seed_range {
            self.attention
                .begin_ced_seed(layer, seed.start, seed.len())?;
        } else {
            self.attention
                .begin_ced_decoder_chunk(layer, range.start, n)?;
        }
        self.ced_command(
            model,
            if seed.is_some() {
                "ced_seed_decoder_rows"
            } else {
                "ced_decoder_rows"
            },
            [
                layer,
                range.start,
                n,
                seed_range.as_ref().map_or(0, Range::len),
            ],
            storage,
            counters,
            |this, enc| {
                if let (Some(range0), Some(seed)) = (&seed_range, &seed) {
                    this.ced_encode_seed(model, enc, layer, range0, Arithmetic::Production, seed)?;
                    // Encode-only seam: neither range has completed or been submitted.
                    #[cfg(test)]
                    super::super::ced::tests::checkpoint("ced_seed_encoded", cancel)?;
                    cancelled(cancel)?;
                    this.attention
                        .chain_ced_seed_to_decoder(layer, range.start, n)?;
                }
                this.ced_block(model, enc, layer, range.start, &x, &incoming, &scratch)?;
                if final_logits {
                    this.ced_head(model, enc, &x, &incoming, n)?;
                }
                cancelled(cancel)
            },
        )?;
        #[cfg(test)]
        if seed.is_some() {
            reuse::compound_completed(&self.attention, &scratch, layer);
        }
        scratch.ffn.validate_status_layers(layer..layer + 1)?;
        self.attention
            .complete_ced_decoder_chunk(layer, range.start, n)?;
        if let Some(range) = &seed_range {
            counters.seed(layer, range.len())?;
            #[cfg(test)]
            super::super::ced::tests::checkpoint("ced_seed_complete", cancel)?;
        }
        #[cfg(test)]
        super::super::ced::qualification::controls(layer, range.start, n, &controls)?;
        counters.decoder(layer, n, model.schedule.produces_index(layer))?;
        #[cfg(test)]
        super::super::ced::tests::checkpoint("ced_decoder_complete", cancel)?;
        #[cfg(test)]
        reuse::decoder_completed(ctx, &scratch, &controls, layer, &x, &incoming);
        cancelled(cancel)
    }

    fn ced_decoder_scalar(
        &mut self,
        model: &Forward<'_>,
        layer: usize,
        position: usize,
        final_logits: bool,
        cancel: Option<&AtomicBool>,
        storage: &mut Storage,
        counters: &mut CedCounters,
    ) -> Result<()> {
        let ctx = model.ctx;
        let c = &model.config;
        let (x, incoming) = storage.input(position..position + 1)?;
        let controls = storage.controls.checked_tile_view(position, 1)?;
        self.attention
            .begin_ced_decoder_scalar(layer, position, &controls)?;
        cancelled(cancel)?;
        self.ced_command(
            model,
            "ced_decoder_scalar",
            [layer, position, 1, 0],
            storage,
            counters,
            |this, enc| {
                metal::encode_copy_offset_f32(ctx, enc, &x, 0, &this.x, 4 * c.hidden)
                    .map_err(|e| e.to_string())?;
                metal::encode_copy_offset_f32(ctx, enc, &incoming, 0, this.ffn.next_pre(), 4)
                    .map_err(|e| e.to_string())?;
                this.layer(model, enc, layer, position, None, true)?;
                metal::encode_copy_offset_f32(ctx, enc, &this.x, 0, &x, 4 * c.hidden)
                    .map_err(|e| e.to_string())?;
                metal::encode_copy_offset_f32(ctx, enc, this.ffn.next_pre(), 0, &incoming, 4)
                    .map_err(|e| e.to_string())?;
                if final_logits {
                    this.ced_head(model, enc, &this.x, this.ffn.next_pre(), 1)?;
                }
                cancelled(cancel)
            },
        )?;
        op::require(
            self.ffn.status(layer) == metal::ROUTE_STATUS_READY,
            "CED scalar decoder FFN status",
        )?;
        self.attention
            .complete_ced_decoder_chunk(layer, position, 1)?;
        #[cfg(test)]
        super::super::ced::qualification::controls(layer, position, 1, &controls)?;
        counters.decoder(layer, 1, model.schedule.produces_index(layer))?;
        #[cfg(test)]
        super::super::ced::tests::checkpoint("ced_decoder_complete", cancel)?;
        storage.observe(ctx);
        cancelled(cancel)
    }

    pub(in super::super) fn run_ced(
        &mut self,
        model: &Forward<'_>,
        plan: &CedPlan,
        tokens: &[u32],
        addresses: &mut engram::AddressState,
        cancel: Option<&AtomicBool>,
        arithmetic: Arithmetic,
        mut storage: Storage,
    ) -> Result<Output> {
        op::require(
            plan.full_chunk_rows == model.prefill_rows,
            "CED Full chunk geometry differs from bound model",
        )?;
        self.commands = 0;
        self.trace_copies = 0;
        let maximum = maximum_decoder_rows(plan, arithmetic)?;
        self.attention.begin_ced(plan)?;
        let chunk_rows = if arithmetic.scalar() {
            1
        } else {
            prefill::DEFAULT_CHUNK_ROWS
        };
        let mut counters = CedCounters::default();
        let encoder_rows = if arithmetic.scalar() {
            1
        } else {
            plan.full_chunk_rows
        };
        for chunk in chunk_views(0..tokens.len(), encoder_rows).map_err(|e| e.to_string())? {
            cancelled(cancel)?;
            let range = chunk.absolute_rows;
            let input = &tokens[range.clone()];
            let mut next = *addresses;
            let mut hashes = Vec::with_capacity(input.len());
            for &token in input {
                let (rows, updated) = next
                    .step_with_config(
                        engram::Token::Compressed(model.token_map[token as usize]),
                        &model.engram_hash,
                    )
                    .map_err(|e| format!("{e:?}"))?;
                hashes.push(rows);
                next = updated;
            }
            objc2::rc::autoreleasepool(|_| {
                if input.len() == 1 {
                    self.ced_encoder_scalar(
                        model,
                        plan,
                        range.start,
                        input[0],
                        &hashes[0],
                        cancel,
                        &mut storage,
                        &mut counters,
                    )
                } else {
                    self.ced_encoder_rows(
                        model,
                        plan,
                        range.start,
                        input,
                        &hashes,
                        cancel,
                        &mut storage,
                        &mut counters,
                    )
                }
            })?;
            *addresses = next;
        }
        #[cfg(test)]
        super::super::ced::qualification::publication(&self.attention, plan.prompt_rows)?;
        let mut pool = DecoderRowsPool {
            maximum,
            backing: None,
        };
        for layer in &plan.decoder {
            let mut fuse_seed =
                !arithmetic.scalar() && layer.fuse_seed_with_packed_output(plan.prompt_rows);
            #[cfg(test)]
            {
                fuse_seed &= !super::super::ced::tests::unfused();
            }
            let seeds = if fuse_seed {
                0..0
            } else {
                layer.seed_rows.clone()
            };
            for chunk in chunk_views(seeds, chunk_rows).map_err(|e| e.to_string())? {
                objc2::rc::autoreleasepool(|_| {
                    self.ced_seed(
                        model,
                        layer.layer,
                        chunk.absolute_rows,
                        arithmetic,
                        cancel,
                        &mut storage,
                        &mut counters,
                    )
                })?;
            }
            let scalar_tail = if arithmetic.scalar() {
                None
            } else {
                plan.scalar_tail()
            };
            let row_end = scalar_tail.unwrap_or(layer.output_rows.end);
            for chunk in chunk_views(layer.output_rows.start..row_end, chunk_rows)
                .map_err(|e| e.to_string())?
            {
                let range = chunk.absolute_rows;
                let final_logits =
                    layer.layer + 1 == model.layers.len() && range.end == plan.prompt_rows;
                objc2::rc::autoreleasepool(|_| {
                    if arithmetic.scalar() {
                        self.ced_decoder_scalar(
                            model,
                            layer.layer,
                            range.start,
                            final_logits,
                            cancel,
                            &mut storage,
                            &mut counters,
                        )
                    } else {
                        self.ced_decoder_rows(
                            model,
                            layer.layer,
                            range,
                            if fuse_seed {
                                Some(layer.seed_rows.clone())
                            } else {
                                None
                            },
                            final_logits,
                            cancel,
                            &mut storage,
                            &mut counters,
                            &mut pool,
                        )
                    }
                })?;
                fuse_seed = false;
            }
            if let Some(position) = scalar_tail {
                objc2::rc::autoreleasepool(|_| {
                    self.ced_decoder_scalar(
                        model,
                        layer.layer,
                        position,
                        layer.layer + 1 == model.layers.len(),
                        cancel,
                        &mut storage,
                        &mut counters,
                    )
                })?;
            }
            #[cfg(test)]
            super::super::ced::qualification::decoder_layer(
                &self.attention,
                layer.layer,
                plan.prompt_rows,
            )?;
        }
        drop(pool);
        cancelled(cancel)?;
        counters.validate(plan)?;
        let logits = f32_slice(&self.logits)?;
        op::require(
            logits.iter().all(|v| v.is_finite()),
            "CED non-finite logits",
        )?;
        let logits = logits.to_vec();
        #[cfg(test)]
        super::super::ced::tests::checkpoint("ced_commit", cancel)?;
        cancelled(cancel)?;
        self.attention.commit_ced(plan.prompt_rows)?;
        Ok(Output {
            logits,
            counters,
            observed_incremental_metal_bytes: storage.peak.get(),
        })
    }
}
