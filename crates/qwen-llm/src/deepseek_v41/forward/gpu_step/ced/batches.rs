use super::super::super::ced_plan::batches::{self as plan_batch, Kind, Unit};
use super::*;

#[cfg(test)]
pub(super) mod tests;

pub(super) fn batch_layers() -> usize {
    #[cfg(test)]
    if let Some(layers) = tests::override_layers() {
        return layers;
    }
    plan_batch::DECODER_BATCH_LAYERS
}

pub(super) struct StatusUnit {
    ffn: Option<MetalTensor>,
    attention: MetalTensor,
    #[cfg(test)]
    canaries: Vec<MetalTensor>,
}

fn status_views(backing: &MetalTensor, units: &[Unit]) -> Result<Vec<StatusUnit>> {
    let words = plan_batch::status_words(units).map_err(|e| e.to_string())?;
    op::require(
        backing.dtype == GgmlType::I32
            && backing.is_writable()
            && backing.offset.is_multiple_of(4)
            && words as u64 <= backing.n_elements()
            && backing
                .offset
                .checked_add(
                    backing
                        .n_elements()
                        .checked_mul(4)
                        .ok_or("CED status ledger overflow")?,
                )
                .is_some_and(|end| end <= backing.buffer.length() as u64),
        "CED status ledger backing",
    )?;
    let mut offset = 0;
    let views: Vec<StatusUnit> = units
        .iter()
        .map(|unit| {
            #[cfg(test)]
            let mut canaries = Vec::with_capacity(5);
            let mut region = |words: usize| {
                let guards = plan_batch::status_guard_words();
                #[cfg(test)]
                canaries.push(backing.view_subrange(offset as u64, vec![guards as u64]));
                offset += guards;
                let view = backing.view_subrange(offset as u64, vec![words as u64]);
                offset += words;
                #[cfg(test)]
                canaries.push(backing.view_subrange(offset as u64, vec![guards as u64]));
                offset += guards;
                view
            };
            let ffn = if unit.kind == Kind::Rows {
                Some(region(unit.rows.len()))
            } else {
                None
            };
            let attention = region(1);
            StatusUnit {
                ffn,
                attention,
                #[cfg(test)]
                canaries,
            }
        })
        .collect();
    #[cfg(test)]
    let views = {
        let mut views = views;
        if (offset as u64) < backing.n_elements() {
            views
                .last_mut()
                .ok_or("CED empty status group")?
                .canaries
                .push(
                    backing
                        .view_subrange(offset as u64, vec![backing.n_elements() - offset as u64]),
                );
        }
        views
    };
    Ok(views)
}

fn attention_status(t: &MetalTensor) -> Result<()> {
    let status = unsafe {
        *t.buffer
            .contents()
            .as_ptr()
            .cast::<i32>()
            .add((t.offset / 4) as usize)
    };
    op::require(status == 0, &format!("CED attention status {status}"))
}

impl Workspace {
    pub(super) fn ced_decoder_groups(
        &mut self,
        model: &Forward<'_>,
        plan: &CedPlan,
        cancel: Option<&AtomicBool>,
        storage: &mut Storage,
        counters: &mut CedCounters,
        pool: &mut DecoderRowsPool,
    ) -> Result<()> {
        let ctx = model.ctx;
        let layers = batch_selection(plan)?.layers;
        op::require(
            layers > 0,
            "CED group execution requires structurally bounded selection",
        )?;
        let words = plan_batch::maximum_status_words(plan, layers).map_err(|e| e.to_string())?;
        #[cfg(test)]
        tests::allocation_checkpoint(tests::AllocationSite::Ledger, words * 4)
            .map_err(|e| e.to_string())?;
        let ledger = prefill::allocate(ctx, words, 1, GgmlType::I32)?;
        let seed_range = plan
            .decoder
            .iter()
            .flat_map(|l| chunk_views(l.seed_rows.clone(), prefill::DEFAULT_CHUNK_ROWS).unwrap())
            .map(|c| c.absolute_rows)
            .max_by_key(Range::len);
        let seed = seed_range
            .map(|range| LocalSeed::new(model, storage, range))
            .transpose()?;
        for group in plan.decoder.chunks(layers) {
            objc2::rc::autoreleasepool(|_| -> Result<()> {
                cancelled(cancel)?;
                let layer_range = group[0].layer..group.last().unwrap().layer + 1;
                let units =
                    plan_batch::units(plan, layer_range.clone()).map_err(|e| e.to_string())?;
                let statuses = status_views(&ledger, &units)?;
                #[cfg(test)]
                let mut diagnostics = tests::Diagnostics::new(model, storage, &units)?;
                let generation = self
                    .attention
                    .begin_ced_group(layer_range.clone(), &storage.controls)?;
                self.ced_command(
                    model,
                    "ced_decoder_batch",
                    [layer_range.start, layer_range.len(), units.len(), 0],
                    storage,
                    counters,
                    |this, enc| {
                        #[cfg(test)]
                        tests::initialize_status_canaries(ctx, enc, &ledger)?;
                        for (unit, status) in units.iter().zip(&statuses) {
                            cancelled(cancel)?;
                            let layer = unit.layer;
                            let range = unit.rows.clone();
                            let n = range.len();
                            match unit.kind {
                                Kind::Seed => {
                                    let seed = seed
                                        .as_ref()
                                        .ok_or("CED seed backing missing")?
                                        .checked_view(model, storage, range.clone())?;
                                    this.attention.begin_ced_seed(layer, range.start, n)?;
                                    this.ced_encode_seed(
                                        model,
                                        enc,
                                        layer,
                                        &range,
                                        Arithmetic::Production,
                                        &seed,
                                    )?;
                                }
                                Kind::Rows => {
                                    let controls =
                                        storage.controls.checked_tile_view(range.start, n)?;
                                    let scratch = pool.checked_rows_status(
                                        model,
                                        this.attention.capacity(),
                                        storage,
                                        &controls,
                                        Some((layer, status.ffn.as_ref().unwrap())),
                                    )?;
                                    let (x, incoming) = storage.input(range.clone())?;
                                    this.attention.begin_ced_decoder_chunk(
                                        layer,
                                        range.start,
                                        n,
                                    )?;
                                    this.ced_block(
                                        model,
                                        enc,
                                        layer,
                                        range.start,
                                        &x,
                                        &incoming,
                                        &scratch,
                                    )?;
                                    if layer + 1 == model.layers.len()
                                        && range.end == plan.prompt_rows
                                    {
                                        this.ced_head(model, enc, &x, &incoming, n)?;
                                    }
                                }
                                Kind::Scalar => {
                                    let c = &model.config;
                                    let (x, incoming) = storage.input(range.clone())?;
                                    let controls =
                                        storage.controls.checked_tile_view(range.start, 1)?;
                                    this.attention.begin_ced_decoder_scalar(
                                        layer,
                                        range.start,
                                        &controls,
                                    )?;
                                    metal::encode_copy_offset_f32(
                                        ctx,
                                        enc,
                                        &x,
                                        0,
                                        &this.x,
                                        4 * c.hidden,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    metal::encode_copy_offset_f32(
                                        ctx,
                                        enc,
                                        &incoming,
                                        0,
                                        this.ffn.next_pre(),
                                        4,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    this.layer(model, enc, layer, range.start, None, true)?;
                                    metal::encode_copy_offset_f32(
                                        ctx,
                                        enc,
                                        &this.x,
                                        0,
                                        &x,
                                        4 * c.hidden,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    metal::encode_copy_offset_f32(
                                        ctx,
                                        enc,
                                        this.ffn.next_pre(),
                                        0,
                                        &incoming,
                                        4,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    if layer + 1 == model.layers.len() {
                                        this.ced_head(model, enc, &this.x, this.ffn.next_pre(), 1)?;
                                    }
                                }
                            }
                            #[cfg(test)]
                            tests::before_copies();
                            #[cfg(test)]
                            diagnostics.fault(&this.attention, ctx, enc, unit, status)?;
                            this.attention
                                .encode_ced_status_copy(ctx, enc, &status.attention)?;
                            #[cfg(test)]
                            diagnostics.encode(ctx, enc, storage, unit)?;
                            #[cfg(test)]
                            tests::after_copies();
                            this.attention.stage_ced_group_unit(
                                generation,
                                layer,
                                range.start,
                                n,
                                unit.kind == Kind::Seed,
                            )?;
                            #[cfg(test)]
                            super::super::super::ced::tests::checkpoint(
                                "ced_batch_unit_encoded",
                                cancel,
                            )?;
                        }
                        cancelled(cancel)
                    },
                )?;
                #[cfg(test)]
                tests::check_status_canaries(&statuses)?;
                #[cfg(test)]
                tests::completed_statuses(&self.attention, &units, &statuses);
                #[cfg(test)]
                super::super::super::ced::tests::checkpoint("ced_batch_completed", cancel)?;
                cancelled(cancel)?;
                // The incumbent compound checks its first output FFN before
                // the sticky seed/attention word. Multichunk seeds retire alone.
                for (unit, status) in units.iter().zip(&statuses) {
                    if let Some(ffn) = &status.ffn {
                        ffn_gpu::validate_ced_status(ffn, unit.layer)?;
                    }
                    if unit.kind == Kind::Scalar {
                        op::require(
                            self.ffn.status(unit.layer) == metal::ROUTE_STATUS_READY,
                            "CED scalar decoder FFN status",
                        )?;
                    }
                    let compound_seed = unit.kind == Kind::Seed
                        && plan.decoder[unit.layer - plan.publication_layer]
                            .fuse_seed_with_packed_output(plan.prompt_rows, plan.full_chunk_rows);
                    if !compound_seed {
                        attention_status(&status.attention)?;
                    }
                }
                self.attention.retire_ced_group(generation, layer_range)?;
                for unit in &units {
                    if unit.kind == Kind::Seed {
                        counters.seed(unit.layer, unit.rows.len())?;
                    } else {
                        counters.decoder(
                            unit.layer,
                            unit.rows.len(),
                            model.schedule.produces_index(unit.layer),
                        )?;
                    }
                }
                #[cfg(test)]
                diagnostics.complete(&self.attention, plan.prompt_rows, &units, cancel)?;
                #[cfg(test)]
                tests::group_retired(&self.attention);
                #[cfg(test)]
                super::super::super::ced::tests::checkpoint("ced_batch_retired", cancel)?;
                Ok(())
            })?;
        }
        Ok(())
    }
}

#[cfg(not(test))]
struct PendingGroup {
    command: super::PendingCed,
    units: Vec<Unit>,
    statuses: Vec<StatusUnit>,
}

#[cfg(not(test))]
impl Workspace {
    /// One-group lookahead: submit group N+1 before waiting N. The command count
    /// and arithmetic are unchanged; the cursor is retired speculatively and the
    /// final status check gates the state after all waits.
    pub(super) fn ced_decoder_groups_pipelined(
        &mut self,
        model: &Forward<'_>,
        plan: &CedPlan,
        cancel: Option<&AtomicBool>,
        storage: &mut Storage,
        counters: &mut CedCounters,
        pool: &mut DecoderRowsPool,
    ) -> Result<()> {
        let ctx = model.ctx;
        let layers = batch_selection(plan)?.layers;
        op::require(
            layers > 0,
            "CED group execution requires structurally bounded selection",
        )?;
        let words = plan_batch::maximum_status_words(plan, layers).map_err(|e| e.to_string())?;
        let ledger = prefill::allocate(ctx, words * 2, 1, GgmlType::I32)?;
        let seed_range = plan
            .decoder
            .iter()
            .flat_map(|l| chunk_views(l.seed_rows.clone(), prefill::DEFAULT_CHUNK_ROWS).unwrap())
            .map(|c| c.absolute_rows)
            .max_by_key(Range::len);
        let seed = seed_range
            .map(|range| LocalSeed::new(model, storage, range))
            .transpose()?;
        let groups = plan
            .decoder
            .chunks(layers)
            .map(|group| {
                let layer_range = group[0].layer..group.last().unwrap().layer + 1;
                let units =
                    plan_batch::units(plan, layer_range.clone()).map_err(|e| e.to_string())?;
                Ok((layer_range, units))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut pending: std::collections::VecDeque<PendingGroup> =
            std::collections::VecDeque::new();
        let result = (|| -> Result<()> {
            for (i, (layer_range, units)) in groups.into_iter().enumerate() {
                cancelled(cancel)?;
                let slot = i % 2;
                let backing = ledger.view_subrange((slot * words) as u64, vec![words as u64]);
                let statuses = status_views(&backing, &units)?;
                let generation = self
                    .attention
                    .begin_ced_group(layer_range.clone(), &storage.controls)?;
                let command = self.ced_submit(
                    model,
                    "ced_decoder_batch",
                    [layer_range.start, layer_range.len(), units.len(), 0],
                    storage,
                    counters,
                    |this, enc| {
                        for (unit, status) in units.iter().zip(&statuses) {
                            cancelled(cancel)?;
                            let layer = unit.layer;
                            let range = unit.rows.clone();
                            let n = range.len();
                            match unit.kind {
                                Kind::Seed => {
                                    let seed = seed
                                        .as_ref()
                                        .ok_or("CED seed backing missing")?
                                        .checked_view(model, storage, range.clone())?;
                                    this.attention.begin_ced_seed(layer, range.start, n)?;
                                    this.ced_encode_seed(
                                        model,
                                        enc,
                                        layer,
                                        &range,
                                        Arithmetic::Production,
                                        &seed,
                                    )?;
                                }
                                Kind::Rows => {
                                    let controls =
                                        storage.controls.checked_tile_view(range.start, n)?;
                                    let scratch = pool.checked_rows_status(
                                        model,
                                        this.attention.capacity(),
                                        storage,
                                        &controls,
                                        Some((layer, status.ffn.as_ref().unwrap())),
                                    )?;
                                    let (x, incoming) = storage.input(range.clone())?;
                                    this.attention.begin_ced_decoder_chunk(
                                        layer,
                                        range.start,
                                        n,
                                    )?;
                                    this.ced_block(
                                        model,
                                        enc,
                                        layer,
                                        range.start,
                                        &x,
                                        &incoming,
                                        &scratch,
                                    )?;
                                    if layer + 1 == model.layers.len()
                                        && range.end == plan.prompt_rows
                                    {
                                        this.ced_head(model, enc, &x, &incoming, n)?;
                                    }
                                }
                                Kind::Scalar => {
                                    let c = &model.config;
                                    let (x, incoming) = storage.input(range.clone())?;
                                    let controls =
                                        storage.controls.checked_tile_view(range.start, 1)?;
                                    this.attention.begin_ced_decoder_scalar(
                                        layer,
                                        range.start,
                                        &controls,
                                    )?;
                                    metal::encode_copy_offset_f32(
                                        ctx,
                                        enc,
                                        &x,
                                        0,
                                        &this.x,
                                        4 * c.hidden,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    metal::encode_copy_offset_f32(
                                        ctx,
                                        enc,
                                        &incoming,
                                        0,
                                        this.ffn.next_pre(),
                                        4,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    this.layer(model, enc, layer, range.start, None, true)?;
                                    metal::encode_copy_offset_f32(
                                        ctx,
                                        enc,
                                        &this.x,
                                        0,
                                        &x,
                                        4 * c.hidden,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    metal::encode_copy_offset_f32(
                                        ctx,
                                        enc,
                                        this.ffn.next_pre(),
                                        0,
                                        &incoming,
                                        4,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    if layer + 1 == model.layers.len() {
                                        this.ced_head(model, enc, &this.x, this.ffn.next_pre(), 1)?;
                                    }
                                }
                            }
                            this.attention
                                .encode_ced_status_copy(ctx, enc, &status.attention)?;
                            this.attention.stage_ced_group_unit(
                                generation,
                                layer,
                                range.start,
                                n,
                                unit.kind == Kind::Seed,
                            )?;
                        }
                        cancelled(cancel)
                    },
                )?;
                pending.push_back(PendingGroup {
                    command,
                    units,
                    statuses,
                });
                self.attention
                    .retire_ced_group_speculative(generation, layer_range.clone())?;
                if pending.len() > 1 {
                    let done = pending.pop_front().unwrap();
                    self.finish_ced_group(model, plan, done, storage, counters)?;
                }
            }
            Ok(())
        })();
        if result.is_err() {
            while let Some(done) = pending.pop_front() {
                let _ = self.ced_wait(model, done.command, storage, counters);
            }
        }
        result?;
        while let Some(done) = pending.pop_front() {
            self.finish_ced_group(model, plan, done, storage, counters)?;
        }
        self.attention.checked_ced_status()?;
        Ok(())
    }

    fn finish_ced_group(
        &mut self,
        model: &Forward<'_>,
        plan: &CedPlan,
        done: PendingGroup,
        storage: &mut Storage,
        counters: &mut CedCounters,
    ) -> Result<()> {
        self.ced_wait(model, done.command, storage, counters)?;
        for (unit, status) in done.units.iter().zip(&done.statuses) {
            if let Some(ffn) = &status.ffn {
                ffn_gpu::validate_ced_status(ffn, unit.layer)?;
            }
            if unit.kind == Kind::Scalar {
                op::require(
                    self.ffn.status(unit.layer) == metal::ROUTE_STATUS_READY,
                    "CED scalar decoder FFN status",
                )?;
            }
            let compound_seed = unit.kind == Kind::Seed
                && plan.decoder[unit.layer - plan.publication_layer]
                    .fuse_seed_with_packed_output(plan.prompt_rows, plan.full_chunk_rows);
            if !compound_seed {
                attention_status(&status.attention)?;
            }
        }
        for unit in &done.units {
            if unit.kind == Kind::Seed {
                counters.seed(unit.layer, unit.rows.len())?;
            } else {
                counters.decoder(
                    unit.layer,
                    unit.rows.len(),
                    model.schedule.produces_index(unit.layer),
                )?;
            }
        }
        Ok(())
    }
}
