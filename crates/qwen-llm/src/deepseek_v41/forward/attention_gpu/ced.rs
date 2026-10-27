//! Narrow CED attention primitives and completion-gated range state.
use super::super::ced_plan::CedPlan;
use super::*;
use std::ops::Range;
use std::sync::{Arc, Mutex};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum GlobalKvMode {
    Ordinary,
    ReusePublished,
}

/// Publication has no query, local-KV, selection, or output scratch.
pub(in super::super) struct PublicationRows {
    value: MetalTensor,
    rows: usize,
}

impl PublicationRows {
    pub(in super::super) fn shapes(c: &Config, rows: usize) -> Result<Vec<usize>> {
        valid_chunk(rows)?;
        Ok(vec![checked_product(&[c.head_dim, rows])?])
    }

    pub(in super::super) fn new(ctx: &MetalContext, c: &Config, rows: usize) -> Result<Self> {
        let sizes = Self::shapes(c, rows)?;
        Ok(Self {
            value: tensor(ctx, sizes[0], GgmlType::F32)?,
            rows,
        })
    }
}

pub(in super::super) struct SeedRows {
    value: MetalTensor,
    rows: usize,
}

impl SeedRows {
    pub(in super::super) fn checked_prefix(&self, c: &Config, rows: usize) -> Result<Self> {
        Ok(Self {
            value: prefill::checked_scratch_view(
                &self.value,
                &[c.head_dim * self.rows],
                &[c.head_dim * rows],
                GgmlType::F32,
            )?,
            rows,
        })
    }
    #[cfg(test)]
    pub(in super::super) fn value(&self) -> &MetalTensor {
        &self.value
    }

    pub(in super::super) fn shapes(c: &Config, rows: usize) -> Result<Vec<usize>> {
        PublicationRows::shapes(c, rows)
    }

    pub(in super::super) fn new(ctx: &MetalContext, c: &Config, rows: usize) -> Result<Self> {
        let sizes = Self::shapes(c, rows)?;
        Ok(Self {
            value: tensor(ctx, sizes[0], GgmlType::F32)?,
            rows,
        })
    }
}

fn valid_chunk(rows: usize) -> Result<()> {
    op::require(
        (1..=prefill::MAX_CHUNK_ROWS).contains(&rows),
        "CED chunk rows",
    )
}

fn checked_range(retained: &Range<usize>, start: usize, rows: usize) -> Result<Range<usize>> {
    valid_chunk(rows)?;
    let end = start.checked_add(rows).ok_or("CED row range overflow")?;
    op::require(
        retained.start <= start && end <= retained.end,
        "CED absolute row view",
    )?;
    Ok(start..end)
}

#[derive(Debug)]
struct Coverage {
    origin: usize,
    selected: Vec<Option<usize>>,
    candidates: Vec<Option<usize>>,
}

impl Coverage {
    fn validate(&self, range: &Range<usize>, schedule: &Schedule, layer: usize) -> Result<()> {
        op::require(layer < schedule.layers, "CED control layer")?;
        let candidate = schedule
            .candidate_source_layer
            .ok_or("CED candidate owner")?;
        let selected = schedule
            .index_source_layers
            .iter()
            .copied()
            .rev()
            .find(|&id| id <= layer);
        op::require(
            range.start >= self.origin
                && range.end >= range.start
                && range.end - self.origin <= self.selected.len(),
            "CED control coverage range",
        )?;
        for p in range.clone() {
            let row = p - self.origin;
            if layer > candidate {
                op::require(
                    self.candidates[row] == Some(candidate),
                    "CED missing owner candidate controls",
                )?;
            }
            if !schedule.produces_index(layer) {
                op::require(
                    self.selected[row] == selected,
                    "CED missing selected-ID producer",
                )?;
            }
        }
        Ok(())
    }

    fn complete(&mut self, range: Range<usize>, schedule: &Schedule, layer: usize) {
        for p in range {
            let row = p - self.origin;
            if schedule.produces_index(layer) {
                self.selected[row] = Some(layer);
            }
            if schedule.candidate_source_layer == Some(layer) {
                self.candidates[row] = Some(layer);
            }
        }
    }
}

/// Both discrete controls survive layer-major passes. Views share storage and
/// completed producer coverage; creating a tile never allocates GPU controls.
pub(in super::super) struct QueryControls {
    selected: MetalTensor,
    candidates: MetalTensor,
    range: Range<usize>,
    selected_width: usize,
    candidate_width: usize,
    capacity: usize,
    coverage: Arc<Mutex<Coverage>>,
}

#[derive(Clone)]
pub(in super::super) struct QueryControlView {
    pub(super) selected: MetalTensor,
    pub(super) candidates: MetalTensor,
    range: Range<usize>,
    selected_width: usize,
    candidate_width: usize,
    capacity: usize,
    coverage: Arc<Mutex<Coverage>>,
}

impl QueryControls {
    pub(in super::super) fn cpu_bytes(range: Range<usize>) -> Result<u64> {
        op::require(range.start < range.end, "CED CPU control coverage range")?;
        range
            .len()
            .checked_mul(std::mem::size_of::<Option<usize>>())
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(1024))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| "CED CPU control coverage overflow".into())
    }

    pub(in super::super) fn shapes(
        c: &Config,
        capacity: usize,
        range: Range<usize>,
    ) -> Result<Vec<usize>> {
        op::require(
            c.index_topk > 0
                && c.candidate_block > 0
                && range.start < range.end
                && range.end <= capacity,
            "CED retained controls geometry",
        )?;
        Ok(vec![
            checked_product(&[range.len(), c.index_topk])?,
            checked_product(&[range.len(), capacity.div_ceil(c.candidate_block)])?,
        ])
    }

    pub(in super::super) fn logical_bytes(
        c: &Config,
        capacity: usize,
        range: Range<usize>,
    ) -> Result<u64> {
        Self::shapes(c, capacity, range)?
            .into_iter()
            .try_fold(0u64, |sum, n| {
                sum.checked_add(n as u64 * 4)
                    .ok_or_else(|| "CED control byte overflow".into())
            })
    }

    pub(in super::super) fn new(
        ctx: &MetalContext,
        c: &Config,
        capacity: usize,
        range: Range<usize>,
    ) -> Result<Self> {
        let sizes = Self::shapes(c, capacity, range.clone())?;
        Ok(Self {
            selected: tensor(ctx, sizes[0], GgmlType::I32)?,
            candidates: tensor(ctx, sizes[1], GgmlType::I32)?,
            selected_width: c.index_topk,
            candidate_width: capacity.div_ceil(c.candidate_block),
            capacity,
            coverage: Arc::new(Mutex::new(Coverage {
                origin: range.start,
                selected: vec![None; range.len()],
                candidates: vec![None; range.len()],
            })),
            range,
        })
    }

    pub(in super::super) fn checked_tile_view(
        &self,
        start: usize,
        rows: usize,
    ) -> Result<QueryControlView> {
        let range = checked_range(&self.range, start, rows)?;
        let offset = start - self.range.start;
        Ok(QueryControlView {
            selected: self.selected.view_subrange(
                (offset * self.selected_width) as u64,
                vec![self.selected_width as u64, rows as u64],
            ),
            candidates: self.candidates.view_subrange(
                (offset * self.candidate_width) as u64,
                vec![self.candidate_width as u64, rows as u64],
            ),
            range,
            selected_width: self.selected_width,
            candidate_width: self.candidate_width,
            capacity: self.capacity,
            coverage: self.coverage.clone(),
        })
    }
}

impl QueryControlView {
    #[cfg(test)]
    pub(in super::super) fn diagnostic_copy(&self, ctx: &MetalContext) -> Result<Self> {
        let mut view = self.clone();
        view.selected = tensor(ctx, self.selected.n_elements() as usize, GgmlType::I32)?;
        view.candidates = tensor(ctx, self.candidates.n_elements() as usize, GgmlType::I32)?;
        Ok(view)
    }

    #[cfg(test)]
    pub(in super::super) fn encode_diagnostic_copy(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        source: &Self,
    ) -> Result<()> {
        copy(
            ctx,
            enc,
            &source.selected,
            &self.selected,
            self.selected.n_elements() as usize,
        )?;
        copy(
            ctx,
            enc,
            &source.candidates,
            &self.candidates,
            self.candidates.n_elements() as usize,
        )
    }
    pub(in super::super) fn rows(&self) -> usize {
        self.range.len()
    }
    #[cfg(test)]
    pub(in super::super) fn absolute_rows(&self) -> Range<usize> {
        self.range.clone()
    }

    #[cfg(test)]
    pub(in super::super) fn selected(&self) -> &MetalTensor {
        &self.selected
    }

    #[cfg(test)]
    pub(in super::super) fn candidates(&self) -> &MetalTensor {
        &self.candidates
    }

    pub(super) fn validate_geometry(&self, c: &Config, capacity: usize) -> Result<()> {
        valid_chunk(self.rows())?;
        op::require(c.candidate_block > 0, "CED candidate block geometry")?;
        op::require(
            self.capacity == capacity
                && self.selected_width == c.index_topk
                && self.candidate_width == capacity.div_ceil(c.candidate_block),
            "CED control view geometry",
        )?;
        for (tensor, width) in [
            (&self.selected, self.selected_width),
            (&self.candidates, self.candidate_width),
        ] {
            let n = checked_product(&[width, self.rows()])?;
            op::require(
                tensor.dtype == GgmlType::I32
                    && tensor.n_elements() == n as u64
                    && tensor.offset % 4 == 0
                    && tensor
                        .offset
                        .checked_add(n as u64 * 4)
                        .is_some_and(|end| end <= tensor.buffer.length() as u64),
                "CED control view storage",
            )?;
        }
        Ok(())
    }

    fn validate_layer(&self, schedule: &Schedule, layer: usize) -> Result<()> {
        self.coverage
            .lock()
            .map_err(|_| "CED control coverage poisoned")?
            .validate(&self.range, schedule, layer)
    }

    fn complete(&self, schedule: &Schedule, layer: usize) -> Result<()> {
        self.coverage
            .lock()
            .map_err(|_| "CED control coverage poisoned")?
            .complete(self.range.clone(), schedule, layer);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct HistoryRange {
    pub(super) layer: usize,
    pub(super) start: usize,
    pub(super) floor: usize,
    pub(super) extent: usize,
}

impl HistoryRange {
    pub(super) fn validate(
        &self,
        layer: usize,
        start: usize,
        rows: usize,
        window: usize,
    ) -> Result<()> {
        let end = start
            .checked_add(rows)
            .ok_or("CED history query overflow")?;
        let history_end = self
            .floor
            .checked_add(self.extent)
            .ok_or("CED history extent overflow")?;
        let first_end = start.checked_add(1).ok_or("CED history query overflow")?;
        op::require(
            rows > 0
                && window > 0
                && self.layer == layer
                && self.start == start
                && self.floor <= first_end.saturating_sub(window)
                && end <= history_end,
            "attention history does not cover full query windows",
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Encoder,
    Seed(usize),
    Decoder(usize),
}

struct Pending {
    phase: Phase,
    range: Range<usize>,
    // Encoded earlier in this serial command, never completed coverage.
    encoded_seed_prefix: Option<Range<usize>>,
    batch: Option<BatchCursor>,
    end_layer: usize,
    encoded: bool,
    finished: std::cell::Cell<bool>,
    controls: Option<QueryControlView>,
}

pub(super) struct Cursor {
    plan: CedPlan,
    encoder_completed: usize,
    publication_completed: usize,
    decoder_next: usize,
    seed_completed: Vec<usize>,
    output_completed: Vec<usize>,
    pending: Option<Pending>,
    group: Option<EncodedGroup>,
    generation: u64,
}

struct EncodedGroup {
    generation: u64,
    layers: Range<usize>,
    next: usize,
    seeds: Vec<usize>,
    outputs: Vec<usize>,
    units: Vec<Pending>,
    coverage: Option<Arc<Mutex<Coverage>>>,
}

pub(in super::super) fn ced_group_cpu_bytes(layers: usize, units: usize) -> Result<u64> {
    // Two frontier vectors and one exact-capacity Pending vector. Each retained
    // control tensor uses the same 1 KiB descriptor allowance as other plans.
    layers
        .checked_mul(2 * std::mem::size_of::<usize>())
        .and_then(|n| {
            units
                .checked_mul(std::mem::size_of::<Pending>() + 2 * 1024)
                .and_then(|u| n.checked_add(u))
        })
        .and_then(|n| n.checked_add(std::mem::size_of::<EncodedGroup>()))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| "CED batch CPU ledger overflow".into())
}

impl Cursor {
    fn new(plan: &CedPlan, c: &Config, schedule: &Schedule, capacity: usize) -> Result<Self> {
        let cut = plan.publication_layer;
        let n = plan.prompt_rows;
        op::require(
            c.window > 0
                && (1..=prefill::MAX_CHUNK_ROWS).contains(&plan.full_chunk_rows)
                && n > 0
                && n <= capacity
                && plan.encoder_layers == (0..cut)
                && cut > 0
                && cut < schedule.layers
                && plan.decoder.len() == schedule.layers - cut
                && schedule.candidate_source_layer == Some(cut)
                && schedule.owns_kv(cut)
                && schedule.ratio(cut) == 1
                && !schedule.kv_source_layers.iter().any(|&id| id > cut),
            "CED attention plan schedule",
        )?;
        let mut previous = plan.boundary_rows.clone();
        for (j, layer) in plan.decoder.iter().enumerate() {
            let distance = (plan.decoder.len() - 1 - j)
                .checked_mul(c.window - 1)
                .and_then(|n| n.checked_add(1))
                .ok_or("CED dependency range overflow")?;
            let a = n.saturating_sub(distance);
            let s = a.saturating_sub(c.window - 1);
            op::require(
                layer.layer == cut + j
                    && layer.input_rows == (s..n)
                    && layer.input_rows == previous
                    && layer.seed_rows == (s..a)
                    && layer.output_rows == (a..n)
                    && schedule.owner(layer.layer) == Some(cut)
                    && schedule.ratio(layer.layer) == 1,
                "CED attention dependency range",
            )?;
            previous = layer.output_rows.clone();
        }
        op::require(
            plan.candidate_rows == plan.decoder[0].output_rows,
            "CED retained control range",
        )?;
        Ok(Self {
            plan: plan.clone(),
            encoder_completed: 0,
            publication_completed: 0,
            decoder_next: 0,
            seed_completed: plan.decoder.iter().map(|l| l.seed_rows.start).collect(),
            output_completed: plan.decoder.iter().map(|l| l.output_rows.start).collect(),
            pending: None,
            group: None,
            generation: 0,
        })
    }

    fn idle(&self) -> Result<()> {
        op::require(
            self.pending.is_none(),
            "CED command still pending completion",
        )
    }

    fn begin_encoder(&mut self, start: usize, rows: usize) -> Result<()> {
        self.idle()?;
        op::require(
            rows <= self.plan.full_chunk_rows,
            "CED encoder chunk geometry",
        )?;
        op::require(self.group.is_none(), "CED encoder during decoder batch")?;
        let range = checked_range(&(0..self.plan.prompt_rows), start, rows)?;
        op::require(
            start == self.encoder_completed && start == self.publication_completed,
            "CED encoder completed frontier",
        )?;
        self.pending = Some(Pending {
            phase: Phase::Encoder,
            range,
            encoded_seed_prefix: None,
            batch: Some(BatchCursor {
                start,
                rows,
                layer: 0,
                row: 0,
            }),
            end_layer: self.plan.publication_layer,
            encoded: false,
            finished: std::cell::Cell::new(false),
            controls: None,
        });
        Ok(())
    }

    fn decoder_index(&self, layer: usize) -> Result<usize> {
        let next = self.group.as_ref().map_or(self.decoder_next, |g| g.next);
        op::require(
            self.encoder_completed == self.plan.prompt_rows
                && self.publication_completed == self.plan.prompt_rows
                && next < self.plan.decoder.len()
                && self.plan.decoder[next].layer == layer
                && self
                    .group
                    .as_ref()
                    .is_none_or(|g| g.layers.contains(&layer)),
            "CED decoder layer/completed publication frontier",
        )?;
        Ok(next)
    }

    fn seed_frontier(&self, j: usize) -> usize {
        self.group
            .as_ref()
            .map_or(self.seed_completed[j], |g| g.seeds[j])
    }

    fn output_frontier(&self, j: usize) -> usize {
        self.group
            .as_ref()
            .map_or(self.output_completed[j], |g| g.outputs[j])
    }

    fn begin_seed(&mut self, layer: usize, start: usize, rows: usize) -> Result<()> {
        self.idle()?;
        op::require(
            rows <= prefill::DEFAULT_CHUNK_ROWS,
            "CED seed chunk geometry",
        )?;
        let j = self.decoder_index(layer)?;
        let range = checked_range(&self.plan.decoder[j].seed_rows, start, rows)?;
        op::require(
            start == self.seed_frontier(j)
                && self.output_frontier(j) == self.plan.decoder[j].output_rows.start,
            "CED seed completed frontier",
        )?;
        self.pending = Some(Pending {
            phase: Phase::Seed(layer),
            range,
            encoded_seed_prefix: None,
            batch: None,
            end_layer: layer,
            encoded: false,
            finished: std::cell::Cell::new(false),
            controls: None,
        });
        Ok(())
    }

    fn begin_decoder(&mut self, layer: usize, start: usize, rows: usize) -> Result<()> {
        self.idle()?;
        op::require(
            rows <= prefill::DEFAULT_CHUNK_ROWS,
            "CED decoder chunk geometry",
        )?;
        let j = self.decoder_index(layer)?;
        let range = checked_range(&self.plan.decoder[j].output_rows, start, rows)?;
        op::require(
            self.seed_frontier(j) == self.plan.decoder[j].seed_rows.end
                && start == self.output_frontier(j),
            "CED output completed frontier/seed coverage",
        )?;
        self.pending = Some(Pending {
            phase: Phase::Decoder(layer),
            range,
            encoded_seed_prefix: None,
            batch: Some(BatchCursor {
                start,
                rows,
                layer,
                row: 0,
            }),
            end_layer: layer + 1,
            encoded: false,
            finished: std::cell::Cell::new(false),
            controls: None,
        });
        Ok(())
    }

    fn chain_seed_to_decoder(&mut self, layer: usize, start: usize, rows: usize) -> Result<()> {
        let j = self.decoder_index(layer)?;
        let seed = &self.plan.decoder[j].seed_rows;
        let p = self.matches(Phase::Seed(layer), seed.start, seed.len())?;
        let range = checked_range(&self.plan.decoder[j].output_rows, start, rows)?;
        op::require(
            p.encoded
                && rows <= prefill::DEFAULT_CHUNK_ROWS
                && p.encoded_seed_prefix.is_none()
                && self.plan.decoder[j]
                    .fuse_seed_with_packed_output(self.plan.prompt_rows, self.plan.full_chunk_rows)
                && range.end <= self.plan.scalar_tail().unwrap_or(self.plan.prompt_rows)
                && self.seed_frontier(j) == seed.start
                && start == seed.end
                && start == self.output_frontier(j),
            "CED compound requires fully encoded whole seed and first decoder chunk",
        )?;
        self.pending = Some(Pending {
            phase: Phase::Decoder(layer),
            range,
            encoded_seed_prefix: Some(seed.clone()),
            batch: Some(BatchCursor {
                start,
                rows,
                layer,
                row: 0,
            }),
            end_layer: layer + 1,
            encoded: false,
            finished: std::cell::Cell::new(false),
            controls: None,
        });
        Ok(())
    }

    fn matches(&self, phase: Phase, start: usize, rows: usize) -> Result<&Pending> {
        let p = self.pending.as_ref().ok_or("CED has no pending command")?;
        op::require(
            p.phase == phase && p.range.start == start && p.range.len() == rows,
            "CED pending range/phase",
        )?;
        Ok(p)
    }

    fn attention_done(p: &Pending) -> bool {
        p.batch
            .as_ref()
            .is_some_and(|b| b.layer == p.end_layer && b.row == 0)
    }

    fn completion_range(
        &self,
        phase: Phase,
        start: usize,
        rows: usize,
    ) -> Result<(usize, Option<usize>)> {
        let p = self.matches(phase, start, rows)?;
        op::require(
            p.encoded
                && (matches!(phase, Phase::Seed(_)) || Self::attention_done(p))
                && (!matches!(phase, Phase::Decoder(_)) || p.finished.get()),
            "CED range not fully encoded",
        )?;
        let end = p.range.end;
        let seed_end = if let Some(seed) = &p.encoded_seed_prefix {
            let j = self.decoder_index(match phase {
                Phase::Decoder(id) => id,
                _ => return Err("CED compound phase".into()),
            })?;
            op::require(
                phase == Phase::Decoder(self.plan.decoder[j].layer)
                    && *seed == self.plan.decoder[j].seed_rows
                    && self.seed_frontier(j) == seed.start
                    && p.range.start == seed.end
                    && self.output_frontier(j) == p.range.start,
                "CED compound completion frontier",
            )?;
            Some(seed.end)
        } else {
            None
        };
        Ok((end, seed_end))
    }

    fn complete(&mut self, phase: Phase, start: usize, rows: usize) -> Result<()> {
        op::require(self.group.is_none(), "CED batch requires atomic retirement")?;
        let (end, seed_end) = self.completion_range(phase, start, rows)?;
        match phase {
            Phase::Encoder => {
                self.encoder_completed = end;
                self.publication_completed = end;
            }
            Phase::Seed(_) => {
                self.seed_completed[self.decoder_next] = end;
            }
            Phase::Decoder(_) => {
                if let Some(end) = seed_end {
                    self.seed_completed[self.decoder_next] = end;
                }
                self.output_completed[self.decoder_next] = end;
                if end == self.plan.prompt_rows {
                    self.decoder_next += 1;
                }
            }
        }
        self.pending = None;
        Ok(())
    }

    fn ready(&self, n: usize) -> bool {
        n == self.plan.prompt_rows
            && self.pending.is_none()
            && self.group.is_none()
            && self.encoder_completed == n
            && self.publication_completed == n
            && self.decoder_next == self.plan.decoder.len()
            && self.output_completed.iter().all(|&end| end == n)
    }

    fn begin_group(&mut self, layers: Range<usize>) -> Result<u64> {
        self.idle()?;
        op::require(
            self.group.is_none() && (1..=4).contains(&layers.len()),
            "CED batch bounds",
        )?;
        self.decoder_index(layers.start)?;
        op::require(
            layers.end <= self.plan.publication_layer + self.plan.decoder.len(),
            "CED batch end",
        )?;
        let generation = self
            .generation
            .checked_add(1)
            .ok_or("CED batch generation overflow")?;
        let units = super::super::ced_plan::batches::units(&self.plan, layers.clone())
            .map_err(|e| e.to_string())?;
        self.group = Some(EncodedGroup {
            generation,
            layers,
            next: self.decoder_next,
            seeds: self.seed_completed.clone(),
            outputs: self.output_completed.clone(),
            units: Vec::with_capacity(units.len()),
            coverage: None,
        });
        self.generation = generation;
        Ok(generation)
    }

    fn stage_group_unit(
        &mut self,
        generation: u64,
        layer: usize,
        start: usize,
        rows: usize,
        seed: bool,
    ) -> Result<()> {
        let phase = if seed {
            Phase::Seed(layer)
        } else {
            Phase::Decoder(layer)
        };
        let (end, seed_end) = self.completion_range(phase, start, rows)?;
        let j = self.decoder_index(layer)?;
        let g = self.group.as_mut().ok_or("CED batch missing")?;
        op::require(g.generation == generation, "CED batch generation")?;
        if seed {
            g.seeds[j] = end;
        } else {
            if let Some(end) = seed_end {
                g.seeds[j] = end;
            }
            g.outputs[j] = end;
            if end == self.plan.prompt_rows {
                g.next += 1;
            }
        }
        g.units.push(self.pending.take().unwrap());
        Ok(())
    }

    fn validate_group(&self, generation: u64, layers: &Range<usize>) -> Result<()> {
        self.idle()?;
        let g = self.group.as_ref().ok_or("CED batch missing")?;
        op::require(
            g.generation == generation
                && &g.layers == layers
                && g.next + self.plan.publication_layer == layers.end
                && !g.units.is_empty(),
            "CED batch retirement range/generation/coverage",
        )
    }

    fn publish_group(&mut self) {
        let g = self.group.take().unwrap();
        self.decoder_next = g.next;
        self.seed_completed = g.seeds;
        self.output_completed = g.outputs;
    }
}

impl Workspace {
    #[cfg(test)]
    pub(in super::super) fn ced_public_decoder_progress(
        &self,
    ) -> Option<(usize, Vec<usize>, Vec<usize>)> {
        let c = self.ced.as_ref()?;
        Some((
            c.decoder_next,
            c.seed_completed.clone(),
            c.output_completed.clone(),
        ))
    }

    pub(in super::super) fn abort_ced(&mut self) {
        // The synchronous CED owner drains submitted work before returning or
        // unwinding to the provisional guard. Drop transient coverage/view owners
        // without resetting any mutated causal data or making it reusable.
        self.ced = None;
    }
    #[cfg(test)]
    pub(in super::super) fn encode_ced_test_nonfinite(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        value: &MetalTensor,
    ) -> Result<()> {
        finite(ctx, enc, value, 1, &self.status)
    }
    fn validate_ced_controls(&self, view: &QueryControlView, layer: usize) -> Result<()> {
        let Some(g) = self.ced.as_ref().and_then(|c| c.group.as_ref()) else {
            return view.validate_layer(&self.schedule, layer);
        };
        let coverage = g
            .coverage
            .as_ref()
            .ok_or("CED batch control storage missing")?;
        op::require(
            Arc::ptr_eq(coverage, &view.coverage),
            "CED batch control generation/storage",
        )?;
        let coverage = coverage
            .lock()
            .map_err(|_| "CED control coverage poisoned")?;
        let candidate = self
            .schedule
            .candidate_source_layer
            .ok_or("CED candidate owner")?;
        let selected = self
            .schedule
            .index_source_layers
            .iter()
            .copied()
            .rev()
            .find(|&id| id <= layer);
        op::require(
            view.range.start >= coverage.origin
                && view.range.end - coverage.origin <= coverage.selected.len(),
            "CED batch control origin",
        )?;
        for position in view.range.clone() {
            let row = position - coverage.origin;
            let mut got_selected = coverage.selected[row];
            let mut got_candidate = coverage.candidates[row];
            for p in &g.units {
                if let Phase::Decoder(id) = p.phase
                    && p.range.contains(&position)
                {
                    if self.schedule.produces_index(id) {
                        got_selected = Some(id);
                    }
                    if id == candidate {
                        got_candidate = Some(id);
                    }
                }
            }
            op::require(
                layer <= candidate || got_candidate == Some(candidate),
                "CED missing encoded owner candidate controls",
            )?;
            op::require(
                self.schedule.produces_index(layer) || got_selected == selected,
                "CED missing encoded selected-ID producer",
            )?;
        }
        Ok(())
    }

    pub(in super::super) fn begin_ced_group(
        &mut self,
        layers: Range<usize>,
        controls: &QueryControls,
    ) -> Result<u64> {
        let ced = self.ced.as_mut().ok_or("CED not active")?;
        op::require(
            controls.range == ced.plan.candidate_rows,
            "CED batch retained control origin",
        )?;
        let generation = ced.begin_group(layers)?;
        ced.group.as_mut().unwrap().coverage = Some(controls.coverage.clone());
        Ok(generation)
    }

    pub(in super::super) fn stage_ced_group_unit(
        &mut self,
        generation: u64,
        layer: usize,
        start: usize,
        rows: usize,
        seed: bool,
    ) -> Result<()> {
        self.ced
            .as_mut()
            .ok_or("CED not active")?
            .stage_group_unit(generation, layer, start, rows, seed)
    }

    pub(in super::super) fn retire_ced_group(
        &mut self,
        generation: u64,
        layers: Range<usize>,
    ) -> Result<()> {
        self.checked_ced_status()?;
        let ced = self.ced.as_mut().ok_or("CED not active")?;
        ced.validate_group(generation, &layers)?;
        let g = ced.group.as_ref().unwrap();
        let coverage = g
            .coverage
            .as_ref()
            .ok_or("CED batch controls missing")?
            .clone();
        let mut coverage = coverage
            .lock()
            .map_err(|_| "CED control coverage poisoned")?;
        for p in &g.units {
            if let Phase::Decoder(_) = p.phase {
                let v = p.controls.as_ref().ok_or("CED staged controls missing")?;
                op::require(
                    v.range == p.range && Arc::ptr_eq(&v.coverage, g.coverage.as_ref().unwrap()),
                    "CED staged controls range/storage",
                )?;
            }
        }
        for p in &g.units {
            if let Phase::Decoder(id) = p.phase {
                coverage.complete(p.range.clone(), &self.schedule, id);
            }
        }
        ced.publish_group();
        Ok(())
    }

    pub(in super::super) fn encode_ced_status_copy(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        out: &MetalTensor,
    ) -> Result<()> {
        // Integer words pass through the existing bit-preserving copy operator.
        copy(
            ctx,
            enc,
            &self.status.view_bytes(0, vec![1]),
            &out.view_bytes(0, vec![1]),
            1,
        )
    }

    pub(in super::super) fn begin_ced(&mut self, plan: &CedPlan) -> Result<()> {
        self.validate_snapshot_position(0)?;
        self.ced = Some(Cursor::new(
            plan,
            &self.config,
            &self.schedule,
            self.capacity,
        )?);
        Ok(())
    }

    pub(in super::super) fn begin_ced_encoder(&mut self, start: usize, rows: usize) -> Result<()> {
        self.ced
            .as_mut()
            .ok_or("CED not active")?
            .begin_encoder(start, rows)
    }

    pub(in super::super) fn begin_ced_seed(
        &mut self,
        layer: usize,
        start: usize,
        rows: usize,
    ) -> Result<()> {
        self.ced
            .as_mut()
            .ok_or("CED not active")?
            .begin_seed(layer, start, rows)
    }

    pub(in super::super) fn begin_ced_decoder_chunk(
        &mut self,
        layer: usize,
        start: usize,
        rows: usize,
    ) -> Result<()> {
        self.ced
            .as_mut()
            .ok_or("CED not active")?
            .begin_decoder(layer, start, rows)
    }

    pub(in super::super) fn chain_ced_seed_to_decoder(
        &mut self,
        layer: usize,
        start: usize,
        rows: usize,
    ) -> Result<()> {
        self.ced
            .as_mut()
            .ok_or("CED not active")?
            .chain_seed_to_decoder(layer, start, rows)
    }

    /// One-row intents let the existing gpu_step::layer use scalar operators.
    pub(in super::super) fn begin_ced_decoder_scalar(
        &mut self,
        layer: usize,
        position: usize,
        controls: &QueryControlView,
    ) -> Result<()> {
        controls.validate_geometry(&self.config, self.capacity)?;
        op::require(
            controls.range == (position..position + 1),
            "CED scalar control origin",
        )?;
        self.validate_ced_controls(controls, layer)?;
        self.begin_ced_decoder_chunk(layer, position, 1)?;
        self.ced
            .as_mut()
            .unwrap()
            .pending
            .as_mut()
            .unwrap()
            .controls = Some(controls.clone());
        Ok(())
    }

    pub(super) fn active_batch(&self) -> Option<&BatchCursor> {
        match &self.ced {
            Some(ced) => ced.pending.as_ref().and_then(|p| p.batch.as_ref()),
            None => self.batch.as_ref(),
        }
    }

    pub(super) fn ced_decoder_prompt_rows(
        &self,
        id: usize,
        start: usize,
        rows: usize,
    ) -> Result<Option<(usize, usize)>> {
        let Some(ced) = &self.ced else {
            return Ok(None);
        };
        let p = ced.pending.as_ref().ok_or("CED has no attention intent")?;
        if matches!(p.phase, Phase::Decoder(_)) {
            ced.matches(Phase::Decoder(id), start, rows)?;
            return Ok(Some((ced.plan.prompt_rows, ced.plan.full_chunk_rows)));
        }
        Ok(None)
    }

    pub(super) fn global_mode(
        &self,
        id: usize,
        position: usize,
        _rows: usize,
    ) -> Result<GlobalKvMode> {
        let Some(ced) = &self.ced else {
            return Ok(GlobalKvMode::Ordinary);
        };
        let p = ced.pending.as_ref().ok_or("CED has no attention intent")?;
        op::require(
            p.range.contains(&position),
            "CED query outside pending range",
        )?;
        match p.phase {
            Phase::Encoder if id < ced.plan.publication_layer => Ok(GlobalKvMode::Ordinary),
            Phase::Decoder(layer) if id == layer => {
                self.validate_global_mode(id, position, GlobalKvMode::ReusePublished)?;
                Ok(GlobalKvMode::ReusePublished)
            }
            _ => Err("CED attention phase/layer".into()),
        }
    }

    pub(super) fn validate_global_mode(
        &self,
        id: usize,
        position: usize,
        mode: GlobalKvMode,
    ) -> Result<()> {
        if mode == GlobalKvMode::ReusePublished {
            let ced = self.ced.as_ref().ok_or("reuse requires active CED")?;
            let p = ced
                .pending
                .as_ref()
                .ok_or("reuse requires decoder intent")?;
            op::require(
                p.phase == Phase::Decoder(id)
                    && p.range.contains(&position)
                    && self.schedule.owner(id) == Some(ced.plan.publication_layer)
                    && ced.publication_completed > position
                    && ced.publication_completed == ced.plan.prompt_rows,
                "reuse owner/completed publication frontier",
            )?;
        } else if let Some(ced) = &self.ced {
            op::require(
                ced.pending.as_ref().is_some_and(|p| {
                    p.phase == Phase::Encoder
                        && id < ced.plan.publication_layer
                        && p.range.contains(&position)
                }),
                "ordinary attention during CED replay",
            )?;
        }
        Ok(())
    }

    pub(super) fn advance_ced_row(&mut self) -> Result<()> {
        let p = self
            .ced
            .as_mut()
            .and_then(|c| c.pending.as_mut())
            .ok_or("CED attention intent missing")?;
        let b = p.batch.as_mut().ok_or("CED intent has no attention")?;
        b.row += 1;
        if b.row == b.rows {
            b.row = 0;
            b.layer += 1;
        }
        op::require(b.layer <= p.end_layer, "CED encoded past layer range")?;
        if matches!(p.phase, Phase::Decoder(_)) && b.layer == p.end_layer {
            p.encoded = true;
        }
        Ok(())
    }

    pub(super) fn bind_ced_rows(&mut self, id: usize, start: usize, rows: &Rows) -> Result<()> {
        let mode = self.global_mode(id, start, rows.rows)?;
        if mode == GlobalKvMode::ReusePublished {
            let controls = rows
                .controls
                .as_ref()
                .ok_or("CED rows require retained controls")?;
            controls.validate_geometry(&self.config, self.capacity)?;
            op::require(
                controls.range == (start..start + rows.rows),
                "CED row controls absolute origin",
            )?;
            self.validate_ced_controls(controls, id)?;
            let p = self.ced.as_mut().unwrap().pending.as_mut().unwrap();
            op::require(p.controls.is_none(), "CED decoder rows already prepared")?;
            p.controls = Some(controls.clone());
        }
        Ok(())
    }

    pub(super) fn finish_ced_rows(&self, rows: &Rows) -> Result<()> {
        if let Some(p) = self.ced.as_ref().and_then(|c| c.pending.as_ref())
            && let Phase::Decoder(_) = p.phase
        {
            op::require(
                p.encoded
                    && Cursor::attention_done(p)
                    && rows.controls.as_ref().is_some_and(|v| v.range == p.range),
                "CED finish rows range/order",
            )?;
            p.finished.set(true);
        }
        Ok(())
    }

    pub(super) fn row_history(&self, id: usize, start: usize, rows: usize) -> Result<HistoryRange> {
        let old = if let Some(ced) = &self.ced {
            if id >= ced.plan.publication_layer {
                let j = ced.decoder_index(id)?;
                let floor = start.saturating_sub(self.config.window - 1);
                let encoded_seed = ced.pending.as_ref().is_some_and(|p| {
                    p.phase == Phase::Decoder(id)
                        && p.range == (start..start + rows)
                        && p.encoded_seed_prefix.as_ref() == Some(&ced.plan.decoder[j].seed_rows)
                        && ced.seed_frontier(j) == ced.plan.decoder[j].seed_rows.start
                        && start == ced.plan.decoder[j].seed_rows.end
                });
                op::require(
                    floor >= ced.plan.decoder[j].seed_rows.start
                        && (ced.seed_frontier(j) == ced.plan.decoder[j].seed_rows.end
                            || encoded_seed)
                        && start == ced.output_frontier(j),
                    "CED local history completed coverage",
                )?;
                start - floor
            } else {
                start.min(self.config.window)
            }
        } else {
            start.min(self.config.window)
        };
        let history = HistoryRange {
            layer: id,
            start,
            floor: start - old,
            extent: old + rows,
        };
        history.validate(id, start, rows, self.config.window)?;
        Ok(history)
    }

    /// Only a matching one-row CED attention intent can enter through encode().
    pub(in super::super) fn encode_ced_scalar(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
    ) -> Result<&MetalTensor> {
        let p = self
            .ced
            .as_ref()
            .and_then(|c| c.pending.as_ref())
            .ok_or("CED scalar intent missing")?;
        op::require(
            p.range == (position..position + 1)
                && p.batch
                    .as_ref()
                    .is_some_and(|b| b.rows == 1 && b.layer == id && b.row == 0),
            "CED scalar intent order",
        )?;
        let mode = self.global_mode(id, position, 1)?;
        let controls = p.controls.clone();
        if mode == GlobalKvMode::ReusePublished {
            let controls = controls
                .as_ref()
                .ok_or("CED scalar decoder controls missing")?;
            self.validate_ced_controls(controls, id)?;
            self.row_history(id, position, 1)?;
            let selected = std::mem::replace(
                &mut self.selected,
                prefill::row(&controls.selected, self.config.index_topk, 0),
            );
            let candidates = std::mem::replace(
                &mut self.candidates,
                prefill::row(
                    &controls.candidates,
                    self.capacity.div_ceil(self.config.candidate_block),
                    0,
                ),
            );
            let result = self
                .encode_inner(ctx, enc, id, position, w, input, None, true, mode)
                .map(|_| ());
            self.selected = selected;
            self.candidates = candidates;
            result?;
            self.ced
                .as_ref()
                .unwrap()
                .pending
                .as_ref()
                .unwrap()
                .finished
                .set(true);
        } else {
            self.encode_inner(ctx, enc, id, position, w, input, None, true, mode)?;
        }
        Ok(&self.output)
    }

    fn validate_publication_intent(&self, owner: usize, start: usize, rows: usize) -> Result<()> {
        let ced = self
            .ced
            .as_ref()
            .ok_or("CED publication without transaction")?;
        let p = ced.matches(Phase::Encoder, start, rows)?;
        op::require(
            owner == ced.plan.publication_layer
                && self.schedule.owns_kv(owner)
                && self.schedule.ratio(owner) == 1
                && Cursor::attention_done(p)
                && !p.encoded,
            "CED publication requires encoded encoder boundary",
        )
    }

    pub(in super::super) fn publish_ced_rows(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        owner: usize,
        start: usize,
        w: &Weights<'_>,
        normalized_u: &MetalTensor,
        scratch: &PublicationRows,
    ) -> Result<()> {
        op::require(
            !enc.is_concurrent(),
            "CED publication requires serial encoder",
        )?;
        self.validate_publication_intent(owner, start, scratch.rows)?;
        validate_weights(&self.config, owner, w, &self.schedule)?;
        valid_view(
            normalized_u,
            checked_product(&[self.config.hidden, scratch.rows])?,
        )?;
        valid_view(
            &scratch.value,
            checked_product(&[self.config.head_dim, scratch.rows])?,
        )?;
        w.compressor
            .as_ref()
            .ok_or("CED missing compressor")?
            .kv
            .encode_rows(enc, normalized_u, &scratch.value, scratch.rows)
            .map_err(|e| e.to_string())?;
        for row in 0..scratch.rows {
            self.publish_owner_transition(
                ctx,
                enc,
                owner,
                start + row,
                w,
                &prefill::row(normalized_u, self.config.hidden, row),
                &prefill::row(&scratch.value, self.config.head_dim, row),
                &self.gate,
                true,
            )?;
        }
        self.validate_published_rows(ctx, enc, owner, start, scratch.rows)?;
        self.ced.as_mut().unwrap().pending.as_mut().unwrap().encoded = true;
        Ok(())
    }

    pub(in super::super) fn publish_ced_scalar(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        owner: usize,
        position: usize,
        w: &Weights<'_>,
        normalized_u: &MetalTensor,
    ) -> Result<()> {
        op::require(
            !enc.is_concurrent(),
            "CED publication requires serial encoder",
        )?;
        self.validate_publication_intent(owner, position, 1)?;
        validate_weights(&self.config, owner, w, &self.schedule)?;
        valid_view(normalized_u, self.config.hidden)?;
        self.publish_owner_transition(
            ctx,
            enc,
            owner,
            position,
            w,
            normalized_u,
            &self.latent,
            &self.gate,
            false,
        )?;
        self.validate_published_rows(ctx, enc, owner, position, 1)?;
        self.ced.as_mut().unwrap().pending.as_mut().unwrap().encoded = true;
        Ok(())
    }

    fn validate_seed_intent(
        &self,
        layer: usize,
        start: usize,
        rows: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
    ) -> Result<()> {
        let p = self
            .ced
            .as_ref()
            .ok_or("CED seed without transaction")?
            .matches(Phase::Seed(layer), start, rows)?;
        op::require(!p.encoded, "CED seed already encoded")?;
        validate_weights(&self.config, layer, w, &self.schedule)?;
        valid_view(input, checked_product(&[self.config.hidden, rows])?)
    }

    fn publish_seed_value(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        layer: usize,
        position: usize,
        value: &MetalTensor,
    ) -> Result<()> {
        let c = &self.config;
        rope(
            ctx,
            enc,
            value,
            &self.constants[layer].frequencies,
            c.head_dim,
            1,
            c.rope_dim / 2,
            position,
            false,
        )?;
        finite(ctx, enc, value, c.head_dim, &self.status)?;
        let raw = self.raw.view_subrange(
            ((layer * c.window + position % c.window) * c.head_dim) as u64,
            vec![c.head_dim as u64],
        );
        copy(ctx, enc, value, &raw, c.head_dim)
    }

    pub(in super::super) fn seed_ced_rows(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        layer: usize,
        start: usize,
        w: &Weights<'_>,
        normalized_u: &MetalTensor,
        scratch: &SeedRows,
    ) -> Result<()> {
        op::require(!enc.is_concurrent(), "CED seed requires serial encoder")?;
        self.validate_seed_intent(layer, start, scratch.rows, w, normalized_u)?;
        valid_view(
            &scratch.value,
            checked_product(&[self.config.head_dim, scratch.rows])?,
        )?;
        w.kv.encode_rows_with_arithmetic(
            enc,
            normalized_u,
            &scratch.value,
            scratch.rows,
            RowsArithmetic::Packed,
        )
        .map_err(|e| e.to_string())?;
        metal::encode_rms_norm_mul_rows_f32(
            ctx,
            enc,
            &scratch.value,
            &self.constants[layer].kv_norm,
            &scratch.value,
            scratch.rows,
            self.config.head_dim,
            w.eps,
        )
        .map_err(|e| e.to_string())?;
        for row in 0..scratch.rows {
            self.publish_seed_value(
                ctx,
                enc,
                layer,
                start + row,
                &prefill::row(&scratch.value, self.config.head_dim, row),
            )?;
        }
        self.ced.as_mut().unwrap().pending.as_mut().unwrap().encoded = true;
        Ok(())
    }

    pub(in super::super) fn seed_ced_scalar(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        layer: usize,
        position: usize,
        w: &Weights<'_>,
        normalized_u: &MetalTensor,
    ) -> Result<()> {
        op::require(!enc.is_concurrent(), "CED seed requires serial encoder")?;
        self.validate_seed_intent(layer, position, 1, w, normalized_u)?;
        linear(ctx, enc, &w.kv, normalized_u, &self.latent)?;
        norm(
            ctx,
            enc,
            &self.latent,
            &self.constants[layer].kv_norm,
            self.config.head_dim,
            w.eps,
        )?;
        self.publish_seed_value(ctx, enc, layer, position, &self.latent)?;
        self.ced.as_mut().unwrap().pending.as_mut().unwrap().encoded = true;
        Ok(())
    }

    fn checked_ced_status(&self) -> Result<()> {
        // Caller must first wait for successful command completion and check FFN
        // status. Encoding intent alone never changes a completed frontier.
        let status = unsafe {
            *self
                .status
                .buffer
                .contents()
                .as_ptr()
                .cast::<i32>()
                .add((self.status.offset / 4) as usize)
        };
        op::require(status == 0, &format!("CED attention status {status}"))
    }

    pub(in super::super) fn complete_ced_encoder(
        &mut self,
        start: usize,
        rows: usize,
    ) -> Result<()> {
        self.checked_ced_status()?;
        self.ced
            .as_mut()
            .ok_or("CED not active")?
            .complete(Phase::Encoder, start, rows)
    }

    pub(in super::super) fn complete_ced_seed(
        &mut self,
        layer: usize,
        start: usize,
        rows: usize,
    ) -> Result<()> {
        self.checked_ced_status()?;
        self.ced
            .as_mut()
            .ok_or("CED not active")?
            .complete(Phase::Seed(layer), start, rows)
    }

    pub(in super::super) fn complete_ced_decoder_chunk(
        &mut self,
        layer: usize,
        start: usize,
        rows: usize,
    ) -> Result<()> {
        self.checked_ced_status()?;
        let ced = self.ced.as_mut().ok_or("CED not active")?;
        op::require(ced.group.is_none(), "CED batch requires atomic retirement")?;
        ced.completion_range(Phase::Decoder(layer), start, rows)?;
        let p = ced.matches(Phase::Decoder(layer), start, rows)?;
        op::require(
            p.encoded && p.finished.get() && Cursor::attention_done(p),
            "CED decoder not fully encoded",
        )?;
        let controls = p
            .controls
            .as_ref()
            .ok_or("CED completed controls missing")?;
        controls.complete(&self.schedule, layer)?;
        ced.complete(Phase::Decoder(layer), start, rows)
    }

    pub(in super::super) fn commit_ced(&mut self, n: usize) -> Result<()> {
        self.checked_ced_status()?;
        op::require(
            self.ced.as_ref().is_some_and(|ced| ced.ready(n)),
            "CED commit requires every completed range",
        )?;
        self.next_layer = 0;
        self.next_position = n;
        self.ced = None;
        Ok(())
    }

    #[cfg(test)]
    pub(in super::super) fn ced_diagnostic_regions(
        &self,
        n: usize,
    ) -> Result<Vec<(&MetalTensor, usize)>> {
        let ced = self
            .ced
            .as_ref()
            .ok_or("CED diagnostic transaction missing")?;
        op::require(
            ced.encoder_completed == n
                && ced.publication_completed == n
                && ced.pending.is_none()
                && ced.group.is_none(),
            "CED diagnostic requires completed publication and no pending command",
        )?;
        self.checked_ced_status()?;
        self.causal_region_views(n)
    }
}

impl Workspace {
    pub(super) fn publish_owner_transition(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
        value: &MetalTensor,
        gate: &MetalTensor,
        prepared: bool,
    ) -> Result<()> {
        let c = &self.config;
        let constants = &self.constants[id];
        let cache = &self.owners[self
            .schedule
            .owner_slot(id)
            .ok_or("missing publication owner")?];
        let ratio = self.schedule.ratio(id);
        let length = (position + 1) / ratio;
        let compressor = w.compressor.as_ref().ok_or("missing compressor")?;
        if !prepared {
            linear(ctx, enc, &compressor.kv, input, value)?;
        }
        if ratio == 2 {
            if !prepared {
                linear(
                    ctx,
                    enc,
                    compressor.gate.as_ref().ok_or("missing gate")?,
                    input,
                    gate,
                )?;
            }
            if position.is_multiple_of(2) {
                copy(ctx, enc, value, &cache.pending_value, c.head_dim)?;
                copy(ctx, enc, gate, &cache.pending_gate, c.head_dim)?;
            } else {
                let pso = pipeline(ctx, enc, "pool")?;
                enc.set_bytes(0, &(c.head_dim as u32));
                enc.set_tensor(1, &cache.pending_value);
                enc.set_tensor(2, &cache.pending_gate);
                enc.set_tensor(3, value);
                enc.set_tensor(4, gate);
                dispatch(enc, &pso, c.head_dim);
            }
        }
        if (position + 1).is_multiple_of(ratio) {
            norm(
                ctx,
                enc,
                value,
                constants.compressor_norm.as_ref().ok_or("missing norm")?,
                c.head_dim,
                w.eps,
            )?;
            let index = w.index.as_ref().ok_or("missing index")?;
            linear(
                ctx,
                enc,
                index.key.as_ref().ok_or("missing key")?,
                value,
                &self.key,
            )?;
            let key_norm = constants.key_norm.as_ref().ok_or("missing key norm")?;
            let group_position = position + 1 - ratio;
            let pairs = c.rope_dim / 2;
            let cache_key = cache.keys.view_subrange(
                ((length - 1) * c.index_dim) as u64,
                vec![c.index_dim as u64],
            );
            let cache_latent = cache
                .compressed
                .view_subrange(((length - 1) * c.head_dim) as u64, vec![c.head_dim as u64]);
            if c.index_dim <= 128 && pairs <= 128 && pairs * 2 <= c.index_dim {
                compress_publish(
                    ctx,
                    enc,
                    &self.key,
                    key_norm,
                    value,
                    &constants.frequencies,
                    &cache_key,
                    &cache_latent,
                    c.index_dim,
                    c.head_dim,
                    pairs,
                    group_position,
                    w.eps,
                )?;
            } else {
                norm(ctx, enc, &self.key, key_norm, c.index_dim, w.eps)?;
                rope(
                    ctx,
                    enc,
                    &self.key,
                    &constants.frequencies,
                    c.index_dim,
                    1,
                    pairs,
                    group_position,
                    false,
                )?;
                rope(
                    ctx,
                    enc,
                    value,
                    &constants.frequencies,
                    c.head_dim,
                    1,
                    pairs,
                    group_position,
                    false,
                )?;
                copy(ctx, enc, &self.key, &cache_key, c.index_dim)?;
                copy(ctx, enc, value, &cache_latent, c.head_dim)?;
            }
        }
        Ok(())
    }

    /// The batched publication form needs the 512-wide latent, the 128-wide
    /// index key, a BF16 key projection, and the row-packed compressor/gate
    /// inputs the prepare pass already produced.
    fn compress_rows_supported(&self, id: usize, w: &Weights<'_>) -> bool {
        let c = &self.config;
        let ratio = self.schedule.ratio(id);
        let pairs = c.rope_dim / 2;
        let Some(compressor) = w.compressor.as_ref() else {
            return false;
        };
        let Some(index) = w.index.as_ref() else {
            return false;
        };
        let Some(key) = index.key.as_ref() else {
            return false;
        };
        c.head_dim == 512
            && c.index_dim == 128
            && (ratio == 1 || ratio == 2)
            && pairs > 0
            && 2 * pairs <= c.index_dim
            && compressor.ratio == ratio
            && key.tensor.dtype == GgmlType::BF16
            && key.input == c.head_dim
            && key.output == c.index_dim
            && self.constants[id].compressor_norm.is_some()
            && self.constants[id].key_norm.is_some()
    }

    /// Publish every complete ratio-group whose rows all lie in this batch
    /// with one dispatch. Returns `(head, body, tail)` row counts; the caller
    /// keeps the per-row transition for the head/tail so the pending pair
    /// state and every absolute group boundary stay identical.
    pub(super) fn publish_owner_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        start: usize,
        rows: usize,
        w: &Weights<'_>,
        value: &MetalTensor,
        gate: &MetalTensor,
    ) -> Result<(usize, usize, usize)> {
        let c = &self.config;
        let ratio = self.schedule.ratio(id);
        if !self.compress_rows_supported(id, w) {
            return Ok((0, 0, 0));
        }
        let body_start = start + (ratio - start % ratio) % ratio;
        let body_end = (start + rows) / ratio * ratio;
        if body_end <= body_start {
            return Ok((0, 0, 0));
        }
        let head = body_start - start;
        let body = body_end - body_start;
        let tail = rows - head - body;
        let constants = &self.constants[id];
        let comp_norm = constants
            .compressor_norm
            .as_ref()
            .ok_or("missing compressor norm")?;
        let key_norm = constants.key_norm.as_ref().ok_or("missing key norm")?;
        let index = w.index.as_ref().ok_or("missing index")?;
        let key = index.key.as_ref().ok_or("missing key")?;
        let cache = &self.owners[self
            .schedule
            .owner_slot(id)
            .ok_or("missing publication owner")?];
        compress_publish_rows(
            ctx,
            enc,
            value,
            gate,
            comp_norm,
            &key.tensor,
            key_norm,
            &constants.frequencies,
            &cache.keys,
            &cache.compressed,
            head,
            body_start / ratio,
            body / ratio,
            ratio,
            c.rope_dim / 2,
            w.eps,
        )?;
        Ok((head, body, tail))
    }

    fn validate_published_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        owner: usize,
        start: usize,
        rows: usize,
    ) -> Result<()> {
        let slot = self
            .schedule
            .owner_slot(owner)
            .ok_or("CED publication owner")?;
        let cache = &self.owners[slot];
        for (tensor, width) in [
            (&cache.compressed, self.config.head_dim),
            (&cache.keys, self.config.index_dim),
        ] {
            let n = checked_product(&[rows, width])?;
            let view =
                tensor.view_subrange((start * width) as u64, vec![width as u64, rows as u64]);
            finite(ctx, enc, &view, n, &self.status)?;
        }
        Ok(())
    }
}

fn finite(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    n: usize,
    status: &MetalTensor,
) -> Result<()> {
    let pso = pipeline(ctx, enc, "finite")?;
    enc.set_bytes(0, &(n as u32));
    enc.set_tensor(1, input);
    enc.set_tensor(2, status);
    dispatch(enc, &pso, n);
    Ok(())
}
