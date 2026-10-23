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
        (1..=prefill::DEFAULT_CHUNK_ROWS).contains(&rows),
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
}

impl Cursor {
    fn new(plan: &CedPlan, c: &Config, schedule: &Schedule, capacity: usize) -> Result<Self> {
        let cut = plan.publication_layer;
        let n = plan.prompt_rows;
        op::require(
            c.window > 0
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
        let range = checked_range(&(0..self.plan.prompt_rows), start, rows)?;
        op::require(
            start == self.encoder_completed && start == self.publication_completed,
            "CED encoder completed frontier",
        )?;
        self.pending = Some(Pending {
            phase: Phase::Encoder,
            range,
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
        op::require(
            self.encoder_completed == self.plan.prompt_rows
                && self.publication_completed == self.plan.prompt_rows
                && self.decoder_next < self.plan.decoder.len()
                && self.plan.decoder[self.decoder_next].layer == layer,
            "CED decoder layer/completed publication frontier",
        )?;
        Ok(self.decoder_next)
    }

    fn begin_seed(&mut self, layer: usize, start: usize, rows: usize) -> Result<()> {
        self.idle()?;
        let j = self.decoder_index(layer)?;
        let range = checked_range(&self.plan.decoder[j].seed_rows, start, rows)?;
        op::require(
            start == self.seed_completed[j]
                && self.output_completed[j] == self.plan.decoder[j].output_rows.start,
            "CED seed completed frontier",
        )?;
        self.pending = Some(Pending {
            phase: Phase::Seed(layer),
            range,
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
        let j = self.decoder_index(layer)?;
        let range = checked_range(&self.plan.decoder[j].output_rows, start, rows)?;
        op::require(
            self.seed_completed[j] == self.plan.decoder[j].seed_rows.end
                && start == self.output_completed[j],
            "CED output completed frontier/seed coverage",
        )?;
        self.pending = Some(Pending {
            phase: Phase::Decoder(layer),
            range,
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

    fn complete(&mut self, phase: Phase, start: usize, rows: usize) -> Result<()> {
        let p = self.matches(phase, start, rows)?;
        op::require(
            p.encoded
                && (matches!(phase, Phase::Seed(_)) || Self::attention_done(p))
                && (!matches!(phase, Phase::Decoder(_)) || p.finished.get()),
            "CED range not fully encoded",
        )?;
        let end = p.range.end;
        match phase {
            Phase::Encoder => {
                self.encoder_completed = end;
                self.publication_completed = end;
            }
            Phase::Seed(_) => {
                self.seed_completed[self.decoder_next] = end;
            }
            Phase::Decoder(_) => {
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
            && self.encoder_completed == n
            && self.publication_completed == n
            && self.decoder_next == self.plan.decoder.len()
            && self.output_completed.iter().all(|&end| end == n)
    }
}

impl Workspace {
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
        controls.validate_layer(&self.schedule, layer)?;
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
    ) -> Result<Option<usize>> {
        let Some(ced) = &self.ced else {
            return Ok(None);
        };
        let p = ced.pending.as_ref().ok_or("CED has no attention intent")?;
        if matches!(p.phase, Phase::Decoder(_)) {
            ced.matches(Phase::Decoder(id), start, rows)?;
            return Ok(Some(ced.plan.prompt_rows));
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
            controls.validate_layer(&self.schedule, id)?;
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
                op::require(
                    floor >= ced.plan.decoder[j].seed_rows.start
                        && ced.seed_completed[j] == ced.plan.decoder[j].seed_rows.end
                        && start == ced.output_completed[j],
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
            controls.validate_layer(&self.schedule, id)?;
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
            ced.encoder_completed == n && ced.publication_completed == n && ced.pending.is_none(),
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
