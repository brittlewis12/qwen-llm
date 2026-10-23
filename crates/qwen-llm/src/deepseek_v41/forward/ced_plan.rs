use super::super::{engram::HashConfig, gguf::Schedule};
use super::{Config, prefill::DEFAULT_CHUNK_ROWS};
use std::{fmt, ops::Range};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CedRequest {
    pub prompt_rows: Option<usize>,
    pub capacity: usize,
    pub start_position: usize,
    pub prefix_reuse: bool,
    pub encoder_replay: bool,
    pub speculative_readers: bool,
    pub prompt_logprobs: bool,
    pub intermediate_checkpoints: bool,
    pub captures: bool,
}

impl CedRequest {
    pub fn cold(prompt_rows: usize, capacity: usize) -> Self {
        Self {
            prompt_rows: Some(prompt_rows),
            capacity,
            start_position: 0,
            prefix_reuse: false,
            encoder_replay: false,
            speculative_readers: false,
            prompt_logprobs: false,
            intermediate_checkpoints: false,
            captures: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CedPlanError {
    UnsupportedRequest(&'static str),
    InvalidSchedule,
    UnsupportedSchedule(&'static str),
    DecoderOwner(usize),
    DecoderEngram(usize),
    InvalidEngram,
    UnsupportedWindow(usize),
    InvalidGeometry(&'static str),
    Overflow(&'static str),
    InvalidRowView,
    InvalidChunkRows(usize),
}

impl fmt::Display for CedPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CED plan: {self:?}")
    }
}

impl std::error::Error for CedPlanError {}

type Result<T> = std::result::Result<T, CedPlanError>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CedLayerPlan {
    pub layer: usize,
    pub input_rows: Range<usize>,
    pub seed_rows: Range<usize>,
    pub output_rows: Range<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CedWorkCounters {
    pub encoder_output_rows: usize,
    pub publication_rows: usize,
    pub decoder_output_rows: usize,
    pub decoder_seed_rows: usize,
    pub total_output_rows: usize,
    pub total_decoder_kv_rows: usize,
    // Operation-row proxy: complete outputs + KV-only seeds + publication.
    pub total_execution_rows: usize,
    pub full_forward_output_rows: usize,
}

// Logical additional storage only. Ordinary scratch, buffer pricing/slack,
// command lifetimes, CPU inputs and reserve still require normal admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CedMemoryLedger {
    pub boundary_residual_bytes: u64,
    pub boundary_mix_bytes: u64,
    pub candidate_i32_per_row: usize,
    pub candidate_bytes: u64,
    pub selected_i32_per_row: usize,
    pub selected_bytes: u64,
    pub total_retained_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CedPlan {
    pub prompt_rows: usize,
    pub encoder_layers: Range<usize>,
    pub publication_layer: usize,
    pub boundary_bound_rows: usize,
    pub boundary_rows: Range<usize>,
    // Both candidate masks and selected IDs retain this absolute query origin.
    pub candidate_rows: Range<usize>,
    pub decoder: Vec<CedLayerPlan>,
    pub work: CedWorkCounters,
    pub memory: CedMemoryLedger,
}

impl CedPlan {
    pub fn shrinking(
        config: &Config,
        schedule: &Schedule,
        engram: &HashConfig,
        request: &CedRequest,
    ) -> Result<Self> {
        let n = request.prompt_rows.ok_or(CedPlanError::UnsupportedRequest(
            "known whole-prompt length required",
        ))?;
        if n == 0 {
            return Err(CedPlanError::UnsupportedRequest("empty prompt"));
        }
        if request.start_position != 0
            || request.prefix_reuse
            || request.encoder_replay
            || request.speculative_readers
            || request.prompt_logprobs
            || request.intermediate_checkpoints
            || request.captures
        {
            return Err(CedPlanError::UnsupportedRequest(
                "cold final-logits/final-state requests only",
            ));
        }
        let capacity =
            u64::try_from(request.capacity).map_err(|_| CedPlanError::Overflow("capacity"))?;
        if n > request.capacity || capacity > engram.max_position {
            return Err(CedPlanError::UnsupportedRequest("request capacity/context"));
        }
        schedule
            .validate()
            .map_err(|_| CedPlanError::InvalidSchedule)?;
        let split = schedule
            .candidate_source_layer
            .ok_or(CedPlanError::UnsupportedSchedule(
                "candidate source required",
            ))?;
        if split == 0 || split >= schedule.layers {
            return Err(CedPlanError::UnsupportedSchedule(
                "nonempty encoder and decoder required",
            ));
        }
        if !schedule.owns_kv(split) || schedule.ratio(split) != 1 {
            return Err(CedPlanError::UnsupportedSchedule(
                "candidate source must own ratio-one KV and index keys",
            ));
        }
        if let Some(&layer) = schedule.kv_source_layers.iter().find(|&&l| l > split) {
            return Err(CedPlanError::DecoderOwner(layer));
        }
        for layer in split..schedule.layers {
            if schedule.owner(layer) != Some(split) || schedule.ratio(layer) != 1 {
                return Err(CedPlanError::UnsupportedSchedule(
                    "all decoder layers must reuse the candidate-source owner",
                ));
            }
        }
        engram.validate().map_err(|_| CedPlanError::InvalidEngram)?;
        for hash in &engram.layers {
            if hash.layer >= schedule.layers {
                return Err(CedPlanError::InvalidEngram);
            }
            if hash.layer >= split {
                return Err(CedPlanError::DecoderEngram(hash.layer));
            }
        }
        if config.window == 0 || config.window > u32::MAX as usize {
            return Err(CedPlanError::UnsupportedWindow(config.window));
        }
        if config.hidden == 0 {
            return Err(CedPlanError::InvalidGeometry("zero hidden width"));
        }
        if config.candidate_block == 0 || config.candidate_block > u32::MAX as usize {
            return Err(CedPlanError::InvalidGeometry("candidate block size"));
        }
        if config.index_topk == 0 {
            return Err(CedPlanError::InvalidGeometry("zero selected-ID width"));
        }

        let depth = schedule.layers - split;
        let halo = config.window - 1;
        let boundary_bound_rows = boundary_bound(depth, halo)?;
        let boundary_rows = n.saturating_sub(boundary_bound_rows)..n;
        let full_forward_output_rows = mul(n, schedule.layers, "full-forward output rows")?;
        let encoder_output_rows = mul(n, split, "encoder output rows")?;
        let mut decoder_output_rows = 0;
        let mut decoder_seed_rows = 0;
        let mut decoder = Vec::with_capacity(depth);
        for j in 0..depth {
            let preceding = mul(depth - 1 - j, halo, "decoder output halo")?;
            let a = n.saturating_sub(add(1, preceding, "decoder output distance")?);
            let s = a.saturating_sub(halo);
            decoder_output_rows = add(decoder_output_rows, n - a, "decoder output rows")?;
            decoder_seed_rows = add(decoder_seed_rows, a - s, "decoder seed rows")?;
            decoder.push(CedLayerPlan {
                layer: split + j,
                input_rows: s..n,
                seed_rows: s..a,
                output_rows: a..n,
            });
        }
        let candidate_rows = decoder[0].output_rows.clone();
        let total_output_rows = add(
            encoder_output_rows,
            decoder_output_rows,
            "total output rows",
        )?;
        let total_decoder_kv_rows = add(decoder_output_rows, decoder_seed_rows, "decoder KV rows")?;
        let total_execution_rows = add(
            add(total_output_rows, decoder_seed_rows, "total execution rows")?,
            n,
            "total execution rows",
        )?;
        let boundary_count = boundary_rows.len();
        let candidate_i32_per_row = request.capacity.div_ceil(config.candidate_block);
        let selected_i32_per_row = config.index_topk;
        // Four streams and four outgoing mix coefficients are fixed by native mHC.
        let residual_width = mul(4, config.hidden, "boundary residual width")?;
        let boundary_residual_bytes = bytes(boundary_count, residual_width)?;
        let boundary_mix_bytes = bytes(boundary_count, 4)?;
        let candidate_bytes = bytes(candidate_rows.len(), candidate_i32_per_row)?;
        let selected_bytes = bytes(candidate_rows.len(), selected_i32_per_row)?;
        let total_retained_bytes = boundary_residual_bytes
            .checked_add(boundary_mix_bytes)
            .and_then(|n| n.checked_add(candidate_bytes))
            .and_then(|n| n.checked_add(selected_bytes))
            .ok_or(CedPlanError::Overflow("retained bytes"))?;
        Ok(Self {
            prompt_rows: n,
            encoder_layers: 0..split,
            publication_layer: split,
            boundary_bound_rows,
            boundary_rows,
            candidate_rows,
            decoder,
            work: CedWorkCounters {
                encoder_output_rows,
                publication_rows: n,
                decoder_output_rows,
                decoder_seed_rows,
                total_output_rows,
                total_decoder_kv_rows,
                total_execution_rows,
                full_forward_output_rows,
            },
            memory: CedMemoryLedger {
                boundary_residual_bytes,
                boundary_mix_bytes,
                candidate_i32_per_row,
                candidate_bytes,
                selected_i32_per_row,
                selected_bytes,
                total_retained_bytes,
            },
        })
    }
}

fn add(a: usize, b: usize, term: &'static str) -> Result<usize> {
    a.checked_add(b).ok_or(CedPlanError::Overflow(term))
}

fn mul(a: usize, b: usize, term: &'static str) -> Result<usize> {
    a.checked_mul(b).ok_or(CedPlanError::Overflow(term))
}

fn boundary_bound(depth: usize, halo: usize) -> Result<usize> {
    add(1, mul(depth, halo, "boundary halo")?, "boundary bound")
}

fn bytes(rows: usize, width: usize) -> Result<u64> {
    let bytes = mul(mul(rows, width, "retained elements")?, 4, "retained bytes")?;
    u64::try_from(bytes).map_err(|_| CedPlanError::Overflow("retained bytes"))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CedChunkView {
    pub absolute_rows: Range<usize>,
    // Relative to the supplied retained range, never a rebased query position.
    pub row_offset: usize,
}

pub fn row_view(retained: &Range<usize>, rows: Range<usize>) -> Result<CedChunkView> {
    if retained.start > retained.end
        || rows.start > rows.end
        || rows.start < retained.start
        || rows.end > retained.end
    {
        return Err(CedPlanError::InvalidRowView);
    }
    if rows.len() > DEFAULT_CHUNK_ROWS {
        return Err(CedPlanError::InvalidChunkRows(rows.len()));
    }
    Ok(CedChunkView {
        row_offset: rows.start - retained.start,
        absolute_rows: rows,
    })
}

pub struct CedChunks {
    origin: usize,
    next: usize,
    end: usize,
    chunk_rows: usize,
}

pub fn chunk_views(rows: Range<usize>, chunk_rows: usize) -> Result<CedChunks> {
    if !(1..=DEFAULT_CHUNK_ROWS).contains(&chunk_rows) {
        return Err(CedPlanError::InvalidChunkRows(chunk_rows));
    }
    if rows.start > rows.end {
        return Err(CedPlanError::InvalidRowView);
    }
    Ok(CedChunks {
        origin: rows.start,
        next: rows.start,
        end: rows.end,
        chunk_rows,
    })
}

impl Iterator for CedChunks {
    type Item = CedChunkView;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.end {
            return None;
        }
        // The increment is bounded by end - next, including at usize::MAX.
        let end = self.next + (self.end - self.next).min(self.chunk_rows);
        let view = CedChunkView {
            absolute_rows: self.next..end,
            row_offset: self.next - self.origin,
        };
        self.next = end;
        Some(view)
    }
}

impl std::iter::FusedIterator for CedChunks {}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(n: usize) -> CedPlan {
        CedPlan::shrinking(
            &Config::default(),
            &Schedule::default(),
            &HashConfig::default(),
            &CedRequest::cold(n, n),
        )
        .unwrap()
    }

    fn assert_ranges(plan: &CedPlan, window: usize) {
        let n = plan.prompt_rows;
        let halo = window - 1;
        let depth = plan.decoder.len();
        assert_eq!(plan.boundary_bound_rows, 1 + depth * halo);
        assert_eq!(plan.boundary_rows, n.saturating_sub(1 + depth * halo)..n);
        assert_eq!(plan.candidate_rows, plan.decoder[0].output_rows);
        let mut previous = &plan.boundary_rows;
        let mut outputs = 0;
        let mut seeds = 0;
        for (j, layer) in plan.decoder.iter().enumerate() {
            let a = n.saturating_sub(1 + (depth - 1 - j) * halo);
            let s = a.saturating_sub(halo);
            assert_eq!(layer.layer, plan.publication_layer + j);
            assert_eq!(layer.input_rows, s..n);
            assert_eq!(&layer.input_rows, previous);
            assert_eq!(layer.seed_rows, s..a);
            assert_eq!(layer.output_rows, a..n);
            assert!(layer.seed_rows.len() <= halo);
            assert!(layer.input_rows.start <= n.saturating_sub(window));
            assert!(!layer.output_rows.is_empty());
            assert!(plan.candidate_rows.start <= layer.output_rows.start);
            for rows in [&layer.input_rows, &layer.seed_rows, &layer.output_rows] {
                let mut next = rows.start;
                for view in chunk_views(rows.clone(), DEFAULT_CHUNK_ROWS).unwrap() {
                    assert_eq!(view.absolute_rows.start, next);
                    assert_eq!(view.row_offset, next - rows.start);
                    assert!((1..=2048).contains(&view.absolute_rows.len()));
                    if rows == &layer.output_rows {
                        assert!(row_view(&plan.candidate_rows, view.absolute_rows.clone()).is_ok());
                    }
                    next = view.absolute_rows.end;
                }
                assert_eq!(next, rows.end);
            }
            outputs += layer.output_rows.len();
            seeds += layer.seed_rows.len();
            previous = &layer.output_rows;
        }
        assert_eq!(plan.decoder.last().unwrap().output_rows, n - 1..n);
        assert_eq!(plan.work.decoder_output_rows, outputs);
        assert_eq!(plan.work.decoder_seed_rows, seeds);
        assert_eq!(plan.work.encoder_output_rows, n * plan.encoder_layers.len());
        assert_eq!(plan.work.publication_rows, n);
        assert_eq!(
            plan.work.total_output_rows,
            plan.work.encoder_output_rows + outputs
        );
        assert_eq!(plan.work.total_decoder_kv_rows, outputs + seeds);
        assert_eq!(
            plan.work.total_execution_rows,
            plan.work.total_output_rows + seeds + n
        );
        assert_eq!(
            plan.work.full_forward_output_rows,
            n * (plan.encoder_layers.len() + depth)
        );
    }

    #[test]
    fn prompt_edges_and_exact_long_prompt_work() {
        for n in [1, 127, 128, 129, 2047, 2048, 2049, 2540, 2541, 2542, 8192] {
            assert_ranges(&plan(n), 128);
        }
        let p = plan(8192);
        assert_eq!(p.encoder_layers, 0..20);
        assert_eq!(p.publication_layer, 20);
        assert_eq!(p.boundary_bound_rows, 2541);
        assert_eq!(p.boundary_rows, 5651..8192);
        assert_eq!(p.candidate_rows, 5778..8192);
        assert_eq!(p.work.decoder_output_rows, 24_150);
        assert_eq!(p.work.decoder_seed_rows, 2540);
        assert_eq!(p.work.encoder_output_rows, 163_840);
        assert_eq!(p.work.total_output_rows, 187_990);
        assert_eq!(p.work.total_decoder_kv_rows, 26_690);
        assert_eq!(p.work.total_execution_rows, 198_722);
        assert_eq!(p.work.full_forward_output_rows, 327_680);
        assert_eq!(plan(2048).work.decoder_output_rows, 23_433);
        assert_eq!(plan(512).work.decoder_output_rows, 8955);
        assert_eq!(plan(1).work.decoder_seed_rows, 0);
        assert_eq!(plan(2540).boundary_rows.start, 0);
        assert_eq!(plan(2541).boundary_rows.start, 0);
        assert_eq!(plan(2542).boundary_rows.start, 1);
    }

    #[test]
    fn short_asymmetric_profiles_and_window_one_remain_general() {
        for (cut, depth) in [
            (1, 1),
            (2, 2),
            (4, 4),
            (20, 20),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 4),
            (7, 2),
            (20, 19),
        ] {
            let schedule = Schedule {
                layers: cut + depth,
                kv_source_layers: vec![cut],
                index_source_layers: (cut..cut + depth).collect(),
                compression_ratios: (0..cut + depth).map(|l| usize::from(l >= cut)).collect(),
                candidate_source_layer: Some(cut),
            };
            let engram = HashConfig {
                layers: Vec::new(),
                ..HashConfig::default()
            };
            for window in [1, 2, 4, 7, 128, 256] {
                let config = Config {
                    hidden: 13,
                    window,
                    candidate_block: 3,
                    index_topk: 7,
                    ..Config::default()
                };
                let bound = 1 + depth * (window - 1);
                for n in [1, window, bound.saturating_sub(1).max(1), bound, bound + 3] {
                    let p =
                        CedPlan::shrinking(&config, &schedule, &engram, &CedRequest::cold(n, n))
                            .unwrap();
                    assert_ranges(&p, window);
                    assert_eq!(p.encoder_layers, 0..cut);
                    assert_eq!(p.publication_layer, cut);
                    assert_eq!(p.decoder.len(), depth);
                    assert_eq!(
                        p.memory.boundary_residual_bytes,
                        p.boundary_rows.len() as u64 * 4 * 13 * 4
                    );
                    assert_eq!(p.memory.candidate_i32_per_row, n.div_ceil(3));
                    assert_eq!(p.memory.selected_i32_per_row, 7);
                    assert_eq!(
                        p.memory.selected_bytes,
                        p.candidate_rows.len() as u64 * 7 * 4
                    );
                    if window == 1 {
                        assert_eq!(p.boundary_rows, n - 1..n);
                        assert_eq!(p.candidate_rows, n - 1..n);
                        assert_eq!(p.work.decoder_output_rows, depth);
                        assert_eq!(p.work.decoder_seed_rows, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn separate_f32_boundary_and_i32_candidate_and_selected_pricing() {
        let p = plan(8192);
        assert_eq!(p.memory.boundary_residual_bytes, 2541 * 4 * 5120 * 4);
        assert_eq!(p.memory.boundary_mix_bytes, 2541 * 4 * 4);
        let boundary = p.memory.boundary_residual_bytes + p.memory.boundary_mix_bytes;
        assert_eq!(boundary, 208_199_376);
        assert_eq!((boundary + (1 << 19)) / (1 << 20), 199);
        assert_eq!(p.memory.candidate_i32_per_row, 1024);
        assert_eq!(p.memory.candidate_bytes, 9_887_744);
        assert_eq!(p.memory.selected_i32_per_row, 512);
        assert_eq!(p.memory.selected_bytes, 4_943_872);
        assert_eq!(
            p.memory.total_retained_bytes,
            boundary + p.memory.candidate_bytes + p.memory.selected_bytes
        );
        for cap in [8192, 8193, 1_000_000, 1_048_576] {
            let p = CedPlan::shrinking(
                &Config::default(),
                &Schedule::default(),
                &HashConfig::default(),
                &CedRequest::cold(8192, cap),
            )
            .unwrap();
            assert_eq!(
                p.memory.candidate_i32_per_row,
                cap / 8 + usize::from(cap % 8 != 0)
            );
            assert_eq!(p.memory.candidate_bytes, 2414 * cap.div_ceil(8) as u64 * 4);
            assert_eq!(p.memory.selected_i32_per_row, 512);
            assert_eq!(p.memory.selected_bytes, 4_943_872);
            assert_eq!(
                p.memory.total_retained_bytes,
                boundary + p.memory.candidate_bytes + p.memory.selected_bytes
            );
            if cap == 1_048_576 {
                assert_eq!(p.memory.candidate_bytes, 1_265_631_232);
                assert_eq!(p.memory.total_retained_bytes, 1_478_774_480);
            }
        }
        let p = plan(1);
        assert_eq!(p.memory.boundary_residual_bytes, 4 * 5120 * 4);
        assert_eq!(p.memory.boundary_mix_bytes, 16);
        assert_eq!(p.memory.candidate_bytes, 4);
        assert_eq!(p.memory.selected_i32_per_row, 512);
        assert_eq!(p.memory.selected_bytes, 2048);
    }

    #[test]
    fn rejects_later_owner_and_decoder_engram_but_allows_reindex() {
        let config = Config::default();
        let mut schedule = Schedule::default();
        let mut engram = HashConfig::default();
        let request = CedRequest::cold(8192, 8192);
        assert!(CedPlan::shrinking(&config, &schedule, &engram, &request).is_ok());
        schedule.kv_source_layers.push(24);
        assert!(schedule.validate().is_ok());
        assert_eq!(
            CedPlan::shrinking(&config, &schedule, &engram, &request),
            Err(CedPlanError::DecoderOwner(24))
        );
        schedule = Schedule::default();
        engram.layers.last_mut().unwrap().layer = 20;
        assert!(engram.validate().is_ok());
        assert_eq!(
            CedPlan::shrinking(&config, &schedule, &engram, &request),
            Err(CedPlanError::DecoderEngram(20))
        );
        engram.layers.last_mut().unwrap().layer = 39;
        assert_eq!(
            CedPlan::shrinking(&config, &schedule, &engram, &request),
            Err(CedPlanError::DecoderEngram(39))
        );
    }

    #[test]
    fn rejects_incompatible_or_malformed_schedules() {
        let check = |s: &Schedule| {
            CedPlan::shrinking(
                &Config::default(),
                s,
                &HashConfig::default(),
                &CedRequest::cold(8, 8),
            )
        };
        let mut s = Schedule {
            candidate_source_layer: None,
            ..Schedule::default()
        };
        assert!(matches!(
            check(&s),
            Err(CedPlanError::UnsupportedSchedule(_))
        ));
        s = Schedule::default();
        s.kv_source_layers.pop();
        s.candidate_source_layer = None;
        s.compression_ratios[20..].fill(2);
        assert!(s.validate().is_ok());
        assert!(matches!(
            check(&s),
            Err(CedPlanError::UnsupportedSchedule(_))
        ));
        s = Schedule::default();
        s.compression_ratios[20..].fill(2);
        s.candidate_source_layer = None;
        assert!(s.validate().is_ok());
        assert!(matches!(
            check(&s),
            Err(CedPlanError::UnsupportedSchedule(_))
        ));
        s = Schedule::default();
        s.layers = 39;
        s.compression_ratios.pop();
        assert!(s.validate().is_ok());
        let p = check(&s).unwrap();
        assert_eq!(p.publication_layer, 20);
        assert_eq!(p.decoder.len(), 19);
        assert_ranges(&p, 128);
        s = Schedule {
            layers: 3,
            kv_source_layers: vec![0],
            index_source_layers: vec![0],
            compression_ratios: vec![1; 3],
            candidate_source_layer: Some(0),
        };
        assert!(s.validate().is_ok());
        assert!(matches!(
            check(&s),
            Err(CedPlanError::UnsupportedSchedule(_))
        ));
        s = Schedule::default();
        s.candidate_source_layer = Some(21);
        assert_eq!(check(&s), Err(CedPlanError::InvalidSchedule));
        s.candidate_source_layer = Some(14);
        assert_eq!(check(&s), Err(CedPlanError::InvalidSchedule));
        s.candidate_source_layer = Some(s.layers);
        assert_eq!(check(&s), Err(CedPlanError::InvalidSchedule));
        s = Schedule::default();
        s.compression_ratios.pop();
        assert_eq!(check(&s), Err(CedPlanError::InvalidSchedule));
        s = Schedule::default();
        s.kv_source_layers.swap(0, 1);
        assert_eq!(check(&s), Err(CedPlanError::InvalidSchedule));
        s = Schedule::default();
        s.candidate_source_layer = Some(usize::MAX);
        assert_eq!(check(&s), Err(CedPlanError::InvalidSchedule));
    }

    #[test]
    fn asymmetric_cut_still_rejects_later_owners_and_decoder_engram() {
        let config = Config::default();
        let mut schedule = Schedule {
            layers: 5,
            kv_source_layers: vec![2],
            index_source_layers: vec![2, 3, 4],
            compression_ratios: vec![0, 0, 1, 1, 1],
            candidate_source_layer: Some(2),
        };
        let mut engram = HashConfig::default();
        engram.layers.truncate(1);
        let request = CedRequest::cold(8192, 8192);
        let p = CedPlan::shrinking(&config, &schedule, &engram, &request).unwrap();
        assert_eq!(p.encoder_layers, 0..2);
        assert_eq!(p.decoder.len(), 3);
        assert_ranges(&p, 128);
        schedule.kv_source_layers.push(4);
        assert!(schedule.validate().is_ok());
        assert_eq!(
            CedPlan::shrinking(&config, &schedule, &engram, &request),
            Err(CedPlanError::DecoderOwner(4))
        );
        schedule.kv_source_layers.pop();
        for layer in [2, 4] {
            engram.layers[0].layer = layer;
            assert!(engram.validate().is_ok());
            assert_eq!(
                CedPlan::shrinking(&config, &schedule, &engram, &request),
                Err(CedPlanError::DecoderEngram(layer))
            );
        }
    }

    #[test]
    fn excludes_non_cold_or_extra_reader_requests() {
        let base = CedRequest::cold(8, 8);
        let check = |r: &CedRequest| {
            CedPlan::shrinking(
                &Config::default(),
                &Schedule::default(),
                &HashConfig::default(),
                r,
            )
        };
        let modifications: [fn(&mut CedRequest); 11] = [
            |r: &mut CedRequest| r.prompt_rows = None,
            |r: &mut CedRequest| r.prompt_rows = Some(0),
            |r: &mut CedRequest| r.start_position = 1,
            |r: &mut CedRequest| r.prefix_reuse = true,
            |r: &mut CedRequest| r.encoder_replay = true,
            |r: &mut CedRequest| r.speculative_readers = true,
            |r: &mut CedRequest| r.prompt_logprobs = true,
            |r: &mut CedRequest| r.intermediate_checkpoints = true,
            |r: &mut CedRequest| r.captures = true,
            |r: &mut CedRequest| r.capacity = 7,
            |r: &mut CedRequest| r.capacity = 1_048_577,
        ];
        for modify in modifications {
            let mut r = base.clone();
            modify(&mut r);
            assert!(matches!(
                check(&r),
                Err(CedPlanError::UnsupportedRequest(_))
            ));
        }
    }

    #[test]
    fn checks_window_geometry_and_arithmetic_overflow() {
        let check = |c: &Config, r: &CedRequest| {
            CedPlan::shrinking(
                c,
                &Schedule::default(),
                &HashConfig {
                    max_position: u64::MAX,
                    ..HashConfig::default()
                },
                r,
            )
        };
        let r = CedRequest::cold(8192, 8192);
        let mut c = Config {
            window: 0,
            ..Config::default()
        };
        assert_eq!(check(&c, &r), Err(CedPlanError::UnsupportedWindow(0)));
        if let Some(window) = (u32::MAX as usize).checked_add(1) {
            c.window = window;
            assert_eq!(
                check(&c, &r),
                Err(CedPlanError::UnsupportedWindow(c.window))
            );
        }
        c = Config::default();
        c.hidden = 0;
        assert!(matches!(
            check(&c, &r),
            Err(CedPlanError::InvalidGeometry(_))
        ));
        c.hidden = usize::MAX;
        assert!(matches!(check(&c, &r), Err(CedPlanError::Overflow(_))));
        c = Config::default();
        c.candidate_block = 0;
        assert!(matches!(
            check(&c, &r),
            Err(CedPlanError::InvalidGeometry(_))
        ));
        c = Config::default();
        c.index_topk = 0;
        assert_eq!(
            check(&c, &r),
            Err(CedPlanError::InvalidGeometry("zero selected-ID width"))
        );
        c.index_topk = usize::MAX;
        assert!(matches!(check(&c, &r), Err(CedPlanError::Overflow(_))));
        c = Config::default();
        assert!(matches!(
            check(&c, &CedRequest::cold(usize::MAX, usize::MAX)),
            Err(CedPlanError::Overflow(_))
        ));
        assert!(matches!(
            check(&c, &CedRequest::cold(8192, usize::MAX)),
            Err(CedPlanError::Overflow(_))
        ));
        let short = Schedule {
            layers: 2,
            kv_source_layers: vec![1],
            index_source_layers: vec![1],
            compression_ratios: vec![0, 1],
            candidate_source_layer: Some(1),
        };
        let engram = HashConfig {
            layers: Vec::new(),
            max_position: u64::MAX,
            ..HashConfig::default()
        };
        let n = usize::MAX / 2;
        assert_eq!(
            CedPlan::shrinking(&c, &short, &engram, &CedRequest::cold(n, n)),
            Err(CedPlanError::Overflow("total execution rows"))
        );
        if usize::BITS == 64 {
            c.hidden = usize::MAX / (2541 * 16);
            assert_eq!(check(&c, &r), Err(CedPlanError::Overflow("retained bytes")));
            c = Config::default();
            c.index_topk = usize::MAX / (2414 * 4);
            assert_eq!(check(&c, &r), Err(CedPlanError::Overflow("retained bytes")));
        }
        assert!(matches!(
            boundary_bound(usize::MAX, 2),
            Err(CedPlanError::Overflow(_))
        ));
        assert!(matches!(
            boundary_bound(usize::MAX, 1),
            Err(CedPlanError::Overflow(_))
        ));
        assert!(matches!(
            bytes(usize::MAX, 4),
            Err(CedPlanError::Overflow(_))
        ));
        assert!(matches!(
            add(usize::MAX, 1, "counter"),
            Err(CedPlanError::Overflow("counter"))
        ));
    }

    #[test]
    fn bounded_views_preserve_absolute_and_retained_origins() {
        let p = plan(8192);
        let views: Vec<_> = chunk_views(p.boundary_rows.clone(), 2048)
            .unwrap()
            .collect();
        assert_eq!(
            views,
            vec![
                CedChunkView {
                    absolute_rows: 5651..7699,
                    row_offset: 0
                },
                CedChunkView {
                    absolute_rows: 7699..8192,
                    row_offset: 2048
                },
            ]
        );
        let output_views: Vec<_> = chunk_views(p.decoder[1].output_rows.clone(), 2048)
            .unwrap()
            .collect();
        let candidate = row_view(&p.candidate_rows, output_views[0].absolute_rows.clone()).unwrap();
        assert_eq!(candidate.row_offset, 127);
        assert_eq!(candidate.absolute_rows.start, 5905);
        let boundary = row_view(
            &p.boundary_rows,
            p.candidate_rows.start..p.candidate_rows.start + 2048,
        )
        .unwrap();
        assert_eq!(boundary.row_offset, 127);
        assert!(matches!(
            row_view(&p.boundary_rows, p.candidate_rows.clone()),
            Err(CedPlanError::InvalidChunkRows(2414))
        ));
        assert!(chunk_views(0..1, 0).is_err());
        assert!(chunk_views(0..1, 2049).is_err());
        assert!(chunk_views(Range { start: 2, end: 1 }, 1).is_err());
        assert!(chunk_views(1..1, 1).unwrap().next().is_none());
        assert!(row_view(&(10..20), 9..11).is_err());
        assert!(row_view(&(10..20), 19..21).is_err());
        assert!(row_view(&(10..20), Range { start: 15, end: 14 }).is_err());
        assert!(row_view(&Range { start: 20, end: 10 }, 10..10).is_err());
        let end = usize::MAX;
        let mut views = chunk_views(end - 2..end, 1).unwrap();
        assert_eq!(views.next().unwrap().absolute_rows, end - 2..end - 1);
        assert_eq!(views.next().unwrap().row_offset, 1);
        assert!(views.next().is_none());
        assert!(views.next().is_none());
    }
}
