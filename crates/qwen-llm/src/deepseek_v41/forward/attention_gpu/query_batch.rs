//! Bounded query tiles; compressor publication and discrete selection stay serial.
//! Raw SWA is layer-local in the scalar contract. Its old-window + chunk history
//! is consumed before this scratch is reused by the next layer. Shared compressed
//! KV/keys remain append-only, with each follower bounded by its query's length.
use super::*;

#[cfg(test)]
#[path = "query_batch/arithmetic.rs"]
mod arithmetic_tests;

pub(super) const TILE: usize = 512;

/// Rows per query batch tile. The default is the production width; the
/// environment override exists for same-binary A/B measurement of the width
/// itself and is read once per process.
pub(super) fn tile_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("QWEN_V41_SINK_TILE")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&rows| (1..=prefill::DEFAULT_CHUNK_ROWS).contains(&rows))
            .unwrap_or(TILE)
    })
}

#[cfg(test)]
thread_local! {
    static F32_ATTENTION_ROWS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(in super::super) fn with_f32_attention<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            F32_ATTENTION_ROWS.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(F32_ATTENTION_ROWS.with(|flag| flag.replace(true)));
    f()
}

/// Scratch planning and execution must agree on whether gathered F32 PV is reachable.
fn mma_geometry(c: &Config) -> bool {
    #[cfg(test)]
    if F32_ATTENTION_ROWS.with(|flag| flag.get()) {
        return false;
    }
    c.head_dim == MMA_DIM && c.heads.is_multiple_of(MMA_HEADS)
}

pub(super) struct Scratch {
    pub(super) history: MetalTensor,
    pub(super) scores: MetalTensor,
    pub(super) attended: MetalTensor,
    pub(super) matrix: Option<ValueMatrix>,
}

impl Scratch {
    pub(super) fn shapes(c: &Config, rows: usize) -> Result<Vec<usize>> {
        op::require(
            (1..=prefill::MAX_CHUNK_ROWS).contains(&rows),
            "query batch scratch rows",
        )?;
        let tile = rows.min(tile_rows());
        let mut sizes = vec![
            checked_product(&[
                c.window.checked_add(rows).ok_or("raw history extent")?,
                c.head_dim,
            ])?,
            checked_product(&[
                tile,
                c.heads,
                c.window
                    .checked_add(c.index_topk)
                    .and_then(|n| n.checked_add(1))
                    .ok_or("query score extent")?,
            ])?,
            checked_product(&[tile, c.heads, c.head_dim])?,
        ];
        if !mma_geometry(c) {
            for size in ValueMatrix::shapes(c)? {
                sizes.push(checked_product(&[tile, size])?);
            }
        }
        Ok(sizes)
    }

    pub(super) fn new(ctx: &MetalContext, c: &Config, rows: usize) -> Result<Self> {
        let sizes = Self::shapes(c, rows)?;
        let f = |i| tensor(ctx, sizes[i], GgmlType::F32);
        Ok(Self {
            history: f(0)?,
            scores: f(1)?,
            attended: f(2)?,
            matrix: if sizes.len() == 5 {
                Some(ValueMatrix {
                    values: f(3)?,
                    probabilities: f(4)?,
                })
            } else {
                None
            },
        })
    }
}

impl Rows {
    // Admission prices one allocation per entry. Round up rather than dropping
    // the fixed old-window prefix or underpricing a short final tile.
    pub(in super::super) fn tile_widths(c: &Config, rows: usize) -> Result<Vec<usize>> {
        Ok(Scratch::shapes(c, rows)?
            .into_iter()
            .map(|n| n.div_ceil(rows))
            .collect())
    }
}

pub(super) fn rope_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    output: &MetalTensor,
    frequencies: &MetalTensor,
    dim: usize,
    heads: usize,
    pairs: usize,
    start: usize,
    rows: usize,
    inverse_grouped: bool,
) -> Result<()> {
    // The grouped PV result is [head, query, dim]; this pass also restores
    // [query, head, dim], including the non-rotary coordinates.
    let pso = pipeline(ctx, enc, "rope_rows")?;
    enc.set_bytes(0, &[dim as u32, heads as u32, pairs as u32, start as u32]);
    enc.set_bytes(3, &[rows as u32, inverse_grouped as u32]);
    enc.set_tensor(1, input);
    enc.set_tensor(2, frequencies);
    enc.set_tensor(4, output);
    dispatch(enc, &pso, rows * heads * dim);
    Ok(())
}

/// Geometry the simdgroup-MMA sink tile kernels are written for.
const MMA_DIM: usize = 512;
const MMA_HEADS: usize = 16;

impl Workspace {
    /// The three steps below in order; production composes them itself so
    /// the stage profiler can sample each.
    #[cfg(test)]
    pub(in super::super) fn encode_rows(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        start: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
        rows: &Rows,
    ) -> Result<()> {
        self.encode_rows_prepare(ctx, enc, id, start, w, input, rows)?;
        self.encode_rows_serial(ctx, enc, id, start, w, input, rows)?;
        self.encode_rows_tiles(ctx, enc, id, start, rows)
    }

    /// Projections, RoPE, history gather and raw-ring publish for the chunk.
    pub(in super::super) fn encode_rows_prepare(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        start: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
        rows: &Rows,
    ) -> Result<()> {
        op::require(!enc.is_concurrent(), "query batch requires serial encoder")?;
        op::require(
            self.active_batch().is_some_and(|b| {
                b.start == start && b.rows == rows.rows && b.layer == id && b.row == 0
            }),
            "query batch layer order",
        )?;
        validate_weights(&self.config, id, w, &self.schedule)?;
        valid_view(input, checked_product(&[self.config.hidden, rows.rows])?)?;
        let mode = self.global_mode(id, start, rows.rows)?;
        self.bind_ced_rows(id, start, rows)?;
        let history = self.row_history(id, start, rows.rows)?;
        self.prepare_rows_mode(ctx, enc, id, w, input, rows, mode)
            .map_err(|e| e.to_string())?;
        let c = &self.config;
        let constants = &self.constants[id];
        let needs_index = self.schedule.produces_index(id) && {
            let length = (start + rows.rows) / self.schedule.ratio(id);
            length > c.index_topk || length.div_ceil(c.candidate_block) > c.candidate_topk
        };
        let tile = tile_rows();
        for first in (0..rows.rows).step_by(tile) {
            let n = (rows.rows - first).min(tile);
            for (tensor, dim, heads) in [
                (&rows.query, c.head_dim, c.heads),
                (&rows.latent, c.head_dim, 1),
                (
                    &rows.index_query,
                    c.index_dim,
                    if needs_index { c.index_heads } else { 0 },
                ),
            ] {
                if heads == 0 || c.rope_dim == 0 {
                    continue;
                }
                let view = tensor.view_subrange(
                    (first * dim * heads) as u64,
                    vec![(dim * heads) as u64, n as u64],
                );
                rope_rows(
                    ctx,
                    enc,
                    &view,
                    &view,
                    &constants.frequencies,
                    dim,
                    heads,
                    c.rope_dim / 2,
                    start + first,
                    n,
                    false,
                )?;
            }
        }
        let raw = self.raw.view_subrange(
            (id * c.window * c.head_dim) as u64,
            vec![(c.window * c.head_dim) as u64],
        );
        let old = start - history.floor;
        let pso = pipeline(ctx, enc, "history")?;
        enc.set_bytes(
            0,
            &[
                c.head_dim as u32,
                c.window as u32,
                start as u32,
                rows.rows as u32,
            ],
        );
        enc.set_tensor(1, &raw);
        enc.set_tensor(2, &rows.latent);
        enc.set_tensor(3, &rows.tile.history);
        enc.set_bytes(4, &[old as u32, history.floor as u32]);
        dispatch(enc, &pso, (old + rows.rows) * c.head_dim);
        // Separate dispatch: no thread may overwrite the ring while another
        // still gathers its old prefix. Each physical slot has one writer.
        let pso = pipeline(ctx, enc, "publish_raw")?;
        dispatch(enc, &pso, rows.rows.min(c.window) * c.head_dim);
        rows.history.set(Some(history));
        Ok(())
    }

    /// Position-ordered compressor, index and selection transitions. For
    /// unpruned rows the per-row compressor/key publication stays ordered,
    /// then selection is batched over query tiles. The pruning path (from
    /// 16,385 visible keys) keeps the incumbent per-row selector.
    pub(in super::super) fn encode_rows_serial(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        start: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
        rows: &Rows,
    ) -> Result<()> {
        let mode = self.global_mode(id, start, rows.rows)?;
        self.validate_global_mode(id, start + rows.rows - 1, mode)?;
        let c = &self.config;
        let hidden = c.hidden;
        let owner = self.schedule.owner_slot(id);
        let ratio = self.schedule.ratio(id);
        let last_length = (start + rows.rows) / ratio.max(1);
        let batched = self.schedule.produces_index(id)
            && owner.is_some()
            && ratio > 0
            && last_length.div_ceil(c.candidate_block) <= c.candidate_topk
            && std::env::var_os("QWEN_V41_SELECT_PER_ROW").is_none();
        for row in 0..rows.rows {
            self.encode_batch_row_mode(
                ctx,
                enc,
                id,
                start + row,
                w,
                &prefill::row(input, hidden, row),
                rows,
                row,
                !batched,
                mode,
            )?;
        }
        if batched {
            let tile = tile_rows();
            for first in (0..rows.rows).step_by(tile) {
                self.select_rows_unpruned(
                    ctx,
                    enc,
                    id,
                    start,
                    rows,
                    first,
                    (rows.rows - first).min(tile),
                )?;
            }
        }
        #[cfg(test)]
        super::super::ced::tests::chunked::encode_control_probes(
            ctx,
            enc,
            id,
            start,
            rows.rows,
            &rows.selected,
            &rows.candidates,
        )?;
        Ok(())
    }

    /// Batched unpruned selection: prepare ids/masks, compute index scores,
    /// then run the incumbent per-row sort and chronological pass. Pruning
    /// (block scores/keep) is not reached before 16,385 visible keys and
    /// stays on the per-row path.
    fn select_rows_unpruned(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        start: usize,
        rows: &Rows,
        first: usize,
        n: usize,
    ) -> Result<()> {
        let c = &self.config;
        let owner = self
            .schedule
            .owner_slot(id)
            .ok_or("missing selection owner")?;
        let ratio = self.schedule.ratio(id);
        let publish = self.schedule.candidate_source_layer == Some(id);
        let last_length = (start + first + n) / ratio;
        let selected_stride = c.index_topk;
        let candidates_stride = self.capacity.div_ceil(c.candidate_block);
        let select_width = Rows::select_width(c, self.capacity);
        // The tile can contain both all-selected rows (length <= top_k) and
        // top-k rows. The last row may be top-k, so bound the x grid by the
        // tile's maximum visible all-selected extent, not by the last row.
        let all_bound = last_length.min(c.index_topk);
        let grid_w = all_bound.max(if publish {
            last_length.div_ceil(c.candidate_block)
        } else {
            0
        });
        if grid_w > 0 {
            pipeline(ctx, enc, "selection_prepare_rows")?;
            enc.set_bytes(
                0,
                &[
                    start as u32,
                    ratio as u32,
                    c.index_topk as u32,
                    c.candidate_block as u32,
                ],
            );
            enc.set_bytes(
                1,
                &[
                    selected_stride as u32,
                    candidates_stride as u32,
                    n as u32,
                    publish as u32,
                ],
            );
            enc.set_tensor(2, &rows.selected);
            enc.set_tensor(3, &rows.candidates);
            enc.set_bytes(4, &(first as u32));
            enc.dispatch(
                MTLSize {
                    width: grid_w.div_ceil(256),
                    height: n,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        if last_length > c.index_topk {
            pipeline(ctx, enc, "index_heads_rows")?;
            enc.set_bytes(
                0,
                &[
                    start as u32,
                    ratio as u32,
                    c.index_topk as u32,
                    last_length as u32,
                ],
            );
            enc.set_bytes(
                1,
                &[
                    c.index_dim as u32,
                    c.index_heads as u32,
                    (c.index_heads * c.index_dim) as u32,
                    c.index_heads as u32,
                ],
            );
            enc.set_bytes(2, &[select_width as u32, n as u32, 0, 0]);
            enc.set_tensor(3, &rows.index_query);
            enc.set_tensor(4, &rows.index_weights);
            enc.set_tensor(5, &self.owners[owner].keys);
            enc.set_tensor(6, &rows.select_scores);
            enc.set_bytes(7, &(first as u32));
            enc.dispatch(
                MTLSize {
                    width: last_length.div_ceil(8),
                    height: n,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            let allow_4k = last_length > 2048
                && last_length <= 4096
                && std::env::var_os("QWEN_V41_SELECT_FUSE_2048").is_none();
            let fused = (last_length <= 2048 || allow_4k)
                && c.index_topk <= 2048
                && std::env::var_os("QWEN_V41_SELECT_SPLIT_SORT").is_none();
            if fused {
                pipeline(
                    ctx,
                    enc,
                    if allow_4k {
                        "select_fused_rows_4k"
                    } else {
                        "select_fused_rows"
                    },
                )?;
                enc.set_bytes(
                    0,
                    &[
                        start as u32,
                        ratio as u32,
                        c.index_topk as u32,
                        select_width as u32,
                    ],
                );
                enc.set_bytes(1, &[first as u32, n as u32, 0, 0]);
                enc.set_tensor(2, &rows.select_scores);
                enc.set_tensor(3, &rows.selected);
                enc.set_tensor(4, &self.status);
                enc.dispatch(
                    MTLSize {
                        width: 1,
                        height: n,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
            } else {
                for row in first..first + n {
                    let length = (start + row + 1) / ratio;
                    if length <= c.index_topk {
                        continue;
                    }
                    let scores = rows.select_scores.view_subrange(
                        ((row - first) * select_width) as u64,
                        vec![select_width as u64],
                    );
                    let selected = std::mem::replace(
                        &mut self.selected,
                        prefill::row(&rows.selected, selected_stride, row),
                    );
                    let candidates = std::mem::replace(
                        &mut self.candidates,
                        prefill::row(&rows.candidates, candidates_stride, row),
                    );
                    let result = (|| -> Result<()> {
                        self.sort(ctx, enc, &scores, length, false)?;
                        let pso = pipeline(ctx, enc, "chronological")?;
                        enc.set_bytes(0, &(c.index_topk as u32));
                        enc.set_tensor(1, &self.sort_ids);
                        enc.set_tensor(2, &self.selected);
                        dispatch(enc, &pso, c.index_topk);
                        Ok(())
                    })();
                    self.selected = selected;
                    self.candidates = candidates;
                    result?;
                }
            }
        }
        Ok(())
    }

    /// Scores, softmax with sink, value accumulation and inverse RoPE per
    /// query tile.
    pub(in super::super) fn encode_rows_tiles(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        start: usize,
        rows: &Rows,
    ) -> Result<()> {
        let tile = tile_rows();
        for first in (0..rows.rows).step_by(tile) {
            self.sink_tile(
                ctx,
                enc,
                id,
                start,
                rows,
                first,
                (rows.rows - first).min(tile),
            )?;
        }
        Ok(())
    }

    pub(super) fn sink_tile(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        start: usize,
        rows: &Rows,
        first: usize,
        n: usize,
    ) -> Result<()> {
        let c = &self.config;
        op::require(
            n > 0 && n <= tile_rows() && first + n <= rows.rows,
            "query tile extent",
        )?;
        let position = start + first;
        let history = match rows.history.get() {
            Some(history) => history,
            None if self.ced.is_none() => ced::HistoryRange {
                layer: id,
                start,
                floor: start - start.min(c.window),
                extent: start.min(c.window) + rows.rows,
            },
            None => return Err("CED history was not prepared".into()),
        };
        history.validate(id, start, rows.rows, c.window)?;
        let count = |p: usize| (p + 1).min(c.window) + self.selected_count(id, p);
        // Simdgroup-MMA QK^T and PV for the released geometry (512-wide
        // latent, head count a multiple of 16); other geometries keep the
        // scalar/gathered-GEMM path below.
        let mma = mma_geometry(c);
        #[cfg(test)]
        if !mma && !ValueMatrix::shapes(c)?.is_empty() {
            op::require(
                rows.tile.matrix.is_some(),
                "F32 attention scratch must be planned and allocated inside its arithmetic scope",
            )?;
        }
        // Preserve the incumbent per-query PV branch. Counts are monotone, so
        // at most one fixed tile per layer/chunk needs this extra boundary split.
        if !mma
            && rows.tile.matrix.is_some()
            && count(position) < 256
            && count(position + n - 1) >= 256
        {
            let split = (1..n).find(|&q| count(position + q) >= 256).unwrap();
            self.sink_tile(ctx, enc, id, start, rows, first, split)?;
            return self.sink_tile(ctx, enc, id, start, rows, first + split, n - split);
        }
        let padded = count(position + n - 1).div_ceil(32) * 32;
        // Heads of one query are rows, not separate grouped projections.
        let matrix = rows.tile.matrix.as_ref().filter(|_| count(position) >= 256);
        let query = rows.query.view_subrange(
            (first * c.heads * c.head_dim) as u64,
            vec![(c.heads * c.head_dim) as u64, n as u64],
        );
        let selected = rows.selected.view_subrange(
            (first * c.index_topk) as u64,
            vec![c.index_topk as u64, n as u64],
        );
        let attended = rows
            .tile
            .attended
            .view_subrange(0, vec![(c.head_dim * n) as u64, c.heads as u64]);
        enc.set_bytes(
            0,
            &[
                c.head_dim as u32,
                c.window as u32,
                c.index_topk as u32,
                (c.window + c.index_topk + 1) as u32,
            ],
        );
        enc.set_bytes(
            1,
            &[
                position as u32,
                history.floor as u32,
                history.extent as u32,
                self.schedule.ratio(id) as u32,
            ],
        );
        enc.set_bytes(11, &[padded as u32, c.heads as u32, n as u32, 0]);
        enc.set_tensor(2, &query);
        enc.set_tensor(3, &rows.tile.history);
        enc.set_tensor(
            4,
            self.schedule
                .owner_slot(id)
                .map_or(&rows.latent, |s| &self.owners[s].compressed),
        );
        enc.set_tensor(5, &selected);
        enc.set_tensor(6, &self.constants[id].sink);
        enc.set_tensor(7, &rows.tile.scores);
        enc.set_tensor(8, &attended);
        enc.set_tensor(10, &self.status);
        let tensor = mma
            && n >= 128
            && metal::moe_grouped_tensor_selected(ctx)
            && std::env::var_os("QWEN_V41_SINK_TENSOR").is_none_or(|v| v != "0");
        // The wider-N score tile is the default tensor-tier form from 128
        // rows; `QWEN_V41_SINK_TENSOR32=0` restores the 16-key form.
        let score32 = tensor && std::env::var_os("QWEN_V41_SINK_TENSOR32").is_none_or(|v| v != "0");
        if tensor {
            tensor_pipeline(
                ctx,
                enc,
                if score32 {
                    "score_tensor32"
                } else {
                    "score_tensor"
                },
            )?;
            enc.dispatch(
                MTLSize {
                    width: c.heads / MMA_HEADS,
                    height: n,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        } else if mma {
            pipeline(ctx, enc, "score_mma")?;
            enc.dispatch(
                MTLSize {
                    width: c.heads / MMA_HEADS,
                    height: n,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            enc.set_threadgroup_memory(0, c.head_dim * std::mem::size_of::<f32>());
            pipeline(ctx, enc, "score_rows_lane8")?;
            enc.dispatch(
                MTLSize {
                    width: count(position + n - 1).div_ceil(32),
                    height: c.heads,
                    depth: n,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        pipeline(ctx, enc, "softmax_rows")?;
        enc.dispatch(
            MTLSize {
                width: c.heads,
                height: n,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        if mma {
            if tensor {
                tensor_pipeline(ctx, enc, "pv_tensor")?;
            } else {
                pipeline(ctx, enc, "pv_mma")?;
            }
            enc.dispatch(
                MTLSize {
                    width: c.heads / MMA_HEADS,
                    height: n,
                    depth: 1,
                },
                MTLSize {
                    width: if tensor { 128 } else { 256 },
                    height: 1,
                    depth: 1,
                },
            );
            let pso = pipeline(ctx, enc, "normalize_rows")?;
            dispatch(enc, &pso, n * c.heads * c.head_dim);
        } else if let Some(matrix) = matrix {
            pipeline(ctx, enc, "matrix_values_rows")?;
            enc.set_tensor(12, &matrix.values);
            enc.set_tensor(13, &matrix.probabilities);
            enc.dispatch(
                MTLSize {
                    width: padded / 16,
                    height: c.head_dim.max(c.heads).div_ceil(16),
                    depth: n,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            metal::encode_mat_mat_grouped_f32(
                ctx,
                enc,
                &matrix
                    .values
                    .view_subrange(0, vec![padded as u64, c.head_dim as u64, n as u64]),
                &matrix
                    .probabilities
                    .view_subrange(0, vec![(padded * n) as u64, c.heads as u64]),
                &attended,
                padded,
                c.head_dim,
                n,
                c.heads,
            )
            .map_err(|e| e.to_string())?;
            let pso = pipeline(ctx, enc, "finite")?;
            enc.set_bytes(0, &((n * c.heads * c.head_dim) as u32));
            enc.set_tensor(1, &attended);
            enc.set_tensor(2, &self.status);
            dispatch(enc, &pso, n * c.heads * c.head_dim);
        } else {
            let pso = pipeline(ctx, enc, "output_rows")?;
            dispatch(enc, &pso, n * c.heads * c.head_dim);
        }
        rope_rows(
            ctx,
            enc,
            &attended,
            &query,
            &self.constants[id].frequencies,
            c.head_dim,
            c.heads,
            c.rope_dim / 2,
            position,
            n,
            true,
        )?;
        Ok(())
    }
}
