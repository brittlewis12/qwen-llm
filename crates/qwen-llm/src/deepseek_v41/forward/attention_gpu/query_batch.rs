//! Bounded query tiles; compressor publication and discrete selection stay serial.
//! Raw SWA is layer-local in the scalar contract. Its old-window + chunk history
//! is consumed before this scratch is reused by the next layer. Shared compressed
//! KV/keys remain append-only, with each follower bounded by its query's length.
use super::*;

pub(super) const TILE: usize = 32;

pub(super) struct Scratch {
    pub(super) history: MetalTensor,
    pub(super) scores: MetalTensor,
    pub(super) attended: MetalTensor,
    pub(super) matrix: Option<ValueMatrix>,
}

impl Scratch {
    pub(super) fn shapes(c: &Config, rows: usize) -> Result<Vec<usize>> {
        op::require(
            (1..=prefill::DEFAULT_CHUNK_ROWS).contains(&rows),
            "query batch scratch rows",
        )?;
        let tile = rows.min(TILE);
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
        for size in ValueMatrix::shapes(c)? {
            sizes.push(checked_product(&[tile, size])?);
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

impl Workspace {
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
        op::require(!enc.is_concurrent(), "query batch requires serial encoder")?;
        op::require(
            self.batch.as_ref().is_some_and(|b| {
                b.start == start && b.rows == rows.rows && b.layer == id && b.row == 0
            }),
            "query batch layer order",
        )?;
        validate_weights(&self.config, id, w, &self.schedule)?;
        valid_view(input, checked_product(&[self.config.hidden, rows.rows])?)?;
        self.prepare_rows(ctx, enc, id, w, input, rows)
            .map_err(|e| e.to_string())?;
        let c = &self.config;
        let constants = &self.constants[id];
        let needs_index = self.schedule.produces_index(id) && {
            let length = (start + rows.rows) / self.schedule.ratio(id);
            length > c.index_topk || length.div_ceil(c.candidate_block) > c.candidate_topk
        };
        for first in (0..rows.rows).step_by(TILE) {
            let n = (rows.rows - first).min(TILE);
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
        let old = start.min(c.window);
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
        dispatch(enc, &pso, (old + rows.rows) * c.head_dim);
        // Separate dispatch: no thread may overwrite the ring while another
        // still gathers its old prefix. Each physical slot has one writer.
        let pso = pipeline(ctx, enc, "publish_raw")?;
        dispatch(enc, &pso, rows.rows.min(c.window) * c.head_dim);

        for row in 0..rows.rows {
            self.encode_batch_row(
                ctx,
                enc,
                id,
                start + row,
                w,
                &prefill::row(input, self.config.hidden, row),
                rows,
                row,
            )?;
        }
        for first in (0..rows.rows).step_by(TILE) {
            self.sink_tile(
                ctx,
                enc,
                id,
                start,
                rows,
                first,
                (rows.rows - first).min(TILE),
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
            n > 0 && n <= TILE && first + n <= rows.rows,
            "query tile extent",
        )?;
        let position = start + first;
        let count = |p: usize| (p + 1).min(c.window) + self.selected_count(id, p);
        // Preserve the incumbent per-query PV branch. Counts are monotone, so
        // at most one fixed tile per layer/chunk needs this extra boundary split.
        if rows.tile.matrix.is_some() && count(position) < 256 && count(position + n - 1) >= 256 {
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
                (start - start.min(c.window)) as u32,
                (start.min(c.window) + rows.rows) as u32,
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
        if std::env::var_os("QWEN_V41_SKIP_SCORE").is_none() {
            let requested = std::env::var("QWEN_V41_SCORE_HEADS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1);
            let heads_per_tg = if requested > 1 && c.heads % requested == 0 {
                requested
            } else {
                1
            };
            enc.set_threadgroup_memory(
                0,
                heads_per_tg * c.head_dim * std::mem::size_of::<f32>(),
            );
            if heads_per_tg > 1 {
                enc.set_bytes(14, &(heads_per_tg as u32));
            }
            pipeline(
                ctx,
                enc,
                if heads_per_tg > 1 {
                    "score_rows_heads"
                } else {
                    "score_rows"
                },
            )?;
            enc.dispatch(
                MTLSize {
                    width: count(position + n - 1).div_ceil(32),
                    height: c.heads / heads_per_tg,
                    depth: n,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        if std::env::var_os("QWEN_V41_SKIP_SOFTMAX").is_none() {
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
        }
        if let Some(matrix) = matrix {
            if std::env::var_os("QWEN_V41_SKIP_PV").is_none() {
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
            }
            let pso = pipeline(ctx, enc, "finite")?;
            enc.set_bytes(0, &((n * c.heads * c.head_dim) as u32));
            enc.set_tensor(1, &attended);
            enc.set_tensor(2, &self.status);
            dispatch(enc, &pso, n * c.heads * c.head_dim);
        } else if std::env::var_os("QWEN_V41_SKIP_PV").is_none() {
            let pso = pipeline(ctx, enc, "output_rows")?;
            dispatch(enc, &pso, n * c.heads * c.head_dim);
        }
        if std::env::var_os("QWEN_V41_SKIP_ROPE").is_none() {
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
        }
        Ok(())
    }
}
