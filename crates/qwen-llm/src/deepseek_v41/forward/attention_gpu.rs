//! Encode-only F32 attention. Workspace metadata is provisional: a failed encode
//! or command completion requires the caller to poison/discard the workspace.
use super::super::gguf::Schedule;
use super::prefill::RowsArithmetic;
use super::*;
use objc2_metal::{MTLComputePipelineState, MTLSize};

mod ced;
mod query_batch;
pub(super) use ced::{GlobalKvMode, PublicationRows, QueryControlView, QueryControls, SeedRows};
#[cfg(test)]
pub(super) use query_batch::with_f32_attention;
#[cfg(test)]
mod tests;

type Weights<'a> = op::AttentionWeights<Linear<'a>>;

struct Constants {
    query_norm: MetalTensor,
    kv_norm: MetalTensor,
    sink: MetalTensor,
    frequencies: MetalTensor,
    compressor_norm: Option<MetalTensor>,
    key_norm: Option<MetalTensor>,
}

struct OwnerCache {
    compressed: MetalTensor,
    keys: MetalTensor,
    pending_value: MetalTensor,
    pending_gate: MetalTensor,
}

pub(super) struct Workspace {
    config: Config,
    schedule: Schedule,
    capacity: usize,
    next_layer: usize,
    next_position: usize,
    batch: Option<BatchCursor>,
    ced: Option<ced::Cursor>,
    constants: Vec<Constants>,
    raw: MetalTensor,
    owners: Vec<OwnerCache>,
    query_rank: MetalTensor,
    query: MetalTensor,
    latent: MetalTensor,
    gate: MetalTensor,
    key: MetalTensor,
    index_query: MetalTensor,
    index_weights: MetalTensor,
    index_scores: MetalTensor,
    block_scores: MetalTensor,
    candidates: MetalTensor,
    sort_scores: MetalTensor,
    sort_ids: MetalTensor,
    selected: MetalTensor,
    attention_scores: MetalTensor,
    value_matrix: Option<ValueMatrix>,
    attended: MetalTensor,
    low: MetalTensor,
    output: MetalTensor,
    status: MetalTensor,
    /// Constant slot ids 0..groups and a READY status, so the eight
    /// output_a group GEMVs dispatch once through the native expert kernel.
    group_ids: MetalTensor,
    ready: MetalTensor,
}

struct BatchCursor {
    start: usize,
    rows: usize,
    layer: usize,
    row: usize,
}

struct ValueMatrix {
    values: MetalTensor,
    probabilities: MetalTensor,
}

impl ValueMatrix {
    fn shapes(c: &Config) -> Result<Vec<usize>> {
        if !c.head_dim.is_multiple_of(32) || c.heads < 8 {
            return Ok(Vec::new());
        }
        let padded = c
            .window
            .checked_add(c.index_topk)
            .and_then(|n| n.checked_add(31))
            .ok_or("attention value matrix extent")?
            / 32
            * 32;
        Ok(vec![
            checked_product(&[padded, c.head_dim])?,
            checked_product(&[padded, c.heads])?,
        ])
    }

    fn new(ctx: &MetalContext, c: &Config) -> Result<Option<Self>> {
        let sizes = Self::shapes(c)?;
        if sizes.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            values: tensor(ctx, sizes[0], GgmlType::F32)?,
            probabilities: tensor(ctx, sizes[1], GgmlType::F32)?,
        }))
    }

    fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        [heads, dim, count]: [usize; 3],
        output: &MetalTensor,
        status: &MetalTensor,
    ) -> Result<()> {
        let padded = count.div_ceil(32) * 32;
        pipeline(ctx, enc, "matrix_values")?;
        enc.set_bytes(11, &[padded as u32, heads as u32]);
        enc.set_tensor(12, &self.values);
        enc.set_tensor(13, &self.probabilities);
        enc.dispatch(
            MTLSize {
                width: padded / 16,
                height: dim.max(heads).div_ceil(16),
                depth: 1,
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
            &self
                .values
                .view_bytes(0, vec![padded as u64, dim as u64, 1]),
            &self
                .probabilities
                .view_bytes(0, vec![padded as u64, heads as u64]),
            &output.view_bytes(0, vec![dim as u64, heads as u64]),
            padded,
            dim,
            1,
            heads,
        )
        .map_err(|e| e.to_string())?;
        let pso = pipeline(ctx, enc, "finite")?;
        enc.set_bytes(0, &((heads * dim) as u32));
        enc.set_tensor(1, output);
        enc.set_tensor(2, status);
        dispatch(enc, &pso, heads * dim);
        Ok(())
    }
}

pub(super) struct Rows {
    tile: query_batch::Scratch,
    rank: MetalTensor,
    query: MetalTensor,
    latent: MetalTensor,
    low: MetalTensor,
    selected: MetalTensor,
    candidates: MetalTensor,
    grouped_input: MetalTensor,
    grouped_low: MetalTensor,
    compressor_value: MetalTensor,
    compressor_gate: MetalTensor,
    index_query: MetalTensor,
    index_weights: MetalTensor,
    /// One index-score row per selection tile row (capacity elements each).
    select_scores: MetalTensor,
    select_pairs: Option<MetalTensor>,
    /// Half copies of projection inputs for the tensor tile; `None` when
    /// the tier is not selected.
    half: Option<prefill::HalfScratch>,
    rows: usize,
    arithmetic: RowsArithmetic,
    controls: Option<QueryControlView>,
    history: std::cell::Cell<Option<ced::HistoryRange>>,
}

impl Rows {
    /// CED borrows the two retained I32 controls instead of allocating them.
    pub(super) fn widths_with_controls(c: &Config, capacity: usize) -> Vec<usize> {
        Self::widths(c, capacity)
            .into_iter()
            .enumerate()
            .filter_map(|(i, width)| (!matches!(i, 3 | 4)).then_some(width))
            .collect()
    }

    /// Grouped buffers are rewritten in full by finish_rows. A changed tile
    /// starts with no valid history or derived layout from its previous origin.
    pub(super) fn reset_grouped_layouts(&self) {
        self.history.set(None);
    }

    pub(super) fn widths(c: &Config, capacity: usize) -> Vec<usize> {
        vec![
            c.query_rank,
            c.heads * c.head_dim,
            c.head_dim,
            c.index_topk,
            capacity.div_ceil(c.candidate_block),
            c.groups * c.output_rank,
            c.heads * c.head_dim,
            c.groups * c.output_rank,
            c.head_dim,
            c.head_dim,
            c.index_heads * c.index_dim,
            c.index_heads,
        ]
    }

    /// Selection never reaches the pruning path above this many visible keys,
    /// so the tiled score scratch is bounded by the unpruned window.
    pub(super) fn select_width(c: &Config, capacity: usize) -> usize {
        capacity.min(c.candidate_topk.saturating_mul(c.candidate_block))
    }

    /// Two separately priced allocations: F32 scores and packed (score bits, ID)
    /// pairs. Round each up independently for short chunks and tile tails.
    pub(super) fn select_widths(c: &Config, capacity: usize, rows: usize) -> Result<Vec<usize>> {
        Ok(
            query_batch::selection_shapes(c, capacity, rows, query_batch::tile_rows())?
                .into_iter()
                .map(|elements| elements.div_ceil(rows))
                .collect(),
        )
    }

    pub(super) fn new(
        ctx: &MetalContext,
        c: &Config,
        capacity: usize,
        rows: usize,
    ) -> Result<Self> {
        Self::allocate(ctx, c, capacity, rows, None)
    }

    pub(super) fn with_controls(
        ctx: &MetalContext,
        c: &Config,
        capacity: usize,
        controls: &QueryControlView,
    ) -> Result<Self> {
        controls.validate_geometry(c, capacity)?;
        Self::allocate(ctx, c, capacity, controls.rows(), Some(controls.clone()))
    }

    fn allocate(
        ctx: &MetalContext,
        c: &Config,
        capacity: usize,
        rows: usize,
        controls: Option<QueryControlView>,
    ) -> Result<Self> {
        let half_elements = prefill::attention_half_scratch_elements(c, rows)?;
        let selection = query_batch::selection_shapes(c, capacity, rows, query_batch::tile_rows())?;
        let widths = Self::widths(c, capacity);
        for &width in &widths {
            checked_product(&[width, rows])?;
        }
        Ok(Self {
            tile: query_batch::Scratch::new(ctx, c, rows)?,
            rank: prefill::allocate(ctx, widths[0], rows, GgmlType::F32)?,
            query: prefill::allocate(ctx, widths[1], rows, GgmlType::F32)?,
            latent: prefill::allocate(ctx, widths[2], rows, GgmlType::F32)?,
            selected: match &controls {
                Some(view) => view.selected.clone(),
                None => prefill::allocate(ctx, widths[3], rows, GgmlType::I32)?,
            },
            candidates: match &controls {
                Some(view) => view.candidates.clone(),
                None => prefill::allocate(ctx, widths[4], rows, GgmlType::I32)?,
            },
            low: prefill::allocate(ctx, widths[5], rows, GgmlType::F32)?,
            grouped_input: prefill::allocate(ctx, widths[6], rows, GgmlType::F32)?,
            grouped_low: prefill::allocate(ctx, widths[7], rows, GgmlType::F32)?,
            compressor_value: prefill::allocate(ctx, widths[8], rows, GgmlType::F32)?,
            compressor_gate: prefill::allocate(ctx, widths[9], rows, GgmlType::F32)?,
            index_query: prefill::allocate(ctx, widths[10], rows, GgmlType::F32)?,
            index_weights: prefill::allocate(ctx, widths[11], rows, GgmlType::F32)?,
            select_scores: prefill::allocate(ctx, selection[0], 1, GgmlType::F32)?,
            select_pairs: if selection[1] == 0 {
                None
            } else {
                Some(prefill::allocate(ctx, selection[1], 1, GgmlType::I32)?)
            },
            half: prefill::HalfScratch::new(ctx, half_elements)?,
            rows,
            arithmetic: if controls.is_some() {
                RowsArithmetic::Packed
            } else {
                RowsArithmetic::Ordinary
            },
            controls,
            history: std::cell::Cell::new(None),
        })
    }

    /// Half copy of the first `width * rows` elements of `x` when the
    /// tensor tile would read it (the scratch exists and at least one of
    /// `consumers` takes the tensor route); `None` leaves every consumer
    /// on its F32 input.
    fn half_input(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        x: &MetalTensor,
        width: usize,
        consumers: &[&Linear<'_>],
    ) -> std::result::Result<Option<MetalTensor>, metal::MetalError> {
        let Some(half) = self.half.as_ref() else {
            return Ok(None);
        };
        if !consumers
            .iter()
            .any(|w| w.tensor_rows_selected_with_arithmetic(self.rows, self.arithmetic))
        {
            return Ok(None);
        }
        let elements = width
            .checked_mul(self.rows)
            .ok_or_else(|| metal::MetalError::BadShape {
                kernel: "v41_half_scratch",
                detail: "attention half input extent overflow".into(),
            })?;
        half.convert(ctx, enc, x, elements).map(Some)
    }
}

impl Workspace {
    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }
    /// Physical rings, completed owner rows, and pending pooling operands only.
    pub(super) fn causal_regions(&self, position: usize) -> Result<Vec<(&MetalTensor, usize)>> {
        op::require(self.ced.is_none(), "snapshot during active CED")?;
        self.causal_region_views(position)
    }

    fn causal_region_views(&self, position: usize) -> Result<Vec<(&MetalTensor, usize)>> {
        op::require(
            position <= self.capacity,
            "snapshot exceeds attention capacity",
        )?;
        let mut regions = vec![(
            &self.raw,
            checked_product(&[
                self.schedule.layers,
                self.config.window,
                self.config.head_dim,
            ])?,
        )];
        for (&layer, owner) in self.schedule.kv_source_layers.iter().zip(&self.owners) {
            let rows = position / self.schedule.ratio(layer);
            regions.extend([
                (
                    &owner.compressed,
                    checked_product(&[rows, self.config.head_dim])?,
                ),
                (
                    &owner.keys,
                    checked_product(&[rows, self.config.index_dim])?,
                ),
                (&owner.pending_value, self.config.head_dim),
                (&owner.pending_gate, self.config.head_dim),
            ]);
        }
        for &(tensor, count) in &regions {
            op::require(f32_slice(tensor)?.len() >= count, "snapshot region extent")?;
        }
        Ok(regions)
    }

    pub(super) fn validate_snapshot_position(&self, position: usize) -> Result<()> {
        op::require(
            self.ced.is_none()
                && self.batch.is_none()
                && self.next_layer == 0
                && self.next_position == position,
            "snapshot requires committed attention state",
        )
    }

    pub(super) fn validate_causal_restore(
        &self,
        position: usize,
        payload_len: usize,
    ) -> Result<()> {
        op::require(self.ced.is_none(), "restore during active CED")?;
        let regions = self.causal_regions(position)?;
        let size = regions
            .iter()
            .try_fold(0usize, |sum, (_, n)| sum.checked_add(*n))
            .ok_or("snapshot extent overflow")?;
        op::require(size == payload_len, "snapshot payload extent")?;
        op::require(
            self.status.dtype == GgmlType::I32
                && self.status.offset.is_multiple_of(4)
                && self.status.buffer.storageMode() == objc2_metal::MTLStorageMode::Shared
                && self
                    .status
                    .offset
                    .checked_add(4)
                    .is_some_and(|end| end <= self.status.buffer.length() as u64),
            "snapshot status extent",
        )?;
        Ok(())
    }

    pub(super) fn restore_causal(&mut self, position: usize, payload: &[f32]) -> Result<()> {
        self.validate_causal_restore(position, payload.len())?;
        let regions = self.causal_regions(position)?;
        let mut offset = 0;
        for (tensor, count) in regions {
            // All regions were validated before the first write. No GPU work is in flight.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    payload.as_ptr().add(offset),
                    tensor
                        .buffer
                        .contents()
                        .as_ptr()
                        .cast::<f32>()
                        .add((tensor.offset / 4) as usize),
                    count,
                );
            }
            offset += count;
        }
        unsafe {
            self.status
                .buffer
                .contents()
                .as_ptr()
                .cast::<i32>()
                .add((self.status.offset / 4) as usize)
                .write(0);
        }
        self.next_layer = 0;
        self.next_position = position;
        self.batch = None;
        Ok(())
    }
}

fn checked_product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .filter(|&n| n <= u32::MAX as usize / 4)
            .ok_or_else(|| "GPU attention extent overflow".into())
    })
}

fn validate(c: &Config, capacity: usize, schedule: &Schedule) -> Result<()> {
    schedule.validate().map_err(|e| e.to_string())?;
    op::require(
        schedule
            .candidate_source_layer
            .is_none_or(|source| (source..schedule.layers).all(|id| schedule.ratio(id) == 1)),
        "candidate mask requires ratio-one coordinates after its source",
    )?;
    op::require(
        capacity > 0 && capacity <= u32::MAX as usize / 8,
        "GPU attention capacity",
    )?;
    op::require(
        [
            c.hidden,
            c.heads,
            c.head_dim,
            c.query_rank,
            c.output_rank,
            c.groups,
            c.index_heads,
            c.index_dim,
            c.index_topk,
            c.window,
            c.candidate_block,
            c.candidate_topk,
        ]
        .iter()
        .all(|&x| x > 0 && x <= u32::MAX as usize / 4)
            && c.heads.is_multiple_of(c.groups)
            && c.rope_dim.is_multiple_of(2)
            && c.rope_dim <= c.head_dim
            && c.rope_dim <= c.index_dim
            && c.norm_eps.is_finite()
            && c.norm_eps > 0.0,
        "GPU attention geometry",
    )?;
    for d in [
        [schedule.layers, c.window, c.head_dim],
        [capacity, c.head_dim, 1],
        [capacity, c.index_dim, 1],
        [c.heads, c.head_dim, 1],
        [c.index_heads, c.index_dim, 1],
        [c.groups, c.output_rank, 1],
        [
            c.heads,
            c.window
                .checked_add(c.index_topk)
                .and_then(|n| n.checked_add(1))
                .ok_or("attention overflow")?,
            1,
        ],
    ] {
        checked_product(&d)?;
    }
    Ok(())
}

fn shapes(c: &Config, capacity: usize, schedule: &Schedule) -> Result<Vec<usize>> {
    validate(c, capacity, schedule)?;
    let mut sizes = vec![
        schedule.layers * c.window * c.head_dim,
        c.query_rank,
        c.heads * c.head_dim,
        c.head_dim,
        c.head_dim,
        c.index_dim,
        c.index_heads * c.index_dim,
        c.index_heads,
        capacity,
        capacity.div_ceil(c.candidate_block),
        capacity.div_ceil(c.candidate_block),
        capacity.next_power_of_two(),
        capacity.next_power_of_two(),
        c.index_topk,
        c.heads * (c.window + c.index_topk + 1),
        c.heads * c.head_dim,
        c.groups * c.output_rank,
        c.hidden,
        1,
        c.groups,
        1,
    ];
    for &owner in &schedule.kv_source_layers {
        let ratio = schedule.ratio(owner);
        sizes.extend([
            capacity.div_ceil(ratio) * c.head_dim,
            capacity.div_ceil(ratio) * c.index_dim,
            c.head_dim,
            c.head_dim,
        ]);
    }
    sizes.extend(ValueMatrix::shapes(c)?);
    for _ in 0..schedule.layers {
        sizes.extend([
            c.query_rank,
            c.head_dim,
            c.heads,
            (c.rope_dim / 2).max(1),
            c.head_dim,
            c.index_dim,
        ]);
    }
    Ok(sizes)
}

#[cfg(test)]
pub(super) fn bytes(c: &Config, capacity: usize) -> Result<u64> {
    bytes_with_schedule(c, capacity, &Schedule::default())
}

pub(super) fn bytes_with_schedule(c: &Config, capacity: usize, schedule: &Schedule) -> Result<u64> {
    shapes(c, capacity, schedule)?
        .into_iter()
        .try_fold(0u64, |sum, n| {
            sum.checked_add(n as u64 * 4 + 17408)
                .ok_or_else(|| "GPU attention memory overflow".into())
        })
}

fn tensor(ctx: &MetalContext, n: usize, dtype: GgmlType) -> Result<MetalTensor> {
    MetalTensor::zeros_dtype_unstaged(ctx, vec![n as u64], dtype).map_err(|e| e.to_string())
}
fn pipeline(ctx: &MetalContext, enc: &KernelEncoder, suffix: &str) -> Result<metal::Pipeline> {
    let pso = ctx
        .pipeline(&format!("kernel_deepseek_v41_attention_{suffix}"))
        .map_err(|e| e.to_string())?;
    enc.set_pipeline(&pso);
    Ok(pso)
}
/// Same names in the Metal 4 tensor library
/// (`tensor/deepseek_v41_attention_tensor.metal`).
fn tensor_pipeline(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    suffix: &str,
) -> Result<metal::Pipeline> {
    let pso = ctx
        .tensor_pipeline(&format!("kernel_deepseek_v41_attention_{suffix}"))
        .map_err(|e| e.to_string())?;
    enc.set_pipeline(&pso);
    Ok(pso)
}
fn dispatch(enc: &KernelEncoder, pso: &metal::Pipeline, n: usize) {
    if n > 0 {
        // These kernels use only global IDs, with no collective operations.
        // Keep the grid exact: RoPE has no padded-thread bounds check.
        let limit = pso.maxTotalThreadsPerThreadgroup().min(128);
        let width = 1usize << n.trailing_zeros().min(limit.ilog2());
        enc.dispatch(
            MTLSize {
                width: n / width,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
}

fn index_dispatch(enc: &KernelEncoder, length: usize) {
    enc.dispatch(
        MTLSize {
            width: length.div_ceil(8),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

// Bindings 0..10 match the incumbent. Each head's last score slot is scratch.
fn sink_passes(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    heads: usize,
    dim: usize,
    count: usize,
    matrix: Option<(&ValueMatrix, &MetalTensor, &MetalTensor)>,
) -> Result<()> {
    if count > 0 {
        pipeline(ctx, enc, "score")?;
        enc.dispatch(
            MTLSize {
                width: count.div_ceil(8),
                height: heads,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }
    let output_count = heads * dim;
    let output_width = 1usize << output_count.trailing_zeros().min(5);
    for (name, groups, width) in [
        ("softmax", heads, 32),
        ("output", output_count / output_width, output_width),
    ] {
        if groups == 0 {
            continue;
        }
        if name == "output"
            && count >= 256
            && let Some((matrix, output, status)) = matrix
        {
            matrix.encode(ctx, enc, [heads, dim, count], output, status)?;
            continue;
        }
        pipeline(ctx, enc, name)?;
        // Exact grids, including tiny toy geometries, need no padded-thread guard.
        enc.dispatch(
            MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(())
}
fn linear(
    _ctx: &MetalContext,
    enc: &KernelEncoder,
    w: &Linear<'_>,
    x: &MetalTensor,
    y: &MetalTensor,
) -> Result<()> {
    w.encode(enc, x, y).map_err(|e| e.to_string())
}
fn norm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    w: &MetalTensor,
    n: usize,
    eps: f32,
) -> Result<()> {
    valid_view(x, n)?;
    metal::encode_rms_norm_mul_f32(ctx, enc, x, w, x, eps).map_err(|e| e.to_string())
}
fn rope(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    freq: &MetalTensor,
    dim: usize,
    heads: usize,
    count: usize,
    position: usize,
    inverse: bool,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    let pso = pipeline(ctx, enc, "rope")?;
    enc.set_bytes(
        0,
        &[dim as u32, count as u32, position as u32, inverse as u32],
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, freq);
    dispatch(enc, &pso, heads * count);
    Ok(())
}
fn copy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
    n: usize,
) -> Result<()> {
    let pso = pipeline(ctx, enc, "copy")?;
    enc.set_bytes(0, &(n as u32));
    enc.set_tensor(1, x);
    enc.set_tensor(2, y);
    dispatch(enc, &pso, n);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compress_publish(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    key: &MetalTensor,
    key_norm: &MetalTensor,
    latent: &MetalTensor,
    freq: &MetalTensor,
    out_key: &MetalTensor,
    out_latent: &MetalTensor,
    index_dim: usize,
    head_dim: usize,
    pairs: usize,
    position: usize,
    eps: f32,
) -> Result<()> {
    pipeline(ctx, enc, "compress_publish")?;
    enc.set_bytes(
        0,
        &[
            index_dim as u32,
            head_dim as u32,
            pairs as u32,
            position as u32,
        ],
    );
    enc.set_bytes(1, &eps);
    enc.set_tensor(2, key);
    enc.set_tensor(3, key_norm);
    enc.set_tensor(4, latent);
    enc.set_tensor(5, freq);
    enc.set_tensor(6, out_key);
    enc.set_tensor(7, out_latent);
    enc.set_threadgroup_memory(0, 136 * std::mem::size_of::<f32>());
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
    Ok(())
}

fn projection(w: &Linear<'_>, input: usize, output: usize) -> Result<()> {
    let t = &w.tensor;
    checked_product(&[input, output])?;
    let alignment = match t.dtype {
        GgmlType::F32 => 4,
        GgmlType::F16 | GgmlType::BF16 | GgmlType::IQ1_S | GgmlType::IQ1_M | GgmlType::IQ2_XXS => 2,
        _ => 16,
    };
    let end = row_bytes(t)?
        .checked_mul(output as u64)
        .and_then(|bytes| bytes.checked_add(t.offset));
    op::require(
        w.input == input
            && w.output == output
            && t.shape == [input as u64, output as u64]
            && dense_linear_supported(t.dtype)
            && t.offset.is_multiple_of(alignment)
            && end.is_some_and(|end| end <= t.buffer.length() as u64),
        "GPU attention projection geometry/storage bounds",
    )
}

fn valid_view(t: &MetalTensor, count: usize) -> Result<()> {
    let elements = t.shape.iter().try_fold(1u64, |n, &d| n.checked_mul(d));
    let end = (count as u64)
        .checked_mul(4)
        .and_then(|n| n.checked_add(t.offset));
    op::require(
        t.dtype == GgmlType::F32
            && elements == Some(count as u64)
            && t.offset.is_multiple_of(4)
            && end.is_some_and(|end| end <= t.buffer.length() as u64),
        "GPU attention buffer bounds/dtype",
    )
}

impl Workspace {
    fn selected_count(&self, id: usize, position: usize) -> usize {
        self.schedule.owner(id).map_or(0, |_| {
            ((position + 1) / self.schedule.ratio(id)).min(self.config.index_topk)
        })
    }

    pub(super) fn capture(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        snapshot: &capture::Snapshot,
        input: &MetalTensor,
        id: usize,
        position: usize,
    ) -> Result<()> {
        let count = self.selected_count(id, position);
        snapshot.encode(
            ctx,
            enc,
            input,
            &self.output,
            &self.selected,
            &self.candidates,
            count,
        )
    }

    pub(super) fn capture_row(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        snapshot: &capture::Snapshot,
        input: &MetalTensor,
        output: &MetalTensor,
        rows: &Rows,
        id: usize,
        position: usize,
        row: usize,
    ) -> Result<()> {
        snapshot.encode(
            ctx,
            enc,
            input,
            output,
            &prefill::row(&rows.selected, self.config.index_topk, row),
            &prefill::row(
                &rows.candidates,
                self.capacity.div_ceil(self.config.candidate_block),
                row,
            ),
            self.selected_count(id, position),
        )
    }

    #[cfg(test)]
    pub(super) fn new(
        ctx: &MetalContext,
        c: &Config,
        capacity: usize,
        weights: &[&Weights<'_>],
    ) -> std::result::Result<Self, StateError> {
        Self::new_with_schedule(ctx, c, capacity, weights, &Schedule::default())
    }

    pub(super) fn new_with_schedule(
        ctx: &MetalContext,
        c: &Config,
        capacity: usize,
        weights: &[&Weights<'_>],
        schedule: &Schedule,
    ) -> std::result::Result<Self, StateError> {
        let size = bytes_with_schedule(c, capacity, schedule)?;
        op::require(
            weights.len() == schedule.layers,
            "GPU attention layer count",
        )?;
        for n in shapes(c, capacity, schedule)? {
            let priced = ctx
                .price_shared_buffer_upper(n as u64 * 4)
                .map_err(|e| e.to_string())?;
            op::require(
                priced.priced_upper_bytes <= n as u64 * 4 + 16384,
                "GPU attention allocation page slack",
            )?;
        }
        let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
            size,
            0,
            0,
            ctx.memory_signals(),
            true,
        );
        if let Some(reason) = admission.refusal() {
            return Err(StateError::MemoryAdmissionDenied(reason));
        }
        let constants = weights
            .iter()
            .enumerate()
            .map(|(id, w)| {
                validate_weights(c, id, w, schedule)?;
                let up = |x: &[f32]| upload(ctx, x, vec![x.len().max(1) as u64]);
                Ok(Constants {
                    query_norm: up(&w.query_norm)?,
                    kv_norm: up(&w.kv_norm)?,
                    sink: up(&w.sink)?,
                    frequencies: if w.frequencies.is_empty() {
                        up(&[0.0])?
                    } else {
                        up(&w.frequencies)?
                    },
                    compressor_norm: w.compressor.as_ref().map(|w| up(&w.norm)).transpose()?,
                    key_norm: w
                        .index
                        .as_ref()
                        .filter(|w| w.key.is_some())
                        .map(|w| up(&w.key_norm))
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let f = |n| tensor(ctx, n, GgmlType::F32);
        let i = |n| tensor(ctx, n, GgmlType::I32);
        let owners = schedule
            .kv_source_layers
            .iter()
            .map(|&owner| {
                let ratio = schedule.ratio(owner);
                Ok(OwnerCache {
                    compressed: f(capacity.div_ceil(ratio) * c.head_dim)?,
                    keys: f(capacity.div_ceil(ratio) * c.index_dim)?,
                    pending_value: f(c.head_dim)?,
                    pending_gate: f(c.head_dim)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config: c.clone(),
            schedule: schedule.clone(),
            capacity,
            next_layer: 0,
            next_position: 0,
            batch: None,
            ced: None,
            constants,
            raw: f(schedule.layers * c.window * c.head_dim)?,
            owners,
            query_rank: f(c.query_rank)?,
            query: f(c.heads * c.head_dim)?,
            latent: f(c.head_dim)?,
            gate: f(c.head_dim)?,
            key: f(c.index_dim)?,
            index_query: f(c.index_heads * c.index_dim)?,
            index_weights: f(c.index_heads)?,
            index_scores: f(capacity)?,
            block_scores: f(capacity.div_ceil(c.candidate_block))?,
            candidates: i(capacity.div_ceil(c.candidate_block))?,
            sort_scores: f(capacity.next_power_of_two())?,
            sort_ids: i(capacity.next_power_of_two())?,
            selected: i(c.index_topk)?,
            attention_scores: f(c.heads * (c.window + c.index_topk + 1))?,
            value_matrix: ValueMatrix::new(ctx, c)?,
            attended: f(c.heads * c.head_dim)?,
            low: f(c.groups * c.output_rank)?,
            output: f(c.hidden)?,
            status: i(1)?,
            group_ids: {
                let ids = i(c.groups)?;
                let p = ids.buffer.contents().as_ptr() as *mut i32;
                for g in 0..c.groups {
                    unsafe { p.add(g).write(g as i32) };
                }
                ids
            },
            ready: {
                let ready = i(1)?;
                unsafe {
                    (ready.buffer.contents().as_ptr() as *mut i32).write(metal::ROUTE_STATUS_READY)
                };
                ready
            },
        })
    }

    /// The eight output_a group matrices as one contiguous [group_input,
    /// output_rank, groups] bank, when their views are adjacent slices of one
    /// resident tensor (how `bind` creates them).
    fn output_a_bank(w: &Weights<'_>, c: &Config) -> Option<MetalTensor> {
        let first = &w.output_a.first()?.tensor;
        let group_input = c.heads / c.groups * c.head_dim;
        let (block, block_bytes) = first.dtype.storage_layout()?;
        if !(group_input as u64).is_multiple_of(block) || w.output_a.len() != c.groups {
            return None;
        }
        let row_bytes = group_input as u64 / block * block_bytes;
        let group_bytes = row_bytes * c.output_rank as u64;
        for (g, linear) in w.output_a.iter().enumerate() {
            let t = &linear.tensor;
            if !std::ptr::eq(&*t.buffer, &*first.buffer)
                || t.offset != first.offset + g as u64 * group_bytes
                || t.dtype != first.dtype
                || linear.input != group_input
                || linear.output != c.output_rank
            {
                return None;
            }
        }
        if first.offset + group_bytes * c.groups as u64 > first.buffer.length() as u64 {
            return None;
        }
        Some(MetalTensor {
            buffer: first.buffer.clone(),
            offset: first.offset,
            shape: vec![group_input as u64, c.output_rank as u64, c.groups as u64],
            dtype: first.dtype,
            provenance: first.provenance(),
        })
    }

    fn select_unpruned(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        length: usize,
    ) -> Result<()> {
        let c = &self.config;
        let blocks = length.div_ceil(c.candidate_block);
        let publish_mask = self.schedule.candidate_source_layer == Some(id);
        let all = if length <= c.index_topk { length } else { 0 };
        if all == 0 && !publish_mask {
            return Ok(());
        }
        let pso = pipeline(ctx, enc, "selection_prepare")?;
        enc.set_bytes(
            0,
            &[
                length as u32,
                blocks as u32,
                publish_mask as u32,
                all as u32,
            ],
        );
        enc.set_tensor(1, &self.selected);
        enc.set_tensor(2, &self.candidates);
        dispatch(enc, &pso, all.max(if publish_mask { blocks } else { 0 }));
        Ok(())
    }

    fn sort(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        scores: &MetalTensor,
        n: usize,
        mask: bool,
    ) -> Result<()> {
        // The shared radix selector rejects nonfinite scores. Here -Inf masked
        // rows still participate in top-k, and forced-last blocks use +Inf.
        let padded = n.max(1).next_power_of_two();
        let pso = pipeline(ctx, enc, "sort_init")?;
        enc.set_bytes(
            0,
            &[
                n as u32,
                padded as u32,
                mask as u32,
                self.config.candidate_block as u32,
            ],
        );
        enc.set_tensor(1, scores);
        enc.set_tensor(2, &self.candidates);
        enc.set_tensor(3, &self.sort_scores);
        enc.set_tensor(4, &self.sort_ids);
        enc.set_tensor(5, &self.status);
        dispatch(enc, &pso, padded);
        let tile = padded.min(1024);
        let sort_tile = |width, full| -> Result<()> {
            pipeline(ctx, enc, "sort_tile")?;
            enc.set_bytes(0, &[padded as u32, tile as u32, width as u32, full as u32]);
            enc.set_tensor(1, &self.sort_scores);
            enc.set_tensor(2, &self.sort_ids);
            enc.set_threadgroup_memory(0, tile * 4);
            enc.set_threadgroup_memory(1, tile * 4);
            enc.dispatch(
                MTLSize {
                    width: padded / tile,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        };
        sort_tile(tile, true)?;
        let mut width = tile * 2;
        while width <= padded {
            let mut stride = width / 2;
            while stride >= tile {
                let pso = pipeline(ctx, enc, "sort_step")?;
                enc.set_bytes(0, &[padded as u32, stride as u32, width as u32, 0]);
                enc.set_tensor(1, &self.sort_scores);
                enc.set_tensor(2, &self.sort_ids);
                dispatch(enc, &pso, padded);
                stride /= 2;
            }
            sort_tile(width, false)?;
            width *= 2;
        }
        Ok(())
    }

    /// Requires a serial encoder and strict token-major schedule order.
    /// Weights and their constants must be the immutable weights passed to new.
    /// The returned tensor aliases scratch and must be consumed before reuse.
    /// No submission, wait, host activation access, or HC occurs here.
    pub(super) fn encode(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
    ) -> Result<&MetalTensor> {
        op::require(self.batch.is_none(), "decode during attention batch")?;
        if self.ced.is_some() {
            return self.encode_ced_scalar(ctx, enc, id, position, w, input);
        }
        self.encode_inner(
            ctx,
            enc,
            id,
            position,
            w,
            input,
            None,
            true,
            GlobalKvMode::Ordinary,
        )
    }

    pub(super) fn begin_batch(&mut self, start: usize, rows: usize) -> Result<()> {
        self.validate_snapshot_position(start)?;
        op::require(
            rows > 0 && rows <= self.capacity.saturating_sub(start),
            "attention batch extent",
        )?;
        self.batch = Some(BatchCursor {
            start,
            rows,
            layer: 0,
            row: 0,
        });
        Ok(())
    }

    fn prepare_rows_mode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
        rows: &Rows,
        mode: GlobalKvMode,
    ) -> std::result::Result<(), metal::MetalError> {
        let c = &self.config;
        let index = if self.schedule.produces_index(id) {
            let needs_index = self.active_batch().is_none_or(|batch| {
                let length = (batch.start + batch.rows) / self.schedule.ratio(id);
                length > c.index_topk || length.div_ceil(c.candidate_block) > c.candidate_topk
            });
            w.index.as_ref().filter(|_| needs_index)
        } else {
            None
        };
        let compressor = w
            .compressor
            .as_ref()
            .filter(|_| mode == GlobalKvMode::Ordinary);
        // Every projection of the layer input first (one half copy serves
        // them all), then the rank-fed ones after the query norm. The
        // order is free: the rank consumers depend only on query_a.
        let mut input_consumers = vec![&w.query_a, &w.kv];
        if let Some(compressor) = compressor {
            input_consumers.push(&compressor.kv);
            input_consumers.extend(compressor.gate.as_ref());
        }
        input_consumers.extend(index.map(|index| &index.weights));
        let input_half = rows.half_input(ctx, enc, input, c.hidden, &input_consumers)?;
        let input_half = input_half.as_ref();
        w.query_a.encode_rows_with_arithmetic_and_half(
            enc,
            input,
            input_half,
            &rows.rank,
            rows.rows,
            rows.arithmetic,
        )?;
        w.kv.encode_rows_with_arithmetic_and_half(
            enc,
            input,
            input_half,
            &rows.latent,
            rows.rows,
            rows.arithmetic,
        )?;
        if let Some(compressor) = compressor {
            compressor.kv.encode_rows_with_arithmetic_and_half(
                enc,
                input,
                input_half,
                &rows.compressor_value,
                rows.rows,
                rows.arithmetic,
            )?;
            if let Some(gate) = &compressor.gate {
                gate.encode_rows_with_arithmetic_and_half(
                    enc,
                    input,
                    input_half,
                    &rows.compressor_gate,
                    rows.rows,
                    rows.arithmetic,
                )?;
            }
        }
        if let Some(index) = index {
            index.weights.encode_rows_with_arithmetic_and_half(
                enc,
                input,
                input_half,
                &rows.index_weights,
                rows.rows,
                rows.arithmetic,
            )?;
        }
        metal::encode_rms_norm_mul_rows_f32(
            ctx,
            enc,
            &rows.rank,
            &self.constants[id].query_norm,
            &rows.rank,
            rows.rows,
            c.query_rank,
            w.eps,
        )?;
        metal::encode_rms_norm_mul_rows_f32(
            ctx,
            enc,
            &rows.latent,
            &self.constants[id].kv_norm,
            &rows.latent,
            rows.rows,
            c.head_dim,
            w.eps,
        )?;
        let mut rank_consumers = vec![&w.query_b];
        rank_consumers.extend(index.map(|index| &index.query));
        let rank_half = rows.half_input(ctx, enc, &rows.rank, c.query_rank, &rank_consumers)?;
        let rank_half = rank_half.as_ref();
        w.query_b.encode_rows_with_arithmetic_and_half(
            enc,
            &rows.rank,
            rank_half,
            &rows.query,
            rows.rows,
            rows.arithmetic,
        )?;
        if let Some(index) = index {
            index.query.encode_rows_with_arithmetic_and_half(
                enc,
                &rows.rank,
                rank_half,
                &rows.index_query,
                rows.rows,
                rows.arithmetic,
            )?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn encode_batch_row(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
        rows: &Rows,
        row: usize,
        select: bool,
    ) -> Result<()> {
        let mode = self.global_mode(id, position, rows.rows)?;
        self.encode_batch_row_mode(ctx, enc, id, position, w, input, rows, row, select, mode)
    }

    fn encode_batch_row_mode(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
        rows: &Rows,
        row: usize,
        select: bool,
        mode: GlobalKvMode,
    ) -> Result<()> {
        op::require(
            self.active_batch().is_some_and(|b| {
                b.rows == rows.rows && b.row == row && b.layer == id && b.start + row == position
            }),
            "attention batch row order",
        )?;
        let selected = std::mem::replace(
            &mut self.selected,
            prefill::row(&rows.selected, self.config.index_topk, row),
        );
        let candidates = std::mem::replace(
            &mut self.candidates,
            prefill::row(
                &rows.candidates,
                self.capacity.div_ceil(self.config.candidate_block),
                row,
            ),
        );
        // Sequential compressor/index transitions operate directly on prepared
        // rows. Their mutations are scratch-only; decode retains its own buffers.
        let latent = std::mem::replace(
            &mut self.latent,
            prefill::row(&rows.compressor_value, self.config.head_dim, row),
        );
        let gate = std::mem::replace(
            &mut self.gate,
            prefill::row(&rows.compressor_gate, self.config.head_dim, row),
        );
        let index_query = std::mem::replace(
            &mut self.index_query,
            prefill::row(
                &rows.index_query,
                self.config.index_heads * self.config.index_dim,
                row,
            ),
        );
        let index_weights = std::mem::replace(
            &mut self.index_weights,
            prefill::row(&rows.index_weights, self.config.index_heads, row),
        );
        let result = self
            .encode_inner(
                ctx,
                enc,
                id,
                position,
                w,
                input,
                Some((rows, row)),
                select,
                mode,
            )
            .map(|_| ());
        self.selected = selected;
        self.candidates = candidates;
        self.latent = latent;
        self.gate = gate;
        self.index_query = index_query;
        self.index_weights = index_weights;
        result
    }

    pub(super) fn finish_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        w: &Weights<'_>,
        rows: &Rows,
        output: &MetalTensor,
    ) -> Result<()> {
        let c = &self.config;
        let group_width = c.heads * c.head_dim / c.groups;
        group_rows(
            ctx,
            enc,
            &rows.query,
            &rows.grouped_input,
            group_width,
            c.groups,
            rows.rows,
        )?;
        // One half copy of the whole grouped input serves every group's
        // output_a; the per-group views are the same offsets in halves.
        let output_a: Vec<&Linear<'_>> = w.output_a.iter().collect();
        let grouped_half = rows
            .half_input(
                ctx,
                enc,
                &rows.grouped_input,
                group_width * c.groups,
                &output_a,
            )
            .map_err(|e| e.to_string())?;
        for (group, projection) in w.output_a.iter().enumerate() {
            let offset = (group * group_width * rows.rows) as u64;
            let shape = vec![group_width as u64, rows.rows as u64];
            let group_half = grouped_half
                .as_ref()
                .map(|h| h.view_subrange(offset, shape.clone()));
            projection
                .encode_rows_with_arithmetic_and_half(
                    enc,
                    &rows.grouped_input.view_subrange(offset, shape),
                    group_half.as_ref(),
                    &rows.grouped_low.view_subrange(
                        (group * c.output_rank * rows.rows) as u64,
                        vec![c.output_rank as u64, rows.rows as u64],
                    ),
                    rows.rows,
                    rows.arithmetic,
                )
                .map_err(|e| e.to_string())?;
        }
        group_rows(
            ctx,
            enc,
            &rows.grouped_low,
            &rows.low,
            c.output_rank,
            rows.rows,
            c.groups,
        )?;
        let low_half = rows
            .half_input(
                ctx,
                enc,
                &rows.low,
                c.groups * c.output_rank,
                &[&w.output_b],
            )
            .map_err(|e| e.to_string())?;
        w.output_b
            .encode_rows_with_arithmetic_and_half(
                enc,
                &rows.low,
                low_half.as_ref(),
                output,
                rows.rows,
                rows.arithmetic,
            )
            .map_err(|e| e.to_string())?;
        let pso = pipeline(ctx, enc, "finite")?;
        enc.set_bytes(0, &((self.config.hidden * rows.rows) as u32));
        enc.set_tensor(1, output);
        enc.set_tensor(2, &self.status);
        dispatch(enc, &pso, self.config.hidden * rows.rows);
        self.finish_ced_rows(rows)?;
        Ok(())
    }

    fn encode_inner(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
        prepared: Option<(&Rows, usize)>,
        select: bool,
        mode: GlobalKvMode,
    ) -> Result<&MetalTensor> {
        op::require(
            !enc.is_concurrent(),
            "GPU attention requires serial encoder",
        )?;
        let c = &self.config;
        op::require(
            self.active_batch().map_or(
                id == self.next_layer && position == self.next_position,
                |b| id == b.layer && position == b.start + b.row,
            ) && position < self.capacity,
            "GPU attention order/capacity",
        )?;
        self.validate_global_mode(id, position, mode)?;
        validate_weights(c, id, w, &self.schedule)?;
        valid_view(input, c.hidden)?;
        let constants = &self.constants[id];
        if prepared.is_none() {
            linear(ctx, enc, &w.query_a, input, &self.query_rank)?;
            norm(
                ctx,
                enc,
                &self.query_rank,
                &constants.query_norm,
                c.query_rank,
                w.eps,
            )?;
            linear(ctx, enc, &w.query_b, &self.query_rank, &self.query)?;
            linear(ctx, enc, &w.kv, input, &self.latent)?;
            norm(
                ctx,
                enc,
                &self.latent,
                &constants.kv_norm,
                c.head_dim,
                w.eps,
            )?;
            rope(
                ctx,
                enc,
                &self.query,
                &constants.frequencies,
                c.head_dim,
                c.heads,
                c.rope_dim / 2,
                position,
                false,
            )?;
            rope(
                ctx,
                enc,
                &self.latent,
                &constants.frequencies,
                c.head_dim,
                1,
                c.rope_dim / 2,
                position,
                false,
            )?;
            let raw = self.raw.view_subrange(
                (id * c.window * c.head_dim) as u64,
                vec![(c.window * c.head_dim) as u64],
            );
            let row = raw.view_subrange(
                (position % c.window * c.head_dim) as u64,
                vec![c.head_dim as u64],
            );
            copy(ctx, enc, &self.latent, &row, c.head_dim)?;
        }
        let owner = self.schedule.owner_slot(id);
        let mut selected_count = 0;
        if let Some(owner) = owner {
            let cache = &self.owners[owner];
            let ratio = self.schedule.ratio(id);
            let length = (position + 1) / ratio;
            if self.schedule.owns_kv(id) && mode == GlobalKvMode::Ordinary {
                self.publish_owner_transition(
                    ctx,
                    enc,
                    id,
                    position,
                    w,
                    input,
                    &self.latent,
                    &self.gate,
                    prepared.is_some(),
                )?;
            }
            selected_count = length.min(c.index_topk);
            if select && self.schedule.produces_index(id) && length > 0 {
                let unpruned = length.div_ceil(c.candidate_block) <= c.candidate_topk;
                if unpruned && length <= c.index_topk {
                    self.select_unpruned(ctx, enc, id, length)?;
                } else {
                    let index = w.index.as_ref().ok_or("missing index")?;
                    if prepared.is_none() {
                        linear(ctx, enc, &index.query, &self.query_rank, &self.index_query)?;
                        linear(ctx, enc, &index.weights, input, &self.index_weights)?;
                        rope(
                            ctx,
                            enc,
                            &self.index_query,
                            &constants.frequencies,
                            c.index_dim,
                            c.index_heads,
                            c.rope_dim / 2,
                            position,
                            false,
                        )?;
                    }
                    pipeline(ctx, enc, "index_heads")?;
                    enc.set_bytes(
                        0,
                        &[length as u32, c.index_dim as u32, c.index_heads as u32, 0],
                    );
                    enc.set_tensor(1, &self.index_query);
                    enc.set_tensor(2, &self.index_weights);
                    enc.set_tensor(3, &cache.keys);
                    enc.set_tensor(4, &self.index_scores);
                    index_dispatch(enc, length);
                    if !unpruned && self.schedule.candidate_source_layer == Some(id) {
                        let blocks = length.div_ceil(c.candidate_block);
                        let pso = pipeline(ctx, enc, "blocks")?;
                        enc.set_bytes(0, &[length as u32, c.candidate_block as u32]);
                        enc.set_tensor(1, &self.index_scores);
                        enc.set_tensor(2, &self.block_scores);
                        dispatch(enc, &pso, blocks);
                        self.sort(ctx, enc, &self.block_scores, blocks, false)?;
                        let pso = pipeline(ctx, enc, "keep")?;
                        enc.set_bytes(0, &[blocks as u32, blocks.min(c.candidate_topk) as u32]);
                        enc.set_tensor(1, &self.sort_ids);
                        enc.set_tensor(2, &self.candidates);
                        dispatch(enc, &pso, blocks);
                    }
                    if unpruned {
                        self.select_unpruned(ctx, enc, id, length)?;
                    }
                    self.sort(
                        ctx,
                        enc,
                        &self.index_scores,
                        length,
                        !unpruned
                            && self
                                .schedule
                                .candidate_source_layer
                                .is_some_and(|source| id > source),
                    )?;
                    let pso = pipeline(ctx, enc, "chronological")?;
                    enc.set_bytes(0, &(selected_count as u32));
                    enc.set_tensor(1, &self.sort_ids);
                    enc.set_tensor(2, &self.selected);
                    dispatch(enc, &pso, selected_count);
                }
            }
        }
        #[cfg(test)]
        if prepared.is_none() {
            super::ced::tests::chunked::encode_control_probes(
                ctx,
                enc,
                id,
                position,
                1,
                &self.selected,
                &self.candidates,
            )?;
        }
        if prepared.is_none() {
            let raw = self.raw.view_subrange(
                (id * c.window * c.head_dim) as u64,
                vec![(c.window * c.head_dim) as u64],
            );
            let raw_count = (position + 1).min(c.window);
            enc.set_bytes(
                0,
                &[
                    c.head_dim as u32,
                    raw_count as u32,
                    ((position + 1 - raw_count) % c.window) as u32,
                    c.window as u32,
                ],
            );
            enc.set_bytes(
                1,
                &[selected_count as u32, (c.window + c.index_topk + 1) as u32],
            );
            enc.set_tensor(2, &self.query);
            enc.set_tensor(3, &raw);
            enc.set_tensor(
                4,
                owner.map_or(&self.latent, |slot| &self.owners[slot].compressed),
            );
            enc.set_tensor(5, &self.selected);
            enc.set_tensor(6, &constants.sink);
            enc.set_tensor(7, &self.attention_scores);
            enc.set_tensor(8, &self.attended);
            let length = owner.map_or(0, |_| (position + 1) / self.schedule.ratio(id));
            enc.set_bytes(9, &(length as u32));
            enc.set_tensor(10, &self.status);
            sink_passes(
                ctx,
                enc,
                c.heads,
                c.head_dim,
                raw_count + selected_count,
                self.value_matrix
                    .as_ref()
                    .map(|matrix| (matrix, &self.attended, &self.status)),
            )?;
            rope(
                ctx,
                enc,
                &self.attended,
                &constants.frequencies,
                c.head_dim,
                c.heads,
                c.rope_dim / 2,
                position,
                true,
            )?;
            let group_width = c.heads / c.groups * c.head_dim;
            match Self::output_a_bank(w, c)
                .filter(|bank| metal::all_slots_down_native_supported(bank.dtype))
            {
                // One dispatch over all groups: slot g reads attended group g
                // and writes low group g through the same native F32 GEMV the
                // per-group path uses (experts = groups, ids = 0..groups).
                Some(bank) => metal::encode_all_slots_down_native(
                    ctx,
                    enc,
                    &bank,
                    &self
                        .attended
                        .view_bytes(0, vec![group_width as u64, c.groups as u64]),
                    &self.group_ids,
                    &self.ready,
                    &self
                        .low
                        .view_bytes(0, vec![c.output_rank as u64, c.groups as u64]),
                    group_width,
                    c.output_rank,
                    c.groups,
                    c.groups,
                )
                .map_err(|e| e.to_string())?,
                None => {
                    for (group, projection) in w.output_a.iter().enumerate() {
                        linear(
                            ctx,
                            enc,
                            projection,
                            &self.attended.view_subrange(
                                (group * group_width) as u64,
                                vec![group_width as u64],
                            ),
                            &self.low.view_subrange(
                                (group * c.output_rank) as u64,
                                vec![c.output_rank as u64],
                            ),
                        )?;
                    }
                }
            }
            linear(ctx, enc, &w.output_b, &self.low, &self.output)?;
            let pso = pipeline(ctx, enc, "finite")?;
            enc.set_bytes(0, &(c.hidden as u32));
            enc.set_tensor(1, &self.output);
            enc.set_tensor(2, &self.status);
            dispatch(enc, &pso, c.hidden);
        }
        if self.ced.is_some() {
            self.advance_ced_row()?;
        } else {
            if let Some(batch) = self.batch.as_mut() {
                batch.row += 1;
                if batch.row == batch.rows {
                    batch.row = 0;
                    batch.layer += 1;
                }
                if batch.layer == self.schedule.layers {
                    self.next_position = batch.start + batch.rows;
                    self.next_layer = 0;
                    self.batch = None;
                }
            } else {
                self.next_layer = (id + 1) % self.schedule.layers;
                if self.next_layer == 0 {
                    self.next_position += 1;
                }
            }
        }
        Ok(&self.output)
    }

    /// Sticky device error bits: 1 NaN selection score, 2 invalid selected ID,
    /// 4 nonfinite attention/published or seeded KV. Inspect only after the
    /// caller's command completes.
    pub(super) fn status(&self) -> &MetalTensor {
        &self.status
    }
}

fn group_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    output: &MetalTensor,
    width: usize,
    groups: usize,
    rows: usize,
) -> Result<()> {
    let pso = ctx
        .pipeline("kernel_deepseek_v41_group_rows")
        .map_err(|e| e.to_string())?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &[width as u32, groups as u32, rows as u32, 0]);
    enc.set_tensor(1, input);
    enc.set_tensor(2, output);
    dispatch(enc, &pso, width * groups * rows);
    Ok(())
}

fn validate_weights(c: &Config, id: usize, w: &Weights<'_>, schedule: &Schedule) -> Result<()> {
    let owns = w.compressor.is_some();
    op::require(
        owns == schedule.owns_kv(id) && w.index.is_some() == schedule.produces_index(id),
        "GPU attention ownership weights",
    )?;
    op::require(
        w.query_norm
            .iter()
            .chain(&w.kv_norm)
            .chain(&w.sink)
            .chain(&w.frequencies)
            .all(|x| x.is_finite()),
        "GPU attention finite constants",
    )?;
    projection(&w.query_a, c.hidden, c.query_rank)?;
    projection(&w.query_b, c.query_rank, c.heads * c.head_dim)?;
    projection(&w.kv, c.hidden, c.head_dim)?;
    projection(&w.output_b, c.groups * c.output_rank, c.hidden)?;
    op::require(
        w.head_dim == c.head_dim
            && w.window == c.window
            && w.query_norm.len() == c.query_rank
            && w.kv_norm.len() == c.head_dim
            && w.sink.len() == c.heads
            && w.frequencies.len() == c.rope_dim / 2
            && w.output_a.len() == c.groups
            && w.eps == c.norm_eps
            && w.candidate_block_size == c.candidate_block
            && w.candidate_topk_blocks == c.candidate_topk,
        "GPU attention weights",
    )?;
    for p in &w.output_a {
        projection(p, c.heads / c.groups * c.head_dim, c.output_rank)?;
    }
    if let Some(index) = &w.index {
        op::require(
            index.heads == c.index_heads
                && index.dim == c.index_dim
                && index.topk == c.index_topk
                && index.key.is_some() == owns
                && (owns || index.key_norm.is_empty())
                && index.key_norm.iter().all(|x| x.is_finite()),
            "GPU index geometry",
        )?;
        projection(&index.query, c.query_rank, c.index_heads * c.index_dim)?;
        projection(&index.weights, c.hidden, c.index_heads)?;
        if owns {
            let compressor = w.compressor.as_ref().ok_or("missing GPU compressor")?;
            op::require(
                compressor.ratio == schedule.ratio(id)
                    && compressor.norm.len() == c.head_dim
                    && compressor.gate.is_some() == (compressor.ratio == 2)
                    && compressor.norm.iter().all(|x| x.is_finite())
                    && index.key_norm.len() == c.index_dim,
                "GPU compressor geometry",
            )?;
            projection(&compressor.kv, c.hidden, c.head_dim)?;
            if compressor.ratio == 2 {
                projection(
                    compressor.gate.as_ref().ok_or("missing GPU gate")?,
                    c.hidden,
                    c.head_dim,
                )?;
            }
            projection(
                index.key.as_ref().ok_or("missing GPU key")?,
                c.head_dim,
                c.index_dim,
            )?;
        }
    }
    Ok(())
}
