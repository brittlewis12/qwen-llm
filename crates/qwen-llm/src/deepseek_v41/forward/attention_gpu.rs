//! Encode-only F32 attention. Workspace metadata is provisional: a failed encode
//! or command completion requires the caller to poison/discard the workspace.
use super::super::gguf::Schedule;
use super::prefill::RowsArithmetic;
use super::*;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLSize};

mod ced;
mod query_batch;
pub(super) use ced::{
    GlobalKvMode, PublicationRows, QueryControlView, QueryControls, SeedRows, ced_group_cpu_bytes,
};
#[cfg(test)]
pub(super) use query_batch::{with_f32_attention, with_old_host_loop};
#[cfg(test)]
mod tests;

type Weights<'a> = op::AttentionWeights<Linear<'a>>;

#[derive(Debug)]
struct OutputANativeLayout {
    group_input: usize,
    group_bytes: u64,
    bank_bytes: u64,
}

fn output_a_native_layout(c: &Config, dtype: GgmlType, offset: u64) -> Option<OutputANativeLayout> {
    if !(1..=metal::ROUTE_MAX_TOP_K).contains(&c.groups)
        || c.heads == 0
        || !c.heads.is_multiple_of(c.groups)
        || c.head_dim == 0
        || c.output_rank == 0
        || c.output_rank > u32::MAX as usize
        || !offset.is_multiple_of(16)
        || !metal::all_slots_down_native_supported(dtype)
    {
        return None;
    }
    let group_input = (c.heads / c.groups).checked_mul(c.head_dim)?;
    if group_input > u32::MAX as usize {
        return None;
    }
    let (block, bytes) = dtype.storage_layout()?;
    if !(group_input as u64).is_multiple_of(block) {
        return None;
    }
    let group_bytes = (group_input as u64 / block)
        .checked_mul(bytes)?
        .checked_mul(c.output_rank as u64)?;
    let bank_bytes = group_bytes.checked_mul(c.groups as u64)?;
    offset.checked_add(bank_bytes)?;
    (group_input as u64)
        .checked_mul(c.output_rank as u64)?
        .checked_mul(c.groups as u64)?;
    (group_input as u64)
        .checked_mul(c.groups as u64)?
        .checked_mul(4)?;
    (c.output_rank as u64)
        .checked_mul(c.groups as u64)?
        .checked_mul(4)?;
    Some(OutputANativeLayout {
        group_input,
        group_bytes,
        bank_bytes,
    })
}

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
    #[cfg(test)]
    mask_publication_experiment: Option<MaskPublicationExperiment>,
    config: Config,
    schedule: Schedule,
    capacity: usize,
    next_layer: usize,
    next_position: usize,
    batch: Option<BatchCursor>,
    ced: Option<ced::Cursor>,
    /// (layer, position, last stage) of a decode step encoded in stages
    /// (`encode_checks` or `encode_front` .. `encode_finish`) whose finish
    /// is still due.
    staged: Option<(usize, usize, Stage)>,
    /// The caller folds the next decode output's finite check into its mHC
    /// post (`defer_finite`); `finite_pending` marks one owed.
    defer_finite: bool,
    finite_pending: bool,
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
    /// One-row scratch of the multi-threadgroup radix selector (decode).
    radix_grid: RadixGridScratch,
    selected: MetalTensor,
    attention_scores: MetalTensor,
    value_matrix: Option<ValueMatrix>,
    /// Split scratch of the fused decode attention (head_dim <= 512).
    flash: Option<FlashScratch>,
    attended: MetalTensor,
    low: MetalTensor,
    output: MetalTensor,
    status: MetalTensor,
    /// Constant slot ids 0..groups and a READY status, so eligible
    /// output_a group GEMVs dispatch once through the native expert kernel.
    group_ids: MetalTensor,
    ready: MetalTensor,
}

/// Last completed stage of a staged decode step (`Workspace::encode_front`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// Checks passed; the caller encodes the rank pair.
    Checked,
    /// Rank norms done; the caller encodes q_b.
    Front,
    /// Attended heads ready; output_a (own or the caller's), the caller's
    /// output_b, then the finish.
    Middle,
}

/// Bindings of a staged decode step's caller-encoded projections: the rank
/// pair writes `query_rank` and `latent`, q_b reads `query_rank` and writes
/// `query`, output_a reads `attended` and writes `low`, output_b reads `low`
/// and writes `output`.
pub(super) struct StagedIo<'a> {
    pub(super) query_rank: &'a MetalTensor,
    pub(super) latent: &'a MetalTensor,
    pub(super) query: &'a MetalTensor,
    pub(super) attended: &'a MetalTensor,
    pub(super) low: &'a MetalTensor,
    pub(super) output: &'a MetalTensor,
}

#[cfg(test)]
struct MaskPublicationExperiment {
    arm: MaskPublicationArm,
    retained_ids: Option<MetalTensor>,
}

const MASK_FUSED_MAX_DOMAIN: usize = 131072;
const MASK_FUSED_TG_BYTES: usize = 4096 * 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MaskPublicationArm {
    Incumbent,
    Fused,
    #[cfg(test)]
    ClearScatter,
}

fn candidate_publication_arm(
    domain: usize,
    candidate_topk: usize,
    concurrent: bool,
    tg_capacity: usize,
    forced: Option<MaskPublicationArm>,
) -> MaskPublicationArm {
    if let Some(arm) = forced {
        return arm;
    }
    let count = domain.min(candidate_topk);
    if !concurrent
        && count >= 2048
        && domain > count
        && domain <= MASK_FUSED_MAX_DOMAIN
        && tg_capacity >= MASK_FUSED_TG_BYTES
    {
        MaskPublicationArm::Fused
    } else {
        MaskPublicationArm::Incumbent
    }
}

fn candidate_fused_words(domain: usize) -> Option<usize> {
    (domain <= MASK_FUSED_MAX_DOMAIN).then(|| (domain + 31) / 32)
}

fn candidate_fused_width(max_threads: usize, static_bytes: usize, tg_capacity: usize) -> usize {
    if tg_capacity < MASK_FUSED_TG_BYTES.max(static_bytes) {
        0
    } else {
        max_threads.min(256)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchCursor {
    start: usize,
    rows: usize,
    layer: usize,
    row: usize,
}

impl BatchCursor {
    fn advanced(&self, layer: usize, position: usize, count: usize) -> Result<Self> {
        let end = self
            .start
            .checked_add(self.rows)
            .ok_or("attention batch overflow")?;
        let current = self
            .start
            .checked_add(self.row)
            .ok_or("attention row overflow")?;
        let row = self
            .row
            .checked_add(count)
            .ok_or("attention range overflow")?;
        op::require(
            count > 0
                && self.row < self.rows
                && row <= self.rows
                && self.layer == layer
                && current == position
                && current < end,
            "attention range order/extent",
        )?;
        let mut next = *self;
        if row == self.rows {
            next.layer = self
                .layer
                .checked_add(1)
                .ok_or("attention layer overflow")?;
            next.row = 0;
        } else {
            next.row = row;
        }
        Ok(next)
    }
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

crate::env_flag!(
    /// Read the tensor64 score Q from the shared half scratch; `QWEN_V41_ATTN_QHALF=0` restores the F32 query.
    default_on prefill_qhalf_enabled,
    "QWEN_V41_ATTN_QHALF"
);

pub(super) struct Rows {
    tile: query_batch::Scratch,
    rank: MetalTensor,
    query: MetalTensor,
    /// Half mirror of the post-RoPE query in the shared half scratch; the
    /// tensor64 score reads it instead of re-converting F32 per key panel.
    query_half: std::cell::RefCell<Option<MetalTensor>>,
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
    // Constructor provenance; checked views must not silently lose half staging.
    half_required: bool,
    rows: usize,
    geometry: [usize; 14],
    arithmetic: RowsArithmetic,
    controls: Option<QueryControlView>,
    history: std::cell::Cell<Option<ced::HistoryRange>>,
}

impl Rows {
    fn geometry(c: &Config, capacity: usize) -> [usize; 14] {
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
            c.rope_dim,
            capacity,
        ]
    }

    fn prefix_widths(c: &Config, capacity: usize) -> Result<Vec<usize>> {
        op::require(
            capacity > 0
                && capacity <= u32::MAX as usize / 8
                && Self::geometry(c, capacity)[..12]
                    .iter()
                    .all(|&n| n > 0 && n <= u32::MAX as usize / 4)
                && c.heads.is_multiple_of(c.groups),
            "attention prefix geometry/capacity",
        )?;
        let query = checked_product(&[c.heads, c.head_dim])?;
        let low = checked_product(&[c.groups, c.output_rank])?;
        let index = checked_product(&[c.index_heads, c.index_dim])?;
        Ok(vec![
            c.query_rank,
            query,
            c.head_dim,
            c.index_topk,
            capacity.div_ceil(c.candidate_block),
            low,
            query,
            low,
            c.head_dim,
            c.head_dim,
            index,
            c.index_heads,
        ])
    }

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
        let mut scratch = Self {
            tile: query_batch::Scratch::new(ctx, c, rows)?,
            rank: prefill::allocate(ctx, widths[0], rows, GgmlType::F32)?,
            query: prefill::allocate(ctx, widths[1], rows, GgmlType::F32)?,
            query_half: std::cell::RefCell::new(None),
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
            half_required: false,
            rows,
            geometry: Self::geometry(c, capacity),
            arithmetic: if controls.is_some() {
                RowsArithmetic::Packed
            } else {
                RowsArithmetic::Ordinary
            },
            controls,
            history: std::cell::Cell::new(None),
        };
        scratch.half_required = scratch.half.is_some();
        Ok(scratch)
    }

    pub(super) fn checked_prefix_with_controls(
        &self,
        c: &Config,
        capacity: usize,
        controls: &QueryControlView,
    ) -> Result<Self> {
        let rows = controls.rows();
        prefill::checked_scratch_rows(self.rows, rows)?;
        op::require(
            self.half.is_some() == self.half_required,
            "attention prefix half backing presence differs from constructor",
        )?;
        op::require(
            self.geometry == Self::geometry(c, capacity),
            "attention prefix incompatible backing geometry",
        )?;
        let widths = Self::prefix_widths(c, capacity)?;
        let selection = query_batch::selection_shapes(c, capacity, rows, query_batch::tile_rows())?;
        let backing_selection =
            query_batch::selection_shapes(c, capacity, self.rows, query_batch::tile_rows())?;
        let f = |tensor, i| {
            checked_product(&[widths[i], rows])?;
            prefill::checked_scratch_view(
                tensor,
                &[widths[i], self.rows],
                &[widths[i], rows],
                GgmlType::F32,
            )
        };
        let selected = prefill::checked_scratch_view(
            &controls.selected,
            &[widths[3], rows],
            &[widths[3], rows],
            GgmlType::I32,
        )?;
        let candidates = prefill::checked_scratch_view(
            &controls.candidates,
            &[widths[4], rows],
            &[widths[4], rows],
            GgmlType::I32,
        )?;
        controls.validate_geometry(c, capacity)?;
        let select_pairs = if selection[1] == 0 {
            None
        } else {
            let pairs = self
                .select_pairs
                .as_ref()
                .ok_or("attention selection pair backing missing")?;
            op::require(pairs.offset.is_multiple_of(8), "selection pair alignment")?;
            Some(prefill::checked_scratch_view(
                pairs,
                &[backing_selection[1], 1],
                &[selection[1], 1],
                GgmlType::I32,
            )?)
        };
        Ok(Self {
            tile: self.tile.checked_prefix(c, rows)?,
            rank: f(&self.rank, 0)?,
            query: f(&self.query, 1)?,
            query_half: std::cell::RefCell::new(None),
            latent: f(&self.latent, 2)?,
            low: f(&self.low, 5)?,
            selected,
            candidates,
            grouped_input: f(&self.grouped_input, 6)?,
            grouped_low: f(&self.grouped_low, 7)?,
            compressor_value: f(&self.compressor_value, 8)?,
            compressor_gate: f(&self.compressor_gate, 9)?,
            index_query: f(&self.index_query, 10)?,
            index_weights: f(&self.index_weights, 11)?,
            select_scores: prefill::checked_scratch_view(
                &self.select_scores,
                &[backing_selection[0], 1],
                &[selection[0], 1],
                GgmlType::F32,
            )?,
            select_pairs,
            half: self
                .half
                .as_ref()
                .map(|half| half.checked_prefix(prefill::attention_half_scratch_elements(c, rows)?))
                .transpose()?,
            half_required: self.half_required,
            rows,
            geometry: self.geometry,
            arithmetic: RowsArithmetic::Packed,
            controls: Some(controls.clone()),
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

#[cfg(test)]
mod scratch_prefix_geometry_tests {
    use super::*;

    #[test]
    #[ignore = "self-leased actual scratch buffers; MTL debug layer; no model or dispatch"]
    fn attention_actual_prefix_rebinds_controls_and_rejects_invalid_parents() {
        assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
        assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
        assert!(!crate::env_flag::read_default_off(
            "QWEN_METAL_LEASE_SKIP_WIRED_GATE"
        ));
        let _lease = metal::acquire_metal_benchmark_lease().unwrap();
        let ctx = MetalContext::new().unwrap();
        let c = Config {
            heads: 8,
            head_dim: 32,
            index_topk: 2,
            ..super::super::toy::config()
        };
        let (maximum, capacity) = (8, 32);
        let retained = 10..22;
        let bytes = Rows::widths_with_controls(&c, capacity)
            .into_iter()
            .map(|w| w * maximum)
            .chain(query_batch::Scratch::shapes(&c, maximum).unwrap())
            .chain(
                query_batch::selection_shapes(&c, capacity, maximum, query_batch::tile_rows())
                    .unwrap(),
            )
            .chain(QueryControls::shapes(&c, capacity, retained.clone()).unwrap())
            .filter(|&n| n != 0)
            .map(|n| (n * 4) as u64)
            .chain([
                (prefill::attention_half_scratch_elements(&c, maximum).unwrap() * 2) as u64,
                18 * 2,
            ])
            .map(|n| ctx.price_shared_buffer_upper(n).unwrap().priced_upper_bytes)
            .sum::<u64>();
        assert!(bytes < 16 * 1024 * 1024);
        let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
            bytes,
            1024 * 1024,
            0,
            ctx.memory_signals(),
            true,
        );
        assert!(admission.refusal().is_none());

        fn scratch(rows: &Rows) -> Vec<&MetalTensor> {
            let mut tensors = vec![
                &rows.rank,
                &rows.query,
                &rows.latent,
                &rows.low,
                &rows.grouped_input,
                &rows.grouped_low,
                &rows.compressor_value,
                &rows.compressor_gate,
                &rows.index_query,
                &rows.index_weights,
                &rows.select_scores,
                &rows.tile.history,
                &rows.tile.scores,
                &rows.tile.attended,
            ];
            tensors.extend(rows.select_pairs.as_ref());
            if let Some(matrix) = &rows.tile.matrix {
                tensors.extend([&matrix.values, &matrix.probabilities]);
            }
            tensors
        }

        objc2::rc::autoreleasepool(|_| {
            let controls = QueryControls::new(&ctx, &c, capacity, retained.clone()).unwrap();
            let initial = controls.checked_tile_view(11, maximum).unwrap();
            let mut backing = Rows::with_controls(&ctx, &c, capacity, &initial).unwrap();
            let history = ced::HistoryRange {
                layer: 4,
                start: 11,
                floor: 9,
                extent: 10,
            };
            backing.history.set(Some(history));
            for (start, n) in [(11, 8), (17, 1), (13, 5)] {
                let controls = controls.checked_tile_view(start, n).unwrap();
                let view = backing
                    .checked_prefix_with_controls(&c, capacity, &controls)
                    .unwrap();
                assert_eq!(view.rows, n);
                assert_eq!(view.arithmetic, RowsArithmetic::Packed);
                assert_eq!(view.half_required, backing.half_required);
                assert_eq!(view.history.get(), None);
                assert_eq!(backing.history.get(), Some(history));
                assert_eq!(
                    view.controls.as_ref().unwrap().absolute_rows(),
                    start..start + n
                );
                for (view, original) in scratch(&view).into_iter().zip(scratch(&backing)) {
                    assert!(std::ptr::eq(&*view.buffer, &*original.buffer));
                    assert_eq!(view.offset, original.offset);
                    assert!(view.n_elements() <= original.n_elements());
                }
                for (actual, expected, width) in [
                    (&view.selected, controls.selected(), c.index_topk),
                    (
                        &view.candidates,
                        controls.candidates(),
                        capacity.div_ceil(c.candidate_block),
                    ),
                ] {
                    assert!(std::ptr::eq(&*actual.buffer, &*expected.buffer));
                    assert_eq!(actual.offset, ((start - retained.start) * width * 4) as u64);
                    assert_eq!(actual.offset, expected.offset);
                    assert_eq!(actual.n_elements(), (width * n) as u64);
                }
                assert!(
                    view.checked_prefix_with_controls(&c, capacity, &controls)
                        .is_err()
                );
                assert!(view.tile.checked_prefix(&c, 1).is_err());
            }
            let current = controls.checked_tile_view(13, 5).unwrap();
            let matrix = backing.tile.matrix.take();
            assert!(
                matrix.is_some(),
                "fixture must exercise the gathered matrix path"
            );
            assert!(
                backing
                    .checked_prefix_with_controls(&c, capacity, &current)
                    .is_err()
            );
            backing.tile.matrix = matrix;
            assert!(
                backing
                    .checked_prefix_with_controls(&c, capacity, &current)
                    .is_ok()
            );
            if let Some(half) = backing.half.take() {
                assert!(backing.half_required);
                assert!(
                    backing
                        .checked_prefix_with_controls(&c, capacity, &current)
                        .is_err()
                );
                assert_eq!(backing.history.get(), Some(history));
                backing.half = Some(half);
                assert!(
                    backing
                        .checked_prefix_with_controls(&c, capacity, &current)
                        .is_ok()
                );
            }

            let half = prefill::allocate(&ctx, 18, 1, GgmlType::F16).unwrap();
            let shifted = half.view_subrange(1, vec![16, 1]);
            let view =
                prefill::checked_scratch_view(&shifted, &[16, 1], &[5], GgmlType::F16).unwrap();
            assert!(std::ptr::eq(&*view.buffer, &*half.buffer));
            assert_eq!(view.offset, 2);
            assert_eq!(view.n_elements(), 5);
            assert!(prefill::checked_scratch_view(&view, &[16, 1], &[1], GgmlType::F16).is_err());
            let mut misaligned = shifted.clone();
            misaligned.offset = 1;
            assert!(
                prefill::checked_scratch_view(&misaligned, &[16, 1], &[5], GgmlType::F16).is_err()
            );
        });
    }

    #[test]
    fn attention_prefix_widths_preserve_fresh_shapes_and_current_group_strides() {
        let c = Config::default();
        let widths = Rows::prefix_widths(&c, 16384).unwrap();
        assert_eq!(widths, Rows::widths(&c, 16384));
        let group_width = c.heads * c.head_dim / c.groups;
        for rows in [1, 3, 127, 512, prefill::DEFAULT_CHUNK_ROWS] {
            let grouped = checked_product(&[widths[6], rows]).unwrap();
            let low = checked_product(&[widths[7], rows]).unwrap();
            assert_eq!(
                (c.groups - 1) * group_width * rows + group_width * rows,
                grouped
            );
            assert_eq!(
                (c.groups - 1) * c.output_rank * rows + c.output_rank * rows,
                low
            );
            let half = prefill::attention_half_scratch_elements(&c, rows).unwrap();
            assert!(half >= grouped && half >= low);
            assert!(RowsArithmetic::Packed.packed_operands(rows));
        }
    }

    #[test]
    fn attention_prefix_geometry_rejects_invalid_capacity_and_products() {
        let mut c = Config::default();
        assert!(Rows::prefix_widths(&c, 0).is_err());
        assert!(Rows::prefix_widths(&c, usize::MAX).is_err());
        c.candidate_block = 0;
        assert!(Rows::prefix_widths(&c, 16384).is_err());
        c = Config::default();
        c.groups = 0;
        assert!(Rows::prefix_widths(&c, 16384).is_err());
        c = Config::default();
        c.heads = usize::MAX;
        assert!(Rows::prefix_widths(&c, 16384).is_err());
        c = Config::default();
        c.head_dim = u32::MAX as usize / 4;
        assert!(Rows::prefix_widths(&c, 16384).is_err());
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
                && self.staged.is_none()
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
        self.staged = None;
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
        RadixGridScratch::domain_words(capacity),
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
    sizes.extend(FlashScratch::shape(
        c.heads,
        c.head_dim,
        c.window + c.index_topk,
    )?);
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

crate::env_flag!(
    /// Exact radix top-k for index selection, fed by F32 index scores from the long-row index kernels (`index_heads_rows_mma`, or `index_heads_rows_ilp` with `QWEN_V41_SELECT_GRID=0`; the scalar kernel's operations, to F32 rounding): decode rows, and prefill chunks past the unpruned selectors' 16K-key domain (batched scores, radix candidate publication and radix top-k over row sub-tiles); disabling restores the per-row scalar index and bitonic sort network for a same-binary A/B. Given the same scores both select the same IDs.
    default_on select_radix_enabled,
    "QWEN_V41_SELECT_RADIX"
);

/// `index_heads_rows_ilp` covers the 32-head x 128-dim index geometry.
fn index_ilp_supported(c: &Config) -> bool {
    c.index_heads == 32 && c.index_dim == 128
}

/// F32 index scores for `n` rows starting at chunk-local row `first` of a
/// chunk at absolute `start` (`kernel_deepseek_v41_attention_index_heads_rows_
/// ilp`: `index_heads_rows` operations, 64 keys per threadgroup). Row scores
/// land at `scores[local * stride ..]`; rows with at most top-k visible keys
/// skip. Decode is the one-row case (`start` = position, `first` = 0).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn encode_index_scores_rows_ilp(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    c: &Config,
    start: usize,
    ratio: usize,
    max_length: usize,
    first: usize,
    n: usize,
    stride: usize,
    query: &MetalTensor,
    weights: &MetalTensor,
    keys: &MetalTensor,
    scores: &MetalTensor,
) -> Result<()> {
    encode_index_scores_rows_kernel(
        ctx,
        enc,
        c,
        IndexRowsKernel::Ilp,
        [start, ratio, max_length, first, n, stride],
        query,
        weights,
        keys,
        scores,
    )
}

/// `encode_index_scores_rows_ilp` on `index_heads_rows_mma` (F32 simdgroup
/// MMA, 128 keys per threadgroup): the same scores up to F32 summation order.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn encode_index_scores_rows_mma(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    c: &Config,
    start: usize,
    ratio: usize,
    max_length: usize,
    first: usize,
    n: usize,
    stride: usize,
    query: &MetalTensor,
    weights: &MetalTensor,
    keys: &MetalTensor,
    scores: &MetalTensor,
) -> Result<()> {
    encode_index_scores_rows_kernel(
        ctx,
        enc,
        c,
        IndexRowsKernel::Mma,
        [start, ratio, max_length, first, n, stride],
        query,
        weights,
        keys,
        scores,
    )
}

/// The radix path's long-row index scores: MMA unless `QWEN_V41_SELECT_GRID`
/// is off, then ILP.
#[allow(clippy::too_many_arguments)]
fn encode_index_scores_rows_long(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    c: &Config,
    start: usize,
    ratio: usize,
    max_length: usize,
    first: usize,
    n: usize,
    stride: usize,
    query: &MetalTensor,
    weights: &MetalTensor,
    keys: &MetalTensor,
    scores: &MetalTensor,
) -> Result<()> {
    let kernel = if select_grid_enabled() {
        IndexRowsKernel::Mma
    } else {
        IndexRowsKernel::Ilp
    };
    encode_index_scores_rows_kernel(
        ctx,
        enc,
        c,
        kernel,
        [start, ratio, max_length, first, n, stride],
        query,
        weights,
        keys,
        scores,
    )
}

#[derive(Clone, Copy)]
enum IndexRowsKernel {
    Ilp,
    Mma,
}

/// Shared binding of the 32x128 long-row index kernels. `g`: chunk start,
/// ratio, longest visible length, first row, rows, score stride.
#[allow(clippy::too_many_arguments)]
fn encode_index_scores_rows_kernel(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    c: &Config,
    kernel: IndexRowsKernel,
    g: [usize; 6],
    query: &MetalTensor,
    weights: &MetalTensor,
    keys: &MetalTensor,
    scores: &MetalTensor,
) -> Result<()> {
    let [start, ratio, max_length, first, n, stride] = g;
    op::require(index_ilp_supported(c), "ILP index geometry")?;
    let (suffix, keys_per_group) = match kernel {
        IndexRowsKernel::Ilp => ("index_heads_rows_ilp", 64),
        IndexRowsKernel::Mma => ("index_heads_rows_mma", 128),
    };
    pipeline(ctx, enc, suffix)?;
    enc.set_bytes(
        0,
        &[
            start as u32,
            ratio as u32,
            c.index_topk as u32,
            max_length as u32,
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
    enc.set_bytes(2, &[stride as u32, n as u32, 0, 0]);
    enc.set_tensor(3, query);
    enc.set_tensor(4, weights);
    enc.set_tensor(5, keys);
    enc.set_tensor(6, scores);
    enc.set_bytes(7, &(first as u32));
    enc.dispatch(
        MTLSize {
            width: max_length.div_ceil(keys_per_group),
            height: n,
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

/// Exact top-k of one row's index scores (`kernel_deepseek_v41_attention_
/// select_radix_rows`), written as chronological IDs to `selected`; `mask`
/// restricts the domain to the row's candidate blocks like the sort path.
#[allow(clippy::too_many_arguments)]
fn encode_select_radix_row(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    c: &Config,
    position: usize,
    ratio: usize,
    scores: &MetalTensor,
    candidates: &MetalTensor,
    selected: &MetalTensor,
    status: &MetalTensor,
    mask: bool,
) -> Result<()> {
    let pso = pipeline(ctx, enc, "select_radix_rows")?;
    op::require(
        pso.maxTotalThreadsPerThreadgroup() >= 1024,
        "radix selection needs 1024-thread threadgroups",
    )?;
    enc.set_bytes(
        0,
        &[
            position as u32,
            ratio as u32,
            c.index_topk as u32,
            c.candidate_block as u32,
        ],
    );
    enc.set_bytes(1, &[0u32, 0, 1, mask as u32]);
    enc.set_bytes(2, &[c.index_topk as u32, 0, c.candidate_topk as u32, 0]);
    enc.set_tensor(3, scores);
    enc.set_tensor(4, candidates);
    enc.set_tensor(5, selected);
    enc.set_tensor(6, status);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

crate::env_flag!(
    /// Wide kernels on the radix selection path (`QWEN_V41_SELECT_RADIX`): decode's one-row top-k by the multi-threadgroup radix selector (`select_radix_grid_hist`/`_place`) and long-row index scores, decode and batched prefill, by the F32 simdgroup-MMA kernel (`index_heads_rows_mma`); disabling restores the single-threadgroup radix kernel and the ILP index kernel for a same-binary A/B. Same IDs given the same scores; MMA scores differ from ILP only by F32 summation order.
    default_on select_grid_enabled,
    "QWEN_V41_SELECT_GRID"
);

/// Words before the per-chunk tables in a radix grid scratch row
/// (`V41_RADIX_GRID_TABLES`: three digit histograms and the selection state).
const RADIX_GRID_HEADER: usize = 5128;
/// Words per chunk table (`V41_RADIX_GRID_TABLE`).
const RADIX_GRID_TABLE: usize = 1028;
/// Threads per `select_radix_grid_*` threadgroup; a chunk holds at most four
/// keys per thread.
const RADIX_GRID_THREADS: usize = 512;
const RADIX_GRID_MAX_CHUNK: usize = 4 * RADIX_GRID_THREADS;
/// `radix_grid_chunk` aims for this many chunks per row.
const RADIX_GRID_TARGET_CHUNKS: usize = 32;

/// Keys per threadgroup for a selection domain: about
/// `RADIX_GRID_TARGET_CHUNKS` chunks of 512 to 2048 keys (the fastest shapes
/// measured on M4 Max from 16K to 128K keys).
fn radix_grid_chunk(domain: usize) -> usize {
    domain
        .div_ceil(RADIX_GRID_TARGET_CHUNKS)
        .next_power_of_two()
        .clamp(RADIX_GRID_THREADS, RADIX_GRID_MAX_CHUNK)
}

/// Scratch rows of the multi-threadgroup radix selector: one per
/// dispatch-local row, each with room for `chunks` chunk tables. Allocated
/// zeroed; the kernels keep the histograms zeroed between selections (see
/// `select_radix_grid_hist`).
struct RadixGridScratch {
    tensor: MetalTensor,
    rows: usize,
    stride: usize,
}

impl RadixGridScratch {
    fn new(ctx: &MetalContext, rows: usize, chunks: usize) -> Result<Self> {
        let stride = RADIX_GRID_HEADER + RADIX_GRID_TABLE * chunks;
        Ok(Self {
            tensor: tensor(ctx, rows.max(1) * stride, GgmlType::I32)?,
            rows,
            stride,
        })
    }

    /// Chunks of every domain up to `domain` at `radix_grid_chunk`'s shapes:
    /// at most `RADIX_GRID_TARGET_CHUNKS` until the chunk size caps.
    fn domain_chunks(domain: usize) -> usize {
        domain
            .div_ceil(RADIX_GRID_MAX_CHUNK)
            .max(RADIX_GRID_TARGET_CHUNKS)
    }

    /// Words of a one-row scratch for domains up to `domain`.
    fn domain_words(domain: usize) -> usize {
        RADIX_GRID_HEADER + RADIX_GRID_TABLE * Self::domain_chunks(domain)
    }

    /// One row with room for every domain up to `domain`.
    fn for_domain(ctx: &MetalContext, domain: usize) -> Result<Self> {
        Self::new(ctx, 1, Self::domain_chunks(domain))
    }
}

/// `select_radix_rows` over `b[2]` rows split across threadgroups
/// (`kernel_deepseek_v41_attention_select_radix_grid_hist` for each digit,
/// then `select_radix_grid_place`): identical arguments, order, masks, modes,
/// NaN status and outputs. `domain` bounds every row's selection domain
/// (visible keys, or blocks in mode 1) and `chunk` is the keys per
/// threadgroup.
#[allow(clippy::too_many_arguments)]
fn encode_select_radix_grid(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    a: [u32; 4],
    b: [u32; 4],
    c: [u32; 4],
    domain: usize,
    chunk: usize,
    scores: &MetalTensor,
    candidates: &MetalTensor,
    selected: &MetalTensor,
    status: &MetalTensor,
    scratch: &RadixGridScratch,
) -> Result<()> {
    let rows = b[2] as usize;
    let chunks = domain.div_ceil(chunk.max(1));
    op::require(
        (1..=RADIX_GRID_MAX_CHUNK).contains(&chunk)
            && rows <= scratch.rows
            && RADIX_GRID_HEADER + RADIX_GRID_TABLE * chunks <= scratch.stride,
        "radix grid scratch",
    )?;
    if rows == 0 || domain == 0 {
        return Ok(());
    }
    for (kernel, pass) in [("hist", 0u32), ("hist", 1), ("hist", 2), ("place", 0)] {
        let pso = pipeline(ctx, enc, &format!("select_radix_grid_{kernel}"))?;
        op::require(
            pso.maxTotalThreadsPerThreadgroup() >= RADIX_GRID_THREADS,
            "radix grid threadgroup size",
        )?;
        enc.set_bytes(0, &a);
        enc.set_bytes(1, &b);
        enc.set_bytes(2, &c);
        enc.set_tensor(3, scores);
        enc.set_tensor(4, candidates);
        enc.set_tensor(5, selected);
        enc.set_tensor(6, status);
        enc.set_tensor(7, &scratch.tensor);
        enc.set_bytes(8, &[pass, chunk as u32, scratch.stride as u32, 0]);
        enc.dispatch(
            MTLSize {
                width: chunks,
                height: rows,
                depth: 1,
            },
            MTLSize {
                width: RADIX_GRID_THREADS,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(())
}

/// `select_radix_rows` on the multi-threadgroup selector at
/// `radix_grid_chunk(domain)`.
#[allow(clippy::too_many_arguments)]
fn encode_select_radix_wide(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    a: [u32; 4],
    b: [u32; 4],
    c: [u32; 4],
    domain: usize,
    scores: &MetalTensor,
    candidates: &MetalTensor,
    selected: &MetalTensor,
    status: &MetalTensor,
    scratch: &RadixGridScratch,
) -> Result<()> {
    encode_select_radix_grid(
        ctx,
        enc,
        a,
        b,
        c,
        domain,
        radix_grid_chunk(domain),
        scores,
        candidates,
        selected,
        status,
        scratch,
    )
}

/// `encode_select_radix_row` on the wide selectors.
#[allow(clippy::too_many_arguments)]
fn encode_select_radix_row_wide(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    c: &Config,
    position: usize,
    ratio: usize,
    scores: &MetalTensor,
    candidates: &MetalTensor,
    selected: &MetalTensor,
    status: &MetalTensor,
    scratch: &RadixGridScratch,
    mask: bool,
) -> Result<()> {
    encode_select_radix_wide(
        ctx,
        enc,
        [
            position as u32,
            ratio as u32,
            c.index_topk as u32,
            c.candidate_block as u32,
        ],
        [0, 0, 1, mask as u32],
        [c.index_topk as u32, 0, c.candidate_topk as u32, 0],
        (position + 1) / ratio,
        scores,
        candidates,
        selected,
        status,
        scratch,
    )
}

crate::env_flag!(
    /// Sink-attention scores (decode and scalar rows) by `attention_score_keys`, which computes 8-head x 4-key register tiles (each lane loads 12 values per 32 products instead of 2 per product); the per-(head, key) arithmetic is `attention_score`'s, so scores are bitwise equal. Disabling restores the one-key-per-simdgroup `attention_score` kernel (A/B).
    default_on score_keys_env,
    "QWEN_V41_SCORE_KEYS"
);

#[cfg(test)]
thread_local! {
    pub(super) static SCORE_KEYS_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Keys and heads per `attention_score_keys` threadgroup (8 simdgroups of
/// 8-head x 4-key tiles); the dimension limit of the dispatch choice.
const SCORE_KEY_BLOCK: usize = 32;
const SCORE_HEAD_BLOCK: usize = 8;
const SCORE_KEYS_MAX_DIM: usize = 512;

fn score_keys_enabled(dim: usize) -> bool {
    #[cfg(test)]
    if let Some(enabled) = SCORE_KEYS_OVERRIDE.with(|o| o.get()) {
        return enabled && dim <= SCORE_KEYS_MAX_DIM;
    }
    score_keys_env() && dim <= SCORE_KEYS_MAX_DIM
}

/// Split scratch of the fused decode sink attention
/// (`attention_flash*` + `attention_flash_combine*`): per split, the
/// unnormalized partial rows (heads x dim), then per (split, head) the split
/// maximum and exp sum, then one invalid-ID flag per split.
struct FlashScratch {
    scratch: MetalTensor,
    splits: usize,
}

impl FlashScratch {
    /// Scratch for rows of up to `max_keys` keys with `flash_kernel`'s
    /// choices, or None when the fused path does not apply (head_dim > 512).
    fn new(ctx: &MetalContext, heads: usize, dim: usize, max_keys: usize) -> Result<Option<Self>> {
        match flash_splits(dim, max_keys) {
            Some(splits) => Self::with_splits(ctx, heads, dim, splits).map(Some),
            None => Ok(None),
        }
    }

    fn with_splits(ctx: &MetalContext, heads: usize, dim: usize, splits: usize) -> Result<Self> {
        op::require(splits <= FLASH_MAX_SPLITS, "flash split count")?;
        Ok(Self {
            scratch: tensor(ctx, Self::elements(heads, dim, splits)?, GgmlType::F32)?,
            splits,
        })
    }

    fn elements(heads: usize, dim: usize, splits: usize) -> Result<usize> {
        checked_product(&[splits, heads, dim])?
            .checked_add(2 * splits * heads + splits)
            .ok_or_else(|| "flash scratch extent".into())
    }

    /// Element counts `new` allocates (for workspace pricing).
    fn shape(heads: usize, dim: usize, max_keys: usize) -> Result<Vec<usize>> {
        match flash_splits(dim, max_keys) {
            Some(splits) => Ok(vec![Self::elements(heads, dim, splits)?]),
            None => Ok(Vec::new()),
        }
    }
}

/// `V41_FLASH_MAX_SPLITS` and the dimension limit of the fused decode
/// attention (two output columns per thread of a 256-thread group); rows of
/// at most `FLASH_SHORT_KEYS` keys use the short-row kernel.
const FLASH_MAX_SPLITS: usize = 64;
const FLASH_MAX_DIM: usize = 512;
const FLASH_SHORT_KEYS: usize = 256;

crate::env_flag!(
    /// Decode sink attention by one fused flash-decoding pass: `attention_flash*` key splits (scores, per-split softmax, unnormalized PV) then `attention_flash_combine_rope` (sink-aware merge plus the attended inverse RoPE) replace score -> softmax -> output or value matrix -> RoPE. F32; outputs match the chain to F32 rounding (softmax and PV summation order differ). Disabling restores the chain (A/B).
    default_on flash_decode_env,
    "QWEN_V41_FLASH_DECODE"
);

#[cfg(test)]
thread_local! {
    pub(super) static FLASH_DECODE_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

fn flash_decode_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = FLASH_DECODE_OVERRIDE.with(|o| o.get()) {
        return enabled;
    }
    flash_decode_env()
}

/// The fused decode partial kernel (suffix, heads and keys per threadgroup)
/// for a row of `count` keys, or None (head_dim > 512) to keep the chain:
/// short rows `flash2_h4_k16` (dim % 4 == 0), otherwise `flash_h8_k32`
/// (`_d512` at head_dim 512). Decode probe on M5 Ultra, 64 heads x 512:
/// 128 + 0 keys 11.9 us (chain 55.9), 128 + 512 keys 19.0 us (chain 42.0).
fn flash_kernel(dim: usize, count: usize) -> Option<(&'static str, usize, usize)> {
    if dim == 0 || dim > FLASH_MAX_DIM {
        return None;
    }
    Some(if count <= FLASH_SHORT_KEYS && dim.is_multiple_of(4) {
        ("flash2_h4_k16", 4, 16)
    } else if dim == 512 {
        ("flash_h8_k32_d512", 8, 32)
    } else {
        ("flash_h8_k32", 8, 32)
    })
}

/// Splits the scratch needs for every row of 1..=max_keys keys, or None
/// when the fused path does not apply to such rows (head_dim > 512, or more
/// than `FLASH_MAX_SPLITS` splits: windows past ~2000 keys keep the chain).
fn flash_splits(dim: usize, max_keys: usize) -> Option<usize> {
    (1..=max_keys)
        .map(|n| flash_kernel(dim, n).map(|(_, _, kb)| n.div_ceil(kb)))
        .try_fold(0, |m, s| s.map(|s| m.max(s)))
        .filter(|&s| s <= FLASH_MAX_SPLITS)
}

/// The fused decode sink attention for one row (bindings 0..10 as
/// `sink_passes`; `count` raw + selected keys): `partial` (an
/// `attention_flash*` suffix with its heads and keys per threadgroup) over
/// ceil(count / kb) key splits, then `attention_flash_combine`, or with
/// `rope` = (frequencies, pairs, position) `attention_flash_combine_rope`,
/// which also applies the attended inverse RoPE (`rope(.., inverse)`).
#[allow(clippy::too_many_arguments)]
fn flash_passes(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    heads: usize,
    dim: usize,
    count: usize,
    scratch: &FlashScratch,
    (partial, hg, kb): (&str, usize, usize),
    rope: Option<(&MetalTensor, usize, usize)>,
) -> Result<()> {
    let splits = count.div_ceil(kb);
    op::require(
        count > 0 && splits <= scratch.splits && dim <= FLASH_MAX_DIM,
        "flash decode geometry",
    )?;
    enc.set_bytes(14, &(heads as u32));
    enc.set_tensor(15, &scratch.scratch);
    enc.set_bytes(16, &[kb as u32, scratch.splits as u32, 0, 0]);
    let threads = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    pipeline(ctx, enc, partial)?;
    enc.dispatch(
        MTLSize {
            width: splits,
            height: heads.div_ceil(hg),
            depth: 1,
        },
        threads,
    );
    if let Some((freq, pairs, position)) = rope {
        enc.set_tensor(17, freq);
        enc.set_bytes(18, &[pairs as u32, position as u32]);
        pipeline(ctx, enc, "flash_combine_rope")?;
    } else {
        pipeline(ctx, enc, "flash_combine")?;
    }
    enc.dispatch(
        MTLSize {
            width: dim.div_ceil(256),
            height: heads,
            depth: 1,
        },
        threads,
    );
    Ok(())
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
    if count > 0 && score_keys_enabled(dim) {
        pipeline(ctx, enc, "score_keys")?;
        enc.set_bytes(14, &(heads as u32));
        enc.dispatch(
            MTLSize {
                width: count.div_ceil(SCORE_KEY_BLOCK),
                height: heads.div_ceil(SCORE_HEAD_BLOCK),
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    } else if count > 0 {
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
/// `split` keeps the decode query/KV norms, KV RoPE and ring copy as four
/// dispatches; otherwise `kernel_deepseek_v41_attention_rank_norms` does them
/// in one (bitwise equal; A/B lever).
const QKV_CHAIN_ENV: &str = "QWEN_V41_QKV_CHAIN";

#[cfg(test)]
thread_local! {
    pub(super) static FUSED_RANK_NORMS_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

fn fused_rank_norms() -> bool {
    #[cfg(test)]
    if let Some(fused) = FUSED_RANK_NORMS_OVERRIDE.with(|o| o.get()) {
        return fused;
    }
    static FUSED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FUSED.get_or_init(|| std::env::var(QKV_CHAIN_ENV).as_deref() != Ok("split"))
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RankNormsArgs {
    q_n: u32,
    kv_n: u32,
    pairs: u32,
    position: u32,
    eps: f32,
}

#[allow(clippy::too_many_arguments)]
fn rank_norms(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    c: &Config,
    constants: &Constants,
    eps: f32,
    position: usize,
    query_rank: &MetalTensor,
    latent: &MetalTensor,
    ring_row: &MetalTensor,
) -> Result<()> {
    valid_view(query_rank, c.query_rank)?;
    valid_view(latent, c.head_dim)?;
    valid_view(ring_row, c.head_dim)?;
    op::require(
        2 * (c.rope_dim / 2) <= c.head_dim
            && constants.query_norm.n_elements() as usize == c.query_rank
            && constants.kv_norm.n_elements() as usize == c.head_dim
            && constants.frequencies.n_elements() as usize >= c.rope_dim / 2,
        "decode rank norms geometry",
    )?;
    pipeline(ctx, enc, "rank_norms")?;
    enc.set_bytes(
        0,
        &RankNormsArgs {
            q_n: c.query_rank as u32,
            kv_n: c.head_dim as u32,
            pairs: (c.rope_dim / 2) as u32,
            position: position as u32,
            eps,
        },
    );
    enc.set_tensor(1, query_rank);
    enc.set_tensor(2, &constants.query_norm);
    enc.set_tensor(3, latent);
    enc.set_tensor(4, &constants.kv_norm);
    enc.set_tensor(5, &constants.frequencies);
    enc.set_tensor(6, ring_row);
    enc.dispatch(
        MTLSize {
            width: 2,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    );
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CompressRowsArgs {
    ratio: u32,
    group0: u32,
    row0: u32,
    pairs: u32,
    eps: f32,
}

/// One threadgroup per completed ratio-group: pool, compressor norm, index
/// key projection/norm, rope and both cache stores. The caller passes the
/// full value/gate rows and the owner cache; `row0` is the group's first
/// chunk row and `group0` its absolute cache row.
#[allow(clippy::too_many_arguments)]
fn compress_publish_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    value: &MetalTensor,
    gate: &MetalTensor,
    comp_norm: &MetalTensor,
    key: &MetalTensor,
    key_norm: &MetalTensor,
    freq: &MetalTensor,
    out_key: &MetalTensor,
    out_latent: &MetalTensor,
    row0: usize,
    group0: usize,
    groups: usize,
    ratio: usize,
    pairs: usize,
    eps: f32,
) -> Result<()> {
    if groups == 0 {
        return Ok(());
    }
    pipeline(ctx, enc, "compress_rows")?;
    enc.set_bytes(
        0,
        &CompressRowsArgs {
            ratio: ratio as u32,
            group0: group0 as u32,
            row0: row0 as u32,
            pairs: pairs as u32,
            eps,
        },
    );
    enc.set_tensor(1, value);
    enc.set_tensor(2, gate);
    enc.set_tensor(3, comp_norm);
    enc.set_tensor(4, key);
    enc.set_tensor(5, key_norm);
    enc.set_tensor(6, freq);
    enc.set_tensor(7, out_latent);
    enc.set_tensor(8, out_key);
    enc.dispatch(
        MTLSize {
            width: groups,
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
            #[cfg(test)]
            mask_publication_experiment: None,
            schedule: schedule.clone(),
            capacity,
            next_layer: 0,
            next_position: 0,
            batch: None,
            ced: None,
            staged: None,
            defer_finite: false,
            finite_pending: false,
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
            radix_grid: RadixGridScratch::for_domain(ctx, capacity)?,
            selected: i(c.index_topk)?,
            attention_scores: f(c.heads * (c.window + c.index_topk + 1))?,
            value_matrix: ValueMatrix::new(ctx, c)?,
            flash: FlashScratch::new(ctx, c.heads, c.head_dim, c.window + c.index_topk)?,
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

    /// Eligible output_a group matrices as one contiguous [group_input,
    /// output_rank, groups] bank, when their views are adjacent slices of one
    /// resident tensor (how `bind` creates them).
    fn output_a_bank(w: &Weights<'_>, c: &Config) -> Option<MetalTensor> {
        let first = &w.output_a.first()?.tensor;
        let OutputANativeLayout {
            group_input,
            group_bytes,
            bank_bytes,
        } = output_a_native_layout(c, first.dtype, first.offset)?;
        if w.output_a.len() != c.groups {
            return None;
        }
        for (g, linear) in w.output_a.iter().enumerate() {
            let t = &linear.tensor;
            if !std::ptr::eq(&*t.buffer, &*first.buffer)
                || t.offset
                    != first
                        .offset
                        .checked_add((g as u64).checked_mul(group_bytes)?)?
                || t.dtype != first.dtype
                || t.shape != [group_input as u64, c.output_rank as u64]
                || linear.input != group_input
                || linear.output != c.output_rank
            {
                return None;
            }
        }
        if first.offset.checked_add(bank_bytes)? > first.buffer.length() as u64 {
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

    fn publish_candidates(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        blocks: usize,
    ) -> Result<()> {
        let count = blocks.min(self.config.candidate_topk);
        let forced = None;
        #[cfg(test)]
        let forced = if let Some(experiment) = &self.mask_publication_experiment {
            if let Some(retained) = &experiment.retained_ids {
                metal::encode_copy_offset_i32(ctx, enc, &self.sort_ids, 0, retained, count)
                    .map_err(|error| error.to_string())?;
            }
            Some(experiment.arm)
        } else {
            forced
        };
        let capacity = ctx.device.maxThreadgroupMemoryLength();
        let arm = candidate_publication_arm(blocks, count, enc.is_concurrent(), capacity, forced);
        let mut width = 0;
        let suffix = match arm {
            MaskPublicationArm::Incumbent => "keep",
            MaskPublicationArm::Fused => {
                op::require(
                    !enc.is_concurrent()
                        && candidate_fused_words(blocks).is_some()
                        && capacity >= MASK_FUSED_TG_BYTES,
                    "fused mask requires serial encoder, bounded domain and on-chip capacity",
                )?;
                "keep_fused"
            }
            #[cfg(test)]
            MaskPublicationArm::ClearScatter => {
                op::require(
                    !enc.is_concurrent(),
                    "mask experiment requires serial encoder",
                )?;
                "keep_clear"
            }
        };
        let mut pso = ctx
            .pipeline(&format!("kernel_deepseek_v41_attention_{suffix}"))
            .map_err(|e| e.to_string())?;
        if arm == MaskPublicationArm::Fused {
            width = candidate_fused_width(
                pso.maxTotalThreadsPerThreadgroup(),
                pso.staticThreadgroupMemoryLength(),
                capacity,
            );
            if width == 0 {
                op::require(
                    forced.is_none(),
                    "forced fused mask pipeline exceeds TG limits",
                )?;
                pso = ctx
                    .pipeline("kernel_deepseek_v41_attention_keep")
                    .map_err(|e| e.to_string())?;
            }
        }
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &[blocks as u32, count as u32]);
        enc.set_tensor(1, &self.sort_ids);
        enc.set_tensor(2, &self.candidates);
        if width > 0 {
            enc.dispatch(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            dispatch(enc, &pso, blocks);
        }
        #[cfg(test)]
        if arm == MaskPublicationArm::ClearScatter {
            let scatter = pipeline(ctx, enc, "keep_scatter")?;
            dispatch(enc, &scatter, count);
        }
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
        op::require(self.staged.is_none(), "decode during a staged decode step")?;
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

    /// Whether a decode step of this workspace can be encoded in stages:
    /// plain decode (no prefill batch, CED cursor or pending staged step) on
    /// the fused rank-norm chain (`QWEN_V41_QKV_CHAIN` not `split`).
    pub(super) fn staged_decode_supported(&self) -> bool {
        self.batch.is_none() && self.ced.is_none() && self.staged.is_none() && fused_rank_norms()
    }

    fn require_stage(&self, id: usize, position: usize, stage: Stage) -> Result<()> {
        op::require(
            self.staged == Some((id, position, stage)),
            "staged decode order",
        )
    }

    /// `encode` in stages, so a caller can encode the dense projections of
    /// several sequences at once. Per sequence, in order:
    /// - `encode_front` (all checks, then the rank pair and the rank norms;
    ///   writes the query rank), or `encode_checks` (checks only), the
    ///   caller's rank pair (input -> `StagedIo::query_rank` and `latent`)
    ///   and `encode_rank_norms`;
    /// - the caller's q_b (`query_rank` -> `query`);
    /// - `encode_middle` (query RoPE through the attended RoPE; writes
    ///   `attended`);
    /// - `encode_output_a`, or the caller's output_a (`attended` -> `low`);
    /// - the caller's output_b (`low` -> `output`);
    /// - `encode_finish` (finite check and the cursor advance).
    ///
    /// With `encode_linear_pair` and `Linear::encode` for the caller's
    /// projections it is `encode`, dispatch for dispatch. Validation happens
    /// first; the cursors move only in the finish, so an error at any stage
    /// leaves them unadvanced (the caller poisons the sequence). The stages
    /// this workspace encodes refuse to run out of order; the caller's
    /// projections are the caller's to order.
    pub(super) fn encode_front(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
    ) -> Result<()> {
        op::require(
            self.staged_decode_supported(),
            "staged decode needs plain fused-chain decode",
        )?;
        let projected = self.encode_qkv(
            ctx,
            enc,
            id,
            position,
            w,
            input,
            None,
            GlobalKvMode::Ordinary,
        )?;
        op::require(!projected, "staged decode query projection")?;
        self.staged = Some((id, position, Stage::Front));
        Ok(())
    }

    /// The checks of a staged decode step whose rank pair the caller
    /// encodes (see `encode_front`). Encodes nothing.
    pub(super) fn encode_checks(
        &mut self,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
    ) -> Result<()> {
        op::require(
            self.staged_decode_supported(),
            "staged decode needs plain fused-chain decode",
        )?;
        self.check_step(enc, id, position, w, input, GlobalKvMode::Ordinary)?;
        self.staged = Some((id, position, Stage::Checked));
        Ok(())
    }

    /// The rank norms after the caller's rank pair (see `encode_front`).
    pub(super) fn encode_rank_norms(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
    ) -> Result<()> {
        self.require_stage(id, position, Stage::Checked)?;
        metal::dispatch_profile_label("attn_qkv", Some(id));
        self.encode_fused_rank_norms(ctx, enc, id, position, w)?;
        self.staged = Some((id, position, Stage::Front));
        Ok(())
    }

    /// The middle stage of a staged decode step (see `encode_front`); the
    /// query must already be projected.
    pub(super) fn encode_middle(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
    ) -> Result<()> {
        self.require_stage(id, position, Stage::Front)?;
        self.encode_core(
            ctx,
            enc,
            id,
            position,
            w,
            input,
            None,
            true,
            true,
            GlobalKvMode::Ordinary,
        )?;
        self.staged = Some((id, position, Stage::Middle));
        Ok(())
    }

    /// This sequence's own output_a after the middle (see `encode_front`).
    pub(super) fn encode_output_a(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
    ) -> Result<()> {
        self.require_stage(id, position, Stage::Middle)?;
        metal::dispatch_profile_label("attn_output", Some(id));
        self.encode_output_a_on(ctx, enc, w, &self.attended, &self.low)
    }

    /// The last stage of a staged decode step (see `encode_front`); the
    /// output must already be projected. Returns the output.
    pub(super) fn encode_finish(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
    ) -> Result<&MetalTensor> {
        self.require_stage(id, position, Stage::Middle)?;
        self.staged = None;
        self.encode_close(ctx, enc, id, true)?;
        Ok(&self.output)
    }

    /// Leave the next decode output's finite check (bit 4 of `status()` for
    /// any non-finite output element) to the caller, which folds it into the
    /// mHC post that reads the output; `take_finite_pending` then reports it
    /// owed, and `encode_finite` encodes it when the post cannot fold it.
    pub(super) fn defer_finite(&mut self) {
        self.defer_finite = true;
    }

    /// Whether a deferred finite check is owed (clears the marker and any
    /// unused deferral).
    pub(super) fn take_finite_pending(&mut self) -> bool {
        self.defer_finite = false;
        std::mem::take(&mut self.finite_pending)
    }

    /// The standalone decode output finite check.
    pub(super) fn encode_finite(&self, ctx: &MetalContext, enc: &KernelEncoder) -> Result<()> {
        let pso = pipeline(ctx, enc, "finite")?;
        enc.set_bytes(0, &(self.config.hidden as u32));
        enc.set_tensor(1, &self.output);
        enc.set_tensor(2, &self.status);
        dispatch(enc, &pso, self.config.hidden);
        Ok(())
    }

    /// Bindings of the projections a staged decode step leaves to the
    /// caller.
    pub(super) fn staged_io(&self) -> StagedIo<'_> {
        StagedIo {
            query_rank: &self.query_rank,
            latent: &self.latent,
            query: &self.query,
            attended: &self.attended,
            low: &self.low,
            output: &self.output,
        }
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
        self.encode_batch_row_mode(
            ctx, enc, id, position, w, input, rows, row, true, select, mode,
        )
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
        publish: bool,
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
            .encode_inner_with(
                ctx,
                enc,
                id,
                position,
                w,
                input,
                Some((rows, row)),
                publish,
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
        self.encode_inner_with(
            ctx, enc, id, position, w, input, prepared, true, select, mode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_inner_with(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
        prepared: Option<(&Rows, usize)>,
        publish: bool,
        select: bool,
        mode: GlobalKvMode,
    ) -> Result<&MetalTensor> {
        if !self.encode_qkv(ctx, enc, id, position, w, input, prepared, mode)? {
            linear(ctx, enc, &w.query_b, &self.query_rank, &self.query)?;
        }
        self.encode_core(
            ctx, enc, id, position, w, input, prepared, publish, select, mode,
        )?;
        if prepared.is_none() {
            metal::dispatch_profile_label("attn_output", Some(id));
            self.encode_output_a_on(ctx, enc, w, &self.attended, &self.low)?;
            linear(ctx, enc, &w.output_b, &self.low, &self.output)?;
        }
        self.encode_close(ctx, enc, id, prepared.is_none())?;
        Ok(&self.output)
    }

    /// Every check of a step, before anything is encoded: serial encoder,
    /// order/capacity, global mode, weights and input.
    fn check_step(
        &self,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
        mode: GlobalKvMode,
    ) -> Result<()> {
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
        valid_view(input, c.hidden)
    }

    /// Layer `id`'s raw KV ring row for `position`.
    fn raw_row(&self, id: usize, position: usize) -> MetalTensor {
        let c = &self.config;
        self.raw
            .view_subrange(
                (id * c.window * c.head_dim) as u64,
                vec![(c.window * c.head_dim) as u64],
            )
            .view_subrange(
                (position % c.window * c.head_dim) as u64,
                vec![c.head_dim as u64],
            )
    }

    /// One dispatch for the query/KV rank norms, the KV RoPE and the ring
    /// store, after both rank projections.
    fn encode_fused_rank_norms(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
    ) -> Result<()> {
        rank_norms(
            ctx,
            enc,
            &self.config,
            &self.constants[id],
            w.eps,
            position,
            &self.query_rank,
            &self.latent,
            &self.raw_row(id, position),
        )
    }

    /// The step's checks, then (unless prepared) its query/KV projections.
    /// Order/capacity, global mode, weights and input are validated before
    /// anything is encoded; nothing advances here. Returns whether the query
    /// up-projection (q_b: `query_rank` -> `query`) is already encoded: false
    /// only on the fused rank-norm decode chain, whose next dispatch is q_b.
    #[allow(clippy::too_many_arguments)]
    fn encode_qkv(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
        prepared: Option<(&Rows, usize)>,
        mode: GlobalKvMode,
    ) -> Result<bool> {
        self.check_step(enc, id, position, w, input, mode)?;
        let c = &self.config;
        let constants = &self.constants[id];
        if prepared.is_some() {
            return Ok(true);
        }
        metal::dispatch_profile_label("attn_qkv", Some(id));
        let row = self.raw_row(id, position);
        if fused_rank_norms() {
            // Both rank projections, then one dispatch for both norms, the
            // KV RoPE and the ring store; the caller projects the query.
            super::encode_linear_pair(
                enc,
                (&w.query_a, &self.query_rank),
                (&w.kv, &self.latent),
                input,
            )
            .map_err(|e| e.to_string())?;
            self.encode_fused_rank_norms(ctx, enc, id, position, w)?;
            return Ok(false);
        }
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
            &self.latent,
            &constants.frequencies,
            c.head_dim,
            1,
            c.rope_dim / 2,
            position,
            false,
        )?;
        copy(ctx, enc, &self.latent, &row, c.head_dim)?;
        Ok(true)
    }

    /// Query RoPE, owner publication, index selection, attention core,
    /// attended RoPE and output_a (writes `low`) of a step whose query is
    /// projected (prepared rows only publish and select).
    #[allow(clippy::too_many_arguments)]
    fn encode_core(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
        prepared: Option<(&Rows, usize)>,
        publish: bool,
        select: bool,
        mode: GlobalKvMode,
    ) -> Result<()> {
        let c = &self.config;
        let constants = &self.constants[id];
        let label = |name| metal::dispatch_profile_label(name, Some(id));
        if prepared.is_none() {
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
        }
        let owner = self.schedule.owner_slot(id);
        let mut selected_count = 0;
        if let Some(owner) = owner {
            let cache = &self.owners[owner];
            let ratio = self.schedule.ratio(id);
            let length = (position + 1) / ratio;
            if publish && self.schedule.owns_kv(id) && mode == GlobalKvMode::Ordinary {
                label("attn_owner");
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
                label("attn_index");
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
                    if select_radix_enabled() && index_ilp_supported(c) {
                        encode_index_scores_rows_long(
                            ctx,
                            enc,
                            c,
                            position,
                            ratio,
                            length,
                            0,
                            1,
                            length,
                            &self.index_query,
                            &self.index_weights,
                            &cache.keys,
                            &self.index_scores,
                        )?;
                    } else {
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
                    }
                    if !unpruned && self.schedule.candidate_source_layer == Some(id) {
                        let blocks = length.div_ceil(c.candidate_block);
                        let pso = pipeline(ctx, enc, "blocks")?;
                        enc.set_bytes(0, &[length as u32, c.candidate_block as u32]);
                        enc.set_tensor(1, &self.index_scores);
                        enc.set_tensor(2, &self.block_scores);
                        dispatch(enc, &pso, blocks);
                        self.sort(ctx, enc, &self.block_scores, blocks, false)?;
                        self.publish_candidates(ctx, enc, blocks)?;
                    }
                    if unpruned {
                        self.select_unpruned(ctx, enc, id, length)?;
                    }
                    let mask = !unpruned
                        && self
                            .schedule
                            .candidate_source_layer
                            .is_some_and(|source| id > source);
                    if select_radix_enabled() && selected_count == c.index_topk {
                        if select_grid_enabled() {
                            encode_select_radix_row_wide(
                                ctx,
                                enc,
                                c,
                                position,
                                ratio,
                                &self.index_scores,
                                &self.candidates,
                                &self.selected,
                                &self.status,
                                &self.radix_grid,
                                mask,
                            )?;
                        } else {
                            encode_select_radix_row(
                                ctx,
                                enc,
                                c,
                                position,
                                ratio,
                                &self.index_scores,
                                &self.candidates,
                                &self.selected,
                                &self.status,
                                mask,
                            )?;
                        }
                    } else {
                        self.sort(ctx, enc, &self.index_scores, length, mask)?;
                        let pso = pipeline(ctx, enc, "chronological")?;
                        enc.set_bytes(0, &(selected_count as u32));
                        enc.set_tensor(1, &self.sort_ids);
                        enc.set_tensor(2, &self.selected);
                        dispatch(enc, &pso, selected_count);
                    }
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
            label("attn_core");
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
            let count = raw_count + selected_count;
            let flash = self
                .flash
                .as_ref()
                .filter(|_| flash_decode_enabled())
                .zip(flash_kernel(c.head_dim, count));
            if let Some((scratch, kernel)) = flash {
                let pairs = c.rope_dim / 2;
                flash_passes(
                    ctx,
                    enc,
                    c.heads,
                    c.head_dim,
                    count,
                    scratch,
                    kernel,
                    (pairs > 0).then_some((&constants.frequencies, pairs, position)),
                )?;
            } else {
                sink_passes(
                    ctx,
                    enc,
                    c.heads,
                    c.head_dim,
                    count,
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
            }
        }
        Ok(())
    }

    /// output_a of one sequence: its attended heads (`attended`, heads x
    /// head_dim) into its output rank (`low`, groups x output_rank).
    fn encode_output_a_on(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        w: &Weights<'_>,
        attended: &MetalTensor,
        low: &MetalTensor,
    ) -> Result<()> {
        let c = &self.config;
        let group_width = c.heads / c.groups * c.head_dim;
        match Self::output_a_bank(w, c) {
            // One dispatch over all groups: slot g reads attended group g
            // and writes low group g through the same native F32 GEMV the
            // per-group path uses (experts = groups, ids = 0..groups).
            Some(bank) => metal::encode_all_slots_down_native(
                ctx,
                enc,
                &bank,
                &attended.view_bytes(0, vec![group_width as u64, c.groups as u64]),
                &self.group_ids,
                &self.ready,
                &low.view_bytes(0, vec![c.output_rank as u64, c.groups as u64]),
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
                        &attended
                            .view_subrange((group * group_width) as u64, vec![group_width as u64]),
                        &low.view_subrange(
                            (group * c.output_rank) as u64,
                            vec![c.output_rank as u64],
                        ),
                    )?;
                }
            }
        }
        Ok(())
    }

    /// output_a of `cols` sequences gathered as columns: `x` holds their
    /// attended heads (`[heads * head_dim, cols]`), `y` receives their output
    /// ranks (`[groups * output_rank, cols]`). A Q5_K output_a bank takes
    /// `metal::encode_grouped_skinny_q5_k` (all groups, up to 8 columns per
    /// dispatch, each weight read once; F32 rounding); otherwise each column
    /// runs `encode_output_a_on` (bitwise per-sequence).
    pub(super) fn encode_output_a_columns(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        w: &Weights<'_>,
        x: &MetalTensor,
        y: &MetalTensor,
        cols: usize,
    ) -> Result<()> {
        let c = &self.config;
        let (width_in, width_out) = (c.heads * c.head_dim, c.groups * c.output_rank);
        op::require(
            cols > 0
                && [(x, width_in), (y, width_out)]
                    .iter()
                    .all(|(t, n)| t.dtype == GgmlType::F32 && t.n_elements() == (n * cols) as u64),
            "output_a column extents",
        )?;
        let group_width = width_in / c.groups;
        let bank = Self::output_a_bank(w, c).filter(|bank| {
            metal::grouped_skinny_supported(bank.dtype, group_width, c.output_rank, c.groups, 2)
                && bank.offset.is_multiple_of(16)
                && x.offset.is_multiple_of(16)
        });
        let mut first = 0;
        while first < cols {
            let n = (cols - first).min(metal::DENSE_SKINNY_MAX_COLS);
            let xs = x.view_bytes(
                (first * width_in * 4) as u64,
                vec![width_in as u64, n as u64],
            );
            let ys = y.view_bytes(
                (first * width_out * 4) as u64,
                vec![width_out as u64, n as u64],
            );
            match bank.as_ref().filter(|_| n >= 2) {
                Some(bank) => metal::encode_grouped_skinny_q5_k(
                    ctx,
                    enc,
                    bank,
                    &xs,
                    &ys,
                    group_width,
                    c.output_rank,
                    c.groups,
                    n,
                )
                .map_err(|e| e.to_string())?,
                None => {
                    for column in 0..n {
                        self.encode_output_a_on(
                            ctx,
                            enc,
                            w,
                            &xs.view_bytes((column * width_in * 4) as u64, vec![width_in as u64]),
                            &ys.view_bytes((column * width_out * 4) as u64, vec![width_out as u64]),
                        )?;
                    }
                }
            }
            first += n;
        }
        Ok(())
    }

    /// The decode output's finite check (`finite`), then the cursor advance
    /// past layer `id`: the only place a step moves the layer/position
    /// cursors.
    fn encode_close(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        finite: bool,
    ) -> Result<()> {
        let c = &self.config;
        if finite && std::mem::take(&mut self.defer_finite) {
            self.finite_pending = true;
        } else if finite {
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
        Ok(())
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
