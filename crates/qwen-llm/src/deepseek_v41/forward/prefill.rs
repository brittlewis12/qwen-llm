//! Bounded layer-major prefill with NativeF32 backbone projections and
//! position-ordered causal attention transitions within each layer.
use super::*;

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "ced/arithmetic.rs"]
mod arithmetic_tests;

#[cfg(test)]
#[path = "prefill/half_staging.rs"]
mod half_staging_tests;

pub const DEFAULT_CHUNK_ROWS: usize = 2048;
/// Global engine ceiling, not the width selected by request admission.
pub const MAX_CHUNK_ROWS: usize = 4096;

/// Planning/geometry ceiling only; execution keeps MAX_CHUNK_ROWS.
pub(super) fn full_chunk_ceiling() -> usize {
    #[cfg(test)]
    if ENCODER8K_FEASIBILITY.with(std::cell::Cell::get) {
        return 8192;
    }
    MAX_CHUNK_ROWS
}

#[cfg(test)]
thread_local! {
    static ENCODER8K_FEASIBILITY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(super) fn with_full_chunk_ceiling_for_test<T>(rows: usize, f: impl FnOnce() -> T) -> Result<T> {
    op::require(
        [MAX_CHUNK_ROWS, 8192].contains(&rows),
        "feasibility ceiling",
    )?;
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ENCODER8K_FEASIBILITY.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(ENCODER8K_FEASIBILITY.with(|flag| flag.replace(rows == 8192)));
    Ok(f())
}

/// Operand lineage of logical rows, independent of a CED replay tile's size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum RowsArithmetic {
    #[default]
    Ordinary,
    Packed,
}

impl RowsArithmetic {
    pub(super) fn packed_operands(self, rows: usize) -> bool {
        match self {
            Self::Ordinary => rows > 1,
            Self::Packed => rows > 0,
        }
    }
}

/// Absolute position of the genuine scalar final chunk in cold Full prefill.
pub(super) fn original_scalar_tail(prompt_rows: usize, chunk_rows: usize) -> Option<usize> {
    (prompt_rows % chunk_rows == 1).then(|| prompt_rows - 1)
}

#[cfg(test)]
thread_local! {
    pub(super) static ROWWISE_LINEAR_ROWS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests holding the F32-operand (NativeF32) contract pin the scalar F32
    /// grouped projection; production prefers the shared half-staged MMA tiles.
    pub(super) static F32_LINEAR_ROWS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(super) fn with_f32_linears<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            F32_LINEAR_ROWS.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(F32_LINEAR_ROWS.with(|flag| flag.replace(true)));
    f()
}

/// Controlled batched F32 operands, not a claim of identical reduction trees.
/// Keep this scope active during both scratch admission/allocation and encoding.
/// Prefer F32 grouped experts; an explicit native-rows override still wins.
#[cfg(test)]
pub(super) fn with_batched_f32<T>(f: impl FnOnce() -> T) -> T {
    with_f32_linears(|| {
        ffn_gpu::tests::with_grouped_rows(|| {
            attention_gpu::with_f32_attention(|| {
                metal::with_grouped_projection_scalar_baseline_override(true, f)
            })
        })
    })
}

/// Multi-row dense projections take the shared half-staged simdgroup-MMA
/// tiles for packed and BF16 weights. F32 weights (router, mHC projections)
/// keep the F32-operand grouped kernel: they are small and route-sensitive.
fn mma_linear_rows_selected(
    dtype: GgmlType,
    input: usize,
    rows: usize,
    arithmetic: RowsArithmetic,
) -> bool {
    let selected = arithmetic.packed_operands(rows)
        && dtype != GgmlType::F32
        && crate::metal_forward::mat_mat_dispatch_supported(dtype)
        && input.is_multiple_of(32);
    #[cfg(test)]
    let selected = selected
        && !ROWWISE_LINEAR_ROWS.with(|flag| flag.get())
        && !F32_LINEAR_ROWS.with(|flag| flag.get());
    selected
}

#[cfg(test)]
fn with_rowwise_linears<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ROWWISE_LINEAR_ROWS.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(ROWWISE_LINEAR_ROWS.with(|flag| flag.replace(true)));
    f()
}

/// Half copies of prefill activations for the tensor tile (`_f16act`):
/// one scratch per row set, converted once per distinct input and read by
/// every tensor-routed projection of that input. The tile rounds its
/// operand to half either way, so the results are bitwise identical to
/// the F32-input form; this only removes the per-tile re-read and cast.
pub(super) struct HalfScratch {
    tensor: MetalTensor,
    prefix_view: bool,
}

pub(super) fn checked_scratch_rows(backing_rows: usize, rows: usize) -> Result<()> {
    op::require(
        (1..=DEFAULT_CHUNK_ROWS).contains(&backing_rows) && (1..=backing_rows).contains(&rows),
        "scratch prefix rows outside original backing",
    )
}

fn scratch_elements(shape: &[u64]) -> Result<u64> {
    op::require(!shape.is_empty(), "empty scratch shape")?;
    shape.iter().try_fold(1u64, |n, &d| {
        op::require(d > 0, "zero scratch dimension")?;
        n.checked_mul(d)
            .ok_or_else(|| "scratch shape overflow".into())
    })
}

fn scratch_byte_extent(offset: u64, elements: u64, bytes: u64, limit: u64) -> Result<()> {
    op::require(offset.is_multiple_of(bytes), "scratch offset alignment")?;
    let end = elements
        .checked_mul(bytes)
        .and_then(|n| offset.checked_add(n))
        .ok_or("scratch byte extent overflow")?;
    op::require(
        end <= limit && end <= isize::MAX as u64,
        "scratch byte extent outside buffer",
    )
}

fn scratch_prefix_extent(
    backing_shape: &[u64],
    shape: &[u64],
    offset: u64,
    bytes: u64,
    limit: u64,
) -> Result<()> {
    let parent = scratch_elements(backing_shape)?;
    let elements = scratch_elements(shape)?;
    op::require(elements <= parent, "scratch prefix logical extent")?;
    scratch_byte_extent(offset, parent, bytes, limit)?;
    scratch_byte_extent(offset, elements, bytes, limit)
}

/// Check both the original logical extent and the physical allocation before
/// entering MetalTensor's asserting view API. Offsets remain dtype-sized.
pub(super) fn checked_scratch_view(
    tensor: &MetalTensor,
    backing_shape: &[usize],
    shape: &[usize],
    dtype: GgmlType,
) -> Result<MetalTensor> {
    op::require(
        tensor.dtype == dtype
            && tensor.is_writable()
            && tensor.shape.len() == backing_shape.len()
            && tensor
                .shape
                .iter()
                .zip(backing_shape)
                .all(|(&actual, &expected)| actual == expected as u64),
        "scratch backing shape/dtype/provenance",
    )?;
    let bytes = match dtype {
        GgmlType::F16 => 2,
        GgmlType::F32 | GgmlType::I32 => 4,
        _ => return Err("unsupported scratch dtype".into()),
    };
    let shape = shape.iter().map(|&n| n as u64).collect::<Vec<_>>();
    scratch_prefix_extent(
        &tensor.shape,
        &shape,
        tensor.offset,
        bytes,
        tensor.buffer.length() as u64,
    )?;
    Ok(tensor.view_subrange(0, shape))
}

#[cfg(test)]
mod scratch_prefix_geometry_tests {
    use super::*;

    #[test]
    fn prefix_rows_require_bounded_original_backing() {
        for backing in [1, 3, 128, DEFAULT_CHUNK_ROWS] {
            for rows in [1, backing] {
                assert!(checked_scratch_rows(backing, rows).is_ok());
            }
            assert!(checked_scratch_rows(backing, 0).is_err());
            assert!(checked_scratch_rows(backing, backing + 1).is_err());
        }
        assert!(checked_scratch_rows(0, 1).is_err());
        assert!(checked_scratch_rows(DEFAULT_CHUNK_ROWS + 1, 1).is_err());
        assert!(checked_scratch_rows(usize::MAX, 1).is_err());
    }

    #[test]
    fn prefix_extents_check_logical_and_physical_limits_before_views() {
        assert!(scratch_prefix_extent(&[24, 8], &[24, 3], 16, 4, 784).is_ok());
        assert!(scratch_prefix_extent(&[4, 4, 8], &[4, 4, 1], 0, 4, 512).is_ok());
        assert!(scratch_prefix_extent(&[8], &[9], 0, 4, 4096).is_err());
        assert!(scratch_prefix_extent(&[8], &[1], 0, 4, 31).is_err());
        assert!(scratch_prefix_extent(&[8], &[1], 2, 4, 64).is_err());
        assert!(scratch_prefix_extent(&[8], &[0], 0, 4, 64).is_err());
        assert!(scratch_prefix_extent(&[0], &[1], 0, 4, 64).is_err());
        assert!(scratch_prefix_extent(&[], &[1], 0, 4, 64).is_err());
        assert!(scratch_prefix_extent(&[u64::MAX, 2], &[1], 0, 4, u64::MAX).is_err());
        assert!(scratch_prefix_extent(&[u64::MAX], &[1], 0, 4, u64::MAX).is_err());
        assert!(scratch_prefix_extent(&[8], &[1], u64::MAX - 3, 4, u64::MAX).is_err());
    }

    #[test]
    fn half_prefix_extents_use_two_byte_offset_units() {
        assert!(scratch_prefix_extent(&[8, 1], &[3], 2, 2, 18).is_ok());
        assert!(scratch_prefix_extent(&[8, 1], &[3], 1, 2, 18).is_err());
        assert!(scratch_prefix_extent(&[8, 1], &[3], 2, 2, 17).is_err());
        assert!(HalfScratch::logical_bytes(usize::MAX).is_err());
    }
}

fn half_scratch_extent(factors: &[usize]) -> Result<usize> {
    let elements = factors.iter().try_fold(1usize, |n, &factor| {
        n.checked_mul(factor).ok_or("half scratch extent overflow")
    })?;
    HalfScratch::logical_bytes(elements)?;
    Ok(elements)
}

pub(super) fn attention_half_scratch_elements(c: &Config, rows: usize) -> Result<usize> {
    let grouped = half_scratch_extent(&[c.heads, c.head_dim])?;
    let low = half_scratch_extent(&[c.groups, c.output_rank])?;
    half_scratch_extent(&[c.hidden.max(c.query_rank).max(grouped).max(low), rows])
}

pub(super) fn ffn_half_scratch_elements(c: &Config, rows: usize) -> Result<[usize; 2]> {
    Ok([
        half_scratch_extent(&[c.hidden, rows])?,
        half_scratch_extent(&[c.intermediate, c.topk.max(1), rows])?,
    ])
}

/// Incremental half buffers owned by one attention/FFN row set. Price each
/// allocation separately, including buffers unused by arithmetic overrides.
pub(super) fn half_scratch_budget(ctx: &MetalContext, c: &Config, rows: usize) -> Result<u64> {
    half_scratch_budget_with(c, rows, HalfScratch::selected(ctx), &|bytes| {
        ctx.price_shared_buffer_upper(bytes)
            .map(|p| p.priced_upper_bytes)
            .map_err(|e| e.to_string())
    })
}

pub(super) fn half_scratch_budget_with(
    c: &Config,
    rows: usize,
    selected: bool,
    price: &impl Fn(u64) -> Result<u64>,
) -> Result<u64> {
    if !selected || rows == 0 {
        return Ok(0);
    }
    let attention = attention_half_scratch_elements(c, rows)?;
    let [input, inner] = ffn_half_scratch_elements(c, rows)?;
    [attention, input, inner]
        .into_iter()
        .filter(|&elements| elements != 0)
        .try_fold(0u64, |total, elements| {
            let priced = price(HalfScratch::logical_bytes(elements)?)?;
            total
                .checked_add(priced)
                .ok_or_else(|| "half scratch budget overflow".into())
        })
}

pub(super) fn full_scratch_budget_with(
    c: &Config,
    layers: usize,
    projections: &[(usize, usize)],
    capacity: usize,
    rows: usize,
    half_selected: bool,
    price: &impl Fn(u64) -> Result<u64>,
) -> Result<(u64, u64)> {
    gpu_step::validate_chunk_geometry(c, capacity, rows)?;
    // A scalar chunk executes the already-budgeted token workspace.
    if rows == 1 {
        return Ok((0, 0));
    }
    let mut metal = half_scratch_budget_with(c, rows, half_selected, price)?;
    if half_selected {
        metal = metal
            .checked_add(price(dense_preweight_bytes(c)?)?)
            .ok_or("prefill dense preweight overflow")?;
    }
    for width in gpu_step::chunk_widths_from_geometry(c, layers, projections, capacity, rows)? {
        let bytes = width
            .checked_mul(rows)
            .and_then(|n| n.checked_mul(4))
            .ok_or("prefill scratch overflow")?;
        metal = metal
            .checked_add(price(bytes as u64)?)
            .ok_or("prefill scratch overflow")?;
    }
    let cpu = upload_cpu_bytes(c, projections, rows)?
        .checked_add(chunk_hash_cpu_bytes(projections.len(), rows)?)
        .ok_or("prefill upload overflow")?;
    Ok((metal, cpu))
}

pub(super) fn upload_cpu_bytes(
    c: &Config,
    projections: &[(usize, usize)],
    rows: usize,
) -> Result<u64> {
    let upload_width = projections
        .iter()
        .map(|&(input, _)| input)
        .max()
        .unwrap_or(0)
        .checked_add(c.hidden.checked_mul(4).ok_or("prefill upload overflow")?)
        .ok_or("prefill upload overflow")?;
    (rows as u64)
        .checked_mul(upload_width as u64)
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| "prefill upload overflow".into())
}

pub(super) fn chunk_hash_cpu_bytes(hash_count: usize, rows: usize) -> Result<u64> {
    (hash_count as u64)
        .checked_mul(1024)
        .and_then(|n| n.checked_add(4096))
        .and_then(|n| n.checked_mul(rows as u64))
        .ok_or_else(|| "chunk hash upload overflow".into())
}

impl HalfScratch {
    #[cfg(test)]
    pub(super) fn backing_for_test(&self) -> &MetalTensor {
        &self.tensor
    }

    fn selected(ctx: &MetalContext) -> bool {
        metal::moe_grouped_tensor_selected(ctx)
    }

    fn logical_bytes(elements: usize) -> Result<u64> {
        u64::try_from(elements)
            .ok()
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| "half scratch byte overflow".into())
    }

    /// Scratch for up to `elements` activations when the tensor tier is
    /// selected on `ctx`; `None` otherwise (nothing would read it).
    pub(super) fn new(ctx: &MetalContext, elements: usize) -> Result<Option<Self>> {
        if !Self::selected(ctx) || elements == 0 {
            return Ok(None);
        }
        Self::logical_bytes(elements)?;
        Ok(Some(Self {
            tensor: allocate(ctx, elements, 1, GgmlType::F16)?,
            prefix_view: false,
        }))
    }

    /// Call on the original allocation, never on a previously narrowed view.
    pub(super) fn checked_prefix(&self, elements: usize) -> Result<Self> {
        op::require(!self.prefix_view, "half prefix requires original backing")?;
        Self::logical_bytes(elements)?;
        let backing_elements = self.tensor.shape.first().copied().unwrap_or(0);
        let backing_elements =
            usize::try_from(backing_elements).map_err(|_| "half scratch host extent overflow")?;
        Ok(Self {
            tensor: checked_scratch_view(
                &self.tensor,
                &[backing_elements, 1],
                &[elements],
                GgmlType::F16,
            )?,
            prefix_view: true,
        })
    }

    /// Convert the first `elements` of `x` (F32) into the scratch and
    /// return the half view. Serial encoders only: every projection that
    /// reads the view must be encoded before the next `convert`.
    pub(super) fn convert(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        x: &MetalTensor,
        elements: usize,
    ) -> std::result::Result<MetalTensor, metal::MetalError> {
        if enc.is_concurrent()
            || x.dtype != GgmlType::F32
            || elements == 0
            || (elements as u64) > x.n_elements()
            || (elements as u64) > self.tensor.n_elements()
        {
            return Err(metal::MetalError::BadShape {
                kernel: "v41_half_scratch",
                detail: format!(
                    "serial F32 input with {elements} elements within the {} element half scratch required",
                    self.tensor.n_elements()
                ),
            });
        }
        let src = x.view_subrange(0, vec![elements as u64]);
        let dst = self.tensor.view_subrange(0, vec![elements as u64]);
        metal::encode_scatter_offset_f32_to_f16(ctx, enc, &src, &dst, 0, elements)?;
        Ok(dst)
    }
}

crate::env_flag!(
    /// Route Q5_K dense prefill linears through a per-chunk F16 dequantized
    /// weight and the dense F16 tensor tile; `QWEN_V41_DENSE_PREWEIGHT=0`
    /// restores the Q5_K tensor tile for a same-binary A/B. Q8_0 stays on its
    /// wide tile.
    default_on dense_preweight_enabled,
    "QWEN_V41_DENSE_PREWEIGHT"
);

/// Below this row count the Q5_K tile wins; production prefill chunks are
/// 2,048/4,096 rows, while 1K appends stay on the tile.
pub(super) const DENSE_PREWEIGHT_MIN_ROWS: usize = 2_048;

/// Largest Q5_K dense linear the preweight scratch must hold. The per-group
/// output_a is smaller than output_b, so the q_b/output_b/shared extents
/// bound it.
pub(super) fn dense_preweight_elements(c: &Config) -> Result<usize> {
    let product = |parts: &[usize]| -> Result<usize> {
        parts.iter().try_fold(1usize, |acc, &n| {
            acc.checked_mul(n)
                .ok_or_else(|| "dense preweight extent overflow".into())
        })
    };
    let q_b = product(&[c.query_rank, c.heads, c.head_dim])?;
    let output_b = product(&[c.groups, c.output_rank, c.hidden])?;
    let shared = product(&[c.hidden, c.intermediate])?;
    Ok(q_b.max(output_b).max(shared))
}

pub(super) fn dense_preweight_bytes(c: &Config) -> Result<u64> {
    u64::try_from(dense_preweight_elements(c)?)
        .ok()
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| "dense preweight scratch overflow".into())
}

crate::env_flag!(
    /// Routes Q5_K dense rows through the dedicated 64x64 mat_mat tensor tile and its
    /// n16/n64 simdgroup variants; `QWEN_V41_Q5K_MATMAT=1` opts in. The dedicated path
    /// is cosine-gated (>= 0.999), not bitwise-equal to the generic dense tensor tile,
    /// so it cannot be the default while the Full/CED pair requires a bitwise endpoint.
    default_off q5k_matmat_enabled,
    "QWEN_V41_Q5K_MATMAT"
);

impl Linear<'_> {
    /// Eligibility uses logical operand lineage, not just the replay tile size.
    pub(super) fn tensor_rows_selected_with_arithmetic(
        &self,
        rows: usize,
        arithmetic: RowsArithmetic,
    ) -> bool {
        let mma = mma_linear_rows_selected(self.tensor.dtype, self.input, rows, arithmetic);
        mma && metal::moe_grouped_tensor_selected(self.ctx)
            && metal::moe_grouped_generic_layout(self.tensor.dtype)
                .is_some_and(|l| self.input.is_multiple_of(l.block_elems))
            && !(self.tensor.dtype == GgmlType::Q5_K && q5k_matmat_enabled())
    }

    pub(super) fn encode_rows(
        &self,
        enc: &KernelEncoder,
        x: &MetalTensor,
        y: &MetalTensor,
        rows: usize,
    ) -> std::result::Result<(), metal::MetalError> {
        self.encode_rows_with(enc, x, None, y, rows)
    }

    pub(super) fn encode_rows_with_arithmetic(
        &self,
        enc: &KernelEncoder,
        x: &MetalTensor,
        y: &MetalTensor,
        rows: usize,
        arithmetic: RowsArithmetic,
    ) -> std::result::Result<(), metal::MetalError> {
        self.encode_rows_with_arithmetic_and_half(enc, x, None, y, rows, arithmetic)
    }

    /// `encode_rows` with an optional half copy of `x` (`x_half`, the first
    /// `input * rows` elements already converted) that the dense tensor
    /// tile reads instead of re-casting `x`; ignored on every other route.
    pub(super) fn encode_rows_with(
        &self,
        enc: &KernelEncoder,
        x: &MetalTensor,
        x_half: Option<&MetalTensor>,
        y: &MetalTensor,
        rows: usize,
    ) -> std::result::Result<(), metal::MetalError> {
        self.encode_rows_with_arithmetic_and_half(enc, x, x_half, y, rows, RowsArithmetic::Ordinary)
    }

    pub(super) fn encode_rows_with_arithmetic_and_half(
        &self,
        enc: &KernelEncoder,
        x: &MetalTensor,
        x_half: Option<&MetalTensor>,
        y: &MetalTensor,
        rows: usize,
        arithmetic: RowsArithmetic,
    ) -> std::result::Result<(), metal::MetalError> {
        if rows == 0
            || x.dtype != GgmlType::F32
            || y.dtype != GgmlType::F32
            || self.input.checked_mul(rows).map(|n| n as u64) != Some(x.n_elements())
            || self.output.checked_mul(rows).map(|n| n as u64) != Some(y.n_elements())
        {
            return Err(metal::MetalError::BadShape {
                kernel: "v41_linear_rows",
                detail: "nonempty matching F32 row extents required".into(),
            });
        }
        let mma = mma_linear_rows_selected(self.tensor.dtype, self.input, rows, arithmetic);
        if mma {
            let x = x.view_bytes(0, vec![self.input as u64, rows as u64]);
            let y = y.view_bytes(0, vec![self.output as u64, rows as u64]);
            // The dense form of the Metal 4 tensor tile where the tier
            // exists: same half operands and F32 sums as the MMA tile.
            if self.tensor_rows_selected_with_arithmetic(rows, arithmetic) {
                let elements = (self.input * rows) as u64;
                let x_half = x_half.map(|h| {
                    if h.dtype == GgmlType::F16 && h.n_elements() >= elements {
                        Ok(h.view_bytes(0, vec![self.input as u64, rows as u64]))
                    } else {
                        Err(metal::MetalError::BadShape {
                            kernel: "v41_linear_rows",
                            detail: format!(
                                "half input must be F16 with >= {elements} elements, got {:?} x {}",
                                h.dtype,
                                h.n_elements()
                            ),
                        })
                    }
                });
                let x_half = x_half.transpose()?;
                return metal::encode_mat_mat_tensor_f32(
                    self.ctx,
                    enc,
                    &self.tensor,
                    x_half.as_ref().unwrap_or(&x),
                    &y,
                    self.input,
                    self.output,
                    rows,
                );
            }
            return crate::metal_forward::encode_mat_mat_dispatch_with_policy(
                self.ctx,
                enc,
                &self.tensor,
                &x,
                &y,
                self.input,
                self.output,
                rows,
                false,
            )
            .map_err(|error| match error {
                crate::metal_forward::MfError::Metal(error) => error,
                other => metal::MetalError::BadShape {
                    kernel: "v41_linear_rows",
                    detail: format!("mma projection: {other}"),
                },
            });
        }
        let grouped = metal::grouped_projection_supported(self.tensor.dtype)
            && self.input.is_multiple_of(32)
            && self
                .tensor
                .dtype
                .storage_layout()
                .is_some_and(|(block, _)| (self.input as u64).is_multiple_of(block));
        #[cfg(test)]
        let grouped = grouped && !ROWWISE_LINEAR_ROWS.with(|flag| flag.get());
        if grouped {
            return metal::encode_mat_mat_grouped_f32(
                self.ctx,
                enc,
                &self
                    .tensor
                    .view_bytes(0, vec![self.input as u64, self.output as u64, 1]),
                &x.view_bytes(0, vec![self.input as u64, rows as u64]),
                &y.view_bytes(0, vec![self.output as u64, rows as u64]),
                self.input,
                self.output,
                1,
                rows,
            );
        }
        for row in 0..rows {
            self.encode(
                enc,
                &x.view_subrange((row * self.input) as u64, vec![self.input as u64]),
                &y.view_subrange((row * self.output) as u64, vec![self.output as u64]),
            )?;
        }
        Ok(())
    }

    /// Q5_K dense preweight route: dequantize the weight once into `dense_weight`
    /// (F16, at least `input * output` elements), then run the dense F16 tensor
    /// tile. Returns false when the shape/dtype/tier does not qualify so the
    /// caller can use the ordinary tensor/Q5_K path.
    pub(super) fn encode_rows_with_arithmetic_and_half_preweight(
        &self,
        enc: &KernelEncoder,
        x_half: Option<&MetalTensor>,
        y: &MetalTensor,
        rows: usize,
        dense_weight: Option<&MetalTensor>,
    ) -> std::result::Result<bool, metal::MetalError> {
        if !dense_preweight_enabled()
            || enc.is_concurrent()
            || self.tensor.dtype != GgmlType::Q5_K
            || rows < DENSE_PREWEIGHT_MIN_ROWS
        {
            return Ok(false);
        }
        let (Some(x_half), Some(dense_weight)) = (x_half, dense_weight) else {
            return Ok(false);
        };
        let elements =
            self.input
                .checked_mul(self.output)
                .ok_or_else(|| metal::MetalError::BadShape {
                    kernel: "v41_dense_preweight",
                    detail: "weight extent overflow".into(),
                })?;
        if x_half.dtype != GgmlType::F16
            || y.dtype != GgmlType::F32
            || dense_weight.dtype != GgmlType::F16
            || dense_weight.n_elements() < elements as u64
            || x_half.n_elements() < (self.input * rows) as u64
            || y.n_elements() < (self.output * rows) as u64
        {
            return Ok(false);
        }
        let weight = dense_weight.view_subrange(0, vec![self.input as u64, self.output as u64]);
        metal::encode_v41_dequant_q5_k_f16(
            self.ctx,
            enc,
            &self.tensor,
            &weight,
            self.input,
            self.output,
        )?;
        let x = x_half.view_bytes(0, vec![self.input as u64, rows as u64]);
        let y = y.view_bytes(0, vec![self.output as u64, rows as u64]);
        metal::encode_mat_mat_tensor_f32(
            self.ctx,
            enc,
            &weight,
            &x,
            &y,
            self.input,
            self.output,
            rows,
        )?;
        Ok(true)
    }
}

#[cfg(test)]
// Test-only token-major reference with batched Engram projections.
fn run_hybrid(
    model: &Forward<'_>,
    state: &mut State,
    tokens: &[u32],
    cancel: Option<&AtomicBool>,
    chunk_rows: usize,
) -> Result<Vec<f32>> {
    op::require(!tokens.is_empty(), "empty prefill suffix")?;
    op::require(
        (1..=MAX_CHUNK_ROWS).contains(&chunk_rows),
        "invalid prefill chunk rows",
    )?;
    op::require(
        state.is_ready(),
        "sequence is provisional or poisoned by partial execution",
    )?;
    op::require(
        Arc::ptr_eq(&state.identity, &model.identity),
        "state belongs to another model",
    )?;
    op::require(state.gpu_step.is_some(), "hybrid prefill requires GPU step")?;
    op::require(
        tokens.len() <= state.capacity().saturating_sub(state.position()),
        "prefill exceeds state capacity",
    )?;
    op::require(
        tokens.iter().all(|&t| (t as usize) < model.token_map.len()),
        "token outside vocabulary",
    )?;
    let mut logits = Vec::new();
    let final_position = state.position() + tokens.len();
    let rows = tokens.len().min(chunk_rows);
    let mut extra_metal = 0u64;
    let mut extra_cpu = 0u64;
    for hash in &model.engram_hash.layers {
        let linear = &model.layers[hash.layer]
            .engram
            .as_ref()
            .ok_or("missing Engram")?
            .projection;
        for width in [linear.input, linear.output] {
            let bytes = width
                .checked_mul(rows)
                .and_then(|n| n.checked_mul(4))
                .ok_or("prefill scratch extent overflow")? as u64;
            let priced = model
                .ctx
                .price_shared_buffer_upper(bytes)
                .map_err(|e| e.to_string())?;
            extra_metal = extra_metal
                .checked_add(priced.priced_upper_bytes)
                .ok_or("prefill scratch budget overflow")?;
            extra_cpu = extra_cpu.max(bytes);
        }
    }
    let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
        extra_metal,
        extra_cpu,
        0,
        model.ctx.memory_signals(),
        true,
    );
    if let Some(reason) = admission.refusal() {
        return Err(StateError::MemoryAdmissionDenied(reason).to_string());
    }
    for chunk in tokens.chunks(chunk_rows) {
        cancelled(cancel)?;
        let mut addresses = state.addresses;
        let mut hashes = Vec::with_capacity(chunk.len());
        let mut next_addresses = Vec::with_capacity(chunk.len());
        for &token in chunk {
            let (rows, next) = addresses
                .step_with_config(
                    engram::Token::Compressed(model.token_map[token as usize]),
                    &model.engram_hash,
                )
                .map_err(|e| format!("{e:?}"))?;
            hashes.push(rows);
            next_addresses.push(next);
            addresses = next;
        }
        let mut projections = Vec::new();
        let mut inputs = Vec::new();
        for hash in &model.engram_hash.layers {
            let id = hash.layer;
            let linear = &model.layers[id]
                .engram
                .as_ref()
                .ok_or("missing Engram")?
                .projection;
            let values = model.engram_rows.decode_tokens(id, &hashes)?;
            inputs.push(upload(
                model.ctx,
                &values,
                vec![linear.input as u64, chunk.len() as u64],
            )?);
            projections.push((
                id,
                MetalTensor::zeros_dtype_unstaged(
                    model.ctx,
                    vec![linear.output as u64, chunk.len() as u64],
                    GgmlType::F32,
                )
                .map_err(|e| e.to_string())?,
            ));
        }
        if !projections.is_empty() {
            submit(
                model.ctx,
                "prefill_engram_rows",
                [chunk.len(), projections.len(), 0, 0],
                |enc| {
                    for ((id, output), input) in projections.iter().zip(&inputs) {
                        model.layers[*id]
                            .engram
                            .as_ref()
                            .unwrap()
                            .projection
                            .encode_rows(enc, input, output, chunk.len())?;
                    }
                    Ok(())
                },
            )?;
        }
        for (row, &token) in chunk.iter().enumerate() {
            cancelled(cancel)?;
            let position = state.position();
            let prepared = projections
                .iter()
                .map(|(id, tensor)| {
                    let width = model.layers[*id].engram.as_ref().unwrap().projection.output;
                    (
                        *id,
                        tensor.view_subrange((row * width) as u64, vec![width as u64]),
                    )
                })
                .collect::<Vec<_>>();
            state.phase = StatePhase::Poisoned;
            logits = state.gpu_step.as_mut().unwrap().run_prepared(
                model,
                position,
                token as usize,
                &hashes[row],
                cancel,
                None,
                None,
                position + 1 == final_position,
                Some(&prepared),
            )?;
            state.addresses = next_addresses[row];
            state.phase = StatePhase::Ready;
        }
    }
    Ok(logits)
}

pub(super) fn allocate(
    ctx: &MetalContext,
    width: usize,
    rows: usize,
    dtype: GgmlType,
) -> Result<MetalTensor> {
    let tensor = MetalTensor::zeros_dtype_unstaged(ctx, vec![width as u64, rows as u64], dtype)
        .map_err(|e| e.to_string())?;
    #[cfg(test)]
    tests::lifetime::allocated(ctx, &tensor);
    Ok(tensor)
}

pub(super) fn row(t: &MetalTensor, width: usize, row: usize) -> MetalTensor {
    t.view_subrange((row * width) as u64, vec![width as u64])
}

/// `split` keeps the prefill mHC pre projection (scalar F32 grouped kernel)
/// and the projected RMS as two passes over the residual; otherwise the F32
/// [4*hidden -> 24] projection reads each row once per threadgroup and folds
/// the projected RMS (`kernel_deepseek_v41_hc_projection_rows`; A/B lever).
const HC_ROWS_ENV: &str = "QWEN_V41_HC_ROWS";

#[cfg(test)]
thread_local! {
    pub(super) static HC_ROWS_FUSED_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HcProjRowsArgs {
    rows: u32,
    k: u32,
    eps: f32,
}

fn hc_projection_rows_fused(projection: &Linear<'_>, c: &Config, x: &MetalTensor) -> bool {
    #[cfg(test)]
    let forced = HC_ROWS_FUSED_OVERRIDE.with(|o| o.get());
    #[cfg(not(test))]
    let forced: Option<bool> = None;
    let enabled = forced.unwrap_or_else(|| {
        static FUSED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *FUSED.get_or_init(|| std::env::var(HC_ROWS_ENV).as_deref() != Ok("split"))
    });
    enabled
        && projection.tensor.dtype == GgmlType::F32
        && projection.output == 24
        && projection.input == 4 * c.hidden
        && projection.input.is_multiple_of(16)
        && projection.tensor.offset.is_multiple_of(16)
        && x.dtype == GgmlType::F32
        && x.offset.is_multiple_of(16)
}

pub(super) struct HcRows {
    pub(super) rows: usize,
    prefix_view: bool,
    raw: MetalTensor,
    pub(super) pre: MetalTensor,
    pub(super) post: MetalTensor,
    pub(super) comb: MetalTensor,
    collapsed: MetalTensor,
    pub(super) normalized: MetalTensor,
}

impl HcRows {
    pub(super) fn widths(c: &Config) -> Vec<usize> {
        vec![24, 4, 4, 16, c.hidden, c.hidden]
    }
    pub(super) fn new(ctx: &MetalContext, c: &Config, rows: usize) -> Result<Self> {
        let f = |w| allocate(ctx, w, rows, GgmlType::F32);
        Ok(Self {
            rows,
            prefix_view: false,
            raw: f(24)?,
            pre: f(4)?,
            post: f(4)?,
            comb: f(16)?.view_bytes(0, vec![4, 4, rows as u64]),
            collapsed: f(c.hidden)?,
            normalized: f(c.hidden)?,
        })
    }

    pub(super) fn checked_prefix(&self, c: &Config, rows: usize) -> Result<Self> {
        op::require(!self.prefix_view, "HC prefix requires original backing")?;
        checked_scratch_rows(self.rows, rows)?;
        let f = |tensor, width| {
            checked_scratch_view(tensor, &[width, self.rows], &[width, rows], GgmlType::F32)
        };
        Ok(Self {
            rows,
            prefix_view: true,
            raw: f(&self.raw, 24)?,
            pre: f(&self.pre, 4)?,
            post: f(&self.post, 4)?,
            comb: checked_scratch_view(
                &self.comb,
                &[4, 4, self.rows],
                &[4, 4, rows],
                GgmlType::F32,
            )?,
            collapsed: f(&self.collapsed, c.hidden)?,
            normalized: f(&self.normalized, c.hidden)?,
        })
    }
    pub(super) fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        c: &Config,
        projection: &Linear<'_>,
        constants: &[MetalTensor],
        x: &MetalTensor,
        incoming: &MetalTensor,
    ) -> std::result::Result<(), metal::MetalError> {
        self.encode_with_arithmetic(
            ctx,
            enc,
            c,
            projection,
            constants,
            x,
            incoming,
            RowsArithmetic::Ordinary,
        )
    }

    pub(super) fn encode_with_arithmetic(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        c: &Config,
        projection: &Linear<'_>,
        constants: &[MetalTensor],
        x: &MetalTensor,
        incoming: &MetalTensor,
        arithmetic: RowsArithmetic,
    ) -> std::result::Result<(), metal::MetalError> {
        if hc_projection_rows_fused(projection, c, x) {
            // One pass over the residual: projection + projected RMS.
            enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_hc_projection_rows")?);
            enc.set_bytes(
                0,
                &HcProjRowsArgs {
                    rows: self.rows as u32,
                    k: (4 * c.hidden) as u32,
                    eps: c.norm_eps,
                },
            );
            enc.set_tensor(1, &projection.tensor);
            enc.set_tensor(2, x);
            enc.set_tensor(3, &self.raw);
            enc.dispatch(
                objc2_metal::MTLSize {
                    width: self.rows.div_ceil(4),
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            projection.encode_rows_with_arithmetic(enc, x, &self.raw, self.rows, arithmetic)?;
            enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_projected_rms")?);
            enc.set_bytes(0, &((4 * c.hidden) as u32));
            enc.set_bytes(1, &c.norm_eps);
            enc.set_tensor(2, x);
            enc.set_tensor(3, &self.raw);
            enc.dispatch(
                objc2_metal::MTLSize {
                    width: self.rows,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        metal::encode_mhc4_controls_rows(
            ctx,
            enc,
            self.rows,
            c.hc_eps,
            &self.raw,
            &constants[0],
            &constants[1],
            &self.pre,
            &self.post,
            &self.comb,
        )?;
        self.normalize_input(ctx, enc, c, &constants[2], x, incoming)
    }

    pub(super) fn normalize_input(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        c: &Config,
        norm: &MetalTensor,
        x: &MetalTensor,
        incoming: &MetalTensor,
    ) -> std::result::Result<(), metal::MetalError> {
        metal::encode_mhc4_collapse_rows(
            ctx,
            enc,
            c.hidden,
            self.rows,
            x,
            incoming,
            &self.collapsed,
        )?;
        metal::encode_rms_norm_mul_rows_f32(
            ctx,
            enc,
            &self.collapsed,
            norm,
            &self.normalized,
            self.rows,
            c.hidden,
            c.norm_eps,
        )
    }
}

pub(super) fn run(
    model: &Forward<'_>,
    state: &mut State,
    tokens: &[u32],
    cancel: Option<&AtomicBool>,
    chunk_rows: usize,
    with_logits: bool,
) -> std::result::Result<Vec<f32>, StateError> {
    run_captured(model, state, tokens, cancel, chunk_rows, with_logits, None)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScratchPolicy {
    Reuse,
    PerChunk,
}

fn scratch_policy() -> ScratchPolicy {
    match std::env::var_os("QWEN_V41_CHUNK_SCRATCH_REUSE")
        .as_deref()
        .and_then(|value| value.to_str())
    {
        Some("0") | Some("off") => ScratchPolicy::PerChunk,
        Some("alternate") => {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            if NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 2 == 0 {
                ScratchPolicy::Reuse
            } else {
                ScratchPolicy::PerChunk
            }
        }
        _ => ScratchPolicy::Reuse,
    }
}

pub(super) fn run_captured(
    model: &Forward<'_>,
    state: &mut State,
    tokens: &[u32],
    cancel: Option<&AtomicBool>,
    chunk_rows: usize,
    with_logits: bool,
    mut capture: Option<&mut dyn capture::Capture>,
) -> std::result::Result<Vec<f32>, StateError> {
    op::require(!tokens.is_empty(), "empty prefill suffix")?;
    op::require(
        (1..=MAX_CHUNK_ROWS).contains(&chunk_rows),
        "invalid prefill chunk rows",
    )?;
    op::require(
        state.is_ready(),
        "sequence is provisional or poisoned by partial execution",
    )?;
    op::require(
        Arc::ptr_eq(&state.identity, &model.identity),
        "state belongs to another model",
    )?;
    op::require(state.gpu_step.is_some(), "prefill requires GPU step")?;
    op::require(
        tokens.len() <= state.capacity().saturating_sub(state.position()),
        "prefill exceeds state capacity",
    )?;
    op::require(
        tokens.iter().all(|&t| (t as usize) < model.token_map.len()),
        "token outside vocabulary",
    )?;
    cancelled(cancel)?;
    let rows = tokens.len().min(chunk_rows);
    if rows > 1 {
        // One summary per process: which routed-expert route each layer takes
        // at the production chunk width (artifact-dependent; see PERF-LOG).
        static ROUTE_CENSUS_ONCE: std::sync::Once = std::sync::Once::new();
        ROUTE_CENSUS_ONCE.call_once(|| {
            let census =
                super::ffn_gpu::routed_route_census(&model.config, &model.layers, rows);
            tracing::info!(
                target: "qwen_diag",
                "deepseek41: prefill rows={rows} routed experts: half_grouped={} f32_grouped={} native_rows={} per_row={}",
                census[0],
                census[1],
                census[2],
                census[3]
            );
        });
    }
    let cost = model.prefill_cost(state.capacity(), rows)?;
    planning::Cost {
        metal: cost.metal,
        cpu: cost
            .cpu
            .checked_add(state.caller_cpu_reserve)
            .ok_or("prefill reserve overflow")?,
    }
    .admit(model.ctx.memory_signals())?;
    op::require(
        state
            .prefill_rows
            .is_none_or(|selected| selected == chunk_rows),
        "prefill geometry is fixed for this request",
    )?;
    state.prefill_rows = Some(chunk_rows);
    state.last_prefill = None;
    let end = state.position() + tokens.len();
    let policy = scratch_policy();
    let mut scratch = if policy == ScratchPolicy::Reuse && rows > 1 {
        let span = profile::Wall("prefill_scratch_prepare", [0; 4], profile::start());
        let scratch = state
            .gpu_step
            .as_ref()
            .unwrap()
            .new_chunk_scratch(model, rows)?;
        drop(span);
        Some(scratch)
    } else {
        None
    };
    let mut logits = Vec::new();
    for chunk in tokens.chunks(chunk_rows) {
        // Admission prices one chunk, not the whole suffix. Drop its completed
        // commands and their resource autoreleases before allocating the next.
        logits = objc2::rc::autoreleasepool(|_| -> std::result::Result<Vec<f32>, StateError> {
            cancelled(cancel)?;
            let start = state.position();
            let last = with_logits && start + chunk.len() == end;
            if chunk.len() == 1 {
                return Ok(model.execute_token(
                    state,
                    chunk[0],
                    cancel,
                    None,
                    capture
                        .as_mut()
                        .map(|sink| &mut **sink as &mut dyn capture::Capture),
                    last,
                )?);
            }
            let per_chunk = if policy == ScratchPolicy::PerChunk {
                let span = profile::Wall("prefill_scratch_prepare", [0; 4], profile::start());
                let scratch = state
                    .gpu_step
                    .as_ref()
                    .unwrap()
                    .new_chunk_scratch(model, chunk.len())?;
                drop(span);
                Some(scratch)
            } else {
                None
            };
            let chunk_scratch = match &per_chunk {
                Some(scratch) => scratch,
                None => {
                    if scratch.as_ref().is_none_or(|s| s.rows != chunk.len()) {
                        // Drop the max-row set before allocating the tail set so
                        // the peak stays one scratch.
                        scratch = None;
                        let span =
                            profile::Wall("prefill_scratch_prepare", [0; 4], profile::start());
                        let tail = state
                            .gpu_step
                            .as_ref()
                            .unwrap()
                            .new_chunk_scratch(model, chunk.len())?;
                        drop(span);
                        scratch = Some(tail);
                    }
                    scratch.as_ref().unwrap()
                }
            };
            let hash_span = profile::Wall("prefill_engram_hash", [0; 4], profile::start());
            let mut next = state.addresses;
            let mut hashes = Vec::with_capacity(chunk.len());
            for &token in chunk {
                let (rows, addresses) = next
                    .step_with_config(
                        engram::Token::Compressed(model.token_map[token as usize]),
                        &model.engram_hash,
                    )
                    .map_err(|e| format!("{e:?}"))?;
                hashes.push(rows);
                next = addresses;
            }
            drop(hash_span);
            state.phase = StatePhase::Poisoned;
            let logits = state.gpu_step.as_mut().unwrap().run_chunk(
                model,
                start,
                chunk,
                &hashes,
                cancel,
                last,
                capture
                    .as_mut()
                    .map(|sink| &mut **sink as &mut dyn capture::Capture),
                chunk_scratch,
            )?;
            state.addresses = next;
            state.phase = StatePhase::Ready;
            Ok(logits)
        })?;
        #[cfg(test)]
        tests::lifetime::completed_chunk(model.ctx, state, chunk.len());
    }
    state.last_prefill = Some(PrefillGeometry {
        start: end - tokens.len(),
        tokens: tokens.len(),
        selected_rows: chunk_rows,
    });
    Ok(logits)
}
