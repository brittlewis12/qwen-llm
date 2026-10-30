//! Bounded dense Half-operand TensorOps route. The KDA owner may supply an
//! already-converted input; no expert/MLA routing or allocations here. Q8 wide
//! defaults on for eligible M5 Fast/Half prefill; 0 restores the narrow tile.

use super::packed::{Arith, FastPrecision, PackedLineage, StageMode};
use crate::metal::checks::{
    bad_shape, check_alignment, check_disjoint, check_physical, check_tensor, require_serial,
    to_u32,
};
use crate::metal::{self, KernelEncoder, MetalContext, MetalError, MetalTensor};
use crate::tensor::GgmlType;
use objc2_metal::{MTLCommandBuffer, MTLComputePipelineState, MTLDevice, MTLResource};

const KERNEL: &str = "glm5_dense_tensor_prefill";
const MIN_ROWS: usize = 1024;
const DEFAULT_ENABLED: bool = true;
const THREADGROUP_BYTES: usize = 8192;

crate::env_flag!(
    /// Default-on GLM dense Q8_0 64-slot TensorOps prefill on eligible M5 Fast/Half; 0 restores the previous narrow tile. Does not bypass Half/Fast, row, tier, binding, or global rollback guards.
    default_on q8_wide_enabled,
    "QWEN_GLM53_DENSE_Q8_WIDE"
);

#[cfg(test)]
thread_local! {
    static OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static Q8_WIDE_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Independent wide/narrow control; never enables the enclosing tensor route.
#[cfg(test)]
pub(crate) struct Q8WideScope {
    previous: Option<bool>,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[cfg(test)]
impl Q8WideScope {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            previous: Q8_WIDE_OVERRIDE.with(|flag| flag.replace(Some(enabled))),
            _thread: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
impl Drop for Q8WideScope {
    fn drop(&mut self) {
        Q8_WIDE_OVERRIDE.with(|flag| flag.set(self.previous));
    }
}

fn wide_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = Q8_WIDE_OVERRIDE.with(std::cell::Cell::get) {
        return enabled;
    }
    q8_wide_enabled()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tile {
    Narrow,
    Q8Wide,
}

impl Tile {
    // Called only after the existing GLM eligibility checks.
    fn selected(dtype: GgmlType) -> Self {
        if dtype == GgmlType::Q8_0 && wide_enabled() {
            Self::Q8Wide
        } else {
            Self::Narrow
        }
    }

    #[cfg(test)]
    fn pipeline_name(self, dtype: GgmlType) -> Option<String> {
        self.pipeline_for(dtype, metal::TensorActivation::F32)
    }

    fn pipeline_for(self, dtype: GgmlType, activation: metal::TensorActivation) -> Option<String> {
        match self {
            Self::Narrow => metal::mat_mat_tensor_pipeline_name(dtype, activation),
            Self::Q8Wide => metal::mat_mat_tensor_wide_pipeline_name(dtype, activation),
        }
    }

    fn threadgroup_bytes(self) -> usize {
        match self {
            Self::Narrow => THREADGROUP_BYTES,
            Self::Q8Wide => 16384,
        }
    }
}

/// Same-thread, panic-safe, nestable A/B control. Cannot bypass arithmetic,
/// geometry, global rollback, tier, or binding checks.
#[cfg(test)]
pub(crate) struct Scope {
    previous: Option<bool>,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[cfg(test)]
impl Scope {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            previous: OVERRIDE.with(|flag| flag.replace(Some(enabled))),
            _thread: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
impl Drop for Scope {
    fn drop(&mut self) {
        OVERRIDE.with(|flag| flag.set(self.previous));
    }
}

#[cfg(test)]
pub(crate) fn with_override<R>(enabled: bool, body: impl FnOnce() -> R) -> R {
    let _scope = Scope::new(enabled);
    body()
}

fn enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = OVERRIDE.with(std::cell::Cell::get) {
        return enabled;
    }
    DEFAULT_ENABLED
}

#[derive(Debug, PartialEq, Eq)]
struct Geometry {
    weight_bytes: usize,
    input_elements: usize,
    output_elements: usize,
}

fn geometry(dtype: GgmlType, k: usize, m: usize, n: usize) -> Result<Geometry, MetalError> {
    for (name, value) in [("K", k), ("M", m), ("N", n)] {
        to_u32(KERNEL, value, name)?;
        if value == 0 {
            return Err(bad_shape(KERNEL, format!("{name} must be positive")));
        }
    }
    // The shared shader uses signed int counts, slot limits and row indices.
    if m > i32::MAX as usize || n > i32::MAX as usize {
        return Err(bad_shape(KERNEL, "M/N exceed signed tensor-tile indexing"));
    }
    let (block, bytes) = dtype
        .storage_layout()
        .filter(|_| matches!(dtype, GgmlType::Q6_K | GgmlType::Q8_0))
        .ok_or_else(|| bad_shape(KERNEL, "only native Q6_K/Q8_0 are eligible"))?;
    if !(k as u64).is_multiple_of(block) || !k.is_multiple_of(32) {
        return Err(bad_shape(
            KERNEL,
            "K must cover whole storage and 32-element tiles",
        ));
    }
    let row_bytes = (k / block as usize)
        .checked_mul(bytes as usize)
        .ok_or_else(|| bad_shape(KERNEL, "weight row-byte overflow"))?;
    to_u32(KERNEL, row_bytes, "weight row-byte stride")?;
    let product = |a: usize, b: usize, name: &str| {
        a.checked_mul(b)
            .ok_or_else(|| bad_shape(KERNEL, format!("{name} size overflow")))
    };
    let weight_bytes = product(row_bytes, m, "weight")?;
    let input_elements = product(k, n, "input")?;
    let output_elements = product(m, n, "output")?;
    product(input_elements, size_of::<f32>(), "input bytes")?;
    product(output_elements, size_of::<f32>(), "output bytes")?;
    Ok(Geometry {
        weight_bytes,
        input_elements,
        output_elements,
    })
}

/// Resolved mode is load-bearing: stage-local Exact/F32 diagnostics still win
/// even in an ordinary Half session. Short packets never reach new checks.
pub(super) fn eligible(
    arith: Arith,
    mode: StageMode,
    dtype: GgmlType,
    k: usize,
    m: usize,
    n: usize,
) -> Result<bool, MetalError> {
    if !enabled()
        || arith.lineage != PackedLineage::Fast
        || arith.precision != FastPrecision::Half
        || mode != StageMode::Fast
        || !matches!(dtype, GgmlType::Q6_K | GgmlType::Q8_0)
        || n < MIN_ROWS
    {
        return Ok(false);
    }
    geometry(dtype, k, m, n)?;
    Ok(true)
}

fn capacity_supported(
    tile: Tile,
    width: usize,
    threads: usize,
    static_bytes: usize,
    limit: usize,
) -> bool {
    width == 32
        && threads >= 128
        && static_bytes
            .checked_add(tile.threadgroup_bytes())
            .is_some_and(|total| total <= limit)
}

/// Resolve/cache the pipeline without touching the encoder. Pipeline failures
/// are errors, not permission to hide a broken candidate behind the baseline.
fn prepare(ctx: &MetalContext, dtype: GgmlType, tile: Tile) -> Result<bool, MetalError> {
    prepare_activation(ctx, dtype, tile, metal::TensorActivation::F32)
}

fn prepare_activation(
    ctx: &MetalContext,
    dtype: GgmlType,
    tile: Tile,
    activation: metal::TensorActivation,
) -> Result<bool, MetalError> {
    let name = tile
        .pipeline_for(dtype, activation)
        .ok_or_else(|| bad_shape(KERNEL, "no dense tensor pipeline"))?;
    let p = ctx.tensor_pipeline(&name)?;
    Ok(capacity_supported(
        tile,
        p.threadExecutionWidth(),
        p.maxTotalThreadsPerThreadgroup(),
        p.staticThreadgroupMemoryLength(),
        ctx.device.maxThreadgroupMemoryLength(),
    ))
}

fn check_binding_extent(
    shape: &[u64],
    dtype: GgmlType,
    expected: usize,
    name: &str,
) -> Result<(), MetalError> {
    let (_, bytes) = metal::checked_ggml_shape_bytes(shape, dtype)?;
    if bytes != expected {
        return Err(bad_shape(KERNEL, format!("{name} byte extent mismatch")));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_bindings(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    n: usize,
) -> Result<(), MetalError> {
    check_bindings_for(ctx, enc, weight, x, y, k, m, n, GgmlType::F32)
}

fn check_bindings_for(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    n: usize,
    activation: GgmlType,
) -> Result<(), MetalError> {
    let g = geometry(weight.dtype, k, m, n)?;
    require_serial(KERNEL, enc)?;
    if enc.ended {
        return Err(bad_shape(KERNEL, "encoder has ended"));
    }
    let device = ctx.device.registryID();
    if enc.parent_command_buffer().device().registryID() != device {
        return Err(bad_shape(KERNEL, "encoder belongs to another device"));
    }
    // Check sizes before helpers that call n_bytes(), so malformed public
    // tensor metadata returns an error rather than panicking on overflow.
    for (t, name, expected) in [
        (weight, "weight", g.weight_bytes),
        (
            x,
            "input",
            g.input_elements * if activation == GgmlType::F16 { 2 } else { 4 },
        ),
        (y, "output", g.output_elements * size_of::<f32>()),
    ] {
        check_binding_extent(&t.shape, t.dtype, expected, name)?;
        if t.buffer.device().registryID() != device {
            return Err(bad_shape(
                KERNEL,
                format!("{name} belongs to another device"),
            ));
        }
    }
    check_tensor(
        KERNEL,
        weight,
        weight.dtype,
        &[k as u64, m as u64],
        false,
        "weight",
    )?;
    // Packed activations may retain multiple leading dimensions (heads).
    check_tensor(KERNEL, x, activation, &x.shape, false, "input")?;
    check_tensor(KERNEL, y, GgmlType::F32, &y.shape, true, "output")?;
    check_physical(KERNEL, weight, 16, false, "weight")?;
    check_alignment(KERNEL, x, 16, "float2x4 input")?;
    check_disjoint(KERNEL, y, &[(weight, "weight"), (x, "input")])?;
    Ok(())
}

/// Frozen tile and bindings. Both activation twins are resolved before any
/// conversion; dispatch never re-selects policy (including Q8 wide rollback).
pub(super) struct PreparedHalf {
    tile: Tile,
    weight: MetalTensor,
    x: MetalTensor,
    y: MetalTensor,
    k: usize,
    m: usize,
    n: usize,
}

impl PreparedHalf {
    pub(super) fn encode(&self, ctx: &MetalContext, enc: &KernelEncoder) -> Result<(), MetalError> {
        match self.tile {
            Tile::Narrow => metal::encode_mat_mat_tensor_f32_narrow(
                ctx,
                enc,
                &self.weight,
                &self.x,
                &self.y,
                self.k,
                self.m,
                self.n,
            ),
            Tile::Q8Wide => metal::encode_mat_mat_tensor_f32_wide(
                ctx,
                enc,
                &self.weight,
                &self.x,
                &self.y,
                self.k,
                self.m,
                self.n,
            ),
        }
    }
}

pub(super) fn prepare_half(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    arith: Arith,
    mode: StageMode,
    weight: &MetalTensor,
    original: &MetalTensor,
    half: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    n: usize,
) -> Result<Option<PreparedHalf>, MetalError> {
    if !eligible(arith, mode, weight.dtype, k, m, n)? || !metal::moe_grouped_tensor_selected(ctx) {
        return Ok(None);
    }
    check_bindings(ctx, enc, weight, original, y, k, m, n)?;
    check_bindings_for(ctx, enc, weight, half, y, k, m, n, GgmlType::F16)?;
    check_disjoint(
        KERNEL,
        half,
        &[
            (weight, "weight"),
            (original, "original input"),
            (y, "output"),
        ],
    )?;
    let tile = Tile::selected(weight.dtype);
    let original_supported = prepare(ctx, weight.dtype, tile)?;
    let half_supported = prepare_activation(ctx, weight.dtype, tile, metal::TensorActivation::F16)?;
    if !original_supported || !half_supported {
        return Ok(None);
    }
    Ok(Some(PreparedHalf {
        tile,
        weight: weight.clone(),
        x: half.clone(),
        y: y.clone(),
        k,
        m,
        n,
    }))
}

/// False means a policy/capability decline before mutation. Invalid bindings,
/// overflow and pipeline errors propagate. After dispatch there is no fallback.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    arith: Arith,
    mode: StageMode,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    n: usize,
) -> Result<bool, MetalError> {
    if !eligible(arith, mode, weight.dtype, k, m, n)? || !metal::moe_grouped_tensor_selected(ctx) {
        return Ok(false);
    }
    check_bindings(ctx, enc, weight, x, y, k, m, n)?;
    let tile = Tile::selected(weight.dtype);
    if !prepare(ctx, weight.dtype, tile)? {
        return Ok(false);
    }
    match tile {
        Tile::Narrow => metal::encode_mat_mat_tensor_f32_narrow(ctx, enc, weight, x, y, k, m, n)?,
        Tile::Q8Wide => metal::encode_mat_mat_tensor_f32_wide(ctx, enc, weight, x, y, k, m, n)?,
    }
    Ok(true)
}

#[cfg(test)]
#[path = "tests/tensor_prefill.rs"]
mod tests;
