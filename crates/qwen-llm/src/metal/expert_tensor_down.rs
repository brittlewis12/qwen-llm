//! Checked, prepared adapter for the existing IQ3_S/Q5_K N32 tensor down tiles.
//! Preparation never dispatches or mutates bindings; encode never falls back.

use super::checks::{
    bad_shape, check_alignment, check_disjoint, check_expert_bank, check_tensor, require_serial,
    to_u32,
};
use super::*;

const KERNEL: &str = "expert_tensor_down";
const DYNAMIC_BYTES: usize = 8192;

#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) enum PreflightFault {
    MissingPipeline,
    Capacity,
}

#[cfg(test)]
thread_local! {
    static FAULT: std::cell::Cell<Option<PreflightFault>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) struct FaultScope {
    previous: Option<PreflightFault>,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[cfg(test)]
impl FaultScope {
    pub(crate) fn new(fault: PreflightFault) -> Self {
        Self {
            previous: FAULT.with(|flag| flag.replace(Some(fault))),
            _thread: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
impl Drop for FaultScope {
    fn drop(&mut self) {
        FAULT.with(|flag| flag.set(self.previous));
    }
}

fn product(a: usize, b: usize, name: &str) -> Result<usize, MetalError> {
    a.checked_mul(b)
        .ok_or_else(|| bad_shape(KERNEL, format!("{name} size overflow")))
}

fn positive_i32(value: usize, name: &str) -> Result<(), MetalError> {
    if value == 0 || value > i32::MAX as usize {
        return Err(bad_shape(KERNEL, format!("{name} must fit positive i32")));
    }
    Ok(())
}

fn bank_bytes(dtype: GgmlType, k: usize, m: usize, e: usize) -> Result<usize, MetalError> {
    let (block, bytes) = dtype
        .storage_layout()
        .ok_or_else(|| bad_shape(KERNEL, "unknown expert storage layout"))?;
    if k == 0 || !(k as u64).is_multiple_of(block) || !k.is_multiple_of(32) {
        return Err(bad_shape(
            KERNEL,
            "K must cover whole storage and 32-element tiles",
        ));
    }
    let row = product(k / block as usize, bytes as usize, "weight row")?;
    to_u32(KERNEL, row, "weight row-byte stride")?;
    // The shaders widen before multiplying expert strides, but the host's
    // element and byte products must also be representable.
    product(product(k, m, "expert elements")?, e, "bank elements")?;
    product(product(row, m, "expert bytes")?, e, "bank bytes")
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    weight_bytes: usize,
    inner_bytes: usize,
    bucket_bytes: usize,
    out_bytes: usize,
    slots: usize,
}

pub(crate) fn geometry(
    dtype: GgmlType,
    k: usize,
    m: usize,
    e: usize,
    top_k: usize,
    rows: usize,
) -> Result<Geometry, MetalError> {
    pipeline_name(dtype)?;
    for (name, value) in [("K", k), ("M", m), ("experts", e), ("rows", rows)] {
        positive_i32(value, name)?;
    }
    if top_k == 0 || top_k > ROUTE_MAX_TOP_K {
        return Err(bad_shape(KERNEL, "top-k must be in 1..=16"));
    }
    let slots = product(rows, top_k, "slots")?;
    positive_i32(slots, "slot limit")?;
    Ok(Geometry {
        weight_bytes: bank_bytes(dtype, k, m, e)?,
        inner_bytes: product(product(k, slots, "inner elements")?, 4, "inner bytes")?,
        bucket_bytes: product(product(e, rows, "bucket elements")?, 4, "bucket bytes")?,
        out_bytes: product(product(m, slots, "output elements")?, 4, "output bytes")?,
        slots,
    })
}

fn check_encoder(ctx: &MetalContext, enc: &KernelEncoder) -> Result<(), MetalError> {
    require_serial(KERNEL, enc)?;
    if enc.ended {
        return Err(bad_shape(KERNEL, "encoder has ended"));
    }
    if enc.parent_command_buffer().device().registryID() != ctx.device.registryID() {
        return Err(bad_shape(KERNEL, "encoder belongs to another device"));
    }
    Ok(())
}

fn check_extent(
    ctx: &MetalContext,
    tensor: &MetalTensor,
    expected: usize,
    name: &str,
) -> Result<(), MetalError> {
    // Must precede check_tensor/check_expert_bank/overlaps and all legacy
    // n_elements()/n_bytes() calls, including flattening the composition.
    let (_, bytes) = checked_ggml_shape_bytes(&tensor.shape, tensor.dtype)?;
    if bytes != expected {
        return Err(bad_shape(KERNEL, format!("{name} byte extent mismatch")));
    }
    if tensor.buffer.device().registryID() != ctx.device.registryID() {
        return Err(bad_shape(
            KERNEL,
            format!("{name} belongs to another device"),
        ));
    }
    Ok(())
}

/// Overflow/device preflight for *every* composition binding before the
/// established grouped shape/physical/alias checks invoke legacy size helpers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn preflight_grouped(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    b: &GroupedExperts<'_>,
    hidden: usize,
    ffn: usize,
    experts: usize,
    top_k: usize,
    rows: usize,
    clamp: f32,
) -> Result<(), MetalError> {
    let g = geometry(b.down_bank.dtype, ffn, hidden, experts, top_k, rows)?;
    check_encoder(ctx, enc)?;
    if !clamp.is_finite() || clamp <= 0.0 {
        return Err(bad_shape(KERNEL, "clamp must be finite and positive"));
    }
    let input = product(product(hidden, rows, "input elements")?, 4, "input bytes")?;
    for (t, bytes, name) in [
        (
            b.gate_bank,
            bank_bytes(b.gate_bank.dtype, hidden, ffn, experts)?,
            "gate bank",
        ),
        (
            b.up_bank,
            bank_bytes(b.up_bank.dtype, hidden, ffn, experts)?,
            "up bank",
        ),
        (b.down_bank, g.weight_bytes, "down bank"),
        (b.input, input, "input"),
        (b.ids, product(g.slots, 4, "route ids")?, "route ids"),
        (
            b.weights,
            product(g.slots, 4, "route weights")?,
            "route weights",
        ),
        (b.counts, product(experts, 4, "counts")?, "counts"),
        (b.slots, g.bucket_bytes, "bucket slots"),
        (b.inner, g.inner_bytes, "inner"),
        (b.slot_out, g.out_bytes, "slot outputs"),
        (b.output, input, "output"),
    ] {
        check_extent(ctx, t, bytes, name)?;
    }
    Ok(())
}

pub(crate) fn capacity_supported(
    width: usize,
    threads: usize,
    static_bytes: usize,
    limit: usize,
) -> bool {
    width == 32
        && threads >= 128
        && static_bytes
            .checked_add(DYNAMIC_BYTES)
            .is_some_and(|total| total <= limit)
}

fn pipeline_name(dtype: GgmlType) -> Result<&'static str, MetalError> {
    match dtype {
        GgmlType::IQ3_S => Ok("kernel_moe_down_iq3_s_f32_grouped_slots_tensor"),
        GgmlType::Q5_K => Ok("kernel_moe_down_q5_K_f32_grouped_slots_tensor"),
        _ => Err(bad_shape(
            KERNEL,
            "only native IQ3_S/Q5_K down are eligible",
        )),
    }
}

pub(crate) struct PreparedDown<'a> {
    ctx: &'a MetalContext,
    enc: &'a KernelEncoder,
    bindings: [MetalTensor; 5],
    dims: [usize; 4],
}

impl PreparedDown<'_> {
    pub(crate) fn encode(self) -> Result<(), MetalError> {
        let [weight, inner, counts, ids, out] = &self.bindings;
        let [k, m, e, rows] = self.dims;
        // Preparation already resolved the immutable cached pipeline and all
        // checks. Use the shared API unchanged, including its census label.
        encode_moe_down_f32_grouped_slots_tensor(
            self.ctx, self.enc, weight, inner, counts, ids, out, k, m, e, rows,
        )
    }
}

/// Counts are in 0..=rows, IDs have expert stride rows and contain unique
/// flattened token*top_k+slot indices (invalid IDs are ignored by the shader).
/// The router establishes this data contract; preparation never reads GPU data.
/// Tier/rollback/capacity decline before mutation; bad bindings and missing
/// eligible pipelines are errors, never silently masked by the generic route.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare<'a>(
    ctx: &'a MetalContext,
    enc: &'a KernelEncoder,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    k: usize,
    m: usize,
    e: usize,
    top_k: usize,
    rows: usize,
) -> Result<Option<PreparedDown<'a>>, MetalError> {
    if !moe_grouped_tensor_selected(ctx) {
        return Ok(None);
    }
    let g = geometry(weight.dtype, k, m, e, top_k, rows)?;
    check_encoder(ctx, enc)?;
    for (t, bytes, dtype, writable, alignment, name) in [
        (weight, g.weight_bytes, weight.dtype, false, 16, "weight"),
        (inner, g.inner_bytes, GgmlType::F32, false, 16, "inner"),
        (
            counts,
            product(e, 4, "counts")?,
            GgmlType::I32,
            false,
            4,
            "counts",
        ),
        (ids, g.bucket_bytes, GgmlType::I32, false, 4, "bucket slots"),
        (out, g.out_bytes, GgmlType::F32, true, 16, "out"),
    ] {
        check_extent(ctx, t, bytes, name)?;
        check_tensor(KERNEL, t, dtype, &t.shape, writable, name)?;
        check_alignment(KERNEL, t, alignment, name)?;
    }
    check_expert_bank(KERNEL, weight, k, m, e, "weight")?;
    check_disjoint(
        KERNEL,
        out,
        &[
            (weight, "weight"),
            (inner, "inner"),
            (counts, "counts"),
            (ids, "ids"),
        ],
    )?;
    let name = pipeline_name(weight.dtype)?.to_owned();
    #[cfg(test)]
    let name = if matches!(
        FAULT.with(std::cell::Cell::get),
        Some(PreflightFault::MissingPipeline)
    ) {
        "__glm_tensor_down_missing_pipeline".to_owned()
    } else {
        name
    };
    let p = ctx.tensor_pipeline(&name)?;
    let limit = ctx.device.maxThreadgroupMemoryLength();
    #[cfg(test)]
    let limit = if matches!(
        FAULT.with(std::cell::Cell::get),
        Some(PreflightFault::Capacity)
    ) {
        0
    } else {
        limit
    };
    if !capacity_supported(
        p.threadExecutionWidth(),
        p.maxTotalThreadsPerThreadgroup(),
        p.staticThreadgroupMemoryLength(),
        limit,
    ) {
        return Ok(None);
    }
    Ok(Some(PreparedDown {
        ctx,
        enc,
        bindings: [
            weight.clone(),
            inner.clone(),
            counts.clone(),
            ids.clone(),
            out.clone(),
        ],
        dims: [k, m, e, rows],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_tensor_down_geometry_prices_native_storage_expert_strides_and_slot_limit() {
        let (block, bytes) = GgmlType::IQ3_S.storage_layout().unwrap();
        for (k, m, e) in [
            (2048, 4096, 288),
            (2048, 6144, 256),
            (4096, 8192, 16),
            (256, 65, 32),
        ] {
            for rows in [1024, 1025, 2048] {
                let g = geometry(GgmlType::IQ3_S, k, m, e, 8, rows).unwrap();
                assert_eq!(
                    g,
                    Geometry {
                        weight_bytes: k / block as usize * bytes as usize * m * e,
                        inner_bytes: k * rows * 8 * 4,
                        bucket_bytes: e * rows * 4,
                        out_bytes: m * rows * 8 * 4,
                        slots: rows * 8,
                    }
                );
            }
        }
    }

    #[test]
    fn cpu_tensor_down_bank_preflight_rejects_row_byte_and_expert_product_overflow() {
        assert!(bank_bytes(GgmlType::F32, u32::MAX as usize / 4 + 1, 256, 16).is_err());
        assert!(bank_bytes(GgmlType::Q8_0, u32::MAX as usize / 32 * 32, 256, 16).is_err());
        assert!(bank_bytes(GgmlType::IQ3_S, 256, usize::MAX, 16).is_err());
        assert!(bank_bytes(GgmlType::IQ3_S, 256, 256, usize::MAX).is_err());
        assert!(bank_bytes(GgmlType::IQ3_S, 255, 256, 16).is_err());
        assert!(bank_bytes(GgmlType::Unknown, 256, 256, 16).is_err());
    }

    #[test]
    fn cpu_tensor_q5_down_native_bytes_pipeline_and_bounded_dtype_contract() {
        for rows in [1024, 2048] {
            let g = geometry(GgmlType::Q5_K, 2048, 4096, 288, 8, rows).unwrap();
            assert_eq!(g.weight_bytes, 2048 / 256 * 176 * 4096 * 288);
            assert_eq!(g.inner_bytes, 2048 * rows * 8 * 4);
            assert_eq!(g.out_bytes, 4096 * rows * 8 * 4);
            assert_eq!(g.bucket_bytes, 288 * rows * 4);
            assert_eq!(g.slots, rows * 8);
        }
        for dtype in [GgmlType::IQ3_S, GgmlType::Q5_K] {
            assert_eq!(
                pipeline_name(dtype).unwrap(),
                moe_grouped_tensor_pipeline_name(
                    MoeGroupedTensorRole::Down,
                    dtype,
                    TensorActivation::F32
                )
                .unwrap()
            );
            assert!(geometry(dtype, 2047, 4096, 288, 8, 1024).is_err());
            assert!(geometry(dtype, 256, usize::MAX, 288, 8, 1024).is_err());
        }
        for dtype in [
            GgmlType::Q4_K,
            GgmlType::Q8_0,
            GgmlType::F32,
            GgmlType::Unknown,
        ] {
            assert!(pipeline_name(dtype).is_err());
            assert!(geometry(dtype, 2048, 4096, 288, 8, 1024).is_err());
        }
        assert!(capacity_supported(32, 128, 512, 8192 + 512));
        assert!(!capacity_supported(32, 128, 512, 8192 + 511));
        assert!(!capacity_supported(16, 128, 0, 8192));
        assert!(!capacity_supported(32, 127, 0, 8192));
        assert!(!capacity_supported(32, 128, usize::MAX, usize::MAX));
    }
}
