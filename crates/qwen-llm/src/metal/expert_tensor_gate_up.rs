//! Prepared N32 gate/up composition. Half input, unweighted F32 inner,
//! incumbent F32-input down, then post-down routing weights. No encoder mutation
//! until every binding and pipeline in the composition has passed preflight.

use super::checks::{bad_shape, check_alignment, check_disjoint, check_tensor, to_u32};
use super::expert::check_grouped_routed_experts;
use super::expert_tensor_down::{bank_bytes, check_encoder, check_extent};
use super::*;

const K: &str = "expert_tensor_gate_up";

#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) enum Fault {
    Missing(usize),
    Capacity(usize),
    CapacityAndMissing,
}

#[cfg(test)]
thread_local! {
    static FAULT: std::cell::Cell<Option<Fault>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) struct FaultScope(Option<Fault>, std::marker::PhantomData<std::rc::Rc<()>>);

#[cfg(test)]
impl FaultScope {
    pub(crate) fn new(fault: Fault) -> Self {
        Self(
            FAULT.with(|f| f.replace(Some(fault))),
            std::marker::PhantomData,
        )
    }
}

#[cfg(test)]
impl Drop for FaultScope {
    fn drop(&mut self) {
        FAULT.with(|f| f.set(self.0));
    }
}

fn product(a: usize, b: usize) -> Result<usize, MetalError> {
    a.checked_mul(b)
        .ok_or_else(|| bad_shape(K, "extent overflow"))
}

pub(crate) fn geometry(
    h: usize,
    f: usize,
    e: usize,
    top: usize,
    rows: usize,
) -> Result<(), MetalError> {
    for d in [h, f, e, top, rows, product(top, rows)?] {
        if d == 0 || d > i32::MAX as usize {
            return Err(bad_shape(K, "dimensions and slots must fit positive i32"));
        }
    }
    if top > ROUTE_MAX_TOP_K || top > e || !h.is_multiple_of(256) || !f.is_multiple_of(256) {
        return Err(bad_shape(K, "invalid top-k or partial K block"));
    }
    to_u32(K, product(h, rows)?, "converter elements")?;
    for (width, count) in [
        (h, product(top, rows)?),
        (f, product(top, rows)?),
        (e, rows),
    ] {
        product(product(width, count)?, 4)?;
    }
    Ok(())
}

pub(crate) fn capacity(
    width: usize,
    threads: usize,
    stat: usize,
    dynamic: usize,
    limit: usize,
    required: usize,
) -> bool {
    width == 32 && threads >= required && stat.checked_add(dynamic).is_some_and(|n| n <= limit)
}

pub(crate) struct Prepared<'a> {
    ctx: &'a MetalContext,
    enc: &'a KernelEncoder,
    b: &'a GroupedExperts<'a>,
    half: MetalTensor,
    dims: [usize; 5],
    clamp: f32,
    tensor_down: Option<super::expert_tensor_down::PreparedDown<'a>>,
}

fn flat(t: &MetalTensor) -> MetalTensor {
    t.view_subrange(0, vec![t.n_elements()])
}

impl Prepared<'_> {
    pub(crate) fn encode(self) -> Result<(), MetalError> {
        let Self {
            ctx,
            enc,
            b,
            half,
            dims: [h, f, e, top, rows],
            clamp,
            tensor_down,
        } = self;
        // normed is rewritten by FFN normalization after KDA. Always recopy.
        encode_scatter_offset_f32_to_f16(ctx, enc, &flat(b.input), &flat(&half), 0, h * rows)?;
        encode_moe_route_bucket_slots_f32(ctx, enc, &flat(b.ids), b.counts, b.slots, e, rows, top)?;
        encode_moe_swiglu_f32_grouped_slots_tensor(
            ctx,
            enc,
            b.gate_bank,
            b.up_bank,
            &flat(&half),
            b.counts,
            b.slots,
            &flat(b.inner),
            h,
            f,
            e,
            top,
            rows,
            Some(clamp),
        )?;
        if let Some(down) = tensor_down {
            down.encode()?;
        } else {
            encode_moe_down_f32_grouped_slots_generic(
                ctx,
                enc,
                b.down_bank,
                &flat(b.inner),
                b.counts,
                b.slots,
                &flat(b.slot_out),
                f,
                h,
                e,
                rows,
            )?;
        }
        encode_moe_weighted_sum_packed_f32(
            ctx,
            enc,
            &flat(b.slot_out),
            &flat(b.weights),
            &flat(b.output),
            h,
            top,
            rows,
        )
    }
}

/// Capability adapter (small fixtures are allowed here, never by production
/// selection). Q5 generic rollback deliberately declines this whole candidate.
pub(crate) fn prepare<'a>(
    ctx: &'a MetalContext,
    enc: &'a KernelEncoder,
    b: &'a GroupedExperts<'a>,
    backing: &MetalTensor,
    h: usize,
    f: usize,
    e: usize,
    top: usize,
    rows: usize,
    clamp: f32,
    policy: GroupedDownPolicy,
) -> Result<Option<Prepared<'a>>, MetalError> {
    if !moe_grouped_tensor_selected(ctx) {
        return Ok(None);
    }
    if !matches!(b.gate_bank.dtype, GgmlType::Q4_K | GgmlType::Q5_K)
        || b.gate_bank.dtype != b.up_bank.dtype
        || !matches!(
            (b.down_bank.dtype, policy),
            (GgmlType::Q5_K, GroupedDownPolicy::Q5KTensor)
                | (GgmlType::Q6_K, GroupedDownPolicy::Incumbent)
        )
        || rows > 2048
    {
        return Ok(None);
    }
    geometry(h, f, e, top, rows)?;
    check_encoder(ctx, enc)?;
    if !clamp.is_finite() || clamp <= 0.0 {
        return Err(bad_shape(K, "clamp must be finite and positive"));
    }
    let slots = product(top, rows)?;
    let bytes = |a, b| product(product(a, b)?, 4);
    for (t, size, name) in [
        (b.gate_bank, bank_bytes(b.gate_bank.dtype, h, f, e)?, "gate"),
        (b.up_bank, bank_bytes(b.up_bank.dtype, h, f, e)?, "up"),
        (b.down_bank, bank_bytes(b.down_bank.dtype, f, h, e)?, "down"),
        (b.input, bytes(h, rows)?, "input"),
        (b.ids, bytes(top, rows)?, "ids"),
        (b.weights, bytes(top, rows)?, "weights"),
        (b.counts, bytes(e, 1)?, "counts"),
        (b.slots, bytes(e, rows)?, "slots"),
        (b.inner, bytes(f, slots)?, "inner"),
        (b.slot_out, bytes(h, slots)?, "slot output"),
        (b.output, bytes(h, rows)?, "output"),
    ] {
        check_extent(ctx, t, size, name)?;
        let (_, size) = checked_ggml_shape_bytes(&t.shape, t.dtype)?;
        if t.offset
            .checked_add(size as u64)
            .is_none_or(|end| end > isize::MAX as u64)
        {
            return Err(bad_shape(K, "binding range overflow"));
        }
    }
    check_grouped_routed_experts(K, enc, b, h, f, e, top, rows)?;
    if backing.shape.len() != 2 || backing.shape[0] != h as u64 || backing.shape[1] < rows as u64 {
        return Err(bad_shape(
            K,
            "Half scratch must retain admitted [hidden, rows] backing",
        ));
    }
    let (_, half_bytes) = checked_ggml_shape_bytes(&backing.shape, backing.dtype)?;
    check_extent(ctx, backing, half_bytes, "Half backing")?;
    check_tensor(
        K,
        backing,
        GgmlType::F16,
        &backing.shape,
        true,
        "Half backing",
    )?;
    check_alignment(K, backing, 16, "Half backing")?;
    if backing
        .offset
        .checked_add(half_bytes as u64)
        .is_none_or(|end| end > isize::MAX as u64)
    {
        return Err(bad_shape(K, "Half backing range overflow"));
    }
    for t in [
        b.gate_bank,
        b.up_bank,
        b.down_bank,
        b.input,
        b.ids,
        b.weights,
        b.counts,
        b.slots,
        b.inner,
        b.slot_out,
        b.output,
    ] {
        check_disjoint(K, backing, &[(t, "composition binding")])?;
    }
    let half = backing.view_subrange(0, vec![h as u64, rows as u64]);
    let gate = moe_grouped_tensor_pipeline_name(
        MoeGroupedTensorRole::Down,
        b.gate_bank.dtype,
        TensorActivation::F16,
    )
    .unwrap();
    let up = moe_grouped_tensor_pipeline_name(
        MoeGroupedTensorRole::UpClamped,
        b.up_bank.dtype,
        TensorActivation::F16,
    )
    .unwrap();
    let down_tensor = policy == GroupedDownPolicy::Q5KTensor;
    let down = if down_tensor {
        moe_grouped_tensor_pipeline_name(
            MoeGroupedTensorRole::Down,
            b.down_bank.dtype,
            TensorActivation::F32,
        )
        .unwrap()
    } else {
        moe_grouped_generic_pipeline_name(MoeGroupedGenericRole::Down, b.down_bank.dtype).unwrap()
    };
    // Resolve ALL pipelines before returning a capacity decline: a missing
    // eligible pipeline is an error even when an earlier one cannot dispatch.
    let specs = [
        ("kernel_scatter_offset_f32_to_f16", false, 0, 1),
        (gate.as_str(), true, 8192, 128),
        (up.as_str(), true, 8192, 128),
        (down.as_str(), down_tensor, 8192, 128),
        ("kernel_moe_route_bucket_slots_f32", false, 0, 256),
        ("kernel_moe_weighted_sum_packed_f32", false, 0, 32),
    ];
    let mut supported = true;
    for (_index, (name, tensor, dynamic, threads)) in specs.into_iter().enumerate() {
        #[cfg(test)]
        let name = if matches!(FAULT.with(std::cell::Cell::get), Some(Fault::Missing(i)) if i == _index)
            || (matches!(
                FAULT.with(std::cell::Cell::get),
                Some(Fault::CapacityAndMissing)
            ) && _index == 5)
        {
            "__glm_gate_up_missing_pipeline"
        } else {
            name
        };
        let p = if tensor {
            ctx.tensor_pipeline(name)?
        } else {
            ctx.pipeline(name)?
        };
        let limit = ctx.device.maxThreadgroupMemoryLength();
        #[cfg(test)]
        let limit = if matches!(FAULT.with(std::cell::Cell::get), Some(Fault::Capacity(i)) if i == _index)
            || (matches!(
                FAULT.with(std::cell::Cell::get),
                Some(Fault::CapacityAndMissing)
            ) && _index == 0)
        {
            0
        } else {
            limit
        };
        supported &= capacity(
            p.threadExecutionWidth(),
            p.maxTotalThreadsPerThreadgroup(),
            p.staticThreadgroupMemoryLength(),
            dynamic,
            limit,
            threads,
        );
        #[cfg(test)]
        if matches!(FAULT.with(std::cell::Cell::get), Some(Fault::Capacity(i)) if i == _index)
            || (matches!(
                FAULT.with(std::cell::Cell::get),
                Some(Fault::CapacityAndMissing)
            ) && _index == 0)
        {
            supported = false;
        }
    }
    let tensor_down = if down_tensor {
        let prepared = super::expert_tensor_down::prepare(
            ctx,
            enc,
            b.down_bank,
            &flat(b.inner),
            b.counts,
            b.slots,
            &flat(b.slot_out),
            f,
            h,
            e,
            top,
            rows,
        )?;
        supported &= prepared.is_some();
        prepared
    } else {
        None
    };
    if !supported {
        return Ok(None);
    }
    Ok(Some(Prepared {
        ctx,
        enc,
        b,
        half,
        dims: [h, f, e, top, rows],
        clamp,
        tensor_down,
    }))
}
