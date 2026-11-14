//! Prepared tensor gate/up composition. Eligible GLM IQ buckets select N32/N64
//! panels by count. Half input, unweighted F32 inner,
//! incumbent F32-input down, then post-down routing weights. No encoder mutation
//! until every binding and pipeline in the composition has passed preflight.

use super::checks::{bad_shape, check_alignment, check_disjoint, check_tensor, to_u32};
use super::expert::check_grouped_routed_experts;
use super::expert_tensor_down::{bank_bytes, check_encoder, check_extent};
use super::*;

const K: &str = "expert_tensor_gate_up";

crate::env_flag!(
    /// Default-on M5 GLM Fast/Half IQ gate/up panel selection at 1024/2048 rows, extended to arbitrary 1024..=4096 with PREFILL_IQ_ANY_ROWS. Buckets through 32 slots retain N32; larger buckets use N64. Zero restores all-N32 panels. Existing tensor, family and precision gates remain mandatory. No added GPU buffer or operand/reduction change.
    default_on adaptive_requested,
    "QWEN_GLM53_PREFILL_PANELS"
);

crate::env_flag!(
    /// Default-off experimental F16-boundary-preserving fused GLM IQ2_S clamped gate/up. Zero retains the tensor `wide_probe`/`up_clamped` path.
    default_off gate_up_fusion_env_requested,
    "QWEN_GLM53_GATE_UP_FUSION"
);

#[cfg(feature = "perf-ablations")]
thread_local! {
    static GATE_UP_FUSION_BENCH_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

fn gate_up_fusion_requested() -> bool {
    #[cfg(feature = "perf-ablations")]
    if let Some(value) = GATE_UP_FUSION_BENCH_OVERRIDE.with(std::cell::Cell::get) {
        return value;
    }
    gate_up_fusion_env_requested()
}

#[cfg(feature = "perf-ablations")]
pub(crate) fn set_gate_up_fusion_bench_override(value: Option<bool>) -> Option<bool> {
    GATE_UP_FUSION_BENCH_OVERRIDE.with(|slot| slot.replace(value))
}

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

/// Down composition admitted by the explicit tensor gate/up adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DownRoute {
    /// Native tensor down; a declined tier or PSO falls back to the ordinary
    /// grouped gate/up instead of this composition.
    Tensor,
    /// Native tensor down, with the small-count retile when the tensor tier
    /// or PSO declines.
    TensorOrSmallCounts,
    /// Incumbent grouped down (generic, or the IQ4_XS kernel).
    Incumbent,
}

/// Maps a routed down policy onto this composition. The grouped adapter admits
/// `Iq3STensorOrSmallCounts` for IQ3_S down, so the explicit adapter must admit
/// the same pairing; otherwise short appends silently take the F32-input
/// grouped gate/up instead of the intended Half-storage composition.
pub(crate) fn down_route(down: GgmlType, policy: GroupedDownPolicy) -> Option<DownRoute> {
    match (down, policy) {
        (GgmlType::Q5_K, GroupedDownPolicy::Q5KTensor) => Some(DownRoute::Tensor),
        (GgmlType::IQ3_S, GroupedDownPolicy::Iq3STensor) => Some(DownRoute::Tensor),
        (GgmlType::IQ3_S, GroupedDownPolicy::Iq3STensorOrSmallCounts) => {
            Some(DownRoute::TensorOrSmallCounts)
        }
        (GgmlType::IQ3_S | GgmlType::IQ4_XS | GgmlType::Q6_K, GroupedDownPolicy::Incumbent) => {
            Some(DownRoute::Incumbent)
        }
        _ => None,
    }
}

enum PreparedDown<'a> {
    Tensor(super::expert_tensor_down::PreparedDown<'a>),
    SmallCounts(super::iq3_s_down_retile::PreparedDown<'a>),
}

impl PreparedDown<'_> {
    fn encode(self) -> Result<(), MetalError> {
        match self {
            Self::Tensor(down) => down.encode(),
            Self::SmallCounts(down) => down.encode(),
        }
    }
}

pub(crate) struct Prepared<'a> {
    ctx: &'a MetalContext,
    enc: &'a KernelEncoder,
    b: &'a GroupedExperts<'a>,
    half: MetalTensor,
    dims: [usize; 5],
    clamp: f32,
    adaptive: bool,
    down: Option<PreparedDown<'a>>,
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
            adaptive,
            down,
        } = self;
        // The prototype reads the original F32 input and performs the same
        // F16 narrowing at the fused kernel boundary. All other routes keep
        // the resident half copy consumed by the incumbent tensor tiles.
        let fused_clamped = gate_up_fusion_requested()
            && !adaptive
            && e == 288
            && top == 8
            && b.gate_bank.dtype == GgmlType::IQ2_S
            && b.up_bank.dtype == GgmlType::IQ2_S;
        if !fused_clamped {
            // normed is rewritten by FFN normalization after KDA. Always recopy.
            encode_scatter_offset_f32_to_f16(ctx, enc, &flat(b.input), &flat(&half), 0, h * rows)?;
        }
        encode_moe_route_bucket_slots_f32_status(
            ctx,
            enc,
            &flat(b.ids),
            b.counts,
            b.slots,
            b.status,
            e,
            rows,
            top,
        )?;
        if fused_clamped {
            encode_moe_swiglu_clamped_f32_grouped_slots_generic(
                ctx,
                enc,
                b.gate_bank,
                b.up_bank,
                &flat(b.input),
                b.counts,
                b.slots,
                &flat(b.inner),
                h,
                f,
                e,
                top,
                rows,
                clamp,
            )?;
        } else if adaptive {
            encode_moe_swiglu_adaptive_tensor(
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
                clamp,
            )?;
        } else {
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
        }
        if let Some(down) = down {
            down.encode()?;
        } else if b.down_bank.dtype == GgmlType::IQ4_XS {
            encode_moe_down_iq4_xs_f32_grouped_slots(
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
    if !matches!(
        b.gate_bank.dtype,
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::IQ2_S | GgmlType::IQ3_S
    ) || b.gate_bank.dtype != b.up_bank.dtype
        || rows > 2048
    {
        return Ok(None);
    }
    let Some(route) = down_route(b.down_bank.dtype, policy) else {
        return Ok(None);
    };
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
    let down_tensor = matches!(route, DownRoute::Tensor | DownRoute::TensorOrSmallCounts);
    let down = if down_tensor {
        moe_grouped_tensor_pipeline_name(
            MoeGroupedTensorRole::Down,
            b.down_bank.dtype,
            TensorActivation::F32,
        )
        .unwrap()
    } else if b.down_bank.dtype == GgmlType::IQ4_XS {
        "kernel_moe_down_iq4_xs_f32_grouped_slots".into()
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
    // The small-count retile owns tensor-down capacity decline; a missing
    // pipeline is still an error below. Other routes require their down PSO.
    let down_required = route != DownRoute::TensorOrSmallCounts;
    let mut supported = true;
    for (_index, (name, tensor, dynamic, threads)) in specs.into_iter().enumerate() {
        let required = down_required || _index != 3;
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
        if required {
            supported &= capacity(
                p.threadExecutionWidth(),
                p.maxTotalThreadsPerThreadgroup(),
                p.staticThreadgroupMemoryLength(),
                dynamic,
                limit,
                threads,
            );
        }
        #[cfg(test)]
        if required
            && (matches!(FAULT.with(std::cell::Cell::get), Some(Fault::Capacity(i)) if i == _index)
                || (matches!(
                    FAULT.with(std::cell::Cell::get),
                    Some(Fault::CapacityAndMissing)
                ) && _index == 0))
        {
            supported = false;
        }
    }
    let adaptive = if (h, f, e, top) == (4096, 2048, 288, 8)
        && (matches!(rows, 1024 | 2048) || crate::glm5_next_metal::row_policy::iq_expanded(rows))
        && matches!(b.gate_bank.dtype, GgmlType::IQ2_S | GgmlType::IQ3_S)
        && ctx.device_facts().gpu_families.apple10
        && adaptive_requested()
    {
        let name = wide_probe_pipeline_name(b.gate_bank.dtype).unwrap();
        let p = ctx.tensor_pipeline(&name)?;
        capacity(
            p.threadExecutionWidth(),
            p.maxTotalThreadsPerThreadgroup(),
            p.staticThreadgroupMemoryLength(),
            16384,
            ctx.device.maxThreadgroupMemoryLength(),
            128,
        )
    } else {
        false
    };
    let down = match route {
        DownRoute::Incumbent => None,
        DownRoute::Tensor | DownRoute::TensorOrSmallCounts => {
            let tensor = super::expert_tensor_down::prepare(
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
            match tensor {
                Some(down) => Some(PreparedDown::Tensor(down)),
                // Borrow the admitted bindings directly: the retile checks
                // element counts, and this keeps the prepared down independent
                // of any flattened temporaries in `prepare`.
                None if route == DownRoute::TensorOrSmallCounts => {
                    super::iq3_s_down_retile::prepare_small_counts(
                        ctx,
                        enc,
                        b.down_bank,
                        b.inner,
                        b.counts,
                        b.slots,
                        b.slot_out,
                        f,
                        h,
                        e,
                        rows,
                    )?
                    .map(PreparedDown::SmallCounts)
                }
                None => None,
            }
        }
    };
    // Preparation covers every binding so invalid down bindings are still
    // errors when an earlier pipeline cannot dispatch.
    if !supported {
        return Ok(None);
    }
    if down.is_none() && route != DownRoute::Incumbent {
        return Ok(None);
    }
    Ok(Some(Prepared {
        ctx,
        enc,
        b,
        half,
        dims: [h, f, e, top, rows],
        clamp,
        adaptive,
        down,
    }))
}
