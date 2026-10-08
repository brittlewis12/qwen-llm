//! Checked IQ3_S grouped down with M128/N16 tiles for counts 1..=16.
use super::checks::{bad_shape, check_disjoint, require_serial};
use super::moe::{checked_moe_product, validate_moe_decode_tensor};
use super::*;

const KERNEL: &str = "iq3_s_down_retile";
const RETILE: &str = "kernel_moe_down_iq3_s_f32_grouped_slots_m128_n16_retile";
const INCUMBENT: &str = "kernel_moe_down_iq3_s_f32_grouped_slots_generic";

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum DownRetile {
    #[default]
    Incumbent,
    Blanket,
    SmallCounts,
}

#[cfg(test)]
thread_local! {
    static STATE: Cell<(Option<DownRetile>, usize)> = const { Cell::new((None, 0)) };
    // Simulate missing pipelines and device limits without changing runtime policy.
    static PREFLIGHT_FAULT: Cell<Option<(&'static str, bool)>> = const { Cell::new(None) };
}

/// Counts successful composition substitutions. Nested scopes restore on unwind.
#[cfg(test)]
pub(crate) fn with_variant<R>(variant: DownRetile, f: impl FnOnce() -> R) -> (R, usize) {
    with_scope(Some(variant), f)
}

#[cfg(test)]
pub(crate) fn with_production<R>(f: impl FnOnce() -> R) -> (R, usize) {
    with_scope(None, f)
}

#[cfg(test)]
fn with_scope<R>(variant: Option<DownRetile>, f: impl FnOnce() -> R) -> (R, usize) {
    struct Restore((Option<DownRetile>, usize));
    impl Drop for Restore {
        fn drop(&mut self) {
            STATE.with(|state| state.set(self.0));
        }
    }
    let _restore = Restore(STATE.with(|state| state.replace((variant, 0))));
    let result = f();
    (result, STATE.with(|state| state.get().1))
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GenericMmArgs {
    m: u32,
    n: u32,
    k: u32,
    nb01: u32,
    stride_b: u32,
    min_count: u32,
    max_count: u32,
    b_div: u32,
    slot_limit: u32,
}

#[allow(clippy::too_many_arguments)]
fn checked_args(
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<GenericMmArgs, MetalError> {
    for (name, value) in [
        ("K", n_in),
        ("M", n_out),
        ("experts", n_expert),
        ("N", n_tokens),
    ] {
        if value == 0 || value > i32::MAX as usize {
            return Err(bad_shape(KERNEL, format!("{name} must fit positive i32")));
        }
    }
    if !n_in.is_multiple_of(256) {
        return Err(bad_shape(KERNEL, "IQ3_S K must be divisible by 256"));
    }
    let bank_elements = checked_moe_product(KERNEL, "bank", &[n_in, n_out, n_expert])?;
    let bucket_elements = checked_moe_product(KERNEL, "buckets", &[n_expert, n_tokens])?;
    let (out_elements, _) = checked_ggml_shape_bytes(&out.shape, out.dtype)?;
    let slots = out_elements / n_out;
    if slots == 0 || slots > i32::MAX as usize || !out_elements.is_multiple_of(n_out) {
        return Err(bad_shape(
            KERNEL,
            "output must contain whole rows and a positive i32 slot count",
        ));
    }
    let input_elements = checked_moe_product(KERNEL, "input", &[slots, n_in])?;
    for (name, tensor, elements, dtype, writable, alignment) in [
        ("weight", weight, bank_elements, GgmlType::IQ3_S, false, 2),
        ("inner", inner, input_elements, GgmlType::F32, false, 16),
        ("counts", counts, n_expert, GgmlType::I32, false, 4),
        ("ids", ids, bucket_elements, GgmlType::I32, false, 4),
        ("out", out, out_elements, GgmlType::F32, true, 4),
    ] {
        validate_moe_decode_tensor(
            KERNEL,
            name,
            tensor,
            elements,
            &[dtype],
            writable,
            alignment,
        )?;
    }
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
    let row_bytes = (n_in / 256)
        .checked_mul(110)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| bad_shape(KERNEL, "weight row byte stride exceeds u32"))?;
    Ok(GenericMmArgs {
        m: n_out as u32,
        n: n_tokens as u32,
        k: n_in as u32,
        nb01: row_bytes,
        stride_b: n_in as u32,
        min_count: 0,
        max_count: i32::MAX as u32,
        b_div: 1,
        slot_limit: slots as u32,
    })
}

fn check_capacity(
    width: usize,
    threads: usize,
    static_bytes: usize,
    dynamic_bytes: usize,
    limit: usize,
) -> Result<(), MetalError> {
    if width != 32
        || threads < 128
        || static_bytes
            .checked_add(dynamic_bytes)
            .is_none_or(|n| n > limit)
    {
        return Err(bad_shape(
            KERNEL,
            format!("requires SIMD32, TG128 and {dynamic_bytes} dynamic TGM bytes"),
        ));
    }
    Ok(())
}

fn pipeline(ctx: &MetalContext, name: &str, dynamic_bytes: usize) -> Result<Pipeline, MetalError> {
    let limit = ctx.device.maxThreadgroupMemoryLength();
    #[cfg(test)]
    let (name, limit) = match PREFLIGHT_FAULT.with(Cell::get) {
        Some((target, missing)) if target == name => {
            if missing {
                ("__iq3_s_retile_missing_pipeline", limit)
            } else {
                (name, 0)
            }
        }
        _ => (name, limit),
    };
    let pso = ctx.pipeline(name)?;
    check_capacity(
        pso.threadExecutionWidth(),
        pso.maxTotalThreadsPerThreadgroup(),
        pso.staticThreadgroupMemoryLength(),
        dynamic_bytes,
        limit,
    )?;
    Ok(pso)
}

struct CheckedDown<'a> {
    ctx: &'a MetalContext,
    enc: &'a KernelEncoder,
    bindings: [&'a MetalTensor; 5],
    args: GenericMmArgs,
    experts: usize,
}

#[allow(clippy::too_many_arguments)]
fn checked_down<'a>(
    ctx: &'a MetalContext,
    enc: &'a KernelEncoder,
    weight: &'a MetalTensor,
    inner: &'a MetalTensor,
    counts: &'a MetalTensor,
    ids: &'a MetalTensor,
    out: &'a MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<CheckedDown<'a>, MetalError> {
    require_serial(KERNEL, enc)?;
    let args = checked_args(
        weight, inner, counts, ids, out, n_in, n_out, n_expert, n_tokens,
    )?;
    let bindings = [weight, inner, counts, ids, out];
    let device = ctx.device.registryID();
    if enc.parent_command_buffer().device().registryID() != device
        || bindings
            .iter()
            .any(|t| t.buffer.device().registryID() != device)
    {
        return Err(bad_shape(
            KERNEL,
            "encoder and bindings must belong to the context device",
        ));
    }
    Ok(CheckedDown {
        ctx,
        enc,
        bindings,
        args,
        experts: n_expert,
    })
}

impl CheckedDown<'_> {
    fn encode_incumbent(&self, min_count: u32) -> Result<(), MetalError> {
        let [weight, inner, counts, ids, out] = self.bindings;
        encode_moe_down_f32_grouped_slots_generic_range(
            self.ctx,
            self.enc,
            weight,
            inner,
            counts,
            ids,
            out,
            self.args.k as usize,
            self.args.m as usize,
            self.experts,
            self.args.n as usize,
            min_count,
            i32::MAX as u32,
        )
    }
}

/// All bindings and both pipeline capabilities are checked before this is returned.
/// Keep preparation ahead of the first grouped dispatch so fallback cannot split a down call.
pub(crate) struct PreparedDown<'a> {
    checked: CheckedDown<'a>,
    retile: Pipeline,
    small_counts: bool,
}

impl<'a> PreparedDown<'a> {
    fn prepare(checked: CheckedDown<'a>, small_counts: bool) -> Result<Self, MetalError> {
        if small_counts {
            pipeline(checked.ctx, INCUMBENT, 8192)?;
        }
        let retile = pipeline(checked.ctx, RETILE, 9216)?;
        Ok(Self {
            checked,
            retile,
            small_counts,
        })
    }

    pub(crate) fn encode(self) -> Result<(), MetalError> {
        self.encode_inner()?;
        #[cfg(test)]
        STATE.with(|state| state.set((state.get().0, state.get().1 + 1)));
        Ok(())
    }

    fn encode_inner(self) -> Result<(), MetalError> {
        let c = self.checked;
        let enc = c.enc;
        let mut args = c.args;
        args.min_count = 1;
        if self.small_counts {
            args.max_count = 16;
        }
        // Preparation precedes gate/up, whose lookup changes the census label.
        census_record_pso(RETILE);
        enc.set_pipeline(&self.retile);
        enc.set_bytes(0, &args);
        for (index, tensor) in c.bindings.into_iter().enumerate() {
            enc.set_tensor(index + 1, tensor);
        }
        enc.set_threadgroup_memory(0, 9216);
        enc.dispatch(
            MTLSize {
                width: if self.small_counts {
                    1
                } else {
                    (args.n as usize).div_ceil(16)
                },
                height: (args.m as usize).div_ceil(128),
                depth: c.experts,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        if self.small_counts {
            c.encode_incumbent(17)?;
        }
        Ok(())
    }
}

/// Prepare counts 1..=16 on the retile and 17..=MAX on generic down.
/// The family caller owns geometry/lineage policy. Counts must be in 0..=N,
/// with unique valid slots across buckets, as produced by routing. A missing
/// pipeline or unsupported capability returns None without encoding anything;
/// invalid bindings remain errors. N is the actual bucket stride, including
/// when the small-count range uses only one N panel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_small_counts<'a>(
    ctx: &'a MetalContext,
    enc: &'a KernelEncoder,
    weight: &'a MetalTensor,
    inner: &'a MetalTensor,
    counts: &'a MetalTensor,
    ids: &'a MetalTensor,
    out: &'a MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<Option<PreparedDown<'a>>, MetalError> {
    let small_counts = true;
    #[cfg(test)]
    let small_counts = match STATE.with(|state| state.get().0) {
        Some(DownRetile::Incumbent) => return Ok(None),
        Some(DownRetile::Blanket) => false,
        _ => small_counts,
    };
    let checked = checked_down(
        ctx, enc, weight, inner, counts, ids, out, n_in, n_out, n_expert, n_tokens,
    )?;
    Ok(PreparedDown::prepare(checked, small_counts).ok())
}

/// Strict diagnostic entry point; unlike production preparation, unsupported
/// capabilities are errors and successful direct calls do not count substitutions.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_variant(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    variant: DownRetile,
    weight: &MetalTensor,
    inner: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    n_tokens: usize,
) -> Result<(), MetalError> {
    let checked = checked_down(
        ctx, enc, weight, inner, counts, ids, out, n_in, n_out, n_expert, n_tokens,
    )?;
    match variant {
        DownRetile::Incumbent => {
            pipeline(ctx, INCUMBENT, 8192)?;
            checked.encode_incumbent(0)
        }
        _ => PreparedDown::prepare(checked, variant == DownRetile::SmallCounts)?.encode_inner(),
    }
}

#[cfg(test)]
mod tests;
