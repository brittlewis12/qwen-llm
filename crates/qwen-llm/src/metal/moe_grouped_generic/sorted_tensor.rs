//! V4.1's native-IQ expert runs. Preparation validates every binding and PSO
//! before dispatch; separate encode phases retain the FFN stage timing points.
//! Keep the incumbent half operands and both F32 storage rounding boundaries
//! before the half down input. Qualify tensor reductions and model outputs
//! against the original chain.
use super::*;
use crate::metal::checks::{
    bad_shape, check_disjoint, check_expert_bank, check_physical, require_serial,
};

pub(crate) fn sorted_iq_supported(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::IQ2_XXS | GgmlType::IQ2_XS | GgmlType::IQ2_S | GgmlType::IQ3_XXS
    )
}

#[derive(Clone, Copy)]
pub(crate) struct SortedExpertGeometry {
    pub hidden: usize,
    pub inner: usize,
    pub experts: usize,
    pub topk: usize,
    pub rows: usize,
    pub clamp: f32,
    pub cooperative: bool,
    pub gather_f32: bool,
}

const KERNEL: &str = "v41_sorted_experts";
const SHARED_BYTES: usize = 128 * 72 * 2;
const GATHER_SHARED_BYTES: usize = SHARED_BYTES + 64 * 72 * 2;

fn extent(dims: &[usize]) -> Result<usize, MetalError> {
    dims.iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n > 0 && n <= u32::MAX as usize)
        .ok_or_else(|| bad_shape(KERNEL, "empty or overflowing extent"))
}

/// Counts/IDs must be the current router's expert buckets. Counts occupy E
/// cells; plan occupies a disjoint 2(E+1)-cell tail in that same allocation.
/// `packed` may alias `output`: only GU reads packed, and down starts afterward.
/// Every other output/scratch binding must be disjoint from live inputs.
pub(crate) struct SortedExperts<'a> {
    pipelines: [Pipeline; 4],
    banks: [&'a MetalTensor; 3],
    x: &'a MetalTensor,
    counts: &'a MetalTensor,
    ids: &'a MetalTensor,
    weights: &'a MetalTensor,
    plan: &'a MetalTensor,
    packed: &'a MetalTensor,
    inner: &'a MetalTensor,
    output: &'a MetalTensor,
    g: SortedExpertGeometry,
    slots: usize,
}

impl<'a> SortedExperts<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        ctx: &MetalContext,
        enc: &KernelEncoder,
        banks: [&'a MetalTensor; 3],
        x: &'a MetalTensor,
        counts: &'a MetalTensor,
        ids: &'a MetalTensor,
        weights: &'a MetalTensor,
        plan: &'a MetalTensor,
        packed: &'a MetalTensor,
        inner: &'a MetalTensor,
        output: &'a MetalTensor,
        g: SortedExpertGeometry,
    ) -> Result<Self, MetalError> {
        require_serial(KERNEL, enc)?;
        if g.cooperative && ctx.device_facts().name != "Apple M5 Ultra" {
            return Err(bad_shape(
                KERNEL,
                "cooperative expert device is not qualified",
            ));
        }
        if g.gather_f32 && !g.cooperative {
            return Err(bad_shape(KERNEL, "F32 gather requires cooperative experts"));
        }
        if !(1..=4096).contains(&g.rows)
            || !(1..=1024).contains(&g.experts)
            || !(1..=g.experts).contains(&g.topk)
            || g.hidden == 0
            || !g.hidden.is_multiple_of(256)
            || g.inner == 0
            || !g.inner.is_multiple_of(256)
            || !g.clamp.is_finite()
            || g.clamp <= 0.0
            || banks[0].dtype != banks[1].dtype
            || banks.iter().any(|w| !sorted_iq_supported(w.dtype))
        {
            return Err(bad_shape(
                KERNEL,
                "unsupported sorted expert geometry or IQ types",
            ));
        }
        let slots = extent(&[g.rows, g.topk])?;
        if slots < 64 {
            return Err(bad_shape(KERNEL, "device operand needs at least 64 slots"));
        }
        for (i, w) in banks.iter().enumerate() {
            let (k, m) = if i == 2 {
                (g.inner, g.hidden)
            } else {
                (g.hidden, g.inner)
            };
            check_expert_bank(KERNEL, w, k, m, g.experts, "expert bank")?;
        }
        for (t, dtype, elements, writable, name) in [
            (
                x,
                if g.gather_f32 {
                    GgmlType::F32
                } else {
                    GgmlType::F16
                },
                extent(&[g.hidden, g.rows])?,
                false,
                "input",
            ),
            (counts, GgmlType::I32, g.experts, false, "counts"),
            (
                ids,
                GgmlType::I32,
                extent(&[g.experts, g.rows])?,
                false,
                "slot ids",
            ),
            (weights, GgmlType::F32, slots, false, "route weights"),
            (plan, GgmlType::I32, 2 * (g.experts + 1), true, "run plan"),
            (
                packed,
                GgmlType::F16,
                extent(&[g.hidden, slots])?,
                true,
                "sorted input",
            ),
            (
                inner,
                GgmlType::F16,
                extent(&[g.inner, slots])?,
                true,
                "sorted inner",
            ),
            (
                output,
                GgmlType::F32,
                extent(&[g.hidden, slots])?,
                true,
                "down output",
            ),
        ] {
            if t.dtype != dtype
                || t.shape.iter().try_fold(1u64, |n, &d| n.checked_mul(d)) != Some(elements as u64)
            {
                return Err(bad_shape(
                    KERNEL,
                    format!("{name}: expected {elements} {dtype:?} elements"),
                ));
            }
            check_physical(
                KERNEL,
                t,
                if dtype == GgmlType::I32 { 4 } else { 16 },
                writable,
                name,
            )?;
        }
        let inputs = [
            (banks[0], "gate"),
            (banks[1], "up"),
            (banks[2], "down"),
            (x, "input"),
            (counts, "counts"),
            (ids, "ids"),
            (weights, "route weights"),
        ];
        for t in [plan, packed, inner, output] {
            check_disjoint(KERNEL, t, &inputs)?;
        }
        check_disjoint(
            KERNEL,
            plan,
            &[(packed, "packed"), (inner, "inner"), (output, "output")],
        )?;
        check_disjoint(KERNEL, inner, &[(packed, "packed"), (output, "output")])?;
        let suffix = |w: &MetalTensor| moe_grouped_generic_layout(w.dtype).unwrap().suffix;
        let family = if g.cooperative { "cooperative_" } else { "" };
        let gu_family = if g.gather_f32 {
            "cooperative_gather_"
        } else {
            family
        };
        let pipelines = [
            ctx.tensor_pipeline("v41_sorted_plan")?,
            ctx.tensor_pipeline("v41_sorted_pack")?,
            ctx.tensor_pipeline(&format!("v41_sorted_gu_{gu_family}{}", suffix(banks[0])))?,
            ctx.tensor_pipeline(&format!("v41_sorted_down_{family}{}", suffix(banks[2])))?,
        ];
        for (i, p) in pipelines[2..].iter().enumerate() {
            let shared = if i == 0 && g.gather_f32 {
                GATHER_SHARED_BYTES
            } else {
                SHARED_BYTES
            };
            if p.threadExecutionWidth() != 32
                || p.maxTotalThreadsPerThreadgroup() < if g.cooperative { 256 } else { 128 }
                || p.staticThreadgroupMemoryLength()
                    .checked_add(shared)
                    .is_none_or(|n| n > ctx.device.maxThreadgroupMemoryLength())
            {
                return Err(bad_shape(KERNEL, "unsupported tensor pipeline resources"));
            }
        }
        Ok(Self {
            pipelines,
            banks,
            x,
            counts,
            ids,
            weights,
            plan,
            packed,
            inner,
            output,
            g,
            slots,
        })
    }

    /// Plan, pack, then fused clipped SwiGLU/weight/half store. No host readback.
    pub(crate) fn encode_gate_up(&self, enc: &KernelEncoder) {
        let g = self.g;
        enc.set_pipeline(&self.pipelines[0]);
        enc.set_tensor(0, self.counts);
        enc.set_tensor(1, self.plan);
        enc.set_bytes(2, &[g.experts as u32, g.rows as u32, g.topk as u32, 0]);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        if !g.gather_f32 {
            enc.set_pipeline(&self.pipelines[1]);
            for (i, t) in [self.x, self.ids, self.plan, self.packed]
                .iter()
                .enumerate()
            {
                enc.set_tensor(i, t);
            }
            enc.set_bytes(
                4,
                &[
                    g.hidden as u32,
                    g.rows as u32,
                    g.experts as u32,
                    g.topk as u32,
                ],
            );
            enc.dispatch(
                MTLSize {
                    width: g.hidden.div_ceil(256),
                    height: self.slots,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        enc.set_pipeline(&self.pipelines[2]);
        enc.set_bytes(
            0,
            &[
                g.hidden as u32,
                g.inner as u32,
                g.experts as u32,
                g.rows as u32,
            ],
        );
        enc.set_tensor(1, self.banks[0]);
        enc.set_tensor(2, self.banks[1]);
        enc.set_tensor(3, if g.gather_f32 { self.x } else { self.packed });
        enc.set_tensor(4, self.plan);
        enc.set_tensor(5, self.ids);
        enc.set_bytes(7, &[self.slots as u32, g.topk as u32]);
        enc.set_bytes(8, &g.clamp);
        enc.set_tensor(9, self.weights);
        enc.set_tensor(10, self.inner);
        self.dispatch(
            enc,
            g.inner / 64,
            if g.gather_f32 {
                GATHER_SHARED_BYTES
            } else {
                SHARED_BYTES
            },
        );
    }

    pub(crate) fn encode_down(&self, enc: &KernelEncoder) {
        let g = self.g;
        enc.set_pipeline(&self.pipelines[3]);
        enc.set_bytes(
            0,
            &[
                g.inner as u32,
                g.hidden as u32,
                g.experts as u32,
                g.rows as u32,
            ],
        );
        enc.set_tensor(1, self.banks[2]);
        enc.set_tensor(3, self.inner);
        enc.set_tensor(4, self.plan);
        enc.set_tensor(5, self.ids);
        enc.set_tensor(6, self.output);
        enc.set_bytes(7, &[self.slots as u32, g.topk as u32]);
        self.dispatch(enc, g.hidden / 128, SHARED_BYTES);
    }

    fn dispatch(&self, enc: &KernelEncoder, columns: usize, shared: usize) {
        enc.set_threadgroup_memory(0, shared);
        enc.dispatch(
            MTLSize {
                width: columns,
                height: self.g.experts + self.slots.div_ceil(64),
                depth: 1,
            },
            MTLSize {
                width: if self.g.cooperative { 256 } else { 128 },
                height: 1,
                depth: 1,
            },
        );
    }
}
