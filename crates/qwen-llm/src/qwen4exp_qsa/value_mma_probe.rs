//! TEST ONLY selected Flash prefill value leaf. No routing or runtime knob.
//!
//! Native test integration: allocate `Scratch::new(ctx, max_queries)` before
//! encoding/measurement. After the unchanged selected-ID/logits work, call the
//! free `encode` function instead of the packed softmax/value leaf, on a serial
//! encoder. Its signature mirrors that leaf with `&Scratch` appended.
//! Logits are destructively converted to masses, as in production. Denominators
//! for active rows are fully overwritten; unused scratch rows are untouched.
//! Keep Scratch immutably borrowed through command completion. Reuse it between
//! serial bands/layers, not across concurrent commands. The caller owns the
//! test-only binding hook and its completion/lifetime discipline.
//!
//! Extra logical scratch: 96 * max_queries bytes (3072 at N32), device-priced as
//! one allocation. Two dispatches: GQA4 stats (3,2,N)/256 threads/144 TGM bytes;
//! gathered PV (4,2,N)/128 threads/10240 TGM bytes. F32 operands and K32 partials
//! preserve operand precision, not the scalar PV reduction order.
//! `Scratch::new_gqa12` selects the single scalar-reuse falsifier instead: the
//! same stats stage, then PV (1,2,N)/256 threads/no TGM. Its twelve named F32
//! accumulators retain production slot order; exact output bits are the leaf
//! gate. Do not promote either candidate if this arithmetic-preserving gate fails.

use super::*;
use crate::metal::Pipeline;
use objc2_metal::MTLCommandQueue;

pub(crate) const SOFTMAX: &str = "kernel_qwen4exp_qsa_selected_softmax_gqa4_f32_test";
pub(crate) const VALUE: &str = "kernel_qwen4exp_qsa_selected_value_m16d64k32_f32_test";
pub(crate) const VALUE_GQA12: &str = "kernel_qwen4exp_qsa_selected_value_gqa12_f32_test";
const BASELINE: &str = "kernel_qwen4exp_qsa_attention_softmax_value_packed_gqa4_f16";
const STATS_TGM: usize = 4 * 9 * 4;
const VALUE_TGM: usize = (16 * 32 + 32 * 64) * 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueVariant {
    Mma,
    Gqa12,
}

impl ValueVariant {
    fn kernel(self) -> &'static str {
        match self {
            Self::Mma => VALUE,
            Self::Gqa12 => VALUE_GQA12,
        }
    }

    fn launch(self, query_count: usize) -> (MTLSize, usize, usize) {
        match self {
            Self::Mma => (size(4, 2, query_count), 128, VALUE_TGM),
            Self::Gqa12 => (size(1, 2, query_count), 256, 0),
        }
    }
}

pub(crate) fn scratch_logical_bytes(query_count: usize) -> Result<u64, Qwen4ExpQsaError> {
    if !(1..=32).contains(&query_count) {
        return invalid("value MMA probe requires 1..=32 selected prefill rows");
    }
    Ok(query_count as u64 * 24 * 4)
}

/// Caller-owned storage for serial selected-prefill commands. No Clone: the
/// native test binding must borrow this owner until its command has completed.
pub(crate) struct Scratch {
    probe: ValueMmaProbe,
    denominators: MetalTensor,
    max_queries: usize,
}

impl Scratch {
    /// Prepare both pipelines, then price/admit and allocate one denominator
    /// buffer under the normal allocation transaction and dynamic reserve.
    pub(crate) fn new(ctx: &MetalContext, max_queries: usize) -> Result<Self, Qwen4ExpQsaError> {
        Self::new_variant(ctx, max_queries, ValueVariant::Mma)
    }

    /// Same storage/lifetime contract as `new`, but scalar GQA12 PV. This is the
    /// only arithmetic-preserving follow-up, gated on production-exact leaf bits.
    pub(crate) fn new_gqa12(
        ctx: &MetalContext,
        max_queries: usize,
    ) -> Result<Self, Qwen4ExpQsaError> {
        Self::new_variant(ctx, max_queries, ValueVariant::Gqa12)
    }

    fn new_variant(
        ctx: &MetalContext,
        max_queries: usize,
        variant: ValueVariant,
    ) -> Result<Self, Qwen4ExpQsaError> {
        let logical = scratch_logical_bytes(max_queries)?;
        let probe = ValueMmaProbe::prepare(ctx, variant)?;
        let priced = ctx.price_shared_buffer_upper(logical).map_err(|error| {
            Qwen4ExpQsaError::Invalid(format!("value MMA scratch pricing: {error}"))
        })?;
        let _transaction = ctx.begin_allocation_transaction();
        let before = ctx.memory_signals();
        let reserve = crate::qwen4exp_text_session::QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES;
        let admission = crate::metal::evaluate_metal_memory_admission(
            priced.priced_upper_bytes,
            reserve,
            before,
            true,
        );
        if let Some(denied) = admission.refusal() {
            return invalid(format!("value MMA scratch: {denied}"));
        }
        let denominators =
            MetalTensor::zeros_dtype_unstaged(ctx, vec![24, max_queries as u64], GgmlType::F32)?;
        let after = ctx.memory_signals();
        if after
            .current_allocated_bytes
            .saturating_sub(before.current_allocated_bytes)
            > priced.priced_upper_bytes
        {
            return invalid("value MMA scratch allocation exceeded device-priced upper bound");
        }
        let remaining = crate::metal::evaluate_metal_memory_admission(0, reserve, after, true);
        if let Some(denied) = remaining.refusal() {
            return invalid(format!(
                "value MMA scratch post-allocation reserve: {denied}"
            ));
        }
        eprintln!(
            "value_mma scratch variant={variant:?} max_queries={max_queries} logical={logical} priced_upper={} alignment={}",
            priced.priced_upper_bytes, priced.alignment
        );
        Ok(Self {
            probe,
            denominators,
            max_queries,
        })
    }
}

/// Same argument order and result type as the parent's packed value leaf, with
/// caller-owned Scratch appended. All checks finish before either dispatch;
/// there is no allocation, prefix-view creation, or pipeline lookup here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query_gate_projection: &MetalTensor,
    value_cache: &MetalTensor,
    token_ids: &MetalTensor,
    selected_count: &MetalTensor,
    selector_status: &MetalTensor,
    logits: &MetalTensor,
    attention: &MetalTensor,
    geometry: QwenSparseAttentionMetalGeometry,
    start_position: usize,
    query_count: usize,
    scratch: &Scratch,
) -> Result<(), MetalError> {
    scratch_logical_bytes(scratch.max_queries).map_err(metal_error)?;
    if query_count > scratch.max_queries {
        return Err(MetalError::BadShape {
            kernel: VALUE,
            detail: format!(
                "query count {query_count} exceeds scratch capacity {}",
                scratch.max_queries
            ),
        });
    }
    require_tensor(
        "value MMA scratch denominators",
        &scratch.denominators,
        GgmlType::F32,
        &[24, scratch.max_queries as u64],
        true,
    )
    .map_err(metal_error)?;
    scratch
        .probe
        .encode(
            ctx,
            enc,
            query_gate_projection,
            value_cache,
            token_ids,
            selected_count,
            selector_status,
            logits,
            attention,
            &scratch.denominators,
            geometry,
            start_position,
            query_count,
        )
        .map_err(metal_error)
}

fn metal_error(error: Qwen4ExpQsaError) -> MetalError {
    match error {
        Qwen4ExpQsaError::Metal(error) => error,
        error => MetalError::BadShape {
            kernel: VALUE,
            detail: error.to_string(),
        },
    }
}

fn validate_shape(
    g: QwenSparseAttentionMetalGeometry,
    start_position: usize,
    query_count: usize,
) -> Result<(), Qwen4ExpQsaError> {
    scratch_logical_bytes(query_count)?;
    if (
        g.query_heads,
        g.kv_heads,
        g.head_dim,
        g.token_budget,
        g.ratio,
    ) != (24, 2, 256, 2048, 4)
    {
        return invalid("value MMA probe requires Q24/KV2/D256/budget2048/ratio4");
    }
    let end = start_position
        .checked_add(query_count)
        .ok_or_else(|| Qwen4ExpQsaError::Invalid("value MMA probe query end overflow".into()))?;
    if start_position < 2051 || end > g.capacity || g.capacity > i32::MAX as usize {
        return invalid(
            "value MMA probe requires a selected suffix within an i32-addressable cache",
        );
    }
    g.validate(g.capacity)?;
    Ok(())
}

#[derive(Clone)]
struct ValueMmaProbe {
    softmax: Pipeline,
    value: Pipeline,
    device: u64,
    variant: ValueVariant,
}

impl ValueMmaProbe {
    /// Resolve research pipelines and check hardware capacity; no GPU dispatch.
    fn prepare(ctx: &MetalContext, variant: ValueVariant) -> Result<Self, Qwen4ExpQsaError> {
        let softmax = ctx.pipeline(SOFTMAX)?;
        let value = ctx.pipeline(variant.kernel())?;
        let (_, value_threads, value_tgm) = variant.launch(1);
        for (pso, threads, tgm) in [
            (&softmax, 256, STATS_TGM),
            (&value, value_threads, value_tgm),
        ] {
            let total = pso.staticThreadgroupMemoryLength().checked_add(tgm);
            if pso.threadExecutionWidth() != 32
                || pso.maxTotalThreadsPerThreadgroup() < threads
                || total.is_none_or(|bytes| bytes > ctx.device.maxThreadgroupMemoryLength())
            {
                return invalid("value MMA probe requires SIMD32 and sufficient threads/TGM");
            }
        }
        Ok(Self {
            softmax,
            value,
            device: ctx.device.registryID(),
            variant,
        })
    }

    /// All refusals precede encoder mutation; allocations/admission are caller-owned.
    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        query_gate_projection: &MetalTensor,
        value_cache: &MetalTensor,
        token_ids: &MetalTensor,
        selected_count: &MetalTensor,
        selector_status: &MetalTensor,
        logits: &MetalTensor,
        attention: &MetalTensor,
        denominators: &MetalTensor,
        g: QwenSparseAttentionMetalGeometry,
        start_position: usize,
        query_count: usize,
    ) -> Result<(), Qwen4ExpQsaError> {
        validate_shape(g, start_position, query_count)?;
        validate_encoder(ctx, enc)?;
        if self.device != ctx.device.registryID() {
            return invalid("value MMA probe pipelines belong to another device");
        }
        let n = query_count as u64;
        let denominator_capacity = match denominators.shape.as_slice() {
            [24, capacity] if (n..=32).contains(capacity) => *capacity,
            _ => return invalid("value MMA denominators must cover the active rows within 1..=32"),
        };
        for (name, tensor, dtype, shape, writable) in [
            (
                "query/gate",
                query_gate_projection,
                GgmlType::F32,
                &[12288, n][..],
                false,
            ),
            (
                "value cache",
                value_cache,
                GgmlType::F16,
                &[256, 2, g.capacity as u64][..],
                false,
            ),
            ("token IDs", token_ids, GgmlType::I32, &[2051, n][..], false),
            (
                "selected counts",
                selected_count,
                GgmlType::I32,
                &[n][..],
                false,
            ),
            (
                "selector status",
                selector_status,
                GgmlType::I32,
                &[n][..],
                false,
            ),
            ("logits", logits, GgmlType::F32, &[2051, 24, n][..], true),
            ("attention", attention, GgmlType::F32, &[6144, n][..], true),
            (
                "denominators",
                denominators,
                GgmlType::F32,
                &[24, denominator_capacity][..],
                true,
            ),
        ] {
            require_tensor(name, tensor, dtype, shape, writable)?;
        }
        let tensors = [
            ("query/gate", query_gate_projection),
            ("value cache", value_cache),
            ("token IDs", token_ids),
            ("selected counts", selected_count),
            ("selector status", selector_status),
            ("logits", logits),
            ("attention", attention),
            ("denominators", denominators),
        ];
        require_same_device(ctx, &tensors)?;
        require_disjoint(&tensors)?;
        let args = packed_attention_args(g, start_position, query_count);
        let (value_grid, value_threads, value_tgm) = self.variant.launch(query_count);
        enc.set_pipeline(&self.softmax);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, selected_count);
        enc.set_tensor(2, selector_status);
        enc.set_tensor(3, logits);
        enc.set_tensor(4, denominators);
        enc.set_threadgroup_memory(0, STATS_TGM);
        enc.dispatch(size(3, 2, query_count), size(256, 1, 1));
        enc.set_pipeline(&self.value);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, query_gate_projection);
        enc.set_tensor(2, value_cache);
        enc.set_tensor(3, token_ids);
        enc.set_tensor(4, selected_count);
        enc.set_tensor(5, selector_status);
        enc.set_tensor(6, logits);
        enc.set_tensor(7, attention);
        enc.set_tensor(8, denominators);
        enc.set_threadgroup_memory(0, value_tgm);
        enc.dispatch(value_grid, size(value_threads, 1, 1));
        Ok(())
    }
}

fn size(width: usize, height: usize, depth: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}

// Leaf-only Fixture: no model weights, selector, QK, publication or planner.
// Unlike the full selected-attention fixture it can price every live allocation
// and exercise N32/8192 without allocating unrelated packed-session scratch.
struct Fixture {
    g: QwenSparseAttentionMetalGeometry,
    start: usize,
    n: usize,
    scratch: Scratch,
    gate: MetalTensor,
    cache: MetalTensor,
    ids: MetalTensor,
    counts: MetalTensor,
    status: MetalTensor,
    logit_bank: MetalTensor,
    logits: Vec<MetalTensor>,
    baseline: MetalTensor,
    candidate: MetalTensor,
    source: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
enum Case {
    Ordinary,
    InvalidIds,
    NoIds,
    AllMasked,
    Metadata,
    Cancellation,
}

impl Fixture {
    fn new(ctx: &MetalContext, start: usize, n: usize, layers: usize, case: Case) -> Self {
        Self::new_variant(ctx, start, n, layers, case, ValueVariant::Mma)
    }

    fn new_variant(
        ctx: &MetalContext,
        start: usize,
        n: usize,
        layers: usize,
        case: Case,
        variant: ValueVariant,
    ) -> Self {
        assert!((1..=12).contains(&layers));
        let config = Qwen4ExpConfig::flash_next_reference();
        let capacity = (start + n + 16).next_multiple_of(4);
        let g = QwenSparseAttentionMetalGeometry::from_config(&config, 3, capacity).unwrap();
        validate_shape(g, start, n).unwrap();
        let scratch = match variant {
            ValueVariant::Mma => Scratch::new(ctx, 32),
            ValueVariant::Gqa12 => Scratch::new_gqa12(ctx, 32),
        }
        .unwrap();
        let baseline_pso = ctx.pipeline(BASELINE).unwrap();
        assert_eq!(baseline_pso.threadExecutionWidth(), 32);
        assert!(baseline_pso.maxTotalThreadsPerThreadgroup() >= 256);
        assert!(
            baseline_pso.staticThreadgroupMemoryLength() + STATS_TGM
                <= ctx.device.maxThreadgroupMemoryLength()
        );
        let logit_count = n * 24 * 2051;
        let logical = [
            n * 12288 * 4,
            g.capacity * 512 * 2,
            n * 2051 * 4,
            n * 4,
            n * 4,
            logit_count * layers * 4,
            n * 6144 * 4,
            n * 6144 * 4,
        ];
        let priced = logical
            .iter()
            .try_fold(0u64, |sum, &bytes| {
                sum.checked_add(
                    ctx.price_shared_buffer_upper((bytes + 24) as u64)
                        .unwrap()
                        .priced_upper_bytes,
                )
            })
            .unwrap();
        let _transaction = ctx.begin_allocation_transaction();
        let before = ctx.memory_signals();
        let reserve = crate::qwen4exp_text_session::QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES;
        // Include host input/staging peaks, F64 oracle, mass/output readbacks.
        let cpu_upper = logical
            .iter()
            .map(|&b| b as u64)
            .sum::<u64>()
            .checked_mul(6)
            .unwrap();
        let admission = crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(
            priced, cpu_upper, reserve, before, true,
        );
        assert!(
            admission.is_admitted(),
            "value MMA fixture admission: {admission:?}"
        );

        let gate_values: Vec<f32> = (0..n * 12288)
            .map(|i| match i % 31 {
                0 => -100.0,
                1 => 100.0,
                _ => sample(i, 71) * 4.0,
            })
            .collect();
        let cache_values: Vec<half::f16> = (0..g.capacity * 512)
            .map(|i| {
                let value = if i / 512 >= start + n {
                    f32::NAN // Future/cache padding must never be gathered.
                } else if matches!(case, Case::Cancellation) {
                    let sign = if (i / 512) % 2 == 0 { 1.0 } else { -1.0 };
                    sign * (8.0 + sample(i % 512, 13) * 0.01)
                } else {
                    sample(i, 29)
                };
                half::f16::from_f32(value)
            })
            .collect();
        let mut ids = vec![i32::MAX; n * 2051];
        let mut counts = vec![512i32; n];
        let mut statuses = vec![0i32; n];
        let mut source = vec![89.0f32; logit_count]; // Poison inactive slots.
        for q in 0..n {
            let sequence = start + q + 1;
            let active = 2048 + sequence % 4;
            if matches!(case, Case::Metadata) {
                counts[q] = [-1, 0, 511, 513, i32::MAX, 512, 512, 512][q % 8];
                statuses[q] = [0, 0, 0, 0, 0, -1, 7, 0][q % 8];
            }
            for slot in 0..active {
                let position = if slot < 2048 {
                    (slot / 4 * (sequence / 4 - 1) / 511) * 4 + slot % 4
                } else {
                    sequence / 4 * 4 + slot - 2048
                };
                ids[q * 2051 + slot] = if matches!(case, Case::NoIds) {
                    -1
                } else if matches!(case, Case::InvalidIds) {
                    match slot % 37 {
                        0 => -1,
                        1 => g.capacity as i32,
                        2 => sequence as i32,
                        3 => i32::MAX,
                        _ => position as i32,
                    }
                } else {
                    position as i32
                };
                for h in 0..24 {
                    source[(q * 24 + h) * 2051 + slot] = if matches!(case, Case::AllMasked) {
                        f32::NEG_INFINITY
                    } else if h % 12 == 0 || matches!(case, Case::Cancellation) {
                        0.0
                    } else if h % 12 == 1 {
                        if slot == active - 1 { 30.0 } else { -30.0 }
                    } else if matches!(case, Case::InvalidIds) && slot % 37 == 0 {
                        f32::NEG_INFINITY
                    } else {
                        sample((q * 24 + h) * 2051 + slot, 47) * 3.0
                    };
                }
            }
        }
        let gate = guarded(ctx, &gate_values, vec![12288, n as u64], GgmlType::F32);
        let cache = guarded(
            ctx,
            &cache_values,
            vec![256, 2, g.capacity as u64],
            GgmlType::F16,
        );
        let ids = guarded(ctx, &ids, vec![2051, n as u64], GgmlType::I32);
        let counts = guarded(ctx, &counts, vec![n as u64], GgmlType::I32);
        let status = guarded(ctx, &statuses, vec![n as u64], GgmlType::I32);
        let logit_bank = guarded(
            ctx,
            &vec![f32::NAN; logit_count * layers],
            vec![logit_count as u64, layers as u64],
            GgmlType::F32,
        );
        let logits = (0..layers)
            .map(|layer| {
                logit_bank.view_subrange((layer * logit_count) as u64, vec![2051, 24, n as u64])
            })
            .collect();
        let baseline = guarded(
            ctx,
            &vec![f32::NAN; n * 6144],
            vec![6144, n as u64],
            GgmlType::F32,
        );
        let candidate = guarded(
            ctx,
            &vec![f32::NAN; n * 6144],
            vec![6144, n as u64],
            GgmlType::F32,
        );
        let after = ctx.memory_signals();
        assert!(
            after
                .current_allocated_bytes
                .saturating_sub(before.current_allocated_bytes)
                <= priced
        );
        assert!(
            crate::metal::evaluate_metal_memory_admission(0, reserve, after, true).is_admitted()
        );
        let (_, _, value_tgm) = variant.launch(n);
        eprintln!(
            "value_mma fixture variant={variant:?} N={n} start={start} layers={layers} case={case:?} input_output_allocation_upper={priced} extra_denominator_capacity=32 extra_denominator_logical=3072 tgm={value_tgm}"
        );
        Self {
            g,
            start,
            n,
            scratch,
            gate,
            cache,
            ids,
            counts,
            status,
            logit_bank,
            logits,
            baseline,
            candidate,
            source,
        }
    }

    fn reset(&self) {
        for logits in &self.logits {
            write(logits, &self.source);
        }
        write(&self.scratch.denominators, &[f32::NAN; 32 * 24]);
    }

    // No tensor allocation, host reset, readback, QK or copy dispatch in here.
    // Each of the 12 layer-equivalent leaves consumes a fresh logit bank view.
    // N32 timing's 12-bank input is 75608064 bytes, shared by both arms; this
    // is a cache-hot leaf screen, not twelve distinct model-layer KV caches.
    fn run(&self, ctx: &MetalContext, candidate: bool) -> f64 {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for logits in &self.logits {
            if candidate {
                encode(
                    ctx,
                    &encoder,
                    &self.gate,
                    &self.cache,
                    &self.ids,
                    &self.counts,
                    &self.status,
                    logits,
                    &self.candidate,
                    self.g,
                    self.start,
                    self.n,
                    &self.scratch,
                )
                .unwrap();
            } else {
                // Call production GQA4 directly, independent of runtime env routing.
                encode_attention_softmax_value_packed_gqa4(
                    ctx,
                    &encoder,
                    &self.gate,
                    &self.cache,
                    &self.ids,
                    &self.counts,
                    &self.status,
                    logits,
                    &self.baseline,
                    self.g,
                    self.start,
                    self.n,
                )
                .unwrap();
            }
        }
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("value MMA command completed");
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none(), "{:?}", command.error());
        let ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3 / self.logits.len() as f64;
        assert!(ms.is_finite() && ms > 0.0, "invalid GPU command time {ms}");
        ms
    }

    fn check_guards(&self) {
        for tensor in [
            &self.gate,
            &self.cache,
            &self.ids,
            &self.counts,
            &self.status,
            &self.logit_bank,
            &self.baseline,
            &self.candidate,
        ] {
            unsafe {
                let base = tensor.buffer.contents().as_ptr().cast::<u8>();
                assert_eq!(std::slice::from_raw_parts(base, 4), &[0xA5; 4]);
                let end = base.add(tensor.buffer.length() - 20);
                assert_eq!(std::slice::from_raw_parts(end, 20), &[0xA5; 20]);
            }
        }
        assert!(
            read::<f32>(&self.scratch.denominators)[self.n * 24..]
                .iter()
                .all(|v| v.is_nan()),
            "unused scratch rows were overwritten"
        );
    }

    fn reference(&self) -> Vec<f64> {
        let values: Vec<f32> = read::<half::f16>(&self.cache)
            .into_iter()
            .map(|v| v.to_f32())
            .collect();
        let gates = read::<f32>(&self.gate);
        let ids = read::<i32>(&self.ids);
        let counts = read::<i32>(&self.counts);
        let status = read::<i32>(&self.status);
        let mut output = vec![0.0; self.n * 6144];
        for q in 0..self.n {
            if counts[q] != 512 || status[q] != 0 {
                continue;
            }
            let sequence = self.start + q + 1;
            let active = 2048 + sequence % 4;
            for h in 0..24 {
                let head = q * 24 + h;
                reference_head(
                    &self.source[head * 2051..head * 2051 + active],
                    &ids[q * 2051..q * 2051 + active],
                    &values,
                    h / 12,
                    sequence,
                    &gates[head * 512 + 256..head * 512 + 512],
                    &mut output[head * 256..head * 256 + 256],
                );
            }
        }
        output
    }
}

fn sample(index: usize, seed: u32) -> f32 {
    let mut x = (index as u32).wrapping_add(seed).wrapping_mul(0x9e37_79b9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85eb_ca6b);
    ((x >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
}

fn guarded<T: bytemuck::Pod>(
    ctx: &MetalContext,
    data: &[T],
    shape: Vec<u64>,
    dtype: GgmlType,
) -> MetalTensor {
    let payload: &[u8] = bytemuck::cast_slice(data);
    assert_eq!(
        payload.len(),
        crate::metal::checked_ggml_shape_bytes(&shape, dtype)
            .unwrap()
            .1
    );
    let mut bytes = vec![0xA5u8; payload.len() + 24];
    bytes[4..4 + payload.len()].copy_from_slice(payload);
    MetalTensor {
        buffer: ctx.buffer_from(&bytes).unwrap(),
        offset: 4,
        shape,
        dtype,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

fn write<T: bytemuck::Pod>(tensor: &MetalTensor, data: &[T]) {
    let bytes = bytemuck::cast_slice(data);
    assert_eq!(bytes.len() as u64, tensor.n_bytes());
    assert!(tensor.is_writable());
    unsafe {
        let dst = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
    }
}

fn read<T: bytemuck::Pod>(tensor: &MetalTensor) -> Vec<T> {
    assert_eq!(
        tensor.n_bytes() as usize,
        tensor.n_elements() as usize * size_of::<T>()
    );
    assert!(tensor.offset.checked_add(tensor.n_bytes()).unwrap() <= tensor.buffer.length() as u64);
    unsafe {
        let src = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<T>();
        std::slice::from_raw_parts(src, tensor.n_elements() as usize).to_vec()
    }
}

fn reference_head(
    logits: &[f32],
    ids: &[i32],
    values: &[f32],
    kv: usize,
    sequence: usize,
    gates: &[f32],
    output: &mut [f64],
) {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let masses: Vec<f64> = logits
        .iter()
        .map(|&v| (f64::from(v) - maximum).exp())
        .collect();
    let denominator: f64 = masses.iter().sum();
    output.fill(0.0);
    if !denominator.is_finite() || denominator <= 0.0 {
        return;
    }
    for (&id, &mass) in ids.iter().zip(&masses) {
        if id < 0 || id as usize >= values.len() / 512 || id as usize >= sequence {
            continue;
        }
        let base = (id as usize * 2 + kv) * 256;
        for (d, total) in output.iter_mut().enumerate() {
            *total += mass * f64::from(values[base + d]);
        }
    }
    for (value, &gate) in output.iter_mut().zip(gates) {
        *value = *value / denominator / (1.0 + (-f64::from(gate)).exp());
    }
}

fn compare(actual: &[f32], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len());
    let (mut error, mut energy, mut maximum) = (0.0f64, 0.0f64, 0.0f64);
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "{label}: nonfinite at {i}");
        let delta = f64::from(a) - b;
        maximum = maximum.max(delta.abs());
        error += delta * delta;
        energy += b * b;
        assert!(
            delta.abs() <= 3e-5 + 3e-5 * b.abs(),
            "{label}[{i}]: {a} != {b}"
        );
    }
    let relative = (error / energy.max(1e-20)).sqrt();
    assert!(
        relative <= 3e-5 || error.sqrt() <= 3e-6,
        "{label}: relative L2={relative} absolute L2={}",
        error.sqrt()
    );
    eprintln!("value_mma {label}: max_abs={maximum:.9e} relative_l2={relative:.9e}");
}

fn assert_output_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "GQA12 nonfinite output at {i}: {a} / {b}"
        );
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "GQA12 exact leaf gate failed at {i}: {a} != {b}; close both candidates, do not retune"
        );
    }
}

fn require_gpu_workflow() {
    for name in ["MTL_DEBUG_LAYER", "QWEN_METAL_LEASE_WAIT"] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok("1"),
            "value MMA GPU tests require {name}=1 before process startup"
        );
    }
}

#[test]
fn bounded_shape_and_scratch_contract() {
    assert_eq!(scratch_logical_bytes(32).unwrap(), 3072);
    for n in [0, 33, usize::MAX] {
        assert!(scratch_logical_bytes(n).is_err());
    }
    let config = Qwen4ExpConfig::flash_next_reference();
    let mut g = QwenSparseAttentionMetalGeometry::from_config(&config, 3, 8256).unwrap();
    for start in [2051, 8192] {
        for n in [1, 7, 31, 32] {
            validate_shape(g, start, n).unwrap();
        }
    }
    for (start, n) in [(2050, 32), (8240, 32), (usize::MAX, 1), (8192, 33)] {
        assert!(validate_shape(g, start, n).is_err());
    }
    g.head_dim = 128;
    assert!(validate_shape(g, 8192, 32).is_err());
    assert_eq!(size_of::<PackedAttentionArgs>(), 40);
    assert_eq!(VALUE_TGM, 10240);
    for (variant, kernel, width, threads, tgm) in [
        (ValueVariant::Mma, VALUE, 4, 128, VALUE_TGM),
        (ValueVariant::Gqa12, VALUE_GQA12, 1, 256, 0),
    ] {
        assert_eq!(variant.kernel(), kernel);
        let (grid, actual_threads, actual_tgm) = variant.launch(32);
        assert_eq!((grid.width, grid.height, grid.depth), (width, 2, 32));
        assert_eq!((actual_threads, actual_tgm), (threads, tgm));
    }
}

#[test]
fn gqa12_exact_gate_rejects_nonfinite_and_bit_drift() {
    assert_output_bits(&[0.0, -0.0, 1.0, -2.0], &[0.0, -0.0, 1.0, -2.0]);
    for (actual, expected) in [
        (-0.0, 0.0),
        (f32::from_bits(1.0f32.to_bits() + 1), 1.0),
        (f32::NAN, f32::NAN),
        (f32::INFINITY, f32::INFINITY),
    ] {
        assert!(std::panic::catch_unwind(|| assert_output_bits(&[actual], &[expected])).is_err());
    }
}

#[test]
fn cpu_reference_retains_invalid_id_mass_and_masked_zero_semantics() {
    let mut values = vec![0.0; 3 * 512];
    values[..256].fill(2.0);
    values[512..768].fill(-1.0);
    let mut output = vec![f64::NAN; 256];
    reference_head(
        &[0.0; 4],
        &[0, 1, -1, 2],
        &values,
        0,
        2,
        &[0.0; 256],
        &mut output,
    );
    assert_eq!(output, vec![0.125; 256]);
    reference_head(
        &[f32::NEG_INFINITY; 4],
        &[0, 1, -1, 3],
        &values,
        0,
        2,
        &[0.0; 256],
        &mut output,
    );
    assert_eq!(output, vec![0.0; 256]);
}

#[test]
#[ignore = "production GPU lease; selected prefill N1/7/31/32 CPU + production leaf qualification"]
fn selected_value_mma_correctness() {
    require_gpu_workflow();
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    for start in [2051, 8192] {
        for n in [1, 7, 31, 32] {
            qualify(&ctx, start, n, Case::Ordinary);
        }
        for case in [
            Case::InvalidIds,
            Case::NoIds,
            Case::AllMasked,
            Case::Metadata,
            Case::Cancellation,
        ] {
            qualify(&ctx, start, 32, case);
        }
    }
}

#[test]
#[ignore = "production GPU lease; GQA12 exact output bits vs production GQA4, existing 18 cases"]
fn selected_value_gqa12_correctness() {
    require_gpu_workflow();
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    for start in [2051, 8192] {
        for n in [1, 7, 31, 32] {
            qualify_fixture(
                &ctx,
                Fixture::new_variant(&ctx, start, n, 1, Case::Ordinary, ValueVariant::Gqa12),
                Case::Ordinary,
            );
        }
        for case in [
            Case::InvalidIds,
            Case::NoIds,
            Case::AllMasked,
            Case::Metadata,
            Case::Cancellation,
        ] {
            qualify_fixture(
                &ctx,
                Fixture::new_variant(&ctx, start, 32, 1, case, ValueVariant::Gqa12),
                case,
            );
        }
    }
}

fn qualify(ctx: &MetalContext, start: usize, n: usize, case: Case) {
    qualify_fixture(ctx, Fixture::new(ctx, start, n, 1, case), case);
}

fn qualify_fixture(ctx: &MetalContext, f: Fixture, case: Case) {
    qualify_fixture_with_product(ctx, f, case, None);
}

fn qualify_fixture_with_product(
    ctx: &MetalContext,
    f: Fixture,
    case: Case,
    product: Option<&selected_value::Pipelines>,
) {
    let (start, n) = (f.start, f.n);
    let variant = f.scratch.probe.variant;
    if start == 2051 && n == 1 {
        check_refusals(ctx, &f);
    }
    let expected = f.reference();
    f.reset();
    f.run(ctx, false);
    let baseline = read::<f32>(&f.baseline);
    compare(&baseline, &expected, "production/F64");
    let masses = read::<f32>(&f.logits[0]);
    f.reset();
    crate::metal::dispatch_census_begin();
    if let Some(pipelines) = product {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        pipelines
            .encode(
                ctx,
                &encoder,
                &f.gate,
                &f.cache,
                &f.ids,
                &f.counts,
                &f.status,
                &f.logits[0],
                &f.candidate,
                &f.scratch.denominators,
                f.g,
                f.start,
                f.n,
            )
            .unwrap();
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("product selected value command completed");
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none(), "{:?}", command.error());
    } else {
        f.run(ctx, true);
    }
    let census = crate::metal::dispatch_census_take();
    let kernels = if product.is_some() {
        [selected_value::SOFTMAX, selected_value::VALUE]
    } else {
        [SOFTMAX, variant.kernel()]
    };
    assert_eq!(
        census.iter().map(|r| r.kernel.as_str()).collect::<Vec<_>>(),
        kernels
    );
    let candidate = read::<f32>(&f.candidate);
    if product.is_some() || variant == ValueVariant::Gqa12 {
        assert_output_bits(&candidate, &baseline);
        let pv = &census[1];
        assert_eq!(
            (
                pv.grid_width,
                pv.grid_height,
                pv.grid_depth,
                pv.threads_width,
                pv.threads_height,
                pv.threads_depth
            ),
            (1, 2, n as u64, 256, 1, 1)
        );
        compare(&candidate, &expected, "GQA12/F64");
    } else {
        compare(&candidate, &expected, "MMA/F64");
        compare(
            &candidate,
            &baseline.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            "MMA/production",
        );
    }
    for (i, (a, b)) in read::<f32>(&f.logits[0]).iter().zip(&masses).enumerate() {
        assert!(
            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
            "softmax mass/tail changed at {i}: {a} != {b}"
        );
    }
    let denominators = read::<f32>(&f.scratch.denominators);
    assert!(denominators[..n * 24].iter().all(|v| !v.is_nan()) || matches!(case, Case::AllMasked));
    for (q, (&count, &status)) in read::<i32>(&f.counts)
        .iter()
        .zip(read::<i32>(&f.status).iter())
        .enumerate()
    {
        if count != 512 || status != 0 {
            assert!(
                candidate[q * 6144..(q + 1) * 6144]
                    .iter()
                    .all(|&v| v == 0.0)
            );
            assert_eq!(&denominators[q * 24..(q + 1) * 24], &[0.0; 24]);
        }
    }
    f.check_guards();
}

#[test]
#[ignore = "production GPU lease; product metallib scalar GQA12 exact-bit qualification, 18 cases"]
fn selected_value_product_correctness() {
    require_gpu_workflow();
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let pipelines = selected_value::Pipelines::prepare(&ctx)
        .unwrap()
        .expect("product scalar GQA12 pipelines required for qualification");
    for start in [2051, 8192] {
        for n in [1, 7, 31, 32] {
            qualify_fixture_with_product(
                &ctx,
                Fixture::new_variant(&ctx, start, n, 1, Case::Ordinary, ValueVariant::Gqa12),
                Case::Ordinary,
                Some(&pipelines),
            );
        }
        for case in [
            Case::InvalidIds,
            Case::NoIds,
            Case::AllMasked,
            Case::Metadata,
            Case::Cancellation,
        ] {
            qualify_fixture_with_product(
                &ctx,
                Fixture::new_variant(&ctx, start, 32, 1, case, ValueVariant::Gqa12),
                case,
                Some(&pipelines),
            );
        }
    }
}

fn check_refusals(ctx: &MetalContext, f: &Fixture) {
    for max_queries in [0, 33, usize::MAX] {
        assert!(Scratch::new(ctx, max_queries).is_err());
        assert!(Scratch::new_gqa12(ctx, max_queries).is_err());
    }
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    crate::metal::dispatch_census_begin();
    for fault in 0..7 {
        let mut denominators = f.scratch.denominators.clone();
        let mut counts = f.counts.clone();
        match fault {
            0 => denominators.shape[0] = 23,
            1 => denominators.offset += 1,
            2 => denominators.offset = u64::MAX - 3,
            3 => denominators = f.gate.view_subrange(0, vec![24, 32]),
            4 => denominators.provenance = MetalTensorProvenance::OwnedWeightReadOnly,
            5 => counts.shape[0] = 0, // Absent metadata tensor: refuse before dispatch.
            _ => {}
        }
        let scratch = Scratch {
            probe: f.scratch.probe.clone(),
            denominators,
            max_queries: if fault == 6 { f.n } else { 32 },
        };
        assert!(
            encode(
                ctx,
                &encoder,
                &f.gate,
                &f.cache,
                &f.ids,
                &counts,
                &f.status,
                &f.logits[0],
                &f.candidate,
                f.g,
                f.start,
                if fault == 6 { f.n + 1 } else { f.n },
                &scratch,
            )
            .is_err(),
            "accepted binding fault {fault}"
        );
    }
    assert!(crate::metal::dispatch_census_take().is_empty());
    encoder.end(); // Intentionally never submitted.
}

#[test]
#[ignore = "production GPU lease; model-free N32 12-layer leaf-only GPU ABBA/BAAB timing"]
fn selected_value_mma_timing_n32() {
    require_gpu_workflow();
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    timing_n32(&ctx, ValueVariant::Mma);
}

#[test]
#[ignore = "production GPU lease; GQA12 N32 exact gate, then 12-layer leaf GPU ABBA/BAAB timing"]
fn selected_value_gqa12_timing_n32() {
    require_gpu_workflow();
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    timing_n32(&ctx, ValueVariant::Gqa12);
}

fn timing_n32(ctx: &MetalContext, variant: ValueVariant) {
    for start in [2051, 8192] {
        let f = Fixture::new_variant(ctx, start, 32, 12, Case::Ordinary, variant);
        let expected = f.reference();
        for arm in [false, true, true, false] {
            f.reset();
            f.run(ctx, arm);
        }
        let exact_reference = if variant == ValueVariant::Gqa12 {
            let baseline = read::<f32>(&f.baseline);
            assert_output_bits(&read::<f32>(&f.candidate), &baseline);
            Some(baseline)
        } else {
            None
        };
        let mut a = Vec::new();
        let mut b = Vec::new();
        for order in [[false, true, true, false], [true, false, false, true]] {
            let mut times = Vec::new();
            for arm in order {
                f.reset(); // Host reset is outside timed calls and GPU command times.
                let ms = f.run(ctx, arm);
                times.push(ms);
                if arm {
                    b.push(ms);
                } else {
                    a.push(ms);
                }
                let observed = read::<f32>(if arm { &f.candidate } else { &f.baseline });
                if let Some(reference) = &exact_reference {
                    assert_output_bits(&observed, reference);
                }
                compare(
                    &observed,
                    &expected,
                    if arm && variant == ValueVariant::Gqa12 {
                        "timed GQA12/F64"
                    } else if arm {
                        "timed MMA/F64"
                    } else {
                        "timed production/F64"
                    },
                );
                f.check_guards();
            }
            eprintln!(
                "value_mma timing variant={variant:?} start={start} N=32 layers=12 order={order:?} gpu_ms_per_layer={times:?}"
            );
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let control = mean(&a);
        let candidate = mean(&b);
        let spread = (a.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - a.iter().copied().fold(f64::INFINITY, f64::min))
            / control;
        eprintln!(
            "value_mma timing variant={variant:?} start={start} N=32 production_ms={control:.6} candidate_ms={candidate:.6} saved_pct={:.3} control_spread={spread:.6} inconclusive={} leaf_only=true extra_dispatch=1 extra_scratch_logical=3072",
            (1.0 - candidate / control) * 100.0,
            spread > 0.05
        );
    }
}
