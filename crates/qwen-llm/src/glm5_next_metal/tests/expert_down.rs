//! Bounded, diagnostic-only IQ3_S down replay. Production routers stay enabled.
//!
//! Owner invocation (new output file; ordinary production lease/admission):
//! ```sh
//! env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
//!   GLM53_EXPERT_DOWN_OUT=/tmp/glm53-expert-down.jsonl \
//!   cargo test --release -p qwen-llm --lib \
//!   glm5_next_metal::packed::expert_down::expert_down_packet \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```
//! Optional GLM53_EXPERT_DOWN_FULL=blanket|small_counts|both runs whole-model
//! ABBA only after all leaf execution/coverage/finite/timing checks pass. There is no
//! speed or numerical-difference promotion gate. Default: leaf screens only.
//! Repeated leaf bindings and CPU readback between attempts are diagnostic replay,
//! not attribution of all 39 layers. GPU duration includes execution stalls;
//! wall minus GPU duration does not isolate CPU work or memory wiring.
//!
//! GLM53_EXPERT_DOWN_DOMAIN=1 instead runs only whole-prefill SmallCounts
//! confirmation: two streams, N32/64/127/128/129/256/512, warm A/B then BAAB.
//! It bypasses ALL capture/leaf work and is incompatible with FULL. Optional
//! GLM53_EXPERT_DOWN_DOMAIN_LONG=1 adds an admitted N4096 / packed512 cell per
//! stream. N128/N512/N4096 reserve and exercise four teacher-forced decode steps.
//! The code stream appends the related renderer because the backend alone has
//! fewer than 4096 tokens. Both sources are fixed at compilation and hashed.
//! With DOMAIN=1, GLM53_EXPERT_DOWN_PRODUCTION=1 selects a final focused screen:
//! N32/128/512 (plus optional N4096), A explicitly forces incumbent, B follows
//! production selection via with_production. This does not change production policy.

use super::super::tests::{
    argmax, choice_regret, kl_divergence, long_qualification_text, perf_lease,
};
use super::*;
use crate::metal::iq3_s_down_retile::{DownRetile, encode_variant, with_production, with_variant};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::fs::File;
use std::io::Write;
use std::rc::Rc;
use std::time::Instant;

const H: usize = 4096;
const K: usize = 2048;
const E: usize = 288;
const TOP: usize = 8;
const READ_FLOATS: usize = 256 * 1024; // One MiB temporary readback.

#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Observation {
    completed: usize,
    valid: usize,
    gpu_ms: f64,
    iq3_layers: usize,
    iq4_layers: usize,
    captures: usize,
}

#[derive(Default)]
struct Observer {
    active: bool,
    capture: Option<Rc<Capture>>,
    observation: Observation,
}

thread_local! {
    static OBSERVER: RefCell<Observer> = RefCell::new(Observer::default());
}

fn observe<T>(capture: Option<Rc<Capture>>, body: impl FnOnce() -> T) -> (T, Observation) {
    struct Restore(Option<Observer>);
    impl Drop for Restore {
        fn drop(&mut self) {
            OBSERVER.with(|s| *s.borrow_mut() = self.0.take().unwrap());
        }
    }
    let old = OBSERVER.with(|s| {
        s.replace(Observer {
            active: true,
            capture,
            ..Observer::default()
        })
    });
    let _restore = Restore(Some(old));
    let result = body();
    (result, OBSERVER.with(|s| s.borrow().observation))
}

fn duration_ms(start: f64, end: f64) -> Option<f64> {
    let ms = (end - start) * 1e3;
    (start.is_finite() && start > 0.0 && end.is_finite() && ms.is_finite() && ms > 0.0)
        .then_some(ms)
}

pub(super) fn record_completed_command(timestamps: impl FnOnce() -> (f64, f64)) {
    OBSERVER.with(|s| {
        let mut s = s.borrow_mut();
        if !s.active {
            return;
        }
        let (start, end) = timestamps();
        s.observation.completed += 1;
        if let Some(ms) = duration_ms(start, end) {
            s.observation.valid += 1;
            s.observation.gpu_ms += ms;
        }
    });
}

pub(super) fn after_grouped(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    block: usize,
    rows: usize,
    dtype: GgmlType,
    p: &PackedScratch,
) -> Result<()> {
    let capture = OBSERVER.with(|s| {
        let mut s = s.borrow_mut();
        if !s.active {
            return None;
        }
        match dtype {
            GgmlType::IQ3_S => s.observation.iq3_layers += 1,
            GgmlType::IQ4_XS => s.observation.iq4_layers += 1,
            _ => panic!("unexpected GLM down dtype {dtype:?}"),
        }
        let capture = s.capture.as_ref().filter(|c| c.block == block).cloned();
        if capture.is_some() {
            s.observation.captures += 1;
        }
        capture
    });
    if let Some(c) = capture {
        assert_eq!((rows, dtype), (c.rows, GgmlType::IQ3_S));
        crate::metal::encode_copy_offset_f32(
            ctx,
            enc,
            &flat(&p.inner, K * TOP * rows),
            0,
            &c.inner,
            K * TOP * rows,
        )?;
        crate::metal::encode_copy_offset_i32(ctx, enc, &p.counts, 0, &c.counts, E)?;
        crate::metal::encode_copy_offset_i32(
            ctx,
            enc,
            &flat(&p.slots, E * rows),
            0,
            &c.slots,
            E * rows,
        )?;
    }
    Ok(())
}

struct Capture {
    block: usize,
    rows: usize,
    inner: MetalTensor,
    counts: MetalTensor,
    slots: MetalTensor,
}

fn cpu_reserve(rows: usize) -> u64 {
    // One slot-output reference, one weighted-output reference, one read chunk,
    // plus the existing packet's 16 MiB for tokens, metadata and endpoint logits.
    (16 * 1024 * 1024 + H * TOP * rows * 4 + H * rows * 4 + READ_FLOATS * 4) as u64
}

impl Capture {
    fn allocate(ctx: &MetalContext, block: usize, rows: usize, out: &mut File) -> Rc<Self> {
        let lengths = [K * TOP * rows * 4, E * 4, E * rows * 4];
        let priced: Vec<u64> = lengths
            .iter()
            .map(|&n| {
                ctx.price_shared_buffer_upper(n as u64)
                    .unwrap()
                    .priced_upper_bytes
            })
            .collect();
        let _allocation = ctx.begin_allocation_transaction();
        let admission = evaluate_metal_memory_admission_with_cpu_bytes(
            priced.iter().sum(),
            cpu_reserve(rows),
            0,
            ctx.memory_signals(),
            true,
        );
        emit(
            out,
            json!({"event":"capture_admission", "block":block, "rows":rows,
            "logical_bytes":lengths, "priced_gpu_bytes":priced, "cpu_reserve_bytes":cpu_reserve(rows),
            "admitted":admission.admitted, "reason":admission.reason.as_str(),
            "current_gpu_bytes":admission.signals.current_allocated_bytes,
            "working_set_headroom_bytes":admission.working_set_headroom_bytes}),
        );
        assert!(admission.admitted, "capture memory admission refused");
        Rc::new(Self {
            block,
            rows,
            inner: MetalTensor::zeros_f32(ctx, vec![(K * TOP * rows) as u64]).unwrap(),
            counts: MetalTensor::zeros_i32(ctx, vec![E as u64]).unwrap(),
            slots: MetalTensor::zeros_i32(ctx, vec![(E * rows) as u64]).unwrap(),
        })
    }
}

fn emit(out: &mut File, event: Value) {
    serde_json::to_writer(&mut *out, &event).unwrap();
    writeln!(out).unwrap();
    out.flush().unwrap();
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn tensor_hash(t: &MetalTensor) -> String {
    let mut digest = Sha256::new();
    for start in (0..t.n_elements() as usize).step_by(READ_FLOATS) {
        let n = READ_FLOATS.min(t.n_elements() as usize - start);
        let view = t.view_subrange(start as u64, vec![n as u64]);
        match t.dtype {
            GgmlType::F32 => digest.update(bytemuck::cast_slice(&read_f32(&view).unwrap())),
            GgmlType::I32 => digest.update(bytemuck::cast_slice(&read_i32(&view).unwrap())),
            _ => panic!("diagnostic hash requires 32-bit elements"),
        }
    }
    format!("{digest:x}", digest = digest.finalize())
}

/// No outstanding command may use this owned, shared scratch while it is written.
fn write_i32(t: &MetalTensor, values: &[i32]) {
    assert_eq!(t.dtype, GgmlType::I32);
    assert!(t.is_writable());
    assert_eq!(t.n_elements() as usize, values.len());
    assert!(t.offset.is_multiple_of(4));
    assert!(t.offset + values.len() as u64 * 4 <= t.buffer.length() as u64);
    // SAFETY: synchronized owned shared scratch, checked dtype, alignment and bounds.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<i32>(),
            values.len(),
        );
    }
}

fn fill_f32(t: &MetalTensor, value: f32) {
    assert_eq!(t.dtype, GgmlType::F32);
    assert!(t.is_writable() && t.offset.is_multiple_of(4));
    assert!(t.offset + t.n_elements() * 4 <= t.buffer.length() as u64);
    // SAFETY: caller has waited; this is bounded, owned shared scratch.
    unsafe {
        let dst = t
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(t.offset as usize)
            .cast::<f32>();
        for i in 0..t.n_elements() as usize {
            dst.add(i).write(value);
        }
    }
}

/// Stream candidate readback; never allocate a second full output vector.
fn metrics(t: &MetalTensor, reference: Option<&[f32]>, constant: Option<f32>) -> (Value, bool) {
    let count = t.n_elements() as usize;
    if let Some(r) = reference {
        assert_eq!(r.len(), count);
    }
    let mut hash = Sha256::new();
    let (mut nonfinite, mut unexpected) = (0usize, 0usize);
    let (mut max_abs, mut error2, mut norm2) = (0.0f64, 0.0f64, 0.0f64);
    for start in (0..count).step_by(READ_FLOATS) {
        let n = READ_FLOATS.min(count - start);
        let values = read_f32(&t.view_subrange(start as u64, vec![n as u64])).unwrap();
        hash.update(bytemuck::cast_slice(&values));
        for (i, &b) in values.iter().enumerate() {
            nonfinite += usize::from(!b.is_finite());
            unexpected += usize::from(constant.is_some_and(|a| a != b));
            if let Some(r) = reference {
                let a = r[start + i];
                if a.is_finite() && b.is_finite() {
                    let d = f64::from(a) - f64::from(b);
                    max_abs = max_abs.max(d.abs());
                    error2 += d * d;
                    norm2 += f64::from(a).powi(2);
                } else {
                    nonfinite += usize::from(!a.is_finite());
                }
            }
        }
    }
    let finite = nonfinite == 0;
    let comparison = (reference.is_some() && finite).then(|| {
        json!({
        "max_abs":max_abs, "relative_l2":(error2 / norm2.max(1e-30)).sqrt()})
    });
    (
        json!({"elements":count, "sha256_f32_le":format!("{:x}",hash.finalize()),
        "nonfinite_values_in_comparison":nonfinite, "comparison_to_A1":comparison,
        "sentinel_mismatches":constant.map(|_| unexpected)}),
        finite && unexpected == 0,
    )
}

fn save_reference(t: &MetalTensor, reference: &mut [f32]) {
    assert_eq!(t.n_elements() as usize, reference.len());
    for (index, chunk) in reference.chunks_mut(READ_FLOATS).enumerate() {
        let values =
            read_f32(&t.view_subrange((index * READ_FLOATS) as u64, vec![chunk.len() as u64]))
                .unwrap();
        chunk.copy_from_slice(&values);
    }
}

#[derive(Clone, Copy)]
enum Candidate {
    Blanket,
    SmallCounts,
}
impl Candidate {
    fn name(self) -> &'static str {
        match self {
            Self::Blanket => "blanket",
            Self::SmallCounts => "small_counts",
        }
    }
    fn variant(self) -> DownRetile {
        match self {
            Self::Blanket => DownRetile::Blanket,
            Self::SmallCounts => DownRetile::SmallCounts,
        }
    }
}

#[derive(Clone, Copy)]
enum Control {
    Natural,
    Concentrated,
    Empty,
}
impl Control {
    fn name(self) -> &'static str {
        match self {
            Self::Natural => "natural",
            Self::Concentrated => "concentrated_e0_to_e7",
            Self::Empty => "empty_launch_floor",
        }
    }
    fn empty(self) -> bool {
        matches!(self, Self::Empty)
    }
}

fn coverage(
    counts: &MetalTensor,
    slots: &MetalTensor,
    rows: usize,
    empty: bool,
    route_ids: Option<&[i32]>,
) -> Value {
    let counts = read_i32(counts).unwrap();
    let ids = read_i32(slots).unwrap();
    assert_eq!((counts.len(), ids.len()), (E, E * rows));
    let mut seen = vec![false; TOP * rows];
    let mut prefixes = Vec::with_capacity(TOP * rows);
    let mut s16 = 0usize;
    let mut s32 = 0usize;
    for (expert, &count) in counts.iter().enumerate() {
        let count = usize::try_from(count).expect("negative route count");
        assert!(count <= rows);
        s16 += count.div_ceil(16);
        s32 += count.div_ceil(32);
        let live = &ids[expert * rows..expert * rows + count];
        assert!(
            live.windows(2).all(|pair| pair[0] < pair[1]),
            "bucket slot order"
        );
        for &slot in live {
            let slot = usize::try_from(slot).expect("negative live slot");
            assert!(
                slot < seen.len() && !seen[slot],
                "invalid or duplicate live slot"
            );
            if let Some(routes) = route_ids {
                assert_eq!(routes[slot], expert as i32);
            }
            seen[slot] = true;
        }
        prefixes.extend_from_slice(live);
    }
    let live = seen.iter().filter(|&&v| v).count();
    assert_eq!(live, if empty { 0 } else { TOP * rows });
    json!({"counts_by_expert":counts, "active_slots":live,
        "coverage_contract":if empty { "zero active slots; no model/weighted-output claim" } else { "every routed slot exactly once" },
        "live_prefix_sha256_i32_le":sha(bytemuck::cast_slice(&prefixes)),
        "bucket_table_sha256_i32_le":sha(bytemuck::cast_slice(&ids)),
        "S16":s16, "S32":s32, "R":(s32 > 0).then(|| s16 as f64 / (2 * s32) as f64)})
}

fn command(
    ctx: &MetalContext,
    encode: impl FnOnce(&KernelEncoder) -> std::result::Result<(), MetalError>,
) -> (std::result::Result<(), MetalError>, f64, Option<f64>) {
    let started = Instant::now();
    let command = ctx
        .queue
        .commandBuffer()
        .expect("diagnostic command buffer");
    let enc = KernelEncoder::begin(&command);
    let result = encode(&enc);
    enc.end();
    let result = result.and_then(|()| {
        command.commit();
        wait_completed(&command)
    });
    let wall = started.elapsed().as_secs_f64() * 1e3;
    let gpu = result
        .is_ok()
        .then(|| duration_ms(command.GPUStartTime(), command.GPUEndTime()))
        .flatten();
    (result, wall, gpu)
}

const ARMS: [(&str, bool); 6] = [
    ("warm_A", false),
    ("warm_B", true),
    ("A1", false),
    ("B1", true),
    ("B2", true),
    ("A2", false),
];

fn leaf_screens(
    ctx: &MetalContext,
    session: &Glm5NextSession<'_>,
    capture: &Capture,
    out: &mut File,
) -> bool {
    let mut timings_valid = true;
    let p = session.packed.as_ref().unwrap();
    let (rows, block) = (capture.rows, capture.block);
    let FfnTensors::Moe(moe) = &session.weights.blocks[block].ffn else {
        panic!("MoE witness")
    };
    assert_eq!(moe.down_experts.dtype, GgmlType::IQ3_S);
    let slot_out = flat(&p.slot_out, H * TOP * rows);
    let weighted = flat(&p.routed, H * rows);
    let weights = flat(&p.routes[block].as_ref().unwrap().weights, TOP * rows);
    let routes = read_i32(&flat(&p.routes[block].as_ref().unwrap().ids, TOP * rows)).unwrap();
    let input_hash = tensor_hash(&capture.inner);
    let count_hash = tensor_hash(&capture.counts);
    let slot_hash = tensor_hash(&capture.slots);
    let weights_hash = tensor_hash(&weights);
    // One reference working set for this layer, materialized before warmup.
    // Nonzero initialization touches its pages; A1 later updates it in chunks.
    let mut slot_reference = vec![0.125f32; H * TOP * rows];
    let mut weighted_reference = vec![0.125f32; H * rows];
    for control in [Control::Natural, Control::Concentrated, Control::Empty] {
        let scratch_slots = flat(&p.slots, E * rows);
        let (counts, slots) = if matches!(control, Control::Natural) {
            (&capture.counts, &capture.slots)
        } else {
            let mut counts = vec![0i32; E];
            let mut ids = vec![-1i32; E * rows];
            if !control.empty() {
                for expert in 0..TOP {
                    counts[expert] = rows as i32;
                    for token in 0..rows {
                        ids[expert * rows + token] = (token * TOP + expert) as i32;
                    }
                }
            }
            write_i32(&p.counts, &counts);
            write_i32(&scratch_slots, &ids);
            (&p.counts, &scratch_slots)
        };
        let counts_hash = tensor_hash(counts);
        let slots_hash = tensor_hash(slots);
        emit(
            out,
            json!({"event":"leaf_binding", "block":block,"rows":rows,"control":control.name(),
            "inner_sha256_f32_le":input_hash, "route_weights_sha256_f32_le":weights_hash,
            "counts_sha256_i32_le":counts_hash, "slots_sha256_i32_le":slots_hash,
            "coverage":coverage(counts, slots, rows, control.empty(), matches!(control,Control::Natural).then_some(routes.as_slice()))}),
        );
        for candidate in [Candidate::Blanket, Candidate::SmallCounts] {
            eprintln!(
                "GLM down N{rows} block{block} {} {}",
                control.name(),
                candidate.name()
            );
            let mut reference_ready = false;
            for (label, b) in ARMS {
                // NaNs expose holes in complete-route writes. Empty controls must
                // preserve a finite sentinel and never feed weighted reduction.
                fill_f32(&slot_out, if control.empty() { 0.125 } else { f32::NAN });
                let (result, wall, gpu) = command(ctx, |enc| {
                    encode_variant(
                        ctx,
                        enc,
                        if b {
                            candidate.variant()
                        } else {
                            DownRetile::Incumbent
                        },
                        &moe.down_experts,
                        &capture.inner,
                        counts,
                        slots,
                        &slot_out,
                        K,
                        H,
                        E,
                        rows,
                    )
                });
                timings_valid &= gpu.is_some();
                emit(
                    out,
                    json!({"event":"leaf_attempt", "block":block,"rows":rows,
                    "control":control.name(),"candidate_policy":candidate.name(),"label":label,"candidate":b,
                    "wall_ms":wall,"command_gpu_ms":gpu,"command_gpu_valid":gpu.is_some(),
                    "down_dispatches_expected":if b && matches!(candidate,Candidate::SmallCounts) {2} else {1},
                    "error":result.as_ref().err().map(ToString::to_string)}),
                );
                result.expect("leaf failed; raw attempt flushed");
                if !control.empty() {
                    let (result, wall, gpu) = command(ctx, |enc| {
                        crate::metal::encode_moe_weighted_sum_packed_f32(
                            ctx, enc, &slot_out, &weights, &weighted, H, TOP, rows,
                        )
                    });
                    timings_valid &= gpu.is_some();
                    emit(
                        out,
                        json!({"event":"weighted_attempt","block":block,"rows":rows,
                        "control":control.name(),"candidate_policy":candidate.name(),"label":label,
                        "wall_ms":wall,"command_gpu_ms":gpu,"command_gpu_valid":gpu.is_some(),
                        "error":result.as_ref().err().map(ToString::to_string)}),
                    );
                    result.expect("weighted replay failed; raw attempt flushed");
                }
                let (slot_metrics, slot_ok) = metrics(
                    &slot_out,
                    reference_ready.then_some(slot_reference.as_slice()),
                    control.empty().then_some(0.125),
                );
                let (weighted_metrics, weighted_ok) = if control.empty() {
                    (Value::Null, true)
                } else {
                    metrics(
                        &weighted,
                        reference_ready.then_some(weighted_reference.as_slice()),
                        None,
                    )
                };
                emit(
                    out,
                    json!({"event":"leaf_output", "block":block,"rows":rows,
                    "control":control.name(),"candidate_policy":candidate.name(),"label":label,
                    "reference":reference_ready.then_some("A1"),
                    "slots":slot_metrics,"weighted":weighted_metrics,
                    "structural_and_finite_success":slot_ok && weighted_ok}),
                );
                assert!(
                    slot_ok && weighted_ok,
                    "nonfinite, incomplete write, or empty-control mutation"
                );
                if label == "A1" && !control.empty() {
                    save_reference(&slot_out, &mut slot_reference);
                    save_reference(&weighted, &mut weighted_reference);
                    reference_ready = true;
                }
            }
        }
        assert_eq!(counts_hash, tensor_hash(counts), "counts mutated");
        assert_eq!(slots_hash, tensor_hash(slots), "slots mutated");
    }
    assert_eq!(
        input_hash,
        tensor_hash(&capture.inner),
        "inner input mutated"
    );
    assert_eq!(count_hash, tensor_hash(&capture.counts));
    assert_eq!(slot_hash, tensor_hash(&capture.slots));
    assert_eq!(weights_hash, tensor_hash(&weights));
    timings_valid
}

fn prefill_attempt(
    ctx: &MetalContext,
    session: &mut Glm5NextSession<'_>,
    tokens: &[u32],
    capture: Option<Rc<Capture>>,
    candidate: Option<Candidate>,
    label: &str,
    phase: &str,
    out: &mut File,
) -> Vec<f32> {
    assert_eq!(super::router_prefill::requested_override(), None);
    let capture_on = capture.is_some();
    let capture_block = capture.as_ref().map(|c| c.block);
    let (((result, wall), obs), substitutions) = with_variant(
        candidate.map_or(DownRetile::Incumbent, Candidate::variant),
        || {
            observe(capture, || {
                let start = Instant::now();
                let result = session.prefill_packed(ctx, tokens);
                (result, start.elapsed().as_secs_f64() * 1e3)
            })
        },
    );
    let gpu = (result.is_ok() && obs.completed == 1 && obs.valid == 1 && obs.gpu_ms.is_finite())
        .then_some(obs.gpu_ms);
    emit(
        out,
        json!({"event":"prefill_attempt","phase":phase,"rows":tokens.len(),"label":label,
        "candidate":candidate.map(Candidate::name),"capture_on":capture_on,"capture_block":capture_block,
        "start_position":0,"chunk_rows":tokens.len(),"wall_ms":wall,
        "command_gpu_ms":gpu,"command_gpu_valid":gpu.is_some(),"command_gpu_completed":obs.completed,
        "command_gpu_valid_samples":obs.valid,"command_gpu_expected":1,
        "iq3_layers_observed":obs.iq3_layers,"iq4_layers_observed":obs.iq4_layers,
        "captures":obs.captures,"substitutions":substitutions,
        "error":result.as_ref().err().map(ToString::to_string)}),
    );
    let logits = result.expect("prefill failed; raw attempt flushed");
    let nonfinite = logits.iter().filter(|v| !v.is_finite()).count();
    emit(
        out,
        json!({"event":"prefill_output","phase":phase,"rows":tokens.len(),"label":label,
        "capture_block":capture_block,"sha256_f32_le":sha(bytemuck::cast_slice(&logits)),
        "nonfinite_logits":nonfinite,"top1":(nonfinite==0).then(|| argmax(&logits))}),
    );
    assert_eq!(
        (obs.iq3_layers, obs.iq4_layers, obs.captures),
        (39, 3, usize::from(capture_on))
    );
    assert_eq!(
        substitutions,
        if candidate.is_some() { 39 } else { 0 },
        "logical down substitution topology"
    );
    assert_eq!(session.position(), tokens.len());
    assert_eq!(nonfinite, 0, "nonfinite logits; raw output summary flushed");
    logits
}

fn whole_screens(
    ctx: &MetalContext,
    weights: &Glm5NextWeights,
    tokens: &[u32],
    candidates: &[Candidate],
    out: &mut File,
) {
    for rows in [128usize, 512] {
        for &candidate in candidates {
            let mut reference: Option<Vec<f32>> = None;
            for (label, b) in ARMS {
                let start = Instant::now();
                let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
                    ctx,
                    weights,
                    rows,
                    rows,
                    16 << 20,
                )
                .unwrap();
                session.set_packed_lineage(PackedLineage::Fast).unwrap();
                emit(
                    out,
                    json!({"event":"session_allocation","phase":"whole","rows":rows,
                    "candidate_policy":candidate.name(),"label":label,"wall_ms":start.elapsed().as_secs_f64()*1e3}),
                );
                let logits = prefill_attempt(
                    ctx,
                    &mut session,
                    &tokens[..rows],
                    None,
                    b.then_some(candidate),
                    label,
                    candidate.name(),
                    out,
                );
                let comparison = reference.as_ref().map(|a| {
                    let mut max_abs = 0.0f64;
                    let (mut error2,mut norm2) = (0.0f64,0.0f64);
                    assert_eq!(a.len(),logits.len());
                    for (&a,&b) in a.iter().zip(&logits) {
                        let d=f64::from(a)-f64::from(b); max_abs=max_abs.max(d.abs());
                        error2+=d*d; norm2+=f64::from(a).powi(2);
                    }
                    let (a_regret,b_regret)=choice_regret(a,&logits);
                    json!({"max_abs":max_abs,"relative_l2":(error2/norm2.max(1e-30)).sqrt(),
                        "kl_reference_candidate":kl_divergence(a,&logits),"kl_candidate_reference":kl_divergence(&logits,a),
                        "reference_top1":argmax(a),"candidate_top1":argmax(&logits),
                        "reference_choice_regret":a_regret,"candidate_choice_regret":b_regret})
                });
                emit(
                    out,
                    json!({"event":"whole_output","rows":rows,"candidate_policy":candidate.name(),"label":label,
                    "sha256_f32_le":sha(bytemuck::cast_slice(&logits)),"top1":argmax(&logits),"comparison_to_A1":comparison}),
                );
                if label == "A1" {
                    reference = Some(logits);
                }
            }
        }
    }
}

const DOMAIN_ARMS: [(&str, bool); 6] = [
    ("warm_A", false),
    ("warm_B", true),
    ("B1", true),
    ("A1", false),
    ("A2", false),
    ("B2", true),
];
const DOMAIN_WIDTHS: [usize; 7] = [32, 64, 127, 128, 129, 256, 512];
// Measurement points only; the production eligibility rule lives in the kernel module.
const PRODUCTION_WIDTHS: [usize; 3] = [32, 128, 512];
const DOMAIN_CPU_RESERVE: u64 = 32 << 20;
const DOMAIN_CODE: &str = include_str!("../../../../qwen-cli/src/serve/backend_glm5_next.rs");
const DOMAIN_RENDER: &str = include_str!("../../../../qwen-cli/src/serve/render_glm5_next.rs");

fn domain_override(production: bool, candidate: bool) -> Option<DownRetile> {
    if !candidate {
        Some(DownRetile::Incumbent)
    } else if production {
        None
    } else {
        Some(DownRetile::SmallCounts)
    }
}

fn with_domain_arm<R>(production: bool, candidate: bool, body: impl FnOnce() -> R) -> (R, usize) {
    match domain_override(production, candidate) {
        Some(variant) => with_variant(variant, body),
        None => with_production(body),
    }
}

fn enabled(name: &str) -> bool {
    match std::env::var(name).as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("") | Ok("0") => false,
        Ok("1") => true,
        _ => panic!("{name} must be absent, 0 or 1"),
    }
}

fn source_binding() -> Value {
    json!({"packed_rs":sha(include_bytes!("../packed.rs")),"packet_rs":sha(include_bytes!("expert_down.rs")),
        "session_rs":sha(include_bytes!("../../glm5_next_metal.rs")),"test_helpers_rs":sha(include_bytes!("../tests.rs")),
        "shared_expert_rs":sha(include_bytes!("../../metal/expert.rs")),"retile_rs":sha(include_bytes!("../../metal/iq3_s_down_retile.rs")),
        "retile_metal":sha(include_bytes!("../../../../../kernels/moe_iq3_s_down_retile.metal")),
        "generic_moe_rs":sha(include_bytes!("../../metal/moe_grouped_generic.rs")),
        "generic_moe_metal":sha(include_bytes!("../../../../../kernels/moe.metal")),
        "quant_tiles_h":sha(include_bytes!("../../../../../kernels/quant_tiles.h")),
        "metal_mod_rs":sha(include_bytes!("../../metal/mod.rs")),"metallib":sha(crate::KERNELS_METALLIB),
        "code_corpus_rs":sha(DOMAIN_CODE.as_bytes()),"code_renderer_rs":sha(DOMAIN_RENDER.as_bytes()),
        "note":"embedded compiled-source hashes, not runtime checkout identity"})
}

#[derive(Clone, Copy)]
struct DomainTiming {
    gpu_ms: Option<f64>,
    wall_ms: f64,
}

// Exactly the measured BAAB arms, excluding warmups. Missing/invalid timing
// cannot become a positive screen. These are observations, never promotion.
fn domain_evidence(times: &[DomainTiming]) -> (Value, bool) {
    if times.len() != 4
        || times.iter().any(|t| {
            !t.wall_ms.is_finite()
                || t.wall_ms <= 0.0
                || !t.gpu_ms.is_some_and(|g| g.is_finite() && g > 0.0)
        })
    {
        return (
            json!({"valid":false,"screen_supports_sampled_cell":false}),
            false,
        );
    }
    let [b1, a1, a2, b2] = [times[0], times[1], times[2], times[3]];
    let gpu_pairs = [
        1.0 - b1.gpu_ms.unwrap() / a1.gpu_ms.unwrap(),
        1.0 - b2.gpu_ms.unwrap() / a2.gpu_ms.unwrap(),
    ];
    let wall_pairs = [1.0 - b1.wall_ms / a1.wall_ms, 1.0 - b2.wall_ms / a2.wall_ms];
    let gpu_mean =
        1.0 - (b1.gpu_ms.unwrap() + b2.gpu_ms.unwrap()) / (a1.gpu_ms.unwrap() + a2.gpu_ms.unwrap());
    let wall_mean = 1.0 - (b1.wall_ms + b2.wall_ms) / (a1.wall_ms + a2.wall_ms);
    let supported = gpu_pairs.iter().all(|&s| s > 0.0)
        && wall_pairs.iter().all(|&s| s >= 0.0)
        && gpu_mean >= 0.02;
    (
        json!({"valid":true,"gpu_paired_saving_fraction":gpu_pairs,"wall_paired_saving_fraction":wall_pairs,
        "gpu_mean_saving_fraction":gpu_mean,"wall_mean_saving_fraction":wall_mean,
        "screen_supports_sampled_cell":supported}),
        supported,
    )
}

fn domain_comparison(reference: &[f32], candidate: &[f32]) -> Value {
    assert_eq!(reference.len(), candidate.len());
    let (mut max_abs, mut error2, mut norm2) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in reference.iter().zip(candidate) {
        assert!(a.is_finite() && b.is_finite());
        let d = f64::from(a) - f64::from(b);
        max_abs = max_abs.max(d.abs());
        error2 += d * d;
        norm2 += f64::from(a).powi(2);
    }
    let (a_top, b_top) = (argmax(reference), argmax(candidate));
    let kl_ab = kl_divergence(reference, candidate);
    let kl_ba = kl_divergence(candidate, reference);
    assert!(kl_ab.is_finite() && kl_ba.is_finite());
    json!({"max_abs":max_abs,"relative_l2":(error2/norm2.max(1e-30)).sqrt(),
        "kl_reference_candidate":kl_ab,"kl_candidate_reference":kl_ba,
        "reference_top1":a_top,"candidate_top1":b_top,
        "reference_choice_regret":f64::from(reference[a_top])-f64::from(reference[b_top]),
        "candidate_choice_regret":f64::from(candidate[b_top])-f64::from(candidate[a_top])})
}

fn domain_routes(session: &Glm5NextSession<'_>, rows: usize, chunk_rows: usize) -> Value {
    let p = session.packed.as_ref().unwrap();
    let live_rows = (rows - 1) % chunk_rows + 1;
    let layers: Vec<Value> = p
        .routes
        .iter()
        .enumerate()
        .filter_map(|(block, route)| {
            let route = route.as_ref()?;
            let ids = read_i32(&flat(&route.ids, live_rows * TOP)).unwrap();
            let mut counts = vec![0usize; E];
            for token in ids.chunks_exact(TOP) {
                for (i, &id) in token.iter().enumerate() {
                    let expert = usize::try_from(id).expect("negative route id");
                    assert!(expert < E && !token[..i].contains(&id));
                    counts[expert] += 1;
                }
            }
            let s16: usize = counts.iter().map(|&c| c.div_ceil(16)).sum();
            let s32: usize = counts.iter().map(|&c| c.div_ceil(32)).sum();
            Some(json!({"block":block,"counts_by_expert":counts,
            "active_experts":counts.iter().filter(|&&c| c>0).count(),
            "small_experts":counts.iter().filter(|&&c| (1..=16).contains(&c)).count(),
            "small_slots":counts.iter().filter(|&&c| c<=16).sum::<usize>(),
            "active_slots":ids.len(),"S16":s16,"S32":s32,"R":s16 as f64/(2*s32) as f64,
            "route_ids_sha256_i32_le":sha(bytemuck::cast_slice(&ids))}))
        })
        .collect();
    json!({"scope":"last packed chunk only; existing per-layer route IDs read after timing; no GPU copies",
        "start_position":rows-live_rows,"rows":live_rows,"layers":layers})
}

fn domain_outputs(
    out: &mut File,
    stream: &str,
    rows: usize,
    label: &str,
    outputs: &[Vec<f32>],
    reference: &[Vec<f32>],
) {
    assert_eq!(outputs.len(), reference.len());
    for (step, (candidate, a)) in outputs.iter().zip(reference).enumerate() {
        emit(
            out,
            json!({"event":"domain_output_comparison","stream":stream,"rows":rows,"label":label,
            "reference_label":"A1","teacher_forced_steps":step,"position":rows+step,
            "comparison":domain_comparison(a,candidate)}),
        );
    }
}

fn domain_cell(
    ctx: &MetalContext,
    weights: &Glm5NextWeights,
    tokens: &[u32],
    stream: &str,
    rows: usize,
    production: bool,
    out: &mut File,
) -> Option<bool> {
    let phase = if production { "production" } else { "domain" };
    let candidate_policy = if production {
        "production_selection"
    } else {
        "small_counts"
    };
    let chunk_rows = rows.min(512);
    let chunks = rows.div_ceil(chunk_rows);
    let continuation = if matches!(rows, 128 | 512 | 4096) {
        4
    } else {
        0
    };
    let capacity = rows + continuation;
    let mut times = Vec::with_capacity(4);
    let mut reference: Option<Vec<Vec<f32>>> = None;
    let mut pending_b1: Option<Vec<Vec<f32>>> = None;
    let mut all_timestamps_valid = true;
    for (label, b) in DOMAIN_ARMS {
        // Same device pricing/admission inputs as the ordinary session constructor.
        // Checking here permits an explicit optional-long skip before allocation.
        let ledger = Glm5NextMemoryLedger::new(
            &weights.config,
            weights.retained_bytes,
            capacity as u64,
            chunk_rows as u64,
            &device_price(ctx),
        )
        .unwrap();
        let admission = evaluate_metal_memory_admission_with_cpu_bytes(
            ledger.peak_bytes() - weights.retained_bytes,
            DOMAIN_CPU_RESERVE,
            0,
            ctx.memory_signals(),
            true,
        );
        emit(
            out,
            json!({"event":"domain_admission","phase":phase,"stream":stream,"rows":rows,"label":label,
            "capacity":capacity,"packed_rows":chunk_rows,"cpu_reserve_bytes":DOMAIN_CPU_RESERVE,
            "session_gpu_upper_bytes":ledger.peak_bytes()-weights.retained_bytes,
            "admitted":admission.admitted,"reason":admission.reason.as_str()}),
        );
        if !admission.admitted {
            emit(
                out,
                json!({"event":"domain_cell_skipped","phase":phase,"stream":stream,"rows":rows,"label":label,
                "reason":"memory admission refused","completed_measured_arms":times.len(),"screen_supports_sampled_cell":false}),
            );
            assert_eq!(
                rows, 4096,
                "required short domain cell was not admitted; raw refusal flushed"
            );
            return None;
        }
        let allocated = Instant::now();
        let result = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
            ctx,
            weights,
            capacity,
            chunk_rows,
            DOMAIN_CPU_RESERVE,
        );
        emit(
            out,
            json!({"event":"session_allocation","phase":phase,"stream":stream,"rows":rows,"label":label,
            "capacity":capacity,"packed_rows":chunk_rows,"wall_ms":allocated.elapsed().as_secs_f64()*1e3,
            "error":result.as_ref().err().map(ToString::to_string)}),
        );
        let mut session =
            result.expect("normal session admission/allocation failed; raw error flushed");
        session.set_packed_lineage(PackedLineage::Fast).unwrap();
        assert_eq!(super::router_prefill::requested_override(), None);
        eprintln!("GLM down {phase} {stream} N{rows} {label}");
        let (((result, wall_ms), obs), substitutions) = with_domain_arm(production, b, || {
            observe(None, || {
                let started = Instant::now();
                let result = session.prefill_packed(ctx, &tokens[..rows]);
                (result, started.elapsed().as_secs_f64() * 1e3)
            })
        });
        let topology_valid = obs.iq3_layers == 39 * chunks
            && obs.iq4_layers == 3 * chunks
            && obs.captures == 0
            && substitutions == if b { 39 * chunks } else { 0 };
        let gpu_ms = (result.is_ok()
            && obs.completed == chunks
            && obs.valid == chunks
            && obs.gpu_ms.is_finite()
            && obs.gpu_ms > 0.0)
            .then_some(obs.gpu_ms);
        all_timestamps_valid &= gpu_ms.is_some();
        emit(
            out,
            json!({"event":"prefill_attempt","phase":phase,"stream":stream,"rows":rows,"label":label,
            "candidate":b,"candidate_policy":candidate_policy,
            "arm_policy":if !b {"forced_incumbent"} else if production {"production_selection_no_override"} else {"forced_small_counts"},
            "capture_on":false,"start_position":0,
            "capacity":capacity,"chunk_rows":chunk_rows,"command_gpu_ms":gpu_ms,"wall_ms":wall_ms,
            "command_gpu_valid":gpu_ms.is_some(),"command_gpu_expected":chunks,"command_gpu_completed":obs.completed,
            "command_gpu_valid_samples":obs.valid,"iq3_layers_observed":obs.iq3_layers,"iq4_layers_observed":obs.iq4_layers,
            "captures":obs.captures,"substitutions":substitutions,"substitutions_expected":if b {39*chunks} else {0},
            "topology_valid":topology_valid,"error":result.as_ref().err().map(ToString::to_string)}),
        );
        let logits = result.expect("domain prefill failed; raw attempt flushed");
        assert!(
            topology_valid,
            "diagnostic eligibility/topology mismatch; raw attempt flushed"
        );
        assert_eq!(session.position(), rows);
        let mut outputs = vec![logits];
        for step in 0..=continuation {
            if step > 0 {
                // The prefill variant and timestamp observer have both ended.
                // Identical teacher-forced tokens exercise normal decode handoff.
                let token = tokens[rows + step - 1];
                let result = session.forward(ctx, token);
                emit(
                    out,
                    json!({"event":"domain_continuation","stream":stream,"rows":rows,"label":label,
                    "step":step,"token":token,"position":session.position(),"error":result.as_ref().err().map(ToString::to_string)}),
                );
                outputs.push(result.expect("teacher-forced handoff failed; error flushed"));
            }
            let logits = outputs.last().unwrap();
            let nonfinite = logits.iter().filter(|v| !v.is_finite()).count();
            emit(
                out,
                json!({"event":"domain_output","stream":stream,"rows":rows,"label":label,
                "teacher_forced_steps":step,"position":session.position(),"nonfinite_logits":nonfinite,
                "sha256_f32_le":sha(bytemuck::cast_slice(logits)),"top1":(nonfinite==0).then(||argmax(logits))}),
            );
            assert_eq!(nonfinite, 0, "nonfinite output; diagnostic flushed");
            assert_eq!(session.position(), rows + step);
            if step == 0 {
                emit(
                    out,
                    json!({"event":"domain_routes","stream":stream,"rows":rows,"label":label,
                    "routes":domain_routes(&session,rows,chunk_rows)}),
                );
            }
        }
        if !label.starts_with("warm_") {
            times.push(DomainTiming { gpu_ms, wall_ms });
        }
        if label == "B1" {
            pending_b1 = Some(outputs);
        } else if label == "A1" {
            domain_outputs(
                out,
                stream,
                rows,
                "B1",
                &pending_b1.take().unwrap(),
                &outputs,
            );
            reference = Some(outputs);
        } else if let Some(a) = reference.as_ref() {
            domain_outputs(out, stream, rows, label, &outputs, a);
        }
    }
    let (evidence, supported) = domain_evidence(&times);
    let supported = supported && all_timestamps_valid;
    emit(
        out,
        json!({"event":"domain_cell_summary","phase":phase,"candidate_policy":candidate_policy,"stream":stream,"rows":rows,"chunk_rows":chunk_rows,
        "status":if all_timestamps_valid {"measured"} else {"invalid_gpu_timestamps"},
        "all_attempt_timestamps_valid":all_timestamps_valid,"evidence":evidence,
        "screen_supports_sampled_cell":supported,"decision":"measurement_only_no_automatic_promotion"}),
    );
    Some(supported)
}

fn domain_packet(
    ctx: &MetalContext,
    gguf: &GgufFile,
    out: &mut File,
    long: bool,
    production: bool,
) {
    let widths: &[usize] = if production {
        &PRODUCTION_WIDTHS
    } else {
        &DOMAIN_WIDTHS
    };
    let schema = if production {
        "glm53.expert_down.production.v1"
    } else {
        "glm53.expert_down.domain.v1"
    };
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(gguf).unwrap();
    preflight_session(ctx, gguf, artifact.model(), 516, 512).unwrap();
    let texts = [
        ("qualification", long_qualification_text()),
        (
            "code_review",
            format!(
                "[gMASK]<sop>Review this Rust inference backend for a prefill or cache-state bug. Explain the likely failure and suggest a focused fix.\n\n{DOMAIN_CODE}\n\nRelated prompt-rendering implementation:\n\n{DOMAIN_RENDER}"
            ),
        ),
    ];
    let streams: Vec<_> = texts
        .iter()
        .map(|(name, text)| {
            let tokens: Vec<u32> = artifact
                .tokenizer()
                .encode(text, false)
                .unwrap()
                .into_iter()
                .take(4100)
                .map(|id| u32::try_from(id).unwrap())
                .collect();
            assert!(
                tokens.len() >= if long { 4100 } else { 516 },
                "corpus too short"
            );
            (*name, tokens)
        })
        .collect();
    let weights = Glm5NextWeights::load(ctx, gguf).unwrap();
    assert_eq!(
        (
            weights.config.hidden_size,
            weights.config.expert_ffn_size,
            weights.config.expert_count,
            weights.config.expert_used_count,
            weights.blocks.len()
        ),
        (4096, 2048, 288, 8, 45)
    );
    let mut iq3 = Vec::new();
    let mut iq4 = Vec::new();
    for (i, block) in weights.blocks.iter().enumerate() {
        if let FfnTensors::Moe(m) = &block.ffn {
            match m.down_experts.dtype {
                GgmlType::IQ3_S => iq3.push(i),
                GgmlType::IQ4_XS => iq4.push(i),
                _ => panic!("unqualified down dtype"),
            }
        }
    }
    assert_eq!(iq3.len(), 39);
    assert_eq!(iq4, vec![11, 12, 44]);
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let shards:Vec<Value>=stamps.iter().map(|s|json!({"path":s.path,"shard":s.shard_idx,"device":s.device,"inode":s.inode,
        "bytes":s.size,"mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,"ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec})).collect();
    let banks:Vec<Value>=gguf.tensors.iter().filter(|t|t.name.ends_with(".ffn_down_exps.weight")).map(|t|
        json!({"name":t.name,"dtype":format!("{:?}",t.dtype),"shape":t.shape,"shard":t.shard_idx,"offset":t.data_offset,"bytes":t.n_bytes})).collect();
    emit(
        out,
        json!({"event":"header","schema":schema,"decision":"measurement_only_no_automatic_promotion",
        "device":ctx.describe(),"fixture":crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.id,"shards":shards,"down_banks":banks,
        "artifact_binding":"retained shard identity/stamps and GGUF tensor descriptors; no full weight-content hash",
        "source_binding":source_binding(),"iq3_layers":iq3,"unchanged_iq4_layers":iq4,"shape":{"K":K,"M":H,"experts":E,"top_k":TOP},
        "widths":widths,"long_requested":long,"long_rows":4096,"long_packed_rows":512,
        "prefills_requested":(widths.len()+usize::from(long))*2*DOMAIN_ARMS.len(),
        "production_verification":production,
        "policy":if production {"A forces incumbent; B uses with_production with no down override; production routers unchanged"}
            else {"A forces incumbent; B forces SmallCounts diagnostic override even when production selection exists; production routers unchanged"},
        "width_scope":"measurement points only; this packet defines no production width limits",
        "order_per_cell":["warm_A","warm_B","B1","A1","A2","B2"],"pairs":[["B1","A1"],["B2","A2"]],
        "screen_criterion":"every attempt has valid GPU timestamps; both measured paired GPU savings >0; both paired wall savings >=0; mean GPU saving >=2%; finite outputs and expected topology",
        "domain_limit":"a width needs supporting cells from BOTH streams; sampled boundaries do not prove every interior; no automatic production promotion",
        "timing":"prefill only; allocation, route readback, output diagnostics and teacher-forced continuation excluded; multi-chunk GPU durations summed without encoder splitting; GPU includes stalls",
        "route_scope":"existing per-layer IDs after timing, last chunk only for multi-chunk prefill; no capture buffers or GPU observer copies",
        "continuation":{"rows":[128,512,4096],"steps":4,"tokens":"next four tokens of the same fixed stream in every arm","capacity":"prompt rows plus four; ordinary session admission"},
        "cpu_reserve_bytes":DOMAIN_CPU_RESERVE,"numerics":"finite f64 norms, bidirectional KL/regret and hashes; no bit-equality or difference threshold"}),
    );
    for ((name, text), (_, tokens)) in texts.iter().zip(&streams) {
        emit(
            out,
            json!({"event":"domain_stream","stream":name,"text_sha256":sha(text.as_bytes()),
            "token_ids":tokens,"token_ids_sha256_u32_le":sha(bytemuck::cast_slice(tokens)),
            "source":if *name=="qualification" {"long_qualification_text() as-is; existing prefix once"} else {"short review request plus compiled qwen-cli/src/serve/backend_glm5_next.rs and related render_glm5_next.rs; prefix once; renderer extends the backend's roughly 3300 tokens without padding"}}),
        );
    }
    let mut results = Vec::new();
    for (name, tokens) in &streams {
        for rows in widths.iter().copied().chain(long.then_some(4096)) {
            let supported = domain_cell(ctx, &weights, tokens, name, rows, production, out);
            results.push((*name, rows, supported));
        }
    }
    let supported_widths: Vec<usize> = widths
        .iter()
        .copied()
        .filter(|&n| {
            streams
                .iter()
                .all(|(name, _)| results.contains(&(*name, n, Some(true))))
        })
        .collect();
    assert_eq!(stamps, gguf.revalidate_retained_shard_stamps().unwrap());
    emit(
        out,
        json!({"event":"complete","schema":schema,"decision":"measurement_only_no_automatic_promotion",
        "production_verification":production,
        "sampled_widths_supported_on_both_streams":supported_widths,"cells":results,
        "warning":"sampled widths are evidence, not an exact-width production allowlist or proof of all interior widths; long cells test packed512 context"}),
    );
}

#[test]
fn expert_down_observer_is_scoped_and_independent() {
    record_completed_command(|| panic!("inactive observer queried timestamps"));
    assert_eq!(super::router_prefill::requested_override(), None);
    let ((), outer) = observe(None, || {
        record_completed_command(|| (1.0, 1.25));
        assert!(std::panic::catch_unwind(|| observe(None, || panic!("observer unwind"))).is_err());
        assert_eq!(super::router_prefill::requested_override(), None);
    });
    assert_eq!((outer.completed, outer.valid, outer.gpu_ms), (1, 1, 250.0));
    let ((), partial) = observe(None, || {
        for _ in 0..7 {
            record_completed_command(|| (1.0, 1.25));
        }
        record_completed_command(|| (0.0, 0.0));
    });
    assert_eq!(
        (partial.completed, partial.valid, partial.gpu_ms),
        (8, 7, 1750.0)
    );
    record_completed_command(|| panic!("observer leaked"));
}

#[test]
fn expert_down_domain_arm_selection_keeps_incumbent_explicit() {
    assert_eq!(domain_override(false, false), Some(DownRetile::Incumbent));
    assert_eq!(domain_override(true, false), Some(DownRetile::Incumbent));
    assert_eq!(domain_override(false, true), Some(DownRetile::SmallCounts));
    assert_eq!(domain_override(true, true), None);
    // Neither scope performs a substitution merely by entering it; the actual
    // shared-expert encode reports substitutions during the owner-run packet.
    for production in [false, true] {
        for candidate in [false, true] {
            let (value, substitutions) = with_domain_arm(production, candidate, || 7);
            assert_eq!((value, substitutions), (7, 0));
        }
    }
}

#[test]
fn expert_down_domain_screen_requires_valid_paired_evidence() {
    let a = DomainTiming {
        gpu_ms: Some(100.0),
        wall_ms: 110.0,
    };
    let b = DomainTiming {
        gpu_ms: Some(90.0),
        wall_ms: 100.0,
    };
    let mut times = [b, a, a, b];
    assert!(domain_evidence(&times).1);
    times[0].gpu_ms = None;
    assert!(!domain_evidence(&times).1);
    times[0].gpu_ms = Some(f64::NAN);
    assert!(!domain_evidence(&times).1);
    times[0] = b;
    times[3].gpu_ms = Some(101.0); // A good average cannot hide a losing pair.
    assert!(!domain_evidence(&times).1);
    times[3] = b;
    times[3].wall_ms = 111.0;
    assert!(!domain_evidence(&times).1);
    times[3].wall_ms = f64::NAN;
    assert!(!domain_evidence(&times).1);
    times = [
        DomainTiming {
            gpu_ms: Some(99.0),
            ..b
        },
        a,
        a,
        DomainTiming {
            gpu_ms: Some(99.0),
            ..b
        },
    ];
    assert!(!domain_evidence(&times).1);
    assert!(!domain_evidence(&[]).1);
}

#[test]
#[ignore = "diagnostic: released GLM fixture, production lease, new GLM53_EXPERT_DOWN_OUT; DOMAIN=1 bypasses capture/leaf; optional PRODUCTION=1 and DOMAIN_LONG=1"]
fn expert_down_packet() {
    assert!(!cfg!(debug_assertions), "packet requires --release");
    let domain = enabled("GLM53_EXPERT_DOWN_DOMAIN");
    let domain_long = enabled("GLM53_EXPERT_DOWN_DOMAIN_LONG");
    let production = enabled("GLM53_EXPERT_DOWN_PRODUCTION");
    assert!(!domain_long || domain, "DOMAIN_LONG requires DOMAIN=1");
    assert!(!production || domain, "PRODUCTION requires DOMAIN=1");
    let full = match std::env::var("GLM53_EXPERT_DOWN_FULL").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("") => vec![],
        Ok("blanket") => vec![Candidate::Blanket],
        Ok("small_counts") => vec![Candidate::SmallCounts],
        Ok("both") => vec![Candidate::Blanket, Candidate::SmallCounts],
        _ => panic!("GLM53_EXPERT_DOWN_FULL must be absent, blanket, small_counts or both"),
    };
    assert!(
        !domain || full.is_empty(),
        "DOMAIN and FULL are separate packets"
    );
    let _lease = perf_lease();
    let output = std::env::var_os("GLM53_EXPERT_DOWN_OUT").expect("GLM53_EXPERT_DOWN_OUT");
    let mut out = File::options()
        .write(true)
        .create_new(true)
        .open(output)
        .expect("new JSONL output required");
    let ctx = MetalContext::new().unwrap();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let gguf = GgufFile::open(&path).unwrap();
    if domain {
        domain_packet(&ctx, &gguf, &mut out, domain_long, production);
        return;
    }
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    preflight_session(&ctx, &gguf, artifact.model(), 512, 512).unwrap();
    let text = format!("[gMASK]<sop>{}", long_qualification_text());
    let tokens: Vec<u32> = artifact
        .tokenizer()
        .encode(&text, false)
        .unwrap()
        .into_iter()
        .take(512)
        .map(|id| u32::try_from(id).unwrap())
        .collect();
    assert_eq!(tokens.len(), 512);
    let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    assert_eq!(
        (
            weights.config.hidden_size,
            weights.config.expert_ffn_size,
            weights.config.expert_count,
            weights.config.expert_used_count,
            weights.blocks.len()
        ),
        (4096, 2048, 288, 8, 45)
    );
    assert_eq!(
        super::router_prefill::requested_override(),
        None,
        "router override must be inactive"
    );
    let mut iq3 = Vec::new();
    let mut iq4 = Vec::new();
    for (block, w) in weights.blocks.iter().enumerate() {
        if let FfnTensors::Moe(moe) = &w.ffn {
            match moe.down_experts.dtype {
                GgmlType::IQ3_S => iq3.push(block),
                GgmlType::IQ4_XS => iq4.push(block),
                _ => panic!("unqualified down dtype"),
            }
        }
    }
    assert_eq!(iq3.len(), 39);
    assert_eq!(iq4, vec![11, 12, 44]);
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let shards:Vec<Value>=stamps.iter().map(|s|json!({"path":s.path,"shard":s.shard_idx,"device":s.device,"inode":s.inode,"bytes":s.size,
        "mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,"ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec})).collect();
    let banks:Vec<Value>=[4usize,21].into_iter().map(|block| {
        let name=format!("blk.{block}.ffn_down_exps.weight");
        let t=gguf.tensors.iter().find(|t|t.name==name).expect("chosen down bank");
        assert_eq!(t.dtype,GgmlType::IQ3_S);
        json!({"block":block,"name":name,"dtype":"IQ3_S","shape":t.shape,"shard":t.shard_idx,"offset":t.data_offset,"bytes":t.n_bytes,"sha256":sha(gguf.slice(t))})
    }).collect();
    emit(
        &mut out,
        json!({"event":"header","schema":"glm53.expert_down.v1","decision":"diagnostic_only_no_promotion",
        "device":ctx.describe(),"fixture":crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.id,"shards":shards,"banks":banks,
        "iq3_layers":iq3,"unchanged_iq4_layers":iq4,"layers":[4,21],"widths":[128,512],"shape":{"K":K,"M":H,"experts":E,"top_k":TOP},
        "prompt":{"text_sha256":sha(text.as_bytes()),"token_ids":tokens,"corpus":"[gMASK]<sop> + long_qualification_text; first N tokens"},
        "router_policy":"production selection in every arm; no router override",
        "whole_requested":full.iter().map(|c|c.name()).collect::<Vec<_>>(),
        "whole_precondition":"all leaf execution, route coverage, input immutability, finite-output and GPU-timestamp validity checks passed; no speed or bit-identity gate",
        "leaf_timing":"one ordinary command per down encode_variant; SmallCounts includes both down dispatches; weighted reduction separately timed; CPU fill/readback excluded",
        "timing_limits":"repeated leaf bindings with CPU diagnostics between attempts are not all-layer attribution; GPU duration includes execution stalls; wall minus GPU is not isolated CPU/wiring cost",
        "capture_timing":"observer copies included; diagnostic setup only, never whole-model performance evidence",
        "whole_timing":"fresh admitted session; allocation excluded; normal weighted reduction; no capture buffers/copies; ordinary command GPU plus prefill wall",
        "empty_control":"zero active slots; output sentinel must be untouched; no weighted/model claim",
        "concentrated_control":"synthetic routing to experts 0..7 with all 8N slots covered; captured natural inner and route weights retained; leaf mechanism control, not model output",
        "order_per_pair":["warm_A","warm_B","A1","B1","B2","A2"],
        "cpu_reference_policy":"one slot and one weighted reference, nonzero-initialized before warmup; A1 saved in place; candidate readback uses 1 MiB chunks",
        "gpu_validity":"successful command and positive finite GPUStartTime and (GPUEndTime-GPUStartTime); whole requires exactly one valid command",
        "source_binding":{"packed_rs":sha(include_bytes!("../packed.rs")),"packet_rs":sha(include_bytes!("expert_down.rs")),
            "session_rs":sha(include_bytes!("../../glm5_next_metal.rs")),"test_helpers_rs":sha(include_bytes!("../tests.rs")),
            "shared_expert_rs":sha(include_bytes!("../../metal/expert.rs")),"retile_rs":sha(include_bytes!("../../metal/iq3_s_down_retile.rs")),
            "retile_metal":sha(include_bytes!("../../../../../kernels/moe_iq3_s_down_retile.metal")),
            "generic_moe_rs":sha(include_bytes!("../../metal/moe_grouped_generic.rs")),
            "generic_moe_metal":sha(include_bytes!("../../../../../kernels/moe.metal")),
            "quant_tiles_h":sha(include_bytes!("../../../../../kernels/quant_tiles.h")),
            "metal_mod_rs":sha(include_bytes!("../../metal/mod.rs")),"metallib":sha(crate::KERNELS_METALLIB),
            "note":"embedded compiled-source hashes, not runtime checkout identity"}}),
    );
    let mut leaf_timings_valid = true;
    for rows in [128usize, 512] {
        for block in [4usize, 21] {
            let start = Instant::now();
            let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
                &ctx,
                &weights,
                rows,
                rows,
                cpu_reserve(rows),
            )
            .unwrap();
            session.set_packed_lineage(PackedLineage::Fast).unwrap();
            emit(
                &mut out,
                json!({"event":"session_allocation","phase":"capture","rows":rows,"block":block,"wall_ms":start.elapsed().as_secs_f64()*1e3}),
            );
            let capture = Capture::allocate(&ctx, block, rows, &mut out);
            let logits = prefill_attempt(
                &ctx,
                &mut session,
                &tokens[..rows],
                Some(capture.clone()),
                None,
                "incumbent",
                "capture",
                &mut out,
            );
            emit(
                &mut out,
                json!({"event":"capture_output","rows":rows,"block":block,"logits_sha256_f32_le":sha(bytemuck::cast_slice(&logits))}),
            );
            leaf_timings_valid &= leaf_screens(&ctx, &session, &capture, &mut out);
            // Session, capture and all leaf CPU references end before the next
            // layer; none survive into optional whole-model measurements.
        }
    }
    emit(
        &mut out,
        json!({"event":"leaf_complete","decision":"unscored","timings_valid":leaf_timings_valid,"whole_requested":!full.is_empty()}),
    );
    if !full.is_empty() && leaf_timings_valid {
        whole_screens(&ctx, &weights, &tokens, &full, &mut out);
    } else if !full.is_empty() {
        emit(
            &mut out,
            json!({"event":"whole_skipped","reason":"invalid leaf GPU timestamps"}),
        );
    }
    assert_eq!(stamps, gguf.revalidate_retained_shard_stamps().unwrap());
    emit(
        &mut out,
        json!({"event":"complete","decision":"unscored_diagnostic"}),
    );
}
