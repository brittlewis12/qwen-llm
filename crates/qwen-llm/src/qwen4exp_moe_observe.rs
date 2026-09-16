use super::*;
use crate::metal::{DispatchCensusRow, dispatch_census_begin, dispatch_census_take};
use objc2_metal::MTLCommandQueue;
use std::cell::RefCell;

const STAGES: [&str; 6] = [
    "router_topk",
    "routed_gate_up",
    "routed_down_sum",
    "shared_gate_up",
    "shared_down",
    "accumulate",
];
const REPEATS: usize = 16;

#[path = "qwen4exp_moe_intervals.rs"]
pub(crate) mod intervals;

#[path = "qwen4exp_moe_topk_screen.rs"]
pub(crate) mod topk_screen;

#[path = "qwen4exp_moe_topk_native.rs"]
pub(crate) mod topk_native;

fn intervals_overlap(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0.max(b.0) < a.1.min(b.1)
}

#[test]
fn interval_overlap_distinguishes_reordered_disjoint_spans() {
    assert!(!intervals_overlap((20, 30), (5, 10)));
    assert!(intervals_overlap((20, 30), (15, 25)));
    assert!(!intervals_overlap((5, 10), (20, 30)));
    assert!(!intervals_overlap((5, 10), (10, 20)));
    assert!(intervals_overlap((5, 20), (10, 15)));
}

pub(crate) struct Capture {
    layer: u32,
    geometry: Qwen4ExpMoeMetalGeometry,
    bank: Vec<MetalTensor>,
    input: MetalTensor,
    output: MetalTensor,
    ids: MetalTensor,
    topk: MetalTensor,
    seen: bool,
}

impl Capture {
    pub(crate) fn new(ctx: &MetalContext, weights: &Qwen4ExpMetalWeights, layer: u32) -> Self {
        let w = Qwen4ExpMoeMetalWeights::bind(weights, layer).unwrap();
        let g = w.geometry;
        Self {
            layer,
            geometry: g,
            bank: [
                w.router,
                w.routed_gate,
                w.routed_up,
                w.routed_down,
                w.shared_router,
                w.shared_gate,
                w.shared_up,
                w.shared_down,
            ]
            .into_iter()
            .cloned()
            .collect(),
            input: MetalTensor::zeros_f32(ctx, vec![g.hidden_size as u64]).unwrap(),
            output: MetalTensor::zeros_f32(ctx, vec![g.hidden_size as u64]).unwrap(),
            ids: MetalTensor::zeros_i32(ctx, vec![g.experts_per_token as u64]).unwrap(),
            topk: MetalTensor::zeros_f32(ctx, vec![g.experts_per_token as u64]).unwrap(),
            seen: false,
        }
    }

    fn weights(&self) -> Qwen4ExpMoeMetalWeights<'_> {
        Qwen4ExpMoeMetalWeights {
            geometry: self.geometry,
            router: &self.bank[0],
            routed_gate: &self.bank[1],
            routed_up: &self.bank[2],
            routed_down: &self.bank[3],
            shared_router: &self.bank[4],
            shared_gate: &self.bank[5],
            shared_up: &self.bank[6],
            shared_down: &self.bank[7],
        }
    }
}

thread_local! {
    static CAPTURES: RefCell<Option<(usize, Vec<Capture>)>> = const { RefCell::new(None) };
}

pub(crate) fn with_capture<T>(captures: Vec<Capture>, f: impl FnOnce() -> T) -> (T, Vec<Capture>) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            CAPTURES.with(|c| {
                c.borrow_mut().take();
            });
        }
    }
    CAPTURES.with(|c| {
        assert!(c.borrow().is_none());
        *c.borrow_mut() = Some((0, captures));
    });
    let _reset = Reset;
    let result = f();
    let (calls, captures) = CAPTURES.with(|c| c.borrow_mut().take().unwrap());
    assert_eq!(calls, 48);
    assert!(captures.iter().all(|c| c.seen));
    (result, captures)
}

pub(super) fn before(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
) -> Result<Option<u32>, Qwen4ExpMoeError> {
    CAPTURES.with(|c| {
        let mut c = c.borrow_mut();
        let Some((calls, captures)) = c.as_mut() else {
            return Ok(None);
        };
        let ordinal = *calls;
        *calls += 1;
        for capture in captures {
            let router = &capture.bank[0];
            if Retained::as_ptr(&router.buffer) == Retained::as_ptr(&weights.router.buffer)
                && router.offset == weights.router.offset
            {
                assert_eq!(ordinal, capture.layer as usize);
                assert!(!capture.seen);
                assert_eq!(weights.geometry, capture.geometry);
                encode_copy_offset_f32(
                    ctx,
                    enc,
                    input,
                    0,
                    &capture.input,
                    weights.geometry.hidden_size,
                )?;
                capture.seen = true;
                return Ok(Some(capture.layer));
            }
        }
        Ok(None)
    })
}

pub(super) fn after(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    layer: Option<u32>,
    buffers: Qwen4ExpMoeSingletonBuffers<'_>,
) -> Result<(), Qwen4ExpMoeError> {
    let Some(layer) = layer else {
        return Ok(());
    };
    CAPTURES.with(|c| {
        let c = c.borrow();
        let capture = c
            .as_ref()
            .unwrap()
            .1
            .iter()
            .find(|c| c.layer == layer)
            .unwrap();
        encode_copy_offset_f32(
            ctx,
            enc,
            buffers.output,
            0,
            &capture.output,
            capture.geometry.hidden_size,
        )?;
        encode_copy_offset_i32(
            ctx,
            enc,
            buffers.topk_ids,
            0,
            &capture.ids,
            capture.geometry.experts_per_token,
        )?;
        encode_copy_offset_f32(
            ctx,
            enc,
            buffers.topk_weights,
            0,
            &capture.topk,
            capture.geometry.experts_per_token,
        )?;
        Ok(())
    })
}

fn bytes(t: &MetalTensor) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize),
            t.n_bytes() as usize,
        )
        .to_vec()
    }
}

fn assert_finite(t: &MetalTensor) {
    assert_eq!(t.dtype, GgmlType::F32);
    assert!(
        bytes(t)
            .chunks_exact(4)
            .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite())
    );
}

fn buffers(w: &Qwen4ExpMoeMetalWorkspace) -> Qwen4ExpMoeSingletonBuffers<'_> {
    Qwen4ExpMoeSingletonBuffers {
        router_logits: &w.router_logits,
        topk_ids: &w.topk_ids,
        topk_weights: &w.topk_weights,
        shared_scale: &w.shared_gate,
        routed_inner: &w.routed_inner,
        routed_expert_output: &w.routed_expert_output,
        shared_inner: &w.shared_inner,
        shared_output: &w.shared_output,
        output: &w.output,
    }
}

fn poison(w: &Qwen4ExpMoeMetalWorkspace) {
    unsafe {
        std::ptr::write_bytes(
            w.topk_ids
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(w.topk_ids.offset as usize),
            0xff,
            w.topk_ids.n_bytes() as usize,
        );
    }
    for t in [
        &w.router_logits,
        &w.topk_weights,
        &w.shared_gate,
        &w.routed_inner,
        &w.routed_expert_output,
        &w.shared_inner,
        &w.shared_output,
        &w.output,
    ] {
        unsafe {
            let ptr = t
                .buffer
                .contents()
                .as_ptr()
                .cast::<u32>()
                .add(t.offset as usize / 4);
            for i in 0..t.n_elements() as usize {
                ptr.add(i).write(f32::NAN.to_bits());
            }
        }
    }
}

fn stage(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    c: &Capture,
    w: &Qwen4ExpMoeMetalWorkspace,
    stage: usize,
) {
    let b = buffers(w);
    let result = match stage {
        0 => encode_singleton_router(ctx, enc, &c.input, c.weights(), b),
        1 => encode_singleton_gate_up(ctx, enc, &c.input, c.weights(), b),
        2 => encode_singleton_down(ctx, enc, c.weights(), b),
        3 => encode_singleton_shared_gate_up(ctx, enc, &c.input, c.weights(), b),
        4 => encode_singleton_shared_down(ctx, enc, c.weights(), b),
        5 => encode_singleton_accumulate(ctx, enc, b),
        _ => unreachable!(),
    };
    result.unwrap();
}

fn row_json(row: &DispatchCensusRow) -> serde_json::Value {
    serde_json::json!({"kernel":row.kernel,"grid":[row.grid_width,row.grid_height,row.grid_depth],"threads":[row.threads_width,row.threads_height,row.threads_depth]})
}

fn replay(
    ctx: &MetalContext,
    c: &Capture,
    w: &Qwen4ExpMoeMetalWorkspace,
    repeats: usize,
    sampled: bool,
    census: bool,
    artifact: &std::path::Path,
    packet: &str,
) -> (serde_json::Value, Vec<DispatchCensusRow>) {
    replay_packet(ctx, c, w, repeats, sampled, census, artifact, packet, false)
}

fn replay_packet(
    ctx: &MetalContext,
    c: &Capture,
    w: &Qwen4ExpMoeMetalWorkspace,
    repeats: usize,
    sampled: bool,
    census: bool,
    artifact: &std::path::Path,
    packet: &str,
    interval_aware: bool,
) -> (serde_json::Value, Vec<DispatchCensusRow>) {
    poison(w);
    let samples = sampled.then(|| ctx.timestamp_sample_buffer(repeats * 12).unwrap());
    if census {
        dispatch_census_begin();
    }
    let command = ctx.queue.commandBuffer().unwrap();
    if let Some(samples) = &samples {
        for repeat in 0..repeats {
            for index in 0..6 {
                let enc = stage_encoder(&command, samples, repeat * 6 + index).unwrap();
                stage(ctx, &enc, c, w, index);
                enc.end();
            }
        }
    } else {
        let enc = KernelEncoder::begin(&command);
        for _ in 0..repeats {
            encode_singleton_step(ctx, &enc, &c.input, c.weights(), buffers(w)).unwrap();
        }
        enc.end();
    }
    let rows = if census {
        dispatch_census_take()
    } else {
        Vec::new()
    };
    command.commit();
    command.waitUntilCompleted();
    let packet_path = artifact.join(format!("layer{}-{packet}-packet.json", c.layer));
    let mut evidence = serde_json::json!({"validation":"pending","layer":c.layer,"packet":packet,"repeats":repeats,"sampled":sampled,"stages":STAGES,"sample_mapping":"2*(repeat*6+stage) start, next index end","status":format!("{:?}",command.status()),"error":command.error().map(|e|e.to_string()),"gpu_start":command.GPUStartTime(),"gpu_end":command.GPUEndTime(),"census":rows.iter().map(row_json).collect::<Vec<_>>()});
    std::fs::write(&packet_path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none());
    let resolved = samples
        .as_ref()
        .map(|s| ctx.resolve_timestamp_samples(s, repeats * 12))
        .transpose();
    evidence["raw_timestamps"] = serde_json::json!(resolved.as_ref().ok().and_then(|v| v.as_ref()));
    evidence["resolution_error"] =
        serde_json::json!(resolved.as_ref().err().map(|e| e.to_string()));
    std::fs::write(&packet_path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
    let raw = resolved.unwrap().unwrap_or_default();
    let gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
    assert!(gpu_ms.is_finite() && gpu_ms > 0.0);
    assert_finite(&w.output);
    assert_finite(&w.topk_weights);
    assert_eq!(
        bytes(&w.output),
        bytes(&c.output),
        "native output layer{}",
        c.layer
    );
    assert_eq!(bytes(&w.topk_ids), bytes(&c.ids));
    assert_eq!(bytes(&w.topk_weights), bytes(&c.topk));
    let mut stages = serde_json::Map::new();
    let mut scale = None;
    let mut stage_sum = 0.0;
    if sampled && !interval_aware {
        assert_eq!(raw.len(), repeats * 12);
        assert!(
            raw.iter().all(|&v| v != 0 && v != u64::MAX),
            "invalid counter sample"
        );
        assert!(raw.windows(2).all(|v| v[1] >= v[0]));
        assert!(
            raw.chunks_exact(2).all(|v| v[1] > v[0]),
            "empty or invalid stage sample"
        );
        assert!(*raw.last().unwrap() > raw[0]);
        let tick_scale = gpu_ms / (*raw.last().unwrap() - raw[0]) as f64;
        scale = Some(tick_scale);
        for (index, name) in STAGES.iter().enumerate() {
            let ticks: u64 = (0..repeats)
                .map(|r| raw[r * 12 + 2 * index + 1] - raw[r * 12 + 2 * index])
                .sum();
            let stage_ms = ticks as f64 * tick_scale / repeats as f64;
            stage_sum += stage_ms;
            stages.insert(name.to_string(), serde_json::json!(stage_ms));
        }
    }
    let result = if interval_aware {
        serde_json::json!({"protocol":"interval-aware-v2","repeats":repeats,"sampled":sampled,"gpu_ms_per_moe":gpu_ms/repeats as f64,"intervals":sampled.then(|| intervals::summarize(&raw, repeats, gpu_ms)),"raw_timestamps":raw,"native_output_and_routes_bitwise":true})
    } else {
        serde_json::json!({"repeats":repeats,"sampled":sampled,"gpu_ms_per_moe":gpu_ms/repeats as f64,"scaled_stage_ms":stages,"normalization_ms_per_tick":scale,"stage_sum_ms":sampled.then_some(stage_sum),"unattributed_ms":sampled.then_some(gpu_ms/repeats as f64-stage_sum),"raw_timestamps":raw,"native_output_and_routes_bitwise":true})
    };
    evidence["validation"] =
        serde_json::json!("replay_checks_passed; caller_census_and_immutability_pending");
    evidence["result"] = result.clone();
    std::fs::write(&packet_path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
    (result, rows)
}

pub(crate) fn profile(
    ctx: &MetalContext,
    captures: &[Capture],
    native: &[DispatchCensusRow],
    artifact: &std::path::Path,
) {
    eprintln!(
        "moe_observe effective_iq3_fast={} effective_iq4_down_fast={}",
        qwen4exp_moe_iq3_fast_enabled(),
        qwen4exp_moe_iq4_down_fast_enabled()
    );
    for c in captures {
        let expected = match c.layer {
            2 => (GgmlType::IQ4_XS, GgmlType::Q8_0, 8),
            4 => (GgmlType::IQ3_XXS, GgmlType::Q8_0, 8),
            5 => (GgmlType::IQ3_XXS, GgmlType::IQ4_NL, 9),
            _ => unreachable!(),
        };
        assert_eq!((c.bank[1].dtype, c.bank[3].dtype), (expected.0, expected.1));
        let tag = format!("moe.native.{}", c.layer);
        let rows: Vec<_> = native
            .iter()
            .filter(|r| r.tag.as_deref() == Some(&tag))
            .map(row_json)
            .collect();
        assert_eq!(rows.len(), expected.2);
        let frozen = bytes(&c.input);
        for (name, t) in [
            ("input", &c.input),
            ("output", &c.output),
            ("ids", &c.ids),
            ("topk", &c.topk),
        ] {
            std::fs::write(
                artifact.join(format!("layer{}-{name}.bin", c.layer)),
                bytes(t),
            )
            .unwrap();
        }
        let w = Qwen4ExpMoeMetalWorkspace::new(ctx, c.geometry).unwrap();
        for t in [&c.input, &c.output, &c.topk] {
            assert_finite(t);
        }
        let ids: Vec<i32> = bytes(&c.ids)
            .chunks_exact(4)
            .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        assert!(
            ids.iter()
                .all(|&id| id >= 0 && (id as usize) < c.geometry.expert_count)
        );
        assert_eq!(
            ids.iter().collect::<std::collections::BTreeSet<_>>().len(),
            ids.len()
        );
        validate_contract(ctx, &c.input, c.weights(), &w).unwrap();
        preflight(ctx, c.weights()).unwrap();
        let (_, replay_rows) = replay(ctx, c, &w, 1, false, true, artifact, "census");
        assert_eq!(rows, replay_rows.iter().map(row_json).collect::<Vec<_>>());
        replay(ctx, c, &w, 3, false, false, artifact, "warm");
        let (before, _) = replay(ctx, c, &w, REPEATS, false, false, artifact, "before");
        let (sampled, sampled_rows) = replay(ctx, c, &w, REPEATS, true, true, artifact, "sampled");
        assert_eq!(sampled_rows.len(), rows.len() * REPEATS);
        for repeated in sampled_rows.chunks_exact(rows.len()) {
            assert_eq!(rows, repeated.iter().map(row_json).collect::<Vec<_>>());
        }
        let (after, _) = replay(ctx, c, &w, REPEATS, false, false, artifact, "after");
        assert_eq!(bytes(&c.input), frozen);
        let ids: Vec<i32> = bytes(&c.ids)
            .chunks_exact(4)
            .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let topk: Vec<f32> = bytes(&c.topk)
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let result = serde_json::json!({"layer":c.layer,"gate_dtype":format!("{:?}",expected.0),"down_dtype":format!("{:?}",expected.1),"weight_shapes":c.bank.iter().map(|t| &t.shape).collect::<Vec<_>>(),"selected_experts":ids,"selected_weights":topk,"census":rows,"before":before,"sampled":sampled,"after":after,"interpretation":"warm isolated replay of captured native input; normalized encoder-stage attribution, not native all-layer cost or speedup"});
        std::fs::write(
            artifact.join(format!("layer{}-profile.json", c.layer)),
            serde_json::to_vec_pretty(&result).unwrap(),
        )
        .unwrap();
        eprintln!("moe_observe {result}");
    }
}

#[test]
#[ignore = "serial production lease; model-free stage timestamp interval diagnostic, not MoE timing"]
fn stage_timestamp_interval_diagnostic() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    std::fs::create_dir_all(&parent).unwrap();
    let path = parent.join(format!(
        "stage-interval-diagnostic-{}.json",
        std::process::id()
    ));
    let ctx = MetalContext::new().unwrap();
    let a = MetalTensor::zeros_f32(&ctx, vec![65536]).unwrap();
    let b = MetalTensor::zeros_f32(&ctx, vec![65536]).unwrap();
    unsafe {
        let ptr = a.buffer.contents().as_ptr().cast::<f32>();
        for i in 0..65536 {
            ptr.add(i).write((i % 101) as f32);
        }
    }
    let original = bytes(&a);
    let samples = ctx.timestamp_sample_buffer(192).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    for stage in 0..96 {
        let enc = stage_encoder(&command, &samples, stage).unwrap();
        let (src, dst) = if stage % 2 == 0 { (&a, &b) } else { (&b, &a) };
        encode_copy_offset_f32(&ctx, &enc, src, 0, dst, 65536).unwrap();
        enc.end();
    }
    command.commit();
    command.waitUntilCompleted();
    let mut evidence = serde_json::json!({"validation":"pending","status":format!("{:?}",command.status()),"error":command.error().map(|e|e.to_string()),"gpu_start":command.GPUStartTime(),"gpu_end":command.GPUEndTime(),"sample_mapping":"pair i is encoder i start/end, one dependent65536-float copy each","purpose":"interval classification only; not MoE cost or speedup"});
    std::fs::write(&path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none());
    let resolved = ctx.resolve_timestamp_samples(&samples, 192);
    evidence["raw_timestamps"] = serde_json::json!(resolved.as_ref().ok());
    evidence["resolution_error"] =
        serde_json::json!(resolved.as_ref().err().map(|e| e.to_string()));
    std::fs::write(&path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
    let raw = resolved.unwrap();
    assert_eq!(bytes(&a), original);
    assert_eq!(bytes(&b), original);
    assert_eq!(raw.len(), 192);
    let invalid = raw.iter().filter(|&&v| v == 0 || v == u64::MAX).count();
    let bad_pairs = raw.chunks_exact(2).filter(|v| v[1] <= v[0]).count();
    let overlaps = (0..95)
        .filter(|&i| {
            intervals_overlap(
                (raw[2 * i], raw[2 * i + 1]),
                (raw[2 * i + 2], raw[2 * i + 3]),
            )
        })
        .count();
    let reversed_starts = (0..95).filter(|&i| raw[2 * i + 2] < raw[2 * i]).count();
    evidence["invalid_samples"] = serde_json::json!(invalid);
    evidence["nonpositive_pairs"] = serde_json::json!(bad_pairs);
    evidence["adjacent_overlap_count"] = serde_json::json!(overlaps);
    evidence["reversed_start_count"] = serde_json::json!(reversed_starts);
    evidence["copies_bitwise"] = serde_json::json!(true);
    evidence["validation"] = serde_json::json!(if invalid == 0 && bad_pairs == 0 {
        "individual_spans_valid"
    } else {
        "invalid_individual_spans"
    });
    std::fs::write(&path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
    eprintln!(
        "stage_interval artifacts={} invalid={invalid} nonpositive_pairs={bad_pairs} adjacent_overlaps={overlaps} reversed_starts={reversed_starts}",
        path.display()
    );
    assert_eq!(invalid, 0);
    assert_eq!(bad_pairs, 0);
}

fn load_saved_layer2(
    ctx: &MetalContext,
    weights: &Qwen4ExpMetalWeights,
    source: &std::path::Path,
) -> Capture {
    use sha2::{Digest, Sha256};
    let c = Capture::new(ctx, weights, 2);
    assert_eq!(
        (c.bank[1].dtype, c.bank[3].dtype),
        (GgmlType::IQ4_XS, GgmlType::Q8_0)
    );
    for (name, tensor, hash) in [
        (
            "input",
            &c.input,
            "35996b3ac153473fcc8978dadccc233308c9d69604956c42df54a8bf46a6b909",
        ),
        (
            "output",
            &c.output,
            "5d808d01d93999b6d816982307b0e6efec840c637798ad7836e59034f2086c95",
        ),
        (
            "ids",
            &c.ids,
            "e1ad77bbfdc0dcd1f90699f9ca17bd6e5de73ae2f67c2aa488c6536a0bb290ae",
        ),
        (
            "topk",
            &c.topk,
            "fad4832ff136c6768edb38516938ef9d2cf28fff89ac1be7a093ab1b7e46c223",
        ),
    ] {
        let data = std::fs::read(source.join(format!("layer2-{name}.bin"))).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&data)), hash);
        assert_eq!(data.len(), tensor.n_bytes() as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(tensor.offset as usize),
                data.len(),
            );
        }
    }
    for t in [&c.input, &c.output, &c.topk] {
        assert_finite(t);
    }
    c
}

pub(crate) fn captured_interval_diagnostic(
    ctx: &MetalContext,
    weights: &Qwen4ExpMetalWeights,
    source: &std::path::Path,
    artifact: &std::path::Path,
) {
    let c = load_saved_layer2(ctx, weights, source);
    let w = Qwen4ExpMoeMetalWorkspace::new(ctx, c.geometry).unwrap();
    validate_contract(ctx, &c.input, c.weights(), &w).unwrap();
    preflight(ctx, c.weights()).unwrap();
    poison(&w);
    let samples = ctx.timestamp_sample_buffer(192).unwrap();
    dispatch_census_begin();
    let command = ctx.queue.commandBuffer().unwrap();
    for repeat in 0..16 {
        for index in 0..6 {
            let enc = stage_encoder(&command, &samples, repeat * 6 + index).unwrap();
            stage(ctx, &enc, &c, &w, index);
            enc.end();
        }
    }
    let census = dispatch_census_take();
    command.commit();
    command.waitUntilCompleted();
    let mut evidence = serde_json::json!({"validation":"pending","source":source.display().to_string(),"status":format!("{:?}",command.status()),"error":command.error().map(|e|e.to_string()),"gpu_start":command.GPUStartTime(),"gpu_end":command.GPUEndTime(),"stages":STAGES,"sample_mapping":"2*(repeat*6+stage) start, next index end","census":census.iter().map(row_json).collect::<Vec<_>>(),"purpose":"saved layer2 interval classification only; not a MoE performance rerun"});
    std::fs::write(artifact, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none());
    let resolved = ctx.resolve_timestamp_samples(&samples, 192);
    evidence["raw_timestamps"] = serde_json::json!(resolved.as_ref().ok());
    evidence["resolution_error"] =
        serde_json::json!(resolved.as_ref().err().map(|e| e.to_string()));
    std::fs::write(artifact, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
    let raw = resolved.unwrap();
    assert_eq!(raw.len(), 192);
    assert_eq!(census.len(), 16 * 8);
    assert_finite(&w.output);
    assert_finite(&w.topk_weights);
    assert_eq!(
        bytes(&c.input),
        std::fs::read(source.join("layer2-input.bin")).unwrap()
    );
    assert_eq!(bytes(&w.output), bytes(&c.output));
    assert_eq!(bytes(&w.topk_ids), bytes(&c.ids));
    assert_eq!(bytes(&w.topk_weights), bytes(&c.topk));
    let invalid = raw.iter().filter(|&&v| v == 0 || v == u64::MAX).count();
    let bad_pairs = raw.chunks_exact(2).filter(|v| v[1] <= v[0]).count();
    let overlaps: Vec<_> = (0..95)
        .filter(|&i| {
            intervals_overlap(
                (raw[2 * i], raw[2 * i + 1]),
                (raw[2 * i + 2], raw[2 * i + 3]),
            )
        })
        .collect();
    let reordered_disjoint: Vec<_> = (0..95).filter(|&i| raw[2 * i + 3] <= raw[2 * i]).collect();
    let reversed_starts: Vec<_> = (0..95).filter(|&i| raw[2 * i + 2] < raw[2 * i]).collect();
    evidence["invalid_samples"] = serde_json::json!(invalid);
    evidence["nonpositive_pairs"] = serde_json::json!(bad_pairs);
    evidence["adjacent_overlap_indices"] = serde_json::json!(overlaps);
    evidence["reordered_disjoint_indices"] = serde_json::json!(reordered_disjoint);
    evidence["reversed_start_indices"] = serde_json::json!(reversed_starts);
    evidence["native_output_and_routes_bitwise"] = serde_json::json!(true);
    evidence["validation"] = serde_json::json!(if invalid == 0 && bad_pairs == 0 {
        "individual_spans_valid"
    } else {
        "invalid_individual_spans"
    });
    std::fs::write(artifact, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
    eprintln!(
        "captured_interval artifacts={} invalid={invalid} nonpositive_pairs={bad_pairs} overlaps={overlaps:?} reversed_starts={reversed_starts:?}",
        artifact.display()
    );
    assert_eq!(invalid, 0);
    assert_eq!(bad_pairs, 0);
}
