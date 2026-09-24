use super::*;

const N: usize = 512;
const K: usize = 10;
const GUARD: usize = 32;
const KERNEL: &str = "kernel_topk_logits_softmax_parallel_f32";

pub(super) fn select(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    logits: &MetalTensor,
    ids: &MetalTensor,
    weights: &MetalTensor,
    parallel: bool,
) {
    if !parallel {
        crate::metal::encode_topk_logits_softmax_f32(ctx, enc, logits, ids, weights, N, K).unwrap();
        return;
    }
    enc.set_pipeline(&ctx.pipeline(KERNEL).unwrap());
    enc.set_bytes_slice(0, &[N as u32, K as u32]);
    enc.set_tensor(1, logits);
    enc.set_tensor(2, ids);
    enc.set_tensor(3, weights);
    for slot in 0..3 {
        enc.set_threadgroup_memory(slot, N * 4);
    }
    enc.dispatch(
        objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: N,
            height: 1,
            depth: 1,
        },
    );
}

pub(super) fn guarded(ctx: &MetalContext, data: &[u8], dtype: GgmlType) -> MetalTensor {
    let mut storage = vec![0xa5; GUARD];
    storage.extend_from_slice(data);
    storage.extend_from_slice(&[0x5a; GUARD]);
    MetalTensor {
        buffer: ctx.buffer_from(&storage).unwrap(),
        offset: GUARD as u64,
        shape: vec![(data.len() / 4) as u64],
        dtype,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

pub(super) fn allocation(t: &MetalTensor) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast::<u8>(), t.buffer.length())
            .to_vec()
    }
}

pub(super) fn guards(t: &MetalTensor) {
    let b = allocation(t);
    assert_eq!(&b[..GUARD], &[0xa5; GUARD]);
    assert_eq!(&b[b.len() - GUARD..], &[0x5a; GUARD]);
}

pub(super) fn f32s(t: &MetalTensor) -> Vec<f32> {
    bytes(t)
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}

pub(super) fn oracle(values: &[f32], ids: &MetalTensor, weights: &MetalTensor) {
    assert!(values.iter().all(|v| v.is_finite()));
    let mut expected: Vec<_> = (0..N).collect();
    expected.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap().then(a.cmp(&b)));
    expected.truncate(K);
    let actual: Vec<usize> = bytes(ids)
        .chunks_exact(4)
        .map(|v| i32::from_le_bytes(v.try_into().unwrap()) as usize)
        .collect();
    assert_eq!(actual, expected);
    let exps: Vec<f64> = expected
        .iter()
        .map(|&i| (f64::from(values[i]) - f64::from(values[expected[0]])).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    for (got, exp) in f32s(weights).into_iter().zip(exps) {
        let reference = exp / sum;
        assert!(got.is_finite() && got >= 0.0);
        assert!((f64::from(got) - reference).abs() <= 2e-6 + 1e-5 * reference);
    }
}

pub(super) fn fixtures(native: Vec<f32>) -> Vec<(&'static str, Vec<f32>)> {
    let mut state = 0x4d595df4d0f33173u64;
    let mut result = Vec::new();
    for label in ["random0", "random1", "random2", "random3"] {
        let values = (0..N)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 32) as u32 as f64 / u32::MAX as f64 * 40.0 - 20.0) as f32
            })
            .collect();
        result.push((label, values));
    }
    result.push(("ascending", (0..N).map(|i| i as f32).collect()));
    result.push(("descending", (0..N).map(|i| (N - i) as f32).collect()));
    result.push(("equal", vec![3.0; N]));
    result.push((
        "signed_zero",
        (0..N)
            .map(|i| if i % 2 == 0 { -0.0 } else { 0.0 })
            .collect(),
    ));
    result.push(("half_ties", (0..N).map(|i| (i % 256) as f32).collect()));
    result.push((
        "cutoff_ties",
        (0..N)
            .map(|i| if i < 9 { 20.0 - i as f32 } else { 1.0 })
            .collect(),
    ));
    let mut ulp = vec![-100.0; N];
    for (i, v) in ulp[..11].iter_mut().enumerate() {
        *v = f32::from_bits(1.0f32.to_bits() + i as u32);
    }
    result.push(("cutoff_ulp", ulp));
    result.push((
        "finite_extremes",
        (0..N)
            .map(|i| {
                if i == 300 {
                    f32::MAX
                } else if i % 2 == 0 {
                    -f32::MAX
                } else {
                    -1000.0
                }
            })
            .collect(),
    ));
    result.push(("native", native));
    result
}

fn primitive(ctx: &MetalContext, native: Vec<f32>, artifact: &std::path::Path) {
    for (name, values) in fixtures(native) {
        let logits = guarded(ctx, bytemuck::cast_slice(&values), GgmlType::F32);
        let frozen = allocation(&logits);
        let mut observations = Vec::new();
        for parallel in [false, true] {
            let ids = guarded(ctx, bytemuck::cast_slice(&[-1i32; K]), GgmlType::I32);
            let weights = guarded(ctx, bytemuck::cast_slice(&[f32::NAN; K]), GgmlType::F32);
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            select(ctx, &enc, &logits, &ids, &weights, parallel);
            enc.end();
            command.commit();
            crate::metal::wait_unchecked(&command);
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert!(command.error().is_none());
            std::fs::write(
                artifact.join(format!("{name}-{parallel}-ids.bin")),
                bytes(&ids),
            )
            .unwrap();
            std::fs::write(
                artifact.join(format!("{name}-{parallel}-weights.bin")),
                bytes(&weights),
            )
            .unwrap();
            oracle(&values, &ids, &weights);
            guards(&ids);
            guards(&weights);
            assert_eq!(allocation(&logits), frozen);
            observations.push((bytes(&ids), bytes(&weights)));
        }
        assert_eq!(observations[0], observations[1], "{name} bitwise");
    }
    eprintln!("topk_screen 13 finite fixtures IDs/weights bitwise + CPU order/softmax PASS");
}

#[derive(Clone, Copy, Debug)]
enum Scope {
    Leaf,
    Router,
    Moe,
}

fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    c: &Capture,
    w: &Qwen4ExpMoeMetalWorkspace,
    native_logits: &MetalTensor,
    scope: Scope,
    parallel: bool,
) {
    if !matches!(scope, Scope::Leaf) {
        encode_mat_vec_dispatch(
            ctx,
            enc,
            c.weights().router,
            &c.input,
            &w.router_logits,
            c.geometry.hidden_size,
            N,
        )
        .unwrap();
    }
    select(
        ctx,
        enc,
        if matches!(scope, Scope::Leaf) {
            native_logits
        } else {
            &w.router_logits
        },
        &w.topk_ids,
        &w.topk_weights,
        parallel,
    );
    if !matches!(scope, Scope::Leaf) {
        encode_dot_sigmoid_f32(
            ctx,
            enc,
            c.weights().shared_router,
            &c.input,
            &w.shared_gate,
            c.geometry.hidden_size,
        )
        .unwrap();
    }
    if matches!(scope, Scope::Moe) {
        for index in 1..6 {
            stage(ctx, enc, c, w, index);
        }
    }
}

fn packet(
    ctx: &MetalContext,
    c: &Capture,
    w: &Qwen4ExpMoeMetalWorkspace,
    native_logits: &MetalTensor,
    scope: Scope,
    parallel: bool,
    repeats: usize,
    census: bool,
    artifact: &std::path::Path,
    name: &str,
) -> (f64, f64, Vec<Vec<u8>>, Vec<serde_json::Value>) {
    poison(w);
    if census {
        dispatch_census_begin();
    }
    let started = std::time::Instant::now();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    for _ in 0..repeats {
        encode(ctx, &enc, c, w, native_logits, scope, parallel);
    }
    enc.end();
    command.commit();
    crate::metal::wait_unchecked(&command);
    let wall = started.elapsed().as_secs_f64() * 1e3;
    let gpu = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
    let rows = if census {
        dispatch_census_take().iter().map(row_json).collect()
    } else {
        Vec::new()
    };
    std::fs::write(artifact.join(format!("{name}-packet.json")),serde_json::to_vec_pretty(&serde_json::json!({"scope":format!("{scope:?}"),"parallel":parallel,"repeats":repeats,"gpu_ms":gpu,"wall_ms":wall,"status":format!("{:?}",command.status()),"error":command.error().map(|e|e.to_string()),"census":rows,"validation":"pending"})).unwrap()).unwrap();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none());
    assert!(gpu.is_finite() && gpu > 0.0 && wall.is_finite() && wall > 0.0);
    let mut tensors = vec![&w.topk_ids, &w.topk_weights];
    if !matches!(scope, Scope::Leaf) {
        tensors.extend([&w.router_logits, &w.shared_gate]);
    }
    if matches!(scope, Scope::Moe) {
        tensors.extend([
            &w.output,
            &w.routed_inner,
            &w.shared_inner,
            &w.shared_output,
        ]);
    }
    let observations = tensors.iter().map(|t| bytes(t)).collect::<Vec<_>>();
    for (i, data) in observations.iter().enumerate() {
        std::fs::write(artifact.join(format!("{name}-tensor{i}.bin")), data).unwrap();
    }
    for t in tensors {
        if t.dtype == GgmlType::F32 {
            assert_finite(t);
        }
    }
    assert_eq!(bytes(&w.topk_ids), bytes(&c.ids));
    assert_eq!(bytes(&w.topk_weights), bytes(&c.topk));
    if matches!(scope, Scope::Moe) {
        assert_eq!(bytes(&w.output), bytes(&c.output));
    }
    (gpu, wall, observations, rows)
}

pub(crate) fn screen(
    ctx: &MetalContext,
    weights: &Qwen4ExpMetalWeights,
    source: &std::path::Path,
    artifact: &std::path::Path,
) {
    let pso = ctx.pipeline(KERNEL).unwrap();
    assert!(pso.maxTotalThreadsPerThreadgroup() >= N);
    assert!(
        pso.staticThreadgroupMemoryLength() + N * 4 * 3 <= ctx.device.maxThreadgroupMemoryLength()
    );
    let c = load_saved_layer2(ctx, weights, source);
    assert_eq!(c.geometry.expert_count, N);
    assert_eq!(c.geometry.experts_per_token, K);
    let w = Qwen4ExpMoeMetalWorkspace::new(ctx, c.geometry).unwrap();
    validate_contract(ctx, &c.input, c.weights(), &w).unwrap();
    preflight(ctx, c.weights()).unwrap();
    let frozen_input = bytes(&c.input);
    let frozen_router = bytes(c.weights().router);
    let frozen_shared_router = bytes(c.weights().shared_router);
    let dummy = MetalTensor::zeros_f32(ctx, vec![N as u64]).unwrap();
    packet(
        ctx,
        &c,
        &w,
        &dummy,
        Scope::Router,
        false,
        1,
        false,
        artifact,
        "capture-router",
    );
    let native_values = f32s(&w.router_logits);
    let native_logits = guarded(ctx, bytemuck::cast_slice(&native_values), GgmlType::F32);
    let frozen_logits = allocation(&native_logits);
    primitive(ctx, native_values, artifact);
    let mut qualified = Vec::new();
    for scope in [Scope::Leaf, Scope::Router, Scope::Moe] {
        let a = packet(
            ctx,
            &c,
            &w,
            &native_logits,
            scope,
            false,
            1,
            true,
            artifact,
            &format!("{scope:?}-qualify-a"),
        );
        let b = packet(
            ctx,
            &c,
            &w,
            &native_logits,
            scope,
            true,
            1,
            true,
            artifact,
            &format!("{scope:?}-qualify-b"),
        );
        assert_eq!(a.2, b.2, "{scope:?} intermediates");
        assert_eq!(
            a.3.len(),
            match scope {
                Scope::Leaf => 1,
                Scope::Router => 3,
                Scope::Moe => 8,
            }
        );
        let mut expected = a.3.clone();
        let mut changed = 0;
        for row in &mut expected {
            if row["kernel"] == "kernel_topk_logits_softmax_f32" {
                row["kernel"] = serde_json::json!(KERNEL);
                row["threads"] = serde_json::json!([N, 1, 1]);
                changed += 1;
            }
        }
        assert_eq!(changed, 1);
        assert_eq!(b.3, expected);
        qualified.push((scope, a.2));
    }
    assert_eq!(bytes(&c.input), frozen_input);
    assert_eq!(bytes(c.weights().router), frozen_router);
    assert_eq!(bytes(c.weights().shared_router), frozen_shared_router);
    assert_eq!(allocation(&native_logits), frozen_logits);
    let mut results = Vec::new();
    for (scope, expected_output) in qualified {
        let mut measured = Vec::new();
        for warm in [true, false] {
            for (i, parallel) in [false, true, true, false].into_iter().enumerate() {
                let observed = packet(
                    ctx,
                    &c,
                    &w,
                    &native_logits,
                    scope,
                    parallel,
                    16,
                    false,
                    artifact,
                    &format!("{scope:?}-warm{warm}-{i}"),
                );
                assert_eq!(observed.2, expected_output);
                if !warm {
                    measured.push((observed.0, observed.1));
                }
            }
        }
        let mut axes = Vec::new();
        for (axis, floor) in [(0, 0.10), (1, 0.05)] {
            let v: Vec<_> = measured
                .iter()
                .map(|t| if axis == 0 { t.0 } else { t.1 })
                .collect();
            let amean = (v[0] + v[3]) * 0.5;
            let bmean = (v[1] + v[2]) * 0.5;
            let drift = (v[0] - v[3]).abs() / amean;
            let useful = bmean <= amean * (1.0 - floor)
                && v[1] <= v[0] * (1.0 - floor)
                && v[2] <= v[3] * (1.0 - floor);
            axes.push(serde_json::json!({"axis":if axis==0{"gpu"}else{"executor_wall"},"abba_ms":v,"saved_fraction":1.0-bmean/amean,"drift_fraction":drift,"verdict":if drift>0.05{"INCONCLUSIVE"}else if useful{"USEFUL"}else{"HOLD"}}));
        }
        let result = serde_json::json!({"scope":format!("{scope:?}"),"axes":axes,"bitwise":true});
        eprintln!("topk_screen {result}");
        results.push(result);
    }
    assert_eq!(bytes(&c.input), frozen_input);
    assert_eq!(bytes(c.weights().router), frozen_router);
    assert_eq!(bytes(c.weights().shared_router), frozen_shared_router);
    assert_eq!(allocation(&native_logits), frozen_logits);
    std::fs::write(artifact.join("result.json"),serde_json::to_vec_pretty(&serde_json::json!({"scope":"finite N512/K10 research only","results":results,"inputs_router_weights_immutable":true,"nonfinite_product_compatibility":"unqualified"})).unwrap()).unwrap();
}
