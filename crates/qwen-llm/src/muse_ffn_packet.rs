const FFN_BASE: usize = 32768;
const FFN_SAVED_BASE: usize = 32640;
const FFN_SAMPLES: usize = 8;
const FFN_LAYERS: usize = 52;
const FFN_HIDDEN: usize = 6656;
const FFN_INNER: usize = 19968;
const FFN_REPEATS: usize = 64;
const FFN_PREFIX_HASH: &str = "2da45d5f9dbba777c32aa3ac871b9cfe8a35d9ddffdc699e1bc3036eabf3cca1";
const FFN_PREFIX_SOURCE: &str = "7d3fab2e4c74e881af0cd37a2e9ab6c715c2c673";
const FFN_MODEL_SHA256: &str = "f2c087d694ca8242a4a436076df7c041703ab051ac4b72bb1bfe2698299b0e86";

pub(super) struct MuseFfnCapture {
    streams: [MetalTensor; 4],
    written: std::cell::RefCell<Vec<[bool; 4]>>,
}

fn ffn_capture_slot(position: usize, layer: usize) -> Option<usize> {
    let sample = position.checked_sub(FFN_BASE)?;
    (sample < FFN_SAMPLES && layer < FFN_LAYERS).then(|| sample * FFN_LAYERS + layer)
}

impl MuseFfnCapture {
    fn widths() -> [usize; 4] {
        [FFN_HIDDEN, FFN_INNER, FFN_INNER, FFN_INNER]
    }

    fn new(ctx: &MetalContext) -> Self {
        let streams = Self::widths().map(|width| {
            let elements = FFN_SAMPLES * FFN_LAYERS * width;
            let tensor = MetalTensor::zeros_f32(ctx, vec![elements as u64]).unwrap();
            ffn_poison_f32(&tensor);
            tensor
        });
        Self {
            streams,
            written: std::cell::RefCell::new(vec![[false; 4]; FFN_SAMPLES * FFN_LAYERS]),
        }
    }

    pub(super) fn copy(
        &self,
        ctx: &MetalContext,
        encoder: &KernelEncoder,
        position: usize,
        layer: usize,
        stream: usize,
        source: &MetalTensor,
    ) -> Result<(), MuseGlimmerTextSessionError> {
        let slot =
            ffn_capture_slot(position, layer).expect("FFN observer outside frozen scalar packet");
        let width = Self::widths()[stream];
        assert_eq!(source.n_elements() as usize, width);
        let mut written = self.written.borrow_mut();
        assert!(!written[slot][stream], "duplicate FFN capture");
        let output = self.streams[stream].view_subrange((slot * width) as u64, vec![width as u64]);
        encode_copy_offset_f32(ctx, encoder, source, 0, &output, width)?;
        written[slot][stream] = true;
        Ok(())
    }

    fn validate(&self) {
        assert!(
            self.written
                .borrow()
                .iter()
                .flatten()
                .all(|written| *written)
        );
        for tensor in &self.streams {
            assert!(
                ffn_f32_slice(tensor).iter().all(|v| v.is_finite()),
                "unwritten/nonfinite FFN capture"
            );
        }
    }
}

fn ffn_byte_range(
    offset: u64,
    tensor_bytes: u64,
    buffer_bytes: u64,
    relative: u64,
    bytes: u64,
) -> Option<usize> {
    let relative_end = relative.checked_add(bytes)?;
    let start = offset.checked_add(relative)?;
    let end = start.checked_add(bytes)?;
    (relative_end <= tensor_bytes && end <= buffer_bytes).then_some(usize::try_from(start).ok()?)
}

fn ffn_tensor_byte_start(tensor: &MetalTensor, relative: u64, bytes: u64) -> usize {
    ffn_byte_range(
        tensor.offset,
        tensor.n_bytes(),
        tensor.buffer.length() as u64,
        relative,
        bytes,
    )
    .expect("FFN packet host access outside tensor/buffer range")
}

fn ffn_poison_f32(tensor: &MetalTensor) {
    assert_eq!(tensor.dtype, GgmlType::F32);
    assert!(tensor.is_writable());
    assert_eq!(tensor.offset % 4, 0);
    let start = ffn_tensor_byte_start(tensor, 0, tensor.n_elements().checked_mul(4).unwrap());
    // All callers own idle output storage; no outstanding GPU command or borrowed CPU slice.
    unsafe {
        std::slice::from_raw_parts_mut(
            (tensor.buffer.contents().as_ptr() as *mut u8)
                .add(start)
                .cast::<f32>(),
            usize::try_from(tensor.n_elements()).unwrap(),
        )
        .fill(f32::NAN);
    }
}

fn ffn_f32_slice(tensor: &MetalTensor) -> &[f32] {
    assert_eq!(tensor.dtype, GgmlType::F32);
    assert_eq!(tensor.offset % 4, 0);
    let start = ffn_tensor_byte_start(tensor, 0, tensor.n_elements().checked_mul(4).unwrap());
    // Only used after synchronous command completion, or before any command is encoded.
    unsafe {
        std::slice::from_raw_parts(
            (tensor.buffer.contents().as_ptr() as *const u8)
                .add(start)
                .cast::<f32>(),
            tensor.n_elements() as usize,
        )
    }
}

fn ffn_json_new(path: &std::path::Path, value: &serde_json::Value) {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    serde_json::to_writer_pretty(&mut file, value).unwrap();
    writeln!(file).unwrap();
    file.sync_all().unwrap();
}

fn ffn_write_stream(
    root: &std::path::Path,
    name: &str,
    tensor: &MetalTensor,
    width: usize,
) -> serde_json::Value {
    use std::io::Write;
    assert!(cfg!(target_endian = "little"));
    let bytes = bytemuck::cast_slice::<f32, u8>(ffn_f32_slice(tensor));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(name))
        .unwrap();
    file.write_all(bytes).unwrap();
    file.sync_all().unwrap();
    serde_json::json!({"name":name,"dtype":"f32le","shape":[FFN_SAMPLES,FFN_LAYERS,width],
        "byte_length":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes))})
}

fn ffn_screen_passes(gpu: [f64; 4], wall: [f64; 4]) -> bool {
    if gpu.iter().chain(&wall).any(|v| !v.is_finite() || *v <= 0.0) {
        return false;
    }
    let mean_a = (gpu[0] + gpu[3]) / 2.0;
    let mean_b = (gpu[1] + gpu[2]) / 2.0;
    let wall_a = (wall[0] + wall[3]) / 2.0;
    let wall_b = (wall[1] + wall[2]) / 2.0;
    (mean_a - mean_b) / mean_a >= 0.10
        && (gpu[0] - gpu[1]) / gpu[0] >= 0.10
        && (gpu[3] - gpu[2]) / gpu[3] >= 0.10
        && (gpu[0] - gpu[3]).abs() / mean_a <= 0.05
        && (wall[0] - wall[3]).abs() / wall_a <= 0.05
        && wall_b <= wall_a
}

#[test]
fn muse_ffn_packet_cpu_contracts() {
    assert_eq!(ffn_capture_slot(FFN_BASE - 1, 0), None);
    assert_eq!(ffn_capture_slot(FFN_BASE, 0), Some(0));
    assert_eq!(ffn_capture_slot(FFN_BASE + 7, 51), Some(415));
    assert_eq!(ffn_capture_slot(FFN_BASE + 8, 0), None);
    assert_eq!(ffn_capture_slot(FFN_BASE, 52), None);
    assert_eq!(ffn_capture_slot(usize::MAX, usize::MAX), None);
    assert_eq!(ffn_byte_range(32, 128, 256, 16, 112), Some(48));
    assert_eq!(ffn_byte_range(32, 128, 256, 16, 113), None);
    assert_eq!(ffn_byte_range(32, 128, 128, 16, 112), None);
    assert_eq!(ffn_byte_range(u64::MAX, 128, u64::MAX, 1, 1), None);
    assert_eq!(ffn_byte_range(0, u64::MAX, u64::MAX, u64::MAX, 1), None);
    assert_eq!(
        MuseFfnCapture::widths().iter().sum::<usize>() * 8 * 52 * 4,
        110755840
    );
    let wall = [100.0, 90.0, 90.0, 100.0];
    assert!(ffn_screen_passes(wall, wall));
    assert!(!ffn_screen_passes([100.0, 89.0, 91.0, 100.0], wall));
    assert!(!ffn_screen_passes([100.0, 80.0, 80.0, 106.0], wall));
    assert!(!ffn_screen_passes(wall, [100.0, 101.0, 101.0, 100.0]));
    assert!(!ffn_screen_passes(wall, [100.0, 90.0, 90.0, 106.0]));
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(!ffn_screen_passes([bad, 80.0, 80.0, 100.0], wall));
        assert!(!ffn_screen_passes(wall, [100.0, bad, 90.0, 100.0]));
    }
}

#[test]
#[ignore = "coordinated GPU only, production lease, all-layer Muse FFN capture and frozen fusion screen"]
fn muse_ffn_capture_and_fusion_screen() {
    let _lease = crate::metal::acquire_metal_benchmark_lease()
        .expect("production lease and wired gate required");
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let source =
        std::env::var("MUSE_FFN_PACKET_SOURCE").expect("declare exact tested source commit");
    assert!(
        source.len() == 40
            && source
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    );
    assert!(
        crate::metal::mat_vec_q8_0_lcpp_enabled(),
        "effective baseline must be default lcpp"
    );
    let profiles = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("target/profiles");
    let fixture_root = profiles.join("muse-tiled-diagnostic");
    let root = profiles.join(format!("muse-ffn-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap_or_else(|error| {
        panic!(
            "FFN artifact preflight before Metal: refuse reuse of {}: {error}",
            root.display()
        )
    });
    ffn_json_new(
        &root.join("attempt.json"),
        &serde_json::json!({
        "source_commit":source,"phase":"started_not_a_result","repeats_per_command":FFN_REPEATS,
        "cells":{"layers":[0,25,51],"samples":[0,7]}}),
    );
    let manifest: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(fixture_root.join("prefix32.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["hash"], FFN_PREFIX_HASH);
    assert_eq!(manifest["producer_source"], FFN_PREFIX_SOURCE);
    assert_eq!(manifest["identity"]["base"], FFN_SAVED_BASE);
    assert_eq!(
        manifest["identity"]["extension_math"],
        "current_online_matrix"
    );
    let path = crate::test_fixtures::MUSE_GLIMMER_Q8_0.path();
    let metadata = std::fs::metadata(path).unwrap();
    let seed = &manifest["identity"]["seed_identity"];
    assert_eq!(seed["model"], path);
    assert_eq!(seed["model_bytes"], metadata.len());
    assert_eq!(
        seed["model_modified_ns"],
        metadata
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .to_string()
    );
    assert_eq!(
        seed["geometry"],
        serde_json::json!({"layers":52,"hidden":6656,"kv_width":256,"head_dim":128,"window":2048,"capacity":8192})
    );
    assert_eq!(
        seed["math"],
        serde_json::json!({"matrix":true,"tiled_f32":true,"chunk":128})
    );
    let gguf = GgufFile::open(path).unwrap();
    let config = MuseGlimmerConfig::from_gguf(&gguf).unwrap();
    let tokens = long_context_tokens(path, &config);
    assert_eq!(
        manifest["identity"]["prefix_tokens"],
        serde_json::json!(&tokens[..FFN_SAVED_BASE])
    );
    assert_eq!(
        seed["prefix_tokens"],
        serde_json::json!(&tokens[..DIAGNOSTIC_BASE])
    );
    let mut prefix_file = std::fs::File::open(fixture_root.join("prefix32.bin")).unwrap();
    assert_eq!(
        prefix_file.metadata().unwrap().len(),
        (52 * 2 * FFN_SAVED_BASE * 256 * 2) as u64
    );

    let ctx = MetalContext::new().unwrap();
    assert_eq!(ctx.device.name().to_string(), "Apple M4 Max");
    assert!(ctx.device.hasUnifiedMemory());
    let transaction = ctx.begin_allocation_transaction();
    let plan = MuseGlimmerMetalWeightPlan::for_authenticated_release(&ctx, &gguf).unwrap();
    assert!(plan.contents_authenticated());
    assert_eq!(plan.authenticated_shard_sha256s(), [FFN_MODEL_SHA256]);
    assert_eq!(
        plan.artifact_profile(),
        crate::muse_glimmer::MuseGlimmerArtifactProfile::UnslothQ8_0
    );
    let geometry = MuseGlimmerTextGeometry::from_config(&config, FFN_BASE + FFN_SAMPLES).unwrap();
    let session_plan =
        MuseGlimmerTextSessionMemoryPlan::for_geometry_with_split_decode(&ctx, &geometry, true)
            .unwrap();
    let price = |bytes| {
        ctx.price_shared_buffer_upper(bytes)
            .unwrap()
            .priced_upper_bytes
    };
    let observer_price = MuseFfnCapture::widths()
        .iter()
        .map(|width| price((FFN_SAMPLES * FFN_LAYERS * width * 4) as u64))
        .try_fold(0u64, |total, value| total.checked_add(value))
        .unwrap();
    let screen_price = price(((FFN_INNER + 16) * 4) as u64).checked_mul(8).unwrap();
    let total = plan
        .memory_plan()
        .priced_upper_bytes()
        .checked_add(session_plan.priced_upper_bytes())
        .unwrap()
        .checked_add(observer_price)
        .unwrap()
        .checked_add(screen_price)
        .unwrap();
    let admission = evaluate_metal_memory_admission_with_cpu_bytes(
        total,
        128 * 1024 * 1024,
        MUSE_GLIMMER_TEXT_SESSION_RESERVE_BYTES,
        ctx.memory_signals(),
        true,
    );
    assert!(
        admission.admitted,
        "FFN packet aggregate admission {admission:?}"
    );
    let weights =
        MuseGlimmerMetalWeights::realize(&ctx, &gguf, plan.admit(ctx.memory_signals()).unwrap())
            .unwrap()
            .into_weights();
    let mut session =
        MuseGlimmerTextSession::new_with_split_decode(&ctx, &config, FFN_BASE + FFN_SAMPLES, true)
            .unwrap();
    assert_eq!(session.memory_plan(), &session_plan);
    let mut forward = MuseGlimmerTextForward::new_with_tiled_prefill(&ctx, &weights, true).unwrap();
    let capture = MuseFfnCapture::new(&ctx);
    let scratch: Vec<MetalTensor> = (0..8)
        .map(|_| {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&vec![-77.0f32; FFN_INNER + 16]),
                vec![(FFN_INNER + 16) as u64],
                GgmlType::F32,
            )
            .unwrap()
        })
        .collect();
    let observer_observed: u64 = capture
        .streams
        .iter()
        .map(|t| t.buffer.length() as u64)
        .sum();
    let screen_observed: u64 = scratch.iter().map(|t| t.buffer.length() as u64).sum();
    assert!(observer_observed <= observer_price && screen_observed <= screen_price);
    drop(transaction);
    replay_prefix_io(&session, &mut prefix_file, true, FFN_SAVED_BASE);
    assert_eq!(
        long_context_prefix_hash(&session, FFN_SAVED_BASE)
            .to_hex()
            .as_str(),
        FFN_PREFIX_HASH
    );
    session.next_position = FFN_SAVED_BASE;
    forward
        .prefill(&tokens[FFN_SAVED_BASE..FFN_BASE], &mut session)
        .unwrap();
    assert_eq!(
        long_context_prefix_hash(&session, FFN_SAVED_BASE)
            .to_hex()
            .as_str(),
        FFN_PREFIX_HASH
    );
    let common_prefix = long_context_prefix_hash(&session, FFN_BASE);
    let mut reference = Vec::new();
    for &token in &tokens[FFN_BASE..FFN_BASE + FFN_SAMPLES] {
        let logits = forward
            .forward_generated_token(token, &mut session)
            .unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(long_context_prefix_hash(&session, FFN_BASE), common_prefix);
        reference.push((
            logits,
            long_context_prefix_hash(&session, session.next_position()),
        ));
    }
    session.rewind_prefix(FFN_BASE).unwrap();
    for layer in 0..FFN_LAYERS {
        let offset = session
            .geometry
            .cache_write_offset(layer, FFN_BASE)
            .unwrap()
            * 2;
        for tensor in [&session.key_cache, &session.value_cache] {
            assert!(tensor.is_writable());
            let bytes = FFN_SAMPLES * 256 * 2;
            let start = ffn_tensor_byte_start(tensor, offset as u64, bytes as u64);
            unsafe {
                (tensor.buffer.contents().as_ptr() as *mut u8)
                    .add(start)
                    .write_bytes(0xff, bytes);
            }
        }
    }
    forward.ffn_census_capture = Some(capture);
    for (sample, &token) in tokens[FFN_BASE..FFN_BASE + FFN_SAMPLES].iter().enumerate() {
        let logits = forward
            .forward_generated_token(token, &mut session)
            .unwrap();
        assert_logits_bitwise_equal("FFN observer full logits", &reference[sample].0, &logits);
        assert_eq!(session.next_position(), FFN_BASE + sample + 1);
        assert_eq!(
            long_context_prefix_hash(&session, session.next_position()),
            reference[sample].1
        );
        assert_eq!(long_context_prefix_hash(&session, FFN_BASE), common_prefix);
    }
    let capture = forward.ffn_census_capture.take().unwrap();
    capture.validate();
    let input = ffn_write_stream(&root, "input.f32", &capture.streams[0], FFN_HIDDEN);
    let gate = ffn_write_stream(&root, "gate.f32", &capture.streams[1], FFN_INNER);
    let up = ffn_write_stream(&root, "up.f32", &capture.streams[2], FFN_INNER);
    let inner = ffn_write_stream(&root, "inner.f32", &capture.streams[3], FFN_INNER);
    let ids: Vec<i32> = tokens[FFN_BASE..FFN_BASE + FFN_SAMPLES]
        .iter()
        .map(|&t| i32::try_from(t).unwrap())
        .collect();
    let capture_manifest = serde_json::json!({"schema_version":2,"source_commit":source,
        "compiled_packet_sha256":format!("{:x}",Sha256::digest(include_bytes!("muse_ffn_packet.rs"))),
        "compiled_graph_sha256":format!("{:x}",Sha256::digest(include_bytes!("muse_glimmer_text_session.rs"))),
        "compiled_q8_shader_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("../../../kernels/mat_vec_q8_0.metal"))),
        "model_content_identity":format!("sha256:{FFN_MODEL_SHA256}"),"model_contents_authenticated":true,
        "model_identity_scope":"authenticated pinned release's single GGUF shard SHA-256",
        "prefix_identity":format!("blake3:{common_prefix}"),"restored_prefix_blake3":FFN_PREFIX_HASH,
        "restored_prefix_producer":FFN_PREFIX_SOURCE,"prefix_scope":"fixed historical replay state, not authenticated fresh-history reconstruction",
        "state_layout":"layer-major full-capacity F16 K then V,52layers/256KV, append-only causal views",
        "weight_dtype":"Q8_0","matvec_variant":"lcpp_nr0_2_nsg_4","layer_count":52,
        "layers":(0..52).collect::<Vec<_>>(),"hidden_size":FFN_HIDDEN,"intermediate_size":FFN_INNER,
        "captured_token_ids":ids,"captured_positions":(FFN_BASE..FFN_BASE+FFN_SAMPLES).collect::<Vec<_>>(),
        "consumed_token_sha256":crate::tokenizer::token_ids_sha256_i32le(&ids),
        "token_semantics":"teacher-forced consumed inputs; logits predict following token",
        "tensors":{"gate":gate,"up":up,"inner":inner},"normalized_input":input,
        "observer":{"logical_bytes":110755840,"priced_bytes":observer_price,"observed_bytes":observer_observed,
            "screen_priced_bytes":screen_price,"screen_observed_bytes":screen_observed,"cpu_oracle_budget_bytes":128*1024*1024,
            "full_logits_and_active_kv_bitwise_each_step":true,
            "immutable_prefix_each_step":true,"poisoned_destinations_finite":true,"encoded_copies":8*52*4}});
    ffn_json_new(&root.join("manifest.json"), &capture_manifest);
    eprintln!(
        "MUSE_FFN_PACKET capture={} observer_bitwise=true",
        root.display()
    );
    muse_ffn_fusion_cells(&ctx, &forward, &capture, &scratch, &root);
}

fn muse_ffn_fusion_cells(
    ctx: &MetalContext,
    forward: &MuseGlimmerTextForward<'_, '_>,
    capture: &MuseFfnCapture,
    scratch: &[MetalTensor],
    root: &std::path::Path,
) {
    let mut results = Vec::new();
    for layer_index in [0, 25, 51] {
        let layer = &forward.weights.layers[layer_index];
        for sample in [0, 7] {
            let slot = sample * FFN_LAYERS + layer_index;
            let x = capture.streams[0]
                .view_subrange((slot * FFN_HIDDEN) as u64, vec![FFN_HIDDEN as u64]);
            let expected =
                capture.streams[3].view_subrange((slot * FFN_INNER) as u64, vec![FFN_INNER as u64]);
            let expected_values = ffn_f32_slice(&expected).to_vec();
            let input_hash = format!(
                "{:x}",
                Sha256::digest(bytemuck::cast_slice::<f32, u8>(ffn_f32_slice(&x)))
            );
            let outputs: Vec<_> = scratch
                .iter()
                .map(|t| t.view_subrange(8, vec![FFN_INNER as u64]))
                .collect();
            for output in &outputs {
                ffn_poison_f32(output);
            }
            let run = |arm: usize, fused: bool, repeats: usize| {
                let started = std::time::Instant::now();
                let command = ctx.queue.commandBuffer().unwrap();
                let encoder = KernelEncoder::begin(&command);
                for _ in 0..repeats {
                    if fused {
                        crate::metal::encode_shared_swiglu_q8_0_f32(
                            ctx,
                            &encoder,
                            layer.feed_forward_gate,
                            layer.feed_forward_up,
                            &x,
                            &outputs[2 * arm],
                            FFN_HIDDEN,
                            FFN_INNER,
                        )
                        .unwrap();
                    } else {
                        encode_mat_vec_dispatch(
                            ctx,
                            &encoder,
                            layer.feed_forward_gate,
                            &x,
                            &outputs[2 * arm],
                            FFN_HIDDEN,
                            FFN_INNER,
                        )
                        .unwrap();
                        encode_mat_vec_dispatch(
                            ctx,
                            &encoder,
                            layer.feed_forward_up,
                            &x,
                            &outputs[2 * arm + 1],
                            FFN_HIDDEN,
                            FFN_INNER,
                        )
                        .unwrap();
                        encode_silu_mul_f32(
                            ctx,
                            &encoder,
                            &outputs[2 * arm],
                            &outputs[2 * arm + 1],
                            &outputs[2 * arm],
                        )
                        .unwrap();
                    }
                }
                encoder.end();
                command.commit();
                command.waitUntilCompleted();
                let wall = started.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                assert!(command.error().is_none());
                let gpu = (command.GPUEndTime() - command.GPUStartTime()) * 1000.0;
                assert!(gpu.is_finite() && gpu > 0.0 && wall.is_finite() && wall > 0.0);
                (gpu, wall)
            };
            for (arm, fused) in [(0, false), (1, true)] {
                crate::metal::dispatch_census_begin();
                run(arm, fused, 1);
                let census = crate::metal::dispatch_census_take();
                let names: Vec<_> = census.iter().map(|row| row.kernel.as_str()).collect();
                assert_eq!(
                    names,
                    if fused {
                        vec!["kernel_shared_swiglu_q8_0_f32_lcpp"]
                    } else {
                        vec![
                            "kernel_mat_vec_q8_0_f32_lcpp",
                            "kernel_mat_vec_q8_0_f32_lcpp",
                            "kernel_silu_mul_f32",
                        ]
                    }
                );
                assert_logits_bitwise_equal(
                    "fused primitive versus captured deployed inner",
                    ffn_f32_slice(&outputs[2 * arm]),
                    &expected_values,
                );
            }
            run(0, false, FFN_REPEATS);
            run(1, true, FFN_REPEATS);
            let mut gpu = [0.0; 4];
            let mut wall = [0.0; 4];
            for (arm, fused) in [false, true, true, false].into_iter().enumerate() {
                (gpu[arm], wall[arm]) = run(arm, fused, FFN_REPEATS);
            }
            for arm in 0..4 {
                let actual = ffn_f32_slice(&outputs[2 * arm]);
                assert!(actual.iter().all(|v| v.is_finite()));
                assert_logits_bitwise_equal(
                    "post-timing FFN full product",
                    actual,
                    &expected_values,
                );
            }
            for tensor in scratch {
                let values = ffn_f32_slice(tensor);
                assert!(
                    values[..8]
                        .iter()
                        .chain(&values[values.len() - 8..])
                        .all(|v| *v == -77.0)
                );
            }
            let pass = ffn_screen_passes(gpu, wall);
            assert_eq!(
                input_hash,
                format!(
                    "{:x}",
                    Sha256::digest(bytemuck::cast_slice::<f32, u8>(ffn_f32_slice(&x)))
                )
            );
            assert_logits_bitwise_equal(
                "captured product immutable",
                &expected_values,
                ffn_f32_slice(&expected),
            );
            let result = serde_json::json!({"layer":layer_index,"sample":sample,"position":FFN_BASE+sample,
                "repeats_per_command":FFN_REPEATS,"order":"ABBA","gpu_ms":gpu,"wall_ms":wall,
                "input_sha256":input_hash,"input_and_captured_product_immutable":true,
                "verdict":if pass {"PASS_ISOLATED_ONLY"} else {"HOLD"},"full_product_bitwise":true,"guards_intact":true});
            ffn_json_new(
                &root.join(format!("cell-{layer_index}-{sample}.json")),
                &result,
            );
            eprintln!("MUSE_FFN_CELL {result}");
            results.push(result);
        }
    }
    let pass = results
        .iter()
        .all(|cell| cell["verdict"] == "PASS_ISOLATED_ONLY");
    ffn_json_new(
        &root.join("screen.json"),
        &serde_json::json!({"cells":results,
        "verdict":if pass {"ADVANCE_TO_NATIVE_QUALIFICATION"} else {"HOLD_NO_RETRY"},"production_changed":false}),
    );
    assert!(pass, "isolated FFN screen HOLD; do not promote or retime");
}
