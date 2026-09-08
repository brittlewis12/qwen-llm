const DIAGNOSTIC_BASE: usize = 8064;
const DIAGNOSTIC_PAIR: usize = 2 * 4096;
const DIAGNOSTIC_RESIDUAL: usize = 3 * 6656;

pub(super) struct PackedNumericalCapture {
    query: MetalTensor,
    online: MetalTensor,
    tiled: MetalTensor,
    actual: MetalTensor,
    residual: MetalTensor,
}

impl PackedNumericalCapture {
    fn new(ctx: &MetalContext) -> Self {
        let pair = || MetalTensor::zeros_f32(ctx, vec![(52 * DIAGNOSTIC_PAIR) as u64]).unwrap();
        Self {
            query: pair(),
            online: pair(),
            tiled: pair(),
            actual: pair(),
            residual: MetalTensor::zeros_f32(ctx, vec![(52 * 2 * DIAGNOSTIC_RESIDUAL) as u64])
                .unwrap(),
        }
    }

    pub(super) fn attention(
        &self,
        ctx: &MetalContext,
        encoder: &KernelEncoder,
        layer: usize,
        base: usize,
        rows: usize,
        session: &MuseGlimmerTextSession,
        query: &MetalTensor,
        actual: &MetalTensor,
        sliding: bool,
    ) -> Result<(), MuseGlimmerTextSessionError> {
        assert_eq!((base, rows), (DIAGNOSTIC_BASE, 128));
        let query = query.view_subrange((84 * 4096) as u64, vec![DIAGNOSTIC_PAIR as u64]);
        let destination = self.query.view_subrange(
            (layer * DIAGNOSTIC_PAIR) as u64,
            vec![DIAGNOSTIC_PAIR as u64],
        );
        encode_copy_offset_f32(ctx, encoder, &query, 0, &destination, DIAGNOSTIC_PAIR)?;
        let destination = self.actual.view_subrange(
            (layer * DIAGNOSTIC_PAIR) as u64,
            vec![DIAGNOSTIC_PAIR as u64],
        );
        encode_copy_offset_f32(
            ctx,
            encoder,
            actual,
            84 * 4096,
            &destination,
            DIAGNOSTIC_PAIR,
        )?;
        let (key, value) = session.cache_prefix_views(layer, base + 86)?;
        for (tiled, storage) in [(false, &self.online), (true, &self.tiled)] {
            let output = storage.view_subrange(
                (layer * DIAGNOSTIC_PAIR) as u64,
                vec![DIAGNOSTIC_PAIR as u64],
            );
            crate::muse_glimmer_metal::with_tiled_prefill(tiled, || {
                crate::muse_glimmer_metal::encode_muse_glimmer_attn_prefill_with_online(
                    ctx,
                    encoder,
                    &query,
                    &key,
                    &value,
                    &output,
                    2,
                    base + 84,
                    32,
                    2,
                    128,
                    sliding.then_some(2048),
                    true,
                )
            })?;
        }
        Ok(())
    }

    pub(super) fn residual(
        &self,
        ctx: &MetalContext,
        encoder: &KernelEncoder,
        layer: usize,
        stage: usize,
        residual: &MetalTensor,
    ) -> Result<(), MuseGlimmerTextSessionError> {
        for (slot, row) in [84, 85, 127].into_iter().enumerate() {
            let offset = (layer * 2 + stage) * DIAGNOSTIC_RESIDUAL + slot * 6656;
            let output = self.residual.view_subrange(offset as u64, vec![6656]);
            encode_copy_offset_f32(ctx, encoder, residual, row * 6656, &output, 6656)?;
        }
        Ok(())
    }
}

fn diagnostic_prefix_io(session: &MuseGlimmerTextSession, file: &mut std::fs::File, restore: bool) {
    use std::io::{Read, Write};
    for layer in 0..52 {
        for tensor in [
            session
                .cache_prefix_views(layer, DIAGNOSTIC_BASE)
                .unwrap()
                .0,
            session
                .cache_prefix_views(layer, DIAGNOSTIC_BASE)
                .unwrap()
                .1,
        ] {
            unsafe {
                let pointer =
                    (tensor.buffer.contents().as_ptr() as *mut u8).add(tensor.offset as usize);
                let bytes = DIAGNOSTIC_BASE * 256 * 2;
                if restore {
                    file.read_exact(std::slice::from_raw_parts_mut(pointer, bytes))
                        .unwrap();
                } else {
                    file.write_all(std::slice::from_raw_parts(pointer, bytes))
                        .unwrap();
                }
            }
        }
    }
}

fn diagnostic_f64(
    query: &[f32],
    key: &MetalTensor,
    value: &MetalTensor,
    head: usize,
    start: usize,
    end: usize,
) -> Vec<f64> {
    let bits = |tensor: &MetalTensor| unsafe {
        std::slice::from_raw_parts(
            (tensor.buffer.contents().as_ptr() as *const u8).add(tensor.offset as usize)
                as *const u16,
            end * 256,
        )
    };
    let key = bits(key);
    let value = bits(value);
    let mut scores = Vec::new();
    for position in start..end {
        let offset = position * 256 + head / 16 * 128;
        scores.push(
            (0..128)
                .map(|d| {
                    query[head * 128 + d] as f64 * half::f16::from_bits(key[offset + d]).to_f64()
                })
                .sum::<f64>()
                / 128.0_f64.sqrt(),
        );
    }
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut output = vec![0.0; 128];
    let mut denominator = 0.0;
    for (position, score) in (start..end).zip(scores) {
        let probability = (score - maximum).exp();
        denominator += probability;
        for d in 0..128 {
            output[d] += probability
                * half::f16::from_bits(value[position * 256 + head / 16 * 128 + d]).to_f64();
        }
    }
    for value in &mut output {
        *value /= denominator;
    }
    output
}

#[test]
#[ignore = "serial Metal, reusable 8K prefix and same-input/per-layer numerical diagnosis; no timing verdict"]
fn tiled_prefill_numerical_diagnostic() {
    use std::io::Write;
    let root = std::path::Path::new("target/profiles/muse-tiled-diagnostic");
    let attempt = std::env::var("MUSE_DIAGNOSTIC_LABEL").unwrap_or_else(|_| "manual-01".into());
    assert!(
        attempt
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
    );
    assert!(root.is_dir());
    for tiled in [false, true] {
        for suffix in [
            "query.f32",
            "online.f32",
            "tiled.f32",
            "actual.f32",
            "residual.f32",
            "final.f32",
            "kv-tail.f16",
        ] {
            assert!(
                !root
                    .join(format!("{attempt}-arm-{tiled}-{suffix}"))
                    .exists(),
                "attempt artifacts already exist"
            );
        }
    }
    let path = crate::test_fixtures::MUSE_GLIMMER_Q8_0.path();
    let gguf = GgufFile::open(path).unwrap();
    let config = MuseGlimmerConfig::from_gguf(&gguf).unwrap();
    let tokens = long_context_tokens(path, &config);
    let model_metadata = std::fs::metadata(path).unwrap();
    let identity = serde_json::json!({"version":1,"model":path,"model_bytes":model_metadata.len(),"model_modified_ns":model_metadata.modified().unwrap().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos().to_string(),"geometry":{"layers":52,"hidden":6656,"kv_width":256,"head_dim":128,"window":2048,"capacity":8192},"math":{"matrix":true,"tiled_f32":true,"chunk":128},"prefix_tokens":&tokens[..DIAGNOSTIC_BASE],"next_tokens":&tokens[DIAGNOSTIC_BASE..8192]});
    let ctx = MetalContext::new().unwrap();
    let transaction = ctx.begin_allocation_transaction();
    let plan = MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf).unwrap();
    let geometry = MuseGlimmerTextGeometry::from_config(&config, 8192).unwrap();
    let session_plan = MuseGlimmerTextSessionMemoryPlan::for_geometry(&ctx, &geometry).unwrap();
    let admission = evaluate_metal_memory_admission_with_cpu_bytes(
        plan.memory_plan().priced_upper_bytes()
            + session_plan.priced_upper_bytes()
            + 16 * 1024 * 1024,
        128 * 1024 * 1024,
        MUSE_GLIMMER_TEXT_SESSION_RESERVE_BYTES,
        ctx.memory_signals(),
        true,
    );
    assert!(admission.admitted, "diagnostic admission {admission:?}");
    let weights =
        MuseGlimmerMetalWeights::realize(&ctx, &gguf, plan.admit(ctx.memory_signals()).unwrap())
            .unwrap()
            .into_weights();
    let mut forward =
        MuseGlimmerTextForward::new_with_optimized_prefill(&ctx, &weights, true).unwrap();
    let mut session = MuseGlimmerTextSession::new(&ctx, weights.config(), 8192).unwrap();
    let capture = PackedNumericalCapture::new(&ctx);
    drop(transaction);
    let manifest_path = root.join("prefix.json");
    if manifest_path.exists() {
        let manifest: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["identity"], identity);
        let mut file = std::fs::File::open(root.join("prefix.bin")).unwrap();
        assert_eq!(
            file.metadata().unwrap().len(),
            (52 * 2 * DIAGNOSTIC_BASE * 256 * 2) as u64
        );
        diagnostic_prefix_io(&session, &mut file, true);
        session.next_position = DIAGNOSTIC_BASE;
        assert_eq!(
            long_context_prefix_hash(&session, DIAGNOSTIC_BASE)
                .to_hex()
                .to_string(),
            manifest["hash"].as_str().unwrap()
        );
        eprintln!(
            "MUSE_DIAGNOSTIC_JSON {}",
            serde_json::json!({"kind":"prefix_restored","positions":DIAGNOSTIC_BASE})
        );
    } else {
        crate::muse_glimmer_metal::with_tiled_prefill(true, || {
            forward.prefill(&tokens[..DIAGNOSTIC_BASE], &mut session)
        })
        .unwrap();
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join("prefix.bin"))
            .unwrap();
        diagnostic_prefix_io(&session, &mut file, false);
        file.sync_all().unwrap();
        let manifest = serde_json::json!({"identity":identity,"producer_source":std::env::var("MUSE_DIAGNOSTIC_SOURCE").unwrap_or_else(|_| "unknown-manual".into()),"hash":long_context_prefix_hash(&session, DIAGNOSTIC_BASE).to_hex().to_string()});
        serde_json::to_writer(
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(manifest_path)
                .unwrap(),
            &manifest,
        )
        .unwrap();
        eprintln!(
            "MUSE_DIAGNOSTIC_JSON {}",
            serde_json::json!({"kind":"prefix_saved","bytes":file.metadata().unwrap().len()})
        );
    }
    let prefix_hash = long_context_prefix_hash(&session, DIAGNOSTIC_BASE);
    eprintln!(
        "MUSE_DIAGNOSTIC_JSON {}",
        serde_json::json!({"kind":"attempt_prefix","attempt":attempt,"hash":prefix_hash.to_hex().to_string()})
    );
    let mut capture = Some(capture);
    let mut residuals = Vec::new();
    let mut finals = Vec::new();
    for tiled in [false, true] {
        session.rewind_prefix(DIAGNOSTIC_BASE).unwrap();
        let plain = crate::muse_glimmer_metal::with_tiled_prefill(tiled, || {
            forward.prefill(&tokens[DIAGNOSTIC_BASE..8192], &mut session)
        })
        .unwrap();
        let plain_hash = long_context_prefix_hash(&session, 8192);
        forward.packed_numerical_capture = capture.take();
        let buffers = forward.packed_numerical_capture.as_ref().unwrap();
        for tensor in [
            &buffers.query,
            &buffers.online,
            &buffers.tiled,
            &buffers.actual,
            &buffers.residual,
        ] {
            unsafe {
                (tensor.buffer.contents().as_ptr() as *mut u8)
                    .add(tensor.offset as usize)
                    .write_bytes(0xff, tensor.n_elements() as usize * 4);
            }
        }
        session.rewind_prefix(DIAGNOSTIC_BASE).unwrap();
        let captured = crate::muse_glimmer_metal::with_tiled_prefill(tiled, || {
            forward.prefill(&tokens[DIAGNOSTIC_BASE..8192], &mut session)
        })
        .unwrap();
        assert_logits_bitwise_equal("diagnostic observer endpoint", &plain, &captured);
        assert_eq!(long_context_prefix_hash(&session, 8192), plain_hash);
        assert_eq!(
            long_context_prefix_hash(&session, DIAGNOSTIC_BASE),
            prefix_hash
        );
        capture = forward.packed_numerical_capture.take();
        let captured = capture.as_ref().unwrap();
        let query = read_f32(&captured.query);
        let online = read_f32(&captured.online);
        let candidate = read_f32(&captured.tiled);
        let actual = read_f32(&captured.actual);
        assert_logits_bitwise_equal(
            "N2 shadow matches N128 consumed attention",
            &actual,
            if tiled { &candidate } else { &online },
        );
        let residual = read_f32(&captured.residual);
        let final_rows = read_f32(
            &session
                .packed
                .views(&session.geometry, 128)
                .unwrap()
                .residual,
        );
        for (label, data) in [
            ("query", &query),
            ("online", &online),
            ("tiled", &candidate),
            ("actual", &actual),
            ("residual", &residual),
            ("final", &final_rows),
        ] {
            assert!(data.iter().all(|v| v.is_finite()));
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(root.join(format!("{attempt}-arm-{tiled}-{label}.f32")))
                .unwrap();
            file.write_all(bytemuck::cast_slice(data)).unwrap();
        }
        let mut tails = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join(format!("{attempt}-arm-{tiled}-kv-tail.f16")))
            .unwrap();
        for layer in 0..52 {
            let (key, value) = session.cache_prefix_views(layer, 8192).unwrap();
            for tensor in [key, value] {
                unsafe {
                    tails
                        .write_all(std::slice::from_raw_parts(
                            (tensor.buffer.contents().as_ptr() as *const u8)
                                .add(tensor.offset as usize + DIAGNOSTIC_BASE * 256 * 2),
                            128 * 256 * 2,
                        ))
                        .unwrap();
                }
            }
            let offset = layer * DIAGNOSTIC_PAIR;
            let online = &online[offset..offset + DIAGNOSTIC_PAIR];
            let candidate = &candidate[offset..offset + DIAGNOSTIC_PAIR];
            let comparison = compare_logits(candidate, online);
            let index = online
                .iter()
                .zip(candidate)
                .enumerate()
                .max_by(|(_, (a, b)), (_, (c, d))| (*a - *b).abs().total_cmp(&(*c - *d).abs()))
                .unwrap()
                .0;
            let row = index / 4096;
            let head = index % 4096 / 128;
            let end = DIAGNOSTIC_BASE + 84 + row + 1;
            let start = if forward.weights.layers[layer].sliding_attention {
                end.saturating_sub(2048)
            } else {
                0
            };
            let (key, value) = session.cache_prefix_views(layer, end).unwrap();
            let expected = diagnostic_f64(
                &query[offset + row * 4096..offset + (row + 1) * 4096],
                &key,
                &value,
                head,
                start,
                end,
            );
            let errors: Vec<f64> = [online, candidate]
                .iter()
                .map(|values| {
                    (0..128)
                        .map(|dim| {
                            (values[row * 4096 + head * 128 + dim] as f64 - expected[dim]).abs()
                        })
                        .fold(0.0_f64, f64::max)
                })
                .collect();
            eprintln!(
                "MUSE_DIAGNOSTIC_JSON {}",
                serde_json::json!({"kind":"same_input_attention","graph_tiled":tiled,"layer":layer,"pair_rms":comparison.relative_rms,"pair_max_abs":comparison.max_abs,"row":row+84,"head":head,"online_f64_max_abs":errors[0],"tiled_f64_max_abs":errors[1]})
            );
        }
        residuals.push(residual);
        finals.push(final_rows);
    }
    for (slot, (a, b)) in residuals[0]
        .chunks_exact(6656)
        .zip(residuals[1].chunks_exact(6656))
        .enumerate()
    {
        let comparison = compare_logits(b, a);
        eprintln!(
            "MUSE_DIAGNOSTIC_JSON {}",
            serde_json::json!({"kind":"layer_residual","layer":slot/6,"stage":if slot/3%2==0 {"post_attention"}else{"post_ffn"},"row":([84,85,127][slot%3]),"relative_rms":comparison.relative_rms,"max_abs":comparison.max_abs,"reference_norm":a.iter().map(|&x|(x as f64).powi(2)).sum::<f64>().sqrt()})
        );
    }
    for row in [84, 85, 113, 119, 127] {
        let mut readouts = Vec::new();
        for residual in &finals {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    residual[row * 6656..(row + 1) * 6656].as_ptr(),
                    (session.residual.buffer.contents().as_ptr() as *mut u8)
                        .add(session.residual.offset as usize) as *mut f32,
                    6656,
                );
            }
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            forward
                .encode_deployed_output_tail(&encoder, &session)
                .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert!(command.error().is_none());
            readouts.push(session.read_logits());
        }
        let comparison = compare_logits(&readouts[1], &readouts[0]);
        eprintln!(
            "MUSE_DIAGNOSTIC_JSON {}",
            serde_json::json!({"kind":"selected_row_deployed_readout","row":row,"cosine":comparison.cosine,"relative_rms":comparison.relative_rms,"max_abs":comparison.max_abs,"top1_equal":comparison.reference_argmax==comparison.candidate_argmax})
        );
        assert!(
            comparison.cosine > 0.999_99
                && comparison.relative_rms < 0.002
                && comparison.max_abs < 0.1,
            "selected row readout {row}: {comparison:?}"
        );
    }
    for (row, (a, b)) in finals[0]
        .chunks_exact(6656)
        .zip(finals[1].chunks_exact(6656))
        .enumerate()
    {
        let comparison = compare_logits(b, a);
        eprintln!(
            "MUSE_DIAGNOSTIC_JSON {}",
            serde_json::json!({"kind":"final_row","row":row,"relative_rms":comparison.relative_rms,"cosine":comparison.cosine,"max_abs":comparison.max_abs})
        );
    }
}
