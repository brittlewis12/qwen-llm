//! Bitwise query-reuse scorer replay, boundaries and native wall probes.
#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires perf-ablations");
}
#[cfg(feature = "perf-ablations")]
fn main() -> anyhow::Result<()> {
    probe::run()
}

#[cfg(feature = "perf-ablations")]
mod probe {
    use anyhow::{Result, ensure};
    use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
    use qwen_llm::{
        gguf::GgufFile,
        glm5_next_metal::{Glm5NextSession, Glm5NextWeights},
        metal::*,
        tensor::GgmlType,
    };
    use serde_json::json;
    use std::{path::Path, time::Instant};

    fn bytes(t: &MetalTensor) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                t.buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(t.offset as usize),
                t.n_bytes() as usize,
            )
        }
    }
    fn floats(t: &MetalTensor) -> &[f32] {
        bytemuck::cast_slice(bytes(t))
    }
    fn ints(t: &MetalTensor) -> &[i32] {
        bytemuck::cast_slice(bytes(t))
    }
    fn emit(v: serde_json::Value) {
        println!("{v}");
    }
    struct GlmIndexInputs {
        position: usize,
        layer: usize,
        queries: MetalTensor,
        weights: MetalTensor,
        keys: MetalTensor,
        visible: MetalTensor,
    }
    fn load(ctx: &MetalContext, dir: &Path) -> Result<GlmIndexInputs> {
        let m: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
        let t = |name: &str, ty| -> Result<MetalTensor> {
            Ok(MetalTensor::from_bytes(
                ctx,
                &std::fs::read(dir.join(format!("{name}.bin")))?,
                serde_json::from_value(m[name].clone())?,
                ty,
            )?)
        };
        Ok(GlmIndexInputs {
            position: m["position"].as_u64().unwrap() as usize,
            layer: m["layer"].as_u64().unwrap() as usize,
            queries: t("queries", GgmlType::F16)?,
            weights: t("weights", GgmlType::F32)?,
            keys: t("keys", GgmlType::F16)?,
            visible: t("visible", GgmlType::I32)?,
        })
    }
    struct ChainScratch {
        scores: MetalTensor,
        pool_ids: MetalTensor,
        pool_counts: MetalTensor,
        row_ids: MetalTensor,
        row_counts: MetalTensor,
        status: MetalTensor,
        visible_rows: MetalTensor,
    }
    struct ChainCapture {
        scores: MetalTensor,
        pools: MetalTensor,
        pool_counts: MetalTensor,
        rows: MetalTensor,
        row_counts: MetalTensor,
    }
    fn chain(
        ctx: &MetalContext,
        x: &GlmIndexInputs,
        scratch: &ChainScratch,
        candidate: bool,
        repeats: usize,
        first: usize,
        capture: Option<&ChainCapture>,
    ) -> Result<(f64, f64)> {
        let count = x.queries.shape[2] as usize;
        let capacity = x.keys.shape[1] as usize;
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let host = Instant::now();
        for _ in 0..repeats {
            for start in (first..count).step_by(64) {
                let n = 64.min(count - start);
                let queries = x
                    .queries
                    .view_subrange((start * 4096) as u64, vec![128, 32, n as u64]);
                let weights = x
                    .weights
                    .view_subrange((start * 32) as u64, vec![32, n as u64]);
                let visible = x.visible.view_subrange(start as u64, vec![n as u64]);
                let visible_rows = scratch
                    .visible_rows
                    .view_subrange(start as u64, vec![n as u64]);
                // Exactly the production reuse: every microbatch starts at zero
                // in scores, pool IDs/counts and expanded IDs/counts. Status is
                // retained by row, as in Glm5NextSession::encode_sparse_rows.
                let scores = scratch
                    .scores
                    .view_subrange(0, vec![capacity as u64, n as u64]);
                let pool_ids = scratch.pool_ids.view_subrange(0, vec![512, n as u64]);
                let pool_counts = scratch.pool_counts.view_subrange(0, vec![n as u64]);
                let row_ids = scratch.row_ids.view_subrange(0, vec![2051, n as u64]);
                let row_counts = scratch.row_counts.view_subrange(0, vec![n as u64]);
                let status = scratch.status.view_subrange(start as u64, vec![n as u64]);
                let max = (x.position + start + n) / 4;
                let inputs = LightningScores {
                    queries: &queries,
                    head_weights: &weights,
                    keys: &x.keys,
                    visible_counts: &visible,
                    scores: &scores,
                };
                if !candidate
                    || !try_encode_glm_index_scores_reuse(ctx, &encoder, &inputs, capacity, max, n)?
                {
                    encode_lightning_scores_f16_matrix(
                        ctx, &encoder, &inputs, 32, 128, capacity, max, n,
                    )?;
                }
                encode_select_top_k_ids(
                    ctx,
                    &encoder,
                    &TopKSelection {
                        scores: &scores,
                        visible_counts: &visible,
                        ids: &pool_ids,
                        counts: &pool_counts,
                        status: &status,
                    },
                    capacity,
                    max,
                    512,
                    n,
                )?;
                encode_indexer_expand_selection(
                    ctx,
                    &encoder,
                    &IndexerSelection {
                        pool_ids: &pool_ids,
                        pool_counts: &pool_counts,
                        visible_rows: &visible_rows,
                        row_ids: &row_ids,
                        row_counts: &row_counts,
                    },
                    512,
                    2051,
                    n,
                )?;
                if let Some(c) = capture {
                    let mut score_bits = scores.clone();
                    score_bits.dtype = GgmlType::I32;
                    for (source, dest, width) in [
                        (&score_bits, &c.scores, capacity),
                        (&pool_ids, &c.pools, 512),
                        (&pool_counts, &c.pool_counts, 1),
                        (&row_ids, &c.rows, 2051),
                        (&row_counts, &c.row_counts, 1),
                    ] {
                        let a = source.view_subrange(0, vec![(width * n) as u64]);
                        let b =
                            dest.view_subrange((width * start) as u64, vec![(width * n) as u64]);
                        encode_copy_offset_i32(ctx, &encoder, &a, 0, &b, width * n)?;
                    }
                }
            }
        }
        encoder.end();
        command.commit();
        wait_completed(&command)?;
        Ok((
            (command.GPUEndTime() - command.GPUStartTime()) * 1e3 / repeats as f64,
            host.elapsed().as_secs_f64() * 1e3 / repeats as f64,
        ))
    }
    fn replay(ctx: &MetalContext, dir: &Path) -> Result<()> {
        for depth in [49152, 131072] {
            let x = load(ctx, &dir.join(depth.to_string()))?;
            let n = x.queries.shape[2];
            let cap = x.keys.shape[1];
            let visible_rows: Vec<i32> = (0..n)
                .map(|r| (x.position + r as usize + 1) as i32)
                .collect();
            let scratch = ChainScratch {
                scores: MetalTensor::zeros_f32(ctx, vec![cap, 64])?,
                pool_ids: MetalTensor::zeros_i32(ctx, vec![512, 64])?,
                pool_counts: MetalTensor::zeros_i32(ctx, vec![64])?,
                row_ids: MetalTensor::zeros_i32(ctx, vec![2051, 64])?,
                row_counts: MetalTensor::zeros_i32(ctx, vec![64])?,
                status: MetalTensor::zeros_i32(ctx, vec![n])?,
                visible_rows: MetalTensor::from_bytes(
                    ctx,
                    bytemuck::cast_slice(&visible_rows),
                    vec![n],
                    GgmlType::I32,
                )?,
            };
            let allocate = || -> Result<ChainCapture> {
                Ok(ChainCapture {
                    scores: MetalTensor::zeros_i32(ctx, vec![cap, n])?,
                    pools: MetalTensor::zeros_i32(ctx, vec![512, n])?,
                    pool_counts: MetalTensor::zeros_i32(ctx, vec![n])?,
                    rows: MetalTensor::zeros_i32(ctx, vec![2051, n])?,
                    row_counts: MetalTensor::zeros_i32(ctx, vec![n])?,
                })
            };
            let a = allocate()?;
            let b = allocate()?;
            for (candidate, c) in [(false, &a), (true, &b)] {
                unsafe {
                    std::slice::from_raw_parts_mut(
                        scratch
                            .scores
                            .buffer
                            .contents()
                            .as_ptr()
                            .cast::<u8>()
                            .add(scratch.scores.offset as usize),
                        scratch.scores.n_bytes() as usize,
                    )
                    .fill(0x5a);
                }
                chain(ctx, &x, &scratch, candidate, 1, 0, Some(c))?;
                ensure!(
                    ints(&scratch.status).iter().all(|s| *s == 0),
                    "replay selection status"
                );
                for r in 0..n as usize {
                    let ids = &ints(&c.pools)[r * 512..(r + 1) * 512];
                    let visible = visible_rows[r] as usize;
                    ensure!(ints(&c.pool_counts)[r] == 512, "pool count");
                    ensure!(
                        ids.windows(2).all(|w| w[0] < w[1])
                            && ids.iter().all(|v| *v >= 0 && (*v as usize) < visible / 4),
                        "ordered/future pool IDs"
                    );
                    let mut expected: Vec<i32> = ids
                        .iter()
                        .flat_map(|p| (0..4).map(move |i| p * 4 + i))
                        .collect();
                    expected.extend((visible / 4 * 4..visible).map(|i| i as i32));
                    ensure!(
                        ints(&c.row_counts)[r] as usize == expected.len(),
                        "expanded count"
                    );
                    let actual = &ints(&c.rows)[r * 2051..(r + 1) * 2051];
                    ensure!(
                        actual[..expected.len()] == expected
                            && actual[expected.len()..].iter().all(|i| *i == -1),
                        "expansion/padding"
                    );
                }
            }
            let mismatches = ints(&a.scores)
                .iter()
                .zip(ints(&b.scores))
                .filter(|(a, b)| a != b)
                .count();
            emit(
                json!({"kind":"score_bits","position":depth,"compared_scores":cap*n,"mismatches":mismatches}),
            );
            ensure!(
                mismatches == 0,
                "scores differ bitwise at {depth}: {mismatches}"
            );
            ensure!(
                bytes(&a.pools) == bytes(&b.pools)
                    && bytes(&a.pool_counts) == bytes(&b.pool_counts)
                    && bytes(&a.rows) == bytes(&b.rows)
                    && bytes(&a.row_counts) == bytes(&b.row_counts),
                "selection/expansion differ bitwise"
            );
            let changed: Vec<usize> = (0..n as usize)
                .filter(|r| {
                    ints(&a.pools)[r * 512..(r + 1) * 512] != ints(&b.pools)[r * 512..(r + 1) * 512]
                })
                .collect();
            let selections:Vec<_>=changed.iter().map(|r|json!({"query":r,"A":&ints(&a.pools)[r*512..(r+1)*512],"B":&ints(&b.pools)[r*512..(r+1)*512]})).collect();
            emit(
                json!({"kind":"chain_functional","position":depth,"queries":n,"changed_queries":changed,"selections":selections,"ordered_ids_counts_status_expansion_valid":true,"score_scratch_bytes":scratch.scores.n_bytes(),"same_scratch_across_arms":true,"copies_outside_timing":true}),
            );
            if std::env::var_os("QWEN_PROBE_FUNCTIONAL_ONLY").is_some() {
                continue;
            }
            for _ in 0..2 {
                chain(ctx, &x, &scratch, false, 3, 0, None)?;
                chain(ctx, &x, &scratch, true, 3, 0, None)?;
            }
            let mut ratios = Vec::new();
            let mut savings = Vec::new();
            for pair in 0..3 {
                let mut times = [0.; 2];
                for candidate in if pair == 1 {
                    [true, false]
                } else {
                    [false, true]
                } {
                    let (gpu, host) = chain(ctx, &x, &scratch, candidate, 10, 0, None)?;
                    times[usize::from(candidate)] = gpu;
                    emit(
                        json!({"kind":"chain_timing","position":depth,"pair":pair,"arm":if candidate{"B"}else{"A"},"gpu_ms":gpu,"host_ms":host,"repetitions":10}),
                    );
                }
                ratios.push(times[0] / times[1]);
                savings.push(times[0] - times[1]);
            }
            let speed = ratios.iter().product::<f64>().powf(1. / 3.);
            emit(
                json!({"kind":"chain_gate","position":depth,"speedup":speed,"pair_speedups":ratios,"pair_savings_ms":savings,"passed":speed>=1.25,"rule":"geomean >=1.25x at BOTH depths; bitwise all scores and selection"}),
            );
        }
        Ok(())
    }
    fn padded(
        ctx: &MetalContext,
        t: &MetalTensor,
        small: bool,
    ) -> Result<(MetalTensor, MetalTensor)> {
        let size = match t.dtype {
            GgmlType::F16 => 2,
            _ => 4,
        };
        let prefix = if small { 4 } else { 16 };
        let mut all = vec![0x5au8; prefix + t.n_bytes() as usize + 16];
        all[prefix..prefix + t.n_bytes() as usize].copy_from_slice(bytes(t));
        let base = MetalTensor::from_bytes(ctx, &all, vec![(all.len() / size) as u64], t.dtype)?;
        let view = base.view_subrange((prefix / size) as u64, t.shape.clone());
        Ok((base, view))
    }
    fn boundaries(ctx: &MetalContext, dir: &Path) -> Result<()> {
        let real = load(ctx, &dir.join("49152"))?;
        let mut tested = 0;
        for small in [false, true] {
            let mut guards = Vec::new();
            let mut pad = |t: MetalTensor| -> Result<MetalTensor> {
                let (base, view) = padded(ctx, &t, small)?;
                guards.push((base, view.offset as usize, view.n_bytes() as usize));
                Ok(view)
            };
            let keys = pad(real.keys.view_subrange(0, vec![128, 1024]))?;
            let scratch = ChainScratch {
                scores: pad(MetalTensor::zeros_f32(ctx, vec![1024, 64])?)?,
                pool_ids: pad(MetalTensor::zeros_i32(ctx, vec![512, 64])?)?,
                pool_counts: pad(MetalTensor::zeros_i32(ctx, vec![64])?)?,
                row_ids: pad(MetalTensor::zeros_i32(ctx, vec![2051, 64])?)?,
                row_counts: pad(MetalTensor::zeros_i32(ctx, vec![64])?)?,
                status: pad(MetalTensor::zeros_i32(ctx, vec![64])?)?,
                visible_rows: pad(MetalTensor::zeros_i32(ctx, vec![64])?)?,
            };
            let capture = ChainCapture {
                scores: pad(MetalTensor::zeros_i32(ctx, vec![1024, 64])?)?,
                pools: pad(MetalTensor::zeros_i32(ctx, vec![512, 64])?)?,
                pool_counts: pad(MetalTensor::zeros_i32(ctx, vec![64])?)?,
                rows: pad(MetalTensor::zeros_i32(ctx, vec![2051, 64])?)?,
                row_counts: pad(MetalTensor::zeros_i32(ctx, vec![64])?)?,
            };
            for residue in 0..4 {
                for q in [31usize, 32, 33] {
                    for ties in [false, true] {
                        let mut position = 2032 + residue;
                        let mut first = 2051 - position;
                        for consecutive in 0..2 {
                            let total = first + q;
                            let qt = if ties {
                                MetalTensor::zeros_f16(ctx, vec![128, 32, total as u64])?
                            } else {
                                real.queries.view_subrange(0, vec![128, 32, total as u64])
                            };
                            let queries = pad(qt)?;
                            let weights =
                                pad(real.weights.view_subrange(0, vec![32, total as u64]))?;
                            let rows: Vec<i32> =
                                (0..total).map(|r| (position + r + 1) as i32).collect();
                            let pools: Vec<i32> = rows.iter().map(|r| r / 4).collect();
                            // Input metadata writes are outside GPU timing, after the
                            // preceding command completed; the same scratch is reused.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    rows.as_ptr().cast::<u8>(),
                                    scratch
                                        .visible_rows
                                        .buffer
                                        .contents()
                                        .as_ptr()
                                        .cast::<u8>()
                                        .add(scratch.visible_rows.offset as usize),
                                    rows.len() * 4,
                                );
                            }
                            let visible = pad(MetalTensor::from_bytes(
                                ctx,
                                bytemuck::cast_slice(&pools),
                                vec![total as u64],
                                GgmlType::I32,
                            )?)?;
                            let x = GlmIndexInputs {
                                position,
                                layer: 0,
                                queries,
                                weights,
                                keys: keys.clone(),
                                visible,
                            };
                            let mut ids_by_arm = Vec::new();
                            let mut ref_scores: Vec<f32> = Vec::new();
                            for candidate in [false, true] {
                                for t in [
                                    &capture.pools,
                                    &capture.pool_counts,
                                    &capture.rows,
                                    &capture.row_counts,
                                ] {
                                    unsafe {
                                        std::slice::from_raw_parts_mut(
                                            t.buffer
                                                .contents()
                                                .as_ptr()
                                                .cast::<u8>()
                                                .add(t.offset as usize),
                                            t.n_bytes() as usize,
                                        )
                                        .fill(0x5a);
                                    }
                                }
                                chain(ctx, &x, &scratch, candidate, 1, first, Some(&capture))?;
                                ensure!(
                                    ints(&scratch.status)[first..total].iter().all(|v| *v == 0),
                                    "boundary status"
                                );
                                ensure!(
                                    ints(&capture.pools)[..first * 512]
                                        .iter()
                                        .all(|v| *v == 0x5a5a5a5a),
                                    "dense rows touched"
                                );
                                for r in first..total {
                                    let ids = &ints(&capture.pools)[r * 512..(r + 1) * 512];
                                    ensure!(
                                        ints(&capture.pool_counts)[r] == 512
                                            && ids.windows(2).all(|w| w[0] < w[1]),
                                        "ordered IDs/count"
                                    );
                                    ensure!(
                                        ids.iter().all(|p| *p >= 0 && *p < pools[r]),
                                        "incomplete/future pool"
                                    );
                                    if ties {
                                        ensure!(
                                            ids.iter().enumerate().all(|(i, v)| *v == i as i32),
                                            "exact tie order"
                                        );
                                    }
                                    let expected: Vec<i32> = ids
                                        .iter()
                                        .flat_map(|p| (0..4).map(move |i| p * 4 + i))
                                        .chain((rows[r] / 4 * 4..rows[r]).map(|v| v as i32))
                                        .collect();
                                    ensure!(
                                        ints(&capture.row_counts)[r] as usize == expected.len(),
                                        "expanded count"
                                    );
                                    let out = &ints(&capture.rows)[r * 2051..(r + 1) * 2051];
                                    ensure!(
                                        out[..expected.len()] == expected
                                            && out[expected.len()..].iter().all(|v| *v == -1),
                                        "expanded rows/padding"
                                    );
                                    for p in 0..pools[r] as usize {
                                        let v = floats(&scratch.scores)[(r - first) * 1024 + p];
                                        if candidate {
                                            let a = ref_scores[(r - first) * 1024 + p];
                                            ensure!(
                                                v.to_bits() == a.to_bits(),
                                                "boundary score error"
                                            );
                                        }
                                    }
                                }
                                if !candidate {
                                    ref_scores = floats(&scratch.scores).to_vec();
                                }
                                ids_by_arm
                                    .push(ints(&capture.pools)[first * 512..total * 512].to_vec());
                            }
                            ensure!(
                                ids_by_arm[0] == ids_by_arm[1],
                                "boundary selected IDs differ"
                            );
                            emit(
                                json!({"kind":"boundary","prefix":position,"prefix_mod4":position%4,"first":first,"sparse_queries":q,"ties":ties,"small_offset":small,"consecutive":consecutive,"candidate_expected":q>=32,"passed":true}),
                            );
                            tested += 1;
                            position += total;
                            first = 0;
                        }
                    }
                }
            }
            for (base, offset, n) in &guards {
                let b = bytes(base);
                ensure!(
                    b[..*offset]
                        .iter()
                        .chain(b[offset + n..].iter())
                        .all(|v| *v == 0x5a),
                    "buffer guard corrupted"
                );
            }
        }
        emit(json!({"kind":"boundary_summary","cases":tested,"passed":true}));
        Ok(())
    }
    fn native(ctx: &MetalContext, input_dir: &Path, _run_dir: &Path) -> Result<()> {
        let path = qwen_llm::test_fixtures::GLM53_FLASH_UD_IQ3_XXS
            .path_or_skip()
            .ok_or_else(|| anyhow::anyhow!("GLM fixture missing"))?;
        let gguf = GgufFile::open(path)?;
        let weights = Glm5NextWeights::load(ctx, &gguf)?;
        let raw = std::fs::read(input_dir.join("tokens.bin"))?;
        let ids: Vec<u32> = raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        ensure!(ids.len() >= 133120, "short token fixture");
        let mut session = Glm5NextSession::with_prefill_rows(ctx, &weights, 135168, 2048)?;
        let empty = session.capture_snapshot()?;
        for residue in 0..4 {
            session.restore_snapshot(&empty)?;
            let position = 2032 + residue;
            with_glm_index_reuse(false, || session.prefill_packed(ctx, &ids[..position]))?;
            let prefix = session.capture_snapshot()?;
            let first = 2051 - position;
            for n in [31usize, 32, 33] {
                let mut baseline = None;
                for candidate in [false, true] {
                    session.restore_snapshot(&prefix)?;
                    let (result, events) = with_glm_index_dispatch_trace(|| {
                        with_glm_index_reuse(candidate, || {
                            session.prefill_index_boundary_probe(
                                ctx,
                                &ids[position..position + first + n],
                            )
                        })
                    });
                    let logits = result?;
                    ensure!(
                        logits.iter().all(|v| v.is_finite()),
                        "nonfinite frontier logits"
                    );
                    ensure!(
                        events.len() == 11
                            && events
                                .iter()
                                .all(|e| *e == (first, first, n, candidate && n >= 32)),
                        "frontier dispatch {:?}",
                        events
                    );
                    let state = session.capture_snapshot()?;
                    if let Some((ref a, ref astate)) = baseline {
                        let a: &Vec<f32> = a;
                        ensure!(a.iter().zip(&logits).all(|(a,b)|a.to_bits()==b.to_bits()), "frontier logits differ");
                        ensure!(state.same_state(astate), "frontier states differ");
                    } else { baseline = Some((logits, state)); }
                    emit(
                        json!({"kind":"native_frontier","position":position,"residue":residue,"first":first,"queries":n,"candidate":candidate,"events":events,"passed":true}),
                    );
                }
            }
        }
        session.restore_snapshot(&empty)?;
        drop(empty);
        for depth in [49152usize, 131072] {
            while session.position() < depth {
                let start = session.position();
                let end = (start + 2048).min(depth);
                with_glm_index_reuse(false, || session.prefill_packed(ctx, &ids[start..end]))?;
                if end % 16384 == 0 {
                    emit(json!({"kind":"native_prefix","position":end}));
                }
            }
            let prefix = session.capture_snapshot()?;
            let tokens = &ids[depth..depth + 2048];
            for candidate in [false, true] {
                session.restore_snapshot(&prefix)?;
                with_glm_index_reuse(candidate, || session.prefill_packed(ctx, tokens))?;
            }
            let mut samples = [Vec::new(), Vec::new()];
            let mut last: [Option<Vec<f32>>; 2] = [None, None];
            let mut states = [None, None];
            for pair in 0..3 {
                for candidate in if pair == 1 {
                    [true, false]
                } else {
                    [false, true]
                } {
                    session.restore_snapshot(&prefix)?;
                    let clock = Instant::now();
                    let logits =
                        with_glm_index_reuse(candidate, || session.prefill_packed(ctx, tokens))?;
                    let wall = clock.elapsed().as_secs_f64() * 1e3;
                    ensure!(
                        logits.iter().all(|v| v.is_finite()),
                        "native nonfinite logits"
                    );
                    samples[usize::from(candidate)].push(wall);
                    last[usize::from(candidate)] = Some(logits);
                    states[usize::from(candidate)] = Some(session.capture_snapshot()?);
                    emit(
                        json!({"kind":"native_timing","position":depth,"rows":2048,"pair":pair,"arm":if candidate{"B"}else{"A"},"wall_ms":wall}),
                    );
                }
                ensure!(
                last[0]
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(last[1].as_ref().unwrap())
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "native logits differ"
            );
            ensure!(
                states[0]
                    .as_ref()
                    .unwrap()
                    .same_state(states[1].as_ref().unwrap()),
                "native states differ"
            );
                emit(json!({"kind":"native_bitwise","position":depth,"pair":pair,"logits":true,"state":true}));
            }
            let ratios: Vec<_> = samples[0]
                .iter()
                .zip(&samples[1])
                .map(|(a, b)| a / b)
                .collect();
            emit(
                json!({"kind":"native_summary","position":depth,"rows":2048,"speedup":ratios.iter().product::<f64>().powf(1./3.),"pair_speedups":ratios,"wall_ms":samples,"logits_bitwise":true,"state_bitwise":true}),
            );
            session.restore_snapshot(&prefix)?;
        }
        Ok(())
    }
    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args().collect();
        ensure!(
            args.len() >= 3,
            "replay|boundaries|native INPUT_DIR [RUN_DIR]"
        );
        let ctx = MetalContext::new()?;
        emit(json!({"kind":"device","description":ctx.describe(),"mode":args[1]}));
        let dir = Path::new(&args[2]);
        with_glm_index_reuse(true, || match args[1].as_str() {
            "replay" => replay(&ctx, dir),
            "boundaries" => boundaries(&ctx, dir),
            "native" => native(&ctx, dir, Path::new(args.get(3).unwrap_or(&args[2]))),
            _ => anyhow::bail!("unknown mode"),
        })
    }
}
