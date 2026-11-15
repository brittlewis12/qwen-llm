//! Captured-operand score/selection gates and one-binary native cold cells.
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
    use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLSize};
    use qwen_llm::{
        deepseek_v41::{
            forward::{Forward, State},
            gguf::{ResidentWeights, resident_storage_plan_with_target},
        },
        gguf::GgufFile,
        metal::*,
        tensor::GgmlType,
        tokenizer::NativeTokenizer,
    };
    use serde_json::json;
    use std::{path::Path, sync::Arc, time::Instant};
    fn emit(x: serde_json::Value) {
        println!("{x}");
    }
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
    fn ints(t: &MetalTensor) -> &[u32] {
        bytemuck::cast_slice(bytes(t))
    }
    fn fill(t: &MetalTensor) {
        unsafe {
            std::slice::from_raw_parts_mut(
                t.buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(t.offset as usize),
                t.n_bytes() as usize,
            )
            .fill(0x5a)
        }
    }
    struct Inputs {
        q: MetalTensor,
        w: MetalTensor,
        k: MetalTensor,
        start: usize,
        ratio: usize,
        rows: usize,
        width: usize,
    }
    fn load(ctx: &MetalContext, dir: &Path) -> Result<Inputs> {
        let j: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("capture.json"))?)?;
        let s = j["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| {
                s["layer"] == 2
                    && s["kernel"]
                        .as_str()
                        .unwrap()
                        .contains("index_scores_tensor")
            })
            .unwrap();
        let slot = |n| {
            s["slots"]
                .as_array()
                .unwrap()
                .iter()
                .find(|v| v["slot"] == n)
                .unwrap()
        };
        let constants = |n| -> Vec<u32> {
            let v: Vec<u8> = serde_json::from_value(slot(n)["constant"].clone()).unwrap();
            v.chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect()
        };
        let tensor = |n| -> Result<MetalTensor> {
            let v = slot(n);
            Ok(MetalTensor::from_bytes(
                ctx,
                &std::fs::read(dir.join(v["file"].as_str().unwrap()))?,
                serde_json::from_value(v["tensor"]["shape"].clone())?,
                GgmlType::F32,
            )?)
        };
        let a = constants(0);
        let c = constants(2);
        let q = tensor(3)?;
        Ok(Inputs {
            rows: q.n_elements() as usize / 4096,
            q,
            w: tensor(4)?,
            k: tensor(5)?,
            start: a[0] as usize,
            ratio: a[1] as usize,
            width: c[0] as usize,
        })
    }
    struct Scratch {
        scores: MetalTensor,
        pairs: MetalTensor,
        ids: MetalTensor,
        status: MetalTensor,
        dummy: MetalTensor,
    }
    fn scratch(ctx: &MetalContext, width: usize, rows: usize) -> Result<Scratch> {
        Ok(Scratch {
            scores: MetalTensor::zeros_f32(ctx, vec![(width * 512) as u64])?,
            pairs: MetalTensor::zeros_i32(ctx, vec![(512 * 4 * 512 * 2) as u64])?,
            ids: MetalTensor::zeros_i32(ctx, vec![512, rows as u64])?,
            status: MetalTensor::zeros_i32(ctx, vec![1])?,
            dummy: MetalTensor::zeros_i32(ctx, vec![1])?,
        })
    }
    fn baseline(
        ctx: &MetalContext,
        enc: &KernelEncoder,
        x: &Inputs,
        s: &Scratch,
        first: usize,
        n: usize,
    ) -> Result<()> {
        encode_v41_root_index_incumbent(
            ctx,
            enc,
            &V41RootIndexReuse {
                query: &x.q,
                weights: &x.w,
                keys: &x.k,
                scores: &s.scores,
                scratch: &s.pairs,
            },
            x.start,
            x.ratio,
            first,
            n,
            x.width,
        )?;
        Ok(())
    }
    fn select(
        ctx: &MetalContext,
        enc: &KernelEncoder,
        x: &Inputs,
        s: &Scratch,
        first: usize,
        n: usize,
    ) -> Result<()> {
        let length = (x.start + first + n) / x.ratio;
        let prep = ctx.pipeline("kernel_deepseek_v41_attention_selection_prepare_rows")?;
        enc.set_pipeline(&prep);
        enc.set_bytes(0, &[x.start as u32, x.ratio as u32, 512, 8]);
        enc.set_bytes(1, &[512u32, 1, n as u32, 0]);
        enc.set_tensor(2, &s.ids);
        enc.set_tensor(3, &s.dummy);
        enc.set_bytes(4, &(first as u32));
        enc.dispatch(
            MTLSize {
                width: length.min(512).div_ceil(256),
                height: n,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        if length <= 512 {
            return Ok(());
        }
        if length <= 4096 {
            let name = if length <= 2048 {
                "kernel_deepseek_v41_attention_select_fused_rows"
            } else {
                "kernel_deepseek_v41_attention_select_fused_rows_4k"
            };
            enc.set_pipeline(&ctx.pipeline(name)?);
            enc.set_bytes(0, &[x.start as u32, x.ratio as u32, 512, x.width as u32]);
            enc.set_bytes(1, &[first as u32, n as u32, 0, 0]);
            enc.set_tensor(2, &s.scores);
            enc.set_tensor(3, &s.ids);
            enc.set_tensor(4, &s.status);
            enc.dispatch(
                MTLSize {
                    width: 1,
                    height: n,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            return Ok(());
        }
        let parts = if length <= 8192 { 2 } else { 4 };
        enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_attention_select_hier_local")?);
        enc.set_bytes(
            0,
            &[x.start as u32, x.ratio as u32, x.width as u32, parts as u32],
        );
        enc.set_bytes(1, &[first as u32, n as u32, 0, 0]);
        enc.set_tensor(2, &s.scores);
        enc.set_tensor(3, &s.pairs);
        enc.set_tensor(4, &s.status);
        enc.dispatch(
            MTLSize {
                width: parts,
                height: n,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_attention_select_hier_merge")?);
        enc.set_tensor(2, &s.pairs);
        enc.set_tensor(3, &s.ids);
        enc.set_threadgroup_memory(0, parts * 512 * 4);
        enc.set_threadgroup_memory(1, parts * 512 * 4);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: n,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }
    fn chain(
        ctx: &MetalContext,
        x: &Inputs,
        s: &Scratch,
        candidate: bool,
        group: usize,
        selector: bool,
        reps: usize,
        capture: Option<&MetalTensor>,
    ) -> Result<(f64, f64)> {
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        let host = Instant::now();
        for _ in 0..reps {
            for first in (0..x.rows).step_by(512) {
                let n = 512.min(x.rows - first);
                if (x.start + first + n) / x.ratio > 512 {
                    if candidate {
                        ensure!(
                            try_encode_v41_root_index_reuse(
                                ctx,
                                &enc,
                                &V41RootIndexReuse {
                                    query: &x.q,
                                    weights: &x.w,
                                    keys: &x.k,
                                    scores: &s.scores,
                                    scratch: &s.pairs
                                },
                                x.start,
                                x.ratio,
                                first,
                                n,
                                x.width,
                                group
                            )?,
                            "unsupported candidate"
                        )
                    } else {
                        baseline(ctx, &enc, x, s, first, n)?
                    }
                }
                if selector {
                    select(ctx, &enc, x, s, first, n)?
                }
                if let Some(out) = capture {
                    let mut source = s.scores.view_subrange(0, vec![(x.width * n) as u64]);
                    source.dtype = GgmlType::I32;
                    let dest =
                        out.view_subrange((first * x.width) as u64, vec![(x.width * n) as u64]);
                    encode_copy_offset_i32(ctx, &enc, &source, 0, &dest, x.width * n)?;
                }
            }
        }
        enc.end();
        cmd.commit();
        wait_completed(&cmd)?;
        Ok((
            (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3 / reps as f64,
            host.elapsed().as_secs_f64() * 1e3 / reps as f64,
        ))
    }
    fn witness(
        ctx: &MetalContext,
        x: &Inputs,
        s: &Scratch,
        group: usize,
        label: &str,
    ) -> Result<()> {
        let a = MetalTensor::zeros_i32(ctx, vec![(x.rows * x.width) as u64])?;
        let b = MetalTensor::zeros_i32(ctx, a.shape.clone())?;
        fill(&s.scores);
        fill(&s.ids);
        chain(ctx, x, s, false, group, true, 1, Some(&a))?;
        let ids = ints(&s.ids).to_vec();
        let status = ints(&s.status).to_vec();
        fill(&s.scores);
        fill(&s.ids);
        chain(ctx, x, s, true, group, true, 1, Some(&b))?;
        let mismatch = ints(&a)
            .iter()
            .zip(ints(&b))
            .filter(|(a, b)| a != b)
            .count();
        let id_diff = ids.iter().zip(ints(&s.ids)).filter(|(a, b)| a != b).count();
        emit(
            json!({"kind":"witness","label":label,"group":group,"rows":x.rows,"score_words":x.rows*x.width,"score_mismatches":mismatch,"selection_mismatches":id_diff,"status":ints(&s.status),"bitwise":mismatch==0&&id_diff==0&&status==ints(&s.status)}),
        );
        ensure!(
            mismatch == 0
                && id_diff == 0
                && status == ints(&s.status)
                && status.iter().all(|v| *v == 0),
            "bitwise witness failed {label} group{group}"
        );
        for row in 0..x.rows {
            let count = ((x.start + row + 1) / x.ratio).min(512);
            let a = &ids[row * 512..row * 512 + count];
            ensure!(
                a.windows(2).all(|v| v[0] < v[1])
                    && a.iter()
                        .all(|v| (*v as usize) < (x.start + row + 1) / x.ratio),
                "ordered causal IDs"
            );
        }
        Ok(())
    }
    fn replay(ctx: &MetalContext, dir: &Path) -> Result<()> {
        let x = load(ctx, dir)?;
        ensure!(x.rows == 4096 && x.start == 28672, "wrong real fixture");
        let s = scratch(ctx, x.width, x.rows)?;
        let mut valid = Vec::new();
        for group in [2, 4, 8, 16] {
            match witness(ctx, &x, &s, group, "real32K") {
                Ok(()) => valid.push(group),
                Err(e) => {
                    emit(json!({"kind":"rejected_group","group":group,"error":e.to_string()}))
                }
            }
        }
        ensure!(!valid.is_empty(), "no bitwise group");
        for group in valid {
            for selector in [false, true] {
                for _ in 0..2 {
                    chain(ctx, &x, &s, false, group, selector, 1, None)?;
                    chain(ctx, &x, &s, true, group, selector, 1, None)?;
                }
                let mut ratios = vec![];
                for pair in 0..3 {
                    let mut times = [0.; 2];
                    for candidate in if pair == 1 {
                        [true, false]
                    } else {
                        [false, true]
                    } {
                        let (gpu, host) = chain(ctx, &x, &s, candidate, group, selector, 3, None)?;
                        times[candidate as usize] = gpu;
                        emit(
                            json!({"kind":"isolated_timing","group":group,"selector":selector,"pair":pair,"arm":if candidate{"B"}else{"A"},"gpu_ms":gpu,"host_ms":host}),
                        );
                    }
                    ratios.push(times[0] / times[1]);
                }
                let speed = ratios.iter().product::<f64>().powf(1. / 3.);
                emit(
                    json!({"kind":"isolated_gate","group":group,"selector":selector,"speedup":speed,"pair_speedups":ratios,"passed":speed>=1.25}),
                );
            }
        }
        Ok(())
    }
    fn padded(
        ctx: &MetalContext,
        t: &MetalTensor,
        offset: usize,
    ) -> Result<(MetalTensor, MetalTensor)> {
        let bytes_per = if t.dtype == GgmlType::F16 { 2 } else { 4 };
        let mut b = vec![0x5a; offset + t.n_bytes() as usize + 16];
        b[offset..offset + t.n_bytes() as usize].copy_from_slice(bytes(t));
        let base = MetalTensor::from_bytes(ctx, &b, vec![(b.len() / bytes_per) as u64], t.dtype)?;
        let view = base.view_subrange((offset / bytes_per) as u64, t.shape.clone());
        Ok((base, view))
    }
    fn boundaries(ctx: &MetalContext, dir: &Path, group: usize) -> Result<()> {
        let real = load(ctx, dir)?;
        let mut cases = 0;
        for offset in [4, 16] {
            for ratio in [1, 2] {
                for start in [
                    1022usize, 1023, 1024, 1025, 2046, 2047, 2048, 4094, 4095, 8190, 8191,
                ] {
                    for rows in [1, 3, group - 1, group, group + 1, 31, 32, 33] {
                        for ties in [false, true] {
                            let q = if ties {
                                MetalTensor::zeros_f32(ctx, vec![4096, rows as u64])?
                            } else {
                                real.q.view_subrange(0, vec![4096, rows as u64])
                            };
                            let w = real.w.view_subrange(0, vec![32, rows as u64]);
                            let keys = ((start + rows) / ratio).next_multiple_of(64).max(512);
                            let k = real.k.view_subrange(0, vec![(keys * 128) as u64]);
                            let mut guards = vec![];
                            let mut pad = |t: &MetalTensor| -> Result<MetalTensor> {
                                let (a, b) = padded(ctx, t, offset)?;
                                guards.push((a, b.offset as usize, b.n_bytes() as usize));
                                Ok(b)
                            };
                            let x = Inputs {
                                q: pad(&q)?,
                                w: pad(&w)?,
                                k: pad(&k)?,
                                start,
                                ratio,
                                rows,
                                width: keys,
                            };
                            let z = scratch(ctx, keys, rows)?;
                            let s = Scratch {
                                scores: pad(&z.scores)?,
                                pairs: pad(&z.pairs)?,
                                ids: pad(&z.ids)?,
                                status: pad(&z.status)?,
                                dummy: pad(&z.dummy)?,
                            };
                            witness(ctx, &x, &s, group, "boundary")?;
                            for (a, off, n) in guards {
                                ensure!(
                                    bytes(&a)[..off]
                                        .iter()
                                        .chain(bytes(&a)[off + n..].iter())
                                        .all(|v| *v == 0x5a),
                                    "guard overwritten"
                                );
                            }
                            cases += 1;
                        }
                    }
                }
            }
        }
        emit(json!({"kind":"boundary_summary","cases":cases,"passed":true}));
        Ok(())
    }
    fn same(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits())
    }
    fn cold(f: &Forward<'_>, s: &mut State, ids: &[u32]) -> Result<Vec<f32>> {
        ensure!(
            f.begin_cold_prefill_episode(s, ids.len())
                .map_err(anyhow::Error::msg)?
        );
        loop {
            let x = f
                .cold_prefill_episode_slice(s, ids, None, 4096)
                .map_err(anyhow::Error::msg)?;
            if x.complete {
                return Ok(x.logits);
            }
        }
    }
    fn native(ctx: &MetalContext, model: &str, corpus: &str) -> Result<()> {
        let g = Arc::new(GgufFile::open(model)?);
        let tok = NativeTokenizer::from_gguf(&g)?;
        let ids: Vec<u32> = tok
            .encode(&std::fs::read_to_string(corpus)?, false)?
            .into_iter()
            .map(|i| i as u32)
            .collect();
        ensure!(ids.len() > 82944);
        let plan = resident_storage_plan_with_target(
            &g,
            host_page_size_bytes()?,
            ctx.max_buffer_length(),
            Some(4_294_967_296),
        )?;
        let w = ResidentWeights::load_with_storage_plan(ctx, g, &plan)?;
        let f = Forward::from_resident(ctx, &w, w.token_map.clone(), w.raw_pad)
            .map_err(anyhow::Error::msg)?;
        for n in [32768usize, 4096, 8192] {
            for candidate in [false, true] {
                let mut s = f.new_state(98304)?;
                with_v41_root_index_reuse(candidate, || cold(&f, &mut s, &ids[..n]))?;
            }
            let mut ratios = vec![];
            for pair in 0..3 {
                let mut times = [0.; 2];
                let mut ref_logits: Option<Vec<f32>> = None;
                let mut ref_state = None;
                let mut ref_next: Option<Vec<f32>> = None;
                for candidate in if pair == 1 {
                    [true, false]
                } else {
                    [false, true]
                } {
                    let mut state = f.new_state(98304)?;
                    take_v41_root_index_reuse_calls();
                    let clock = Instant::now();
                    let logits =
                        with_v41_root_index_reuse(candidate, || cold(&f, &mut state, &ids[..n]))?;
                    let ms = clock.elapsed().as_secs_f64() * 1e3;
                    let calls = take_v41_root_index_reuse_calls();
                    let digest = f.capture_snapshot(&state)?.diagnostic_digest();
                    let next = f
                        .step(&mut state, ids[n], None)
                        .map_err(anyhow::Error::msg)?;
                    ensure!(
                        !candidate && calls == 0 || candidate && calls > 0,
                        "rollback/active calls mismatch"
                    );
                    if let (Some(a), Some(b), Some(c)) = (&ref_logits, &ref_state, &ref_next) {
                        ensure!(
                            same(a, &logits) && b == &digest && same(c, &next),
                            "native logits/state/decode differ"
                        );
                    } else {
                        ref_logits = Some(logits);
                        ref_state = Some(digest);
                        ref_next = Some(next);
                    }
                    times[candidate as usize] = ms;
                    emit(
                        json!({"kind":"native_timing","rows":n,"cell":"cold","pair":pair,"arm":if candidate{"B"}else{"A"},"wall_ms":ms,"reuse_calls":calls,"bitwise":true}),
                    );
                }
                ratios.push(times[0] / times[1]);
            }
            emit(
                json!({"kind":"native_gate","cell":"cold","rows":n,"speedup":ratios.iter().product::<f64>().powf(1./3.),"pair_speedups":ratios,"logits_state_decode_bitwise":true}),
            );
        }
        let mut root = f.new_state(98304)?;
        with_v41_root_index_reuse(false, || cold(&f, &mut root, &ids[..81920]))?;
        let prefix = f.capture_snapshot(&root)?;
        drop(root);
        for candidate in [false, true] {
            let mut s = f.new_state(98304)?;
            f.restore_snapshot(&mut s, &prefix)?;
            with_v41_root_index_reuse(candidate, || f.prefill(&mut s, &ids[81920..82944], None))?;
        }
        let mut ratios = vec![];
        for pair in 0..3 {
            let mut times = [0.; 2];
            let mut reference = None;
            for candidate in if pair == 1 {
                [true, false]
            } else {
                [false, true]
            } {
                let mut s = f.new_state(98304)?;
                f.restore_snapshot(&mut s, &prefix)?;
                take_v41_root_index_reuse_calls();
                let clock = Instant::now();
                let l = with_v41_root_index_reuse(candidate, || {
                    f.prefill(&mut s, &ids[81920..82944], None)
                })?;
                let ms = clock.elapsed().as_secs_f64() * 1e3;
                let calls = take_v41_root_index_reuse_calls();
                ensure!(calls == 0, "F32 append path must not use Half reuse");
                let d = f.capture_snapshot(&s)?.diagnostic_digest();
                if let Some((ref a, ref b)) = reference {
                    let a: &Vec<f32> = a;
                    ensure!(same(a, &l) && b == &d, "append state");
                } else {
                    reference = Some((l, d));
                }
                times[candidate as usize] = ms;
                emit(
                    json!({"kind":"native_timing","cell":"append80K","rows":1024,"pair":pair,"arm":if candidate{"B"}else{"A"},"wall_ms":ms,"reuse_calls":calls,"bitwise":true}),
                );
            }
            ratios.push(times[0] / times[1]);
        }
        emit(
            json!({"kind":"native_gate","cell":"append80K","speedup":ratios.iter().product::<f64>().powf(1./3.),"pair_speedups":ratios,"logits_state_bitwise":true,"path_active":false}),
        );
        Ok(())
    }
    pub fn run() -> Result<()> {
        let a: Vec<_> = std::env::args().collect();
        ensure!(
            a.len() >= 3,
            "replay|boundaries INPUT | native MODEL CORPUS"
        );
        let ctx = MetalContext::new()?;
        with_v41_root_index_reuse(true, || match a[1].as_str() {
            "replay" => replay(&ctx, Path::new(&a[2])),
            "boundaries" => boundaries(
                &ctx,
                Path::new(&a[2]),
                a.get(3).map_or(Ok(4), |v| v.parse())?,
            ),
            "native" => native(&ctx, &a[2], a.get(3).expect("corpus")),
            _ => anyhow::bail!("mode"),
        })
    }
}
