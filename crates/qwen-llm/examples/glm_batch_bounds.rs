//! Whole-round price probes. --features perf-ablations; MODEL [DEPTH] [FRAMES].
//! No candidate arithmetic: held writes preserve the captured trajectory;
//! private-buffer concurrency deliberately removes dependencies as a predictor.
#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires --features perf-ablations");
}
#[cfg(feature = "perf-ablations")]
mod probe {
    use objc2_foundation::NSString;
    use objc2_metal::*;
    use qwen_llm::{gguf::GgufFile, glm5_next_metal::*, metal::*};
    use serde_json::{Value, json};
    use std::{collections::BTreeMap, io::Write, time::Instant};
    const GROUPS: [&str; 16] = [
        "q6_kda_qkv",
        "q6_shared",
        "q6_other",
        "head",
        "q8_batch",
        "iq_gate_up",
        "iq_down",
        "mla_member_projections",
        "index_cache",
        "index_scores",
        "selection",
        "attention",
        "kda",
        "hc",
        "router",
        "pointwise",
    ];
    fn emit(v: Value) {
        println!("{v}");
        std::io::stdout().flush().unwrap();
    }
    fn thermal() {
        let s = std::process::Command::new("notifyutil")
            .args(["-g", "com.apple.system.thermalpressurelevel"])
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8(s.stdout)
                .unwrap()
                .split_whitespace()
                .last(),
            Some("0")
        );
    }
    fn digest(v: &[Vec<f32>]) -> String {
        let mut h = blake3::Hasher::new();
        for r in v {
            assert!(r.iter().all(|v| v.is_finite()));
            h.update(bytemuck::cast_slice(r));
        }
        h.finalize().to_hex().to_string()
    }
    fn gpu(ctx: &MetalContext, concurrent: bool, f: impl FnOnce(&KernelEncoder)) -> (f64, f64) {
        let cmd = ctx.queue.commandBuffer().unwrap();
        let start = Instant::now();
        let enc = if concurrent {
            KernelEncoder::begin_concurrent(&cmd)
        } else {
            KernelEncoder::begin(&cmd)
        };
        f(&enc);
        enc.end();
        let encode = start.elapsed().as_secs_f64() * 1e3;
        cmd.commit();
        wait_completed(&cmd).unwrap();
        ((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3, encode)
    }
    fn group(row: &Value, vocab: usize) -> usize {
        let k = row["kernel"].as_str().unwrap();
        if k.contains("mat_mat_q6_K_fewrow") {
            let args = row["bindings"]
                .as_array()
                .unwrap()
                .iter()
                .find(|b| b["slot"] == 0)
                .unwrap();
            let a = args["u32"].as_array().unwrap();
            let (n, m) = (
                a[0].as_u64().unwrap() as usize,
                a[1].as_u64().unwrap() as usize,
            );
            return if m == vocab {
                3
            } else if (n, m) == (4096, 8192) {
                0
            } else if [(4096, 2048), (2048, 4096)].contains(&(n, m)) {
                1
            } else {
                2
            };
        }
        if k.contains("mat_mat_q8_0_fewrow") {
            4
        } else if k.contains("expert_native_gate_up") {
            5
        } else if k.contains("expert_native_") {
            6
        } else if k.contains("mat_vec_q8") || k.contains("mat_vec_f32_f32_lcpp") {
            7
        } else if k.contains("indexer_append") || k.contains("scatter_offset_f32_to_f16") {
            8
        } else if k.contains("lightning_indexer") {
            9
        } else if k.contains("select_top_k") || k.contains("indexer_expand_selection") {
            10
        } else if k.contains("selected_attention") {
            11
        } else if k.contains("kda_decode") {
            12
        } else if k.contains("hc_") || k.contains("mhc4_") {
            13
        } else if k.contains("route_learned") || k.contains("mat_vec_f32_members") {
            14
        } else {
            15
        }
    }
    pub fn run() -> anyhow::Result<()> {
        let a: Vec<_> = std::env::args().collect();
        let depth = a
            .get(2)
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(8192usize);
        let frames = a.get(3).map(|s| s.parse()).transpose()?.unwrap_or(4usize);
        assert!(std::env::var_os("QWEN_METAL_LEASE_FD").is_some());
        assert!((1..=4).contains(&frames));
        thermal();
        let mut ctx = MetalContext::new()?;
        assert!(ctx.device_facts().gpu_families.apple10);
        let old = ctx.library.clone();
        let opts = MTLCompileOptions::new();
        opts.setLanguageVersion(MTLLanguageVersion::Version3_1);
        opts.setFastMathEnabled(true);
        ctx.library = ctx
            .device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!(
                    "../../../scripts/research/glm-batch-next/controls.metal"
                )),
                Some(&opts),
            )
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let read = ctx.pipeline("roof4_read")?;
        let copy = ctx.pipeline("roof4_substitute")?;
        ctx.library = old;
        let flush = MetalTensor::zeros_f32(&ctx, vec![268435456])?;
        let check = MetalTensor::zeros_f32(&ctx, vec![131072])?;
        gpu(&ctx, false, |e| {
            encode_fill_f32(&ctx, e, &flush, 1.).unwrap()
        });
        let flush_cache = || {
            gpu(&ctx, false, |e| {
                e.set_pipeline(&read);
                e.set_tensor(0, &flush);
                e.set_tensor(1, &check);
                e.dispatch(
                    MTLSize {
                        width: 16384,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
            })
        };
        let gg = GgufFile::open(&a[1])?;
        let w = Glm5NextWeights::load(&ctx, &gg)?;
        anyhow::ensure!(
            evaluate_metal_memory_admission(16u64 << 30, 512 << 20, ctx.memory_signals(), true)
                .is_admitted(),
            "probe storage admission"
        );
        let docs: Value = serde_json::from_slice(include_bytes!(
            "../../../scripts/reference/glm53/quality-v1.json"
        ))?;
        let docs: Vec<Vec<u32>> = docs["documents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| {
                d["tokens"]
                    .as_str()
                    .unwrap()
                    .split_whitespace()
                    .map(|s| s.parse().unwrap())
                    .collect()
            })
            .collect();
        let mut ss: Vec<_> = (0..4)
            .map(|i| {
                Glm5NextSession::with_prefill_rows(
                    &ctx,
                    &w,
                    depth + 512,
                    if i == 0 { 2048 } else { 4 },
                )
                .unwrap()
            })
            .collect();
        let mut batch = Glm5NextBatch::new(&ctx, &w, 4)?;
        let empty = ss[0].capture_snapshot()?;
        let mut banks = Vec::new();
        emit(
            json!({"kind":"header","depth":depth,"frames":frames,"width":4,"groups":GROUPS,"device":ctx.device_facts(),"commit":std::process::Command::new("git").args(["rev-parse","HEAD"]).output().map(|o|String::from_utf8_lossy(&o.stdout).trim().to_string()).ok()}),
        );
        for r in 0..4 {
            ss[0].restore_snapshot(&empty)?;
            ss[0].prefill_packed(
                &ctx,
                &docs[[0, 8, 16, 9][r]]
                    .iter()
                    .copied()
                    .cycle()
                    .take(depth + 17 * r)
                    .collect::<Vec<_>>(),
            )?;
            banks.push(ss[0].capture_snapshot()?);
            emit(json!({"kind":"prefix","member":r}));
        }
        drop(empty);
        for r in 0..4 {
            ss[r].restore_snapshot(&banks[r])?;
        }
        drop(banks);
        for p in 0..64 {
            let ts: Vec<_> = (0..4).map(|r| docs[[1, 10, 17, 11][r]][p]).collect();
            batch.step(&ctx, &mut ss.iter_mut().collect::<Vec<_>>(), &ts)?;
        }
        let entry: Vec<_> = ss.iter().map(|s| s.capture_snapshot().unwrap()).collect();
        let tokens = |p: usize| {
            (0..4)
                .map(|r| docs[[1, 10, 17, 11][r]][64 + p])
                .collect::<Vec<_>>()
        };
        let mut native_hash = None;
        for rep in 0..3 {
            for r in 0..4 {
                ss[r].restore_snapshot(&entry[r])?;
            }
            let mut h = blake3::Hasher::new();
            let mut times = Vec::new();
            for p in 0..16 {
                let t = Instant::now();
                let out = batch.step(&ctx, &mut ss.iter_mut().collect::<Vec<_>>(), &tokens(p))?;
                times.push(t.elapsed().as_secs_f64() * 1e3);
                h.update(digest(&out).as_bytes());
            }
            let h = h.finalize().to_hex().to_string();
            if let Some(v) = &native_hash {
                assert_eq!(v, &h);
            } else {
                native_hash = Some(h.clone());
            }
            emit(json!({"kind":"native","rep":rep,"wall_ms":times,"digest":h}));
        }
        for r in 0..4 {
            ss[r].restore_snapshot(&entry[r])?;
        }
        for phase in 0..frames {
            thermal();
            let guard = glm_decode_capture_begin();
            let out = batch.step(&ctx, &mut ss.iter_mut().collect::<Vec<_>>(), &tokens(phase))?;
            let graph = glm_decode_capture_finish(guard);
            let output_hash = blake3::hash(bytemuck::cast_slice(&out.concat()))
                .to_hex()
                .to_string();
            let state_hash = graph.mutable_hash();
            let meta = graph.metadata();
            anyhow::ensure!(
                meta["snapshot_error"].is_null(),
                "capture {:?}",
                meta["snapshot_error"]
            );
            let steps = meta["steps"].as_array().unwrap();
            let groups: Vec<_> = steps
                .iter()
                .map(|r| group(r, w.config.vocab_size as usize))
                .collect();
            let mut counts = BTreeMap::<String, usize>::new();
            for r in steps {
                *counts
                    .entry(r["kernel"].as_str().unwrap().into())
                    .or_default() += 1;
            }
            emit(
                json!({"kind":"capture","phase":phase,"calls":steps.len(),"snapshot_bytes":meta["snapshot_bytes"],"kernels":counts,"group_calls":(0..16).map(|g|groups.iter().filter(|x|**x==g).count()).collect::<Vec<_>>()}),
            );
            let mut access = BTreeMap::new();
            for kernel in graph.kernels() {
                let f = ctx
                    .library
                    .newFunctionWithName(&NSString::from_str(&kernel))
                    .unwrap();
                let mut reflection = None;
                let _p = unsafe {
                    ctx.device
                        .newComputePipelineStateWithFunction_options_reflection_error(
                            &f,
                            MTLPipelineOption::BindingInfo,
                            Some(&mut reflection),
                        )
                }
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let b = reflection.unwrap().bindings();
                access.insert(
                    kernel,
                    b.iter()
                        .filter(|b| b.r#type().0 == 0 && b.isUsed())
                        .map(|b| (b.index(), b.access().0 != 0))
                        .collect::<Vec<_>>(),
                );
            }
            emit(
                json!({"kind":"weight_bytes","phase":phase,"bounds":graph.weight_byte_bounds(&groups,&access)}),
            );
            let writes: BTreeMap<_, _> = access
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        v.iter()
                            .filter(|(_, w)| *w)
                            .map(|(slot, _)| *slot)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            let held = graph.prepare_substitutions(&ctx, &writes);
            emit(
                json!({"kind":"hold_storage","phase":phase,"bytes":held.metadata(),"group_bytes":(0..16).map(|g|held.group_bytes(&groups,g)).collect::<Vec<_>>()}),
            );
            let full = 0u64;
            for rep in 0..6 {
                let mut modes: Vec<_> = (0..16).map(|g| (GROUPS[g], 1u64 << g)).collect();
                modes.insert(0, ("reference", full));
                modes.push(("mla_all", (7..=11).map(|g| 1u64 << g).sum()));
                modes.push(("q6_all", 15));
                if rep % 2 != 0 {
                    modes.reverse();
                }
                for (name, mask) in modes {
                    graph.restore();
                    flush_cache();
                    let (t, cpu) = gpu(&ctx, false, |e| {
                        held.replay_hold_groups(e, &copy, &groups, mask)
                    });
                    if rep == 0 {
                        assert_eq!(
                            blake3::hash(bytemuck::cast_slice(&graph.output()))
                                .to_hex()
                                .as_str(),
                            output_hash
                        );
                        assert_eq!(
                            graph.mutable_hash(),
                            state_hash,
                            "held {name} changed state"
                        );
                    }
                    emit(
                        json!({"kind":"hold","phase":phase,"rep":rep,"group":name,"gpu_ms":t,"encode_ms":cpu,"bits_checked":rep==0}),
                    );
                }
            }
            for rep in 0..6 {
                for dag in if rep % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    graph.restore();
                    flush_cache();
                    let mut barriers = 0;
                    let (ms, cpu) = gpu(&ctx, dag, |e| {
                        if dag {
                            barriers = graph.replay_dag(e, &access)
                        } else {
                            graph.replay(e, None, 1)
                        }
                    });
                    let bits = blake3::hash(bytemuck::cast_slice(&graph.output()))
                        .to_hex()
                        .as_str()
                        == output_hash
                        && graph.mutable_hash() == state_hash;
                    assert!(bits, "causal predictor differs");
                    emit(
                        json!({"kind":"causal","phase":phase,"rep":rep,"dag":dag,"gpu_ms":ms,"encode_ms":cpu,"barriers":barriers,"bits":bits}),
                    );
                }
            }
            let wavefront = graph.wavefront(&access);
            emit(
                json!({"kind":"wavefront_metadata","phase":phase,"metadata":wavefront.metadata()}),
            );
            for rep in 0..6 {
                graph.restore();
                flush_cache();
                let (ms, cpu) = gpu(&ctx, true, |e| wavefront.replay(e));
                let bits = blake3::hash(bytemuck::cast_slice(&graph.output()))
                    .to_hex()
                    .as_str()
                    == output_hash
                    && graph.mutable_hash() == state_hash;
                assert!(bits, "reordered wavefront differs");
                emit(
                    json!({"kind":"wavefront","phase":phase,"rep":rep,"gpu_ms":ms,"encode_ms":cpu,"bits":bits}),
                );
            }
            let overlap_mask = (7..=11).map(|g| 1u64 << g).sum::<u64>() | (1u64 << 15);
            let member_waves = graph.wavefront_groups(&access, &groups, overlap_mask);
            emit(
                json!({"kind":"member_waves_metadata","phase":phase,"metadata":member_waves.metadata(),"groups":[7,8,9,10,11,15]}),
            );
            for rep in 0..6 {
                graph.restore();
                flush_cache();
                let (ms, cpu) = gpu(&ctx, true, |e| member_waves.replay(e));
                let bits = blake3::hash(bytemuck::cast_slice(&graph.output()))
                    .to_hex()
                    .as_str()
                    == output_hash
                    && graph.mutable_hash() == state_hash;
                assert!(bits, "member wavefront differs");
                emit(
                    json!({"kind":"member_waves","phase":phase,"rep":rep,"gpu_ms":ms,"encode_ms":cpu,"bits":bits}),
                );
            }
            let frozen = graph.freeze(&ctx, &access);
            emit(json!({"kind":"private_storage","phase":phase,"metadata":frozen.metadata()}));
            for rep in 0..6 {
                let mut order: Vec<_> = (0..17).collect();
                if rep % 2 != 0 {
                    order.reverse();
                }
                for g in order {
                    let mask = if g == 16 { 0 } else { 1u64 << g };
                    frozen.restore();
                    flush_cache();
                    let (ms, cpu) = gpu(&ctx, false, |e| {
                        frozen.replay_skip_groups(e, &groups, mask, false)
                    });
                    let checked = rep == 0;
                    if checked {
                        assert!(frozen.check_except_groups(&groups, mask));
                    }
                    emit(
                        json!({"kind":"private_skip","phase":phase,"rep":rep,"group":if g==16{"reference"}else{GROUPS[g]},"gpu_ms":ms,"encode_ms":cpu,"non_skipped_bits_checked":checked}),
                    );
                }
            }
            for rep in 0..6 {
                let mut modes = vec!["serial", "members", "global"];
                if rep % 2 != 0 {
                    modes.reverse();
                }
                for mode in modes {
                    frozen.restore();
                    flush_cache();
                    let (ms, cpu) = gpu(&ctx, mode != "serial", |e| {
                        frozen.replay(e, mode == "members")
                    });
                    assert!(frozen.check());
                    emit(
                        json!({"kind":"private_concurrency","phase":phase,"rep":rep,"mode":mode,"gpu_ms":ms,"encode_ms":cpu,"bits":true}),
                    );
                }
            }
        }
        for r in 0..4 {
            ss[r].restore_snapshot(&entry[r])?;
        }
        let mut h = blake3::Hasher::new();
        let mut times = Vec::new();
        for p in 0..16 {
            let t = Instant::now();
            let out = batch.step(&ctx, &mut ss.iter_mut().collect::<Vec<_>>(), &tokens(p))?;
            times.push(t.elapsed().as_secs_f64() * 1e3);
            h.update(digest(&out).as_bytes());
        }
        assert_eq!(h.finalize().to_hex().as_str(), native_hash.unwrap());
        emit(json!({"kind":"after","wall_ms":times,"bits":true}));
        thermal();
        emit(json!({"kind":"complete"}));
        Ok(())
    }
}
#[cfg(feature = "perf-ablations")]
fn main() -> anyhow::Result<()> {
    // This probe prices the serial control. Native causal waves have their
    // own on/off witness in glm_batch_waves.
    qwen_llm::glm5_next_metal::with_batch_waves(false, probe::run)
}
