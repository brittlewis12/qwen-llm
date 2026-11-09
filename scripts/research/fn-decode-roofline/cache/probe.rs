//! MODEL METAL_SOURCE [mode: cache|model|all]. Commit and commands reproduce it.
use objc2_foundation::NSString;
use objc2_metal::*;
use qwen_llm::{
    gguf::GgufFile,
    metal::*,
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity},
};
use serde_json::json;
use std::{collections::BTreeMap, time::Instant};
fn emit(v: serde_json::Value) {
    println!("{v}");
}
fn sz(n: usize) -> MTLSize {
    MTLSize {
        width: n,
        height: 1,
        depth: 1,
    }
}
fn thermal() {
    let o = std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .unwrap();
    let s = String::from_utf8(o.stdout).unwrap();
    emit(json!({"kind":"thermal","value":s.trim()}));
    assert!(s.trim().ends_with(" 0"));
}
fn timed(ctx: &MetalContext, concurrent: bool, f: impl FnOnce(&KernelEncoder)) -> f64 {
    let c = ctx.queue.commandBuffer().unwrap();
    let e = if concurrent {
        KernelEncoder::begin_concurrent(&c)
    } else {
        KernelEncoder::begin(&c)
    };
    f(&e);
    e.end();
    c.commit();
    wait_completed(&c).unwrap();
    (c.GPUEndTime() - c.GPUStartTime()) * 1e6
}
fn barrier(e: &KernelEncoder) {
    e.buffer_barrier_probe();
}
fn ids(n: usize, v: u32, mut s: u64) -> Vec<u32> {
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % u64::from(v)) as u32
        })
        .collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let mode = a.get(3).map(String::as_str).unwrap_or("all");
    let mut ctx = MetalContext::new()?;
    let original = ctx.library.clone();
    let opts = MTLCompileOptions::new();
    opts.setLanguageVersion(MTLLanguageVersion::Version3_1);
    opts.setFastMathEnabled(true);
    ctx.library = ctx
        .device
        .newLibraryWithSource_options_error(
            &NSString::from_str(&std::fs::read_to_string(&a[2])?),
            Some(&opts),
        )
        .map_err(|e| e.to_string())?;
    let names = [
        "cache_touch",
        "cache_fill",
        "layer_gdn_fused",
        "layer_qsa_flash",
        "layer_qsa_flash_merge",
        "layer_ffn_epilogue",
        "layer_shared_down",
        "roof4_substitute",
        "roof4_noop",
    ];
    let mut pipes = BTreeMap::new();
    let mut access = BTreeMap::new();
    for name in names {
        pipes.insert(name.to_string(), ctx.pipeline(name)?);
        let f = ctx
            .library
            .newFunctionWithName(&NSString::from_str(name))
            .unwrap();
        let mut r = None;
        let _p = unsafe {
            ctx.device
                .newComputePipelineStateWithFunction_options_reflection_error(
                    &f,
                    MTLPipelineOption::BindingInfo,
                    Some(&mut r),
                )
        }
        .map_err(|e| e.to_string())?;
        access.insert(
            name.to_string(),
            r.unwrap()
                .bindings()
                .iter()
                .filter(|b| b.r#type().0 == 0 && b.isUsed())
                .map(|b| (b.index(), b.access().0 != 0))
                .collect::<Vec<_>>(),
        );
    }
    let f = ctx
        .library
        .newFunctionWithName(&NSString::from_str("cache_touch"))
        .unwrap();
    let encoder = unsafe { f.newArgumentEncoderWithBufferIndex(0) };
    ctx.library = original;
    let data = MetalTensor::zeros_i32(&ctx, vec![268435456])?;
    let flush_data = MetalTensor::zeros_i32(&ctx, vec![268435456])?;
    for t in [&data, &flush_data] {
        timed(&ctx, false, |e| {
            e.set_pipeline(&pipes["cache_fill"]);
            e.set_bytes(0, &268435456u32);
            e.set_tensor(1, t);
            e.dispatch(sz(1048576), sz(256));
        });
    }
    let flush = CachePacket::sweep(
        &ctx,
        &encoder,
        &pipes["cache_touch"],
        &flush_data.buffer,
        1 << 30,
        0,
    );
    let flush_cache = || {
        let t = timed(&ctx, false, |e| flush.encode(e));
        assert!(flush.check());
        t
    };
    if mode != "model" {
        thermal();
        let sizes = [
            1usize, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
        ];
        for rep in 0..4 {
            let mut order = sizes.to_vec();
            if rep % 2 != 0 {
                order.reverse();
            }
            for mib in order {
                let p = CachePacket::sweep(
                    &ctx,
                    &encoder,
                    &pipes["cache_touch"],
                    &data.buffer,
                    mib << 20,
                    1024,
                );
                let flush_us = flush_cache();
                let cold = timed(&ctx, false, |e| p.encode(e));
                assert!(p.check());
                let n = 16;
                let hot = timed(&ctx, false, |e| {
                    for _ in 0..n {
                        p.encode(e);
                    }
                }) / n as f64;
                assert!(p.check());
                emit(
                    json!({"kind":"working_set","rep":rep,"mib":mib,"traffic_bytes":p.traffic(),"cold_us":cold,"hot_us":hot,"hot_gb_s":p.traffic()as f64/hot/1000.,"flush_us":flush_us,"checksum":true}),
                );
            }
        }
        thermal();
    }
    if mode == "cache" {
        return Ok(());
    }
    let g = GgufFile::open(&a[1])?;
    let c = Qwen4ExpConfig::from_gguf(&g)?;
    let cap = Qwen4ExpSessionCapacity::for_forward_limit(&c, 8448)?;
    let mut model = Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &g, cap, 8448)?;
    let mut runner = model.create_runner(&ctx)?;
    runner.prefill(&ids(64, c.vocab_size, 1))?;
    runner.reset()?;
    runner.prefill(&ids(8192, c.vocab_size, 1 ^ 0x9e3779b97f4a7c15))?;
    for t in ids(64, c.vocab_size, 7) {
        runner.forward_token(t)?;
    }
    let entry = runner.capture_snapshot()?;
    let targets = ids(4, c.vocab_size, 13);
    for phase in [0usize, 3] {
        thermal();
        runner.restore_snapshot(&entry)?;
        for &t in &targets[..phase] {
            runner.forward_token(t)?;
        }
        let guard = decode_capture_begin();
        let reference = runner.forward_token(targets[phase])?.to_vec();
        let graph = decode_capture_finish(guard);
        let meta = graph.metadata();
        let rows = meta["steps"].as_array().unwrap();
        emit(
            json!({"kind":"capture_summary","phase":phase,"steps":rows.len(),"native_finite":reference.iter().all(|x|x.is_finite())}),
        );
        for kernel in graph.kernels() {
            if access.contains_key(&kernel) {
                continue;
            }
            let f = ctx
                .library
                .newFunctionWithName(&NSString::from_str(&kernel))
                .unwrap();
            let mut r = None;
            let _p = unsafe {
                ctx.device
                    .newComputePipelineStateWithFunction_options_reflection_error(
                        &f,
                        MTLPipelineOption::BindingInfo,
                        Some(&mut r),
                    )
            }
            .map_err(|e| e.to_string())?;
            access.insert(
                kernel,
                r.unwrap()
                    .bindings()
                    .iter()
                    .filter(|b| b.r#type().0 == 0 && b.isUsed())
                    .map(|b| (b.index(), b.access().0 != 0))
                    .collect::<Vec<_>>(),
            );
        }
        let ends: Vec<_> = rows
            .iter()
            .enumerate()
            .filter_map(|(i, s)| (s["kernel"] == "kernel_qwen4exp_hc_inject_f32").then_some(i))
            .collect();
        let bounds = |layer: usize| {
            let start = (ends[layer * 2 - 1] + 1..ends[layer * 2])
                .find(|&i| rows[i]["kernel"] == "kernel_qwen4exp_hc_rms_norm_f32")
                .unwrap();
            (start, ends[layer * 2 + 1] + 1)
        };
        for (current, next) in [(2usize, 3usize), (3, 4), (4, 5)] {
            thermal();
            let (start, end) = bounds(current);
            let (ns, ne) = bounds(next);
            let cur = graph.prepare_layer(&ctx, start, end, &access);
            let nxt = graph.prepare_layer(&ctx, ns, ne, &access);
            let cv = cur.variant(&ctx, &pipes, &access, 1, 64, 24);
            let nv = nxt.variant(&ctx, &pipes, &access, 1, 64, 24);
            emit(
                json!({"kind":"pair_metadata","phase":phase,"current":current,"next":next,"current_layer":cur.metadata(),"next_layer":nxt.metadata()}),
            );
            let prepare = || {
                graph.restore();
                timed(&ctx, false, |e| graph.replay(e, None, 1));
                graph.restore();
                timed(&ctx, false, |e| graph.replay_interval(e, 0, start));
                flush_cache()
            };
            for scope in ["hc", "early", "proj_shared", "all_known"] {
                let packet = nxt.cache_packets(&ctx, &encoder, &pipes["cache_touch"], scope, 0);
                let chunks =
                    nxt.cache_packets(&ctx, &encoder, &pipes["cache_touch"], scope, 4 << 20);
                emit(
                    json!({"kind":"prefetch_set","phase":phase,"current":current,"next":next,"scope":scope,"bytes":packet.iter().map(CachePacket::bytes).sum::<usize>(),"chunks":chunks.len()}),
                );
                let matvecs = nxt.cache_matvecs(&ctx, scope);
                // Immediate warm-cache consumer bound; pair timing below charges overlap.
                for rep in 0..4 {
                    for warm in if rep % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        prepare();
                        matvecs.restore();
                        if warm {
                            timed(&ctx, false, |e| {
                                for p in &packet {
                                    p.encode(e);
                                }
                            });
                        }
                        let us = timed(&ctx, false, |e| matvecs.replay(e));
                        assert!(matvecs.check());
                        assert!(packet.iter().all(CachePacket::check) || !warm);
                        emit(
                            json!({"kind":"matvec_cache","phase":phase,"current":current,"next":next,"scope":scope,"rep":rep,"warm":warm,"gpu_us":us,"outputs_exact":true}),
                        );
                    }
                }
                let placements = [
                    "none", "serial", "chain", "publish", "route", "tail", "spread",
                ];
                for rep in 0..6 {
                    let mut order = placements.to_vec();
                    if rep % 2 != 0 {
                        order.reverse();
                    }
                    for placement in order {
                        let flush_us = prepare();
                        let ps = if placement == "spread" {
                            &chunks
                        } else {
                            &packet
                        };
                        let us = timed(&ctx, true, |e| {
                            cv.prefetched_replay(e, ps, placement);
                            barrier(e);
                            nv.replay(e, true);
                        });
                        let check = nxt.check();
                        assert_eq!(check["boundary_mismatched_bytes"], 0);
                        assert!(cur.states_match());
                        if placement != "none" {
                            assert!(ps.iter().all(CachePacket::check));
                        }
                        emit(
                            json!({"kind":"prefetch_pair","phase":phase,"current":current,"next":next,"scope":scope,"rep":rep,"placement":placement,"gpu_us":us,"flush_us":flush_us,"outputs_exact":true}),
                        );
                    }
                }
                // Encoder-boundary timing is diagnostic; unsampled pair is the gate.
                for placement in ["none", "chain", "spread"] {
                    prepare();
                    let samples = ctx.timestamp_sample_buffer(4)?;
                    let command = ctx.queue.commandBuffer().unwrap();
                    let ps = if placement == "spread" {
                        &chunks
                    } else {
                        &packet
                    };
                    let e = KernelEncoder::begin_sampled(&command, &samples, 0, 1, true);
                    cv.prefetched_replay(&e, ps, placement);
                    e.end();
                    let e = KernelEncoder::begin_sampled(&command, &samples, 2, 3, true);
                    nv.replay(&e, true);
                    e.end();
                    command.commit();
                    wait_completed(&command)?;
                    let t = ctx.resolve_timestamp_samples(&samples, 4)?;
                    assert_eq!(nxt.check()["boundary_mismatched_bytes"], 0);
                    emit(
                        json!({"kind":"sampled_pair","phase":phase,"current":current,"next":next,"scope":scope,"placement":placement,"gpu_us":(command.GPUEndTime()-command.GPUStartTime())*1e6,"raw_ns":t}),
                    );
                }
            }
            if current == 3 {
                let plan = cv.cache_attribution(&ctx, &cur, &access);
                emit(
                    json!({"kind":"attribution_metadata","phase":phase,"metadata":plan.metadata()}),
                );
                for rep in 0..3 {
                    let masks: Vec<_> = (0usize..64).map(|i| (i * 37 + rep * 13) % 64).collect();
                    for copy_only in [false, true] {
                        for &mask in &masks {
                            prepare();
                            let us = timed(&ctx, true, |e| {
                                plan.replay(
                                    e,
                                    &pipes["roof4_substitute"],
                                    &pipes["roof4_noop"],
                                    mask as u64,
                                    copy_only,
                                    None,
                                )
                            });
                            if !copy_only {
                                assert_eq!(cur.check()["boundary_mismatched_bytes"], 0);
                            }
                            emit(
                                json!({"kind":"attribution_mask","phase":phase,"rep":rep,"copy_only":copy_only,"mask":mask,"gpu_us":us,"exact":!copy_only}),
                            );
                        }
                    }
                }
                for rep in 0..3 {
                    for omit in 0..plan.len() {
                        for copy_only in [false, true] {
                            prepare();
                            let us = timed(&ctx, true, |e| {
                                plan.replay(
                                    e,
                                    &pipes["roof4_substitute"],
                                    &pipes["roof4_noop"],
                                    63,
                                    copy_only,
                                    Some(omit),
                                )
                            });
                            if !copy_only {
                                assert_eq!(cur.check()["boundary_mismatched_bytes"], 0);
                            }
                            emit(
                                json!({"kind":"dispatch_marginal","phase":phase,"rep":rep,"omit":omit,"copy_only":copy_only,"gpu_us":us}),
                            );
                        }
                    }
                }
            }
            thermal();
        }
    }
    Ok(())
}
