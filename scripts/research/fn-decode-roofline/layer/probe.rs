//! MODEL FUSED_METAL [MODE: screen|confirm]. 8K native causal layer probe.
use objc2_foundation::NSString;
use objc2_metal::*;
use qwen_llm::{
    gguf::GgufFile,
    metal::*,
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity},
};
use serde_json::json;
use std::collections::BTreeMap;
fn sz(x: usize, y: usize) -> MTLSize {
    MTLSize {
        width: x,
        height: y,
        depth: 1,
    }
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
fn emit(v: serde_json::Value) {
    println!("{v}");
}
fn thermal() {
    let v = std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .unwrap();
    let text = String::from_utf8(v.stdout).unwrap();
    emit(json!({"kind":"thermal","value":text.trim()}));
    assert!(
        text.trim().ends_with(" 0"),
        "thermal pressure during layer packet"
    );
}
fn timed(ctx: &MetalContext, concurrent: bool, f: impl FnOnce(&KernelEncoder)) -> (f64, f64) {
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = if concurrent {
        KernelEncoder::begin_concurrent(&cmd)
    } else {
        KernelEncoder::begin(&cmd)
    };
    let t = std::time::Instant::now();
    f(&enc);
    enc.end();
    cmd.commit();
    wait_completed(&cmd).unwrap();
    (
        (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6,
        t.elapsed().as_secs_f64() * 1e6,
    )
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let mode = args.get(3).map(String::as_str).unwrap_or("screen");
    let mut ctx = MetalContext::new()?;
    let original = ctx.library.clone();
    let opts = MTLCompileOptions::new();
    opts.setLanguageVersion(MTLLanguageVersion::Version3_1);
    opts.setFastMathEnabled(true);
    let source = std::fs::read_to_string(&args[2])?;
    ctx.library = ctx
        .device
        .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&opts))
        .map_err(|e| e.to_string())?;
    let mut pipes = BTreeMap::new();
    let mut access = BTreeMap::new();
    for name in [
        "layer_gdn_fused",
        "layer_qsa_flash",
        "layer_qsa_flash_merge",
        "layer_ffn_epilogue",
        "layer_shared_down",
    ] {
        pipes.insert(name.to_string(), ctx.pipeline(name)?);
    }
    // Keep a standalone streaming flush in the same source packet.
    let read = ctx.pipeline("roof4_read")?;
    for name in pipes.keys() {
        let f = ctx
            .library
            .newFunctionWithName(&NSString::from_str(name))
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
        .map_err(|e| e.to_string())?;
        access.insert(
            name.clone(),
            reflection
                .unwrap()
                .bindings()
                .iter()
                .filter(|b| b.r#type().0 == 0 && b.isUsed())
                .map(|b| (b.index(), b.access().0 != 0))
                .collect::<Vec<_>>(),
        );
    }
    ctx.library = original;
    let flush = MetalTensor::zeros_f32(&ctx, vec![268435456])?;
    let out = MetalTensor::zeros_f32(&ctx, vec![131072])?;
    timed(&ctx, false, |e| {
        encode_fill_f32(&ctx, e, &flush, 1.).unwrap()
    });
    let flush_cache = || {
        timed(&ctx, false, |e| {
            e.set_pipeline(&read);
            e.set_tensor(0, &flush);
            e.set_tensor(1, &out);
            e.dispatch(sz(16384, 1), sz(256, 1));
        })
        .0
    };
    let g = GgufFile::open(&args[1])?;
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
    let targets = ids(32, c.vocab_size, 13);
    for phase in if mode == "screen" {
        vec![0usize]
    } else {
        vec![0usize, 3]
    } {
        runner.restore_snapshot(&entry)?;
        for &t in &targets[..phase] {
            runner.forward_token(t)?;
        }
        let guard = decode_capture_begin();
        let reference = runner.forward_token(targets[phase])?.to_vec();
        let graph = decode_capture_finish(guard);
        let meta = graph.metadata();
        emit(
            json!({"kind":"capture","phase":phase,"metadata":meta,"native_hash":blake3::hash(bytemuck::cast_slice(&reference)).to_hex().to_string()}),
        );
        let greedy = ctx.pipeline("kernel_argmax_f32_greedy")?;
        let expected = reference.iter().enumerate().fold(0usize, |best, (i, x)| {
            if x.total_cmp(&reference[best]).is_gt() {
                i
            } else {
                best
            }
        }) as i32;
        for rep in 0..8 {
            graph.restore();
            // Restore full logits after the entry snapshot reset.
            timed(&ctx, false, |e| graph.replay(e, None, 1));
            let (us, _) = timed(&ctx, false, |e| {
                for _ in 0..32 {
                    graph.feedback_probe(e, &greedy);
                }
            });
            assert_eq!(graph.feedback_id(), expected);
            emit(
                json!({"kind":"gpu_feedback","phase":phase,"rep":rep,"us_per_argmax_and_embedding":us/32.,"token":expected,"token_check":true}),
            );
        }
        for kernel in graph.kernels() {
            if access.contains_key(&kernel) {
                continue;
            }
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
            .map_err(|e| e.to_string())?;
            access.insert(
                kernel,
                reflection
                    .unwrap()
                    .bindings()
                    .iter()
                    .filter(|b| b.r#type().0 == 0 && b.isUsed())
                    .map(|b| (b.index(), b.access().0 != 0))
                    .collect::<Vec<_>>(),
            );
        }
        let rows = meta["steps"].as_array().unwrap();
        let ends: Vec<_> = rows
            .iter()
            .enumerate()
            .filter_map(|(i, s)| (s["kernel"] == "kernel_qwen4exp_hc_inject_f32").then_some(i))
            .collect();
        for (layer, kind, gate) in [(2usize, "gdn", 140.), (3, "qsa", 180.)] {
            thermal();
            let start = (ends[layer * 2 - 1] + 1..ends[layer * 2])
                .find(|&i| rows[i]["kernel"] == "kernel_qwen4exp_hc_rms_norm_f32")
                .unwrap();
            let end = ends[layer * 2 + 1] + 1;
            let slice = graph.prepare_layer(&ctx, start, end, &access);
            let oracle = slice.operator_oracle(&ctx);
            emit(
                json!({"kind":"layer","phase":phase,"layer":layer,"type":kind,"gate_us":gate,"metadata":slice.metadata()}),
            );
            let configs: Vec<(String, usize, usize, u32)> = if kind == "gdn" {
                vec![
                    ("common".into(), 1, 64, 24),
                    ("gdn_p1".into(), 1, 64, 25),
                    ("gdn_p2".into(), 2, 64, 25),
                    ("gdn_p4".into(), 4, 64, 25),
                    ("gdn_p8".into(), 8, 64, 25),
                    ("shared_p2".into(), 2, 64, 29),
                ]
            } else {
                vec![
                    ("common".into(), 1, 64, 24),
                    ("flash_s16".into(), 1, 16, 26),
                    ("flash_s32".into(), 1, 32, 26),
                    ("flash_s64".into(), 1, 64, 26),
                    ("flash_s128".into(), 1, 128, 26),
                    ("shared_s64".into(), 1, 64, 30),
                ]
            };
            let mut variants = Vec::new();
            variants.push((
                "incumbent_dag".to_string(),
                slice.variant(&ctx, &pipes, &access, 1, 64, 0),
                0u32,
                1usize,
                64usize,
            ));
            for (name, parts, splits, mask) in configs {
                let variant = slice.variant(&ctx, &pipes, &access, parts, splits, mask);
                emit(
                    json!({"kind":"plan","phase":phase,"layer":layer,"variant":name,"mask":mask,"parts":parts,"splits":splits,"metadata":variant.metadata()}),
                );
                variants.push((name, variant, mask, parts, splits));
            }
            let reps = if mode == "screen" { 6 } else { 12 };
            for rep in 0..reps {
                let mut order: Vec<_> = (0..=variants.len()).collect();
                if rep % 2 != 0 {
                    order.reverse();
                }
                for cell in order {
                    graph.restore();
                    let warm_us = timed(&ctx, false, |e| graph.replay(e, None, 1)).0;
                    slice.restore();
                    if cell > 0 {
                        variants[cell - 1].1.restore();
                    }
                    // Prime before the cold-cache stream. No repeated hot-layer timing.
                    let flush_us = flush_cache();
                    let (name, concurrent) = if cell == 0 {
                        ("incumbent_serial", false)
                    } else {
                        (variants[cell - 1].0.as_str(), true)
                    };
                    let (us, wall_us) = timed(&ctx, concurrent, |e| {
                        if cell == 0 {
                            slice.replay(e)
                        } else {
                            variants[cell - 1].1.replay(e, true)
                        }
                    });
                    let check = slice.check();
                    assert!(check["finite"].as_bool().unwrap());
                    // The MoE status is READY=1, with no error bits. QSA's
                    // earlier selector status uses zero for success.
                    let statuses = check["status_codes"].as_array().unwrap();
                    assert!(!statuses.is_empty());
                    assert!(
                        statuses
                            .iter()
                            .enumerate()
                            .all(|(i, v)| v == if i + 1 == statuses.len() { 1 } else { 0 }),
                        "GPU status: {check}"
                    );
                    if cell <= 1 {
                        assert_eq!(check["boundary_mismatched_bytes"], 0);
                    }
                    if rep == 0 {
                        emit(
                            json!({"kind":"f64_operator","phase":phase,"layer":layer,"variant":name,"metrics":oracle.evaluate()}),
                        );
                    }
                    emit(
                        json!({"kind":"timing","phase":phase,"layer":layer,"type":kind,"rep":rep,"variant":name,"gpu_us":us,"wall_us":wall_us,"flush_us":flush_us,"warm_us":warm_us,"check":check}),
                    );
                }
            }
            thermal();
        }
    }
    Ok(())
}
