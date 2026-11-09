//! Research predictor. MODEL CONTROLS_METAL [DEPTH]. Requires capture.patch.
use objc2_foundation::NSString;
use objc2_metal::*;
use qwen_llm::{
    gguf::GgufFile,
    metal::{
        KernelEncoder, MetalContext, MetalTensor, decode_capture_begin, decode_capture_finish,
        encode_fill_f32, wait_completed,
    },
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity},
};
use serde_json::json;
use std::{collections::BTreeMap, time::Instant};
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
fn sz(n: usize) -> MTLSize {
    MTLSize {
        width: n,
        height: 1,
        depth: 1,
    }
}
fn gpu(ctx: &MetalContext, concurrent: bool, f: impl FnOnce(&KernelEncoder)) -> f64 {
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = if concurrent {
        KernelEncoder::begin_concurrent(&cmd)
    } else {
        KernelEncoder::begin(&cmd)
    };
    f(&enc);
    enc.end();
    cmd.commit();
    wait_completed(&cmd).unwrap();
    (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3
}
fn hash(v: &[f32]) -> String {
    assert!(v.iter().all(|x| x.is_finite()));
    blake3::hash(bytemuck::cast_slice(v)).to_hex().to_string()
}
fn emit(v: serde_json::Value) {
    println!("{v}");
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let depth: usize = a.get(3).map(|s| s.parse()).transpose()?.unwrap_or(8192);
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
    let read = ctx.pipeline("roof4_read")?;
    let noop = ctx.pipeline("roof4_noop")?;
    ctx.library = original;
    let flush = MetalTensor::zeros_f32(&ctx, vec![268435456])?;
    let output = MetalTensor::zeros_f32(&ctx, vec![1048576])?;
    gpu(&ctx, false, |e| {
        encode_fill_f32(&ctx, e, &flush, 1.).unwrap()
    });
    let flush_cache = || {
        gpu(&ctx, false, |e| {
            e.set_pipeline(&read);
            e.set_tensor(0, &flush);
            e.set_tensor(1, &output);
            e.dispatch(sz(16384), sz(256));
        })
    };
    let g = GgufFile::open(&a[1])?;
    let c = Qwen4ExpConfig::from_gguf(&g)?;
    let cap = Qwen4ExpSessionCapacity::for_forward_limit(&c, depth + 256)?;
    let mut model = Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &g, cap, depth + 256)?;
    let mut r = model.create_runner(&ctx)?;
    r.prefill(&ids(64, c.vocab_size, 1))?;
    r.reset()?;
    r.prefill(&ids(depth, c.vocab_size, 1 ^ 0x9e3779b97f4a7c15))?;
    for t in ids(64, c.vocab_size, 7) {
        r.forward_token(t)?;
    }
    let entry = r.capture_snapshot()?;
    let targets = ids(32, c.vocab_size, 13);
    let mut first = Vec::new();
    let mut baseline = String::new();
    for rep in 0..3 {
        r.restore_snapshot(&entry)?;
        let mut h = blake3::Hasher::new();
        let mut samples = Vec::new();
        for &t in &targets {
            let start = Instant::now();
            let v = r.forward_token(t)?.to_vec();
            let wall = start.elapsed().as_secs_f64() * 1e3;
            if first.is_empty() {
                first = v.clone();
            }
            assert!(v.iter().all(|x| x.is_finite()));
            h.update(bytemuck::cast_slice(&v));
            let p = r.last_token_timing().unwrap();
            samples.push(json!({"wall_ms":wall,"gpu_ms":p.gpu_ms,"encode_ms":p.encode_cpu_ms,"wait_ms":p.completion_wait_ms}));
        }
        let digest = h.finalize().to_hex().to_string();
        if rep == 0 {
            baseline = digest.clone();
        } else {
            assert_eq!(baseline, digest);
        }
        emit(json!({"kind":"native","rep":rep,"depth":depth,"samples":samples,"hash":digest}));
    }
    r.restore_snapshot(&entry)?;
    let guard = decode_capture_begin();
    let v = r.forward_token(targets[0])?.to_vec();
    let graph = decode_capture_finish(guard);
    assert_eq!(hash(&v), hash(&first));
    emit(json!({"kind":"capture","metadata":graph.metadata(),"bitwise":true}));
    let calls = graph.metadata()["steps"].as_array().unwrap().len();
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
        .map_err(|e| e.to_string())?;
        let bindings = reflection.unwrap().bindings();
        access.insert(
            kernel,
            bindings
                .iter()
                .filter(|b| b.r#type().0 == 0 && b.isUsed())
                .map(|b| (b.index(), b.access().0 != 0))
                .collect::<Vec<_>>(),
        );
    }
    emit(json!({"kind":"access","bindings":access}));
    for rep in 0..8 {
        for mode in if rep % 2 == 0 {
            ["serial", "dag"]
        } else {
            ["dag", "serial"]
        } {
            graph.restore();
            flush_cache();
            let mut barriers = 0;
            let ms = gpu(&ctx, mode == "dag", |e| {
                if mode == "dag" {
                    barriers = graph.replay_dag(e, &access);
                } else {
                    graph.replay(e, None, 1);
                }
            });
            assert_eq!(hash(&graph.output()), hash(&first));
            emit(
                json!({"kind":"replay","rep":rep,"mode":mode,"gpu_ms":ms,"barriers":barriers,"bitwise":true}),
            );
        }
    }
    for rep in 0..8 {
        for n in if rep % 2 == 0 {
            vec![48, 96, 192, 384, 768, calls]
        } else {
            vec![calls, 768, 384, 192, 96, 48]
        } {
            graph.restore();
            gpu(&ctx, false, |e| graph.replay(e, None, 1));
            let before = unsafe { *output.buffer.contents().as_ptr().cast::<u32>() };
            let ms = gpu(&ctx, false, |e| {
                e.set_pipeline(&noop);
                e.set_tensor(0, &output);
                e.set_bytes(1, &[1u32, 1]);
                for _ in 0..n {
                    e.dispatch(sz(1), sz(32));
                }
            });
            let after = unsafe { *output.buffer.contents().as_ptr().cast::<u32>() };
            assert_eq!(after.wrapping_sub(before), n as u32);
            emit(
                json!({"kind":"launch_floor","rep":rep,"calls":n,"gpu_ms":ms,"counter_check":true}),
            );
        }
    }
    for rep in 0..9 {
        let ms = flush_cache();
        let checksum = unsafe {
            std::slice::from_raw_parts(output.buffer.contents().as_ptr().cast::<f32>(), 131072)
        };
        assert!(checksum.iter().all(|&v| v == 2048.));
        emit(
            json!({"kind":"bandwidth","rep":rep,"gpu_ms":ms,"gb_s":1073741824f64/(ms*1e6),"checksum":true}),
        );
    }
    let frozen = graph.freeze(&ctx, &access);
    emit(json!({"kind":"frozen_metadata","metadata":frozen.metadata()}));
    for rep in 0..8 {
        let mut modes = vec!["serial", "segments", "all"];
        if rep % 2 != 0 {
            modes.reverse();
        }
        for mode in modes {
            frozen.restore();
            flush_cache();
            let ms = gpu(&ctx, mode != "serial", |e| {
                frozen.replay(e, mode == "segments")
            });
            assert!(frozen.check(), "frozen outputs differ in {mode}");
            emit(
                json!({"kind":"frozen","rep":rep,"mode":mode,"gpu_ms":ms,"all_outputs_bitwise":true}),
            );
        }
    }
    r.restore_snapshot(&entry)?;
    let mut h = blake3::Hasher::new();
    let mut samples = Vec::new();
    for &t in &targets {
        let start = Instant::now();
        let v = r.forward_token(t)?.to_vec();
        let wall = start.elapsed().as_secs_f64() * 1e3;
        h.update(bytemuck::cast_slice(&v));
        let p = r.last_token_timing().unwrap();
        samples.push(json!({"wall_ms":wall,"gpu_ms":p.gpu_ms,"encode_ms":p.encode_cpu_ms}));
    }
    assert_eq!(h.finalize().to_hex().to_string(), baseline);
    emit(json!({"kind":"after","samples":samples,"hash":baseline}));
    Ok(())
}
