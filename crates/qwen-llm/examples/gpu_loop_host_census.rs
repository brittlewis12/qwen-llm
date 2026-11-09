//! Ordinary decode host census: FAMILY MODEL DEPTH [STEPS=32].
//! Run under the canonical GPU lease. No sampled encoders or graph changes.
use anyhow::{Result, ensure};
use qwen_llm::{
    deepseek_v41::{
        forward::{Forward, PrefillExecution, profile},
        gguf::{ResidentWeights, resident_storage_plan_with_target},
    },
    gguf::GgufFile,
    glm5_next_metal::{Glm5NextSession, Glm5NextWeights},
    metal::{self, MetalContext},
    sampling::{Sampler, SamplingConfig},
    tokenizer::NativeTokenizer,
};
use serde_json::{Value, json};
use std::{
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
fn unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
fn rustc() -> String {
    std::process::Command::new("pgrep")
        .args(["-x", "rustc"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}
fn emit(v: Value) {
    println!("{v}");
}
fn corpus(g: &GgufFile, depth: usize) -> Result<Vec<u32>> {
    let tok = NativeTokenizer::from_gguf(g)?;
    let text = concat!(
        include_str!("../../../README.md"),
        include_str!("../src/sampling.rs")
    );
    let ids = tok.encode(text, false)?;
    ensure!(!ids.is_empty());
    Ok(ids.iter().cycle().take(depth).map(|&i| i as u32).collect())
}
#[derive(Default)]
struct Sum {
    wall: f64,
    gpu: f64,
    encode: f64,
    sample: f64,
    embedding: f64,
    engram: f64,
    readback: f64,
    envelope: f64,
}
impl Sum {
    fn report(&self, n: usize) -> Value {
        let q = n as f64;
        json!({"token_ms":self.wall/q,"gpu_ms":self.gpu/q,"outside_gpu_ms":(self.wall-self.gpu)/q,"encode_ms":self.encode/q,"sampling_ms":self.sample/q,"embedding_cpu_ms":self.embedding/q,"engram_cpu_ms":self.engram/q,"readback_ms":self.readback/q,"gpu_envelope_ms":self.envelope/q,"tps":q*1e3/self.wall})
    }
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let family = &a[1];
    let depth: usize = a[3].parse()?;
    let steps: usize = a.get(4).map(|x| x.parse()).transpose()?.unwrap_or(32);
    let g = Arc::new(GgufFile::open(&a[2])?);
    let ids = corpus(&g, depth)?;
    let ctx = MetalContext::new()?;
    emit(
        json!({"kind":"header","family":family,"depth":depth,"steps":steps,"device":ctx.device_facts()}),
    );
    if family == "v41" {
        let storage = resident_storage_plan_with_target(
            &g,
            metal::host_page_size_bytes()?,
            ctx.max_buffer_length(),
            Some(4_294_967_296),
        )?;
        let w = ResidentWeights::load_with_storage_plan(&ctx, g.clone(), &storage)?;
        let f = Forward::from_resident(&ctx, &w, w.token_map.clone(), w.raw_pad)
            .map_err(anyhow::Error::msg)?;
        let mut state = f
            .new_state(depth + steps + 64)
            .map_err(anyhow::Error::msg)?;
        f.prefill_with_execution(&mut state, &ids, None, PrefillExecution::ShrinkingCed)?;
        for &token in ids.iter().take(8) {
            f.step(&mut state, token, None)
                .map_err(anyhow::Error::msg)?;
        }
        let entry = f.capture_snapshot(&state).map_err(anyhow::Error::msg)?;
        for temperature in [0., 0.8] {
            for (arm, observed) in [false, true, true, false].into_iter().enumerate() {
                f.restore_snapshot(&mut state, &entry)
                    .map_err(anyhow::Error::msg)?;
                let mut sampler = Sampler::new(SamplingConfig {
                    temperature,
                    top_k: 20,
                    top_p: 0.95,
                    min_p: 0.02,
                    seed: 19,
                })?;
                let before = rustc();
                let started = unix();
                let mut token = ids[0];
                let mut sum = Sum::default();
                let mut last = Vec::new();
                for _ in 0..steps {
                    let scope = observed
                        .then(profile::Scope::begin)
                        .transpose()
                        .map_err(anyhow::Error::msg)?;
                    let wall = Instant::now();
                    let row = f
                        .step(&mut state, token, None)
                        .map_err(anyhow::Error::msg)?;
                    let sample = Instant::now();
                    token = sampler.sample(&row)?.token as u32;
                    sum.sample += sample.elapsed().as_secs_f64() * 1e3;
                    sum.wall += wall.elapsed().as_secs_f64() * 1e3;
                    if let Some(scope) = scope {
                        let p = scope.finish();
                        let split = p.operations.iter().any(|o| o.operation == "decode_submit");
                        for o in p.operations {
                            if o.operation == if split { "decode_submit" } else { "gpu_step" } {
                                sum.gpu += o.gpu_ms.unwrap_or(0.);
                                sum.encode += o.encode_ms;
                            }
                            match o.operation {
                                "decode_embedding" => sum.embedding += o.wall_ms,
                                "decode_engram_rows" => sum.engram += o.wall_ms,
                                "decode_readback" => sum.readback += o.wall_ms,
                                "gpu_step" => sum.envelope += o.gpu_ms.unwrap_or(0.),
                                _ => {}
                            }
                        }
                    }
                    last = row;
                }
                emit(
                    json!({"kind":"arm","family":family,"depth":depth,"temperature":temperature,"arm":arm,"observed":observed,"started_unix":started,"finished_unix":unix(),"rustc_before":before,"rustc_after":rustc(),"summary":sum.report(steps),"last_head_hash":blake3::hash(bytemuck::cast_slice(&last)).to_hex().as_str()}),
                );
            }
        }
    } else {
        ensure!(family == "glm");
        let w = Glm5NextWeights::load(&ctx, &g)?;
        let mut state = Glm5NextSession::with_prefill_rows(&ctx, &w, depth + steps + 64, 2048)?;
        state.prefill_packed(&ctx, &ids)?;
        for &token in ids.iter().take(8) {
            state.forward(&ctx, token)?;
        }
        let entry = state.capture_snapshot()?;
        for temperature in [0., 0.8] {
            for (arm, observed) in [false, true, true, false].into_iter().enumerate() {
                state.restore_snapshot(&entry)?;
                let mut sampler = Sampler::new(SamplingConfig {
                    temperature,
                    top_k: 20,
                    top_p: 0.95,
                    min_p: 0.02,
                    seed: 19,
                })?;
                let before = rustc();
                let started = unix();
                let mut token = ids[0];
                let mut sum = Sum::default();
                let mut last = Vec::new();
                for _ in 0..steps {
                    let wall = Instant::now();
                    let row = if observed {
                        let (row, t) = state.forward_timed(&ctx, token)?;
                        sum.gpu += t.gpu_ms;
                        sum.encode += t.encode_ms;
                        sum.readback += t.readback_ms;
                        row
                    } else {
                        state.forward(&ctx, token)?
                    };
                    let sample = Instant::now();
                    token = sampler.sample(&row)?.token as u32;
                    sum.sample += sample.elapsed().as_secs_f64() * 1e3;
                    sum.wall += wall.elapsed().as_secs_f64() * 1e3;
                    last = row;
                }
                emit(
                    json!({"kind":"arm","family":family,"depth":depth,"temperature":temperature,"arm":arm,"observed":observed,"started_unix":started,"finished_unix":unix(),"rustc_before":before,"rustc_after":rustc(),"summary":sum.report(steps),"last_head_hash":blake3::hash(bytemuck::cast_slice(&last)).to_hex().as_str()}),
                );
            }
        }
    }
    Ok(())
}
