//! Reproducible GLM byte ledger and Q6 microbatch screen.
//! Usage: cargo run --release -p qwen-llm --example glm_batch_q6 -- MODEL ledger|sweep
//! Sweep requires the canonical Metal lease; timing does not take the build lock.
use objc2_metal::{
    MTLBuffer as _, MTLCommandBuffer as _, MTLCommandQueue as _, MTLComputePipelineState as _,
};
use qwen_llm::{
    gguf::GgufFile,
    glm5_next::{Glm5NextConfig, MixerKind},
    metal::*,
    tensor::GgmlType,
};
use std::{collections::BTreeMap, io::Write};
fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    k: u32,
    m: u32,
    b: u32,
}
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let g = GgufFile::open(args.get(1).expect("MODEL ledger|sweep"))?;
    let c = Glm5NextConfig::from_gguf(&g)?;
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()?;
    emit(
        serde_json::json!({"event":"source","commit":String::from_utf8(commit.stdout)?.trim(),"argv":args}),
    );
    if args.get(2).map(String::as_str) == Some("ledger") {
        let mut categories: BTreeMap<String, u64> = BTreeMap::new();
        let mut shapes: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        for t in &g.tensors {
            if t.name.starts_with("blk.")
                && t.name.split('.').nth(1).unwrap().parse::<u32>()? >= c.executed_block_count()
            {
                continue;
            }
            if t.name == "token_embd.weight" {
                continue;
            }
            let category = if t.name.contains("_exps.weight") {
                "routed"
            } else {
                "always"
            };
            let bytes = if category == "routed" {
                t.n_bytes / u64::from(c.expert_count) * u64::from(c.expert_used_count)
            } else {
                t.n_bytes
            };
            *categories
                .entry(format!("{category}/{:?}", t.dtype))
                .or_default() += bytes;
            if t.dtype == GgmlType::Q6_K && t.shape.len() == 2 {
                let s = shapes
                    .entry(format!("{}x{}", t.shape[0], t.shape[1]))
                    .or_default();
                s.0 += 1;
                s.1 += t.n_bytes;
            }
            emit(
                serde_json::json!({"event":"tensor","name":t.name,"dtype":format!("{:?}",t.dtype),"shape":t.shape,"bytes":t.n_bytes,"active_bytes":bytes,"category":category}),
            );
        }
        let mla = c.block_count(MixerKind::Mla) as u64;
        let kda = c.block_count(MixerKind::Kda) as u64;
        let always: u64 = categories
            .iter()
            .filter(|(k, _)| k.starts_with("always/"))
            .map(|(_, v)| v)
            .sum();
        let routed: u64 = categories
            .iter()
            .filter(|(k, _)| k.starts_with("routed/"))
            .map(|(_, v)| v)
            .sum();
        emit(
            serde_json::json!({"event":"ledger","categories":categories,"q6_shapes":shapes,"always_bytes":always,"routed_bytes_per_member":routed,"trunk_blocks":c.executed_block_count(),"mla_blocks":mla,"kda_blocks":kda,"experts":c.expert_count,"top_k":c.expert_used_count,"vocab":c.vocab_size,"latent_rank":c.kv_lora_rank,"selected_latents":c.indexer_top_k,"pool_stride":c.indexer_pool}),
        );
        for b in [2u64, 3, 4] {
            let weights = always + b * routed;
            for depth in [8192u64, 32768, 65536] {
                // A useful-byte lower bound, not hardware DRAM traffic: distinct
                // selected F16 latents, all completed F16 index keys, KDA state R+W.
                let latent =
                    mla * depth.min(u64::from(c.indexer_top_k)) * u64::from(c.kv_lora_rank) * 2;
                let index =
                    mla * (depth / u64::from(c.indexer_pool)) * u64::from(c.indexer_head_dim) * 2;
                let recurrent =
                    kda * u64::from(c.head_count) * u64::from(c.kda_head_dim).pow(2) * 4 * 2;
                emit(
                    serde_json::json!({"event":"floor","b":b,"depth":depth,"shared_plus_routed_bytes":weights,"latent_bytes_per_member":latent,"index_bytes_per_member":index,"kda_rw_bytes_per_member":recurrent,"useful_bytes":weights+b*(latent+index+recurrent),"floor_ms_at_1TBps":(weights+b*(latent+index+recurrent))as f64/1e9}),
                );
            }
        }
        return Ok(());
    }
    anyhow::ensure!(
        args.get(2).map(String::as_str) == Some("sweep")
            || args.get(2).map(String::as_str) == Some("routers"),
        "ledger|sweep|routers required"
    );
    anyhow::ensure!(
        std::env::var_os("QWEN_METAL_LEASE_FD").is_some(),
        "run via with-metal-lease"
    );
    let ctx = MetalContext::new()?;
    emit(serde_json::json!({"event":"device","device":ctx.device_facts()}));
    if args.get(2).map(String::as_str) == Some("routers") {
        return routers(&ctx, &g);
    }
    let mut cases: BTreeMap<(usize, usize), Vec<_>> = BTreeMap::new();
    for t in &g.tensors {
        if t.dtype == GgmlType::Q6_K
            && t.shape.len() == 2
            && t.name != "token_embd.weight"
            && t.shape[0] >= 1024
            && t.shape[1] >= 1024
            && !t.name.starts_with("blk.45.")
        {
            cases
                .entry((t.shape[0] as usize, t.shape[1] as usize))
                .or_default()
                .push(t);
        }
    }
    for ((k, m), ds) in cases {
        let count = ds.len();
        let ds: Vec<_> = ds.into_iter().take(8).collect();
        let bytes = ds.iter().map(|t| t.n_bytes).sum();
        anyhow::ensure!(
            evaluate_metal_memory_admission(bytes, 512 << 20, ctx.memory_signals(), true)
                .is_admitted(),
            "normal admission"
        );
        let ws: Vec<_> = ds
            .iter()
            .map(|d| MetalTensor::from_gguf_tensor(&ctx, d, g.try_slice(d).unwrap()).unwrap())
            .collect();
        let coeff: Vec<_> = ds
            .iter()
            .map(|d| {
                let mut sd = (**d).clone();
                sd.shape = vec![k as u64, 8];
                sd.n_bytes = k as u64 / 256 * 210 * 8;
                qwen_llm::codec::dequant_to_f32(
                    &sd,
                    &g.try_slice(d).unwrap()[..sd.n_bytes as usize],
                )
                .unwrap()
            })
            .collect();
        for b in [2, 3, 4] {
            let x: Vec<_> = (0..k * b)
                .map(|i| ((i * 37 + 11) % 97) as f32 * 0.01 - 0.48)
                .collect();
            let xt = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&x),
                vec![k as u64, b as u64],
                GgmlType::F32,
            )?;
            let y = MetalTensor::zeros_f32(&ctx, vec![m as u64, b as u64])?;
            let expected: Vec<Vec<f64>> = coeff
                .iter()
                .map(|c| {
                    (0..b)
                        .flat_map(|t| (0..8).map(move |r| (t, r)))
                        .map(|(t, r)| {
                            (0..k)
                                .map(|j| c[r * k + j] as f64 * x[t * k + j] as f64)
                                .sum()
                        })
                        .collect()
                })
                .collect();
            let mut configs = Vec::new();
            for nt in [1, 2, 4] {
                for sg in [8, 16, 32] {
                    configs.push((
                        format!("kernel_mat_mat_q6_K_fewrow_gen_f32w_nt{nt}_rt1_nsg{sg}_f32"),
                        8 * nt,
                        sg,
                    ));
                }
            }
            for out in [1, 2, 4] {
                for sg in [2, 4] {
                    configs.push((
                        format!("glm_batch_q6_vector_b{b}_o{out}_s{sg}"),
                        out * sg,
                        sg,
                    ));
                }
            }
            emit(
                serde_json::json!({"event":"case","k":k,"m":m,"b":b,"matrices":count,"rotating_matrices":ds.len(),"bytes_per_matrix":ds[0].n_bytes}),
            );
            for out in [1, 2, 4] {
                for sg in [2, 4] {
                    configs.push((
                        format!("glm_batch_q6_scaled_b{b}_o{out}_s{sg}"),
                        out * sg,
                        sg,
                    ));
                }
            }
            for rep in -2i32..10 {
                let mut order = configs.clone();
                order.rotate_left((rep + 2) as usize % configs.len());
                if rep % 2 != 0 {
                    order.reverse();
                }
                for (index, (name, rows, sg)) in order.into_iter().enumerate() {
                    let wi = ((rep + 2) as usize * configs.len() + index) % ws.len();
                    let pso = ctx.pipeline(&name)?;
                    if pso.maxTotalThreadsPerThreadgroup() < 32 * sg {
                        emit(serde_json::json!({"event":"skip","kernel":name}));
                        continue;
                    }
                    let cmd = ctx.queue.commandBuffer().unwrap();
                    let enc = KernelEncoder::begin(&cmd);
                    enc.set_pipeline(&pso);
                    enc.set_bytes(
                        0,
                        &Args {
                            k: k as u32,
                            m: m as u32,
                            b: b as u32,
                        },
                    );
                    enc.set_tensor(1, &ws[wi]);
                    enc.set_tensor(2, &xt);
                    enc.set_tensor(3, &y);
                    enc.dispatch(
                        objc2_metal::MTLSize {
                            width: m.div_ceil(rows),
                            height: 1,
                            depth: 1,
                        },
                        objc2_metal::MTLSize {
                            width: 32 * sg,
                            height: 1,
                            depth: 1,
                        },
                    );
                    enc.end();
                    cmd.commit();
                    wait_completed(&cmd)?;
                    let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6;
                    let vals = unsafe {
                        std::slice::from_raw_parts(
                            y.buffer.contents().as_ptr().cast::<f32>(),
                            m * b,
                        )
                    };
                    anyhow::ensure!(vals.iter().all(|v| v.is_finite()), "finite {name}");
                    let mut error = 0.;
                    let mut energy = 0.;
                    for t in 0..b {
                        for row in 0..8 {
                            let e = expected[wi][t * 8 + row];
                            error += (vals[t * m + row] as f64 - e).powi(2);
                            energy += e * e;
                        }
                    }
                    emit(
                        serde_json::json!({"event":"timing","k":k,"m":m,"b":b,"kernel":name,"weight":ds[wi].name,"rep":rep,"gpu_us":gpu,"GBps":ds[wi].n_bytes as f64/(gpu*1e3),"f64_relative_rms":(error/energy).sqrt()}),
                    );
                }
            }
        }
    }
    emit(serde_json::json!({"event":"complete"}));
    Ok(())
}

fn routers(ctx: &MetalContext, g: &GgufFile) -> anyhow::Result<()> {
    let ds: Vec<_> = g
        .tensors
        .iter()
        .filter(|t| t.name.ends_with("ffn_gate_inp.weight") && t.dtype == GgmlType::F32)
        .take(8)
        .collect();
    let ws: Vec<_> = ds
        .iter()
        .map(|d| MetalTensor::from_gguf_tensor(ctx, d, g.try_slice(d).unwrap()).unwrap())
        .collect();
    let k = 4096usize;
    let m = 288usize;
    for b in [2, 3, 4] {
        let x: Vec<_> = (0..k * b)
            .map(|i| ((i * 37 + 11) % 97) as f32 * 0.01 - 0.48)
            .collect();
        let xt = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&x),
            vec![k as u64, b as u64],
            GgmlType::F32,
        )?;
        let y = MetalTensor::zeros_f32(ctx, vec![m as u64, b as u64])?;
        for rep in -8i32..32 {
            let wi = (rep + 8) as usize % ws.len();
            let mut reference = None;
            for route in if rep % 2 == 0 { [0, 1] } else { [1, 0] } {
                let cmd = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&cmd);
                if route == 0 {
                    for row in 0..b {
                        encode_mat_vec_f32(
                            ctx,
                            &enc,
                            &ws[wi],
                            &xt.view_subrange((row * k) as u64, vec![k as u64]),
                            &y.view_subrange((row * m) as u64, vec![m as u64]),
                            k,
                            m,
                        )?;
                    }
                } else {
                    enc.set_pipeline(&ctx.pipeline(&format!("glm_batch_router_b{b}"))?);
                    enc.set_bytes(
                        0,
                        &Args {
                            k: k as u32,
                            m: m as u32,
                            b: b as u32,
                        },
                    );
                    enc.set_tensor(1, &ws[wi]);
                    enc.set_tensor(2, &xt);
                    enc.set_tensor(3, &y);
                    enc.dispatch(
                        objc2_metal::MTLSize {
                            width: m.div_ceil(2),
                            height: 1,
                            depth: 1,
                        },
                        objc2_metal::MTLSize {
                            width: 128,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                enc.end();
                cmd.commit();
                wait_completed(&cmd)?;
                let vals = unsafe {
                    std::slice::from_raw_parts(y.buffer.contents().as_ptr().cast::<f32>(), m * b)
                }
                .to_vec();
                anyhow::ensure!(vals.iter().all(|v| v.is_finite()), "router finite");
                if let Some(ref old) = reference {
                    let old: &Vec<f32> = old;
                    anyhow::ensure!(
                        old.iter()
                            .zip(&vals)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "router arithmetic differs"
                    );
                } else {
                    reference = Some(vals);
                }
                emit(
                    serde_json::json!({"event":"router","b":b,"rep":rep,"route":route,"gpu_us":(cmd.GPUEndTime()-cmd.GPUStartTime())*1e6,"bitwise":true}),
                );
            }
        }
    }
    emit(serde_json::json!({"event":"complete"}));
    Ok(())
}
