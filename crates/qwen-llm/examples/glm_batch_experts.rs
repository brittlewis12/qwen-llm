//! Kill screen: F32-coefficient MMA over the actual per-layer expert union.
//! Build normally, then use canonical Metal lease; MODEL argument; no snapshots.
use objc2_metal::{MTLBuffer as _, MTLCommandBuffer as _, MTLCommandQueue as _};
use qwen_llm::{
    gguf::GgufFile,
    metal::*,
    tensor::{GgmlType, TensorDesc},
};
use std::{collections::BTreeSet, io::Write};
fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Native {
    row_bytes: u64,
    k: u32,
    m: u32,
    e: u32,
    top: u32,
    mode: u32,
    clamp: f32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Mma {
    k: u32,
    m: u32,
    b: u32,
    top: u32,
    mode: u32,
    clamp: f32,
}
fn tensor<T: bytemuck::Pod>(
    ctx: &MetalContext,
    v: &[T],
    shape: Vec<u64>,
    ty: GgmlType,
) -> MetalTensor {
    MetalTensor::from_bytes(ctx, bytemuck::cast_slice(v), shape, ty).unwrap()
}
fn read(t: &MetalTensor) -> Vec<f32> {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<f32>(),
            t.n_elements() as usize,
        )
    }
    .to_vec()
}
fn sample(g: &GgufFile, d: &TensorDesc, expert: usize) -> Vec<f32> {
    let mut sd = d.clone();
    let k = d.shape[0];
    let stride = d.n_bytes / d.shape[2];
    sd.shape = vec![k, 8];
    sd.n_bytes = stride / d.shape[1] * 8;
    qwen_llm::codec::dequant_to_f32(
        &sd,
        &g.try_slice(d).unwrap()
            [expert * stride as usize..expert * stride as usize + sd.n_bytes as usize],
    )
    .unwrap()
}
fn main() -> anyhow::Result<()> {
    assert!(std::env::var_os("QWEN_METAL_LEASE_FD").is_some());
    let ctx = MetalContext::new()?;
    let g = GgufFile::open(std::env::args().nth(1).expect("MODEL"))?;
    let gd = g
        .tensors
        .iter()
        .find(|t| {
            t.name.ends_with("ffn_gate_exps.weight")
                && t.dtype == GgmlType::IQ2_S
                && g.find(&t.name.replace("gate_exps", "down_exps"))
                    .is_some_and(|d| d.dtype == GgmlType::IQ3_S)
        })
        .unwrap();
    let ud = g.find(&gd.name.replace("gate_exps", "up_exps")).unwrap();
    let dd = g.find(&gd.name.replace("gate_exps", "down_exps")).unwrap();
    let (k, m, e) = (
        gd.shape[0] as usize,
        gd.shape[1] as usize,
        gd.shape[2] as usize,
    );
    assert_eq!((k, m, e), (4096, 2048, 288));
    assert!(
        evaluate_metal_memory_admission(
            gd.n_bytes + ud.n_bytes + dd.n_bytes,
            512 << 20,
            ctx.memory_signals(),
            true
        )
        .is_admitted()
    );
    let gate = MetalTensor::from_gguf_tensor(&ctx, gd, g.try_slice(gd)?)?;
    let up = MetalTensor::from_gguf_tensor(&ctx, ud, g.try_slice(ud)?)?;
    let down = MetalTensor::from_gguf_tensor(&ctx, dd, g.try_slice(dd)?)?;
    emit(
        serde_json::json!({"event":"header","gate":gd.name,"down":dd.name,"device":ctx.device_facts(),"note":"two shared experts per member, otherwise independent; full F32 coefficients"}),
    );
    for b in [2, 3, 4] {
        let xv: Vec<f32> = (0..k * b)
            .map(|i| ((i * 37 + 11) % 97) as f32 * 0.01 - 0.48)
            .collect();
        let dv: Vec<f32> = (0..m * 8 * b)
            .map(|i| ((i * 13 + 7) % 83) as f32 * 0.01 - 0.41)
            .collect();
        let x = tensor(&ctx, &xv, vec![k as u64, b as u64], GgmlType::F32);
        let di = tensor(&ctx, &dv, vec![m as u64, (8 * b) as u64], GgmlType::F32);
        let ys: Vec<_> = (0..2)
            .map(|_| MetalTensor::zeros_f32(&ctx, vec![k as u64, (8 * b) as u64]).unwrap())
            .collect();
        let gs: Vec<_> = (0..2)
            .map(|_| MetalTensor::zeros_f32(&ctx, vec![m as u64, (8 * b) as u64]).unwrap())
            .collect();
        let status = tensor(&ctx, &vec![1i32; b], vec![b as u64], GgmlType::I32);
        for rep in -2i32..10 {
            let iv: Vec<i32> = (0..b * 8)
                .map(|i| {
                    let (r, s) = (i / 8, i % 8);
                    ((s * 11 + if s < 2 { 0 } else { r * 37 } + (rep + 2) as usize * 23) % e) as i32
                })
                .collect();
            assert_eq!(iv[..8].iter().collect::<BTreeSet<_>>().len(), 8);
            let ids = tensor(&ctx, &iv, vec![8, b as u64], GgmlType::I32);
            for mode in [1u32, 2] {
                let (ni, no, ty, bank, input, outputs, desc) = if mode == 1 {
                    (k, m, GgmlType::IQ2_S, &gate, &x, &gs, gd)
                } else {
                    (m, k, GgmlType::IQ3_S, &down, &di, &ys, dd)
                };
                let mut expected = Vec::new();
                for selected in 0..8 * b {
                    let coef = sample(&g, desc, iv[selected] as usize);
                    let uc = if mode == 1 {
                        Some(sample(&g, ud, iv[selected] as usize))
                    } else {
                        None
                    };
                    let inputs = if mode == 1 {
                        &xv[(selected / 8) * ni..][..ni]
                    } else {
                        &dv[selected * ni..][..ni]
                    };
                    for row in 0..8 {
                        let dot = |c: &[f32]| {
                            (0..ni)
                                .map(|i| c[row * ni + i] as f64 * inputs[i] as f64)
                                .sum::<f64>()
                        };
                        let v = dot(&coef);
                        let v = if let Some(c) = &uc {
                            let g = v.min(7.);
                            let u = dot(c).clamp(-7., 7.);
                            g / (1. + (-g).exp()) * u
                        } else {
                            v
                        };
                        expected.push((selected * no + row, v));
                    }
                }
                let mut configs = vec![(0, 0)];
                for nt in [1, 2, 4] {
                    for nsg in [8, 16] {
                        configs.push((nt, nsg));
                    }
                }
                if rep % 2 != 0 {
                    configs.reverse();
                }
                for (nt, nsg) in configs {
                    let route = usize::from(nt != 0);
                    let cmd = ctx.queue.commandBuffer().unwrap();
                    let enc = KernelEncoder::begin(&cmd);
                    if nt == 0 {
                        let name = if mode == 1 {
                            "kernel_expert_native_gate_up_iq2_s_f32_k4096_r4"
                        } else {
                            "kernel_expert_native_iq3_s_f32_k2048_r4"
                        };
                        enc.set_pipeline(&ctx.pipeline(name)?);
                        let (block, bytes) = ty.storage_layout().unwrap();
                        enc.set_bytes(
                            0,
                            &Native {
                                row_bytes: ni as u64 / block * bytes,
                                k: ni as u32,
                                m: no as u32,
                                e: e as u32,
                                top: 8,
                                mode,
                                clamp: 7.,
                            },
                        );
                        for (slot, t) in [
                            (1, bank),
                            (2, input),
                            (3, &ids),
                            (4, &status),
                            (5, &outputs[route]),
                        ] {
                            enc.set_tensor(slot, t);
                        }
                        if mode == 1 {
                            enc.set_tensor(6, &up);
                        }
                        enc.dispatch(
                            objc2_metal::MTLSize {
                                width: no.div_ceil(if mode == 1 { 8 } else { 16 }),
                                height: 8,
                                depth: b,
                            },
                            objc2_metal::MTLSize {
                                width: 128,
                                height: 1,
                                depth: 1,
                            },
                        );
                    } else {
                        let name = format!(
                            "kernel_glm_expert_{}_mma_nt{nt}_nsg{nsg}",
                            if mode == 1 { "iq2_s" } else { "iq3_s" }
                        );
                        enc.set_pipeline(&ctx.pipeline(&name)?);
                        for pass in if mode == 1 { vec![0, 1] } else { vec![2] } {
                            enc.set_bytes(
                                0,
                                &Mma {
                                    k: ni as u32,
                                    m: no as u32,
                                    b: b as u32,
                                    top: 8,
                                    mode: pass,
                                    clamp: 7.,
                                },
                            );
                            for (slot, t) in [
                                (1, if pass == 1 { &up } else { bank }),
                                (2, input),
                                (3, &outputs[route]),
                                (4, &ids),
                                (5, &status),
                            ] {
                                enc.set_tensor(slot, t);
                            }
                            enc.dispatch(
                                objc2_metal::MTLSize {
                                    width: no.div_ceil(8 * nt),
                                    height: 8 * b,
                                    depth: 1,
                                },
                                objc2_metal::MTLSize {
                                    width: 32 * nsg,
                                    height: 1,
                                    depth: 1,
                                },
                            );
                        }
                    }
                    enc.end();
                    cmd.commit();
                    wait_completed(&cmd)?;
                    let us = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6;
                    let got = read(&outputs[route]);
                    assert!(got.iter().all(|v| v.is_finite()));
                    let err = expected
                        .iter()
                        .map(|(i, e)| (got[*i] as f64 - e).powi(2))
                        .sum::<f64>();
                    let energy = expected.iter().map(|(_, e)| e * e).sum::<f64>();
                    let rms = (err / energy).sqrt();
                    anyhow::ensure!(rms < 1e-3, "wrong expert mapping: {rms}");
                    emit(
                        serde_json::json!({"event":"timing","b":b,"mode":mode,"rep":rep,"nt":nt,"nsg":nsg,"gpu_us":us,"f64_relative_rms":rms,"unique_experts":iv.iter().collect::<BTreeSet<_>>().len()}),
                    );
                }
            }
        }
    }
    emit(serde_json::json!({"event":"complete"}));
    Ok(())
}
