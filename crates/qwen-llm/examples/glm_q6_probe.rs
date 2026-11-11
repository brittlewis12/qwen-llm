#![cfg_attr(not(feature = "perf-ablations"), allow(unused))]
//! Isolated GLM Q6 projection census and mapping screen on the real shapes.
//! MODEL [reps] [mode=census|candidates]
//! Requires `--features perf-ablations`; run under the canonical Metal lease.
//!
//! Every shape is timed over a rotating working set large enough to leave the
//! M5 system cache, so the GB/s figures describe the cold decode stream rather
//! than a cache-resident replay. Candidate rows reuse the bench-only research
//! kernels in `kernels/research/glm_batch_q6.metal` (B=1 instantiations).
use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _};
use qwen_llm::{
    gguf::GgufFile,
    metal::*,
    tensor::{GgmlType, TensorDesc},
};
use std::{io::Write, time::Instant};

/// Keep the rotating set comfortably above the M5 shared cache.
const TARGET_BYTES: u64 = 384 << 20;

fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}

fn thermal() -> u32 {
    std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .split_whitespace()
                .last()?
                .parse()
                .ok()
        })
        .unwrap_or(u32::MAX)
}

fn xorshift(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 0.2
        })
        .collect()
}

fn desc<'a>(g: &'a GgufFile, name: &str) -> &'a TensorDesc {
    g.tensors
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("missing tensor {name}"))
}

fn bytes<'a>(g: &'a GgufFile, t: &TensorDesc) -> &'a [u8] {
    let shard = &g.shards[t.shard_idx];
    let lo = t.data_offset as usize;
    &shard.mmap_bytes()[lo..lo + t.n_bytes as usize]
}

fn reps_for(n_bytes: u64) -> u64 {
    TARGET_BYTES.div_ceil(n_bytes).clamp(1, 256)
}

/// `reps` back-to-back copies of one real matrix.
fn bank(ctx: &MetalContext, g: &GgufFile, t: &TensorDesc) -> anyhow::Result<(MetalTensor, u64)> {
    let one = t.n_bytes as usize;
    let reps = reps_for(t.n_bytes);
    let src = bytes(g, t);
    let mut all = Vec::with_capacity(one * reps as usize);
    for _ in 0..reps {
        all.extend_from_slice(src);
    }
    let shape = vec![t.shape[0], t.shape[1] * reps];
    let big = MetalTensor::from_bytes(ctx, &all, shape, GgmlType::Q6_K)?;
    Ok((big, reps))
}

#[derive(Clone, Copy, Debug)]
struct Case {
    label: &'static str,
    k: usize,
    n: usize,
    n_bytes: u64,
    calls: usize,
}

#[derive(Clone, Copy)]
enum Kind {
    Needle,
    X3,
    SharedSwiglu,
    SplitK {
        nsg: usize,
    },
    PairSplitK {
        nsg: usize,
        swiglu: bool,
    },
    Research {
        name: &'static str,
        out: usize,
        nsg: usize,
    },
}

fn incumbent(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    n: usize,
    two_row: bool,
) -> anyhow::Result<()> {
    let _guard = diagnostic_q6k_r1(if two_row { 2 } else { 1 });
    encode_mat_vec_q6_k_f32(ctx, enc, w, x, y, k, n)?;
    Ok(())
}

fn research(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    name: &str,
    w: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    n: usize,
    out: usize,
    nsg: usize,
) -> anyhow::Result<()> {
    let pso = ctx.pipeline(name)?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_cols: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: k as u32,
            n_out: n as u32,
            n_cols: 1,
        },
    );
    enc.set_tensor(1, w);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.dispatch(
        objc2_metal::MTLSize {
            width: n.div_ceil(nsg * out),
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: (nsg * 32) as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn splitk(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    name: &str,
    w: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    n: usize,
    nsg: usize,
) -> anyhow::Result<()> {
    let pso = ctx.pipeline(name)?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        clamp: f32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: k as u32,
            n_out: n as u32,
            clamp: 10.0,
        },
    );
    enc.set_tensor(1, w);
    enc.set_tensor(2, x);
    enc.set_tensor(3, y);
    enc.dispatch(
        objc2_metal::MTLSize {
            width: n,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: nsg * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn pair_splitk(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    name: &str,
    gate: &MetalTensor,
    up: &MetalTensor,
    x: &MetalTensor,
    yg: &MetalTensor,
    yu: &MetalTensor,
    k: usize,
    n: usize,
    nsg: usize,
) -> anyhow::Result<()> {
    let pso = ctx.pipeline(name)?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        clamp: f32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: k as u32,
            n_out: n as u32,
            clamp: 10.0,
        },
    );
    enc.set_tensor(1, gate);
    enc.set_tensor(2, up);
    enc.set_tensor(3, x);
    enc.set_tensor(4, yg);
    enc.set_tensor(5, yu);
    enc.dispatch(
        objc2_metal::MTLSize {
            width: n,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: nsg * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Time one dispatch per command so successive calls cannot overlap (the real
/// B=1 token is one serial encoder), rotating `reps` cold replicas. Returns the
/// median per-call GPU milliseconds.
fn time<F>(ctx: &MetalContext, reps: u64, mut encode: F) -> anyhow::Result<f64>
where
    F: FnMut(&KernelEncoder, u64) -> anyhow::Result<()>,
{
    let mut one = |rep: u64| -> anyhow::Result<f64> {
        let cmd = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("no command buffer"))?;
        let enc = KernelEncoder::begin(&cmd);
        encode(&enc, rep)?;
        enc.end();
        cmd.commit();
        wait_completed(&cmd)?;
        anyhow::ensure!(cmd.error().is_none(), "command error: {:?}", cmd.error());
        Ok((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3)
    };
    for pass in 0..2 {
        for rep in 0..reps {
            one((rep + pass) % reps)?;
        }
    }
    let mut times: Vec<f64> = Vec::with_capacity(5 * reps as usize);
    for pass in 0..5 {
        for rep in 0..reps {
            times.push(one((rep + pass) % reps)?);
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(times[times.len() / 2])
}

fn main() -> anyhow::Result<()> {
    assert!(std::env::var_os("QWEN_METAL_LEASE_FD").is_some());
    let args: Vec<String> = std::env::args().collect();
    let g = GgufFile::open(&args[1])?;
    let _mode = args.get(2).map(String::as_str).unwrap_or("candidates");
    let ctx = MetalContext::new()?;
    emit(
        serde_json::json!({"kind":"header","thermal":thermal(),"device":ctx.device_facts(),"target_bytes":TARGET_BYTES}),
    );

    // The real B=1 Q6 call set, keyed by representative tensor and count.
    let cases = [
        Case {
            label: "KDA qkv (per matrix)",
            k: 4096,
            n: 8192,
            n_bytes: desc(&g, "blk.0.attn_q.weight").n_bytes,
            calls: 34,
        },
        Case {
            label: "KDA output",
            k: 8192,
            n: 4096,
            n_bytes: desc(&g, "blk.0.attn_output.weight").n_bytes,
            calls: 34,
        },
        Case {
            label: "MLA output",
            k: 16384,
            n: 4096,
            n_bytes: desc(&g, "blk.3.attn_output.weight").n_bytes,
            calls: 11,
        },
        Case {
            label: "MLA q_a",
            k: 4096,
            n: 1536,
            n_bytes: desc(&g, "blk.3.attn_q_a.weight").n_bytes,
            calls: 11,
        },
        Case {
            label: "shared gate+up (per matrix)",
            k: 4096,
            n: 2048,
            n_bytes: desc(&g, "blk.3.ffn_gate_shexp.weight").n_bytes,
            calls: 42,
        },
        Case {
            label: "shared down",
            k: 2048,
            n: 4096,
            n_bytes: desc(&g, "blk.3.ffn_down_shexp.weight").n_bytes,
            calls: 42,
        },
        Case {
            label: "dense gate+up (per matrix)",
            k: 4096,
            n: 12288,
            n_bytes: desc(&g, "blk.0.ffn_gate.weight").n_bytes,
            calls: 3,
        },
        Case {
            label: "dense down",
            k: 12288,
            n: 4096,
            n_bytes: desc(&g, "blk.0.ffn_down.weight").n_bytes,
            calls: 3,
        },
        Case {
            label: "head",
            k: 4096,
            n: 154880,
            n_bytes: desc(&g, "output.weight").n_bytes,
            calls: 1,
        },
    ];

    let variants = [
        ("incumbent_r1", Kind::Needle),
        ("incumbent_two_row", Kind::Needle),
        (
            "research_scaled_o1_s4",
            Kind::Research {
                name: "glm_batch_q6_scaled_b1_o1_s4",
                out: 1,
                nsg: 4,
            },
        ),
        (
            "research_scaled_o1_s2",
            Kind::Research {
                name: "glm_batch_q6_scaled_b1_o1_s2",
                out: 1,
                nsg: 2,
            },
        ),
        (
            "research_scaled_o2_s4",
            Kind::Research {
                name: "glm_batch_q6_scaled_b1_o2_s4",
                out: 2,
                nsg: 4,
            },
        ),
        (
            "research_scaled_o2_s2",
            Kind::Research {
                name: "glm_batch_q6_scaled_b1_o2_s2",
                out: 2,
                nsg: 2,
            },
        ),
        (
            "research_vector_o1_s4",
            Kind::Research {
                name: "glm_batch_q6_vector_b1_o1_s4",
                out: 1,
                nsg: 4,
            },
        ),
        (
            "research_vector_o2_s4",
            Kind::Research {
                name: "glm_batch_q6_vector_b1_o2_s4",
                out: 2,
                nsg: 4,
            },
        ),
        ("splitk_s2", Kind::SplitK { nsg: 2 }),
        ("splitk_s4", Kind::SplitK { nsg: 4 }),
        ("splitk_s8", Kind::SplitK { nsg: 8 }),
    ];

    for case in cases {
        let needle = match case.label {
            "KDA output" => "blk.0.attn_output.weight",
            "MLA output" => "blk.3.attn_output.weight",
            "MLA q_a" => "blk.3.attn_q_a.weight",
            "shared gate+up (per matrix)" => "blk.3.ffn_gate_shexp.weight",
            "shared down" => "blk.3.ffn_down_shexp.weight",
            "dense gate+up (per matrix)" => "blk.0.ffn_gate.weight",
            "dense down" => "blk.0.ffn_down.weight",
            "head" => "output.weight",
            _ => "blk.0.attn_q.weight",
        };
        let t = desc(&g, needle);
        let (w, reps) = bank(&ctx, &g, t)?;
        let x = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&xorshift(case.k, 1)),
            vec![case.k as u64],
            GgmlType::F32,
        )?;
        let y = MetalTensor::zeros_f32(&ctx, vec![case.n as u64])?;
        let y2 = MetalTensor::zeros_f32(&ctx, vec![case.n as u64])?;
        let y3 = MetalTensor::zeros_f32(&ctx, vec![case.n as u64])?;
        let y_swiglu = MetalTensor::zeros_f32(&ctx, vec![(case.n * 3) as u64])?;
        let per_call_bytes = case.n_bytes;
        let bytes_per_pass = per_call_bytes * reps;
        emit(serde_json::json!({
            "kind":"shape","label":case.label,"k":case.k,"n":case.n,
            "bytes":per_call_bytes,"calls_per_token":case.calls,
            "reps":reps,"pass_bytes":bytes_per_pass
        }));
        let mut variants = variants.to_vec();
        if case.label.starts_with("KDA qkv") {
            variants.push(("incumbent_x3_two_row", Kind::X3));
        }
        if case.label.starts_with("shared gate+up") {
            variants.push(("shared_swiglu_fused", Kind::SharedSwiglu));
        }
        if case.label.starts_with("shared gate+up") || case.label.starts_with("dense gate+up") {
            variants.push((
                "pair_splitk_s4",
                Kind::PairSplitK {
                    nsg: 4,
                    swiglu: false,
                },
            ));
            variants.push((
                "pair_swiglu_s2",
                Kind::PairSplitK {
                    nsg: 2,
                    swiglu: true,
                },
            ));
            variants.push((
                "pair_swiglu_s4",
                Kind::PairSplitK {
                    nsg: 4,
                    swiglu: true,
                },
            ));
            variants.push((
                "pair_swiglu_s8",
                Kind::PairSplitK {
                    nsg: 8,
                    swiglu: true,
                },
            ));
        }
        let mut base_ms = 0.0;
        for (name, kind) in variants {
            let started = Instant::now();
            let (ms, variant_bytes) = match kind {
                Kind::Needle => {
                    let two_row = name == "incumbent_two_row";
                    (
                        time(&ctx, reps, |enc, rep| {
                            let view = w.view_bytes(
                                rep * per_call_bytes,
                                vec![case.k as u64, case.n as u64],
                            );
                            incumbent(&ctx, enc, &view, &x, &y, case.k, case.n, two_row)
                        })?,
                        per_call_bytes,
                    )
                }
                Kind::X3 => (
                    time(&ctx, reps, |enc, rep| {
                        let views: Vec<_> = (0..3)
                            .map(|i| {
                                w.view_bytes(
                                    ((rep + i) % reps) * per_call_bytes,
                                    vec![case.k as u64, case.n as u64],
                                )
                            })
                            .collect();
                        encode_mat_vec_q6_k_x3_f32(
                            &ctx,
                            enc,
                            [&views[0], &views[1], &views[2]],
                            &x,
                            [&y, &y2, &y3],
                            case.k,
                            case.n,
                        )?;
                        Ok(())
                    })?,
                    3 * per_call_bytes,
                ),
                Kind::SharedSwiglu => (
                    time(&ctx, reps, |enc, rep| {
                        let gate =
                            w.view_bytes(rep * per_call_bytes, vec![case.k as u64, case.n as u64]);
                        let up = w.view_bytes(
                            ((rep + 1) % reps) * per_call_bytes,
                            vec![case.k as u64, case.n as u64],
                        );
                        encode_ds4_shared_swiglu_q6_k_f32(
                            &ctx, enc, &gate, &up, &x, &y_swiglu, case.k, case.n, 10.0,
                        )?;
                        Ok(())
                    })?,
                    2 * per_call_bytes,
                ),
                Kind::SplitK { nsg } => (
                    time(&ctx, reps, |enc, rep| {
                        let view =
                            w.view_bytes(rep * per_call_bytes, vec![case.k as u64, case.n as u64]);
                        let name = format!("kernel_glm_q6_splitk_s{nsg}");
                        splitk(&ctx, enc, &name, &view, &x, &y, case.k, case.n, nsg)
                    })?,
                    per_call_bytes,
                ),
                Kind::PairSplitK { nsg, swiglu } => (
                    time(&ctx, reps, |enc, rep| {
                        let gate =
                            w.view_bytes(rep * per_call_bytes, vec![case.k as u64, case.n as u64]);
                        let up = w.view_bytes(
                            ((rep + 1) % reps) * per_call_bytes,
                            vec![case.k as u64, case.n as u64],
                        );
                        let name = format!(
                            "kernel_glm_q6_pair_splitk_s{nsg}_swiglu{}",
                            if swiglu { 1 } else { 0 }
                        );
                        pair_splitk(
                            &ctx, enc, &name, &gate, &up, &x, &y, &y2, case.k, case.n, nsg,
                        )
                    })?,
                    2 * per_call_bytes,
                ),
                Kind::Research { name, out, nsg } => {
                    // Views are aligned to `out` rows so a group's rows stay in
                    // one replica.
                    let rows = case.n.div_ceil(out * nsg) * out * nsg;
                    anyhow::ensure!(rows == case.n, "case {name} n={} not tile aligned", case.n);
                    (
                        time(&ctx, reps, |enc, rep| {
                            let view = w.view_bytes(
                                rep * per_call_bytes,
                                vec![case.k as u64, case.n as u64],
                            );
                            research(&ctx, enc, name, &view, &x, &y, case.k, case.n, out, nsg)
                        })?,
                        per_call_bytes,
                    )
                }
            };
            if name == "incumbent_r1" {
                base_ms = ms;
            }
            emit(serde_json::json!({
                "kind":"variant","label":case.label,"name":name,"us":ms*1e3,
                "gbps":variant_bytes as f64/(ms*1e6),
                "vs_r1":if base_ms>0.0 {base_ms/ms}else{0.0},
                "per_token_ms":ms*case.calls as f64,
                "wall_ms":started.elapsed().as_secs_f64()*1e3,
                "thermal":thermal()
            }));
        }
        drop((w, x, y, y2, y3, y_swiglu));
    }
    Ok(())
}
