//! Routed-expert decode GEMV attribution probe (research; opt-in).
//!
//! Times the production native gate/up/down dispatches at the DeepSeek V4.1
//! decode shape against the `kernel_xprobe_*` research variants that change
//! only the activation access pattern and rows per simdgroup. Expert IDs
//! rotate every iteration so selected rows stream from DRAM, not the SLC.
//! Run with `QWEN_EXPERT_DECODE_PROBE=1` under the Metal lease.

use super::*;

const K: usize = 5120;
const M: usize = 2304;
const TOP_K: usize = 6;
const BANK_EXPERTS: usize = 64;
const ITERS: usize = 12;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ProbeArgs {
    row_bytes: u64,
    n_in: u32,
    n_out: u32,
    experts: u32,
    top_k: u32,
    mode: u32,
    clamp: f32,
}

struct Shape {
    n_in: usize,
    n_out: usize,
}

fn row_bytes(dtype: GgmlType, n_in: usize) -> u64 {
    let (block, bytes) = dtype.storage_layout().expect("bank layout");
    (n_in as u64) / block * bytes
}

fn bank_tensor(
    ctx: &MetalContext,
    dtype: GgmlType,
    k: usize,
    m: usize,
    seed: usize,
) -> MetalTensor {
    let bytes = bank_bytes_for_experts(dtype, k, m, seed, BANK_EXPERTS);
    MetalTensor::from_bytes(
        ctx,
        &bytes,
        vec![k as u64, m as u64, BANK_EXPERTS as u64],
        dtype,
    )
    .expect("bank")
}

fn f32_tensor(ctx: &MetalContext, values: &[f32], shape: &[u64]) -> MetalTensor {
    MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(values),
        shape.to_vec(),
        GgmlType::F32,
    )
    .expect("f32 tensor")
}

fn write_ids(ids: &MetalTensor, iteration: usize) {
    let base = (iteration * TOP_K) % BANK_EXPERTS;
    let p = ids.buffer.contents().as_ptr() as *mut i32;
    for j in 0..TOP_K {
        unsafe { p.add(j).write(((base + j) % BANK_EXPERTS) as i32) }
    }
}

fn timed<F>(ctx: &MetalContext, encode: F) -> f64
where
    F: FnOnce(&KernelEncoder) -> Result<(), MetalError>,
{
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd);
    encode(&enc).expect("encode");
    enc.end();
    cmd.commit();
    wait_completed(&cmd).expect("completed");
    (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

struct Fixture {
    gate: MetalTensor,
    up: MetalTensor,
    down: MetalTensor,
    x: MetalTensor,
    ids: MetalTensor,
    status: MetalTensor,
    inner: MetalTensor,
    out: MetalTensor,
}

impl Fixture {
    fn new(ctx: &MetalContext, gate_up: GgmlType, down: GgmlType) -> Self {
        let x = values(K, 3);
        Self {
            gate: bank_tensor(ctx, gate_up, K, M, 1),
            up: bank_tensor(ctx, gate_up, K, M, 2),
            down: bank_tensor(ctx, down, M, K, 3),
            x: f32_tensor(ctx, &x, &[K as u64]),
            ids: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&[0i32; TOP_K]),
                vec![TOP_K as u64],
                GgmlType::I32,
            )
            .expect("ids"),
            status: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&[ROUTE_STATUS_READY]),
                vec![1],
                GgmlType::I32,
            )
            .expect("status"),
            inner: MetalTensor::zeros_f32(ctx, vec![M as u64, TOP_K as u64]).expect("inner"),
            out: MetalTensor::zeros_f32(ctx, vec![K as u64, TOP_K as u64]).expect("out"),
        }
    }

    /// Touch every expert's rows in all three banks from the GPU before any
    /// timing, so fresh-allocation page mapping is not attributed to a kernel.
    fn warm(&self, ctx: &MetalContext) {
        for it in 0..(BANK_EXPERTS / TOP_K + 1) {
            write_ids(&self.ids, it);
            timed(ctx, |enc| {
                encode_all_slots_gate_up_swiglu_native(
                    ctx,
                    enc,
                    &self.gate,
                    &self.up,
                    &self.x,
                    &self.ids,
                    &self.status,
                    &self.inner,
                    K,
                    M,
                    BANK_EXPERTS,
                    TOP_K,
                    10.0,
                )?;
                encode_all_slots_down_native(
                    ctx,
                    enc,
                    &self.down,
                    &self.inner,
                    &self.ids,
                    &self.status,
                    &self.out,
                    M,
                    K,
                    BANK_EXPERTS,
                    TOP_K,
                )
            });
        }
    }

    /// Sustained GPU work so clocks are at their steady state before timing.
    fn heat(&self, ctx: &MetalContext) {
        let start = std::time::Instant::now();
        let mut it = 0;
        while start.elapsed() < std::time::Duration::from_millis(400) {
            write_ids(&self.ids, it);
            it += 1;
            timed(ctx, |enc| {
                encode_all_slots_gate_up_swiglu_native(
                    ctx,
                    enc,
                    &self.gate,
                    &self.up,
                    &self.x,
                    &self.ids,
                    &self.status,
                    &self.inner,
                    K,
                    M,
                    BANK_EXPERTS,
                    TOP_K,
                    10.0,
                )?;
                encode_all_slots_down_native(
                    ctx,
                    enc,
                    &self.down,
                    &self.inner,
                    &self.ids,
                    &self.status,
                    &self.out,
                    M,
                    K,
                    BANK_EXPERTS,
                    TOP_K,
                )
            });
        }
    }

    fn stage_bytes(&self) -> f64 {
        (TOP_K as f64)
            * ((M as f64) * row_bytes(self.gate.dtype, K) as f64 * 2.0
                + (K as f64) * row_bytes(self.down.dtype, M) as f64)
    }
}

fn probe_dispatch(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    name: &str,
    rows: usize,
    bank: &MetalTensor,
    input: &MetalTensor,
    f: &Fixture,
    output: &MetalTensor,
    shape: Shape,
    mode: u32,
) -> Result<(), MetalError> {
    let pso = ctx.pipeline(name)?;
    let args = ProbeArgs {
        row_bytes: row_bytes(bank.dtype, shape.n_in),
        n_in: shape.n_in as u32,
        n_out: shape.n_out as u32,
        experts: BANK_EXPERTS as u32,
        top_k: TOP_K as u32,
        mode,
        clamp: 10.0,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, &f.ids);
    enc.set_tensor(4, &f.status);
    enc.set_tensor(5, output);
    enc.dispatch(
        MTLSize {
            width: shape.n_out.div_ceil(4 * rows),
            height: TOP_K,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn report(label: &str, f: &Fixture, stage_us: Vec<f64>, gate_us: Vec<f64>) {
    let stage_min = stage_us.iter().cloned().fold(f64::INFINITY, f64::min);
    let stage = median(stage_us);
    let gate = median(gate_us);
    let gbps = f.stage_bytes() / (stage * 1e-6) / 1e9;
    eprintln!(
        "xprobe {label:<34} stage {stage:8.1} us (min {stage_min:7.1})  gate-only {gate:7.1} us  eff {gbps:6.0} GB/s"
    );
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RouteArgs {
    experts: u32,
    topk: u32,
    temperature: f32,
    scale: f32,
}

/// The production single-thread router at the V4.1 decode shape
/// (kernel_deepseek_v41_route: 384 experts, top-6, one row).
fn route_probe(ctx: &MetalContext) {
    const EXPERTS: usize = 384;
    let logits: Vec<f32> = (0..EXPERTS)
        .map(|i| (((i * 37 + 11) % 101) as f32 - 50.0) * 0.07)
        .collect();
    let bias: Vec<f32> = (0..EXPERTS)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.01)
        .collect();
    let logits_t = f32_tensor(ctx, &logits, &[EXPERTS as u64]);
    let bias_t = f32_tensor(ctx, &bias, &[EXPERTS as u64]);
    let scores_t = MetalTensor::zeros_f32(ctx, vec![EXPERTS as u64]).expect("scores");
    let ids_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&[0i32; TOP_K]),
        vec![TOP_K as u64],
        GgmlType::I32,
    )
    .expect("ids");
    let weights_t = MetalTensor::zeros_f32(ctx, vec![TOP_K as u64]).expect("weights");
    let status_t =
        MetalTensor::from_bytes(ctx, bytemuck::cast_slice(&[0i32]), vec![1], GgmlType::I32)
            .expect("status");
    let encode = |enc: &KernelEncoder, n: usize| -> Result<(), MetalError> {
        let pso = ctx.pipeline("kernel_deepseek_v41_route")?;
        for _ in 0..n {
            enc.set_pipeline(&pso);
            enc.set_bytes(
                0,
                &RouteArgs {
                    experts: EXPERTS as u32,
                    topk: TOP_K as u32,
                    temperature: 1.0,
                    scale: 1.5,
                },
            );
            for (slot, t) in [&logits_t, &bias_t, &scores_t, &ids_t, &weights_t, &status_t]
                .iter()
                .enumerate()
            {
                enc.set_tensor(slot + 1, t);
            }
            enc.dispatch(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Ok(())
    };
    for n in [1usize, 10, 40] {
        let mut us = Vec::new();
        for it in 0..ITERS + 2 {
            let t = timed(ctx, |enc| encode(enc, n));
            if it >= 2 {
                us.push(t);
            }
        }
        let med = median(us.clone());
        let min = us.iter().cloned().fold(f64::INFINITY, f64::min);
        eprintln!(
            "xprobe route x{n:<3} (384 experts, top-6, 256 threads)   total {med:8.1} us (min {min:7.1})  per-route {:6.1} us",
            med / n as f64
        );
    }
    let ids = unsafe {
        std::slice::from_raw_parts(ids_t.buffer.contents().as_ptr() as *const i32, TOP_K)
    };
    let status = unsafe { *(status_t.buffer.contents().as_ptr() as *const i32) };
    eprintln!("xprobe route ids={ids:?} status={status}");
}

#[test]
fn routed_decode_attribution_probe() {
    if std::env::var_os("QWEN_EXPERT_DECODE_PROBE").is_none() {
        return;
    }
    let _lease = production_lease();
    let Some(ctx) = metal_test_context() else {
        return;
    };
    eprintln!(
        "xprobe shape K={K} M={M} top_k={TOP_K} bank_experts={BANK_EXPERTS} iters={ITERS} (median GPU us)"
    );
    route_probe(&ctx);

    // Production cohorts: the kernels V4.1 decode runs today.
    for (gate_up, down) in [
        (GgmlType::Q3_K, GgmlType::Q3_K),
        (GgmlType::Q3_K, GgmlType::MXFP4),
        (GgmlType::IQ2_XS, GgmlType::MXFP4),
        (GgmlType::Q8_0, GgmlType::Q8_0),
    ] {
        let f = Fixture::new(&ctx, gate_up, down);
        f.warm(&ctx);
        f.heat(&ctx);
        let mut stage = Vec::new();
        let mut gate = Vec::new();
        for it in 0..ITERS + 2 {
            write_ids(&f.ids, it);
            let s = timed(&ctx, |enc| {
                encode_all_slots_gate_up_swiglu_native(
                    &ctx,
                    enc,
                    &f.gate,
                    &f.up,
                    &f.x,
                    &f.ids,
                    &f.status,
                    &f.inner,
                    K,
                    M,
                    BANK_EXPERTS,
                    TOP_K,
                    10.0,
                )?;
                encode_all_slots_down_native(
                    &ctx,
                    enc,
                    &f.down,
                    &f.inner,
                    &f.ids,
                    &f.status,
                    &f.out,
                    M,
                    K,
                    BANK_EXPERTS,
                    TOP_K,
                )
            });
            write_ids(&f.ids, it + 1000);
            let g = timed(&ctx, |enc| {
                encode_all_slots_gate_up_swiglu_native(
                    &ctx,
                    enc,
                    &f.gate,
                    &f.gate,
                    &f.x,
                    &f.ids,
                    &f.status,
                    &f.inner,
                    K,
                    M,
                    BANK_EXPERTS,
                    TOP_K,
                    10.0,
                )
            }) / 2.0;
            if it >= 2 {
                stage.push(s);
                gate.push(g);
            }
        }
        report(&format!("production {gate_up:?}/{down:?}"), &f, stage, gate);
        // Same kernels with a decode-token-sized CPU gap before each command
        // buffer, as the real loop has (sample, encode, Engram reads).
        for gap_ms in [0.5f64, 1.0, 2.0, 5.0] {
            let mut stage = Vec::new();
            let mut gate = Vec::new();
            for it in 0..ITERS + 2 {
                write_ids(&f.ids, it + 31);
                std::thread::sleep(std::time::Duration::from_secs_f64(gap_ms * 1e-3));
                let s = timed(&ctx, |enc| {
                    encode_all_slots_gate_up_swiglu_native(
                        &ctx,
                        enc,
                        &f.gate,
                        &f.up,
                        &f.x,
                        &f.ids,
                        &f.status,
                        &f.inner,
                        K,
                        M,
                        BANK_EXPERTS,
                        TOP_K,
                        10.0,
                    )?;
                    encode_all_slots_down_native(
                        &ctx,
                        enc,
                        &f.down,
                        &f.inner,
                        &f.ids,
                        &f.status,
                        &f.out,
                        M,
                        K,
                        BANK_EXPERTS,
                        TOP_K,
                    )
                });
                if it >= 2 {
                    stage.push(s);
                    gate.push(0.0);
                }
            }
            report(
                &format!("production {gate_up:?}/{down:?} gap {gap_ms} ms"),
                &f,
                stage,
                gate,
            );
        }
    }

    // Research variants on Q3_K (gate/up/down all Q3_K) and Q8_0.
    for (dtype, suffix, variants) in [
        (
            GgmlType::Q3_K,
            "q3_K",
            &[
                ("r1_x0", 1usize),
                ("r1_x1", 1),
                ("r1_x2", 1),
                ("r1_x3", 1),
                ("r1_x4", 1),
                ("r1_x5", 1),
                ("prod", 1),
                ("r2_x1", 2),
                ("r4_x0", 4),
                ("r4_x1", 4),
                ("r4_x2", 4),
                ("r4_x3", 4),
                ("r8_x1", 8),
                ("r8_x2", 8),
            ][..],
        ),
        (
            GgmlType::Q8_0,
            "q8_0",
            &[
                ("r1_x0", 1usize),
                ("r1_x3", 1),
                ("r4_x1", 4),
                ("r1_x1", 1),
                ("r1_x5", 1),
                ("prod", 1),
            ][..],
        ),
    ] {
        let f = Fixture::new(&ctx, dtype, dtype);
        f.warm(&ctx);
        f.heat(&ctx);
        let mut stage: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
        let mut gate: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
        for it in 0..ITERS + 2 {
            for (v, &(variant, rows)) in variants.iter().enumerate() {
                let name = if variant == "prod" {
                    format!("kernel_xprobe_prod_{suffix}")
                } else {
                    format!("kernel_xprobe_{suffix}_{variant}")
                };
                write_ids(&f.ids, it * variants.len() + v + 7);
                let s = timed(&ctx, |enc| {
                    probe_dispatch(
                        &ctx,
                        enc,
                        &name,
                        rows,
                        &f.gate,
                        &f.x,
                        &f,
                        &f.inner,
                        Shape { n_in: K, n_out: M },
                        0,
                    )?;
                    probe_dispatch(
                        &ctx,
                        enc,
                        &name,
                        rows,
                        &f.up,
                        &f.x,
                        &f,
                        &f.inner,
                        Shape { n_in: K, n_out: M },
                        1,
                    )?;
                    probe_dispatch(
                        &ctx,
                        enc,
                        &name,
                        rows,
                        &f.down,
                        &f.inner,
                        &f,
                        &f.out,
                        Shape { n_in: M, n_out: K },
                        2,
                    )
                });
                write_ids(&f.ids, it * variants.len() + v + 1007);
                let g = timed(&ctx, |enc| {
                    probe_dispatch(
                        &ctx,
                        enc,
                        &name,
                        rows,
                        &f.gate,
                        &f.x,
                        &f,
                        &f.inner,
                        Shape { n_in: K, n_out: M },
                        0,
                    )
                });
                if it >= 2 {
                    stage[v].push(s);
                    gate[v].push(g);
                }
            }
        }
        for (v, &(variant, _)) in variants.iter().enumerate() {
            report(
                &format!("{suffix} {variant}"),
                &f,
                stage[v].clone(),
                gate[v].clone(),
            );
        }
    }
}
