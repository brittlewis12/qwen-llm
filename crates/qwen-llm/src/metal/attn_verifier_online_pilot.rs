//! Model-free falsifier, not a verifier selector or an endpoint benchmark.

use super::*;
use objc2_metal::MTLCommandBufferStatus;

const ROWS: usize = 16;
const NQ: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const GROUP: usize = NQ / NKV;
const QDIM: usize = NQ * HD;
const KVDIM: usize = NKV * HD;

struct Pilot {
    q: MetalTensor,
    k: MetalTensor,
    v: MetalTensor,
    vt: MetalTensor,
    scores: MetalTensor,
    ml: MetalTensor,
    partial_o: MetalTensor,
    partial_ml: MetalTensor,
    scalar: MetalTensor,
    online: MetalTensor,
    n_pos: usize,
}

impl Pilot {
    fn new(ctx: &MetalContext, n_pos: usize) -> Self {
        // Deterministic nonperiodic inputs; Q/K variance ~1 gives nonuniform
        // softmax, unlike tiny periodic values that nearly average V uniformly.
        let mut state = 0x6a09_e667_f3bc_c909u64;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 16_777_216.0 * 2.0 - 1.0) * 1.732_050_8
        };
        let q: Vec<f32> = (0..ROWS * QDIM).map(|_| sample()).collect();
        let q = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&q),
            vec![(ROWS * QDIM) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let mut kv = || {
            let values: Vec<u16> = (0..n_pos * KVDIM)
                .map(|_| half::f16::from_f32(sample()).to_bits())
                .collect();
            MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&values),
                vec![values.len() as u64],
                GgmlType::F16,
            )
            .unwrap()
        };
        let k = kv();
        let v = kv();
        let before = ctx.current_allocated_size();
        let vt = MetalTensor::zeros_f16(ctx, vec![(n_pos * KVDIM) as u64]).unwrap();
        let scores = MetalTensor::zeros_f16(ctx, vec![(ROWS * NQ * n_pos) as u64]).unwrap();
        let ml = MetalTensor::zeros_f32(ctx, vec![attn_matrix_ml_elems(ROWS, NQ, n_pos) as u64])
            .unwrap();
        let delta = ctx.current_allocated_size() - before;
        let logical: u64 = [&vt, &scores, &ml].iter().map(|t| t.n_bytes()).sum();
        println!(
            "VERIFY_ONLINE_JSON {}",
            serde_json::json!({
                "kind": "workspace", "n_pos": n_pos, "logical_bytes": logical,
                "allocated_delta_bytes": delta, "scope": "incremental VT/scores/ml only",
            })
        );
        assert!(
            delta <= 128 * 1024 * 1024,
            "incremental workspace exceeds screen cap"
        );
        let nwg = attn_v4_choose_nwg(n_pos, GROUP);
        assert_eq!(nwg, 128);
        assert_eq!(attn_v4_choose_tile_c(n_pos, GROUP), 32);
        Self {
            q,
            k,
            v,
            vt,
            scores,
            ml,
            partial_o: MetalTensor::zeros_f32(ctx, vec![(NQ * nwg * HD) as u64]).unwrap(),
            partial_ml: MetalTensor::zeros_f32(ctx, vec![(NQ * nwg * 2) as u64]).unwrap(),
            scalar: MetalTensor::zeros_f32(ctx, vec![(ROWS * QDIM) as u64]).unwrap(),
            online: MetalTensor::zeros_f32(ctx, vec![(ROWS * QDIM) as u64]).unwrap(),
            n_pos,
        }
    }

    fn encode_matrix(&self, ctx: &MetalContext, enc: &KernelEncoder, part: &str) {
        if matches!(part, "B" | "transpose") {
            encode_attn_matrix_transpose_v_f16(
                ctx, enc, &self.v, &self.vt, 0, self.n_pos, self.n_pos, KVDIM, self.n_pos, NKV, HD,
            )
            .unwrap();
        }
        if matches!(part, "B" | "kq") {
            encode_attn_matrix_kq_online_f32(
                ctx,
                enc,
                &self.q,
                &self.k,
                &self.scores,
                &self.ml,
                ROWS,
                self.n_pos - ROWS,
                self.n_pos,
                KVDIM,
                NQ,
                NKV,
                GROUP,
                HD,
                true,
            )
            .unwrap();
        }
        if matches!(part, "B" | "kqv") {
            encode_attn_matrix_kqv_norm_f32(
                ctx,
                enc,
                &self.scores,
                &self.ml,
                &self.vt,
                &self.online,
                ROWS,
                self.n_pos - ROWS,
                self.n_pos,
                self.n_pos,
                NQ,
                NKV,
                GROUP,
                HD,
                true,
            )
            .unwrap();
        }
    }

    fn run(&self, ctx: &MetalContext, arm: &str, repeats: usize) -> (f64, f64) {
        let start = std::time::Instant::now();
        let cmd = ctx.queue.commandBuffer().expect("pilot command buffer");
        for _ in 0..repeats {
            if arm == "A" {
                for row in 0..ROWS {
                    let enc = KernelEncoder::begin(&cmd);
                    let q = self.q.view_subrange((row * QDIM) as u64, vec![QDIM as u64]);
                    let out = self
                        .scalar
                        .view_subrange((row * QDIM) as u64, vec![QDIM as u64]);
                    let extent = self.n_pos - ROWS + row + 1;
                    encode_attn_decode_v4_f32(
                        ctx,
                        &enc,
                        &q,
                        &self.k,
                        &self.v,
                        &self.partial_o,
                        &self.partial_ml,
                        &out,
                        NQ,
                        NKV,
                        HD,
                        extent,
                        attn_v4_choose_nwg(extent, GROUP),
                        attn_v4_choose_tile_c(extent, GROUP),
                    )
                    .unwrap();
                    enc.end();
                }
            } else {
                let enc = KernelEncoder::begin(&cmd);
                self.encode_matrix(ctx, &enc, arm);
                enc.end();
            }
        }
        cmd.commit();
        crate::metal::wait_unchecked(&cmd);
        let wall_ms = start.elapsed().as_secs_f64() * 1e3 / repeats as f64;
        assert_eq!(
            cmd.status(),
            MTLCommandBufferStatus::Completed,
            "{:?}",
            cmd.error()
        );
        assert!(cmd.error().is_none(), "{:?}", cmd.error());
        let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3 / repeats as f64;
        assert!(gpu_ms.is_finite() && gpu_ms > 0.0);
        (wall_ms, gpu_ms)
    }

    fn check(&self) {
        let a = read_back_f32(&self.scalar.buffer, ROWS * QDIM);
        let b = read_back_f32(&self.online.buffer, ROWS * QDIM);
        let mut worst_cos = 1.0f64;
        let mut max_abs = 0.0f32;
        for (a, b) in a.chunks_exact(QDIM).zip(b.chunks_exact(QDIM)) {
            assert!(a.iter().chain(b).all(|v| v.is_finite()));
            let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
            let aa: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
            let bb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum();
            worst_cos = worst_cos.min(dot / (aa * bb).sqrt());
            max_abs = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| (x - y).abs())
                .fold(max_abs, f32::max);
        }
        println!(
            "VERIFY_ONLINE_JSON {}",
            serde_json::json!({
                "kind": "correctness", "n_pos": self.n_pos,
                "worst_row_cosine": worst_cos, "max_abs": max_abs,
            })
        );
        assert!(worst_cos >= 0.9999 && max_abs < 0.005);
    }
}

#[test]
#[ignore]
fn verifier_online_n16_attention_screen() {
    for (key, _) in std::env::vars_os() {
        let key = key.to_string_lossy();
        assert!(
            !key.starts_with("QWEN_ATTN_"),
            "unexpected attention override {key}"
        );
    }
    let ctx = MetalContext::new().expect("required Metal context");
    let pilot = Pilot::new(&ctx, 32768);
    for arm in ["A", "B", "B", "A"] {
        pilot.run(&ctx, arm, 1);
    }
    pilot.check();
    for (index, arm) in ["A", "B", "B", "A"].into_iter().enumerate() {
        let (wall, gpu) = pilot.run(&ctx, arm, 8);
        println!(
            "VERIFY_ONLINE_JSON {}",
            serde_json::json!({
                "kind": "sample", "index": index, "arm": arm, "repeats": 8,
                "wall_ms": wall, "gpu_ms": gpu, "n_pos": pilot.n_pos,
            })
        );
    }
    pilot.check();
    for arm in ["transpose", "kq", "kqv"] {
        let (wall, gpu) = pilot.run(&ctx, arm, 8);
        println!(
            "VERIFY_ONLINE_JSON {}",
            serde_json::json!({
                "kind": "diagnostic", "arm": arm, "wall_ms": wall, "gpu_ms": gpu,
            })
        );
    }
    drop(pilot);
    let edge = Pilot::new(&ctx, 32769);
    edge.run(&ctx, "A", 1);
    edge.run(&ctx, "B", 1);
    edge.check();
}
