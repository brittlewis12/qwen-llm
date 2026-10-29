//! Resident synthetic adapter for the existing query-batch packing/value sink.
//! No workspace, model, routing override, or duplicate attention implementation.
use super::*;
use metal::GroupedF32TestVariant::{self, Production, Scalar};

const M: usize = 512;
const H: usize = 64;
const WINDOW: usize = 128;
const TOPK: usize = 512;
const SCORE_STRIDE: usize = WINDOW + TOPK + 1;
// Keep timed operands 256-byte aligned; shared operator fixtures separately test
// the minimum four-byte alignment with 20-byte offsets.
const GUARD: usize = 64;
const CANARY: f32 = -777.0;

struct Fixture {
    k: usize,
    n: usize,
    first: usize,
    start: usize,
    topk: usize,
    ratio: usize,
    query: MetalTensor,
    query_half: MetalTensor,
    _query_half_backing: MetalTensor,
    history: MetalTensor,
    compressed: MetalTensor,
    history_half: MetalTensor,
    compressed_half: MetalTensor,
    _half_backings: Vec<MetalTensor>,
    selected: MetalTensor,
    sink: MetalTensor,
    scores: MetalTensor,
    values: MetalTensor,
    probabilities: MetalTensor,
    output: MetalTensor,
    status: MetalTensor,
    guards: Vec<MetalTensor>,
}

impl Fixture {
    fn new(ctx: &MetalContext, k: usize, n: usize, ragged: bool, first: usize) -> Self {
        assert!([256, 512, 640].contains(&k) && n > 0 && n <= 512);
        let topk = TOPK;
        // K256 cannot have count<256 on the current matrix branch. It still has
        // distinct selection per query; larger K also has ragged visibility.
        let ratio = if !ragged && k < 640 { 128 } else { 2 };
        let position = if ragged {
            (k - WINDOW - 1) * ratio - n
        } else {
            (k - WINDOW + if k == 640 { 32 } else { 0 }) * ratio - 1
        };
        let start = position - first;
        let end = (position + n) / ratio;
        assert!(start >= WINDOW);
        assert_eq!((WINDOW + end.min(topk)).div_ceil(32) * 32, k);
        assert!(WINDOW + ((position + 1) / ratio).min(topk) >= 256);
        let mut guards = Vec::new();
        let mut alloc = |data: Vec<f32>, shape: Vec<u64>| {
            let mut backing = vec![CANARY; GUARD];
            backing.extend_from_slice(&data);
            backing.extend_from_slice(&[CANARY; GUARD]);
            let t = upload(ctx, &backing, vec![backing.len() as u64]).unwrap();
            let view = t.view_subrange(GUARD as u64, shape);
            guards.push(t);
            view
        };
        let data = |len: usize, seed: usize| {
            (0..len)
                .map(|i| {
                    ((i * 17 + i / M * 7 + seed) % 61) as f32 / 64.0 - 0.46875
                        + ((i + seed) % 3) as f32 / 8192.0
                })
                .collect::<Vec<_>>()
        };
        let query = alloc(
            data((first + n) * H * M, 11),
            vec![(H * M) as u64, (first + n) as u64],
        )
        .view_subrange((first * H * M) as u64, vec![(H * M) as u64, n as u64]);
        let history_data = data((WINDOW + first + n) * M, 23);
        let compressed_data = data(end * M, 37);
        let history = alloc(
            history_data.clone(),
            vec![((WINDOW + first + n) * M) as u64],
        );
        let compressed = alloc(compressed_data.clone(), vec![(end * M) as u64]);
        let half_alloc = |data: &[f32], shape: Vec<u64>| -> (MetalTensor, MetalTensor) {
            let mut raw: Vec<u16> = Vec::with_capacity(data.len() + 2 * GUARD);
            raw.extend(std::iter::repeat_n(
                half::f16::from_f32(CANARY).to_bits(),
                GUARD,
            ));
            raw.extend(data.iter().map(|&v| half::f16::from_f32(v).to_bits()));
            raw.extend(std::iter::repeat_n(
                half::f16::from_f32(CANARY).to_bits(),
                GUARD,
            ));
            let len = raw.len() as u64;
            let backing =
                MetalTensor::from_bytes(ctx, bytemuck::cast_slice(&raw), vec![len], GgmlType::F16)
                    .unwrap();
            let view = backing.view_subrange(GUARD as u64, shape);
            (backing, view)
        };
        let (history_half_backing, history_half) =
            half_alloc(&history_data, vec![((WINDOW + first + n) * M) as u64]);
        let (compressed_half_backing, compressed_half) =
            half_alloc(&compressed_data, vec![(end * M) as u64]);
        let query_data = f32_slice(&query).unwrap();
        let (query_half_backing, query_half) =
            half_alloc(&query_data, vec![(H * M) as u64, n as u64]);
        let selections: Vec<u32> = (0..n)
            .flat_map(|q| {
                let visible = (start + first + q + 1) / ratio;
                (0..topk).map(move |i| {
                    if i < visible.min(topk) {
                        ((visible - 1 - i + (q + first) * 7) % visible) as u32
                    } else {
                        u32::MAX
                    }
                })
            })
            .collect();
        let selected = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&selections),
            vec![topk as u64, n as u64],
            GgmlType::I32,
        )
        .unwrap();
        let sink = alloc(vec![-0.25; H], vec![H as u64]);
        let scores = alloc(
            vec![f32::NAN; n * H * SCORE_STRIDE],
            vec![(n * H * SCORE_STRIDE) as u64],
        );
        let values = alloc(
            vec![f32::NAN; n * M * k],
            vec![k as u64, M as u64, n as u64],
        );
        let probabilities = alloc(vec![f32::NAN; n * H * k], vec![(k * n) as u64, H as u64]);
        let output = alloc(vec![f32::NAN; n * H * M], vec![(M * n) as u64, H as u64]);
        let status =
            MetalTensor::from_bytes(ctx, bytemuck::cast_slice(&[0u32]), vec![1], GgmlType::I32)
                .unwrap();
        Self {
            k,
            n,
            first,
            start,
            topk,
            ratio,
            query,
            query_half,
            _query_half_backing: query_half_backing,
            history,
            compressed,
            history_half,
            compressed_half,
            _half_backings: vec![history_half_backing, compressed_half_backing],
            selected,
            sink,
            scores,
            values,
            probabilities,
            output,
            status,
            guards,
        }
    }

    fn count(&self, q: usize) -> usize {
        WINDOW + ((self.start + self.first + q + 1) / self.ratio).min(self.topk)
    }

    fn bind(&self, enc: &KernelEncoder) {
        enc.set_bytes(
            0,
            &[
                M as u32,
                WINDOW as u32,
                self.topk as u32,
                SCORE_STRIDE as u32,
            ],
        );
        enc.set_bytes(
            1,
            &[
                (self.start + self.first) as u32,
                (self.start - WINDOW) as u32,
                (WINDOW + self.first + self.n) as u32,
                self.ratio as u32,
            ],
        );
        enc.set_bytes(11, &[self.k as u32, H as u32, self.n as u32, 0]);
        for (slot, tensor) in [
            (2, &self.query),
            (3, &self.history),
            (4, &self.compressed),
            (5, &self.selected),
            (6, &self.sink),
            (7, &self.scores),
            (8, &self.output),
            (10, &self.status),
            (12, &self.values),
            (13, &self.probabilities),
        ] {
            enc.set_tensor(slot, tensor);
        }
    }

    fn prepare(&self, ctx: &MetalContext, enc: &KernelEncoder) {
        self.bind(enc);
        pipeline(ctx, enc, "score_rows").unwrap();
        enc.set_threadgroup_memory(0, M * std::mem::size_of::<f32>());
        grid(enc, self.count(self.n - 1).div_ceil(32), H, self.n, 256);
        pipeline(ctx, enc, "softmax_rows").unwrap();
        grid(enc, H, self.n, 1, 32);
    }

    fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        variant: GroupedF32TestVariant,
        scope: Scope,
    ) {
        if scope == Scope::Sink {
            self.prepare(ctx, enc);
        }
        if scope != Scope::Pv {
            self.bind(enc);
            pipeline(ctx, enc, "matrix_values_rows").unwrap();
            grid(enc, self.k / 16, M / 16, self.n, 256);
        }
        metal::encode_mat_mat_grouped_f32_for_test(
            ctx,
            enc,
            &self.values,
            &self.probabilities,
            &self.output,
            self.k,
            M,
            self.n,
            H,
            variant,
        )
        .unwrap();
        if scope != Scope::Pv {
            let pso = pipeline(ctx, enc, "finite").unwrap();
            enc.set_bytes(0, &((self.n * H * M) as u32));
            enc.set_tensor(1, &self.output);
            enc.set_tensor(2, &self.status);
            dispatch(enc, &pso, self.n * H * M);
        }
    }

    fn checked(&self, ctx: &MetalContext, variant: GroupedF32TestVariant, scope: Scope) {
        metal::dispatch_census_begin();
        execute(ctx, |enc| self.encode(ctx, enc, variant, scope));
        let census = metal::dispatch_census_take();
        let mut names = Vec::new();
        if scope == Scope::Sink {
            names.extend([
                "kernel_deepseek_v41_attention_score_rows",
                "kernel_deepseek_v41_attention_softmax_rows",
            ]);
        }
        if scope != Scope::Pv {
            names.push("kernel_deepseek_v41_attention_matrix_values_rows");
        }
        let mma = variant == Production && self.n >= 8;
        let kernel = if mma {
            "kernel_latent_grouped_f32_f32_m16n64k32"
        } else {
            "kernel_latent_grouped_f32_f32"
        };
        names.push(kernel);
        if scope != Scope::Pv {
            names.push("kernel_deepseek_v41_attention_finite");
        }
        assert_eq!(
            census.iter().map(|r| r.kernel.as_str()).collect::<Vec<_>>(),
            names
        );
        let pv = census.iter().find(|r| r.kernel == kernel).unwrap();
        let (x, y) = if mma {
            (H / 64, M / 16)
        } else {
            (M / 4, H / 4)
        };
        assert_eq!(
            (pv.grid_width, pv.grid_height, pv.grid_depth, pv.tg_threads),
            (x as u64, y as u64, self.n as u64, 128)
        );
        assert!(census.iter().all(|r| !r.encoder_concurrent));
    }

    fn check_packing_and_f64(&self) {
        let values = f32_slice(&self.values).unwrap();
        let p = f32_slice(&self.probabilities).unwrap();
        let scores = f32_slice(&self.scores).unwrap();
        let output = f32_slice(&self.output).unwrap();
        let raw = f32_slice(&self.history).unwrap();
        let compressed = f32_slice(&self.compressed).unwrap();
        let selected = ids(&self.selected, self.topk * self.n);
        for q in 0..self.n {
            let count = self.count(q);
            let value = |i: usize, j: usize| {
                if i < WINDOW {
                    raw[(self.first + q + 1 + i) * M + j]
                } else if i < count {
                    compressed[selected[q * self.topk + i - WINDOW] as usize * M + j]
                } else {
                    0.0
                }
            };
            for j in 0..M {
                for i in 0..self.k {
                    assert_eq!(
                        values[(q * M + j) * self.k + i].to_bits(),
                        value(i, j).to_bits()
                    );
                }
            }
            for h in 0..H {
                let base = (q * H + h) * SCORE_STRIDE;
                for i in 0..self.k {
                    let expected = if i < count {
                        scores[base + i] / scores[base + SCORE_STRIDE - 1]
                    } else {
                        0.0
                    };
                    let actual = p[(h * self.n + q) * self.k + i];
                    assert!(actual.is_finite() && (actual - expected).abs() <= 1e-7);
                    if i >= count {
                        assert_eq!(actual.to_bits(), 0);
                    }
                }
            }
            // Independent F64 score/softmax/PV, sampled heads and dimensions;
            // all output elements are also compared against current scalar PV.
            let queries = f32_slice(&self.query).unwrap();
            for h in [0, 31, 63] {
                let logits: Vec<f64> = (0..count)
                    .map(|i| {
                        (0..M)
                            .map(|j| {
                                f64::from(queries[(q * H + h) * M + j]) * f64::from(value(i, j))
                            })
                            .sum::<f64>()
                            / (M as f64).sqrt()
                    })
                    .collect();
                let max = logits.iter().copied().fold(-0.25, f64::max);
                let mass: Vec<f64> = logits.iter().map(|s| (s - max).exp()).collect();
                let denom = mass.iter().sum::<f64>() + (-0.25 - max).exp();
                for j in [0, 1, 7, 8, 255, 511] {
                    let expected = (0..count)
                        .map(|i| mass[i] * f64::from(value(i, j)))
                        .sum::<f64>()
                        / denom;
                    let got = f64::from(output[(h * self.n + q) * M + j]);
                    assert!(
                        got.is_finite() && (got - expected).abs() <= 2e-5 * expected.abs().max(1.0),
                        "q={q} h={h} j={j}: {got} vs {expected}"
                    );
                }
            }
        }
        self.check_guards();
        assert_eq!(ids(&self.status, 1), [0]);
    }

    fn check_guards(&self) {
        for t in &self.guards {
            let data = f32_slice(t).unwrap();
            assert!(
                data[..GUARD]
                    .iter()
                    .chain(&data[data.len() - GUARD..])
                    .all(|v| v.to_bits() == CANARY.to_bits())
            );
        }
    }
}

fn grid(enc: &KernelEncoder, width: usize, height: usize, depth: usize, threads: usize) {
    enc.dispatch(
        MTLSize {
            width,
            height,
            depth,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scope {
    Pv,
    PackPvFinite,
    Sink,
}

fn cases() -> impl Iterator<Item = (usize, usize, bool, usize)> {
    [256, 512, 640].into_iter().flat_map(|k| {
        [1, 8, 32].into_iter().flat_map(move |n| {
            [false, true]
                .into_iter()
                .filter(move |&ragged| !ragged || k > 256)
                .map(move |ragged| (k, n, ragged, if n == 1 { 32 } else { 0 }))
        })
    })
}

#[test]
fn f32_pv_batch_correctness() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    for (k, n, ragged, first) in cases() {
        let f = Fixture::new(&ctx, k, n, ragged, first);
        f.checked(&ctx, Scalar, Scope::Sink);
        let scalar = f32_slice(&f.output).unwrap().to_vec();
        for variant in [Scalar, Production] {
            for scope in [Scope::Sink, Scope::PackPvFinite, Scope::Pv] {
                f.checked(&ctx, variant, scope);
                close(f32_slice(&f.output).unwrap(), &scalar);
            }
            f.check_packing_and_f64();
        }
    }
}

#[test]
fn f32_pv_batch_invalid_visibility_and_finite_guards() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    for variant in [Scalar, Production] {
        let mut f = Fixture::new(&ctx, 512, 8, true, 0);
        f.checked(&ctx, variant, Scope::Sink);
        let valid = f32_slice(&f.output).unwrap().to_vec();
        let mut selected: Vec<u32> = ids(&f.selected, f.topk * f.n)
            .into_iter()
            .map(|id| id as u32)
            .collect();
        selected[f.topk] = ((f.start + 2) / f.ratio) as u32; // Populated but not visible to query 1.
        f.selected = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&selected),
            vec![f.topk as u64, f.n as u64],
            GgmlType::I32,
        )
        .unwrap();
        f.checked(&ctx, variant, Scope::Sink);
        assert_eq!(ids(&f.status, 1), [2]);
        let got = f32_slice(&f.output).unwrap();
        for h in 0..H {
            for q in 0..f.n {
                let off = (h * f.n + q) * M;
                if q == 1 {
                    assert!(got[off..off + M].iter().all(|&v| v == 0.0));
                } else {
                    close(&got[off..off + M], &valid[off..off + M]);
                }
            }
        }
        let poison = upload(&ctx, &[f32::NAN], vec![1]).unwrap();
        execute(&ctx, |enc| {
            copy(
                &ctx,
                enc,
                &poison,
                &f.history.view_subrange(M as u64, vec![1]),
                1,
            )
            .unwrap()
        });
        f.checked(&ctx, variant, Scope::Sink);
        assert_eq!(ids(&f.status, 1), [6]);
        f.check_guards();
    }
}

/// Production attention tiles dispatch 512 query rows; the older probe used
/// at most 32 and therefore ran far below the GPU's threadgroup occupancy.
fn production_cases() -> impl Iterator<Item = (usize, usize, bool, usize)> {
    [(640, 512, false, 0), (640, 512, true, 0)].into_iter()
}

#[test]
#[ignore = "resident GPU timing; unset MTL_DEBUG_LAYER and QWEN_REQUIRE_METAL_TESTS"]
fn attention_half_kv_tile_probe() {
    for name in ["MTL_DEBUG_LAYER", "QWEN_REQUIRE_METAL_TESTS"] {
        assert!(
            std::env::var_os(name).is_none(),
            "{name} must be unset for performance"
        );
    }
    let ctx = MetalContext::new().expect("explicit timing requires Metal");
    const REPEATS: usize = 3;
    const LAUNCHES: usize = 4;
    // Production tiles are 512 queries; the tiny fixture cases include a
    // pre-existing n=1 tensor64 sentinel edge and are not the probe target.
    for (k, n, ragged, first) in production_cases() {
        let f = Fixture::new(&ctx, k, n, ragged, first);
        execute(&ctx, |enc| f.encode_tensor64(&ctx, enc));
        let base_out = f32_slice(&f.output).unwrap().to_vec();
        let base_scores = f32_slice(&f.scores).unwrap().to_vec();
        for (label, encode) in [
            (
                "kvhalf",
                Fixture::encode_tensor64_half as fn(&Fixture, &MetalContext, &KernelEncoder),
            ),
            ("qhalf", Fixture::encode_tensor64_qhalf),
            ("qkvhalf", Fixture::encode_tensor64_qkvhalf),
        ] {
            execute(&ctx, |enc| encode(&f, &ctx, enc));
            let out = f32_slice(&f.output).unwrap();
            let scores = f32_slice(&f.scores).unwrap();
            // The fixture's first query hits a pre-existing tensor64 sentinel
            // edge (base output is non-finite); compare finite lanes only.
            let mut base_nonfinite = 0;
            let mut mismatches = 0;
            for (a, b) in base_out.iter().zip(out) {
                if !a.is_finite() {
                    base_nonfinite += 1;
                } else if a.to_bits() != b.to_bits() {
                    mismatches += 1;
                }
            }
            for q in 0..n {
                let count = f.count(q);
                for h in 0..H {
                    let base = (q * H + h) * SCORE_STRIDE;
                    for i in 0..count {
                        if base_scores[base + i].is_finite()
                            && base_scores[base + i].to_bits() != scores[base + i].to_bits()
                        {
                            mismatches += 1;
                        }
                    }
                }
            }
            assert_eq!(mismatches, 0, "{label} finite mismatch");
            eprintln!(
                "half_kv_check variant={label} K={k} n={n} ragged={ragged} base_nonfinite={base_nonfinite} finite_mismatches={mismatches}"
            );
        }
        let timed = |arm: usize| {
            let wall = std::time::Instant::now();
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            for _ in 0..LAUNCHES {
                match arm {
                    0 => f.encode_tensor64(&ctx, &enc),
                    1 => f.encode_tensor64_half(&ctx, &enc),
                    2 => f.encode_tensor64_qhalf(&ctx, &enc),
                    _ => f.encode_tensor64_qkvhalf(&ctx, &enc),
                }
            }
            enc.end();
            metal::commit_and_wait(&command).unwrap();
            let wall = wall.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64;
            let start = command.GPUStartTime();
            let end = command.GPUEndTime();
            assert!(
                start.is_finite() && end.is_finite() && start > 0.0 && end > start,
                "GPU timestamps unavailable"
            );
            ((end - start) * 1e6 / LAUNCHES as f64, wall)
        };
        for arm in 0..4 {
            timed(arm);
        }
        let mut samples = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..REPEATS {
            for arm in [0, 1, 2, 3, 3, 2, 1, 0] {
                samples[arm].push(timed(arm));
            }
        }
        let median = |arm: usize| {
            let mut x: Vec<f64> = samples[arm].iter().map(|&(g, _)| g).collect();
            x.sort_by(f64::total_cmp);
            (x[2] + x[3]) * 0.5
        };
        let base = median(0);
        let kv = median(1);
        let q = median(2);
        let qkv = median(3);
        eprintln!(
            "half_kv_probe K={k} n={n} ragged={ragged} first={first} f32_us={base:.3} kvhalf_us={kv:.3} qhalf_us={q:.3} qkvhalf_us={qkv:.3} kv_speedup={:.3} q_speedup={:.3} qkv_speedup={:.3}",
            base / kv,
            base / q,
            base / qkv
        );
        f.check_guards();
    }
}

#[test]
#[ignore = "worker-only resident GPU timing; unset MTL_DEBUG_LAYER and QWEN_REQUIRE_METAL_TESTS"]
fn f32_pv_batch_performance() {
    for name in ["MTL_DEBUG_LAYER", "QWEN_REQUIRE_METAL_TESTS"] {
        assert!(
            std::env::var_os(name).is_none(),
            "{name} must be unset for performance"
        );
    }
    let ctx = MetalContext::new().expect("explicit timing requires Metal");
    const REPEATS: usize = 3;
    const LAUNCHES: usize = 4;
    eprintln!(
        "F32 PV resident synthetic timing: M=512 heads=64; each query is a separate group/value matrix. GPU command timestamps and encode+commit+wait wall us per launch; excludes allocation, compilation, readback, oracle. Sink=score+softmax+pack+PV+finite, excludes RoPE/model. No device peak/DRAM claim. Candidate is unforced production dispatch; Scalar is explicitly forced with census proof."
    );
    for (k, n, ragged, first) in cases() {
        let f = Fixture::new(&ctx, k, n, ragged, first);
        f.checked(&ctx, Scalar, Scope::Sink);
        let scalar = f32_slice(&f.output).unwrap().to_vec();
        let variant = Production;
        f.checked(&ctx, variant, Scope::Sink);
        close(f32_slice(&f.output).unwrap(), &scalar);
        f.check_packing_and_f64();
        for scope in [Scope::Pv, Scope::PackPvFinite, Scope::Sink] {
            let timed = |v| {
                assert!(!metal::dispatch_census_is_active());
                let wall = std::time::Instant::now();
                let command = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&command);
                for _ in 0..LAUNCHES {
                    f.encode(&ctx, &enc, v, scope);
                }
                enc.end();
                metal::commit_and_wait(&command).unwrap();
                let wall = wall.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64;
                let start = command.GPUStartTime();
                let end = command.GPUEndTime();
                assert!(
                    start.is_finite() && end.is_finite() && start > 0.0 && end > start,
                    "GPU timestamps unavailable"
                );
                ((end - start) * 1e6 / LAUNCHES as f64, wall)
            };
            timed(Scalar);
            timed(variant);
            let mut samples = [Vec::new(), Vec::new()];
            for repeat in 0..REPEATS {
                for (arm, v) in [(0, Scalar), (1, variant), (1, variant), (0, Scalar)] {
                    let (gpu, wall) = timed(v);
                    samples[arm].push((gpu, wall));
                    eprintln!(
                        "pv_batch sample K={k} queries={n} count={}..{} compression_ratio={} first={first} ragged={ragged} scope={scope:?} candidate={variant:?} repeat={repeat} arm={v:?} gpu_us={gpu:.3} wall_us={wall:.3}",
                        f.count(0),
                        f.count(n - 1),
                        f.ratio
                    );
                }
            }
            let median = |arm: usize, wall: bool| {
                let mut x: Vec<f64> = samples[arm]
                    .iter()
                    .map(|&(g, w)| if wall { w } else { g })
                    .collect();
                x.sort_by(f64::total_cmp);
                (x[2] + x[3]) * 0.5
            };
            let a = median(0, false);
            let b = median(1, false);
            let useful = 2 * M * H * (0..n).map(|q| f.count(q)).sum::<usize>();
            let padded = 2 * M * H * k * n;
            eprintln!(
                "pv_batch median K={k} queries={n} ragged={ragged} scope={scope:?} candidate={variant:?} ABBA_repeats={REPEATS} scalar_gpu_us={a:.3} mma_gpu_us={b:.3} speedup={:.3} scalar_wall_us={:.3} mma_wall_us={:.3} useful_PV_FLOPs={useful} padded_PV_FLOPs={padded}",
                a / b,
                median(0, true),
                median(1, true)
            );
            close(f32_slice(&f.output).unwrap(), &scalar);
            assert_eq!(ids(&f.status, 1), [0]);
            f.check_guards();
        }
    }
}

fn close_half(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= 2e-3 + 4e-3 * b.abs(),
            "{i}: {a} != {b}"
        );
    }
}

impl Fixture {
    /// The simdgroup-MMA sink tile: QK^T into the score slab, the shared
    /// softmax, PV straight from gathered latent rows, then normalize/finite.
    fn encode_mma(&self, ctx: &MetalContext, enc: &KernelEncoder) {
        self.bind(enc);
        pipeline(ctx, enc, "score_mma").unwrap();
        grid(enc, H / 16, self.n, 1, 128);
        pipeline(ctx, enc, "softmax_rows").unwrap();
        grid(enc, H, self.n, 1, 32);
        pipeline(ctx, enc, "pv_mma").unwrap();
        grid(enc, H / 16, self.n, 1, 256);
        let pso = pipeline(ctx, enc, "normalize_rows").unwrap();
        dispatch(enc, &pso, self.n * H * M);
    }

    /// The Metal 4 tensor-ops form of the same tile (same slabs, softmax
    /// and normalization; matmul2d per key block).
    fn encode_tensor(&self, ctx: &MetalContext, enc: &KernelEncoder) {
        self.bind(enc);
        tensor_pipeline(ctx, enc, "score_tensor").unwrap();
        grid(enc, H / 16, self.n, 1, 128);
        pipeline(ctx, enc, "softmax_rows").unwrap();
        grid(enc, H, self.n, 1, 32);
        tensor_pipeline(ctx, enc, "pv_tensor").unwrap();
        grid(enc, H / 16, self.n, 1, 128);
        let pso = pipeline(ctx, enc, "normalize_rows").unwrap();
        dispatch(enc, &pso, self.n * H * M);
    }

    /// Wider-N tensor score (32 keys, two 256-wide latent chunks) with the
    /// existing softmax, tensor PV and normalize.
    fn encode_tensor32(&self, ctx: &MetalContext, enc: &KernelEncoder) {
        self.bind(enc);
        tensor_pipeline(ctx, enc, "score_tensor32").unwrap();
        grid(enc, H / 16, self.n, 1, 128);
        pipeline(ctx, enc, "softmax_rows").unwrap();
        grid(enc, H, self.n, 1, 32);
        tensor_pipeline(ctx, enc, "pv_tensor").unwrap();
        grid(enc, H / 16, self.n, 1, 128);
        let pso = pipeline(ctx, enc, "normalize_rows").unwrap();
        dispatch(enc, &pso, self.n * H * M);
    }

    /// All 64 heads in one threadgroup (32 keys, four 128-wide latent chunks).
    fn encode_tensor64(&self, ctx: &MetalContext, enc: &KernelEncoder) {
        self.bind(enc);
        tensor_pipeline(ctx, enc, "score_tensor64").unwrap();
        grid(enc, 1, self.n, 1, 128);
        pipeline(ctx, enc, "softmax_rows").unwrap();
        grid(enc, H, self.n, 1, 32);
        tensor_pipeline(ctx, enc, "pv_tensor64").unwrap();
        grid(enc, 1, self.n, 1, 128);
        let pso = pipeline(ctx, enc, "normalize_rows").unwrap();
        dispatch(enc, &pso, self.n * H * M);
    }

    /// Same tile with the K/V latent read from the half mirrors.
    fn bind_half(&self, enc: &KernelEncoder) {
        self.bind(enc);
        enc.set_tensor(3, &self.history_half);
        enc.set_tensor(4, &self.compressed_half);
    }

    fn encode_tensor64_half(&self, ctx: &MetalContext, enc: &KernelEncoder) {
        self.bind_half(enc);
        tensor_pipeline(ctx, enc, "score_tensor64_half").unwrap();
        grid(enc, 1, self.n, 1, 128);
        pipeline(ctx, enc, "softmax_rows").unwrap();
        grid(enc, H, self.n, 1, 32);
        tensor_pipeline(ctx, enc, "pv_tensor64_half").unwrap();
        grid(enc, 1, self.n, 1, 128);
        let pso = pipeline(ctx, enc, "normalize_rows").unwrap();
        dispatch(enc, &pso, self.n * H * M);
    }

    /// Score with the Q half mirror; K/V stay F32.
    fn encode_tensor64_qhalf(&self, ctx: &MetalContext, enc: &KernelEncoder) {
        self.bind(enc);
        enc.set_tensor(2, &self.query_half);
        tensor_pipeline(ctx, enc, "score_tensor64_qhalf").unwrap();
        grid(enc, 1, self.n, 1, 128);
        pipeline(ctx, enc, "softmax_rows").unwrap();
        grid(enc, H, self.n, 1, 32);
        tensor_pipeline(ctx, enc, "pv_tensor64").unwrap();
        grid(enc, 1, self.n, 1, 128);
        let pso = pipeline(ctx, enc, "normalize_rows").unwrap();
        dispatch(enc, &pso, self.n * H * M);
    }

    /// Both Q and K/V half mirrors.
    fn encode_tensor64_qkvhalf(&self, ctx: &MetalContext, enc: &KernelEncoder) {
        self.bind_half(enc);
        enc.set_tensor(2, &self.query_half);
        tensor_pipeline(ctx, enc, "score_tensor64_qhalf").unwrap();
        grid(enc, 1, self.n, 1, 128);
        pipeline(ctx, enc, "softmax_rows").unwrap();
        grid(enc, H, self.n, 1, 32);
        tensor_pipeline(ctx, enc, "pv_tensor64_half").unwrap();
        grid(enc, 1, self.n, 1, 128);
        let pso = pipeline(ctx, enc, "normalize_rows").unwrap();
        dispatch(enc, &pso, self.n * H * M);
    }
}

#[test]
fn mma_sink_tile_matches_scalar_sink() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    for (k, n, ragged, first) in cases() {
        let f = Fixture::new(&ctx, k, n, ragged, first);
        f.checked(&ctx, Scalar, Scope::Sink);
        let scalar = f32_slice(&f.output).unwrap().to_vec();
        let scalar_scores = f32_slice(&f.scores).unwrap().to_vec();
        execute(&ctx, |enc| f.encode_mma(&ctx, enc));
        assert_eq!(ids(&f.status, 1), [0], "k={k} n={n} ragged={ragged}");
        // Half-operand products: scores (post-softmax exp values and
        // denominators) and outputs agree with the F32 scalar sink at a
        // half-rounding tolerance.
        let mma_scores = f32_slice(&f.scores).unwrap();
        for q in 0..n {
            let count = f.count(q);
            for h in 0..H {
                let base = (q * H + h) * SCORE_STRIDE;
                close_half(
                    &mma_scores[base..base + count],
                    &scalar_scores[base..base + count],
                );
                close_half(
                    &mma_scores[base + SCORE_STRIDE - 1..base + SCORE_STRIDE],
                    &scalar_scores[base + SCORE_STRIDE - 1..base + SCORE_STRIDE],
                );
            }
        }
        close_half(f32_slice(&f.output).unwrap(), &scalar);
        f.check_guards();
    }
}

/// The tensor tile against the scalar sink (half tolerance) and against
/// the MMA tile it replaces (same half operands, F32 sums, different
/// reduction order), with the MMA tile's visibility and finite guards.
#[test]
fn tensor_sink_tile_matches_mma_sink() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    if !metal::moe_grouped_tensor_available(&ctx) {
        eprintln!("tensor tier unavailable: {:?}", ctx.tensor_ops_status());
        return;
    }
    for (k, n, ragged, first) in cases() {
        let f = Fixture::new(&ctx, k, n, ragged, first);
        f.checked(&ctx, Scalar, Scope::Sink);
        let scalar = f32_slice(&f.output).unwrap().to_vec();
        execute(&ctx, |enc| f.encode_mma(&ctx, enc));
        let mma = f32_slice(&f.output).unwrap().to_vec();
        let mma_scores = f32_slice(&f.scores).unwrap().to_vec();
        execute(&ctx, |enc| f.encode_tensor(&ctx, enc));
        assert_eq!(ids(&f.status, 1), [0], "k={k} n={n} ragged={ragged}");
        let tensor_scores = f32_slice(&f.scores).unwrap();
        for q in 0..n {
            let count = f.count(q);
            for h in 0..H {
                let base = (q * H + h) * SCORE_STRIDE;
                close_tile(
                    &tensor_scores[base..base + count],
                    &mma_scores[base..base + count],
                );
                close_tile(
                    &tensor_scores[base + SCORE_STRIDE - 1..base + SCORE_STRIDE],
                    &mma_scores[base + SCORE_STRIDE - 1..base + SCORE_STRIDE],
                );
            }
        }
        let got = f32_slice(&f.output).unwrap();
        close_tile(got, &mma);
        close_half(got, &scalar);
        f.check_guards();
    }
}

#[test]
fn tensor32_sink_tile_matches_tensor_sink() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    if !metal::moe_grouped_tensor_available(&ctx) {
        eprintln!("tensor tier unavailable: {:?}", ctx.tensor_ops_status());
        return;
    }
    for (k, n, ragged, first) in cases() {
        let f = Fixture::new(&ctx, k, n, ragged, first);
        execute(&ctx, |enc| f.encode_tensor(&ctx, enc));
        let base_scores = f32_slice(&f.scores).unwrap().to_vec();
        let base_out = f32_slice(&f.output).unwrap().to_vec();
        execute(&ctx, |enc| f.encode_tensor32(&ctx, enc));
        assert_eq!(ids(&f.status, 1), [0], "k={k} n={n} ragged={ragged}");
        let got_scores = f32_slice(&f.scores).unwrap();
        for q in 0..n {
            let count = f.count(q);
            for h in 0..H {
                let base = (q * H + h) * SCORE_STRIDE;
                close_tile(
                    &got_scores[base..base + count],
                    &base_scores[base..base + count],
                );
                close_tile(
                    &got_scores[base + SCORE_STRIDE - 1..base + SCORE_STRIDE],
                    &base_scores[base + SCORE_STRIDE - 1..base + SCORE_STRIDE],
                );
            }
        }
        close_tile(f32_slice(&f.output).unwrap(), &base_out);
        f.check_guards();
    }
}

#[test]
fn tensor64_sink_tile_matches_tensor32_sink() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    if !metal::moe_grouped_tensor_available(&ctx) {
        eprintln!("tensor tier unavailable: {:?}", ctx.tensor_ops_status());
        return;
    }
    for (k, n, ragged, first) in cases() {
        let f = Fixture::new(&ctx, k, n, ragged, first);
        execute(&ctx, |enc| f.encode_tensor32(&ctx, enc));
        let base_scores = f32_slice(&f.scores).unwrap().to_vec();
        let base_out = f32_slice(&f.output).unwrap().to_vec();
        execute(&ctx, |enc| f.encode_tensor64(&ctx, enc));
        assert_eq!(ids(&f.status, 1), [0], "k={k} n={n} ragged={ragged}");
        let got_scores = f32_slice(&f.scores).unwrap();
        for q in 0..n {
            let count = f.count(q);
            for h in 0..H {
                let base = (q * H + h) * SCORE_STRIDE;
                close_tile(
                    &got_scores[base..base + count],
                    &base_scores[base..base + count],
                );
                close_tile(
                    &got_scores[base + SCORE_STRIDE - 1..base + SCORE_STRIDE],
                    &base_scores[base + SCORE_STRIDE - 1..base + SCORE_STRIDE],
                );
            }
        }
        close_tile(f32_slice(&f.output).unwrap(), &base_out);
        f.check_guards();
    }
}

/// Two half-operand F32-sum tiles differing only in reduction order.
fn close_tile(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= 1e-4 + 2e-3 * b.abs(),
            "{i}: {a} != {b}"
        );
    }
}

#[test]
fn mma_sink_tile_invalid_visibility_and_finite_guards() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let mut f = Fixture::new(&ctx, 512, 8, true, 0);
    execute(&ctx, |enc| f.encode_mma(&ctx, enc));
    let valid = f32_slice(&f.output).unwrap().to_vec();
    let mut selected: Vec<u32> = ids(&f.selected, f.topk * f.n)
        .into_iter()
        .map(|id| id as u32)
        .collect();
    selected[f.topk] = ((f.start + 2) / f.ratio) as u32; // Populated but not visible to query 1.
    f.selected = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&selected),
        vec![f.topk as u64, f.n as u64],
        GgmlType::I32,
    )
    .unwrap();
    execute(&ctx, |enc| f.encode_mma(&ctx, enc));
    assert_eq!(ids(&f.status, 1), [2]);
    let got = f32_slice(&f.output).unwrap();
    for h in 0..H {
        for q in 0..f.n {
            let off = (h * f.n + q) * M;
            if q == 1 {
                assert!(got[off..off + M].iter().all(|&v| v == 0.0));
            } else {
                assert_eq!(got[off..off + M], valid[off..off + M]);
            }
        }
    }
    let poison = upload(&ctx, &[f32::NAN], vec![1]).unwrap();
    execute(&ctx, |enc| {
        copy(
            &ctx,
            enc,
            &poison,
            &f.history.view_subrange(M as u64, vec![1]),
            1,
        )
        .unwrap()
    });
    execute(&ctx, |enc| f.encode_mma(&ctx, enc));
    assert_eq!(ids(&f.status, 1), [6]);
    f.check_guards();
}

/// Opt-in timing of each sink-tile dispatch, old path versus MMA, at the
/// production tile shape. `QWEN_V41_SINK_TIMING=1`; prints, asserts nothing.
#[test]
fn mma_sink_tile_dispatch_timing() {
    if std::env::var_os("QWEN_V41_SINK_TIMING").is_none() {
        return;
    }
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = MetalContext::new().expect("Metal");
    let timed = |label: &str, f: &Fixture, encode: &dyn Fn(&KernelEncoder)| {
        let mut us = Vec::new();
        for _ in 0..12 {
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            encode(&enc);
            enc.end();
            metal::commit_and_wait(&command).unwrap();
            us.push((command.GPUEndTime() - command.GPUStartTime()) * 1e6);
        }
        us.sort_by(f64::total_cmp);
        eprintln!(
            "sink_timing k={} n={} {label:<26} median {:8.1} us  min {:8.1}",
            f.k, f.n, us[6], us[0]
        );
    };
    for (k, n) in [
        (640usize, 32usize),
        (512, 32),
        (256, 32),
        (640, 64),
        (640, 128),
        (640, 256),
        (640, 512),
    ] {
        let f = Fixture::new(&ctx, k, n, k == 640, 0);
        // Heat.
        for _ in 0..20 {
            execute(&ctx, |enc| f.encode_mma(&ctx, enc));
        }
        timed("old score_rows", &f, &|enc| {
            f.bind(enc);
            pipeline(&ctx, enc, "score_rows").unwrap();
            grid(enc, f.count(f.n - 1).div_ceil(8), H, f.n, 256);
        });
        timed("mma score", &f, &|enc| {
            f.bind(enc);
            pipeline(&ctx, enc, "score_mma").unwrap();
            grid(enc, H / 16, f.n, 1, 128);
        });
        timed("softmax_rows", &f, &|enc| {
            f.bind(enc);
            pipeline(&ctx, enc, "softmax_rows").unwrap();
            grid(enc, H, f.n, 1, 32);
        });
        timed("old gather+pv(F32 MMA)+finite", &f, &|enc| {
            f.encode(&ctx, enc, Production, Scope::PackPvFinite);
        });
        timed("mma pv", &f, &|enc| {
            f.bind(enc);
            pipeline(&ctx, enc, "pv_mma").unwrap();
            grid(enc, H / 16, f.n, 1, 256);
        });
        timed("normalize_rows", &f, &|enc| {
            f.bind(enc);
            let pso = pipeline(&ctx, enc, "normalize_rows").unwrap();
            dispatch(enc, &pso, f.n * H * M);
        });
        timed("full old sink", &f, &|enc| {
            f.encode(&ctx, enc, Production, Scope::Sink)
        });
        timed("full mma sink", &f, &|enc| f.encode_mma(&ctx, enc));
        if metal::moe_grouped_tensor_available(&ctx) {
            timed("tensor score", &f, &|enc| {
                f.bind(enc);
                tensor_pipeline(&ctx, enc, "score_tensor").unwrap();
                grid(enc, H / 16, f.n, 1, 128);
            });
            timed("tensor32 score", &f, &|enc| {
                f.bind(enc);
                tensor_pipeline(&ctx, enc, "score_tensor32").unwrap();
                grid(enc, H / 16, f.n, 1, 128);
            });
            timed("full tensor sink", &f, &|enc| f.encode_tensor(&ctx, enc));
            timed("full tensor32 sink", &f, &|enc| {
                f.encode_tensor32(&ctx, enc)
            });
        }
    }
}
