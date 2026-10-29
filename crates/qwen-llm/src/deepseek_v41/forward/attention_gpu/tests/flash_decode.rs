//! Fused decode sink attention (`attention_flash_*` + `attention_flash_combine`)
//! against the production chain (`sink_passes`: score_keys -> softmax ->
//! output or value matrix) and an F64 reference, plus the opt-in kill-test
//! timing (`QWEN_V41_FLASH_TIMING`, under the Metal lease).

use super::*;

/// Any value prints the chain vs fused decode attention kill-test timing.
const FLASH_TIMING_ENV: &str = "QWEN_V41_FLASH_TIMING";

const WINDOW: usize = 128;
const POSITION: usize = 40960;

/// Attended inverse RoPE after the attention: none, the production `rope`
/// dispatch, or folded into `attention_flash_combine_rope` (fused only).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Rope {
    None,
    Separate,
    Fused,
}
const TOPK: usize = 512;
/// The production partial kernels (forced for every row length here; the
/// decode path picks one with `flash_kernel`).
const CONFIGS: [(&str, usize, usize); 3] = [
    ("flash_h8_k32", 8, 32),
    ("flash_h8_k32_d512", 8, 32),
    ("flash2_h4_k16", 4, 16),
];

fn make(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            (((i * 2_654_435_761usize + seed * 97) % 1_000_003) as f32 / 250_000.0 - 2.0) * scale
        })
        .collect()
}

/// One decode row's inputs: query, raw ring, compressed rows and selected IDs.
struct Row {
    heads: usize,
    dim: usize,
    query: MetalTensor,
    raw: MetalTensor,
    compressed: MetalTensor,
    ids: MetalTensor,
    sink: MetalTensor,
    ring_start: usize,
    length: u32,
    host: (Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<f32>),
}

impl Row {
    fn new(
        ctx: &MetalContext,
        heads: usize,
        dim: usize,
        rows: usize,
        length: u32,
        seed: usize,
    ) -> Self {
        let up = |v: &[f32]| upload(ctx, v, vec![v.len() as u64]).unwrap();
        let query = make(heads * dim, 3 + seed, 1.0);
        let raw = make(WINDOW * dim, 5 + seed, 1.0);
        let compressed = make(rows * dim, 7 + seed, 1.0);
        let ids: Vec<u32> = (0..TOPK as u32)
            .map(|i| {
                (i.wrapping_mul(2_654_435_761)
                    .wrapping_add(seed as u32 * 7919))
                    % length
            })
            .collect();
        let sink: Vec<f32> = (0..heads).map(|h| (h % 7) as f32 * 0.5 - 1.0).collect();
        Self {
            heads,
            dim,
            query: up(&query),
            raw: up(&raw),
            compressed: up(&compressed),
            ids: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&ids),
                vec![TOPK as u64],
                GgmlType::I32,
            )
            .unwrap(),
            sink: up(&sink),
            ring_start: 17 + seed % 50,
            length,
            host: (query, raw, compressed, ids, sink),
        }
    }

    fn bind(
        &self,
        enc: &KernelEncoder,
        nr: usize,
        nk: usize,
        scores: &MetalTensor,
        out: &MetalTensor,
        status: &MetalTensor,
    ) {
        enc.set_bytes(
            0,
            &[
                self.dim as u32,
                nr as u32,
                self.ring_start as u32,
                WINDOW as u32,
            ],
        );
        enc.set_bytes(1, &[nk as u32, (WINDOW + TOPK + 1) as u32]);
        enc.set_tensor(2, &self.query);
        enc.set_tensor(3, &self.raw);
        enc.set_tensor(4, &self.compressed);
        enc.set_tensor(5, &self.ids);
        enc.set_tensor(6, &self.sink);
        enc.set_tensor(7, scores);
        enc.set_tensor(8, out);
        enc.set_bytes(9, &self.length);
        enc.set_tensor(10, status);
    }

    /// F64 sink attention of the row (invalid IDs give zeros, as the chain).
    fn reference(&self, nr: usize, nk: usize) -> Vec<f64> {
        let (q, raw, compressed, ids, sink) = &self.host;
        let (heads, dim) = (self.heads, self.dim);
        if ids[..nk].iter().any(|&id| id >= self.length) {
            return vec![0.0; heads * dim];
        }
        let key = |i: usize| -> &[f32] {
            if i < nr {
                let r = (self.ring_start + i) % WINDOW;
                &raw[r * dim..][..dim]
            } else {
                &compressed[ids[i - nr] as usize * dim..][..dim]
            }
        };
        let mut out = vec![0.0f64; heads * dim];
        for h in 0..heads {
            let qh = &q[h * dim..][..dim];
            let s: Vec<f64> = (0..nr + nk)
                .map(|i| {
                    qh.iter()
                        .zip(key(i))
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum::<f64>()
                        / (dim as f64).sqrt()
                })
                .collect();
            let m = s.iter().cloned().fold(f64::from(sink[h]), f64::max);
            let e: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
            let denom = e.iter().sum::<f64>() + (f64::from(sink[h]) - m).exp();
            for (i, w) in e.iter().enumerate() {
                for (d, &v) in key(i).iter().enumerate() {
                    out[h * dim + d] += w / denom * f64::from(v);
                }
            }
        }
        out
    }
}

struct Buffers {
    scores: MetalTensor,
    out: MetalTensor,
    status: MetalTensor,
    matrix: ValueMatrix,
    flash: Vec<FlashScratch>,
    freq: MetalTensor,
    pairs: usize,
}

impl Buffers {
    fn new(ctx: &MetalContext, heads: usize, dim: usize) -> Self {
        let padded = (WINDOW + TOPK).div_ceil(32) * 32;
        Self {
            scores: tensor(ctx, heads * (WINDOW + TOPK + 1), GgmlType::F32).unwrap(),
            out: tensor(ctx, heads * dim, GgmlType::F32).unwrap(),
            status: tensor(ctx, 1, GgmlType::I32).unwrap(),
            matrix: ValueMatrix {
                values: tensor(ctx, padded * dim, GgmlType::F32).unwrap(),
                probabilities: tensor(ctx, padded * heads, GgmlType::F32).unwrap(),
            },
            flash: CONFIGS
                .iter()
                .map(|&(_, _, kb)| {
                    FlashScratch::with_splits(ctx, heads, dim, (WINDOW + TOPK).div_ceil(kb))
                        .unwrap()
                })
                .collect(),
            freq: upload(
                ctx,
                &(0..32)
                    .map(|j| 10000f32.powf(-(j as f32) / 32.0))
                    .collect::<Vec<_>>(),
                vec![32],
            )
            .unwrap(),
            pairs: 32.min(dim / 4),
        }
    }
}

/// `None`: the production chain; `Some(c)`: CONFIGS[c] fused.
fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    row: &Row,
    b: &Buffers,
    nr: usize,
    nk: usize,
    config: Option<usize>,
    rope_mode: Rope,
) {
    row.bind(enc, nr, nk, &b.scores, &b.out, &b.status);
    match config {
        None => {
            let matrix = (row.heads >= 8 && row.dim.is_multiple_of(32))
                .then_some((&b.matrix, &b.out, &b.status));
            sink_passes(ctx, enc, row.heads, row.dim, nr + nk, matrix).unwrap()
        }
        Some(c) => flash_passes(
            ctx,
            enc,
            row.heads,
            row.dim,
            nr + nk,
            &b.flash[c],
            CONFIGS[c],
            (rope_mode == Rope::Fused).then_some((&b.freq, b.pairs, POSITION)),
        )
        .unwrap(),
    }
    if rope_mode == Rope::Separate {
        rope(
            ctx, enc, &b.out, &b.freq, row.dim, row.heads, b.pairs, POSITION, true,
        )
        .unwrap();
    }
}

fn run(
    ctx: &MetalContext,
    row: &Row,
    b: &Buffers,
    nr: usize,
    nk: usize,
    config: Option<usize>,
    rope_mode: Rope,
) -> (Vec<f32>, u32) {
    let nan = vec![f32::NAN; row.heads * row.dim];
    b.out.buffer.contents();
    unsafe {
        std::ptr::copy_nonoverlapping(
            nan.as_ptr(),
            b.out.buffer.contents().as_ptr() as *mut f32,
            nan.len(),
        );
        *(b.status.buffer.contents().as_ptr() as *mut u32) = 0;
    }
    execute(ctx, |enc| {
        encode(ctx, enc, row, b, nr, nk, config, rope_mode)
    });
    let status = unsafe { *(b.status.buffer.contents().as_ptr() as *const u32) };
    (f32_slice(&b.out).unwrap().to_vec(), status)
}

/// Kernels compiled for a fixed head dimension run only at it.
fn usable(c: usize, dim: usize) -> bool {
    !CONFIGS[c].0.contains("_d512") || dim == 512
}

/// The decode path's choice for a row of `count` keys, as a CONFIGS index.
fn production(dim: usize, count: usize) -> usize {
    let (name, _, _) = flash_kernel(dim, count).unwrap();
    CONFIGS.iter().position(|c| c.0 == name).unwrap()
}

/// Max |a - b| over max |b|; panics on non-finite or mismatched inputs.
fn rel(a: &[f32], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    crate::compare::assert_max_abs_diff_f64("flash decode relative error", a, b) / scale
}

/// Every fused configuration matches the chain and the F64 reference to
/// F32 rounding at decode geometry and toy shapes (raw ring wraparound, raw
/// only, partial last split), invalid IDs zero the row and set status bit 2,
/// a non-finite query sets status bit 4.
#[test]
fn flash_decode_matches_chain_and_f64() {
    let ctx = MetalContext::new().unwrap();
    for (heads, dim) in [(64usize, 512usize), (3, 32), (12, 96)] {
        let row = Row::new(&ctx, heads, dim, 2048, 1500, 0);
        let b = Buffers::new(&ctx, heads, dim);
        for (nr, nk) in [
            (1, 0),
            (5, 7),
            (17, 16),
            (128, 0),
            (128, 129),
            (128, 256),
            (128, 512),
        ] {
            let reference = row.reference(nr, nk);
            let (chain, chain_status) = run(&ctx, &row, &b, nr, nk, None, Rope::None);
            let (chain_rope, _) = run(&ctx, &row, &b, nr, nk, None, Rope::Separate);
            let chain_rope64: Vec<f64> = chain_rope.iter().map(|&v| f64::from(v)).collect();
            let chain_err = rel(&chain, &reference);
            assert!(
                chain_err < 1e-5,
                "chain {heads}x{dim} {nr}+{nk}: {chain_err:.2e}"
            );
            for c in (0..CONFIGS.len()).filter(|&c| usable(c, dim)) {
                let (fused, status) = run(&ctx, &row, &b, nr, nk, Some(c), Rope::None);
                // RoPE folded into the combine vs the separate dispatch: the
                // same expression, F32 rounding (fast-math trig/contraction).
                let (fused_rope, rope_status) = run(&ctx, &row, &b, nr, nk, Some(c), Rope::Fused);
                let (separate, _) = run(&ctx, &row, &b, nr, nk, Some(c), Rope::Separate);
                let separate64: Vec<f64> = separate.iter().map(|&v| f64::from(v)).collect();
                let rope_err = rel(&fused_rope, &separate64);
                assert!(rope_err < 1e-6, "rope {:?}: {rope_err:.2e}", CONFIGS[c]);
                assert_eq!(rope_status, status);
                let rope_chain = rel(&fused_rope, &chain_rope64);
                assert!(
                    rope_chain < 1e-5,
                    "{heads}x{dim} {nr}+{nk} {:?}: rope vs chain {rope_chain:.2e} (unroped {:.2e})",
                    CONFIGS[c],
                    rel(
                        &fused,
                        &chain.iter().map(|&v| f64::from(v)).collect::<Vec<_>>()
                    )
                );
                let err = rel(&fused, &reference);
                let chain64: Vec<f64> = chain.iter().map(|&v| f64::from(v)).collect();
                let vs_chain = rel(&fused, &chain64);
                assert_eq!(
                    status, chain_status,
                    "{heads}x{dim} {nr}+{nk} {:?}",
                    CONFIGS[c]
                );
                assert!(
                    err < 1e-5 && vs_chain < 1e-5,
                    "{heads}x{dim} {nr}+{nk} {:?}: f64 {err:.2e} chain {vs_chain:.2e}",
                    CONFIGS[c]
                );
                if heads == 64 && c == production(dim, nr + nk) {
                    eprintln!(
                        "flash {heads}x{dim} {nr}+{nk}: max err / max|out| vs F64 chain {chain_err:.2e} fused {err:.2e}; fused vs chain {vs_chain:.2e}"
                    );
                }
            }
        }
        // Invalid selected ID: zero row, status bit 2, in both.
        let bad = Row::new(&ctx, heads, dim, 2048, 1500, 1);
        let mut ids = bad.host.3.clone();
        ids[100] = 1500;
        let bad = Row {
            ids: MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&ids),
                vec![TOPK as u64],
                GgmlType::I32,
            )
            .unwrap(),
            host: (
                bad.host.0.clone(),
                bad.host.1.clone(),
                bad.host.2.clone(),
                ids,
                bad.host.4.clone(),
            ),
            ..bad
        };
        for config in
            std::iter::once(None).chain((0..CONFIGS.len()).filter(|&c| usable(c, dim)).map(Some))
        {
            let (out, status) = run(&ctx, &bad, &b, 128, 512, config, Rope::None);
            assert!(out.iter().all(|&v| v == 0.0), "{config:?}");
            assert_eq!(status & 2, 2, "{config:?}");
            assert_eq!(status & 4, 0, "{config:?}");
        }
        // Non-finite query: status bit 4 in both.
        let mut q = row.host.0.clone();
        q[dim + 3] = f32::INFINITY;
        let inf = Row {
            query: upload(&ctx, &q, vec![q.len() as u64]).unwrap(),
            ..Row::new(&ctx, heads, dim, 2048, 1500, 0)
        };
        for config in
            std::iter::once(None).chain((0..CONFIGS.len()).filter(|&c| usable(c, dim)).map(Some))
        {
            let (_, status) = run(&ctx, &inf, &b, 128, 256, config, Rope::None);
            assert_eq!(status & 4, 4, "{config:?}");
        }
    }
}

fn timed(ctx: &MetalContext, iters: usize, f: &dyn Fn(&KernelEncoder, usize)) -> f64 {
    let mut samples = Vec::new();
    for round in 0..8 {
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        for i in 0..iters {
            f(&enc, i);
        }
        enc.end();
        metal::commit_and_wait(&command).unwrap();
        if round > 1 {
            samples.push((command.GPUEndTime() - command.GPUStartTime()) * 1e6 / iters as f64);
        }
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// Kill test: chain vs fused per decode row at 64 x 512, back to back.
/// Warm: one row repeated (keys cache-resident). Cold: 40 rows (one per
/// layer) rotated, each with its own query, raw ring and 512 selected rows
/// of a 64K-row compressed bank.
#[test]
fn flash_decode_kill_test_timing() {
    if std::env::var_os(FLASH_TIMING_ENV).is_none() {
        return;
    }
    let ctx = MetalContext::new().unwrap();
    let (heads, dim, layers) = (64usize, 512usize, 40usize);
    let rows: Vec<Row> = (0..layers)
        .map(|l| Row::new(&ctx, heads, dim, 65536, 65536, l + 1))
        .collect();
    let b = Buffers::new(&ctx, heads, dim);
    // Heat the GPU.
    let heat = std::time::Instant::now();
    while heat.elapsed() < std::time::Duration::from_millis(500) {
        timed(&ctx, 40, &|enc, i| {
            encode(&ctx, enc, &rows[i % layers], &b, 128, 512, None, Rope::None)
        });
    }
    for (nr, nk) in [(1, 0), (128, 0), (128, 256), (128, 512)] {
        for (label, rotate, chain_rope, fused_rope) in [
            ("warm", 1usize, Rope::None, Rope::None),
            ("cold", layers, Rope::None, Rope::None),
            ("cold+rope", layers, Rope::Separate, Rope::Fused),
        ] {
            let chain = timed(&ctx, 80, &|enc, i| {
                encode(&ctx, enc, &rows[i % rotate], &b, nr, nk, None, chain_rope)
            });
            let mut line = format!(
                "flash kill {nr}+{nk} {label} (decode uses {}): chain {chain:6.2} us",
                CONFIGS[production(dim, nr + nk)].0
            );
            for c in 0..CONFIGS.len() {
                let fused = timed(&ctx, 80, &|enc, i| {
                    encode(
                        &ctx,
                        enc,
                        &rows[i % rotate],
                        &b,
                        nr,
                        nk,
                        Some(c),
                        fused_rope,
                    )
                });
                line += &format!(" | {} {fused:6.2} us {:4.2}x", CONFIGS[c].0, chain / fused);
            }
            eprintln!("{line}");
        }
        // Where the fused time goes: each kernel alone (cold rows).
        let mut line = format!("flash parts {nr}+{nk} cold:");
        for c in 0..CONFIGS.len() {
            let (partial_name, hg, kb) = CONFIGS[c];
            let part = |enc: &KernelEncoder, i: usize, combine: bool| {
                let row = &rows[i % layers];
                row.bind(enc, nr, nk, &b.scores, &b.out, &b.status);
                enc.set_bytes(14, &(heads as u32));
                enc.set_tensor(15, &b.flash[c].scratch);
                enc.set_bytes(16, &[kb as u32, b.flash[c].splits as u32, 0, 0]);
                let (name, grid) = if combine {
                    ("flash_combine".to_string(), (dim.div_ceil(256), heads))
                } else {
                    (
                        partial_name.to_string(),
                        ((nr + nk).div_ceil(kb), heads.div_ceil(hg)),
                    )
                };
                pipeline(&ctx, enc, &name).unwrap();
                enc.dispatch(
                    MTLSize {
                        width: grid.0,
                        height: grid.1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
            };
            let partial = timed(&ctx, 80, &|enc, i| part(enc, i, false));
            let combine = timed(&ctx, 80, &|enc, i| part(enc, i, true));
            line += &format!(" | {partial_name} partial {partial:5.2} combine {combine:5.2} us");
        }
        eprintln!("{line}");
    }
}

/// Decode path A/B on the toy model (40 layers x 7 positions, one command):
/// the fused attention (`flash2_h4_k16` + `flash_combine_rope` per layer, no
/// chain or RoPE dispatch) matches the chain and the CPU oracle to F32
/// rounding, with the same status.
#[test]
fn flash_decode_toy_chain_matches_chain_and_oracle() {
    let ctx = MetalContext::new().unwrap();
    let data = super::fixture();
    let cpu: Vec<_> = (0..40).map(|id| super::cpu_weights(&data, id)).collect();
    let gpu: Vec<_> = cpu.iter().map(|w| super::resident(&ctx, w)).collect();
    let mut c = super::super::super::toy::config();
    c.hidden = 8;
    let run = |flash: bool| {
        FLASH_DECODE_OVERRIDE.with(|o| o.set(Some(flash)));
        let mut workspace = Workspace::new(&ctx, &c, 7, &gpu.iter().collect::<Vec<_>>()).unwrap();
        let outputs: Vec<_> = (0..280)
            .map(|_| tensor(&ctx, 8, GgmlType::F32).unwrap())
            .collect();
        metal::dispatch_census_begin();
        execute(&ctx, |enc| {
            for p in 0..7 {
                let mut input =
                    upload(&ctx, &data[&format!("p.{p}.l.0.attn_input")], vec![8]).unwrap();
                for id in 0..40 {
                    let output = workspace
                        .encode(&ctx, enc, id, p, &gpu[id], &input)
                        .unwrap();
                    copy(&ctx, enc, output, &outputs[p * 40 + id], 8).unwrap();
                    input = output.clone();
                }
            }
        });
        let census = metal::dispatch_census_take();
        FLASH_DECODE_OVERRIDE.with(|o| o.set(None));
        let count = |name: &str| {
            census
                .iter()
                .filter(|d| d.kernel == format!("kernel_deepseek_v41_attention_{name}"))
                .count()
        };
        let counts = [
            "flash2_h4_k16",
            "flash_combine_rope",
            "score_keys",
            "softmax",
            "rope",
        ]
        .map(count);
        let values: Vec<Vec<f32>> = outputs
            .iter()
            .map(|t| f32_slice(t).unwrap().to_vec())
            .collect();
        (values, counts, super::ids(workspace.status(), 1))
    };
    let (fused, fused_counts, fused_status) = run(true);
    let (chain, chain_counts, chain_status) = run(false);
    // Per decode row: fused = partial + combine_rope; chain = score_keys,
    // softmax, output and the inverse RoPE (query RoPE is shared).
    assert_eq!(fused_counts[..3], [280, 280, 0]);
    assert_eq!(chain_counts[..4], [0, 0, 280, 280]);
    assert_eq!(chain_counts[4], fused_counts[4] + 280);
    assert_eq!(fused_status, chain_status);
    assert_eq!(fused_status, [0]);
    let mut oracle = op::Session::default();
    for p in 0..7 {
        let mut input = data[&format!("p.{p}.l.0.attn_input")].clone();
        for id in 0..40 {
            input = oracle.attention(id, p, &input, &cpu[id]).unwrap();
            super::close(&fused[p * 40 + id], &input);
            super::close(&fused[p * 40 + id], &chain[p * 40 + id]);
        }
    }
}
