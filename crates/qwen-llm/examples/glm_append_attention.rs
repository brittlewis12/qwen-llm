//! Reproducible selected-attention timing, F64 diagnostics and buffer guards.
//! Run under scripts/bench/with-metal-lease; no model or ignored tests.
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLSize};
use qwen_llm::metal::*;

fn emit(v: serde_json::Value) {
    println!("{v}");
}
fn read<T: Copy>(t: &MetalTensor) -> &[T] {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast(),
            t.n_bytes() as usize / std::mem::size_of::<T>(),
        )
    }
}
fn tensor<T: bytemuck::Pod>(
    c: &MetalContext,
    v: &[T],
    s: Vec<u64>,
    dt: qwen_llm::tensor::GgmlType,
) -> MetalTensor {
    MetalTensor::from_bytes(c, bytemuck::cast_slice(v), s, dt).unwrap()
}
fn value(i: usize, salt: u64) -> f32 {
    let mut x = (i as u64).wrapping_add(salt);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((x as u32 as f64 / 4294967296. - 0.5) * 0.6) as f32
}
fn gpu(c: &MetalContext, f: impl FnOnce(&KernelEncoder)) -> f64 {
    let cmd = c.queue.commandBuffer().unwrap();
    let e = KernelEncoder::begin(&cmd);
    f(&e);
    e.end();
    cmd.commit();
    wait_completed(&cmd).unwrap();
    (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3
}
fn sz(w: usize, h: usize, d: usize) -> MTLSize {
    MTLSize {
        width: w,
        height: h,
        depth: d,
    }
}
struct Op {
    q: MetalTensor,
    k: MetalTensor,
    ids: MetalTensor,
    counts: MetalTensor,
    visible: MetalTensor,
    sinks: MetalTensor,
    ys: Vec<MetalTensor>,
    scores: MetalTensor,
    pv: MetalTensor,
    ps: MetalTensor,
    shape: SelectedAttentionShape,
}
impl Op {
    fn new(c: &MetalContext, n: usize) -> Self {
        use qwen_llm::tensor::GgmlType::*;
        let (slots, cap, total) = (2051usize, 65536usize, n + 8);
        let f = |v: &[f32], s| tensor(c, v, s, F32);
        let q = f(
            &(0..total * 32768).map(|i| value(i, 19)).collect::<Vec<_>>(),
            vec![32768, total as u64],
        );
        let k = tensor(
            c,
            &(0..cap * 512)
                .map(|i| half::f16::from_f32(value(i, 31)))
                .collect::<Vec<_>>(),
            vec![512, cap as u64],
            F16,
        );
        let ids = tensor(
            c,
            &(0..n * slots)
                .map(|i| {
                    if i % 13 == 0 {
                        -1
                    } else if i % 17 == 0 {
                        cap as i32
                    } else {
                        ((i / slots * 997 + i % slots * 17) % cap) as i32
                    }
                })
                .collect::<Vec<_>>(),
            vec![slots as u64, n as u64],
            I32,
        );
        let counts = tensor(
            c,
            &(0..n)
                .map(|r| if r == 1 { 0 } else { (slots - r % 3) as i32 })
                .collect::<Vec<_>>(),
            vec![n as u64],
            I32,
        );
        let visible = tensor(
            c,
            &(0..n)
                .map(|r| if r == 0 { cap as i32 / 2 } else { cap as i32 })
                .collect::<Vec<_>>(),
            vec![n as u64],
            I32,
        );
        let sinks = f(
            &(0..64)
                .map(|h| {
                    if h == 0 {
                        -1e30
                    } else if h == 1 {
                        20.
                    } else {
                        (h % 3) as f32 - 1.
                    }
                })
                .collect::<Vec<_>>(),
            vec![64],
        );
        let ys = (0..3)
            .map(|_| f(&vec![f32::NAN; total * 32768], vec![32768, total as u64]))
            .collect();
        let scores = f(&vec![f32::NAN; 2176 * 64 * n], vec![2176, 64, n as u64]);
        let units = 16 * 64 * 17;
        Self {
            q,
            k,
            ids,
            counts,
            visible,
            sinks,
            ys,
            scores,
            pv: MetalTensor::zeros_f32(c, vec![512, units]).unwrap(),
            ps: MetalTensor::zeros_f32(c, vec![2, units]).unwrap(),
            shape: SelectedAttentionShape {
                head_count: 64,
                query_count: n,
                query_token_offset: 4,
                token_count: total,
                chunk_start_position: 65536,
                window: 0,
                raw_cache_is_chunk: false,
                selected_slots: slots,
                compressed_capacity: cap,
                scale: 0.0625,
                direct: true,
            },
        }
    }
    fn buffers(&self, m: usize) -> SelectedAttention<'_> {
        SelectedAttention {
            queries: &self.q,
            raw_cache: &self.k,
            raw_cache_before_chunk: &self.k,
            compressed_cache: &self.k,
            selected_ids: &self.ids,
            selected_counts: &self.counts,
            visible_counts: &self.visible,
            sinks: &self.sinks,
            output: &self.ys[m],
        }
    }
    fn run(&self, c: &MetalContext, m: usize) -> f64 {
        let g = self.shape;
        gpu(c, |e| match m {
            0 => {
                for start in (0..g.query_count).step_by(16) {
                    let n = 16.min(g.query_count - start);
                    let ids = self.ids.view_subrange(
                        (start * g.selected_slots) as u64,
                        vec![g.selected_slots as u64, n as u64],
                    );
                    let counts = self.counts.view_subrange(start as u64, vec![n as u64]);
                    let visible = self.visible.view_subrange(start as u64, vec![n as u64]);
                    let b = SelectedAttention {
                        selected_ids: &ids,
                        selected_counts: &counts,
                        visible_counts: &visible,
                        ..self.buffers(m)
                    };
                    let u = n * 64 * 17;
                    encode_online_selected_attention_split_f16(
                        c,
                        e,
                        &b,
                        SelectedAttentionShape {
                            query_count: n,
                            query_token_offset: g.query_token_offset + start,
                            ..g
                        },
                        &SelectedAttentionPartials {
                            values: &self.pv.view_subrange(0, vec![512, u as u64]),
                            stats: &self.ps.view_subrange(0, vec![2, u as u64]),
                        },
                    )
                    .unwrap();
                }
            }
            1 => assert!(
                try_encode_selected_attention_tensor_f32(c, e, &self.buffers(m), g, &self.scores)
                    .unwrap()
            ),
            2 => assert!(
                try_encode_selected_attention_tensor_f16(
                    c,
                    e,
                    &self.buffers(m),
                    g,
                    &self
                        .scores
                        .view_subrange(0, vec![2080, 64, g.query_count as u64])
                )
                .unwrap()
            ),
            _ => unreachable!(),
        })
    }
    fn check(&self) {
        let g = self.shape;
        for (m, y) in self.ys.iter().enumerate() {
            let a = read::<f32>(y);
            assert!(a[..g.query_token_offset * 32768].iter().all(|v| v.is_nan()));
            assert!(
                a[(g.query_token_offset + g.query_count) * 32768..]
                    .iter()
                    .all(|v| v.is_nan())
            );
            assert!(
                a[g.query_token_offset * 32768..(g.query_token_offset + g.query_count) * 32768]
                    .iter()
                    .all(|v| v.is_finite()),
                "method {m}"
            );
        }
        let q = read::<f32>(&self.q);
        let k = read::<half::f16>(&self.k);
        let ids = read::<i32>(&self.ids);
        let counts = read::<i32>(&self.counts);
        let visible = read::<i32>(&self.visible);
        let sinks = read::<f32>(&self.sinks);
        for r in [0, g.query_count / 2, g.query_count - 1] {
            for h in [0, 7, 31, 63] {
                let mut values = Vec::new();
                let mut max = sinks[h] as f64;
                for i in 0..counts[r].max(0) as usize {
                    let id = ids[r * g.selected_slots + i];
                    if id < 0 || id >= visible[r] {
                        continue;
                    }
                    let mut s = 0.;
                    for d in 0..512 {
                        s += q[((g.query_token_offset + r) * 64 + h) * 512 + d] as f64
                            * k[id as usize * 512 + d].to_f64();
                    }
                    s *= g.scale as f64;
                    max = max.max(s);
                    values.push((id as usize, s));
                }
                let den = (sinks[h] as f64 - max).exp()
                    + values.iter().map(|(_, s)| (s - max).exp()).sum::<f64>();
                let mut truth = vec![0.; 512];
                for (id, s) in values {
                    let p = (s - max).exp() / den;
                    for d in 0..512 {
                        truth[d] += p * k[id * 512 + d].to_f64();
                    }
                }
                let mut squared = vec![0.; self.ys.len()];
                let mut absolute = vec![0f64; self.ys.len()];
                for (m, y) in self.ys.iter().enumerate() {
                    let got = read::<f32>(y);
                    for d in 0..512 {
                        let diff =
                            got[((g.query_token_offset + r) * 64 + h) * 512 + d] as f64 - truth[d];
                        squared[m] += diff * diff;
                        absolute[m] = absolute[m].max(diff.abs());
                    }
                }
                emit(
                    serde_json::json!({"kind":"f64","queries":g.query_count,"row":r,"head":h,"squared":squared,"max_abs":absolute,"elements":512}),
                );
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    let c = MetalContext::new()?;
    emit(
        serde_json::json!({"kind":"device","facts":c.device_facts(),"methods":["incumbent-f32-split","candidate-f32-tensor","accepted-aligned-half-replay"],"f64":"absolute diagnostic, not a ratio gate"}),
    );
    for n in [16, 32, 64] {
        let f = Op::new(&c, n);
        let original_q = blake3::hash(bytemuck::cast_slice(read::<f32>(&f.q)));
        let original_cache = blake3::hash(bytemuck::cast_slice(read::<half::f16>(&f.k)));
        for _ in 0..4 {
            for m in 0..3 {
                f.run(&c, m);
            }
        }
        for rep in 0..8 {
            for m in if rep % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] } {
                emit(
                    serde_json::json!({"kind":"operator","queries":n,"method":m,"rep":rep,"gpu_ms":f.run(&c,m)}),
                );
            }
        }
        f.run(&c, 1);
        f.check();
        assert_eq!(
            original_q,
            blake3::hash(bytemuck::cast_slice(read::<f32>(&f.q)))
        );
        assert_eq!(
            original_cache,
            blake3::hash(bytemuck::cast_slice(read::<half::f16>(&f.k)))
        );
        let s = read::<f32>(&f.scores);
        for r in 0..n {
            for h in 0..64 {
                assert!(
                    s[(r * 64 + h) * 2176 + 2051..(r * 64 + h + 1) * 2176]
                        .iter()
                        .all(|&v| v == 0.)
                );
            }
        }
        let y = read::<f32>(&f.ys[1]);
        assert!(
            y[(f.shape.query_token_offset + 1) * 32768..(f.shape.query_token_offset + 2) * 32768]
                .iter()
                .all(|&v| v == 0.)
        );
        gpu(&c, |enc| {
            let bad = SelectedAttention {
                output: &f.q,
                ..f.buffers(1)
            };
            assert!(
                try_encode_selected_attention_tensor_f32(&c, enc, &bad, f.shape, &f.scores)
                    .is_err()
            );
            assert!(
                try_encode_selected_attention_tensor_f32(
                    &c,
                    enc,
                    &f.buffers(1),
                    SelectedAttentionShape {
                        head_count: 32,
                        ..f.shape
                    },
                    &f.scores
                )
                .is_ok_and(|v| !v)
            );
        });
    }
    Ok(())
}
