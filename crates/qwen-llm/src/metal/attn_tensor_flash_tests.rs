//! Retained contracts and numerics for the Apple10 Qwen tensor loop.
use super::*;
use crate::metal::context::acquire_metal_benchmark_lease;
use crate::metal::test_support::{cpu_matrix_attn_reference, metal_test_context};

fn gpu(tensor_required: bool, f: impl FnOnce(&MetalContext)) {
    let _lease = acquire_metal_benchmark_lease().expect("canonical correctness GPU lease");
    let Some(ctx) = metal_test_context() else {
        return;
    };
    if tensor_required && !ctx.tensor_ops_status().is_available() {
        eprintln!("tensor flash numerical check requires Apple10/Metal4; unsupported device");
        return;
    }
    f(&ctx);
}

struct Data {
    rows: usize,
    pos: usize,
    q: Vec<f32>,
    k: Vec<u16>,
    v: Vec<u16>,
}
impl Data {
    fn random(rows: usize, pos: usize) -> Self {
        let mut state = 0x6a09_e667_f3bc_c909u64;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 16_777_216.0 * 2.0 - 1.0) * 1.732_050_8
        };
        Self {
            rows,
            pos,
            q: (0..rows * 24 * 256).map(|_| sample()).collect(),
            k: (0..pos * 1024)
                .map(|_| half::f16::from_f32(sample()).to_bits())
                .collect(),
            v: (0..pos * 1024)
                .map(|_| half::f16::from_f32(sample()).to_bits())
                .collect(),
        }
    }
    fn base(&self) -> usize {
        self.pos - self.rows
    }
    fn controlled(rows: usize, scores: &[f32]) -> Self {
        let pos = scores.len() * 64;
        let mut d = Self {
            rows,
            pos,
            q: vec![0.0; rows * 24 * 256],
            k: vec![0; pos * 1024],
            v: vec![0; pos * 1024],
        };
        for row in 0..rows {
            for h in 0..24 {
                // Interleave increasing, uniform and decreasing maxima WITHIN
                // each cooperative output tile, not merely between threadgroups.
                d.q[(row * 24 + h) * 256] = match (row * 24 + h) % 3 {
                    0 => 16.0,
                    1 => 0.0,
                    _ => -16.0,
                };
            }
        }
        for p in 0..pos {
            for h in 0..4 {
                for c in 0..256 {
                    let i = p * 1024 + h * 256 + c;
                    if c == 0 {
                        d.k[i] = half::f16::from_f32(scores[p / 64]).to_bits();
                    }
                    d.v[i] = half::f16::from_f32(
                        0.25 + (p / 64 % 5) as f32 * 0.125 + (c % 17) as f32 / 256.0,
                    )
                    .to_bits();
                }
            }
        }
        d
    }
}

#[derive(Clone)]
struct Buffers {
    q: MetalTensor,
    k: MetalTensor,
    v: MetalTensor,
    masks: MetalTensor,
    out: MetalTensor,
}
impl Buffers {
    fn upload(ctx: &MetalContext, d: &Data) -> Self {
        let make =
            |b: &[u8], n: usize, t| MetalTensor::from_bytes(ctx, b, vec![n as u64], t).unwrap();
        let layout = tensor_flash_qwen27_layout(d.rows, d.base(), d.pos).unwrap();
        Self {
            q: make(bytemuck::cast_slice(&d.q), d.q.len(), GgmlType::F32),
            k: make(bytemuck::cast_slice(&d.k), d.k.len(), GgmlType::F16),
            v: make(bytemuck::cast_slice(&d.v), d.v.len(), GgmlType::F16),
            masks: MetalTensor::zeros_f32(ctx, vec![layout.mask_bytes.div_ceil(4) as u64]).unwrap(),
            out: MetalTensor::zeros_f32(ctx, vec![d.q.len() as u64]).unwrap(),
        }
    }
    fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        rows: usize,
        base: usize,
        pos: usize,
    ) -> Result<(), MetalError> {
        encode_attn_tensor_flash_qwen27(
            ctx,
            enc,
            &self.q,
            &self.k,
            &self.v,
            &self.masks,
            &self.out,
            rows,
            base,
            pos,
        )
    }
    fn run(&self, ctx: &MetalContext, d: &Data) -> Vec<f32> {
        one_shot(ctx, |enc| self.encode(ctx, enc, d.rows, d.base(), d.pos)).unwrap();
        unsafe {
            std::slice::from_raw_parts(
                self.out
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(self.out.offset as usize)
                    .cast::<f32>(),
                d.q.len(),
            )
            .to_vec()
        }
    }
    fn matrix(&self, ctx: &MetalContext, d: &Data) -> Vec<f32> {
        // Bound the reference workspace even for 4096 queries at context 262144:
        // a full score matrix would require 48 GiB and exceed the buffer/index contract.
        // Each incumbent batch still uses the original total KV and causal base.
        const QUERY_BATCH: usize = 64;
        let batch_rows = d.rows.min(QUERY_BATCH);
        let vt = MetalTensor::zeros_f16(ctx, vec![(d.pos * 1024) as u64]).unwrap();
        let scores = MetalTensor::zeros_f16(ctx, vec![(batch_rows * 24 * d.pos) as u64]).unwrap();
        let ml = MetalTensor::zeros_f32(
            ctx,
            vec![attn_matrix_ml_elems(batch_rows, 24, d.pos) as u64],
        )
        .unwrap();
        let out = MetalTensor::zeros_f32(ctx, vec![d.q.len() as u64]).unwrap();
        one_shot(ctx, |enc| {
            encode_attn_matrix_transpose_v_f16(
                ctx, enc, &self.v, &vt, 0, d.pos, d.pos, 1024, d.pos, 4, 256,
            )
        })
        .unwrap();
        for first in (0..d.rows).step_by(QUERY_BATCH) {
            let rows = (d.rows - first).min(QUERY_BATCH);
            let q = self
                .q
                .view_subrange((first * 24 * 256) as u64, vec![(rows * 24 * 256) as u64]);
            let o = out.view_subrange((first * 24 * 256) as u64, vec![(rows * 24 * 256) as u64]);
            let scores = scores.view_subrange(0, vec![(rows * 24 * d.pos) as u64]);
            let ml = ml.view_subrange(0, vec![attn_matrix_ml_elems(rows, 24, d.pos) as u64]);
            one_shot(ctx, |enc| {
                encode_attn_matrix_kq_online_f32(
                    ctx,
                    enc,
                    &q,
                    &self.k,
                    &scores,
                    &ml,
                    rows,
                    d.base() + first,
                    d.pos,
                    1024,
                    24,
                    4,
                    6,
                    256,
                    true,
                )?;
                encode_attn_matrix_kqv_norm_f32(
                    ctx,
                    enc,
                    &scores,
                    &ml,
                    &vt,
                    &o,
                    rows,
                    d.base() + first,
                    d.pos,
                    d.pos,
                    24,
                    4,
                    6,
                    256,
                    true,
                )
            })
            .unwrap();
        }
        read_back_f32(&out.buffer, d.q.len())
    }
}

fn compare_vectors(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: output length");
    let mut error = 0f64;
    let mut min_cos = 1f64;
    for (a, b) in a.chunks_exact(256).zip(b.chunks_exact(256)) {
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            assert!(x.is_finite() && y.is_finite(), "{label}: nonfinite");
            error = error.max((f64::from(x) - f64::from(y)).abs());
            ab += f64::from(x) * f64::from(y);
            aa += f64::from(x).powi(2);
            bb += f64::from(y).powi(2);
        }
        min_cos = min_cos.min(if aa == 0.0 && bb == 0.0 {
            1.0
        } else {
            ab / (aa * bb).sqrt()
        });
    }
    assert!(
        error <= 0.02 && min_cos >= 0.9999,
        "{label}: max_abs={error} min_cos={min_cos}"
    );
}

fn sampled_f64(d: &Data, actual: &[f32]) {
    let mut error = 0f64;
    let mut min_cos = 1f64;
    for row in [0, d.rows / 2, d.rows - 1] {
        for h in [0, 5, 6, 12, 18, 23] {
            let visible = d.base() + row + 1;
            let q = &d.q[(row * 24 + h) * 256..(row * 24 + h + 1) * 256];
            let uniform = q.iter().all(|&x| x == 0.0);
            let mut scores = vec![0f64; visible];
            let mut max = f64::NEG_INFINITY;
            for (p, s) in scores.iter_mut().enumerate() {
                if !uniform {
                    for (c, &q) in q.iter().enumerate() {
                        *s += f64::from(q)
                            * f64::from(
                                half::f16::from_bits(d.k[p * 1024 + h / 6 * 256 + c]).to_f32(),
                            );
                    }
                    *s /= 16.0;
                }
                max = max.max(*s);
            }
            let mut reference = [0f64; 256];
            let mut sum = 0f64;
            for (p, &s) in scores.iter().enumerate() {
                let w = if uniform { 1.0 } else { (s - max).exp() };
                sum += w;
                for (c, x) in reference.iter_mut().enumerate() {
                    *x += w * f64::from(
                        half::f16::from_bits(d.v[p * 1024 + h / 6 * 256 + c]).to_f32(),
                    );
                }
            }
            let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
            for (c, x) in reference.iter_mut().enumerate() {
                *x /= sum;
                let y = f64::from(actual[(row * 24 + h) * 256 + c]);
                assert!(y.is_finite());
                error = error.max((y - *x).abs());
                ab += y * *x;
                aa += y * y;
                bb += *x * *x;
            }
            min_cos = min_cos.min(if aa == 0.0 && bb == 0.0 {
                1.0
            } else {
                ab / (aa * bb).sqrt()
            });
        }
    }
    eprintln!(
        "tensor_flash F64 rows={} KV={} sampled_heads=18 max_abs={error:e} min_cos={min_cos}",
        d.rows, d.pos
    );
    assert!(error <= 0.02 && min_cos >= 0.9999);
}

#[test]
fn tensor_flash_layout_bounds_cover_64bit_indices_and_scratch() {
    assert_eq!(TENSOR_FLASH_SCRATCH_LAYOUT_BYTES, 28804);
    assert_eq!(TENSOR_FLASH_SCRATCH_BYTES, 28816);
    assert_eq!(TENSOR_FLASH_SCRATCH_BYTES % 16, 0);
    assert!(TENSOR_FLASH_SCRATCH_BYTES <= 32 * 1024);
    let n = TENSOR_FLASH_MAX_POSITIONS;
    let a = tensor_flash_qwen27_layout(n, 0, n).unwrap();
    // These are accepted logical shapes whose products caused the original
    // uint shader wrap. Physical buffer validation is a separate contract.
    assert!(a.q_elements - 1 > u64::from(u32::MAX));
    assert!(a.mask_bytes as u64 - 1 > u64::from(u32::MAX));
    assert_eq!(a.q_elements, n as u64 * 24 * 256);
    assert_eq!(a.kv_elements, n as u64 * 1024);
    assert_eq!(a.mask_bytes, a.tiles * (n / 64));
    assert!(tensor_flash_qwen27_layout(64, n, n + 64).is_err());
    for (rows, base, pos) in [
        (0, 64, 64),
        (1, usize::MAX, 0),
        (usize::MAX, 1, 0),
        (64, 0, usize::MAX),
    ] {
        assert!(tensor_flash_qwen27_layout(rows, base, pos).is_err());
    }
}

fn rejected(change: impl FnOnce(&MetalContext, &Data, &mut Buffers), needle: &str) {
    gpu(false, |ctx| {
        let d = Data::random(64, 128);
        let mut b = Buffers::upload(ctx, &d);
        change(ctx, &d, &mut b);
        one_shot(ctx, |enc| {
            let e = b.encode(ctx, enc, d.rows, d.base(), d.pos).unwrap_err();
            assert!(e.to_string().contains(needle), "{e}");
            Ok(())
        })
        .unwrap();
    });
}

#[test]
fn tensor_flash_rejects_unaligned_kv_independently() {
    gpu(false, |ctx| {
        let d = Data::random(63, 128);
        let b = Buffers::upload(ctx, &d);
        one_shot(ctx, |enc| {
            let e = b.encode(ctx, enc, 63, 64, 127).unwrap_err();
            assert!(e.to_string().contains("64-aligned"));
            Ok(())
        })
        .unwrap();
    });
}
#[test]
fn tensor_flash_rejects_base_position_independently() {
    gpu(false, |ctx| {
        let d = Data::random(64, 128);
        let b = Buffers::upload(ctx, &d);
        one_shot(ctx, |enc| {
            let e = b.encode(ctx, enc, 64, 63, 128).unwrap_err();
            assert!(e.to_string().contains("base_pos + n_rows"));
            Ok(())
        })
        .unwrap();
    });
}
#[test]
fn tensor_flash_rejects_concurrent_encoder_before_dispatch() {
    gpu(false, |ctx| {
        let d = Data::random(64, 128);
        let b = Buffers::upload(ctx, &d);
        let cb = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin_concurrent(&cb);
        let e = b.encode(ctx, &enc, 64, 64, 128).unwrap_err();
        assert!(e.to_string().contains("serial encoder"));
        enc.end();
    });
}
#[test]
fn tensor_flash_rejects_readonly_masks_and_output() {
    rejected(
        |_, _, b| b.masks.provenance = MetalTensorProvenance::OwnedWeightReadOnly,
        "masks must be writable",
    );
    rejected(
        |_, _, b| b.out.provenance = MetalTensorProvenance::OwnedWeightReadOnly,
        "out must be writable",
    );
}
#[test]
fn tensor_flash_rejects_each_dtype_and_element_extent() {
    rejected(|_, _, b| b.q.dtype = GgmlType::I32, "q requires F32");
    rejected(|_, _, b| b.out.dtype = GgmlType::I32, "out requires F32");
    rejected(|_, _, b| b.k.dtype = GgmlType::F32, "k requires F16");
    rejected(|_, _, b| b.v.dtype = GgmlType::F32, "v requires F16");
    rejected(
        |_, d, b| b.q.shape = vec![(d.q.len() - 4) as u64],
        "q element count",
    );
    rejected(
        |_, d, b| b.out.shape = vec![(d.q.len() - 4) as u64],
        "out element count",
    );
    rejected(
        |_, d, b| b.k.shape = vec![(d.k.len() - 1) as u64],
        "k cache too small",
    );
    rejected(
        |_, d, b| b.v.shape = vec![(d.v.len() - 1) as u64],
        "v cache too small",
    );
    rejected(
        |_, _, b| b.masks.shape = vec![5],
        "mask workspace too small",
    );
}
#[test]
fn tensor_flash_rejects_each_out_of_buffer_view() {
    rejected(|_, _, b| b.q.offset = 16, "q view exceeds");
    rejected(|_, _, b| b.k.offset = 2, "k view exceeds");
    rejected(|_, _, b| b.v.offset = 2, "v view exceeds");
    rejected(|_, _, b| b.masks.offset = 1, "masks view exceeds");
    rejected(|_, _, b| b.out.offset = 4, "out view exceeds");
}
#[test]
fn tensor_flash_rejects_overflowed_metadata_without_panicking() {
    rejected(
        |_, _, b| b.q.shape = vec![u64::MAX, 2],
        "q shape/byte extent",
    );
    rejected(
        |_, _, b| b.masks.shape = vec![u64::MAX],
        "masks shape/byte extent",
    );
    rejected(
        |_, _, b| b.q.offset = u64::MAX - 15,
        "q buffer range overflow",
    );
}
#[test]
fn tensor_flash_rejects_q_float4_and_other_typed_misalignment() {
    for offset in [4, 8, 12] {
        rejected(
            |ctx, d, b| {
                let parent = MetalTensor::zeros_f32(ctx, vec![(d.q.len() + 4) as u64]).unwrap();
                b.q = parent.view_bytes(offset, vec![d.q.len() as u64]);
            },
            "q offset requires 16-byte alignment",
        );
    }
    rejected(
        |ctx, d, b| {
            let p = MetalTensor::zeros_f16(ctx, vec![(d.k.len() + 1) as u64]).unwrap();
            b.k = p.view_bytes(1, vec![d.k.len() as u64]);
        },
        "k offset requires 2-byte alignment",
    );
    rejected(
        |ctx, d, b| {
            let p = MetalTensor::zeros_f16(ctx, vec![(d.v.len() + 1) as u64]).unwrap();
            b.v = p.view_bytes(1, vec![d.v.len() as u64]);
        },
        "v offset requires 2-byte alignment",
    );
    rejected(
        |ctx, d, b| {
            let p = MetalTensor::zeros_f32(ctx, vec![(d.q.len() + 1) as u64]).unwrap();
            b.out = p.view_bytes(2, vec![d.q.len() as u64]);
        },
        "out offset requires 4-byte alignment",
    );
}
#[test]
fn tensor_flash_rejects_write_aliases() {
    rejected(
        |_, _, b| {
            b.masks = b.q.clone();
            b.masks.shape = vec![6];
        },
        "aliases q",
    );
    rejected(|_, _, b| b.out = b.q.clone(), "aliases q");
    rejected(
        |_, d, b| {
            b.k = b.out.clone();
            b.k.dtype = GgmlType::F16;
            b.k.shape = vec![d.k.len() as u64];
        },
        "aliases k",
    );
    rejected(
        |_, _, b| {
            b.masks = b.out.clone();
            b.masks.shape = vec![6];
        },
        "masks and out overlap",
    );
}
#[test]
fn tensor_flash_accepts_aligned_nonzero_offsets() {
    gpu(true, |ctx| {
        let d = Data::random(64, 128);
        let mut b = Buffers::upload(ctx, &d);
        let mut qp = vec![0f32; d.q.len() + 4];
        qp[4..].copy_from_slice(&d.q);
        b.q = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&qp),
            vec![qp.len() as u64],
            GgmlType::F32,
        )
        .unwrap()
        .view_subrange(4, vec![d.q.len() as u64]);
        let op = MetalTensor::zeros_f32(ctx, vec![(d.q.len() + 1) as u64]).unwrap();
        b.out = op.view_subrange(1, vec![d.q.len() as u64]);
        let mp = MetalTensor::zeros_f32(ctx, vec![7]).unwrap();
        b.masks = mp.view_bytes(1, vec![6]);
        sampled_f64(&d, &b.run(ctx, &d));
    });
}

#[test]
fn tensor_flash_small_edges_f64() {
    gpu(true, |ctx| {
        for (rows, pos) in [(1, 64), (20, 64), (33, 192), (64, 64)] {
            for pattern in 0..3 {
                let mut d = Data::random(rows, pos);
                if pattern == 1 {
                    d.q.fill(0.0);
                }
                if pattern == 2 {
                    for h in 0..24 {
                        for c in 0..256 {
                            d.q[((rows - 1) * 24 + h) * 256 + c] =
                                half::f16::from_bits(d.k[(pos - 1) * 1024 + h / 6 * 256 + c])
                                    .to_f32()
                                    * 8.0;
                        }
                    }
                }
                let b = Buffers::upload(ctx, &d);
                let actual = b.run(ctx, &d);
                let k: Vec<f32> =
                    d.k.iter()
                        .map(|&x| half::f16::from_bits(x).to_f32())
                        .collect();
                let v: Vec<f32> =
                    d.v.iter()
                        .map(|&x| half::f16::from_bits(x).to_f32())
                        .collect();
                let reference =
                    cpu_matrix_attn_reference(&d.q, &k, &v, rows, d.base(), pos, 24, 4, 6, 256);
                compare_vectors("small complete F64", &actual, &reference);
            }
        }
    });
}

macro_rules! long_case {
    ($name:ident,$pos:expr) => {
        #[test]
        fn $name() {
            gpu(true, |ctx| {
                let d = Data::random(64, $pos);
                let b = Buffers::upload(ctx, &d);
                let a = b.run(ctx, &d);
                compare_vectors("long incumbent", &a, &b.matrix(ctx, &d));
                sampled_f64(&d, &a);
            });
        }
    };
}
long_case!(tensor_flash_kv_8192_sampled_f64, 8192);
long_case!(tensor_flash_kv_32768_sampled_f64, 32768);
long_case!(tensor_flash_kv_65536_sampled_f64, 65536);
// Qwen3.8-27B metadata max_position_embeddings/context_length, not the much
// broader integer address contract tested without enormous GPU allocations.
long_case!(tensor_flash_model_context_262144_sampled_f64, 262144);

// Exercise the actual complete tensor query grid at both automatic boundaries,
// not a 64-query proxy. The incumbent uses bounded score batches; all output
// rows/heads/channels are compared, plus independent F64 at first/middle/last
// queries across all four GQA groups. 129 includes the final partial 32-row tile.
macro_rules! auto_query_case {
    ($name:ident, $rows:expr, $pos:expr) => {
        #[test]
        fn $name() {
            gpu(true, |ctx| {
                let d = Data::random($rows, $pos);
                let b = Buffers::upload(ctx, &d);
                let actual = b.run(ctx, &d);
                compare_vectors(
                    "automatic query boundary incumbent",
                    &actual,
                    &b.matrix(ctx, &d),
                );
                sampled_f64(&d, &actual);
            });
        }
    };
}
auto_query_case!(
    tensor_flash_auto_128_queries_kv8192_f64_incumbent,
    128,
    8192
);
auto_query_case!(
    tensor_flash_auto_4096_queries_kv8192_f64_incumbent,
    4096,
    8192
);
auto_query_case!(
    tensor_flash_auto_128_queries_kv262144_f64_incumbent,
    128,
    262144
);
auto_query_case!(
    tensor_flash_auto_4096_queries_kv262144_f64_incumbent,
    4096,
    262144
);
auto_query_case!(
    tensor_flash_auto_129_queries_kv8192_f64_incumbent,
    129,
    8192
);
auto_query_case!(
    tensor_flash_auto_129_queries_kv262144_f64_incumbent,
    129,
    262144
);

#[test]
fn tensor_flash_incumbent_query_batching_matches_f64() {
    // Runs without TensorOps too, protecting reference row views, causal base
    // increments and the final one-row batch on the incumbent-only hardware.
    gpu(false, |ctx| {
        let d = Data::random(129, 192);
        let b = Buffers::upload(ctx, &d);
        let to_f32 = |xs: &[u16]| {
            xs.iter()
                .map(|&x| half::f16::from_bits(x).to_f32())
                .collect::<Vec<_>>()
        };
        let reference = cpu_matrix_attn_reference(
            &d.q,
            &to_f32(&d.k),
            &to_f32(&d.v),
            d.rows,
            d.base(),
            d.pos,
            24,
            4,
            6,
            256,
        );
        compare_vectors(
            "batched incumbent complete F64",
            &b.matrix(ctx, &d),
            &reference,
        );
    });
}

#[test]
fn tensor_flash_long_uniform_distribution_keeps_float_accumulator() {
    gpu(true, |ctx| {
        let mut d = Data::controlled(64, &vec![0.0; 65536 / 64]);
        d.q.fill(0.0);
        let b = Buffers::upload(ctx, &d);
        let a = b.run(ctx, &d);
        sampled_f64(&d, &a);
        compare_vectors("uniform incumbent", &a, &b.matrix(ctx, &d));
    });
}
#[test]
fn tensor_flash_rescale_below_at_above_eight_and_partial_rows() {
    gpu(true, |ctx| {
        for step in [7.99609375, 8.0, 8.0078125] {
            let scores: Vec<f32> = (0..8).map(|b| step * b as f32).collect();
            let d = Data::controlled(65, &scores);
            let b = Buffers::upload(ctx, &d);
            let a = b.run(ctx, &d);
            sampled_f64(&d, &a);
            compare_vectors("threshold/partial", &a, &b.matrix(ctx, &d));
        }
    });
}
#[test]
fn tensor_flash_repeated_cross_key_tile_maximum_jumps() {
    gpu(true, |ctx| {
        let d = Data::controlled(
            65,
            &[
                0.0, 0.0, 0.0, 16.0, 16.0, 24.015625, 8.0, 32.0, 32.0, 40.0, 32.0, 48.0, 56.0, 8.0,
                56.0, 64.0,
            ],
        );
        let b = Buffers::upload(ctx, &d);
        let a = b.run(ctx, &d);
        sampled_f64(&d, &a);
        compare_vectors("cross-tile", &a, &b.matrix(ctx, &d));
    });
}

#[test]
fn tensor_flash_nonfinite_policy_matches_incumbent_for_read_operands() {
    gpu(true, |ctx| {
        for kind in 0..3 {
            let mut d = Data::controlled(64, &[0.0, 0.0]);
            d.q.fill(0.0);
            for row in 0..64 {
                for h in 0..24 {
                    d.q[(row * 24 + h) * 256] = 16.0;
                }
            }
            for h in 0..4 {
                if kind == 0 {
                    d.k[h * 256] = half::f16::INFINITY.to_bits();
                } else {
                    d.k[h * 256] = half::f16::from_f32(-64.0).to_bits();
                    d.v[h * 256] = if kind == 1 {
                        half::f16::INFINITY.to_bits()
                    } else {
                        half::f16::NAN.to_bits()
                    };
                }
            }
            // kind0: +Inf scores; kind1/2: zero half-P via underflow;
            // Non-finite inputs are not sanitized, even when half P underflows to zero.
            let b = Buffers::upload(ctx, &d);
            let a = b.run(ctx, &d);
            let inc = b.matrix(ctx, &d);
            assert!(
                a[0].is_nan() && inc[0].is_nan(),
                "nonfinite kind={kind}: {}/{}",
                a[0],
                inc[0]
            );
            for (&a, &i) in a.iter().zip(&inc) {
                assert_eq!(
                    (a.is_nan(), a.is_infinite()),
                    (i.is_nan(), i.is_infinite()),
                    "nonfinite kind={kind}"
                );
            }
        }
    });
}

#[test]
fn tensor_flash_nonfinite_causal_mask_documents_tile_granularity_difference() {
    gpu(true, |ctx| {
        let mut d = Data::controlled(64, &[0.0, 0.0]);
        d.q.fill(0.0);
        for h in 0..4 {
            d.v[127 * 1024 + h * 256] = half::f16::NAN.to_bits();
        }
        let b = Buffers::upload(ctx, &d);
        let a = b.run(ctx, &d);
        let inc = b.matrix(ctx, &d);
        // Query0 sees keys0..64. Flash reads a64-key PV panel containing
        // key127 with P=0; incumbent skips the wholly future32-key panel.
        // Both expose NaN once that key is visible. This is explicitly OUTSIDE
        // the finite-input contract, not a claim of non-finite mask parity.
        assert!(a[0].is_nan());
        assert!(inc[0].is_finite());
        let last = (63 * 24) * 256;
        assert!(a[last].is_nan() && inc[last].is_nan());
    });
}

#[test]
fn tensor_flash_restored_prefix_to_unaligned_matrix_rebuild() {
    gpu(true, |ctx| {
        const PREFIX: usize = 64;
        const FLASH: usize = 64;
        const TAIL: usize = 33;
        let full = Data::random(FLASH + TAIL, PREFIX + FLASH + TAIL);
        let half_f32 = |xs: &[u16]| {
            xs.iter()
                .map(|&x| half::f16::from_bits(x).to_f32())
                .collect::<Vec<_>>()
        };
        let make16 = |xs: &[u16]| {
            MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(xs),
                vec![xs.len() as u64],
                GgmlType::F16,
            )
            .unwrap()
        };
        let make32 = |xs: &[f32]| {
            MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(xs),
                vec![xs.len() as u64],
                GgmlType::F32,
            )
            .unwrap()
        };
        // Restoring a checkpoint supplies raw KV, never the derived VT. Poison
        // every missing row so an omitted or too-short rebuild is observable.
        let mut restored_k = vec![half::f16::NAN.to_bits(); full.k.len()];
        let mut restored_v = restored_k.clone();
        restored_k[..PREFIX * 1024].copy_from_slice(&full.k[..PREFIX * 1024]);
        restored_v[..PREFIX * 1024].copy_from_slice(&full.v[..PREFIX * 1024]);
        let k = make16(&restored_k);
        let v = make16(&restored_v);
        let vt_stride = full.pos + 7;
        let vt = make16(&vec![half::f16::NAN.to_bits(); vt_stride * 1024]);
        let q_flash = make32(&full.q[..FLASH * 24 * 256]);
        let q_tail = make32(&full.q[FLASH * 24 * 256..]);
        let masks = MetalTensor::zeros_f32(ctx, vec![128]).unwrap();
        let o_flash = MetalTensor::zeros_f32(ctx, vec![(FLASH * 24 * 256) as u64]).unwrap();
        let k_now = make32(&half_f32(&full.k[PREFIX * 1024..(PREFIX + FLASH) * 1024]));
        let v_now = make32(&half_f32(&full.v[PREFIX * 1024..(PREFIX + FLASH) * 1024]));
        one_shot(ctx, |enc| {
            encode_scatter_offset_f32_to_f16_kv_vt(
                ctx,
                enc,
                &k_now,
                &v_now,
                &k,
                &v,
                &vt,
                PREFIX * 1024,
                FLASH * 1024,
                PREFIX,
                1024,
                256,
                vt_stride,
            )?;
            encode_attn_tensor_flash_qwen27(
                ctx,
                enc,
                &q_flash,
                &k,
                &v,
                &masks,
                &o_flash,
                FLASH,
                PREFIX,
                PREFIX + FLASH,
            )
        })
        .unwrap();
        let valid =
            crate::metal_dflash::attn_matrix_vt_after_tensor_flash(0, PREFIX, PREFIX + FLASH);
        assert_eq!(valid, 0);
        let rebuild =
            crate::metal_dflash::attn_matrix_vt_prefix_rebuild_rows(true, valid, PREFIX + FLASH)
                .unwrap();
        assert_eq!(rebuild, PREFIX + FLASH);
        let k_tail = make32(&half_f32(&full.k[(PREFIX + FLASH) * 1024..]));
        let v_tail = make32(&half_f32(&full.v[(PREFIX + FLASH) * 1024..]));
        let scores = MetalTensor::zeros_f16(ctx, vec![(TAIL * 24 * full.pos) as u64]).unwrap();
        let ml = MetalTensor::zeros_f32(ctx, vec![attn_matrix_ml_elems(TAIL, 24, full.pos) as u64])
            .unwrap();
        let out = MetalTensor::zeros_f32(ctx, vec![(TAIL * 24 * 256) as u64]).unwrap();
        one_shot(ctx, |enc| {
            encode_scatter_offset_f32_to_f16_kv_vt(
                ctx,
                enc,
                &k_tail,
                &v_tail,
                &k,
                &v,
                &vt,
                (PREFIX + FLASH) * 1024,
                TAIL * 1024,
                PREFIX + FLASH,
                1024,
                256,
                vt_stride,
            )?;
            encode_attn_matrix_transpose_v_f16(
                ctx, enc, &v, &vt, 0, rebuild, full.pos, 1024, vt_stride, 4, 256,
            )?;
            encode_attn_matrix_kq_online_f32(
                ctx,
                enc,
                &q_tail,
                &k,
                &scores,
                &ml,
                TAIL,
                PREFIX + FLASH,
                full.pos,
                1024,
                24,
                4,
                6,
                256,
                true,
            )?;
            encode_attn_matrix_kqv_norm_f32(
                ctx,
                enc,
                &scores,
                &ml,
                &vt,
                &out,
                TAIL,
                PREFIX + FLASH,
                full.pos,
                vt_stride,
                24,
                4,
                6,
                256,
                true,
            )
        })
        .unwrap();
        let kf = half_f32(&full.k);
        let vf = half_f32(&full.v);
        let reference = cpu_matrix_attn_reference(
            &full.q[FLASH * 24 * 256..],
            &kf,
            &vf,
            TAIL,
            PREFIX + FLASH,
            full.pos,
            24,
            4,
            6,
            256,
        );
        compare_vectors(
            "restored flash then matrix VT rebuild",
            &read_back_f32(&out.buffer, TAIL * 24 * 256),
            &reference,
        );
        // Check all derived VT values, including original restored prefix,
        // the flashed chunk and the unaligned tail; padding stays irrelevant.
        let actual = unsafe {
            std::slice::from_raw_parts(
                vt.buffer.contents().as_ptr().cast::<u16>(),
                vt_stride * 1024,
            )
        };
        for p in 0..full.pos {
            for h in 0..4 {
                for c in 0..256 {
                    assert_eq!(
                        actual[(h * 256 + c) * vt_stride + p],
                        full.v[p * 1024 + h * 256 + c]
                    );
                }
            }
        }
    });
}

#[test]
#[ignore = "requires the Qwen3.8-27B-Q4_K_M model fixture and Apple10"]
fn tensor_flash_restored_model_checkpoint_and_matrix_tail() {
    tensor_flash_restored_model_case(64, 33, false);
}

#[test]
#[ignore = "requires the model fixture, M5 Ultra and an unset tensor-flash knob"]
fn tensor_flash_default_restored_checkpoint_and_aligned_tail() {
    tensor_flash_restored_model_case(7168, 64, true);
}

fn tensor_flash_restored_model_case(prefix_rows: usize, tail_rows: usize, automatic: bool) {
    use crate::runtime::{Runtime, SequenceConfig};
    let _lease = acquire_metal_benchmark_lease().expect("canonical model correctness lease");
    let runtime = Runtime::metal().unwrap();
    if !runtime.context().tensor_ops_status().is_available() {
        return;
    }
    if automatic && runtime.context().device_facts().name != "Apple M5 Ultra" {
        return;
    }
    let path = crate::test_fixtures::required_filename("Qwen3.8-27B-Q4_K_M.gguf");
    let model = runtime.load_model(path).unwrap();
    assert_eq!(model.gguf().get_u64("qwen35.context_length"), Some(262144));
    let total = prefix_rows + 1024 + tail_rows;
    let capacity = total.div_ceil(64) * 64;
    let ids: Vec<i32> = (0..total)
        .map(|i| ((i * 13 + 7) % model.arch().vocab_size as usize) as i32)
        .collect();
    let mut prefix = model
        .create_sequence(SequenceConfig::new(capacity))
        .unwrap();
    let mut scratch = model
        .allocate_packed_prefill_scratch(model.plan_packed_prefill_scratch(1024, capacity).unwrap())
        .unwrap();
    crate::metal_dflash::with_prefill_attn_tensor_flash_override(false, || {
        model.prefill(&mut prefix, &mut scratch, &ids[..prefix_rows])
    })
    .unwrap();
    let checkpoint = model
        .prepare_checkpoint_boundary(&prefix, ids[..prefix_rows].to_vec(), None, None, None, 0)
        .unwrap();
    let mut results = Vec::new();
    for enabled in [false, true] {
        let mut seq = model
            .create_sequence(SequenceConfig::new(capacity))
            .unwrap();
        model
            .restore_prepared_checkpoint(&checkpoint, &mut seq, &ids)
            .unwrap();
        assert_eq!(seq.position(), prefix_rows);
        let mut scratch = model
            .allocate_packed_prefill_scratch(
                model.plan_packed_prefill_scratch(1024, capacity).unwrap(),
            )
            .unwrap();
        dispatch_census_begin();
        let logits = if enabled && automatic {
            model.prefill(&mut seq, &mut scratch, &ids[prefix_rows..])
        } else {
            crate::metal_dflash::with_prefill_attn_tensor_flash_override(enabled, || {
                model.prefill(&mut seq, &mut scratch, &ids[prefix_rows..])
            })
        }
        .unwrap();
        let census = dispatch_census_take();
        assert_eq!(seq.position(), total);
        let flash = census
            .iter()
            .filter(|x| x.kernel == "kernel_attn_flash_probe_f16_d256")
            .count();
        let matrix = census
            .iter()
            .filter(|x| x.kernel.starts_with("kernel_attn_matrix_kq_online"))
            .count();
        let rebuild = census
            .iter()
            .filter(|x| x.kernel.starts_with("kernel_attn_matrix_transpose_v_f16"))
            .count();
        assert_eq!(flash, if enabled { 16 } else { 0 });
        assert_eq!(matrix, if enabled { 16 } else { 32 });
        assert_eq!(rebuild, 16);
        results.push(logits);
    }
    compare_vectors(
        "restored model flash chunk plus matrix tail",
        &results[0],
        &results[1],
    );
    let argmax = |x: &[f32]| {
        x.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    assert_eq!(argmax(&results[0]), argmax(&results[1]));
}
