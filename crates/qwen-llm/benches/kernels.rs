//! Kernel-level performance benchmarks.
//!
//! Measures *per-dispatch* time on persistent buffers — i.e. excludes
//! buffer creation / copy-in / copy-out, which is what real inference
//! sees. Two regimes per shape:
//!
//! 1. **single-dispatch**: one kernel per command buffer, with
//!    `waitUntilCompleted` at the end. Captures dispatch + wait overhead.
//!    This is the worst case; serialized GPU access.
//! 2. **chained**: 64 dispatches in a single command buffer, one wait at
//!    the end. Approximates what a 64-layer forward pass with persistent
//!    weights would see. The honest "is this fast at scale" number.
//!
//! The 27B model file is required; benches that need it skip cleanly if
//! it's missing.
//!
//! Run a single one:
//!     cargo bench -p qwen-llm --bench kernels -- 'q4_k mat_vec/embed.*chained'
//! Hyperfine the comparison against llama-bench:
//!     hyperfine --warmup 3 './target/release/qwen-bench …' '… llama-bench …'

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::{mat_vec_q4_k_f32_bufs, MetalContext};
use qwen_llm::tensor::{GgmlType, TensorDesc};

const MODEL_27B: &str = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";

fn pick_q4k_shapes(g: &GgufFile) -> Vec<(&'static str, &TensorDesc)> {
    let names = [
        ("attn_gate_27b", "blk.0.attn_gate.weight"),
        ("ffn_gate_27b", "blk.0.ffn_gate.weight"),
        ("ffn_up_27b", "blk.0.ffn_up.weight"),
        ("embed_27b", "token_embd.weight"),
    ];
    names
        .iter()
        .filter_map(|(label, name)| {
            g.find(name).and_then(|t| {
                if t.dtype == GgmlType::Q4_K {
                    Some((*label, t))
                } else {
                    None
                }
            })
        })
        .collect()
}

fn bench_q4k_mat_vec(c: &mut Criterion) {
    if !std::path::Path::new(MODEL_27B).exists() {
        eprintln!("[bench] skipping q4_k — {MODEL_27B} not present");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[bench] no metal context: {e}");
            return;
        }
    };
    let g = GgufFile::open(MODEL_27B).expect("open 27B");
    let shapes = pick_q4k_shapes(&g);

    let mut group = c.benchmark_group("q4_k mat_vec");
    for (label, t) in &shapes {
        let n_in = t.shape[0] as usize;
        let n_out = t.shape[1] as usize;

        // Persistent resources for this shape.
        let buf_w = ctx.buffer_from(g.slice(t)).expect("w");
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 1e-3).sin()).collect();
        let buf_x = ctx.buffer_from(&x).expect("x");
        let buf_y = ctx
            .buffer_uninit(n_out * std::mem::size_of::<f32>())
            .expect("y");

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Args {
            n_in: u32,
            n_out: u32,
        }
        let buf_args = ctx
            .buffer_from(&[Args {
                n_in: n_in as u32,
                n_out: n_out as u32,
            }])
            .expect("args");

        // Throughput annotation: bytes-of-weights touched per dispatch.
        group.throughput(Throughput::Bytes(t.n_bytes));

        // 1) Single dispatch per command buffer.
        group.bench_with_input(
            BenchmarkId::new("single", label),
            &(label, t.n_bytes),
            |b, _| {
                b.iter(|| {
                    mat_vec_q4_k_f32_bufs(&ctx, &buf_args, &buf_w, &buf_x, &buf_y, n_out)
                        .expect("dispatch");
                    black_box(&buf_y);
                });
            },
        );

        // 2) 64 dispatches per command buffer, single wait. Approximates
        // what a layered forward pass sees with persistent buffers and
        // chained command encoding.
        group.bench_with_input(
            BenchmarkId::new("chained64", label),
            &(label, t.n_bytes),
            |b, _| {
                b.iter(|| {
                    qwen_llm::metal::mat_vec_q4_k_f32_chained(
                        &ctx, &buf_args, &buf_w, &buf_x, &buf_y, n_out, 64,
                    )
                    .expect("chained");
                    black_box(&buf_y);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_q4k_mat_vec);
criterion_main!(benches);
