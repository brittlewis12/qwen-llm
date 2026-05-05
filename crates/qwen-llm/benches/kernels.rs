//! Kernel-level performance benchmarks (post-API-rewrite).
//!
//! Each bench creates persistent `MetalTensor`s once, then issues N
//! dispatches against them inside a single command buffer. This is the
//! shape the production forward pass uses, so numbers here translate
//! directly to per-step inference cost.
//!
//! Two regimes per shape:
//! * **single**: 1 dispatch per command buffer → captures dispatch +
//!   wait overhead.
//! * **chained64**: 64 dispatches per command buffer, single wait at end
//!   → approximates the per-step economics of a 64-layer forward.
//!
//! Run all:           cargo bench -p qwen-llm --bench kernels
//! One filter:        cargo bench -p qwen-llm --bench kernels -- 'q4_k mat_vec/chained64/embed'

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::{MetalContext, MetalTensor, bench_q4_k_chained, bench_q6_k_chained};
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
            g.find(name)
                .filter(|t| t.dtype == GgmlType::Q4_K)
                .map(|t| (*label, t))
        })
        .collect()
}

fn pick_q6k_shapes(g: &GgufFile) -> Vec<(&'static str, &TensorDesc)> {
    let names = [
        ("attn_qkv_27b", "blk.0.attn_qkv.weight"),
        ("attn_v_27b", "blk.3.attn_v.weight"),
        ("ffn_down_27b", "blk.0.ffn_down.weight"),
        ("output_27b", "output.weight"),
    ];
    names
        .iter()
        .filter_map(|(label, name)| {
            g.find(name)
                .filter(|t| t.dtype == GgmlType::Q6_K)
                .map(|t| (*label, t))
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

        // Persistent tensors.
        let w_t = MetalTensor::from_gguf_tensor(&ctx, t, g.slice(t)).expect("w");
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 1e-3).sin()).collect();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x");
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y");

        group.throughput(Throughput::Bytes(t.n_bytes));
        group.bench_with_input(
            BenchmarkId::new("single", label),
            &(label, t.n_bytes),
            |b, _| {
                b.iter(|| {
                    bench_q4_k_chained(&ctx, &w_t, &x_t, &y_t, n_in, n_out, 1).expect("dispatch");
                    black_box(&y_t);
                });
            },
        );

        group.throughput(Throughput::Bytes(t.n_bytes * 64));
        group.bench_with_input(
            BenchmarkId::new("chained64", label),
            &(label, t.n_bytes),
            |b, _| {
                b.iter(|| {
                    bench_q4_k_chained(&ctx, &w_t, &x_t, &y_t, n_in, n_out, 64).expect("chained");
                    black_box(&y_t);
                });
            },
        );
    }
    group.finish();
}

fn bench_q6k_mat_vec(c: &mut Criterion) {
    if !std::path::Path::new(MODEL_27B).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    let g = GgufFile::open(MODEL_27B).expect("open");
    let shapes = pick_q6k_shapes(&g);

    let mut group = c.benchmark_group("q6_k mat_vec");
    for (label, t) in &shapes {
        let n_in = t.shape[0] as usize;
        let n_out = t.shape[1] as usize;

        let w_t = MetalTensor::from_gguf_tensor(&ctx, t, g.slice(t)).expect("w");
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 1e-3).sin()).collect();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x");
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y");

        group.throughput(Throughput::Bytes(t.n_bytes));
        group.bench_with_input(
            BenchmarkId::new("single", label),
            &(label, t.n_bytes),
            |b, _| {
                b.iter(|| {
                    bench_q6_k_chained(&ctx, &w_t, &x_t, &y_t, n_in, n_out, 1).expect("dispatch");
                    black_box(&y_t);
                });
            },
        );

        group.throughput(Throughput::Bytes(t.n_bytes * 64));
        group.bench_with_input(
            BenchmarkId::new("chained64", label),
            &(label, t.n_bytes),
            |b, _| {
                b.iter(|| {
                    bench_q6_k_chained(&ctx, &w_t, &x_t, &y_t, n_in, n_out, 64).expect("chained");
                    black_box(&y_t);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_q4k_mat_vec, bench_q6k_mat_vec);
criterion_main!(benches);
