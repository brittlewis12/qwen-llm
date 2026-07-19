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
use qwen_llm::metal::{
    MetalContext, MetalTensor, Trellis3Variant, bench_q4_k_chained, bench_q4_k_mat_mat_chained,
    bench_q6_k_chained, bench_trellis3_chained, trellis3_compressed_bytes, trellis3_synthetic,
    trellis3_upload,
};
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
        let w_t =
            MetalTensor::from_gguf_tensor(&ctx, t, g.try_slice(t).expect("slice")).expect("w");
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

        let w_t =
            MetalTensor::from_gguf_tensor(&ctx, t, g.try_slice(t).expect("slice")).expect("w");
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

/// H5.3b.1-3 isolated bench for the lifted Q4_K mat-mat kernel.
///
/// For each production-shape Q4_K weight, we measure THREE timings at
/// the DFlash N_QUERY=16 use case:
///
///   * `mat_vec_n_query_times` — N_QUERY=16 successive single-row mat-vec
///     dispatches in ONE command buffer. The naive H5.3a baseline
///     (matches what `packed_verify` does today for these weights).
///     Throughput: weight bytes × N_QUERY (we re-read weights N times).
///
///   * `mat_mat_single` — ONE mat-mat dispatch covering N_QUERY=16 cols.
///     Throughput: weight bytes × 1 (weights read ONCE; activations &
///     output scale with N_QUERY, but N_QUERY × n_in × 4 B is < 1% of
///     weight bytes at our shapes).
///
///   * `mat_mat_chained64` — 64 mat-mat dispatches in ONE command buffer
///     (approximates the steady-state per-step cost of doing one mat-mat
///     at every layer of a 64-layer forward).
///
/// The HEADLINE NUMBER is the wall-time RATIO of `mat_vec_n_query_times`
/// to `mat_mat_single` at the same weight shape. Per H5.3b plan rev 6,
/// we expect ≈ N_QUERY-fold speedup if the kernel hits the BW ceiling.
/// Anything < 4× at N_QUERY=16 means the lifted tile underdelivers and
/// we need to revisit (smaller tile, partial-tile path overhead, etc).
fn bench_q4k_mat_mat(c: &mut Criterion) {
    if !std::path::Path::new(MODEL_27B).exists() {
        eprintln!("[bench] skipping q4_k mat_mat — {MODEL_27B} not present");
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

    const N_QUERY: usize = 16; // matches DFlash block_size

    let mut group = c.benchmark_group("q4_k mat_mat");
    for (label, t) in &shapes {
        let n_in = t.shape[0] as usize;
        let n_out = t.shape[1] as usize;

        // Skip shapes whose n_out isn't a multiple of NR0_MM=64; the lifted
        // tile requires this for correctness without a partial-row path.
        if n_out % 64 != 0 {
            eprintln!("[bench q4_k mat_mat] skip {label} (n_out={n_out} not % 64)");
            continue;
        }

        // Persistent tensors.
        let w_t =
            MetalTensor::from_gguf_tensor(&ctx, t, g.try_slice(t).expect("slice")).expect("w");
        // Mat-vec activation: just one row.
        let x_vec: Vec<f32> = (0..n_in).map(|i| (i as f32 * 1e-3).sin()).collect();
        let x_vec_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x_vec),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x_vec");
        let y_vec_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y_vec");

        // Mat-mat activation: [N_QUERY, n_in] row-major.
        let x_mat: Vec<f32> = (0..N_QUERY * n_in)
            .map(|i| (i as f32 * 1e-3).sin())
            .collect();
        let x_mat_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x_mat),
            vec![N_QUERY as u64, n_in as u64],
            GgmlType::F32,
        )
        .expect("x_mat");
        let y_mat_t =
            MetalTensor::zeros_f32(&ctx, vec![n_out as u64, N_QUERY as u64]).expect("y_mat");

        // Baseline: N_QUERY successive mat-vec dispatches in one cmd buffer.
        // This is what naive H5.3a packed_verify does at this weight today.
        // Throughput: weight bytes × N_QUERY (weights re-read N times).
        group.throughput(Throughput::Bytes(t.n_bytes * N_QUERY as u64));
        group.bench_with_input(
            BenchmarkId::new("mat_vec_n_query_times", label),
            &(label, t.n_bytes),
            |b, _| {
                b.iter(|| {
                    bench_q4_k_chained(&ctx, &w_t, &x_vec_t, &y_vec_t, n_in, n_out, N_QUERY)
                        .expect("dispatch");
                    black_box(&y_vec_t);
                });
            },
        );

        // Single mat-mat at N_QUERY=16. Throughput: weight bytes × 1
        // (weights read ONCE; ratio to mat_vec_n_query_times in wall time
        // is the headline H5.3b.1-3 number).
        group.throughput(Throughput::Bytes(t.n_bytes));
        group.bench_with_input(
            BenchmarkId::new("mat_mat_single", label),
            &(label, t.n_bytes),
            |b, _| {
                b.iter(|| {
                    bench_q4_k_mat_mat_chained(
                        &ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 1,
                    )
                    .expect("dispatch");
                    black_box(&y_mat_t);
                });
            },
        );

        // Steady-state chained64 — approximates one mat-mat at every layer
        // of a 64-layer forward.
        group.throughput(Throughput::Bytes(t.n_bytes * 64));
        group.bench_with_input(
            BenchmarkId::new("mat_mat_chained64", label),
            &(label, t.n_bytes),
            |b, _| {
                b.iter(|| {
                    bench_q4_k_mat_mat_chained(
                        &ctx, &w_t, &x_mat_t, &y_mat_t, n_in, n_out, N_QUERY, 64,
                    )
                    .expect("chained");
                    black_box(&y_mat_t);
                });
            },
        );
    }
    group.finish();
}

/// Trellis3 decode floor (docs/bench/2026-07-19-trellis3-gemv-floor/).
///
/// Synthetic buffers (no model file needed). Same harness discipline as
/// `q4_k mat_vec`: persistent tensors, single + chained64 regimes.
/// Throughput charges compressed weight+scale bytes; the Lut8x2 device
/// LUT (1 KB) is deliberately uncharged and shows up as reduced achieved
/// GB/s instead.
fn bench_trellis3_mat_vec(c: &mut Criterion) {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[bench] no metal context: {e}");
            return;
        }
    };
    // Primary cell: ffn_gate class (5120 x 17408). Diagnostic: transpose
    // class (17408 x 5120, ffn_down-like).
    // embed_t3_27b (476 MB) is an SLC-uncacheable control row: the two
    // production shapes' 33.4 MB buffers could partially ride the ~48 MB
    // SLC across chained dispatches, so the big row guards the primary
    // result against cache inflation (diagnostic, not the gate cell).
    let shapes: [(&str, usize, usize); 3] = [
        ("ffn_gate_27b", 5120, 17408),
        ("ffn_down_t_27b", 17408, 5120),
        ("embed_t3_27b", 5120, 248320),
    ];
    let variants = [
        Trellis3Variant::ThreeInst,
        Trellis3Variant::ThreeInstV2,
        Trellis3Variant::Lut8x2,
        Trellis3Variant::HybV2,
    ];

    let mut group = c.benchmark_group("trellis3 mat_vec");
    for (label, n_in, n_out) in shapes {
        let syn = trellis3_synthetic(n_in, n_out, 0xF10D);
        let (w_t, s_t, l_t) = trellis3_upload(&ctx, &syn).expect("upload");
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 1e-3).sin()).collect();
        let x_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&x),
            vec![n_in as u64],
            GgmlType::F32,
        )
        .expect("x");
        let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).expect("y");
        let bytes = trellis3_compressed_bytes(n_in, n_out);

        for variant in variants {
            let row = format!("{}/{label}", variant.label());
            group.throughput(Throughput::Bytes(bytes));
            group.bench_with_input(BenchmarkId::new("single", &row), &row, |b, _| {
                b.iter(|| {
                    bench_trellis3_chained(
                        &ctx,
                        variant,
                        &w_t,
                        &s_t,
                        Some(&l_t),
                        &x_t,
                        &y_t,
                        n_in,
                        n_out,
                        1,
                    )
                    .expect("dispatch");
                    black_box(&y_t);
                });
            });

            group.throughput(Throughput::Bytes(bytes * 64));
            group.bench_with_input(BenchmarkId::new("chained64", &row), &row, |b, _| {
                b.iter(|| {
                    bench_trellis3_chained(
                        &ctx,
                        variant,
                        &w_t,
                        &s_t,
                        Some(&l_t),
                        &x_t,
                        &y_t,
                        n_in,
                        n_out,
                        64,
                    )
                    .expect("chained");
                    black_box(&y_t);
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_q4k_mat_vec,
    bench_q6k_mat_vec,
    bench_q4k_mat_mat,
    bench_trellis3_mat_vec
);
criterion_main!(benches);
