//! Primitive capacity/coverage diagnostic; never loads a model or changes residency.
//! Run with a NEW output file and the ordinary production lease:
//! ```sh
//! env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
//!   IQ_CAPACITY_DTYPE=iq2_xs IQ_CAPACITY_OUT=/tmp/iq2-xs-capacity.jsonl \
//!   cargo test --release -p qwen-llm --lib \
//!   metal_forward::tests::iq_capacity::iq_capacity_packet \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```
//! IQ_CAPACITY_DTYPE selects iq2_xs, iq2_xxs, iq1_s or iq1_m. IQ1_M also
//! checks real embedding gather without an inflated embedding baseline. IQ_CAPACITY_GGUF
//! optionally overrides the documented Saluki artifact. Numerical differences
//! are reported, not automatically qualified; no bitwise or speed gate.

use crate::codec::{dequant_to_f32, dequant_to_f32_into};
use crate::gguf::GgufFile;
use crate::metal::*;
use crate::tensor::{GgmlType, TensorDesc};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Write, mem::MaybeUninit, time::Instant};

const ARTIFACT: &str =
    "/Volumes/wdblack/weights-archive/underdog-saluki-27b/Underdog-Saluki-27B-1.0-IQ2-mix.gguf";
const READ: usize = 256 * 1024; // F32 readback chunk, one MiB.
const DECODE_ROWS: usize = 16;
const META_CPU: u64 = 16 << 20; // Future diagnostic/codec/census bookkeeping.
const WIDTHS: [usize; 6] = [1, 2, 8, 32, 128, 512];
const ARMS: [(&str, bool); 6] = [
    ("warm_A", false),
    ("warm_B", true),
    ("B1", true),
    ("A1", false),
    ("A2", false),
    ("B2", true),
];

fn emit(out: &mut File, value: Value) {
    serde_json::to_writer(&mut *out, &value).unwrap();
    writeln!(out).unwrap();
    out.flush().unwrap();
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn anchors(n: usize) -> Vec<usize> {
    assert!(n > 0);
    let mut a = vec![0, n / 2, n - 1];
    a.dedup();
    a
}
fn dimensions(d: &TensorDesc) -> (usize, usize, usize) {
    assert_eq!(d.shape.len(), 2, "only ordinary matrices: {}", d.name);
    let (k, m) = (
        usize::try_from(d.shape[0]).unwrap(),
        usize::try_from(d.shape[1]).unwrap(),
    );
    assert!(k > 0 && k.is_multiple_of(256) && m >= 3);
    let expected_block_bytes = match d.dtype {
        GgmlType::IQ2_XS => 74,
        GgmlType::IQ2_XXS => 66,
        GgmlType::IQ1_S => 50,
        GgmlType::IQ1_M => 56,
        _ => panic!("cohort dtype"),
    };
    let (block_elements, block_bytes) = d.dtype.storage_layout().expect("canonical GGML layout");
    assert_eq!((block_elements, block_bytes), (256, expected_block_bytes));
    let row_bytes = k / block_elements as usize * block_bytes as usize;
    assert_eq!(d.n_bytes, (row_bytes * m) as u64);
    (k, m, row_bytes)
}
fn descriptor(d: &TensorDesc) -> Value {
    json!({"name":d.name,"shape":d.shape,"dtype":format!("{:?}",d.dtype),
        "shard":d.shard_idx,"offset":d.data_offset,"source_bytes":d.n_bytes,
        "converted_f32_bytes":d.n_elements()*4,"native_logical_bytes":d.n_bytes,
        "avoided_persistent_logical_bytes":d.n_elements()*4-d.n_bytes})
}

// Admission runs inside the caller's allocation transaction, before CPU/GPU
// payload allocation. Existing allocations are already in memory_signals().
fn admit(
    ctx: &MetalContext,
    out: &mut File,
    phase: &str,
    name: &str,
    gpu: &[(&str, u64)],
    cpu: &[(&str, u64)],
) {
    let priced: Vec<Value> = gpu
        .iter()
        .map(|(name, bytes)| {
            let p = ctx
                .price_shared_buffer_upper(*bytes)
                .expect("device buffer pricing");
            json!({"name":name,"logical_bytes":bytes,"priced_upper_bytes":p.priced_upper_bytes})
        })
        .collect();
    let gpu_bytes: u64 = priced
        .iter()
        .map(|v| v["priced_upper_bytes"].as_u64().unwrap())
        .sum();
    let cpu_bytes: u64 = cpu.iter().map(|(_, n)| n).sum();
    let signals = ctx.memory_signals();
    let a = evaluate_metal_memory_admission_with_cpu_bytes(gpu_bytes, cpu_bytes, 0, signals, true);
    emit(
        out,
        json!({"event":"admission","phase":phase,"tensor":name,"gpu_buffers":priced,
        "gpu_upper_bytes":gpu_bytes,"cpu_components":cpu,"cpu_upper_bytes":cpu_bytes,
        "current_metal_allocated_bytes":ctx.current_allocated_size(),
        "recommended_working_set_bytes":signals.recommended_max_bytes,
        "process_limit_remaining_bytes":signals.process_limit_remaining_bytes,
        "admitted":a.admitted,"reason":a.reason.as_str()}),
    );
    assert!(
        a.admitted,
        "ordinary memory admission refused; raw record flushed"
    );
}

// Host slices are used only before first dispatch or after wait_completed.
// All buffers below are owned shared storage; no retained mmap is mutated.
fn f32_values(t: &MetalTensor) -> &[f32] {
    assert_eq!(t.dtype, GgmlType::F32);
    assert!(t.offset.is_multiple_of(4));
    assert!(
        t.offset
            .checked_add(t.n_bytes())
            .is_some_and(|end| end <= t.buffer.length() as u64)
    );
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<f32>(),
            t.n_elements() as usize,
        )
    }
}
fn initialize(t: &mut MetalTensor, mut value: impl FnMut(usize) -> f32) {
    assert!(t.is_writable());
    assert_eq!(t.dtype, GgmlType::F32);
    assert_eq!(t.offset, 0);
    assert!(t.n_bytes() <= t.buffer.length() as u64);
    // SAFETY: exclusive initialization of owned shared storage; prior work waited.
    unsafe {
        let p = t.buffer.contents().as_ptr().cast::<f32>();
        for i in 0..t.n_elements() as usize {
            p.add(i).write(value(i));
        }
    }
}
fn input_value(index: usize) -> f32 {
    let mut x = (index as u32).wrapping_add(0x9e3779b9);
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    ((x & 65535) as f32 - 32768.0) / 32768.0
}

#[derive(Default)]
struct Difference {
    max_abs: f64,
    diff2: f64,
    ref2: f64,
    peak: f64,
}
impl Difference {
    fn add(&mut self, reference: f64, actual: f64) {
        assert!(reference.is_finite() && actual.is_finite());
        let delta = actual - reference;
        self.max_abs = self.max_abs.max(delta.abs());
        self.diff2 += delta * delta;
        self.ref2 += reference * reference;
        self.peak = self.peak.max(reference.abs());
    }
    fn json(&self) -> Value {
        json!({"max_abs":self.max_abs,"relative_l2":(self.diff2/self.ref2.max(1e-30)).sqrt(),
            "max_abs_over_reference_peak":self.max_abs/self.peak.max(1e-30),"reference_l2":self.ref2.sqrt()})
    }
}

#[allow(clippy::too_many_arguments)]
fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    n: usize,
) -> Result<(), MetalError> {
    // New native primitives validate rank as well as element count. Public
    // GEMV takes [K]/[M], whereas the mat-mat APIs take [K,N]/[M,N].
    let vectors = (n == 1).then(|| {
        (
            x.view_subrange(0, vec![k as u64]),
            y.view_subrange(0, vec![m as u64]),
        )
    });
    let (x, y) = vectors.as_ref().map(|(x, y)| (x, y)).unwrap_or((x, y));
    match (w.dtype, n == 1) {
        (GgmlType::F32, true) => encode_mat_vec_f32(ctx, enc, w, x, y, k, m),
        (GgmlType::F32, false) => encode_mat_mat_f32(ctx, enc, w, x, y, k, m, n),
        (GgmlType::IQ2_XS, true) => encode_mat_vec_iq2_xs_f32(ctx, enc, w, x, y, k, m),
        (GgmlType::IQ2_XS, false) => encode_mat_mat_iq2_xs_f32(ctx, enc, w, x, y, k, m, n),
        (GgmlType::IQ2_XXS, true) => encode_mat_vec_iq2_xxs_f32(ctx, enc, w, x, y, k, m),
        (GgmlType::IQ2_XXS, false) => encode_mat_mat_iq2_xxs_f32(ctx, enc, w, x, y, k, m, n),
        (GgmlType::IQ1_S | GgmlType::IQ1_M, true) => {
            encode_mat_vec_iq1_f32(ctx, enc, w, x, y, k, m)
        }
        (GgmlType::IQ1_S | GgmlType::IQ1_M, false) => {
            encode_mat_mat_iq1_f32(ctx, enc, w, x, y, k, m, n)
        }
        _ => panic!("unexpected diagnostic dtype"),
    }
}

fn command(
    ctx: &MetalContext,
    out: &mut File,
    mut event: Value,
    census: bool,
    f: impl FnOnce(&KernelEncoder) -> Result<(), MetalError>,
) {
    if census {
        dispatch_census_begin();
    }
    let start = Instant::now();
    let cmd = ctx.queue.commandBuffer().expect("ordinary command buffer");
    let enc = KernelEncoder::begin(&cmd);
    let result = f(&enc);
    enc.end();
    let result = result.and_then(|()| {
        cmd.commit();
        wait_completed(&cmd)
    });
    let wall_ms = start.elapsed().as_secs_f64() * 1e3;
    let (gpu_start, gpu_end) = if result.is_ok() {
        (cmd.GPUStartTime(), cmd.GPUEndTime())
    } else {
        (0.0, 0.0)
    };
    let gpu_ms = (gpu_end - gpu_start) * 1e3;
    let valid = result.is_ok()
        && gpu_start.is_finite()
        && gpu_start > 0.0
        && gpu_end.is_finite()
        && gpu_ms.is_finite()
        && gpu_ms > 0.0;
    let dispatches: Option<Vec<Value>> = census.then(|| dispatch_census_take().into_iter().map(|d|
        json!({"kernel":d.kernel,"tag":d.tag,"grid":[d.grid_width,d.grid_height,d.grid_depth],
            "threads":[d.threads_width,d.threads_height,d.threads_depth],"concurrent":d.encoder_concurrent})).collect());
    event.as_object_mut().unwrap().extend(
        json!({"wall_ms":wall_ms,"gpu_start_s":gpu_start,"gpu_end_s":gpu_end,
        "command_gpu_ms":valid.then_some(gpu_ms),"command_gpu_valid":valid,
        "dispatch_census":dispatches,"error":result.as_ref().err().map(ToString::to_string)})
        .as_object()
        .unwrap()
        .clone(),
    );
    emit(out, event);
    result.expect("encode/command failed; raw attempt flushed");
    assert!(valid, "invalid GPU timestamps; raw attempt flushed");
}

fn oracle_values(
    weights: &[f32],
    x: &[f32],
    k: usize,
    m: usize,
    n: usize,
    selected_rows: &[usize],
    selected_tokens: &[usize],
) -> Vec<(usize, f64)> {
    assert_eq!(weights.len(), k * m);
    assert_eq!(x.len(), k * n);
    let mut result = Vec::new();
    for &token in selected_tokens {
        for &row in selected_rows {
            let value = weights[row * k..(row + 1) * k]
                .iter()
                .zip(&x[token * k..(token + 1) * k])
                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                .sum::<f64>();
            assert!(value.is_finite());
            result.push((token * m + row, value));
        }
    }
    result
}

// Exactly one bounded traversal per arm; replacement B1 -> A1 happens in place.
fn outputs(
    out: &mut File,
    event: Value,
    y: &MetalTensor,
    oracle: &[(usize, f64)],
    reference: Option<&mut [f32]>,
    reference_label: Option<&str>,
    save: bool,
) {
    let mut reference = reference;
    let mut hash = Sha256::new();
    let mut difference = Difference::default();
    let mut oracle_difference = Difference::default();
    let mut nonfinite = 0usize;
    let values = f32_values(y);
    for (chunk_index, chunk) in values.chunks(READ).enumerate() {
        hash.update(bytemuck::cast_slice(chunk));
        for (j, &v) in chunk.iter().enumerate() {
            let i = chunk_index * READ + j;
            nonfinite += usize::from(!v.is_finite());
            if let Some(r) = reference.as_deref_mut() {
                if reference_label.is_some() && v.is_finite() {
                    difference.add(f64::from(r[i]), f64::from(v));
                }
                if save {
                    r[i] = v;
                }
            }
        }
    }
    for &(i, expected) in oracle {
        if values[i].is_finite() {
            oracle_difference.add(expected, f64::from(values[i]));
        }
    }
    let mut event = event;
    event.as_object_mut().unwrap().extend(
        json!({"event":"output_comparison","nonfinite":nonfinite,
        "sha256_f32_le":format!("{:x}",hash.finalize()),"elements":values.len(),
        "reference_label":reference_label,"difference":reference_label.map(|_|difference.json()),
        "cpu_codec_f64_samples":oracle.len(),"cpu_codec_f64_difference":oracle_difference.json(),
        "cpu_codec_f64_points":oracle.iter().map(|&(i,v)|json!({"index":i,"reference":v,"actual":values[i]})).collect::<Vec<_>>(),
        "numerical_decision":"unscored_no_bit_gate"})
        .as_object()
        .unwrap()
        .clone(),
    );
    emit(out, event);
    assert_eq!(
        nonfinite, 0,
        "nonfinite/unwritten output; raw record flushed"
    );
}

fn sample_tensor(ctx: &MetalContext, gguf: &GgufFile, d: &TensorDesc, out: &mut File) {
    let (k, m, row_bytes) = dimensions(d);
    let rows = anchors(m);
    let transaction = ctx.begin_allocation_transaction();
    admit(
        ctx,
        out,
        "all_tensor_sample",
        &d.name,
        &[
            ("native_three_rows", (3 * row_bytes) as u64),
            ("input_N3", (k * 3 * 4) as u64),
            ("output_M3_N3", 36),
        ],
        &[
            ("packed_rows_and_codec_alignment", (6 * row_bytes) as u64),
            ("decoded_three_rows", (3 * k * 4) as u64),
            ("diagnostic_metadata", META_CPU),
        ],
    );
    let mut bytes = vec![0u8; 3 * row_bytes];
    for (i, &row) in rows.iter().enumerate() {
        gguf.read_shard_exact_at(
            d.shard_idx,
            d.data_offset + (row * row_bytes) as u64,
            &mut bytes[i * row_bytes..(i + 1) * row_bytes],
        )
        .unwrap();
    }
    let small = TensorDesc {
        name: d.name.clone(),
        shape: vec![k as u64, 3],
        dtype: d.dtype,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: bytes.len() as u64,
    };
    let decoded = dequant_to_f32(&small, &bytes).unwrap();
    assert!(decoded.iter().all(|v| v.is_finite()));
    let mut w = MetalTensor::from_bytes(ctx, &bytes, small.shape.clone(), d.dtype).unwrap();
    w.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    let mut x = MetalTensor::zeros_f32(ctx, vec![k as u64, 3]).unwrap();
    initialize(&mut x, input_value);
    let mut y = MetalTensor::zeros_f32(ctx, vec![3, 3]).unwrap();
    drop(transaction);
    emit(
        out,
        json!({"event":"sample_binding","tensor":descriptor(d),"source_rows":rows,
        "source_row_offsets":rows.iter().map(|r|d.data_offset+(r*row_bytes) as u64).collect::<Vec<_>>(),
        "packed_sample_sha256":sha(&bytes),"decoded_sample_sha256_f32_le":sha(bytemuck::cast_slice(&decoded)),
        "input_N3_sha256_f32_le":sha(bytemuck::cast_slice(f32_values(&x))),
        "layout":"first/middle/last physical rows concatenated as [K,3]; synthetic fixed F32 activations, not captured model activations"}),
    );
    for n in [1usize, 3] {
        initialize(&mut y, |_| f32::NAN);
        let xv = x.view_subrange(0, vec![k as u64, n as u64]);
        let yv = y.view_subrange(0, vec![3, n as u64]);
        let oracle = oracle_values(
            &decoded,
            f32_values(&xv),
            k,
            3,
            n,
            &[0, 1, 2],
            &(0..n).collect::<Vec<_>>(),
        );
        command(
            ctx,
            out,
            json!({"event":"oracle_attempt","tensor":d.name,"K":k,"M":3,"N":n,"measured":false}),
            true,
            |enc| encode(ctx, enc, &w, &xv, &yv, k, 3, n),
        );
        outputs(
            out,
            json!({"phase":"all_tensor_sample","tensor":d.name,"N":n,"arm":"native"}),
            &yv,
            &oracle,
            None,
            None,
            false,
        );
    }
}

// Decode bounded row chunks directly into the F32 Metal destination. There is
// no model-sized CPU vector and no full compressed CPU payload copy.
fn load_pair(
    ctx: &MetalContext,
    gguf: &GgufFile,
    d: &TensorDesc,
    out: &mut File,
) -> (MetalTensor, MetalTensor) {
    let (k, m, row_bytes) = dimensions(d);
    let start = Instant::now();
    let mut native = MetalTensor {
        buffer: ctx.buffer_uninit(d.n_bytes as usize).unwrap(),
        offset: 0,
        shape: d.shape.clone(),
        dtype: d.dtype,
        provenance: MetalTensorProvenance::OwnedWritable,
    };
    let mut decoded = MetalTensor::zeros_f32(ctx, d.shape.clone()).unwrap();
    let mut bytes = vec![0u8; row_bytes * DECODE_ROWS];
    let mut hash = Sha256::new();
    for row in (0..m).step_by(DECODE_ROWS) {
        let count = DECODE_ROWS.min(m - row);
        let bytes = &mut bytes[..count * row_bytes];
        gguf.read_shard_exact_at(d.shard_idx, d.data_offset + (row * row_bytes) as u64, bytes)
            .unwrap();
        hash.update(&*bytes);
        let chunk = TensorDesc {
            name: d.name.clone(),
            shape: vec![k as u64, count as u64],
            dtype: d.dtype,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: bytes.len() as u64,
        };
        // SAFETY: newly allocated shared buffers, exclusive initialization,
        // checked full tensor dimensions; row ranges partition each destination.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                native
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(row * row_bytes),
                bytes.len(),
            );
            let destination = std::slice::from_raw_parts_mut(
                decoded
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<MaybeUninit<f32>>()
                    .add(row * k),
                count * k,
            );
            dequant_to_f32_into(&chunk, bytes, destination).unwrap();
        }
    }
    assert!(f32_values(&decoded).iter().all(|v| v.is_finite()));
    native.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    decoded.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    emit(
        out,
        json!({"event":"representative_loaded","tensor":descriptor(d),"source_payload_sha256":format!("{:x}",hash.finalize()),
        "load_and_decode_wall_ms":start.elapsed().as_secs_f64()*1e3,
        "native_buffer_bytes":native.buffer.length(),"f32_buffer_bytes":decoded.buffer.length(),
        "decode_rows_per_chunk":DECODE_ROWS,"full_cpu_f32_weight_bytes":0,"full_cpu_compressed_weight_bytes":0,
        "provenance":"both owned weight read-only after initialization; retained descriptor positional reads"}),
    );
    (native, decoded)
}

fn representative(ctx: &MetalContext, gguf: &GgufFile, d: &TensorDesc, out: &mut File) {
    let (k, m, row_bytes) = dimensions(d);
    let transaction = ctx.begin_allocation_transaction();
    admit(
        ctx,
        out,
        "representative",
        &d.name,
        &[
            ("native_weight", d.n_bytes),
            ("decoded_f32_weight", d.n_elements() * 4),
            ("input_N512", (k * 512 * 4) as u64),
            ("output_N512", (m * 512 * 4) as u64),
        ],
        &[
            ("one_output_reference_N512", (m * 512 * 4) as u64),
            (
                "decode_chunk_plus_codec_alignment",
                (2 * row_bytes * DECODE_ROWS) as u64,
            ),
            ("readback_allowance", (READ * 4) as u64),
            ("diagnostic_metadata", META_CPU),
        ],
    );
    let (native, decoded) = load_pair(ctx, gguf, d, out);
    let mut x = MetalTensor::zeros_f32(ctx, vec![k as u64, 512]).unwrap();
    initialize(&mut x, input_value);
    let y = MetalTensor::zeros_f32(ctx, vec![m as u64, 512]).unwrap();
    let mut reference = vec![0.125f32; m * 512];
    drop(transaction);
    for n in WIDTHS {
        let xv = x.view_subrange(0, vec![k as u64, n as u64]);
        let mut yv = y.view_subrange(0, vec![m as u64, n as u64]);
        let oracle = oracle_values(
            f32_values(&decoded),
            f32_values(&xv),
            k,
            m,
            n,
            &anchors(m),
            &anchors(n),
        );
        emit(
            out,
            json!({"event":"performance_binding","tensor":d.name,"K":k,"M":m,"N":n,
            "input_sha256_f32_le":sha(bytemuck::cast_slice(f32_values(&xv))),
            "cpu_oracle_output_indices":oracle.iter().map(|(i,_)|i).collect::<Vec<_>>(),
            "cpu_oracle_values_f64":oracle.iter().map(|(_,v)|v).collect::<Vec<_>>(),
            "metal_allocated_bytes":ctx.current_allocated_size()}),
        );
        let mut reference_label = None;
        for (label, b) in ARMS {
            // Poison the active output before every arm, outside all timing.
            // yv aliases y, but there is no active GPU command or host slice.
            initialize(&mut yv, |_| f32::NAN);
            eprintln!("IQ capacity {} N{n} {label}", d.name);
            command(
                ctx,
                out,
                json!({"event":"performance_attempt","tensor":d.name,"K":k,"M":m,"N":n,
                "label":label,"arm":if b {"native"} else {"decoded_f32"},"measured":!label.starts_with("warm_")}),
                label.starts_with("warm_"),
                |enc| {
                    encode(
                        ctx,
                        enc,
                        if b { &native } else { &decoded },
                        &xv,
                        &yv,
                        k,
                        m,
                        n,
                    )
                },
            );
            let save = matches!(label, "B1" | "A1");
            outputs(
                out,
                json!({"phase":"representative","tensor":d.name,"N":n,"label":label,"arm":if b {"native"} else {"decoded_f32"}}),
                &yv,
                &oracle,
                Some(&mut reference[..m * n]),
                reference_label,
                save,
            );
            if save {
                reference_label = Some(label);
            }
        }
    }
}

// Full compressed embedding only. Retained positional reads avoid a full CPU
// payload copy; each CPU oracle decodes just one actual vocabulary row.
fn embedding_gather(
    ctx: &MetalContext,
    gguf: &GgufFile,
    d: &TensorDesc,
    out: &mut File,
    corpus_ids: &[i32],
    corpus_origins: &[Value],
) {
    assert_eq!(d.dtype, GgmlType::IQ1_M);
    let (k, vocab, row_bytes) = dimensions(d);
    let ids: Vec<i32> = [0, vocab / 2, vocab - 1, vocab / 2, 0, vocab - 1]
        .into_iter()
        .map(|id| i32::try_from(id).unwrap())
        .collect();
    assert_eq!(corpus_ids.len(), ids.len());
    assert_eq!(corpus_origins.len(), corpus_ids.len());
    assert!(
        corpus_ids
            .iter()
            .all(|&id| id >= 0 && (id as usize) < vocab)
    );
    let transaction = ctx.begin_allocation_transaction();
    admit(
        ctx,
        out,
        "embedding_gather",
        &d.name,
        &[
            ("native_embedding", d.n_bytes),
            ("ids_N6", (ids.len() * 4) as u64),
            ("output_N6", (k * ids.len() * 4) as u64),
        ],
        &[
            ("compressed_read_chunk", (READ * 4) as u64),
            (
                "one_compressed_row_and_codec_alignment",
                (2 * row_bytes) as u64,
            ),
            ("one_decoded_row", (k * 4) as u64),
            ("diagnostic_metadata", META_CPU),
        ],
    );
    let mut native = MetalTensor {
        buffer: ctx
            .buffer_uninit(usize::try_from(d.n_bytes).unwrap())
            .unwrap(),
        offset: 0,
        shape: d.shape.clone(),
        dtype: d.dtype,
        provenance: MetalTensorProvenance::OwnedWritable,
    };
    let mut bytes = vec![0u8; READ * 4];
    let mut hash = Sha256::new();
    let mut offset = 0u64;
    while offset < d.n_bytes {
        let count = (d.n_bytes - offset).min(bytes.len() as u64) as usize;
        gguf.read_shard_exact_at(d.shard_idx, d.data_offset + offset, &mut bytes[..count])
            .unwrap();
        hash.update(&bytes[..count]);
        // SAFETY: exclusive initialization; the checked range is within the
        // newly allocated native buffer and no command references it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                native
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(offset as usize),
                count,
            );
        }
        offset += count as u64;
    }
    drop(bytes);
    native.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    let ids_gpu = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&ids),
        vec![ids.len() as u64],
        GgmlType::I32,
    )
    .unwrap();
    let y = MetalTensor::zeros_f32(ctx, vec![k as u64, ids.len() as u64]).unwrap();
    drop(transaction);
    emit(
        out,
        json!({"event":"embedding_gather_binding","tensor":descriptor(d),"ids":ids,
        "corpus_ids":corpus_ids,"corpus_origins":corpus_origins,
        "corpus_ids_sha256_i32_le":sha(bytemuck::cast_slice(corpus_ids)),
        "source_payload_sha256":format!("{:x}",hash.finalize()),"native_buffer_bytes":native.buffer.length(),
        "full_cpu_compressed_weight_bytes":0,"full_cpu_f32_weight_bytes":0,"f32_gpu_embedding_bytes":0,
        "read_chunk_bytes":READ*4,"oracle":"canonical CPU codec, one selected vocabulary row at a time"}),
    );
    let mut compressed_row = vec![0u8; row_bytes];
    let row_desc = TensorDesc {
        name: d.name.clone(),
        shape: vec![k as u64, 1],
        dtype: d.dtype,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: row_bytes as u64,
    };
    let probes = [
        ("anchors_N1", &ids[..1], None),
        ("anchors_N6", ids.as_slice(), None),
        ("whole_corpus_N6", corpus_ids, Some(corpus_origins)),
    ];
    for (probe, batch_ids, origins) in probes {
        let n = batch_ids.len();
        // SAFETY: owned writable IDs buffer; n<=6 was checked, and the prior
        // command completed before this overwrite. Reuse the same allocations.
        unsafe {
            std::ptr::copy_nonoverlapping(
                batch_ids.as_ptr(),
                ids_gpu.buffer.contents().as_ptr().cast::<i32>(),
                n,
            );
        }
        let id_view = ids_gpu.view_subrange(0, vec![n as u64]);
        let mut y_view = y.view_subrange(0, vec![k as u64, n as u64]);
        initialize(&mut y_view, |_| f32::NAN);
        command(
            ctx,
            out,
            json!({"event":"embedding_gather_attempt","tensor":d.name,"N":n,"K":k,
            "probe":probe,"ids":batch_ids,"ids_sha256_i32_le":sha(bytemuck::cast_slice(batch_ids)),
            "origins":origins,"measured":false,"comparison":"CPU selected-row oracle; every row coefficient, no F32 GPU table"}),
            true,
            |enc| encode_get_rows_iq1_m_f32(ctx, enc, &native, &id_view, &y_view, n, k),
        );
        let values = f32_values(&y_view);
        for (slot, &id) in batch_ids.iter().enumerate() {
            let source_offset = d.data_offset + id as u64 * row_bytes as u64;
            gguf.read_shard_exact_at(d.shard_idx, source_offset, &mut compressed_row)
                .unwrap();
            let decoded = dequant_to_f32(&row_desc, &compressed_row).unwrap();
            let actual = &values[slot * k..(slot + 1) * k];
            let mut difference = Difference::default();
            let mut nonfinite = 0;
            let mut unequal_values = 0;
            for (&reference, &value) in decoded.iter().zip(actual) {
                if reference.is_finite() && value.is_finite() {
                    difference.add(f64::from(reference), f64::from(value));
                    unequal_values += usize::from(reference != value);
                } else {
                    nonfinite += 1;
                }
            }
            emit(
                out,
                json!({"event":"embedding_gather_comparison","tensor":d.name,"N":n,"slot":slot,"id":id,
                "probe":probe,"origin":origins.map(|o| &o[slot]),
                "source_offset":source_offset,"source_row_sha256":sha(&compressed_row),
                "cpu_f32_sha256":sha(bytemuck::cast_slice(&decoded)),"native_f32_sha256":sha(bytemuck::cast_slice(actual)),
                "elements":k,"nonfinite_pairs":nonfinite,"unequal_values":unequal_values,"difference":difference.json(),
                "numerical_decision":"unscored_no_bit_or_error_threshold_gate"}),
            );
            assert_eq!(
                nonfinite, 0,
                "nonfinite embedding gather; raw comparison flushed"
            );
        }
    }
}

#[test]
fn iq_capacity_cpu_bookkeeping() {
    assert_eq!(anchors(1), vec![0]);
    assert_eq!(anchors(3), vec![0, 1, 2]);
    assert_eq!(anchors(512), vec![0, 256, 511]);
    let oracle = oracle_values(
        &[1.0, 2.0, -3.0, 4.0],
        &[2.0, 1.0, 0.0, 3.0],
        2,
        2,
        2,
        &[0, 1],
        &[0, 1],
    );
    assert_eq!(oracle, vec![(0, 4.0), (1, -2.0), (2, 6.0), (3, 12.0)]);
    let mut d = Difference::default();
    d.add(f64::from(f32::MAX), -f64::from(f32::MAX));
    assert!(d.diff2.is_finite());
    assert_eq!(d.json()["relative_l2"], json!(2.0));
}

#[test]
#[ignore = "Saluki retained GGUF; release; production GPU lease; new IQ_CAPACITY_OUT"]
fn iq_capacity_packet() {
    assert!(!cfg!(debug_assertions), "release diagnostic required");
    assert!(
        std::env::var_os("MTL_DEBUG_LAYER").is_none(),
        "timing requires no Metal debug layer"
    );
    let dtype = match std::env::var("IQ_CAPACITY_DTYPE").as_deref() {
        Ok("iq2_xs") => GgmlType::IQ2_XS,
        Ok("iq2_xxs") => GgmlType::IQ2_XXS,
        Ok("iq1_s") => GgmlType::IQ1_S,
        Ok("iq1_m") => GgmlType::IQ1_M,
        _ => panic!("IQ_CAPACITY_DTYPE must be iq2_xs, iq2_xxs, iq1_s or iq1_m"),
    };
    let mut out = File::options()
        .write(true)
        .create_new(true)
        .open(std::env::var_os("IQ_CAPACITY_OUT").expect("IQ_CAPACITY_OUT required"))
        .expect("new output file required");
    let path = std::env::var_os("IQ_CAPACITY_GGUF").unwrap_or_else(|| ARTIFACT.into());
    let gguf = GgufFile::open(path).unwrap();
    // Same metadata-only vocabulary initialization as the whole packet, before
    // Metal context/admission. The helper drops the tokenizer and full streams.
    let gather_probe = (dtype == GgmlType::IQ1_M)
        .then(|| super::iq_capacity_model::qualification_gather_ids(&gguf));
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let _lease = acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = MetalContext::new().unwrap();
    let cohort: Vec<_> = gguf.tensors.iter().filter(|d| d.dtype == dtype).collect();
    let expected = match dtype {
        GgmlType::IQ2_XS => 39,
        GgmlType::IQ2_XXS => 119,
        GgmlType::IQ1_S => 36,
        GgmlType::IQ1_M => 31,
        _ => unreachable!(),
    };
    let inventory: Vec<_> = cohort.iter().map(|d| descriptor(d)).collect();
    let shapes = match dtype {
        GgmlType::IQ2_XS => [
            ("ffn_gate.weight", 5120, 17408),
            ("ssm_out.weight", 6144, 5120),
        ],
        GgmlType::IQ2_XXS => [
            ("ffn_down.weight", 17408, 5120),
            ("ffn_gate.weight", 5120, 17408),
        ],
        GgmlType::IQ1_S => [
            ("ffn_gate.weight", 5120, 17408),
            ("ffn_down.weight", 17408, 5120),
        ],
        GgmlType::IQ1_M => [
            ("ffn_up.weight", 5120, 17408),
            ("ffn_down.weight", 17408, 5120),
        ],
        _ => unreachable!(),
    };
    let representatives: Vec<_> = shapes
        .iter()
        .map(|(suffix, k, m)| {
            *cohort
                .iter()
                .find(|d| d.name.ends_with(*suffix) && d.shape == [*k as u64, *m as u64])
                .unwrap_or_else(|| panic!("missing {dtype:?} {suffix} [{k},{m}]"))
        })
        .collect();
    let shards:Vec<_>=stamps.iter().map(|s|json!({"path":s.path,"shard":s.shard_idx,"device":s.device,"inode":s.inode,"bytes":s.size,
        "mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,"ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec})).collect();
    let source: u64 = cohort.iter().map(|d| d.n_bytes).sum();
    let inflated: u64 = cohort.iter().map(|d| d.n_elements() * 4).sum();
    let kernel_env: Vec<_> = ["QWEN_MATVEC_F32_LCPP_R2", "QWEN_MATVEC_IQ2_XS_FAST"]
        .into_iter()
        .map(|name| (name, std::env::var(name).ok()))
        .collect();
    let source_binding: Value = json!({"packet":sha(include_bytes!("iq_capacity.rs")),"codec":sha(include_bytes!("../../codec.rs")),
            "model_packet_corpus":sha(include_bytes!("iq_capacity_model.rs")),"tokenizer":sha(include_bytes!("../../tokenizer.rs")),
            "gguf":sha(include_bytes!("../../gguf.rs")),"metal_module":sha(include_bytes!("../../metal/mod.rs")),
            "mat_vec":sha(include_bytes!("../../metal/mat_vec.rs")),"mat_mat":sha(include_bytes!("../../metal/mat_mat.rs")),
            "xxs_dispatch":sha(include_bytes!("../../metal/iq2_xxs.rs")),
            "xxs_kernel":sha(include_bytes!("../../../../../kernels/iq2_xxs.metal")),
            "xxs_grid":sha(include_bytes!("../../../../../kernels/iq2_xxs_grid.metalh")),
            "iq1_dispatch":sha(include_bytes!("../../metal/iq1.rs")),
            "iq1_kernel_decoder":sha(include_bytes!("../../../../../kernels/iq1.metal")),
            "iq1_grid":sha(include_bytes!("../../../../../kernels/iq1_grid.metalh")),
            "xs_grid":sha(include_bytes!("../../../../../kernels/iq2_xs_grid.metalh")),
            "gemv_metal":sha(include_bytes!("../../../../../kernels/mat_vec.metal")),
            "dispatch":sha(include_bytes!("../dispatch.rs")),"xs_gemm":sha(include_bytes!("../../../../../kernels/mat_mat_iq2_xs.metal")),
            "metallib":sha(crate::KERNELS_METALLIB)});
    emit(
        &mut out,
        json!({"event":"header","schema":"iq_capacity.primitive.v2","device":ctx.describe(),"dtype":format!("{:?}",dtype),
        "canonical_storage_layout":dtype.storage_layout(),"embedding_gather":dtype==GgmlType::IQ1_M,
        "cohort_count":cohort.len(),"expected_saluki_count":expected,"shards":shards,
        "cohort_source_bytes":source,"cohort_converted_f32_bytes":inflated,"cohort_native_logical_bytes":source,
        "cohort_avoidable_persistent_logical_bytes":inflated-source,
        "inventory_sha256":sha(&serde_json::to_vec(&inventory).unwrap()),"inventory":inventory,
        "representatives":representatives.iter().map(|d|d.name.as_str()).collect::<Vec<_>>(),"widths":WIDTHS,
        "order":["warm_A","warm_B","B1","A1","A2","B2"],
        "policy":"A canonical CPU-decoded F32 via public F32 GEMV/GEMM; B public native IQ GEMV/GEMM including its automatic selection; no residency-policy dependency",
        "comparisons":"A1 versus B1, then A2 repeat/B2 native versus A1; each arm also has sampled CPU-codec F64 dot products",
        "scope":"all selected tensors including embedding: 3 physical rows, N1/N3; performance: two whole representative matrices; IQ1_M gather: anchor N1/N6 plus one N6 actual whole-corpus token batch, CPU every-coefficient oracle, no full F32 table; tokenizer metadata only, no model/session/loader realization",
        "capacity":"logical avoided inflation, not measured full-model RSS; both representative weight formats coexist solely for A/B; no whole CPU weight copies",
        "timing":"ordinary single command per attempt; warm arms census on, measured arms census off; initialization/readback/CPU oracle excluded; GPU time includes stalls",
        "decision":"unscored_diagnostic_no_promotion_or_bit_gate",
        "xxs_selection":"public Auto with capability fallback; no environment override",
        "env":kernel_env,
        "source_binding":source_binding}),
    );
    assert_eq!(
        cohort.len(),
        expected,
        "artifact cohort differs; inventory flushed"
    );
    // Bind header bytes with retained positional reads; never hash/touch the
    // whole mmap or imply that sampled payload hashes cover the full artifact.
    {
        let transaction = ctx.begin_allocation_transaction();
        admit(
            &ctx,
            &mut out,
            "header_hash",
            "all_shards",
            &[],
            &[("read_chunk", (READ * 4) as u64), ("metadata", META_CPU)],
        );
        let mut bytes = vec![0u8; READ * 4];
        drop(transaction);
        for (i, shard) in gguf.shards.iter().enumerate() {
            let mut h = Sha256::new();
            let mut offset = 0u64;
            while offset < shard.tensor_data_start {
                let n = (shard.tensor_data_start - offset).min(bytes.len() as u64) as usize;
                gguf.read_shard_exact_at(i, offset, &mut bytes[..n])
                    .unwrap();
                h.update(&bytes[..n]);
                offset += n as u64;
            }
            emit(
                &mut out,
                json!({"event":"header_binding","shard":i,"bytes":offset,"sha256":format!("{:x}",h.finalize()),"includes_alignment_padding":true}),
            );
        }
    }
    for d in &cohort {
        sample_tensor(&ctx, &gguf, d, &mut out);
    }
    emit(
        &mut out,
        json!({"event":"oracle_phase_complete","tensors":cohort.len(),"numerical_decision":"inspect_reported_errors"}),
    );
    for d in representatives {
        representative(&ctx, &gguf, d, &mut out);
    }
    if dtype == GgmlType::IQ1_M {
        let embedding = cohort
            .iter()
            .find(|d| d.name == "token_embd.weight")
            .expect("IQ1_M cohort must contain the real token embedding");
        let (ids, origins) = gather_probe
            .as_ref()
            .expect("whole-corpus gather IDs prepared");
        embedding_gather(&ctx, &gguf, embedding, &mut out, ids, origins);
    }
    assert_eq!(stamps, gguf.revalidate_retained_shard_stamps().unwrap());
    emit(
        &mut out,
        json!({"event":"complete","schema":"iq_capacity.primitive.v2","tensors_sampled":cohort.len(),
        "performance_attempts":72,"embedding_gather_attempts":if dtype==GgmlType::IQ1_M {3} else {0},
        "decision":"unscored_diagnostic_no_model_capacity_or_quality_claim"}),
    );
}
