//! Native IQ2_XS leaf qualification with explicit Scalar versus Mma/Auto test scopes.
//! ```sh
//! env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
//!   IQ2_XS_MMA_OUT=/tmp/saluki-iq2-xs-mma-leaf.jsonl \
//!   cargo test --release -p qwen-llm --lib \
//!   metal_forward::tests::iq2_xs_mma::iq2_xs_mma_packet \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```
//! Optional IQ2_XS_MMA_GGUF overrides the artifact; IQ2_XS_MMA_ROUNDS defaults
//! to two ABBA rounds per width; IQ2_XS_MMA_CANDIDATE=mma|auto defaults to strict
//! MMA. Auto records actual substitutions, including capability fallback.
//! No full-model loading or F32 weight bank. Numerical results are unscored.

use crate::codec::dequant_to_f32;
use crate::gguf::GgufFile;
use crate::metal::*;
use crate::tensor::{GgmlType, TensorDesc};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Write, time::Instant};

const ARTIFACT: &str =
    "/Volumes/wdblack/weights-archive/underdog-saluki-27b/Underdog-Saluki-27B-1.0-IQ2-mix.gguf";
const WIDTHS: [usize; 11] = [1, 2, 8, 31, 32, 33, 127, 128, 129, 512, 1024];
const MAX_N: usize = 1024;
const READ_BYTES: usize = 1 << 20;
const META_CPU: u64 = 16 << 20;
const ABBA: [(&str, bool); 4] = [("A1", false), ("B1", true), ("B2", true), ("A2", false)];

fn emit(out: &mut File, event: Value) {
    serde_json::to_writer(&mut *out, &event).unwrap();
    writeln!(out).unwrap();
    out.flush().unwrap();
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn anchors(n: usize) -> Vec<usize> {
    assert!(n > 0);
    let mut result = vec![0, n / 2, n - 1];
    result.dedup();
    result
}
fn descriptor(d: &TensorDesc) -> Value {
    json!({"name":d.name,"dtype":format!("{:?}",d.dtype),"shape":d.shape,
        "shard":d.shard_idx,"offset":d.data_offset,"native_bytes":d.n_bytes})
}
fn dimensions(d: &TensorDesc) -> (usize, usize, usize) {
    assert_eq!(d.dtype, GgmlType::IQ2_XS);
    assert_eq!(d.dtype.storage_layout(), Some((256, 74)));
    assert_eq!(d.shape.len(), 2);
    let (k, m) = (
        usize::try_from(d.shape[0]).unwrap(),
        usize::try_from(d.shape[1]).unwrap(),
    );
    assert!(k > 0 && k.is_multiple_of(256) && m >= 3);
    let row_bytes = (k / 256).checked_mul(74).unwrap();
    assert_eq!(d.n_bytes, row_bytes.checked_mul(m).unwrap() as u64);
    (k, m, row_bytes)
}
fn admit(
    ctx: &MetalContext,
    out: &mut File,
    phase: &str,
    gpu: &[(&str, u64)],
    cpu: &[(&str, u64)],
) {
    let buffers: Vec<_> = gpu
        .iter()
        .map(|(name, bytes)| {
            let price = ctx
                .price_shared_buffer_upper(*bytes)
                .unwrap()
                .priced_upper_bytes;
            json!({"name":name,"logical_bytes":bytes,"priced_upper_bytes":price})
        })
        .collect();
    let gpu_bytes = buffers
        .iter()
        .map(|b| b["priced_upper_bytes"].as_u64().unwrap())
        .sum();
    let cpu_bytes = cpu.iter().map(|(_, n)| n).sum();
    let signals = ctx.memory_signals();
    let decision =
        evaluate_metal_memory_admission_with_cpu_bytes(gpu_bytes, cpu_bytes, 0, signals, true);
    emit(
        out,
        json!({"event":"admission","phase":phase,"gpu_buffers":buffers,"cpu_components":cpu,
        "gpu_upper_bytes":gpu_bytes,"cpu_upper_bytes":cpu_bytes,"admitted":decision.admitted,
        "reason":decision.reason.as_str(),"metal_current_allocated_bytes":signals.current_allocated_bytes,
        "recommended_working_set_bytes":signals.recommended_max_bytes,"process_limit_remaining_bytes":signals.process_limit_remaining_bytes}),
    );
    assert!(
        decision.admitted,
        "normal admission refused; record flushed"
    );
}
fn f32_values(t: &MetalTensor) -> &[f32] {
    assert_eq!(t.dtype, GgmlType::F32);
    assert!(t.offset.is_multiple_of(4));
    assert!(
        t.offset
            .checked_add(t.n_bytes())
            .is_some_and(|end| end <= t.buffer.length() as u64)
    );
    // SAFETY: callers use host slices only before dispatch or after completion.
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
    assert_eq!(t.offset, 0);
    assert_eq!(t.dtype, GgmlType::F32);
    assert!(t.n_bytes() <= t.buffer.length() as u64);
    // SAFETY: owned writable buffer, exclusive initialization between commands.
    unsafe {
        let p = t.buffer.contents().as_ptr().cast::<f32>();
        for i in 0..t.n_elements() as usize {
            p.add(i).write(value(i));
        }
    }
}
fn input_value(i: usize) -> f32 {
    let mut x = (i as u32).wrapping_add(0x9e3779b9);
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    ((x & 65535) as f32 - 32768.0) / 32768.0
}

// One compressed destination, copied via bounded retained-descriptor reads.
fn load_native(ctx: &MetalContext, gguf: &GgufFile, d: &TensorDesc, out: &mut File) -> MetalTensor {
    let mut w = MetalTensor {
        buffer: ctx
            .buffer_uninit(usize::try_from(d.n_bytes).unwrap())
            .unwrap(),
        offset: 0,
        shape: d.shape.clone(),
        dtype: d.dtype,
        provenance: MetalTensorProvenance::OwnedWritable,
    };
    let mut bytes = vec![0u8; READ_BYTES];
    let mut hash = Sha256::new();
    let mut offset = 0u64;
    while offset < d.n_bytes {
        let count = (d.n_bytes - offset).min(bytes.len() as u64) as usize;
        gguf.read_shard_exact_at(d.shard_idx, d.data_offset + offset, &mut bytes[..count])
            .unwrap();
        hash.update(&bytes[..count]);
        // SAFETY: new buffer, bounded range, no GPU work references it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                w.buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(offset as usize),
                count,
            );
        }
        offset += count as u64;
    }
    w.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    emit(
        out,
        json!({"event":"weight_binding","tensor":descriptor(d),"payload_sha256":format!("{:x}",hash.finalize()),
        "buffer_bytes":w.buffer.length(),"full_f32_weight_bytes":0,"full_cpu_compressed_weight_bytes":0,
        "arms_share_weight_buffer":true,"provenance":format!("{:?}",w.provenance())}),
    );
    w
}
fn decoded_rows(gguf: &GgufFile, d: &TensorDesc, out: &mut File) -> Vec<(usize, Vec<f32>)> {
    let (k, m, row_bytes) = dimensions(d);
    let mut bytes = vec![0u8; row_bytes];
    anchors(m).into_iter().map(|row| {
        let offset=d.data_offset+(row*row_bytes) as u64;
        gguf.read_shard_exact_at(d.shard_idx,offset,&mut bytes).unwrap();
        let desc=TensorDesc {name:d.name.clone(),shape:vec![k as u64,1],dtype:d.dtype,
            shard_idx:0,data_offset:0,n_bytes:row_bytes as u64};
        let values=dequant_to_f32(&desc,&bytes).unwrap();
        assert_eq!(values.len(),k); assert!(values.iter().all(|v|v.is_finite()));
        emit(out,json!({"event":"oracle_row_binding","tensor":d.name,"row":row,"offset":offset,
            "source_sha256":sha(&bytes),"decoded_sha256_f32_le":sha(bytemuck::cast_slice(&values))}));
        (row,values)
    }).collect()
}
fn oracle(
    rows: &[(usize, Vec<f32>)],
    input: &[f32],
    k: usize,
    m: usize,
    n: usize,
) -> Vec<(usize, f64)> {
    assert_eq!(input.len(), k * n);
    anchors(n)
        .into_iter()
        .flat_map(|token| {
            rows.iter().map(move |(row, weights)| {
                assert_eq!(weights.len(), k);
                assert!(*row < m);
                let value = weights
                    .iter()
                    .zip(&input[token * k..(token + 1) * k])
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum::<f64>();
                assert!(value.is_finite());
                (token * m + *row, value)
            })
        })
        .collect()
}
#[derive(Default)]
struct Difference {
    max_abs: f64,
    error2: f64,
    reference2: f64,
}
impl Difference {
    fn add(&mut self, a: f64, b: f64) {
        let d = b - a;
        self.max_abs = self.max_abs.max(d.abs());
        self.error2 += d * d;
        self.reference2 += a * a;
    }
    fn json(&self) -> Value {
        json!({"max_abs":self.max_abs,"relative_l2":(self.error2/self.reference2.max(1e-30)).sqrt(),"reference_l2":self.reference2.sqrt()})
    }
}
fn compare(
    out: &mut File,
    mut event: Value,
    y: &MetalTensor,
    samples: &[(usize, f64)],
    reference: &mut [f32],
    compare_a1: bool,
    save_a1: bool,
) {
    let values = f32_values(y);
    assert_eq!(values.len(), reference.len());
    let mut difference = Difference::default();
    let mut cpu = Difference::default();
    let mut nonfinite = 0;
    let mut hash = Sha256::new();
    for (chunk_index, chunk) in values.chunks(READ_BYTES / 4).enumerate() {
        hash.update(bytemuck::cast_slice(chunk));
        for (j, &v) in chunk.iter().enumerate() {
            let i = chunk_index * (READ_BYTES / 4) + j;
            if v.is_finite() {
                if compare_a1 {
                    difference.add(f64::from(reference[i]), f64::from(v));
                }
            } else {
                nonfinite += 1;
            }
            if save_a1 {
                reference[i] = v;
            }
        }
    }
    for &(i, v) in samples {
        if values[i].is_finite() {
            cpu.add(v, f64::from(values[i]));
        }
    }
    let points: Vec<_> = samples
        .iter()
        .map(|&(i, v)| json!({"index":i,"reference":v,"actual":values[i]}))
        .collect();
    let metrics = json!({"elements":values.len(),"nonfinite":nonfinite,"output_sha256_f32_le":format!("{:x}",hash.finalize()),
        "reference_label":compare_a1.then_some("A1_scalar_native_same_round"),"difference":compare_a1.then(||difference.json()),
        "cpu_codec_f64_difference":cpu.json(),"cpu_codec_f64_points":points,"decision":"unscored_no_bit_or_error_threshold_gate"});
    event
        .as_object_mut()
        .unwrap()
        .extend(metrics.as_object().unwrap().clone());
    emit(out, event);
    assert_eq!(nonfinite, 0, "nonfinite output; comparison flushed");
}

#[derive(Clone, Copy)]
struct Timing {
    gpu_ms: f64,
    wall_ms: f64,
    substitutions: usize,
}
struct CensusReset(bool);
impl Drop for CensusReset {
    fn drop(&mut self) {
        if self.0 {
            let _ = dispatch_census_take();
        }
    }
}
fn attempt(
    ctx: &MetalContext,
    out: &mut File,
    mut event: Value,
    variant: Iq2XsMatMatVariant,
    n: usize,
    witness: bool,
    encode: impl FnOnce(&KernelEncoder) -> Result<(), MetalError>,
) -> Option<Timing> {
    assert!(
        !dispatch_census_is_active(),
        "no external census in timing path"
    );
    let _reset = CensusReset(witness);
    if witness {
        dispatch_census_begin();
    }
    let start = (!witness).then(Instant::now);
    let cmd = ctx.queue.commandBuffer().expect("ordinary command buffer");
    let enc = KernelEncoder::begin(&cmd);
    let (encoded, substitutions) = with_iq2_xs_matmat_variant(Some(variant), || encode(&enc));
    enc.end();
    let result = encoded.and_then(|()| {
        cmd.commit();
        wait_completed(&cmd)
    });
    let wall_ms = start.map(|start| start.elapsed().as_secs_f64() * 1e3);
    let (gpu_start, gpu_end) = if !witness && result.is_ok() {
        (cmd.GPUStartTime(), cmd.GPUEndTime())
    } else {
        (0.0, 0.0)
    };
    let gpu_ms = (gpu_end - gpu_start) * 1e3;
    let valid = !witness
        && result.is_ok()
        && gpu_start.is_finite()
        && gpu_start > 0.0
        && gpu_end.is_finite()
        && gpu_ms.is_finite()
        && gpu_ms > 0.0;
    let census = if witness {
        dispatch_census_take()
    } else {
        Vec::new()
    };
    let rows:Vec<_>=census.iter().map(|r|json!({"kernel":r.kernel,"tag":r.tag,"concurrent":r.encoder_concurrent,
        "grid":[r.grid_width,r.grid_height,r.grid_depth],"threads":[r.threads_width,r.threads_height,r.threads_depth]})).collect();
    let measurements = json!({"requested_variant":format!("{variant:?}"),"mma_substitutions":substitutions,
        "expected_mma_substitutions":if n==1||variant==Iq2XsMatMatVariant::Scalar {Some(0)} else if variant==Iq2XsMatMatVariant::Mma {Some(1)} else {None},
        "census_observer":witness,"measured":!witness,"dispatch_census":rows,"wall_ms":wall_ms,
        "gpu_start_s":(!witness).then_some(gpu_start),"gpu_end_s":(!witness).then_some(gpu_end),
        "command_gpu_ms":valid.then_some(gpu_ms),"command_gpu_valid":(!witness).then_some(valid),
        "error":result.as_ref().err().map(ToString::to_string)});
    event
        .as_object_mut()
        .unwrap()
        .extend(measurements.as_object().unwrap().clone());
    emit(out, event);
    result.expect("encode/command failed; attempt flushed");
    if n == 1 || variant == Iq2XsMatMatVariant::Scalar {
        assert_eq!(substitutions, 0);
    } else if variant == Iq2XsMatMatVariant::Mma {
        assert_eq!(substitutions, 1);
    } else {
        assert!(substitutions <= 1);
    }
    if witness {
        assert_eq!(census.len(), 1, "one native GEMM dispatch");
        let expected = if substitutions == 1 {
            "kernel_mat_mat_iq2_xs_f32_mma"
        } else {
            "kernel_mat_mat_iq2_xs_f32"
        };
        assert_eq!(census[0].kernel, expected);
        None
    } else {
        assert!(valid, "invalid GPU timestamps; attempt flushed");
        Some(Timing {
            gpu_ms,
            wall_ms: wall_ms.unwrap(),
            substitutions,
        })
    }
}

fn screen(
    ctx: &MetalContext,
    gguf: &GgufFile,
    d: &TensorDesc,
    out: &mut File,
    candidate: Iq2XsMatMatVariant,
    rounds: usize,
) {
    let (k, m, row_bytes) = dimensions(d);
    let transaction = ctx.begin_allocation_transaction();
    admit(
        ctx,
        out,
        &d.name,
        &[
            ("native_weight_shared_by_arms", d.n_bytes),
            ("input_N1024", (k * MAX_N * 4) as u64),
            ("output_N1024", (m * MAX_N * 4) as u64),
        ],
        &[
            ("one_A1_output_reference", (m * MAX_N * 4) as u64),
            ("compressed_read_chunk", READ_BYTES as u64),
            ("three_decoded_oracle_rows", (3 * k * 4) as u64),
            ("source_row_and_codec_staging", (2 * row_bytes) as u64),
            ("metadata", META_CPU),
        ],
    );
    let w = load_native(ctx, gguf, d, out);
    let rows = decoded_rows(gguf, d, out);
    let mut x = MetalTensor::zeros_f32(ctx, vec![k as u64, MAX_N as u64]).unwrap();
    initialize(&mut x, input_value);
    let y = MetalTensor::zeros_f32(ctx, vec![m as u64, MAX_N as u64]).unwrap();
    let mut reference = vec![0.0; m * MAX_N];
    drop(transaction);
    for n in WIDTHS {
        let xv = x.view_subrange(0, vec![k as u64, n as u64]);
        let mut yv = y.view_subrange(0, vec![m as u64, n as u64]);
        let samples = oracle(&rows, f32_values(&xv), k, m, n);
        emit(
            out,
            json!({"event":"shape_binding","tensor":d.name,"K":k,"M":m,"N":n,
            "input_sha256_f32_le":sha(bytemuck::cast_slice(f32_values(&xv))),"same_weight_input_output_buffers":true,
            "is_primary_width":matches!(n,32|128|512|1024),"N1_control":"both overrides use incumbent scalar GEMM",
            "metal_current_allocated_bytes":ctx.current_allocated_size()}),
        );
        for (label, variant) in [
            ("warm_A_witness", Iq2XsMatMatVariant::Scalar),
            ("warm_B_witness", candidate),
        ] {
            initialize(&mut yv, |_| f32::NAN);
            attempt(
                ctx,
                out,
                json!({"event":"witness_attempt","tensor":d.name,"N":n,"label":label}),
                variant,
                n,
                true,
                |enc| encode_mat_mat_iq2_xs_f32(ctx, enc, &w, &xv, &yv, k, m, n),
            );
            compare(
                out,
                json!({"event":"output_comparison","tensor":d.name,"N":n,"label":label,"round":0}),
                &yv,
                &samples,
                &mut reference[..m * n],
                false,
                false,
            );
        }
        for round in 1..=rounds {
            let mut times = Vec::with_capacity(4);
            for (label, b) in ABBA {
                initialize(&mut yv, |_| f32::NAN);
                let variant = if b {
                    candidate
                } else {
                    Iq2XsMatMatVariant::Scalar
                };
                eprintln!("IQ2_XS MMA {} N{n} round{round} {label}", d.name);
                times.push(attempt(ctx,out,json!({"event":"leaf_attempt","tensor":d.name,"N":n,"round":round,"label":label,
                    "arm":if b {"candidate_native"} else {"scalar_native"}}),variant,n,false,
                    |enc|encode_mat_mat_iq2_xs_f32(ctx,enc,&w,&xv,&yv,k,m,n)).unwrap());
                compare(
                    out,
                    json!({"event":"output_comparison","tensor":d.name,"N":n,"round":round,"label":label}),
                    &yv,
                    &samples,
                    &mut reference[..m * n],
                    label != "A1",
                    label == "A1",
                );
            }
            let [a1, b1, b2, a2]: [Timing; 4] = times.try_into().ok().unwrap();
            emit(
                out,
                json!({"event":"abba_summary","tensor":d.name,"N":n,"round":round,
                "paired_gpu_savings":[1.0-b1.gpu_ms/a1.gpu_ms,1.0-b2.gpu_ms/a2.gpu_ms],
                "gpu_mean_saving":1.0-(b1.gpu_ms+b2.gpu_ms)/(a1.gpu_ms+a2.gpu_ms),
                "wall_mean_saving":1.0-(b1.wall_ms+b2.wall_ms)/(a1.wall_ms+a2.wall_ms),
                "scalar_A2_over_A1_gpu":a2.gpu_ms/a1.gpu_ms,"candidate_mma_dispatches":b1.substitutions+b2.substitutions,
                "decision":"unscored_leaf_only_no_production_promotion"}),
            );
        }
    }
}

#[test]
fn iq2_xs_mma_cpu_bookkeeping() {
    assert_eq!(anchors(1), vec![0]);
    assert_eq!(anchors(7), vec![0, 3, 6]);
    let rows = vec![(1, vec![1.0, 2.0]), (3, vec![-3.0, 4.0])];
    assert_eq!(
        oracle(&rows, &[2.0, 1.0, 0.0, 3.0], 2, 5, 2),
        vec![(1, 4.0), (3, -2.0), (6, 6.0), (8, 12.0)]
    );
    let mut d = Difference::default();
    d.add(f64::from(f32::MAX), -f64::from(f32::MAX));
    assert_eq!(d.json()["relative_l2"], json!(2.0));
}

#[test]
#[ignore = "actual Saluki native IQ2_XS scalar/MMA; release; lease; new IQ2_XS_MMA_OUT"]
fn iq2_xs_mma_packet() {
    assert!(!cfg!(debug_assertions), "release diagnostic required");
    assert!(
        std::env::var_os("MTL_DEBUG_LAYER").is_none(),
        "timing requires no debug layer"
    );
    let mut out = File::options()
        .write(true)
        .create_new(true)
        .open(std::env::var_os("IQ2_XS_MMA_OUT").expect("IQ2_XS_MMA_OUT required"))
        .expect("new output file required");
    let rounds = std::env::var("IQ2_XS_MMA_ROUNDS")
        .map(|s| s.parse::<usize>().expect("positive rounds"))
        .unwrap_or(2);
    assert!(rounds > 0);
    let candidate = match std::env::var("IQ2_XS_MMA_CANDIDATE").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("mma") => Iq2XsMatMatVariant::Mma,
        Ok("auto") => Iq2XsMatMatVariant::Auto,
        _ => panic!("IQ2_XS_MMA_CANDIDATE must be mma or auto"),
    };
    let path = std::env::var_os("IQ2_XS_MMA_GGUF").unwrap_or_else(|| ARTIFACT.into());
    let gguf = GgufFile::open(path).unwrap();
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let shapes = [
        ("ffn_gate.weight", 5120, 17408),
        ("ssm_out.weight", 6144, 5120),
    ];
    let reps: Vec<_> = shapes
        .iter()
        .map(|(suffix, k, m)| {
            gguf.tensors
                .iter()
                .find(|d| {
                    d.dtype == GgmlType::IQ2_XS && d.name.ends_with(*suffix) && d.shape == [*k, *m]
                })
                .expect("actual Saluki IQ2_XS representative")
        })
        .collect();
    let _lease = acquire_metal_benchmark_lease().expect("production adaptive-wait lease");
    let ctx = MetalContext::new().unwrap();
    let shards:Vec<_>=stamps.iter().map(|s|json!({"path":s.path,"shard":s.shard_idx,"bytes":s.size,"device":s.device,"inode":s.inode,
        "mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,"ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec})).collect();
    let bindings: Vec<_> = reps.iter().map(|d| descriptor(d)).collect();
    let source_binding = json!({"packet":sha(include_bytes!("iq2_xs_mma.rs")),"selector":sha(include_bytes!("../../metal/iq2_xs.rs")),
        "mat_mat":sha(include_bytes!("../../metal/mat_mat.rs")),"scalar_kernel":sha(include_bytes!("../../../../../kernels/mat_vec.metal")),
        "mma_kernel":sha(include_bytes!("../../../../../kernels/iq2_xs_dense.metal")),"grid":sha(include_bytes!("../../../../../kernels/iq2_xs_grid.metalh")),
        "codec":sha(include_bytes!("../../codec.rs")),"gguf":sha(include_bytes!("../../gguf.rs")),"metallib":sha(crate::KERNELS_METALLIB)});
    emit(
        &mut out,
        json!({"event":"header","schema":"iq2_xs.native_mma.leaf.v1","device":ctx.describe(),"shards":shards,
        "representatives":bindings,"widths":WIDTHS,"rounds":rounds,"candidate":format!("{candidate:?}"),"source_binding":source_binding,
        "order":"untimed census warm_A/warm_B, then A1/B1/B2/A2 per round; timed paths census off",
        "inputs":"fixed synthetic F32 activations; real retained-descriptor compressed weights; identical buffers for both arms",
        "oracle":"canonical CPU codec first/middle/last rows, F64 dots at first/middle/last token; one A1 output reference for full comparisons",
        "scope":"leaf only; no full model, no F32 weight bank, no throughput or language-quality claim",
        "production":"None follows guarded Auto; this leaf uses explicit Scalar versus Mma/Auto test scopes; N1 is scalar under both arms",
        "decision":"unscored_no_numerical_or_performance_promotion_gate"}),
    );
    {
        let transaction = ctx.begin_allocation_transaction();
        admit(
            &ctx,
            &mut out,
            "header_binding",
            &[],
            &[("read_chunk", READ_BYTES as u64), ("metadata", META_CPU)],
        );
        let mut bytes = vec![0u8; READ_BYTES];
        drop(transaction);
        for (i, shard) in gguf.shards.iter().enumerate() {
            let mut offset = 0u64;
            let mut hash = Sha256::new();
            while offset < shard.tensor_data_start {
                let n = (shard.tensor_data_start - offset).min(bytes.len() as u64) as usize;
                gguf.read_shard_exact_at(i, offset, &mut bytes[..n])
                    .unwrap();
                hash.update(&bytes[..n]);
                offset += n as u64;
            }
            emit(
                &mut out,
                json!({"event":"header_binding","shard":i,"bytes":offset,"sha256":format!("{:x}",hash.finalize()),"full_artifact_payload_hash":false}),
            );
        }
    }
    for d in reps {
        screen(&ctx, &gguf, d, &mut out, candidate, rounds);
        emit(
            &mut out,
            json!({"event":"representative_dropped","tensor":d.name,"metal_current_allocated_bytes":ctx.current_allocated_size()}),
        );
    }
    assert_eq!(stamps, gguf.revalidate_retained_shard_stamps().unwrap());
    emit(
        &mut out,
        json!({"event":"complete","schema":"iq2_xs.native_mma.leaf.v1",
        "timed_attempts":2*WIDTHS.len()*rounds*4,"untimed_witness_attempts":2*WIDTHS.len()*2,
        "decision":"unscored_leaf_only_no_promotion"}),
    );
}
