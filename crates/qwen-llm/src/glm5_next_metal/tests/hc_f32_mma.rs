//! Bounded normalized F32 HC experiment. Separate leaf and optional whole N2048
//! packets; GLM53_GGUF=Maya, GLM53_STAGE_OUT=new.jsonl. No production promotion.
use super::*;
use crate::glm5_next_metal::tests::{kl_divergence, long_qualification_text, perf_lease};
use crate::metal::{DispatchCensusRow, dispatch_census_begin, dispatch_census_take};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    time::Instant,
};

const ROWS: usize = 2048;
const SAMPLE_TOKENS: [usize; 10] = [0, 31, 32, 63, 64, 127, 128, 1023, 1024, ROWS - 1];
const ARMS: [(&str, bool); 6] = [
    ("warm_A", false),
    ("warm_B", true),
    ("A1", false),
    ("B1", true),
    ("B2", true),
    ("A2", false),
];
thread_local! { static HC_F32_MMA: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

// Calling-thread scope only: never touches the global Q6/scalar diagnostics.
fn with_hc_f32_mma_override<R>(candidate: bool, body: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            HC_F32_MMA.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(HC_F32_MMA.with(|s| s.replace(candidate)));
    body()
}
pub(super) fn enabled(arith: Arith, dtype: GgmlType) -> bool {
    arith.lineage == PackedLineage::Fast
        && arith.precision == FastPrecision::Half
        && dtype == GgmlType::F32
        && HC_F32_MMA.with(|s| s.get())
}
#[test]
fn permission_is_half_only_nested_and_unwind_safe() {
    let half = Arith {
        lineage: PackedLineage::Fast,
        precision: FastPrecision::Half,
    };
    assert!(!enabled(half, GgmlType::F32));
    with_hc_f32_mma_override(true, || {
        for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
            for precision in [
                FastPrecision::Half,
                FastPrecision::DenseF32,
                FastPrecision::F32,
            ] {
                assert_eq!(
                    enabled(Arith { lineage, precision }, GgmlType::F32),
                    lineage == PackedLineage::Fast && precision == FastPrecision::Half
                );
            }
        }
        assert!(!enabled(half, GgmlType::Q8_0));
        assert!(!enabled(half, GgmlType::Q6_K));
        with_hc_f32_mma_override(false, || assert!(!enabled(half, GgmlType::F32)));
        assert!(
            std::panic::catch_unwind(|| with_hc_f32_mma_override(false, || panic!("restore")))
                .is_err()
        );
        assert!(enabled(half, GgmlType::F32));
    });
    assert!(!enabled(half, GgmlType::F32));
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn emit(out: &mut std::fs::File, value: Value) {
    serde_json::to_writer(&mut *out, &value).unwrap();
    writeln!(out).unwrap();
    out.flush().unwrap();
}
fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
fn census(out: &mut std::fs::File, label: &str, rows: &[DispatchCensusRow]) {
    emit(
        out,
        json!({"event":"untimed_census","label":label,
        "census_payload_bytes":rows.len()*size_of::<DispatchCensusRow>() + rows.iter().map(|r|r.kernel.capacity()+r.tag.as_ref().map_or(0,String::capacity)).sum::<usize>(),
        "dispatches":rows.iter().map(|r| json!({"kernel":r.kernel,"tag":r.tag,"encoder":r.encoder_ordinal,
            "grid":[r.grid_width,r.grid_height,r.grid_depth],"threads":[r.threads_width,r.threads_height,r.threads_depth]})).collect::<Vec<_>>() }),
    );
}
fn unchanged_signature(rows: &[DispatchCensusRow]) -> Vec<(String, [u64; 3], [u64; 3])> {
    rows.iter()
        .filter(|r| !r.kernel.starts_with("kernel_latent_grouped_f32"))
        .map(|r| {
            (
                r.kernel.clone(),
                [r.grid_width, r.grid_height, r.grid_depth],
                [r.threads_width, r.threads_height, r.threads_depth],
            )
        })
        .collect()
}

// Existing grouped_native_tests::grouped_f32_mma_fragments_match_all_channel_f64_oracle
// tolerance: 2e-5 * max(abs(f64 dot), activation scale). For actual normalized
// inputs the activation scale is the maximum absolute entry of the sampled row.
// Check each arm independently; the captured scalar output only verifies that
// the surviving scratch belongs to these weights, never defines MMA accuracy.
fn dot_accuracy(weight: &[f32], input: &[f32], gpu: f32) -> Value {
    assert_eq!(weight.len(), input.len());
    assert!(weight.iter().chain(input).all(|v| v.is_finite()));
    let mut reference = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sequential = 0.0f32;
    for (&w, &x) in weight.iter().zip(input) {
        let product = f64::from(w) * f64::from(x);
        reference += product;
        sum_abs += product.abs();
        sequential += w * x;
    }
    let scale = input.iter().map(|v| f64::from(v.abs())).fold(0.0, f64::max);
    let tolerance = 2e-5 * reference.abs().max(scale);
    let error = (f64::from(gpu) - reference).abs();
    json!({"f64_dot":reference,"sum_abs_products_f64":sum_abs,"cpu_sequential_f32":sequential,
        "gpu":gpu,"activation_scale":scale,"abs_error":error,"tolerance":tolerance,
        "within_bound":gpu.is_finite() && reference.is_finite() && error <= tolerance})
}
#[test]
fn independent_dot_tolerance_rejects_wrong_channel() {
    assert_eq!(
        dot_accuracy(&[1.0, -2.0], &[3.0, 4.0], -5.0)["within_bound"],
        true
    );
    assert_eq!(
        dot_accuracy(&[1.0, -2.0], &[3.0, 4.0], -4.0)["within_bound"],
        false
    );
}
const F32_MMA: &str = "kernel_latent_grouped_f32_f32_r2c16k64";
const F32_SCALAR: &str = "kernel_latent_grouped_f32_f32";
const Q6_MMA: &str = "kernel_latent_grouped_q6_K_f32_r2c16k64";
fn check_hc_census(rows: &[DispatchCensusRow], candidate: bool, calls: usize) {
    assert_eq!(
        rows.iter().filter(|r| r.kernel == F32_MMA).count(),
        if candidate { calls } else { 0 }
    );
    assert_eq!(
        rows.iter().filter(|r| r.kernel == F32_SCALAR).count(),
        if candidate { 0 } else { calls }
    );
    if candidate {
        for r in rows.iter().filter(|r| r.kernel == F32_MMA) {
            assert_eq!([r.grid_width, r.grid_height, r.grid_depth], [16, 2, 1]);
        }
    }
}

fn leaf(ctx: &MetalContext, session: &Glm5NextSession<'_>, out: &mut std::fs::File) {
    let p = session.packed.as_ref().unwrap();
    let block = session.weights.blocks.len() - 1;
    let mix = &session.weights.blocks[block].ffn_hc.mix;
    let k = 4 * session.weights.config.hidden_size as usize;
    assert_eq!(mix.dtype, GgmlType::F32);
    assert_eq!(mix.shape, [k as u64, 24]);
    // Only HC writes hc_normalized/mixes; the final FFN HC is their last writer.
    // FFN post consumes controls, and the last-row head uses decode scratch.
    let input = rows_view(p.hc_normalized.as_ref().unwrap(), ROWS);
    let output = rows_view(&p.mixes, ROWS);
    let captured = read_f32(&output).unwrap();
    let weights = read_f32(mix).unwrap();
    let samples: Vec<_> = SAMPLE_TOKENS
        .iter()
        .map(|&t| read_f32(&input.view_subrange((t * k) as u64, vec![k as u64])).unwrap())
        .collect();
    emit(
        out,
        json!({"event":"actual_binding","block":block,"component":"ffn_hc","dtype":"F32",
        "K":k,"M":24,"N":ROWS,"groups":1,"normalized_before_projection":true,
        "normalized_scratch_bytes_reused":input.n_bytes(),"output_scratch_bytes_reused":output.n_bytes(),
        "extra_gpu_buffers_bytes":0,"candidate_threadgroups":32,"weights_sha256":sha(bytemuck::cast_slice(&weights)),
        "sample_tokens":SAMPLE_TOKENS,"sample_input_hashes":samples.iter().map(|v| sha(bytemuck::cast_slice(v))).collect::<Vec<_>>(),
        "cpu_capture_bytes":4*(captured.capacity()+weights.capacity()+samples.iter().map(Vec::capacity).sum::<usize>()),
        "captured_mixes_sha256":sha(bytemuck::cast_slice(&captured))}),
    );
    let mut timings = Vec::new();
    for (label, candidate) in ARMS {
        let warm = label.starts_with("warm");
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        if warm {
            dispatch_census_begin();
        }
        let start = Instant::now();
        crate::metal::encode_mat_mat_grouped_f32_with_f32_mma_policy(
            ctx, &enc, mix, &input, &output, k, 24, 1, ROWS, true, candidate,
        )
        .unwrap();
        enc.end();
        command.commit();
        wait_completed(&command).unwrap();
        let wall_ms = start.elapsed().as_secs_f64() * 1e3;
        let gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
        assert!(gpu_ms.is_finite() && gpu_ms > 0.0, "invalid GPU duration");
        if warm {
            let rows = dispatch_census_take();
            census(out, label, &rows);
            check_hc_census(&rows, candidate, 1);
        }
        let values = read_f32(&output).unwrap();
        assert!(values.iter().all(|x| x.is_finite()));
        let mut accuracy = Vec::new();
        for (&token, x) in SAMPLE_TOKENS.iter().zip(&samples) {
            for channel in 0..24 {
                let mut a = dot_accuracy(
                    &weights[channel * k..(channel + 1) * k],
                    x,
                    values[token * 24 + channel],
                );
                a["token"] = json!(token);
                a["channel"] = json!(channel);
                accuracy.push(a);
            }
        }
        let passed = accuracy.iter().all(|a| a["within_bound"] == true);
        emit(
            out,
            json!({"event":"leaf_arm","label":label,"candidate":candidate,"measured":!warm,
            "wall_ms":wall_ms,"gpu_ms":gpu_ms,"finite_full_output":true,"output_sha256":sha(bytemuck::cast_slice(&values)),
            "bits_equal_captured":bits_equal(&captured,&values),"dot_samples":accuracy,"f64_contract_passed":passed,"cpu_output_readback_bytes":values.capacity()*4}),
        );
        assert!(passed, "independent dot contract failed; samples flushed");
        if !candidate {
            assert!(
                bits_equal(&captured, &values),
                "surviving scratch does not match final FFN HC projection"
            );
        }
        if !warm {
            timings.push((candidate, wall_ms, gpu_ms));
        }
    }
    let mean = |b, gpu| {
        timings
            .iter()
            .filter(|t| t.0 == b)
            .map(|t| if gpu { t.2 } else { t.1 })
            .sum::<f64>()
            / 2.0
    };
    emit(
        out,
        json!({"event":"leaf_summary","scalar_over_mma_wall":mean(false,false)/mean(true,false),
        "scalar_over_mma_gpu":mean(false,true)/mean(true,true),
        "interpretation":"entire HC pre budget previously 263 ms; this is only one M24/N2048 projection, not HC end-to-end. Stop on leaf loss; owner decides whether to run separate whole test."}),
    );
}

#[test]
#[ignore = "release; internal perf lease; GLM53_GGUF=Maya, new GLM53_STAGE_OUT"]
fn normalized_hc_f32_mma_leaf_2048() {
    packet(false);
}
#[test]
#[ignore = "optional ONLY after promising leaf; release; GLM53_GGUF=Maya, new GLM53_STAGE_OUT"]
fn normalized_hc_f32_mma_whole_2048_abba() {
    packet(true);
}

fn packet(whole: bool) {
    assert!(!cfg!(debug_assertions));
    let _lease = perf_lease();
    let path = std::path::PathBuf::from(std::env::var_os("GLM53_GGUF").expect("Maya GLM53_GGUF"));
    let out_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(std::env::var_os("GLM53_STAGE_OUT").expect("new GLM53_STAGE_OUT"));
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out_path)
        .unwrap();
    let binary = std::env::current_exe().unwrap();
    let mut file = std::fs::File::open(&binary).unwrap();
    let mut hash = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    emit(
        &mut out,
        json!({"event":"header","schema":"glm53.normalized_hc_f32_mma.v1","whole":whole,
        "artifact":path,"binary":binary,"binary_sha256":format!("{:x}",hash.finalize()),
        "source_sha256":{"packet":sha(include_bytes!("hc_f32_mma.rs")),"mhc":sha(include_bytes!("../../metal/mhc.rs")),
            "latent":sha(include_bytes!("../../metal/latent.rs")),"packed":sha(include_bytes!("../packed.rs")),"metallib":sha(crate::KERNELS_METALLIB)},
        "msl_language":include_str!("../../../build.rs").lines().find(|l|l.starts_with("const METAL_LANGUAGE:")).unwrap().split('"').nth(1).unwrap(),
        "product_target":crate::PRODUCT_METALLIB_TARGET,"research_target":crate::RESEARCH_METALLIB_TARGET,
        "precision":"Half","lineage":"Fast","rows":ROWS,"workspace_rows":ROWS,"prefix":0,
        "production_default_f32_hc_mma":false,"global_q6_override":false,
        "dot_tolerance":"existing native grouped all-channel F64 oracle: 2e-5 * max(abs(dot), sampled input maxabs)",
        "method":"warm A/B then A1 B1 B2 A2; census only warm; allocation/load/readback checks excluded from leaf timing; whole wall includes ordinary logits readback; no profiling or numeric whole promotion gate"}),
    );
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    let capacity = ROWS + if whole { 3 } else { 0 };
    preflight_session(&ctx, &gguf, artifact.model(), capacity, ROWS).unwrap();
    let text = format!("[gMASK]<sop>{}", long_qualification_text());
    let tokens: Vec<u32> = artifact
        .tokenizer()
        .encode(&text, false)
        .unwrap()
        .into_iter()
        .take(capacity)
        .map(|t| u32::try_from(t).unwrap())
        .collect();
    assert_eq!(tokens.len(), capacity);
    drop(text);
    emit(
        &mut out,
        json!({"event":"input","tokens":tokens,"shards":stamps.iter().map(|s|json!({"path":s.path,"bytes":s.size,"device":s.device,"inode":s.inode,"mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec})).collect::<Vec<_>>() }),
    );
    let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    assert!(
        weights
            .blocks
            .iter()
            .all(|b| b.attention_hc.mix.dtype == GgmlType::F32
                && b.ffn_hc.mix.dtype == GgmlType::F32)
    );
    let k = 4 * weights.config.hidden_size as usize;
    // Whole: four reference logits + one current vector + two f64 KL vectors.
    // Leaf: sampled normalized rows, F32 weights, captured/current projection;
    // capture-prefill logits are dropped before those readbacks are allocated.
    // Small census/JSON metadata is reported separately from these vector reserves.
    let cpu_bytes = tokens.capacity() * 4
        + if whole {
            weights.config.vocab_size as usize * 36
        } else {
            4 * (SAMPLE_TOKENS.len() * k + 24 * k + 2 * 24 * ROWS)
                .max(weights.config.vocab_size as usize)
                + SAMPLE_TOKENS.len() * size_of::<Vec<f32>>()
        };
    let mut reference: Vec<Vec<f32>> = Vec::new();
    let mut baseline_census = None;
    let mut times = Vec::new();
    for (label, candidate) in if whole {
        ARMS.to_vec()
    } else {
        vec![("capture", false)]
    } {
        let warm = label.starts_with("warm") || !whole;
        let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
            &ctx,
            &weights,
            capacity,
            ROWS,
            cpu_bytes as u64,
        )
        .unwrap();
        assert_eq!(session.packed_lineage(), Some(PackedLineage::Fast));
        assert_eq!(session.fast_precision(), Some(FastPrecision::Half));
        emit(
            &mut out,
            json!({"event":"session","label":label,"ledger":session.ledger().terms(),"cpu_payload_reserve_bytes":cpu_bytes,"gpu_current_allocated_bytes":ctx.current_allocated_size()}),
        );
        assert!(!crate::metal::dispatch_census_is_active());
        if warm {
            dispatch_census_begin();
        }
        let start = Instant::now();
        let mut logits = with_hc_f32_mma_override(candidate, || {
            session.prefill_packed(&ctx, &tokens[..ROWS]).unwrap()
        });
        let wall_ms = start.elapsed().as_secs_f64() * 1e3;
        if warm {
            let rows = dispatch_census_take();
            census(&mut out, label, &rows);
            assert!(
                rows.iter().any(|r| r.kernel == Q6_MMA),
                "shipped Q6 MMA must remain active"
            );
            check_hc_census(&rows, candidate, 2 * weights.blocks.len());
            let signature = unchanged_signature(&rows);
            if let Some(a) = &baseline_census {
                // Routing can respond to changed HC arithmetic; record other
                // dispatch differences descriptively, while requiring Q6 policy
                // and geometry to stay identical in both arms.
                let q6 = |v: &Vec<(String, [u64; 3], [u64; 3])>| {
                    v.iter()
                        .filter(|r| r.0 == Q6_MMA)
                        .cloned()
                        .collect::<Vec<_>>()
                };
                assert_eq!(q6(a), q6(&signature));
                emit(
                    &mut out,
                    json!({"event":"surrounding_dispatches","label":label,
                    "non_hc_signature_equal":a==&signature,"q6_mma_equal":true}),
                );
            } else {
                baseline_census = Some(signature);
            }
        }
        emit(
            &mut out,
            json!({"event":"prefill","label":label,"candidate":candidate,"wall_ms":wall_ms,"measured":!warm}),
        );
        assert_eq!(logits.len(), weights.config.vocab_size as usize);
        assert!(logits.iter().all(|x| x.is_finite()));
        if !whole {
            drop(logits);
            leaf(&ctx, &session, &mut out);
            break;
        }
        if !warm {
            times.push((candidate, wall_ms));
        }
        for step in 0..=3 {
            assert_eq!(logits.len(), weights.config.vocab_size as usize);
            assert!(logits.iter().all(|x| x.is_finite()));
            let comparison = if !warm && label != "A1" {
                Some(
                    json!({"kl_a1_to_run":kl_divergence(&reference[step],&logits),"bits_equal_a1":bits_equal(&reference[step],&logits)}),
                )
            } else {
                None
            };
            emit(
                &mut out,
                json!({"event":"logits","label":label,"step":step,"finite":true,"sha256":sha(bytemuck::cast_slice(&logits)),"comparison":comparison}),
            );
            if label == "A1" {
                reference.push(std::mem::take(&mut logits));
            }
            if step < 3 {
                logits = session.forward(&ctx, tokens[ROWS + step]).unwrap();
            }
        }
    }
    if whole {
        let mean = |candidate| {
            times
                .iter()
                .filter(|t| t.0 == candidate)
                .map(|t| t.1)
                .sum::<f64>()
                / 2.0
        };
        emit(
            &mut out,
            json!({"event":"whole_summary","scalar_over_mma_wall":mean(false)/mean(true),"decision":"descriptive; stop on loss or inconclusive whole result, no sweep"}),
        );
    }
    assert_eq!(gguf.revalidate_retained_shard_stamps().unwrap(), stamps);
    emit(&mut out, json!({"event":"complete","whole":whole}));
}
