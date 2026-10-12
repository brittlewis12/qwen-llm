//! Current production Fast Half, one N2048 chunk at prefix zero. Parent costs,
//! not leaf attribution or a qualification gate. GLM53_GGUF=Maya;
//! GLM53_STAGE_OUT=new.jsonl (relative paths are workspace-relative).
use super::*;
use crate::glm5_next_metal::tests::{long_qualification_text, perf_lease};
use crate::metal::{dispatch_census_begin, dispatch_census_take};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    time::Instant,
};

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn emit(out: &mut std::fs::File, value: Value) {
    serde_json::to_writer(&mut *out, &value).unwrap();
    writeln!(out).unwrap();
    out.flush().unwrap();
}

#[test]
#[ignore = "release, internal production perf lease; GLM53_GGUF=Maya and new GLM53_STAGE_OUT"]
fn current_half_2048_cost() {
    assert!(!cfg!(debug_assertions), "requires release test binary");
    let _lease = perf_lease();
    const ROWS: usize = 2048;
    let path = std::env::var_os("GLM53_GGUF").expect("GLM53_GGUF must name Maya");
    let out_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(std::env::var_os("GLM53_STAGE_OUT").expect("GLM53_STAGE_OUT"));
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
        json!({"event":"header","schema":"glm53.current_half_2048_cost.v1",
        "artifact":path,"binary":binary,"binary_sha256":format!("{hash:x}",hash=hash.finalize()),
        "rows":ROWS,"prefix":0,"precision":"Half","lineage":"Fast","production_overrides":false,
        "msl_language":include_str!("../../../build.rs").lines().find(|l|l.starts_with("const METAL_LANGUAGE:")).unwrap().split('"').nth(1).unwrap(),
        "product_deployment_target":crate::PRODUCT_METALLIB_TARGET,"research_deployment_target":crate::RESEARCH_METALLIB_TARGET,
        "source_sha256":{"packet":sha(include_bytes!("prefill_cost.rs")),"packed":sha(include_bytes!("../packed.rs")),
            "runtime":sha(include_bytes!("../../glm5_next_metal.rs")),"build":sha(include_bytes!("../../../build.rs")),
            "metallib":sha(crate::KERNELS_METALLIB)},
        "method":"warm ordinary with census, two ordinary plain calls, one stage-profiled call; fresh zero-prefix session each; ordinary call includes GPU wait and logits readback; session allocation, checks, hashes and JSON excluded; profiled API wall includes sample allocation/resolution",
        "limits":"existing packed boundaries combine bucket/gate-up/down/reduction in routed_experts, KDA projections/prepare/recurrence in kda, and MLA dense/indexer/latent/attention in mla_projection; dispatch counts are NOT leaf times; scaled timestamps do not supply independent raw coverage"}),
    );
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(std::path::PathBuf::from(path)).unwrap();
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    preflight_session(&ctx, &gguf, artifact.model(), ROWS, ROWS).unwrap();
    let text = format!("[gMASK]<sop>{}", long_qualification_text());
    let tokens: Vec<u32> = artifact
        .tokenizer()
        .encode(&text, false)
        .unwrap()
        .into_iter()
        .take(ROWS)
        .map(|t| u32::try_from(t).unwrap())
        .collect();
    assert_eq!(tokens.len(), ROWS);
    emit(
        &mut out,
        json!({"event":"input","text_sha256":sha(text.as_bytes()),"tokens":tokens,
        "shards":stamps.iter().map(|s|json!({"path":s.path,"bytes":s.size,"device":s.device,"inode":s.inode,
            "mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,"ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec})).collect::<Vec<_>>() }),
    );
    drop(text);
    let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    // try_map visits identical field order on descriptors and live Metal bindings.
    for (block, (desc, live)) in artifact
        .model()
        .blocks
        .iter()
        .zip(&weights.blocks)
        .enumerate()
    {
        let mut descriptors = Vec::new();
        desc.try_map(&mut |t| {
            descriptors.push(*t);
            Ok::<_, ()>(())
        })
        .unwrap();
        let mut index = 0;
        live.try_map(&mut |t| {
            let d = descriptors[index]; index += 1;
            assert_eq!(d.dtype, t.dtype);
            emit(&mut out, json!({"event":"weight_binding","block":block,"name":d.name,
                "dtype":format!("{:?}",t.dtype),"stored_shape":d.shape,"execution_shape":t.shape,"bytes":t.n_bytes()}));
            Ok::<_, ()>(())
        }).unwrap();
        assert_eq!(index, descriptors.len());
    }
    let cpu_payload = 2 * u64::from(weights.config.vocab_size) * 4 + tokens.capacity() as u64 * 4;
    let mut reference: Option<Vec<f32>> = None;
    let mut plain = Vec::new();
    for label in ["warm", "plain1", "plain2", "profiled"] {
        let warm = label == "warm";
        let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
            &ctx,
            &weights,
            ROWS,
            ROWS,
            cpu_payload,
        )
        .unwrap();
        assert_eq!(session.position(), 0);
        assert_eq!(session.packed_lineage(), Some(PackedLineage::Fast));
        assert_eq!(session.fast_precision(), Some(FastPrecision::Half));
        assert_eq!(session.ledger().prefill_rows(), ROWS as u64);
        emit(
            &mut out,
            json!({"event":"session","label":label,"rows":session.ledger().prefill_rows(),
            "ledger":session.ledger().terms(),"observed_allocation_delta":session.observed_allocation_delta(),
            "actual_precision":session.fast_precision().unwrap().name(),"actual_lineage":format!("{:?}",session.packed_lineage().unwrap()),
            "cpu_vector_payload_reserve_bytes":cpu_payload,"gpu_current_allocated_bytes":ctx.current_allocated_size()}),
        );
        assert!(!crate::metal::dispatch_census_is_active());
        if warm {
            dispatch_census_begin();
        }
        let start = Instant::now();
        let (logits, report) = if label == "profiled" {
            let (logits, report) = session.prefill_chunk_stage_profiled(&ctx, &tokens).unwrap();
            (logits, Some(report))
        } else {
            (session.prefill_packed(&ctx, &tokens).unwrap(), None)
        };
        let wall = start.elapsed().as_secs_f64() * 1e3;
        let census = if warm {
            dispatch_census_take()
        } else {
            Vec::new()
        };
        let finite = logits.len() == weights.config.vocab_size as usize
            && logits.iter().all(|x| x.is_finite());
        let equal = reference.as_ref().is_none_or(|a| {
            a.iter()
                .zip(&logits)
                .all(|(a, b)| a.to_bits() == b.to_bits())
                && a.len() == logits.len()
        });
        emit(
            &mut out,
            json!({"event":"run","label":label,"wall_ms":wall,"ordinary_timing_eligible":label.starts_with("plain"),
            "finite_full_vocab":finite,"bits_equal_warm":equal,"logits_sha256":sha(bytemuck::cast_slice(&logits)),"position":session.position()}),
        );
        assert!(finite && equal, "fresh/profiling execution changed logits");
        assert_eq!(session.position(), ROWS);
        if warm {
            emit(
                &mut out,
                json!({"event":"warm_census","timing_eligible":false,"dispatches":census.len(),
                "production_q6_latent_mma_dispatches":census.iter().filter(|r|r.kernel=="kernel_latent_grouped_q6_K_f32_r2c16k64" && r.grid_depth==u64::from(weights.config.head_count)).count(),
                "census_heap_bytes":census.capacity()*size_of::<crate::metal::DispatchCensusRow>()+census.iter().map(|r|r.kernel.capacity()+r.tag.as_ref().map_or(0,String::capacity)).sum::<usize>(),
                "rows":census.iter().map(|r|json!({"family":r.family,"tag":r.tag,"encoder":r.encoder_ordinal,"kernel":r.kernel,
                    "grid":[r.grid_width,r.grid_height,r.grid_depth],"threads":[r.threads_width,r.threads_height,r.threads_depth]})).collect::<Vec<_>>() }),
            );
            reference = Some(logits);
        }
        if label.starts_with("plain") {
            plain.push(wall);
        }
        if let Some(report) = report {
            let mut stages = BTreeMap::<&str, f64>::new();
            for s in &report.spans {
                *stages.entry(s.stage.as_str()).or_default() += s.gpu_ms;
            }
            let mean = (plain[0] + plain[1]) * 0.5;
            emit(
                &mut out,
                json!({"event":"profile","plain_wall_ms":plain,"profiled_wall_ms":wall,
                "profiled_over_plain_mean":wall/mean,"plain2_over_plain1":plain[1]/plain[0],
                "command_gpu_ms":report.command_gpu_ms,"scaled_span_sum_ms":report.span_sum_ms,
                "command_minus_scaled_spans_ms":report.command_gpu_ms-report.span_sum_ms,"stages_ms":stages,
                "spans":report.spans.iter().map(|s|json!({"block":s.block,"stage":s.stage.as_str(),"gpu_ms":s.gpu_ms})).collect::<Vec<_>>(),
                "warning":"profile splits encoders and scales timestamps; inspect overhead and plain drift before trusting attribution. No ordinary-wall fractions, raw-coverage claim, leaf extrapolation or numeric acceptance bound."}),
            );
            eprintln!(
                "N2048 Half plain {:?} ms, profiled {wall:.2} ms ({:.3}x mean); scaled stage costs are diagnostic, not ordinary-wall fractions",
                plain,
                wall / mean
            );
        }
    }
    assert_eq!(gguf.revalidate_retained_shard_stamps().unwrap(), stamps);
    emit(
        &mut out,
        json!({"event":"complete","execution_complete":true,"performance_verdict":null}),
    );
}
