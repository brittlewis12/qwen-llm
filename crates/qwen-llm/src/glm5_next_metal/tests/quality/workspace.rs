//! Bounded frozen quality-v1 workspace screen, not a quality/noninferiority gate.
//! Timings include Metal validation and are diagnostic only, not ABBA evidence.
use super::*;
use std::{fs::File, io::Write, os::unix::fs::OpenOptionsExt, time::Instant};

const FROZEN: &[u8] = include_bytes!("../../../../../../scripts/reference/glm53/quality-v1.json");
const ARMS: [(&str, PackedLineage, usize); 3] = [
    ("Exact512", PackedLineage::Exact, 512),
    ("Fast512", PackedLineage::Fast, 512),
    ("Fast2048", PackedLineage::Fast, 2048),
];

fn emit(out: &mut File, value: Value) {
    serde_json::to_writer(&mut *out, &value).unwrap();
    writeln!(out).unwrap();
    out.flush().unwrap();
}

#[test]
#[ignore = "bounded workspace quality screen: MTL_DEBUG_LAYER=1, GLM53_GGUF, new GLM53_WORKSPACE_REPORT, idle GPU"]
fn workspace_quality_screen() {
    let fixture: Value = serde_json::from_slice(FROZEN).unwrap();
    assert_eq!(fixture["name"], "glm53-quality-v1");
    assert_eq!(fixture["continuation"], CONTINUATION);
    // Same first-in-stratum document selection as latent_mma, with no tool tasks.
    let documents = [("prose", 2048), ("code", 2048), ("long", 4352)].map(|(stratum, n)| {
        let d = fixture["documents"]
            .as_array()
            .unwrap()
            .iter()
            .find(|d| d["stratum"] == stratum)
            .unwrap();
        assert!(d["prefix_lengths"].as_array().unwrap().contains(&json!(n)));
        let tokens = parse_ids(&d["tokens"]);
        assert!(tokens.len() >= n + CONTINUATION);
        (d, n, tokens)
    });
    let mut out = File::options()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(std::env::var_os("GLM53_WORKSPACE_REPORT").expect("new private JSONL path"))
        .unwrap();
    let _lease = production_lease(); // Requires MTL_DEBUG_LAYER=1 and the production memory gate.
    let path = PathBuf::from(std::env::var_os("GLM53_GGUF").expect("GLM53_GGUF"));
    let gguf = GgufFile::open(&path).unwrap();
    let ctx = MetalContext::new().unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    let vocab = weights.config.vocab_size as usize;
    assert!(vocab > 0);
    for (_, n, tokens) in &documents {
        assert!(
            tokens[..n + CONTINUATION]
                .iter()
                .all(|&t| (t as usize) < vocab)
        );
    }
    // Two retained 64-row f32 arms; two live f32 rows (clone/forward peak);
    // KL's two f64 log-softmax vectors; bounded metrics, row headers and JSON.
    // Fixture/token arrays are already resident before admission. Reserve the
    // full bound on every arm, conservatively including already-resident rows.
    let cpu_terms = [
        2 * CONTINUATION as u64 * vocab as u64 * 4,
        2 * vocab as u64 * 4,
        2 * vocab as u64 * 8,
        64 * 1024,
    ];
    let cpu_reserve: u64 = cpu_terms.iter().sum();
    emit(
        &mut out,
        json!({
            "event":"header", "schema":"glm53.workspace_quality.v1",
            "fixture_sha256":sha256_hex(FROZEN), "model":path,
            "artifact_layout":artifact_layout(&gguf), "config":format!("{:?}", weights.config),
            "arms":ARMS.map(|(name, _, _)| name), "continuation":CONTINUATION,
            "documents":documents.iter().map(|(d,n,_)| json!({"path":d["path"],"stratum":d["stratum"],"prefix":n})).collect::<Vec<_>>(),
            "cpu_reserve_bytes":cpu_reserve, "cpu_terms":cpu_terms,
            "cpu_names":["exact_and_fast512_logits","two_live_logits","kl_f64_vectors","metrics_and_serialization"],
            "policy":"descriptive only; no quality threshold or noninferiority pass claim; finite/state/alignment assertions only; timings diagnostic, not ABBA; incomplete unless final complete event",
            "token_source":"frozen repository token IDs only; no retokenization or tokenizer compatibility claim",
            "mtl_debug_layer":"1"
        }),
    );
    for (document, n, tokens) in documents {
        let mut exact = Vec::<Vec<f32>>::with_capacity(CONTINUATION);
        let mut baseline = Vec::<Vec<f32>>::with_capacity(CONTINUATION);
        for (arm_index, (arm, lineage, rows)) in ARMS.into_iter().enumerate() {
            assert_eq!(exact.len(), if arm_index == 0 { 0 } else { CONTINUATION });
            assert_eq!(
                baseline.len(),
                if arm_index == 2 { CONTINUATION } else { 0 }
            );
            let started = Instant::now();
            let mut s = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
                &ctx,
                &weights,
                n + CONTINUATION,
                rows,
                cpu_reserve,
            )
            .unwrap();
            s.set_packed_lineage(lineage).unwrap();
            s.set_fast_precision(FastPrecision::Half).unwrap();
            let session_ms = started.elapsed().as_secs_f64() * 1e3;
            let actual_rows = s.ledger().prefill_rows();
            assert_eq!(actual_rows, rows as u64);
            assert_eq!(s.position(), 0);
            let mut chunks = Vec::with_capacity(n.div_ceil(rows));
            let started = Instant::now();
            let mut logits = s
                .prefill_packed_with_observer(
                    &ctx,
                    &tokens[..n],
                    &mut || Ok(()),
                    &mut |s, count| {
                        assert!(!s.is_poisoned());
                        chunks.push(count);
                        assert_eq!(s.position(), chunks.iter().sum::<usize>());
                    },
                )
                .unwrap();
            let prefill_ms = started.elapsed().as_secs_f64() * 1e3;
            assert_eq!(
                chunks,
                (0..n)
                    .step_by(rows)
                    .map(|i| (n - i).min(rows))
                    .collect::<Vec<_>>()
            );
            let mut nll = Vec::with_capacity(CONTINUATION);
            let mut hits = Vec::with_capacity(CONTINUATION);
            let mut exact_kl = Vec::with_capacity(CONTINUATION);
            let mut width_kl = Vec::with_capacity(CONTINUATION);
            let mut decode_ms = 0.0;
            for (i, &token) in tokens[n..n + CONTINUATION].iter().enumerate() {
                assert!(!s.is_poisoned());
                assert_eq!(s.position(), n + i, "held-out target alignment");
                assert_eq!(s.packed_lineage(), Some(lineage));
                assert_eq!(s.fast_precision(), Some(FastPrecision::Half));
                assert_eq!(s.ledger().prefill_rows(), actual_rows);
                assert_eq!(logits.len(), vocab);
                assert_finite("workspace logits", &logits);
                let loss = -log_prob(&logits, token);
                assert!(loss.is_finite());
                nll.push(loss);
                hits.push(argmax(&logits) == token as usize);
                if arm_index > 0 {
                    exact_kl.push(kl_divergence(&exact[i], &logits));
                }
                if arm_index == 2 {
                    width_kl.push(kl_divergence(&baseline[i], &logits));
                }
                match arm_index {
                    0 => exact.push(logits.clone()),
                    1 => baseline.push(logits.clone()),
                    _ => {}
                }
                if i + 1 < CONTINUATION {
                    let started = Instant::now();
                    logits = s.forward(&ctx, token).unwrap();
                    decode_ms += started.elapsed().as_secs_f64() * 1e3;
                }
            }
            assert_eq!(nll.len(), CONTINUATION);
            let mean_nll = nll.iter().sum::<f64>() / CONTINUATION as f64;
            assert!(mean_nll.is_finite());
            emit(
                &mut out,
                json!({
                    "event":"document_arm", "arm":arm, "path":document["path"],
                    "stratum":document["stratum"], "prefix":n, "capacity":n+CONTINUATION,
                    "target_positions":(n..n+CONTINUATION).collect::<Vec<_>>(),
                    "target_tokens":&tokens[n..n+CONTINUATION], "final_position":s.position(),
                    "logits_finite":true, "logit_rows":CONTINUATION, "vocab":vocab,
                    "nll":nll, "mean_nll":mean_nll, "top1_hits":hits.iter().filter(|&&h| h).count(),
                    "top1_hit_per_token":hits, "kl_exact_to_arm":exact_kl,
                    "kl_fast512_to_fast2048":width_kl, "kl_scope":"per-token descriptive KL(reference || arm)",
                    "actual_prefill_rows":actual_rows, "prefill_chunks":chunks,
                    "lineage":format!("{:?}", s.packed_lineage().unwrap()),
                    "selected_fast_precision":format!("{:?}", s.fast_precision().unwrap()),
                    "effective_precision":if lineage == PackedLineage::Fast { "Half" } else { "Exact (Fast precision unused)" },
                    "session_ms":session_ms, "prefill_ms":prefill_ms, "decode_ms":decode_ms,
                    "decode_calls":CONTINUATION-1, "timing_scope":"diagnostic only, Metal validation on; no warmup or ABBA"
                }),
            );
        }
    }
    emit(
        &mut out,
        json!({"event":"complete", "documents":3, "arms":3,
        "quality_verdict":"none; descriptive screen only"}),
    );
    out.sync_all().unwrap();
}
