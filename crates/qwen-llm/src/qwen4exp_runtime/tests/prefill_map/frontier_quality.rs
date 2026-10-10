//! Frozen technical-text NLL + independent retrieval, not a performance packet.
//! FLASH_FRONTIER_QUALITY_ARTIFACT=ud|gsq, FLASH_PREFILL_MODEL, NEW FLASH_PREFILL_OUT.
//! One artifact per process; normal lease and capacity4160 native admission.
use super::*;
use serde::Deserialize;

#[path = "frontier_retrieval_v2.rs"]
mod frontier_retrieval_v2;

const FIXTURE: &[u8] =
    include_bytes!("../../../../../../docs/bench/2026-10-09-flash-frontier-quality/fixtures.json");
const FIXTURE_SHA: &str = "d592ccda9273cdc03b26bca284e8bb48e4aa8db404a3b140c2a51b37f009e73e";
const QUALITY_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/bench/2026-10-09-flash-frontier-quality"
);
const N: usize = 4096;
const TARGETS: usize = 64;
const VOCAB: usize = 248320;
const STOP: u32 = 248046;
const LIMIT: f64 = 0.009950330853168092;
const NAMES: [&str; 8] = [
    "bash",
    "csh",
    "curl",
    "find",
    "launchctl",
    "security",
    "tar",
    "tcpdump",
];

#[derive(Deserialize)]
struct Binding {
    file: String,
    sha256: String,
    bytes: usize,
}
#[derive(Deserialize)]
struct Natural {
    id: String,
    text_file: String,
    rendered_sha256: String,
    complete_source_tokens: usize,
    token_ids: Vec<u32>,
    tokens_sha256_u32le: String,
    prompt_range: [usize; 2],
    target_range: [usize; 2],
}
#[derive(Debug, Deserialize, serde::Serialize, PartialEq)]
struct Answer {
    code: String,
    record: String,
}
#[derive(Deserialize)]
struct Retrieval {
    id: String,
    text_file: String,
    rendered_sha256: String,
    token_ids: Vec<u32>,
    tokens_sha256_u32le: String,
    expected: Answer,
    fact_text: String,
    distractor_code: String,
    fact_token_range: [usize; 2],
    maximum_generated_tokens: usize,
}
#[derive(Deserialize)]
struct Fixtures {
    schema: String,
    freeze_revision: usize,
    tokenizer_identity_sha256: String,
    vocab_size: usize,
    stop_token: u32,
    prefix: usize,
    continuation: usize,
    natural: Vec<Natural>,
    retrieval: Vec<Retrieval>,
    files: Vec<Binding>,
    templates: BTreeMap<String, Binding>,
    metadata_models: Value,
}
fn frozen() -> PacketResult<Fixtures> {
    require(
        sha256_bytes(FIXTURE) == FIXTURE_SHA,
        "compiled fixture freeze hash changed",
    )?;
    let f: Fixtures = serde_json::from_slice(FIXTURE)?;
    require(
        f.schema == "flash.frontier_quality.fixtures.v1"
            && f.freeze_revision == 2
            && f.vocab_size == VOCAB
            && f.stop_token == STOP
            && f.prefix == N
            && f.continuation == TARGETS,
        "frozen geometry changed",
    )?;
    require(
        f.natural.len() == 8 && f.retrieval.len() == 2,
        "frozen cohort incomplete",
    )?;
    for (i, d) in f.natural.iter().enumerate() {
        require(
            d.id == NAMES[i]
                && d.token_ids.len() == N + TARGETS
                && d.complete_source_tokens >= N + TARGETS
                && d.prompt_range == [0, N]
                && d.target_range == [N, N + TARGETS],
            "natural fixture boundaries changed",
        )?;
        require(
            d.token_ids.iter().all(|&id| (id as usize) < VOCAB)
                && sha256_u32_le(b"", &d.token_ids) == d.tokens_sha256_u32le,
            "natural token hash/range invalid",
        )?;
    }
    for (i, d) in f.retrieval.iter().enumerate() {
        let [start, end] = d.fact_token_range;
        require(
            d.id == ["retrieval_before", "retrieval_after"][i]
                && d.token_ids.len() == N
                && d.maximum_generated_tokens == TARGETS
                && start < end
                && end < N
                && if i == 0 { end <= 2051 } else { start >= 2051 },
            "retrieval boundary changed",
        )?;
        require(
            d.token_ids.iter().all(|&id| (id as usize) < VOCAB)
                && sha256_u32_le(b"", &d.token_ids) == d.tokens_sha256_u32le,
            "retrieval token hash/range invalid",
        )?;
        let rendered = text(&d.text_file, &d.rendered_sha256)?;
        require(
            !d.expected.code.is_empty()
                && !d.distractor_code.contains(d.expected.code.as_str())
                && rendered.matches(d.fact_text.as_str()).count() == 1
                && d.fact_text.matches(d.expected.code.as_str()).count() == 1
                && rendered.matches(d.expected.code.as_str()).count() == 1
                && !rendered
                    .replacen(d.fact_text.as_str(), "", 1)
                    .contains(d.expected.code.as_str())
                && rendered.contains(&format!(
                    "RATIFIED RECORD [Z-99]\nLaunch code: {}.",
                    d.distractor_code
                )),
            "correct code must occur only in the authoritative fact; Z-99 must be distinct",
        )?;
    }
    for b in &f.files {
        require(
            std::path::Path::new(&b.file).components().count() == 1,
            "fixture filename must be local",
        )?;
        let bytes = std::fs::read(std::path::Path::new(QUALITY_DIR).join(&b.file))?;
        require(
            bytes.len() == b.bytes && sha256_bytes(&bytes) == b.sha256,
            "frozen file hash changed",
        )?;
    }
    Ok(f)
}
fn text(name: &str, hash: &str) -> PacketResult<String> {
    let bytes = std::fs::read(std::path::Path::new(QUALITY_DIR).join(name))?;
    require(sha256_bytes(&bytes) == hash, "rendered text hash changed")?;
    Ok(String::from_utf8(bytes)?)
}
fn validate_tokenizer(
    gguf: &GgufFile,
    f: &Fixtures,
    artifact: &str,
    out: &mut std::fs::File,
) -> PacketResult<Tokenizer> {
    let identity = crate::tokenizer::qwen4exp_tokenizer_identity_sha256(gguf)?;
    let hex = identity
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    require(
        identity == crate::tokenizer::QWEN4EXP_RELEASE_TOKENIZER_IDENTITY_SHA256
            && hex == f.tokenizer_identity_sha256,
        "released tokenizer identity mismatch",
    )?;
    require(
        gguf.stop_token_ids()? == vec![STOP as i32],
        "stop-token set changed",
    )?;
    let template = gguf
        .get_str("tokenizer.chat_template")
        .ok_or("missing actual template")?;
    require(
        sha256_bytes(template.as_bytes()) == f.templates[artifact].sha256,
        "actual template changed",
    )?;
    let stamps = gguf.revalidate_retained_shard_stamps()?;
    let first = stamps.first().ok_or("missing retained shard")?;
    let expected = &f.metadata_models[artifact];
    require(
        first.path.file_name()
            == std::path::Path::new(expected["path"].as_str().unwrap()).file_name()
            && first.size == expected["size"].as_u64().unwrap(),
        "artifact label/header binding mismatch",
    )?;
    let tokenizer = Tokenizer::from_gguf(gguf)?;
    let encode = |t: &str| -> PacketResult<Vec<u32>> {
        Ok(tokenizer
            .encode(t, false)?
            .into_iter()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()?)
    };
    for d in &f.natural {
        let actual = encode(&text(&d.text_file, &d.rendered_sha256)?)?;
        require(
            actual.len() == d.complete_source_tokens
                && actual.get(..N + TARGETS) == Some(d.token_ids.as_slice()),
            "natural tokenizer drift",
        )?;
    }
    for d in &f.retrieval {
        let rendered = text(&d.text_file, &d.rendered_sha256)?;
        require(
            encode(&rendered)? == d.token_ids && rendered.ends_with("</think>\n\n"),
            "rendered retrieval drift",
        )?;
        let offset = rendered
            .find(&d.fact_text)
            .ok_or("missing complete authoritative fact")?;
        let start = encode(&rendered[..offset])?;
        let end = encode(&rendered[..offset + d.fact_text.len()])?;
        require(
            [start.len(), end.len()] == d.fact_token_range
                && d.token_ids.starts_with(&start)
                && d.token_ids.starts_with(&end),
            "fact token span drift",
        )?;
    }
    emit(
        out,
        json!({"event":"fixture_verified","artifact":artifact,"manifest_sha256":FIXTURE_SHA,
        "tokenizer_identity_sha256":hex,"actual_template_sha256":sha256_bytes(template.as_bytes()),
        "natural_documents":8,"targets_per_document":TARGETS,"retrieval_cases":2,"all_token_arrays_match_current_native_tokenizer":true}),
    );
    Ok(tokenizer)
}

fn nll(logits: &[f32], target: usize) -> PacketResult<(f64, usize)> {
    require(
        !logits.is_empty() && target < logits.len() && logits.iter().all(|v| v.is_finite()),
        "invalid score row",
    )?;
    let max = logits
        .iter()
        .map(|&v| f64::from(v))
        .fold(f64::NEG_INFINITY, f64::max);
    let z = logits
        .iter()
        .map(|&v| (f64::from(v) - max).exp())
        .sum::<f64>();
    let loss = z.ln() - (f64::from(logits[target]) - max);
    require(loss.is_finite(), "nonfinite NLL")?;
    Ok((loss, argmax(logits)))
}
fn parse_answer(bytes: &[u8]) -> Option<Answer> {
    let text = std::str::from_utf8(bytes).ok()?;
    let offset = text.find("FINAL_JSON:")? + "FINAL_JSON:".len();
    let tail = text[offset..].trim_start_matches([' ', '\r', '\n', '\t']);
    // Serde structs also accept positional JSON arrays; this policy requires
    // an object with named fields, so reject that representation explicitly.
    if !tail.starts_with('{') {
        return None;
    }
    let mut parser = serde_json::Deserializer::from_str(tail);
    Answer::deserialize(&mut parser).ok()
}
fn state(r: &Qwen4ExpTextRunner<'_, '_, '_>) -> Value {
    json!({"position":r.next_position(),"qsa_lengths":r.workspace.qsa_committed_lengths(),
        "ple_prior_tokens":r.workspace.ple_prior_tokens()})
}
fn fresh_prefill(
    r: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    ids: &[u32],
    b: bool,
    out: &mut std::fs::File,
    id: &str,
) -> PacketResult<()> {
    require(ids.len() == N, "quality requires exactly4096 prompt tokens")?;
    require(
        !crate::metal::dispatch_census_is_active(),
        "census must stay off",
    )?;
    r.reset()?;
    zero_persistent_state(r);
    require(
        r.next_position() == 0 && r.logits().is_err(),
        "reset failed",
    )?;
    let plan = normal_plan(r, N)?;
    require(
        plan.packed_ranges == ranges(0, b) && plan.scalar_start == N && plan.contains_selection,
        "schedule scope failed",
    )?;
    r.prefill(ids)?;
    let t = r
        .last_prefill_timing()
        .ok_or("missing prefill completion metadata")?;
    require(
        r.next_position() == N
            && t.command_count == plan.packed_ranges.len()
            && t.packed_token_count == N,
        "prefill publication mismatch",
    )?;
    emit(
        out,
        json!({"event":"prefill_complete","id":id,"arm":if b {"B"} else {"A"},
        "ranges":plan.packed_ranges.iter().map(|r|[r.start,r.end]).collect::<Vec<_>>(),
        "state":state(r),"command_count":t.command_count,"performance_claim":false}),
    );
    Ok(())
}
struct NaturalResult {
    mean: f64,
    hits: usize,
}
fn natural(
    r: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    d: &Natural,
    b: bool,
    out: &mut std::fs::File,
) -> PacketResult<NaturalResult> {
    with_qwen4exp_frontier_schedule(b, || {
        fresh_prefill(r, &d.token_ids[..N], b, out, &d.id)?;
        let mut losses = Vec::with_capacity(TARGETS);
        let mut hits = 0;
        for (step, &target) in d.token_ids[N..].iter().enumerate() {
            require(
                r.next_position() == N + step,
                "teacher-forced input frontier mismatch",
            )?;
            let logits = r.logits()?;
            require(
                logits.as_slice().len() == VOCAB,
                "wrong full-vocabulary size",
            )?;
            let (loss, top) = nll(logits.as_slice(), target as usize)?;
            hits += usize::from(top == target as usize);
            losses.push(loss);
            emit(
                out,
                json!({"event":"target","id":d.id,"arm":if b {"B"} else {"A"},
                "step":step,"target_id":target,"nll":loss,"top1":top,"correct":top==target as usize,
                "logit_count":logits.as_slice().len(),"all_finite":true,"state_before_feed":state(r)}),
            );
            drop(logits);
            r.forward_token(target)?;
        }
        require(
            r.next_position() == N + TARGETS,
            "teacher-forced terminal frontier mismatch",
        )?;
        let terminal = r.logits()?;
        require(
            terminal.as_slice().len() == VOCAB && terminal.as_slice().iter().all(|v| v.is_finite()),
            "nonfinite unscored terminal row",
        )?;
        let mean = losses.iter().sum::<f64>() / TARGETS as f64;
        emit(
            out,
            json!({"event":"natural_complete","id":d.id,"arm":if b {"B"} else {"A"},
            "nll":losses,"mean_nll":mean,"hits":hits,"scored_tokens":TARGETS,"terminal_row_scored":false,"state":state(r)}),
        );
        Ok(NaturalResult { mean, hits })
    })
}
struct RetrievalResult {
    parsed: bool,
    correct: bool,
}
fn retrieval(
    r: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tok: &Tokenizer,
    d: &Retrieval,
    b: bool,
    out: &mut std::fs::File,
) -> PacketResult<RetrievalResult> {
    with_qwen4exp_frontier_schedule(b, || {
        fresh_prefill(r, &d.token_ids, b, out, &d.id)?;
        let mut bytes = Vec::new();
        let mut ids = Vec::new();
        let mut stopped = false;
        for step in 0..TARGETS {
            let logits = r.logits()?;
            require(
                logits.as_slice().len() == VOCAB && logits.as_slice().iter().all(|v| v.is_finite()),
                "invalid retrieval logits",
            )?;
            let token = argmax(logits.as_slice()) as u32;
            drop(logits);
            if token == STOP {
                stopped = true;
                break;
            }
            bytes.extend_from_slice(tok.try_decode_piece_bytes_exact(i32::try_from(token)?)?);
            ids.push(token);
            r.forward_token(token)?;
            require(
                r.next_position() == N + step + 1,
                "greedy frontier mismatch",
            )?;
        }
        let terminal = r.logits()?;
        require(
            terminal.as_slice().len() == VOCAB && terminal.as_slice().iter().all(|v| v.is_finite()),
            "nonfinite retrieval terminal logits",
        )?;
        let answer = parse_answer(&bytes);
        let correct = answer.as_ref() == Some(&d.expected);
        emit(
            out,
            json!({"event":"retrieval_complete","id":d.id,"arm":if b {"B"} else {"A"},
            "generated_token_ids":ids,"generated_bytes":bytes,"generated_text":String::from_utf8_lossy(&bytes),
            "stopped_on_eos":stopped,"max_generated_tokens":TARGETS,"parsed_answer":answer,"expected":d.expected,
            "correct":correct,"fact_token_range":d.fact_token_range,"state":state(r),"all_finite":true}),
        );
        Ok(RetrievalResult {
            parsed: answer.is_some(),
            correct,
        })
    })
}
fn attempt<T>(
    out: &mut std::fs::File,
    id: &str,
    b: bool,
    work: impl FnOnce(&mut std::fs::File) -> PacketResult<T>,
) -> PacketResult<T> {
    emit(
        out,
        json!({"event":"attempt_begin","id":id,"arm":if b {"B"} else {"A"}}),
    );
    let result = work(out);
    if let Err(error) = &result {
        emit(
            out,
            json!({"event":"acquisition_error","id":id,"arm":if b {"B"} else {"A"},
            "error":error.to_string(),"disposition":if b {"SEMANTIC_FAIL"} else {"INVALID_ACQUISITION"}}),
        );
    }
    result
}
fn packet(out: &mut std::fs::File, artifact: &str) -> PacketResult<()> {
    let f = frozen()?;
    for (name, _) in std::env::vars_os() {
        let name = name.to_string_lossy();
        require(
            !name.starts_with("QWEN_") || name == "QWEN_METAL_LEASE_WAIT",
            "remove ambient QWEN override; quality uses committed defaults",
        )?;
        require(
            !name.starts_with("QWEN4EXP_"),
            "remove ambient Flash override; quality uses committed defaults",
        )?;
    }
    with_native_artifact(out, |ctx, gguf, out| {
        let tok = validate_tokenizer(gguf, &f, artifact, out)?;
        with_native_runner(ctx, gguf, N + TARGETS, 96 << 20, out, |r, out| {
            require(r.packed_qsa_dense_end()? == 2051, "wrong selected frontier")?;
            let mut deltas = Vec::new();
            let mut all_hits = [0usize; 2];
            for (i, d) in f.natural.iter().enumerate() {
                let mut arms: [Option<NaturalResult>; 2] = [None, None];
                for b in if i % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    arms[usize::from(b)] =
                        Some(attempt(out, &d.id, b, |out| natural(r, d, b, out))?);
                }
                let a = arms[0].as_ref().unwrap();
                let b = arms[1].as_ref().unwrap();
                let delta = b.mean - a.mean;
                deltas.push(delta);
                all_hits[0] += a.hits;
                all_hits[1] += b.hits;
                emit(
                    out,
                    json!({"event":"document_pair","artifact":artifact,"id":d.id,"mean_nll_A":a.mean,"mean_nll_B":b.mean,
                    "delta_B_minus_A":delta,"hits_A":a.hits,"hits_B":b.hits,"targets":TARGETS}),
                );
            }
            let delta = deltas.iter().sum::<f64>() / deltas.len() as f64;
            let nll_pass = delta <= LIMIT;
            let mut retrieval_fail = false;
            let mut retrieval_invalid = false;
            for (i, d) in f.retrieval.iter().enumerate() {
                let mut arms: [Option<RetrievalResult>; 2] = [None, None];
                for b in if i % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    arms[usize::from(b)] =
                        Some(attempt(out, &d.id, b, |out| retrieval(r, &tok, d, b, out))?);
                }
                let a = arms[0].as_ref().unwrap();
                let b = arms[1].as_ref().unwrap();
                let verdict = if !a.parsed {
                    retrieval_invalid = true;
                    "INCONCLUSIVE_INVALID_FIXTURE"
                } else if !b.correct {
                    retrieval_fail = true;
                    "FAIL"
                } else {
                    "PASS"
                };
                emit(
                    out,
                    json!({"event":"retrieval_pair","artifact":artifact,"id":d.id,"A_parsed":a.parsed,
                    "A_correct":a.correct,"B_parsed":b.parsed,"B_correct":b.correct,"verdict":verdict}),
                );
            }
            let verdict = if !nll_pass || retrieval_fail {
                "FAIL"
            } else if retrieval_invalid {
                "INCONCLUSIVE"
            } else {
                "PASS"
            };
            emit(
                out,
                json!({"event":"quality_summary","artifact":artifact,"document_deltas":deltas,
                "pooled_delta_nll":delta,"limit":LIMIT,"natural_pass":nll_pass,"scored_tokens_per_arm":8*TARGETS,
                "hits_A":all_hits[0],"hits_B":all_hits[1],"retrieval_failed":retrieval_fail,"retrieval_inconclusive":retrieval_invalid,
                "verdict":verdict,"uncertainty":"analyze.py resamples documents, not tokens; descriptive only",
                "claim":"frozen technical-text continuation and two controlled no-thinking retrieval cases at4096, separately per artifact",
                "promotion":false}),
            );
            Ok(())
        })
    })
}

#[test]
fn frontier_quality_metric_scorer_and_boundaries() {
    assert!((nll(&[0.0, 0.0], 0).unwrap().0 - 2.0_f64.ln()).abs() < 1e-14);
    assert_eq!(nll(&[2.0, 2.0, 1.0], 1).unwrap().1, 0);
    assert!(
        (nll(&[10000.0, 10001.0], 0).unwrap().0 - (1.0 + (-1.0_f64).exp().ln_1p())).abs() < 1e-12
    );
    assert!(nll(&[f32::NAN], 0).is_err());
    assert!(nll(&[], 0).is_err());
    assert!(nll(&[0.0], 1).is_err());
    let answer =
        parse_answer(b"reason\nFINAL_JSON: {\"record\":\"R\",\"code\":\"C\",\"extra\":1} tail")
            .unwrap();
    assert_eq!(
        answer,
        Answer {
            record: "R".into(),
            code: "C".into()
        }
    );
    for bad in [
        b"{\"record\":\"R\",\"code\":\"C\"}".as_slice(),
        b"FINAL_JSON: []",
        b"FINAL_JSON: [\"C\",\"R\"]",
        b"FINAL_JSON: {\"record\":1,\"code\":\"C\"}",
        b"FINAL_JSON: {\"record\":\"R\",\"code\":\"C\",\"code\":\"D\"}",
        b"FINAL_JSON: invalid FINAL_JSON: {\"record\":\"R\",\"code\":\"C\"}",
    ] {
        assert!(parse_answer(bad).is_none());
    }
    let f = frozen().unwrap();
    assert_eq!(
        f.natural
            .iter()
            .map(|d| d.token_ids[N..].len())
            .sum::<usize>(),
        512
    );
    assert_eq!(f.retrieval[0].fact_token_range, [1033, 1074]);
    assert_eq!(f.retrieval[1].fact_token_range, [3083, 3122]);
}

#[test]
#[ignore = "release; production lease/admission; frozen UD or GSQ; NEW FLASH_PREFILL_OUT"]
fn native_frontier_quality() {
    let artifact = std::env::var("FLASH_FRONTIER_QUALITY_ARTIFACT")
        .expect("FLASH_FRONTIER_QUALITY_ARTIFACT=ud|gsq required");
    assert!(["ud", "gsq"].contains(&artifact.as_str()));
    run_packet(
        "flash.frontier_quality.v1",
        include_bytes!("frontier_quality.rs"),
        json!({
            "artifact":artifact,"fixture_sha256":FIXTURE_SHA,"freeze_revision":2,"protocol_sha256":sha256_bytes(include_bytes!("../../../../../../docs/bench/2026-10-09-flash-frontier-quality/PROTOCOL.md")),
            "schedule_packet_sha256":sha256_bytes(include_bytes!("frontier_schedule.rs")),
            "order":"alternating AB/BA by frozen document index; independently alternating retrieval index",
            "A":"production2048+3+2045","B":"test-only2048+2048","arithmetic_and_router":"production in both arms",
            "capacity":N+TARGETS,"documents":8,"targets_per_document":TARGETS,"retrieval_cases":2,
            "nll_limit":LIMIT,"uncertainty":"descriptive document bootstrap only; no token pseudo-replication",
            "observers":false,"performance_claim":false,"automatic_promotion":false
        }),
        |out| packet(out, &artifact),
    );
}
