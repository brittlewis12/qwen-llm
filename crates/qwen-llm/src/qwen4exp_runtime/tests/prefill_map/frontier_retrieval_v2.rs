//! Separately frozen raw-JSON retrieval follow-up. No v1 reclassification.
//! FLASH_FRONTIER_RETRIEVAL_V2_ARTIFACT=ud|gsq; normal FLASH_PREFILL_MODEL/OUT.
use super::*;

const FIXTURE: &[u8] = include_bytes!(
    "../../../../../../docs/bench/2026-10-09-flash-frontier-retrieval-v2/fixtures.json"
);
const FIXTURE_SHA: &str = "352557183db3eac478ed4a40aa265fe1f366e982b7958b81ff33a900cd9d8bce";
const DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/bench/2026-10-09-flash-frontier-retrieval-v2"
);

#[derive(Deserialize)]
struct V2Fixtures {
    schema: String,
    tokenizer_identity_sha256: String,
    vocab_size: usize,
    stop_token: u32,
    prefix: usize,
    continuation: usize,
    retrieval: Vec<Retrieval>,
    files: Vec<Binding>,
    templates: BTreeMap<String, Binding>,
    metadata_models: Value,
}
fn text(name: &str, hash: &str) -> PacketResult<String> {
    require(
        std::path::Path::new(name).components().count() == 1,
        "v2 fixture must be a local filename",
    )?;
    let bytes = std::fs::read(std::path::Path::new(DIR).join(name))?;
    require(sha256_bytes(&bytes) == hash, "v2 text binding changed")?;
    Ok(String::from_utf8(bytes)?)
}
fn frozen() -> PacketResult<V2Fixtures> {
    require(sha256_bytes(FIXTURE) == FIXTURE_SHA, "v2 freeze changed")?;
    let f: V2Fixtures = serde_json::from_slice(FIXTURE)?;
    require(
        f.schema == "flash.frontier_retrieval.fixtures.v2"
            && f.prefix == N
            && f.continuation == TARGETS
            && f.vocab_size == VOCAB
            && f.stop_token == STOP
            && f.retrieval.len() == 2,
        "v2 geometry/cohort changed",
    )?;
    for binding in &f.files {
        let rendered = text(&binding.file, &binding.sha256)?;
        require(rendered.len() == binding.bytes, "v2 file size changed")?;
    }
    for (i, d) in f.retrieval.iter().enumerate() {
        let [start, end] = d.fact_token_range;
        require(
            d.id == ["v2_before", "v2_after"][i]
                && d.token_ids.len() == N
                && d.maximum_generated_tokens == TARGETS
                && start < end
                && end < N
                && if i == 0 { end <= 2051 } else { start >= 2051 },
            "v2 boundaries changed",
        )?;
        require(
            d.token_ids.iter().all(|&id| (id as usize) < VOCAB)
                && sha256_u32_le(b"", &d.token_ids) == d.tokens_sha256_u32le,
            "v2 token IDs/hash changed",
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
            "v2 answer must occur solely in authoritative fact",
        )?;
    }
    Ok(f)
}
fn validate_tokenizer(
    gguf: &GgufFile,
    f: &V2Fixtures,
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
            && hex == f.tokenizer_identity_sha256
            && gguf.stop_token_ids()? == vec![STOP as i32],
        "v2 tokenizer/stop identity changed",
    )?;
    let template = gguf
        .get_str("tokenizer.chat_template")
        .ok_or("missing template")?;
    require(
        sha256_bytes(template.as_bytes()) == f.templates[artifact].sha256,
        "v2 actual template changed",
    )?;
    let stamps = gguf.revalidate_retained_shard_stamps()?;
    let first = stamps.first().ok_or("missing retained shard")?;
    let expected = &f.metadata_models[artifact];
    require(
        first.path.file_name()
            == std::path::Path::new(expected["path"].as_str().ok_or("missing source path")?)
                .file_name()
            && Some(first.size) == expected["size"].as_u64(),
        "v2 artifact label/header mismatch",
    )?;
    let tok = Tokenizer::from_gguf(gguf)?;
    let encode = |s: &str| -> PacketResult<Vec<u32>> {
        Ok(tok
            .encode(s, false)?
            .into_iter()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()?)
    };
    for d in &f.retrieval {
        let rendered = text(&d.text_file, &d.rendered_sha256)?;
        require(
            encode(&rendered)? == d.token_ids && rendered.ends_with("</think>\n\n"),
            "v2 rendered/tokenizer drift",
        )?;
        let offset = rendered
            .find(&d.fact_text)
            .ok_or("missing authoritative fact")?;
        let start = encode(&rendered[..offset])?;
        let end = encode(&rendered[..offset + d.fact_text.len()])?;
        require(
            [start.len(), end.len()] == d.fact_token_range
                && d.token_ids.starts_with(&start)
                && d.token_ids.starts_with(&end),
            "v2 fact span drift",
        )?;
    }
    emit(
        out,
        json!({"event":"fixture_verified","artifact":artifact,
        "manifest_sha256":FIXTURE_SHA,"tokenizer_identity_sha256":hex,
        "all_token_arrays_match_current_native_tokenizer":true,"retrieval_cases":2}),
    );
    Ok(tok)
}
fn parse_answer(bytes: &[u8]) -> Option<Answer> {
    let text = std::str::from_utf8(bytes).ok()?;
    let text = text.trim_start_matches([' ', '\t', '\r', '\n']);
    // Struct deserialization also accepts arrays; the contract requires an object.
    if !text.starts_with('{') {
        return None;
    }
    let mut parser = serde_json::Deserializer::from_str(text);
    let answer = Answer::deserialize(&mut parser).ok()?;
    parser.end().ok()?; // Only JSON whitespace may follow the first object.
    Some(answer)
}
fn case_verdict(a: &RetrievalResult, b: &RetrievalResult) -> &'static str {
    if !a.parsed {
        "INCONCLUSIVE_INVALID_FIXTURE"
    } else if !b.correct {
        "FAIL"
    } else {
        "PASS"
    }
}
fn overall(failed: bool, inconclusive: bool) -> &'static str {
    if failed {
        "FAIL"
    } else if inconclusive {
        "INCONCLUSIVE"
    } else {
        "PASS"
    }
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
                "v2 nonfinite/wrong-size logits",
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
                "v2 greedy frontier mismatch",
            )?;
        }
        let terminal = r.logits()?;
        require(
            terminal.as_slice().len() == VOCAB && terminal.as_slice().iter().all(|v| v.is_finite()),
            "v2 invalid terminal row",
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
fn packet(out: &mut std::fs::File, artifact: &str) -> PacketResult<()> {
    let f = frozen()?;
    for (name, _) in std::env::vars_os() {
        let name = name.to_string_lossy();
        require(
            (!name.starts_with("QWEN_") || name == "QWEN_METAL_LEASE_WAIT")
                && !name.starts_with("QWEN4EXP_"),
            "remove ambient arithmetic/diagnostic overrides",
        )?;
    }
    with_native_artifact(out, |ctx, gguf, out| {
        let tok = validate_tokenizer(gguf, &f, artifact, out)?;
        with_native_runner(ctx, gguf, N + TARGETS, 96 << 20, out, |r, out| {
            require(r.packed_qsa_dense_end()? == 2051, "wrong selected frontier")?;
            let mut failed = false;
            let mut inconclusive = false;
            for (i, d) in f.retrieval.iter().enumerate() {
                let mut arms: [Option<RetrievalResult>; 2] = [None, None];
                for b in if i == 0 { [false, true] } else { [true, false] } {
                    arms[usize::from(b)] =
                        Some(attempt(out, &d.id, b, |out| retrieval(r, &tok, d, b, out))?);
                }
                let a = arms[0].as_ref().unwrap();
                let b = arms[1].as_ref().unwrap();
                let verdict = case_verdict(a, b);
                failed |= verdict == "FAIL";
                inconclusive |= verdict == "INCONCLUSIVE_INVALID_FIXTURE";
                emit(
                    out,
                    json!({"event":"retrieval_pair","artifact":artifact,"id":d.id,
                    "A_parsed":a.parsed,"A_correct":a.correct,"B_parsed":b.parsed,"B_correct":b.correct,"verdict":verdict}),
                );
            }
            emit(
                out,
                json!({"event":"retrieval_v2_summary","artifact":artifact,
                "verdict":overall(failed,inconclusive),"retrieval_failed":failed,"retrieval_inconclusive":inconclusive,
                "v1_reclassified":false,"nll_rerun":false,"automatic_promotion":false}),
            );
            Ok(())
        })
    })
}

#[test]
fn frontier_retrieval_v2_scorer_and_boundaries() {
    let expected = Answer {
        record: "R".into(),
        code: "C".into(),
    };
    for good in [
        b"{\"record\":\"R\",\"code\":\"C\"}".as_slice(),
        b" \t\r\n{\"code\":\"C\",\"record\":\"R\",\"extra\":1}\n\t",
    ] {
        assert_eq!(
            parse_answer(good),
            Some(Answer {
                record: expected.record.clone(),
                code: expected.code.clone()
            })
        );
    }
    for bad in [
        b"FINAL_JSON: {\"code\":\"C\",\"record\":\"R\"}".as_slice(),
        b"[\"C\",\"R\"]",
        b"{\"code\":\"C\",\"record\":\"R\"} commentary",
        b"{\"code\":\"C\",\"record\":\"R\"}{}",
        b"{\"code\":\"C\",\"record\":\"R\",\"code\":\"C\"}",
        b"{\"code\":\"C\",\"record\":\"R\",\"record\":\"R\"}",
        b"{\"code\":1,\"record\":\"R\"}",
        b"{\"code\":\"C\"}",
        b"{\"code\":\"C\",\"record\":\"R\",\"x\":NaN}",
        b"{\"code\":\"C\",\"record\":\"R\"}\x0b",
        b"\xff",
        b"{\"code\":",
    ] {
        assert!(parse_answer(bad).is_none(), "accepted {bad:?}");
    }
    let invalid = RetrievalResult {
        parsed: false,
        correct: false,
    };
    let wrong = RetrievalResult {
        parsed: true,
        correct: false,
    };
    let correct = RetrievalResult {
        parsed: true,
        correct: true,
    };
    assert_eq!(
        case_verdict(&invalid, &correct),
        "INCONCLUSIVE_INVALID_FIXTURE"
    );
    assert_eq!(case_verdict(&wrong, &wrong), "FAIL");
    assert_eq!(case_verdict(&wrong, &correct), "PASS");
    assert_eq!(case_verdict(&correct, &invalid), "FAIL");
    assert_eq!(overall(true, true), "FAIL");
    assert_eq!(overall(false, true), "INCONCLUSIVE");
    assert_eq!(overall(false, false), "PASS");
    let f = frozen().unwrap();
    assert_eq!(f.retrieval[0].expected.record, "P-28");
    assert_eq!(f.retrieval[1].expected.record, "W-57");
}

#[test]
#[ignore = "release; normal production lease/admission; NEW FLASH_PREFILL_OUT; ud or gsq"]
fn native_frontier_retrieval_v2() {
    let artifact = std::env::var("FLASH_FRONTIER_RETRIEVAL_V2_ARTIFACT")
        .expect("FLASH_FRONTIER_RETRIEVAL_V2_ARTIFACT=ud|gsq required");
    assert!(["ud", "gsq"].contains(&artifact.as_str()));
    run_packet(
        "flash.frontier_retrieval.v2",
        include_bytes!("frontier_retrieval_v2.rs"),
        json!({
            "artifact":artifact,"fixture_sha256":FIXTURE_SHA,
            "protocol_sha256":sha256_bytes(include_bytes!("../../../../../../docs/bench/2026-10-09-flash-frontier-retrieval-v2/PROTOCOL.md")),
            "quality_helpers_sha256":sha256_bytes(include_bytes!("frontier_quality.rs")),
            "schedule_packet_sha256":sha256_bytes(include_bytes!("frontier_schedule.rs")),
            "A":"production2048+3+2045","B":"test-only2048+2048","arithmetic_and_router":"production in both arms",
            "capacity":N+TARGETS,"order":"v2_before AB; v2_after BA","contract":"strict raw JSON; no marker; trailing JSON whitespace only",
            "independent_replication":false,"v1_reclassified":false,"nll_rerun":false,"performance_claim":false,"automatic_promotion":false
        }),
        |out| packet(out, &artifact),
    );
}
