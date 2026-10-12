//! One resident model, ordinary A/B, two requests per arm. No HTTP or planner override.
use super::*;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const PACKET_SHA: &str = "e6e72a87bbb973c72e5e7dc11519f405296a790d0cda7f0d65b54503fb70a2d4";
const BRANCH_QUESTION: &str = "Using only this repository source, explain how cancellation during prefill versus decode affects retained live state and previously captured snapshots. Can a retry reuse a snapshot after decode cancellation? Contrast Fast shared-prefix snapshots with Exact transcript snapshots and explain the Fast canonicality check. Cite concrete paths and function/test names. Keep the answer under 400 words. Do not call tools.";

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn edited_question(packet: &str) -> String {
    let (source, _) = packet
        .rsplit_once("\n===== QUESTION =====\n")
        .expect("frozen question boundary");
    format!("{source}\n===== QUESTION =====\n{BRANCH_QUESTION}")
}

fn sha_file(path: &Path) -> String {
    let mut file = std::fs::File::open(path).unwrap();
    let mut hash = Sha256::new();
    let mut bytes = [0; 65536];
    loop {
        let n = file.read(&mut bytes).unwrap();
        if n == 0 {
            break;
        }
        hash.update(&bytes[..n]);
    }
    format!("{:x}", hash.finalize())
}

fn sha_bits(bits: &[u32]) -> String {
    let mut hash = Sha256::new();
    for value in bits {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn emit(out: &mut std::fs::File, mut value: Value) {
    value["recorded_unix_ms"] = json!(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
    serde_json::to_writer(&mut *out, &value).unwrap();
    writeln!(out).unwrap();
    out.flush().unwrap();
    eprintln!("chunk_checkpoint_ab {value}");
}

struct TimedSink {
    bytes: Vec<u8>,
    started: Instant,
    first_piece_ms: Option<f64>,
}

impl GenerationSink for TimedSink {
    fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.first_piece_ms
            .get_or_insert_with(|| self.started.elapsed().as_secs_f64() * 1e3);
        if self.bytes.len() + bytes.len() > self.bytes.capacity() {
            return Err(io::Error::other(
                "decoded output exceeded priced token-piece bound",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn tick(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn frozen_packet_edit_changes_only_final_question() {
    let path =
        repository_root().join("docs/bench/2026-10-12-glm-maya-repository-context/packet.txt");
    assert_eq!(sha_file(&path), PACKET_SHA);
    let packet = std::fs::read_to_string(path).unwrap();
    let branch = edited_question(&packet);
    let source_end = packet.rfind("\n===== QUESTION =====\n").unwrap();
    assert_eq!(&packet[..source_end], &branch[..source_end]);
    assert_ne!(packet, branch);
    assert!(branch.ends_with(BRANCH_QUESTION));
}

#[test]
#[ignore = "GLM53_GGUF + new GLM53_STAGE_OUT path; bounded 64K repository A/B, production internal lease, no server"]
fn gpu_repository_chunk_checkpoint_ab() {
    let root = repository_root();
    let packet_path = root.join("docs/bench/2026-10-12-glm-maya-repository-context/packet.txt");
    assert_eq!(sha_file(&packet_path), PACKET_SHA);
    let packet = std::fs::read_to_string(&packet_path).unwrap();
    let branch = edited_question(&packet);
    let manifest_path = packet_path.with_file_name("packet.json");
    let manifest: Value = serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let instructions = manifest["instructions"].as_str().unwrap();
    let out_path = std::env::var("GLM53_STAGE_OUT").expect("new GLM53_STAGE_OUT JSONL path");
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out_path)
        .unwrap();
    let model = gguf_path();
    let model_metadata = std::fs::metadata(&model).unwrap();
    let binary = std::env::current_exe().unwrap();
    let git = |args: &[&str]| {
        String::from_utf8(
            std::process::Command::new("git")
                .args(args)
                .current_dir(&root)
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
    };
    let hashes: Vec<_> = [
        "crates/qwen-cli/src/serve/backend_glm5_next.rs",
        "crates/qwen-cli/src/serve/backend_glm5_next/chunk_checkpoint.rs",
        "crates/qwen-cli/src/serve/backend_glm5_next/tests/chunk_checkpoint_ab.rs",
        "crates/qwen-llm/src/glm5_next_metal/packed.rs",
    ]
    .iter()
    .map(|path| json!({"path":path,"sha256":sha_file(&root.join(path))}))
    .collect();
    let overrides: std::collections::BTreeMap<_, _> = std::env::vars()
        .filter(|(key, _)| {
            key.starts_with("QWEN_GLM_")
                || key.starts_with("QWEN_METAL_")
                || key.starts_with("QWEN_SERVE_")
                || key.starts_with("MTL_")
                || key == "RUST_LOG"
        })
        .collect();
    emit(
        &mut out,
        json!({"event":"run","schema":"glm53.backend_chunk_checkpoint_ab.v1",
        "model":model,"model_first_shard_bytes":model_metadata.len(),
        "model_first_shard_mtime":format!("{:?}",model_metadata.modified().unwrap()),
        "model_identity_note":"path/stat, no weight-content hash", "binary":binary,"binary_sha256":sha_file(&binary),
        "source_revision":git(&["rev-parse","HEAD"]).trim(),"source_dirty":git(&["status","--porcelain"]),
        "source_revision_scope":"checkout hashes; binary source is not attested", "source_hashes":hashes,
        "packet_path":packet_path,"packet_sha256":PACKET_SHA,"packet_manifest_sha256":sha_file(&manifest_path),
        "packet_bytes":packet.len(),"branch_bytes":branch.len(),"branch_question":BRANCH_QUESTION,
        "inherited_overrides":overrides,"capacity":65536,"precision":"half","max_output_tokens":1024,
        "timing_mode":if ["MTL_DEBUG_LAYER","MTL_SHADER_VALIDATION"].iter().any(|key|std::env::var(key).is_ok_and(|v|v!="0" && !v.is_empty())) { "validation-on" } else { "validation-not-requested" },
        "temperature":0,"effort":"low","order":"A seed/branch, B seed/branch",
        "checkpoint_policy":{"A":"forced incumbent (Some(false))","B":"production default (None)"},
        "timing":"ordinary backend wall and first decoded piece; not HTTP first-answer latency; load, warmup, request rendering, sink allocation and comparisons excluded",
        "limits":"single AB, no speed threshold or task-quality verdict; seed timings may include first-use work"}),
    );
    let source = GgufFile::open(&model).unwrap();
    let inv = invocation(&model, Some(65536), Some(1024));
    let prepared = Prepared::new(&source, &inv).unwrap();
    assert_eq!(
        prepared.fast_precision,
        FastPrecision::Half,
        "diagnostic requires default Half; remove caller precision override"
    );
    // CLI tests link the production library. Its context owns the real lease;
    // no external/test lease and no arbitrary reserve/headroom adjustment.
    let ctx = MetalContext::new().unwrap();
    let load_started = Instant::now();
    let weights = load(&ctx, &source, &prepared).unwrap();
    let load_ms = load_started.elapsed().as_secs_f64() * 1e3;
    let cache_plan =
        crate::serve::SnapshotCachePlan::resolve(None, Default::default(), ctx.memory_signals())
            .unwrap();
    let mut warm = Glm5NextBackend::new(
        &ctx,
        &weights,
        prepared,
        "glm".into(),
        std::time::Duration::ZERO,
        cache_plan,
    );
    let warm_ms = warm.warm_up().unwrap();
    drop(warm);
    emit(
        &mut out,
        json!({"event":"startup","load_ms":load_ms,"warmup_ms":warm_ms,"cache_budget":cache_plan.bytes,
        "cpu_storage":"resident packet/prompts/token vectors count in process signals; output buffer allocated before timing at 1024 * tokenizer max_piece_bytes; capture uses existing exact snapshot/key admission"}),
    );
    let mut references: Vec<(Vec<u32>, Vec<u8>, GenerationEnd)> = Vec::new();
    let mut rows_reference = None;
    let mut branch_walls = Vec::new();
    for candidate in [false, true] {
        let arm = if candidate { "B" } else { "A" };
        let prepared = Prepared::new(&source, &inv).unwrap();
        let mut backend = Glm5NextBackend::new(
            &ctx,
            &weights,
            prepared,
            "glm".into(),
            std::time::Duration::ZERO,
            cache_plan,
        );
        backend.chunk_checkpoint_override = if candidate { None } else { Some(false) };
        assert!(
            backend.prefix_reuse
                && backend.fast_snapshots
                && backend.lineage == PackedLineage::Fast,
            "diagnostic requires default Fast shared-split reuse"
        );
        let requests: Vec<_> = [&packet, &branch]
            .iter()
            .map(|input| {
                prepared_request(
                    &backend,
                    json!({
                        "model":"glm","instructions":instructions,"tools":[],"tool_choice":"auto",
                        "input":[{"role":"user","content":input}],"reasoning":{"effort":"low"},
                        "temperature":0,"max_output_tokens":1024
                    }),
                )
            })
            .collect();
        let tokenizer = backend.prepared.artifact.tokenizer();
        let encode = |text: &str| -> Result<Vec<u32>, ServeError> {
            decode_loop::encode_checked(tokenizer, text, false, weights.config.vocab_size, FAMILY)
        };
        let tokens: Vec<_> = requests
            .iter()
            .map(|p| encode(&p.prompt).unwrap())
            .collect();
        let cuts: Vec<_> = requests
            .iter()
            .zip(&tokens)
            .map(|(p, t)| {
                cut_positions(
                    encode,
                    &p.prompt,
                    t,
                    p.boundaries,
                    Schedule::FastSharedSplitV1,
                )
            })
            .collect();
        assert_eq!(cuts[0], cuts[1]);
        assert_eq!(cuts[0].len(), 1);
        assert!(
            !tokens[1].starts_with(&tokens[0]),
            "branch must edit the prompt, not extend it"
        );
        emit(
            &mut out,
            json!({"event":"arm","arm":arm,"input_tokens":[tokens[0].len(),tokens[1].len()],"shared_cut":cuts[0][0],
            "prompt_storage_bytes":requests.iter().map(|p|p.prompt.capacity()).sum::<usize>(),
            "token_storage_bytes":tokens.iter().map(|t|t.capacity()*size_of::<u32>()).sum::<usize>(),
            "reference_storage_bytes":references.iter().map(|(bits,bytes,_)| bits.capacity()*4+bytes.capacity()).sum::<usize>()}),
        );
        for (phase, request) in requests.iter().enumerate() {
            if phase == 1 {
                let rows = backend.frozen_rows.unwrap();
                let cut = chunk_checkpoint::target(tokens[0].len(), cuts[0][0], rows)
                    .expect("full suffix chunk");
                assert!(
                    tokens[1].len() > cut && tokens[1][..cut] == tokens[0][..cut],
                    "edited question must preserve checkpoint token prefix"
                );
                assert_eq!(
                    decode_loop::extending_prefix(
                        &backend.history,
                        &tokens[1],
                        backend.session.as_ref().unwrap().position(),
                        true
                    ),
                    0
                );
                emit(
                    &mut out,
                    json!({"event":"checkpoint_prefix_verified","arm":arm,
                    "seed_prompt_tokens":tokens[0].len(),"branch_prompt_tokens":tokens[1].len(),
                    "shared_cut":cuts[0][0],"rows":rows,"checkpoint":cut,
                    "seed_remainder":tokens[0].len()-cut,"branch_remainder":tokens[1].len()-cut,
                    "shared_checkpoint_prefix_sha256":sha_bits(&tokens[0][..cut])}),
                );
            }
            let output_bytes = 1024usize
                .checked_mul(backend.prepared.max_piece_bytes)
                .unwrap();
            let mut sink = TimedSink {
                bytes: Vec::with_capacity(output_bytes),
                started: Instant::now(),
                first_piece_ms: None,
            };
            emit(
                &mut out,
                json!({"event":"request_begin","arm":arm,"phase":phase,"output_capacity_bytes":sink.bytes.capacity()}),
            );
            sink.started = Instant::now();
            let outcome = backend
                .generate_prepared(Arc::clone(request), &mut sink)
                .unwrap();
            let wall_ms = sink.started.elapsed().as_secs_f64() * 1e3;
            let bits = &backend.last_prefill_logits;
            let finite = bits.iter().all(|b| f32::from_bits(*b).is_finite());
            let matches_reference = !candidate
                || (*bits == references[phase].0
                    && sink.bytes == references[phase].1
                    && outcome.end == references[phase].2);
            let (reasoning, answer) = split(&sink.bytes, outcome.end);
            emit(
                &mut out,
                json!({"event":"request_end","arm":arm,"phase":phase,"wall_ms":wall_ms,
                "first_piece_ms":sink.first_piece_ms,"input_tokens":outcome.usage.input_tokens,"cached_tokens":outcome.usage.cached_tokens,
                "output_tokens":outcome.usage.output_tokens,"truncated":outcome.end.is_token_limit(),"end":format!("{:?}",outcome.end),
                "prefill_logits_sha256":sha_bits(bits),"prefill_logits_finite":finite,"full_logits_and_greedy_bytes_equal":candidate.then_some(matches_reference),
                "output_sha256":format!("{:x}",Sha256::digest(&sink.bytes)),"reasoning":reasoning,"answer":answer,
                "phases":backend.last_chunk_checkpoint,"session_bytes":backend.session.as_ref().unwrap().ledger().session_buffer_bytes(),
                "metal_current_allocated_bytes":ctx.memory_signals().current_allocated_bytes}),
            );
            assert!(
                finite && matches_reference,
                "candidate must match same-schedule incumbent, see flushed evidence"
            );
            assert!(
                backend.chunk_origin.is_none(),
                "decode must invalidate live chunk provenance"
            );
            if phase == 0 {
                assert_eq!(outcome.usage.cached_tokens, 0);
                let rows = backend.frozen_rows.unwrap();
                if let Some(expected) = rows_reference {
                    assert_eq!(rows, expected, "workspace must not change between arms");
                } else {
                    rows_reference = Some(rows);
                }
                assert!(rows > 0);
            } else {
                let expected = if candidate {
                    chunk_checkpoint::target(
                        tokens[0].len(),
                        cuts[0][0],
                        backend.frozen_rows.unwrap(),
                    )
                    .unwrap()
                } else {
                    cuts[0][0]
                };
                // Admission refusal stays a correct fallback, never a fake performance win.
                emit(
                    &mut out,
                    json!({"event":"branch_selection","arm":arm,"expected_reuse_if_admitted":expected,
                    "actual_reuse":outcome.usage.cached_tokens,"deep_selected":backend.last_chunk_checkpoint.as_ref().unwrap()["restore_origin"]=="fast_chunk_checkpoint_v1"}),
                );
                branch_walls.push(wall_ms);
            }
            if !candidate {
                references.push((bits.clone(), sink.bytes, outcome.end));
            }
        }
        drop(backend);
        emit(
            &mut out,
            json!({"event":"arm_dropped","arm":arm,"metal_current_allocated_bytes":ctx.memory_signals().current_allocated_bytes}),
        );
    }
    emit(
        &mut out,
        json!({"event":"end","branch_wall_ratio_a_over_b":branch_walls[0]/branch_walls[1],
        "status":"same-schedule output comparison complete; inspect admission/restore origin and answers; no quality or performance threshold"}),
    );
}
