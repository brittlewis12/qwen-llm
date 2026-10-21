//! PRIVATE bounded raw-text likelihood witness. No capture feature or route hooks.
//! See v41_text_likelihood.md for the packet policy, commands, and evidence limits.
use anyhow::{Context, Result, ensure};
use qwen_llm::{
    deepseek_v41::{
        forward::{Forward, PREFILL_CHUNK_ROWS},
        gguf::ResidentWeights,
    },
    gguf::GgufFile,
    metal::MetalContext,
    tokenizer::NativeTokenizer,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
    process::{Command, ExitCode},
    sync::Arc,
};

const PREFIX: usize = 512;
const TARGETS: usize = 64;
const VOCAB: usize = 129280;
const DOCUMENTS: usize = 6;
const SCHEMA: &str = "private.v41.text-likelihood.packet.v1";
const POLICY: &str = "NativeTokenizer::from_gguf; encode(whole text, false) exactly once; no chat template, GLM IDs, trimming, or normalization by this helper; no automatic BOS/EOS; literal special-token text recognized by NativeTokenizer; freeze IDs [0,512) and [512,576); discard only the declared remaining token suffix; no document selection";
const RECIPE: &str =
    "cargo build --offline --locked --release -p qwen-cli --example v41_text_likelihood";

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Document {
    name: String,
    stratum: String,
    source_commit: String,
    path: String,
    text: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    documents: Vec<Document>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FrozenDocument {
    source: Document,
    whole_text_tokens: usize,
    prefix_ids: Vec<u32>,
    target_ids: Vec<u32>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    schema: String,
    policy: String,
    prefix_length: usize,
    target_length: usize,
    vocab: usize,
    model: Value,
    tokenizer: Value,
    prepare_build: Value,
    documents: Vec<FrozenDocument>,
}

fn validate_documents<'a>(documents: impl Iterator<Item = &'a Document>) -> Result<()> {
    let mut names = BTreeSet::new();
    let mut sources = BTreeSet::new();
    let (mut prose, mut code) = (0, 0);
    for doc in documents {
        ensure!(
            !doc.name.trim().is_empty() && names.insert(&doc.name),
            "empty or duplicate document name"
        );
        ensure!(
            !doc.path.trim().is_empty()
                && !doc.text.trim().is_empty()
                && matches!(doc.source_commit.len(), 40 | 64)
                && doc.source_commit.bytes().all(|b| b.is_ascii_hexdigit()),
            "{}: require raw text, source path, and full source commit",
            doc.name
        );
        ensure!(
            sources.insert((doc.source_commit.to_ascii_lowercase(), &doc.path)),
            "duplicate source commit/path"
        );
        match doc.stratum.as_str() {
            "prose" => prose += 1,
            "code" => code += 1,
            _ => anyhow::bail!("{}: stratum must be prose or code", doc.name),
        }
    }
    ensure!(
        prose == 3 && code == 3,
        "v1 requires exactly 3 prose + 3 code documents"
    );
    Ok(())
}

fn freeze(source: Document, ids: Vec<i32>) -> Result<FrozenDocument> {
    ensure!(
        ids.len() >= PREFIX + TARGETS,
        "{}: too short: {} tokens, need {} (no document skipped)",
        source.name,
        ids.len(),
        PREFIX + TARGETS
    );
    ensure!(
        ids.iter().all(|&id| id >= 0 && (id as usize) < VOCAB),
        "{}: tokenizer ID outside vocabulary",
        source.name
    );
    Ok(FrozenDocument {
        source,
        whole_text_tokens: ids.len(),
        prefix_ids: ids[..PREFIX].iter().map(|&id| id as u32).collect(),
        target_ids: ids[PREFIX..PREFIX + TARGETS]
            .iter()
            .map(|&id| id as u32)
            .collect(),
    })
}

fn validate_packet(packet: &Packet) -> Result<()> {
    ensure!(
        packet.schema == SCHEMA
            && packet.policy == POLICY
            && packet.prefix_length == PREFIX
            && packet.target_length == TARGETS
            && packet.vocab == VOCAB,
        "unsupported packet schema, policy, lengths, or vocabulary"
    );
    validate_documents(packet.documents.iter().map(|doc| &doc.source))?;
    for doc in &packet.documents {
        ensure!(
            doc.whole_text_tokens >= PREFIX + TARGETS
                && doc.prefix_ids.len() == PREFIX
                && doc.target_ids.len() == TARGETS,
            "{}: invalid packet lengths",
            doc.source.name
        );
        ensure!(
            doc.prefix_ids
                .iter()
                .chain(&doc.target_ids)
                .all(|&id| (id as usize) < VOCAB),
            "{}: packet ID outside vocabulary",
            doc.source.name
        );
    }
    Ok(())
}

fn create_new(path: &Path) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create fresh {}", path.display()))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = create_new(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    write_new(path, &serde_json::to_vec_pretty(value)?)
}

fn command_text(program: &str, args: &[&str]) -> Option<String> {
    Command::new(program)
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_owned())
}

fn build_identity() -> Value {
    let (compiler, sdk) = qwen_llm::metallib_build_identity();
    json!({
        "commit":env!("QWEN_BUILD_COMMIT"), "dirty":env!("QWEN_BUILD_DIRTY"),
        "source_state":env!("QWEN_BUILD_SOURCE_STATE"),
        "stamp_source":env!("QWEN_BUILD_STAMP_SOURCE"), "stamp_error":env!("QWEN_BUILD_STAMP_ERROR"),
        "debug_assertions":cfg!(debug_assertions), "recipe":RECIPE,
        "metal_compiler":compiler, "metal_sdk":sdk,
        "diagnostic_features":qwen_llm::enabled_diagnostic_features(),
        "rustflags":option_env!("RUSTFLAGS"), "encoded_rustflags":option_env!("CARGO_ENCODED_RUSTFLAGS"),
        "runtime_rustc_not_build_attestation":command_text("rustc", &["-vV"]),
        "runtime_commit":command_text("git", &["rev-parse", "HEAD"]),
        "runtime_status":command_text("git", &["status", "--porcelain=v1", "--untracked-files=all"]),
        "executable":std::env::current_exe().ok(),
    })
}

fn model_info(gguf: &GgufFile) -> Result<Value> {
    ensure!(
        gguf.architecture().as_deref() == Some("deepseek41"),
        "requires deepseek41 GGUF"
    );
    Ok(json!({
        "path":gguf.primary_shard().path, "file_bytes":gguf.total_mapped_len(),
        "architecture":gguf.architecture(), "authenticated":false, "hashes":false,
        "shards":gguf.shards.iter().map(|s| json!({"path":s.path, "bytes":s.mmap_len()})).collect::<Vec<_>>(),
    }))
}

fn tokenizer_info(gguf: &GgufFile, tokenizer: &NativeTokenizer) -> Result<Value> {
    ensure!(
        tokenizer.n_vocab() as usize == VOCAB,
        "requires {VOCAB} tokenizer vocabulary"
    );
    Ok(json!({
        "implementation":"qwen_llm::tokenizer::NativeTokenizer", "vocab":tokenizer.n_vocab(),
        "model":gguf.get_str("tokenizer.ggml.model"), "pre":gguf.get_str("tokenizer.ggml.pre"),
        "bos":tokenizer.bos(), "eos":tokenizer.eos(),
        "gguf_add_bos":gguf.get_bool("tokenizer.ggml.add_bos_token")?,
        "gguf_add_eos":gguf.get_bool("tokenizer.ggml.add_eos_token")?,
        "encode_add_special":false, "literal_special_tokens":"recognized, not stripped or escaped",
    }))
}

fn prepare(model: &Path, corpus: &Path, output: &Path) -> Result<()> {
    let mut file = create_new(output)?;
    let build = build_identity();
    let result = (|| -> Result<Packet> {
        qwen_cli::shutdown::install()?;
        let corpus: Corpus = serde_json::from_slice(&fs::read(corpus)?)?;
        validate_documents(corpus.documents.iter())?;
        qwen_cli::shutdown::checkpoint()?;
        let gguf = GgufFile::open(model.canonicalize()?)?;
        let model = model_info(&gguf)?;
        let tokenizer = NativeTokenizer::from_gguf(&gguf)?;
        let tokenizer_metadata = tokenizer_info(&gguf, &tokenizer)?;
        let mut documents = Vec::with_capacity(DOCUMENTS);
        for source in corpus.documents {
            qwen_cli::shutdown::checkpoint()?;
            let ids = tokenizer
                .encode(&source.text, false)
                .with_context(|| source.name.clone())?;
            documents.push(freeze(source, ids)?);
        }
        let packet = Packet {
            schema: SCHEMA.into(),
            policy: POLICY.into(),
            prefix_length: PREFIX,
            target_length: TARGETS,
            vocab: VOCAB,
            model,
            tokenizer: tokenizer_metadata,
            prepare_build: build.clone(),
            documents,
        };
        validate_packet(&packet)?;
        qwen_cli::shutdown::checkpoint()?;
        Ok(packet)
    })();
    let value = match &result {
        Ok(packet) => serde_json::to_value(packet)?,
        Err(error) => json!({
            "schema":"private.v41.text-likelihood.prepare-failure.v1", "status":"failed",
            "model_argument":model, "corpus_argument":corpus, "build":build, "error":format!("{error:#}"),
        }),
    };
    file.write_all(&serde_json::to_vec_pretty(&value)?)?;
    file.sync_all()?;
    result.map(|_| ())
}

#[derive(Debug, Serialize)]
struct TokenScore {
    target_index: usize,
    absolute_target_position: usize,
    target_id: u32,
    target_logit: f32,
    nll: f64,
    target_rank: usize,
    top1_id: u32,
    top1_correct: bool,
}

fn token_score(logits: &[f32], vocab: usize, target: u32, index: usize) -> Result<TokenScore> {
    ensure!(
        vocab > 0 && logits.len() == vocab,
        "truncated/invalid logit row length"
    );
    ensure!((target as usize) < vocab, "target outside vocabulary");
    ensure!(logits.iter().all(|v| v.is_finite()), "nonfinite logits");
    let mut top1 = 0;
    for i in 1..vocab {
        if logits[i] > logits[top1] {
            top1 = i;
        }
    }
    let max = f64::from(logits[top1]);
    let target_logit = logits[target as usize];
    let sum: f64 = logits.iter().map(|&v| (f64::from(v) - max).exp()).sum();
    // Subtract before adding ln(sum): even f64 loses ln(2) at f32::MAX.
    let nll = (max - f64::from(target_logit)) + sum.ln();
    ensure!(nll.is_finite() && nll >= 0.0, "invalid NLL");
    let target_rank = 1 + logits
        .iter()
        .enumerate()
        .filter(|&(id, &v)| v > target_logit || (v == target_logit && id < target as usize))
        .count();
    Ok(TokenScore {
        target_index: index,
        absolute_target_position: PREFIX + index,
        target_id: target,
        target_logit,
        nll,
        target_rank,
        top1_id: top1 as u32,
        top1_correct: top1 == target as usize,
    })
}

fn teacher_force(
    targets: &[u32],
    mut logits: Vec<f32>,
    mut step: impl FnMut(u32) -> Result<Vec<f32>>,
    mut observe: impl FnMut(usize, u32, &[f32]) -> Result<()>,
) -> Result<()> {
    for (index, &target) in targets.iter().enumerate() {
        observe(index, target, &logits)?;
        if index + 1 < targets.len() {
            logits = step(target)?;
        }
    }
    Ok(())
}

#[derive(Serialize)]
struct Means {
    nll: f64,
    target_rank: f64,
    top1_accuracy: f64,
}

fn means(scores: &[TokenScore]) -> Means {
    let count = scores.len() as f64;
    Means {
        nll: scores.iter().map(|s| s.nll).sum::<f64>() / count,
        target_rank: scores.iter().map(|s| s.target_rank as f64).sum::<f64>() / count,
        top1_accuracy: scores.iter().filter(|s| s.top1_correct).count() as f64 / count,
    }
}

fn score_document(
    forward: &Forward<'_>,
    doc: &FrozenDocument,
    output: &Path,
    index: usize,
) -> Result<Means> {
    let mut raw = BufWriter::new(create_new(
        &output.join(format!("{index:03}.logits.f32le")),
    )?);
    let mut tokens = BufWriter::new(create_new(
        &output.join(format!("{index:03}.tokens.jsonl")),
    )?);
    let mut state = forward.new_state_with_gpu_step(PREFIX + TARGETS)?;
    ensure!(
        state.capacity() == PREFIX + TARGETS && state.position() == 0,
        "invalid initial state"
    );
    let cancel = Some(qwen_cli::shutdown::cancellation_flag());
    let logits = forward.prefill(&mut state, &doc.prefix_ids, cancel)?;
    ensure!(state.position() == PREFIX, "prefill position mismatch");
    let mut scores = Vec::with_capacity(TARGETS);
    teacher_force(
        &doc.target_ids,
        logits,
        |token| {
            qwen_cli::shutdown::checkpoint()?;
            let position = state.position();
            let logits = forward
                .step(&mut state, token, cancel)
                .map_err(anyhow::Error::msg)?;
            ensure!(state.position() == position + 1, "step position mismatch");
            Ok(logits)
        },
        |i, target, logits| {
            // Retain returned bytes even if validation rejects this row.
            for v in logits {
                raw.write_all(&v.to_le_bytes())?;
            }
            raw.flush()?;
            let score = token_score(logits, VOCAB, target, i)?;
            serde_json::to_writer(&mut tokens, &score)?;
            tokens.write_all(b"\n")?;
            tokens.flush()?;
            scores.push(score);
            Ok(())
        },
    )?;
    ensure!(
        state.position() == PREFIX + TARGETS - 1,
        "final position mismatch"
    );
    raw.get_ref().sync_all()?;
    tokens.get_ref().sync_all()?;
    ensure!(
        raw.get_ref().metadata()?.len() == (TARGETS * VOCAB * 4) as u64 && scores.len() == TARGETS,
        "truncated logit file or incomplete scores"
    );
    qwen_cli::shutdown::checkpoint()?;
    Ok(means(&scores))
}

fn require_gpu_run(build: &Value) -> Result<()> {
    ensure!(
        env!("QWEN_BUILD_DIRTY") == "0"
            && env!("QWEN_BUILD_STAMP_ERROR") == "none"
            && env!("QWEN_BUILD_COMMIT").len() >= 40
            && build["runtime_commit"] == env!("QWEN_BUILD_COMMIT")
            && build["runtime_status"] == ""
            && !cfg!(debug_assertions)
            && qwen_llm::enabled_diagnostic_features().is_empty(),
        "score requires a matching clean committed release build without diagnostics; {RECIPE}"
    );
    for name in ["MTL_DEBUG_LAYER", "QWEN_METAL_LEASE_WAIT"] {
        ensure!(
            std::env::var(name).ok().as_deref() == Some("1"),
            "score requires {name}=1"
        );
    }
    ensure!(
        PREFILL_CHUNK_ROWS == 256,
        "requires production prefill chunk 256"
    );
    qwen_cli::shutdown::checkpoint()?;
    Ok(())
}

fn score_inner(model: &Path, packet_path: &Path, output: &Path, build: &Value) -> Result<Value> {
    qwen_cli::shutdown::install()?;
    let bytes = fs::read(packet_path)?;
    write_new(&output.join("packet.json"), &bytes)?;
    let packet: Packet = serde_json::from_slice(&bytes)?;
    validate_packet(&packet)?;
    require_gpu_run(build)?;
    let gguf = Arc::new(GgufFile::open(model.canonicalize()?)?);
    let info = model_info(&gguf)?;
    write_json(&output.join("model.json"), &info)?;
    ensure!(
        info == packet.model,
        "model paths/sizes differ from packet (not a hash check)"
    );
    let tokenizer = NativeTokenizer::from_gguf(&gguf)?;
    ensure!(
        tokenizer_info(&gguf, &tokenizer)? == packet.tokenizer,
        "tokenizer metadata differs from packet"
    );
    drop(tokenizer); // Score frozen IDs; never re-tokenize text.
    let plan = ResidentWeights::plan(&gguf)?;
    ensure!(plan.config.vocab == VOCAB, "model vocabulary mismatch");
    write_json(
        &output.join("geometry.json"),
        &json!({
            "config":format!("{:?}", plan.config), "layers":plan.layers,
            "dtype_inventory_bytes":plan.dtype_inventory,
        }),
    )?;
    drop(plan);
    qwen_cli::shutdown::checkpoint()?;
    let ctx = MetalContext::new()?; // Normal production process lease; no private capture.
    write_json(
        &output.join("device.json"),
        &json!({"description":ctx.describe()}),
    )?;
    let weights = ResidentWeights::load(&ctx, gguf)?;
    let forward =
        Forward::from_resident(&ctx, &weights, weights.token_map.clone(), weights.raw_pad)
            .map_err(anyhow::Error::msg)?;
    let mut docs = Vec::with_capacity(DOCUMENTS);
    let mut doc_means = Vec::with_capacity(DOCUMENTS);
    for (index, doc) in packet.documents.iter().enumerate() {
        qwen_cli::shutdown::checkpoint()?;
        let mean = score_document(&forward, doc, output, index)
            .with_context(|| format!("document {index}: {}", doc.source.name))?;
        let summary = json!({"index":index, "name":doc.source.name, "stratum":doc.source.stratum,
            "targets":TARGETS, "mean":mean});
        write_json(&output.join(format!("{index:03}.summary.json")), &summary)?;
        docs.push(summary);
        doc_means.push(mean);
    }
    qwen_cli::shutdown::checkpoint()?;
    Ok(json!({
        "status":"complete", "documents":docs,
        "corpus":{"documents":DOCUMENTS, "targets":DOCUMENTS * TARGETS,
            "mean_document_nll":doc_means.iter().map(|m| m.nll).sum::<f64>() / DOCUMENTS as f64,
            "mean_document_target_rank":doc_means.iter().map(|m| m.target_rank).sum::<f64>() / DOCUMENTS as f64,
            "mean_document_top1_accuracy":doc_means.iter().map(|m| m.top1_accuracy).sum::<f64>() / DOCUMENTS as f64},
        "interpretation":"descriptive document means, not independent token trials; no task acceptance threshold or baseline-exactness criterion",
    }))
}

fn score(model: &Path, packet: &Path, output: &Path) -> Result<()> {
    fs::create_dir(output).context("output must be fresh with an existing parent")?;
    let build = build_identity();
    write_json(
        &output.join("manifest.json"),
        &json!({
            "schema":"private.v41.text-likelihood.score.v1", "status":"started", "build":build,
            "model_argument":model, "packet_argument":packet,
            "argv":std::env::args_os().map(|s| s.to_string_lossy().into_owned()).collect::<Vec<_>>(),
            "environment":{"MTL_DEBUG_LAYER":std::env::var("MTL_DEBUG_LAYER").ok(),
                "QWEN_METAL_LEASE_WAIT":std::env::var("QWEN_METAL_LEASE_WAIT").ok()},
            "prefix":PREFIX, "targets":TARGETS, "capacity":PREFIX + TARGETS,
            "prefill_chunk_rows":PREFILL_CHUNK_ROWS, "continuation_steps":TARGETS - 1,
            "logits":{"file":"NNN.logits.f32le", "shape":[TARGETS, VOCAB],
                "layout":"target index, raw vocabulary ID", "dtype":"F32", "byte_order":"little", "header":false,
                "expected_bytes_per_document":TARGETS * VOCAB * 4, "temperature":1},
            "token_scores":"NNN.tokens.jsonl; natural-log NLL in F64; rank is 1-based, ties ordered by lower token ID; top1 uses the same tie order",
            "limitations":["no task acceptance threshold", "no route capture or forced routes",
                "model paths/sizes are not authentication", "partial files are not completed evidence",
                "raw completion, not chat/task performance", "paired TV/KL computation is deferred"],
        }),
    )?;
    let result = score_inner(model, packet, output, &build);
    let report = match &result {
        Ok(report) => report.clone(),
        Err(error) => json!({"status":"failed", "error":format!("{error:#}")}),
    };
    write_json(&output.join("result.json"), &report)?;
    result.map(|_| ())
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        [mode, model, corpus, out] if mode == "prepare" => {
            prepare(Path::new(model), Path::new(corpus), Path::new(out))
        }
        [mode, model, packet, out] if mode == "score" => {
            score(Path::new(model), Path::new(packet), Path::new(out))
        }
        _ => anyhow::bail!(
            "PRIVATE: prepare MODEL CORPUS.json FRESH_PACKET.json | score MODEL PACKET.json FRESH_OUTPUTDIR\n{RECIPE}"
        ),
    }
}

fn main() -> ExitCode {
    qwen_cli::shutdown::finish(run())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(index: usize) -> Document {
        Document {
            name: format!("doc-{index}"),
            stratum: if index < 3 { "prose" } else { "code" }.into(),
            source_commit: "a".repeat(40),
            path: format!("retained/{index}"),
            text: "raw source\n".into(),
        }
    }

    fn packet() -> Packet {
        Packet {
            schema: SCHEMA.into(),
            policy: POLICY.into(),
            prefix_length: PREFIX,
            target_length: TARGETS,
            vocab: VOCAB,
            model: json!({}),
            tokenizer: json!({}),
            prepare_build: json!({}),
            documents: (0..DOCUMENTS)
                .map(|i| freeze(source(i), (0..600).collect()).unwrap())
                .collect(),
        }
    }

    #[test]
    fn stable_nll_extreme_finite_values_and_ties() -> Result<()> {
        for value in [0.0, f32::MAX, -f32::MAX] {
            let score = token_score(&[value, value], 2, 1, 0)?;
            assert!((score.nll - 2.0_f64.ln()).abs() < 1e-14);
            assert_eq!(
                (score.top1_id, score.target_rank, score.top1_correct),
                (0, 2, false)
            );
        }
        let score = token_score(&[f32::MAX, -f32::MAX], 2, 1, 0)?;
        assert_eq!(score.nll, 2.0 * f64::from(f32::MAX));
        assert_eq!(token_score(&[1000.0, -1000.0], 2, 0, 0)?.nll, 0.0);
        let score = token_score(&[0.0, 1.0, 2.0], 3, 1, 0)?;
        assert!(
            (score.nll - (1.0 + ((-2.0_f64).exp() + (-1.0_f64).exp() + 1.0).ln())).abs() < 1e-14
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_logits_and_targets() {
        for values in [
            vec![],
            vec![0.0],
            vec![0.0, f32::NAN],
            vec![f32::INFINITY, 0.0],
            vec![0.0, f32::NEG_INFINITY],
        ] {
            assert!(token_score(&values, 2, 0, 0).is_err());
        }
        assert!(token_score(&[0.0, 1.0], 2, 2, 0).is_err());
        assert!(token_score(&[], 0, 0, 0).is_err());
    }

    #[test]
    fn whole_text_freeze_rejects_short_or_invalid_ids() -> Result<()> {
        let doc = freeze(source(0), (0..600).collect())?;
        assert_eq!(doc.prefix_ids, (0..512).collect::<Vec<_>>());
        assert_eq!(doc.target_ids, (512..576).collect::<Vec<_>>());
        assert_eq!(doc.whole_text_tokens, 600);
        assert!(freeze(source(0), vec![0; 575]).is_err());
        for id in [-1, VOCAB as i32] {
            let mut ids = vec![0; 577];
            ids[576] = id; // Validate even the declared unused suffix.
            assert!(freeze(source(0), ids).is_err());
        }
        Ok(())
    }

    #[test]
    fn fresh_outputs_retain_failures_without_opening_a_model() -> Result<()> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "v41-text-likelihood-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let corpus = root.join("corpus.json");
        let packet = root.join("packet.json");
        let model = root.join("nonexistent-model.gguf");
        let output = root.join("score");
        write_new(&corpus, br#"{"documents":[]}"#)?;
        let error = prepare(&model, &corpus, &packet).unwrap_err();
        assert!(error.to_string().contains("exactly 3 prose + 3 code"));
        let failure = fs::read(&packet)?;
        assert_eq!(
            serde_json::from_slice::<Value>(&failure)?["status"],
            "failed"
        );
        assert!(serde_json::from_slice::<Packet>(&failure).is_err());
        assert!(prepare(&model, &corpus, &packet).is_err());
        assert_eq!(fs::read(&packet)?, failure);
        assert!(score(&model, &packet, &output).is_err());
        assert_eq!(fs::read(output.join("packet.json"))?, failure);
        let report = fs::read(output.join("result.json"))?;
        assert_eq!(
            serde_json::from_slice::<Value>(&report)?["status"],
            "failed"
        );
        assert!(output.join("manifest.json").exists());
        assert!(!output.join("device.json").exists());
        assert!(score(&model, &packet, &output).is_err());
        assert_eq!(fs::read(output.join("result.json"))?, report);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn packet_roundtrip_rejects_lengths_bounds_policy_and_truncation() -> Result<()> {
        let valid = packet();
        let bytes = serde_json::to_vec(&valid)?;
        validate_packet(&serde_json::from_slice::<Packet>(&bytes)?)?;
        assert!(serde_json::from_slice::<Packet>(&bytes[..bytes.len() - 1]).is_err());
        for mutate in [
            |p: &mut Packet| {
                p.documents[0].prefix_ids.pop();
            },
            |p: &mut Packet| {
                p.documents[0].target_ids.push(0);
            },
            |p: &mut Packet| {
                p.documents[0].whole_text_tokens = 575;
            },
            |p: &mut Packet| {
                p.documents[0].prefix_ids[0] = VOCAB as u32;
            },
            |p: &mut Packet| {
                p.documents[0].target_ids[63] = VOCAB as u32;
            },
            |p: &mut Packet| {
                p.prefix_length = 511;
            },
            |p: &mut Packet| {
                p.target_length = 65;
            },
            |p: &mut Packet| {
                p.vocab += 1;
            },
            |p: &mut Packet| {
                p.policy.clear();
            },
            |p: &mut Packet| {
                p.documents.pop();
            },
            |p: &mut Packet| {
                p.documents[0].source.stratum = "code".into();
            },
            |p: &mut Packet| {
                p.documents[0].source.name = "doc-1".into();
            },
            |p: &mut Packet| {
                p.documents[0].source.path = "retained/1".into();
            },
        ] {
            let mut invalid = valid.clone();
            mutate(&mut invalid);
            assert!(validate_packet(&invalid).is_err());
        }
        Ok(())
    }

    #[test]
    fn prefill_scores_target_zero_then_exactly_63_steps() -> Result<()> {
        let targets: Vec<u32> = (1..=TARGETS as u32).collect();
        let mut consumed = Vec::new();
        let mut scores = Vec::new();
        let row = |id: usize| {
            let mut logits = vec![0.0; 65];
            logits[id] = 10.0;
            logits
        };
        teacher_force(
            &targets,
            row(1),
            |token| {
                consumed.push(token);
                Ok(row(token as usize + 1))
            },
            |i, target, logits| {
                scores.push(token_score(logits, 65, target, i)?);
                Ok(())
            },
        )?;
        assert_eq!(consumed, targets[..63]);
        assert_eq!(scores.len(), TARGETS);
        for (i, score) in scores.iter().enumerate() {
            assert_eq!(score.target_index, i);
            assert_eq!(score.absolute_target_position, PREFIX + i);
            assert!(score.top1_correct);
            assert_eq!(score.target_rank, 1);
        }
        assert_eq!(means(&scores).top1_accuracy, 1.0);
        Ok(())
    }
}
