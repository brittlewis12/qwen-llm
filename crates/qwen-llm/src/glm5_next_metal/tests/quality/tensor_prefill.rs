//! Same-binary Fast/Half2048 screen; all wall times are diagnostic with validation on.
use super::super::super::tensor_prefill::Scope;
use super::*;
use crate::glm5_next::admission::Glm5NextArtifactLayout;
use crate::metal::{
    DispatchCensusRow, diagnostics_observer_active_counts, dispatch_census_begin,
    dispatch_census_is_active, dispatch_census_take,
};
use crate::tokenizer::NativeTokenizer;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    time::Instant,
};

const FROZEN: &[u8] = include_bytes!("../../../../../../scripts/reference/glm53/quality-v1.json");
const ROWS: usize = 2048;
const CENSUS_BYTES: u64 = 128 << 20;
const REPORT_BYTES: u64 = 8 << 20;
const TOKENIZER_BYTES: u64 = 128 << 20;
const ARMS: [(&str, bool); 2] = [("old", false), ("tensor", true)];
const LONG_TASKS: [&str; 4] = ["T09", "T10", "T11", "T12"];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Packet {
    Quality,
    Plain,
    Extended,
}

impl Packet {
    fn report_env(self) -> &'static str {
        match self {
            Self::Quality => "GLM53_TENSOR_QUALITY_REPORT",
            Self::Plain => "GLM53_TENSOR_TIMING_REPORT",
            Self::Extended => "GLM53_TENSOR_EXTENDED_REPORT",
        }
    }

    fn schema(self) -> &'static str {
        match self {
            Self::Quality => "glm53.tensor_quality.v1",
            Self::Plain => "glm53.tensor_plain_abba.v1",
            Self::Extended => "glm53.tensor_extended_quality.v1",
        }
    }
}

struct Document {
    path: String,
    stratum: String,
    prefix: usize,
    tokens: Vec<u32>,
}

fn documents() -> Vec<Document> {
    let fixture: Value = serde_json::from_slice(FROZEN).unwrap();
    assert_eq!(fixture["name"], "glm53-quality-v1");
    assert_eq!(fixture["continuation"], CONTINUATION);
    [
        ("prose", "docs/GLM53-FLASH-PLAN.md", 2048),
        ("code", "crates/qwen-llm/src/sampling.rs", 2048),
        ("long", "docs/SERVE.md", 4352),
    ]
    .map(|(stratum, path, prefix)| {
        let d = fixture["documents"]
            .as_array()
            .unwrap()
            .iter()
            .find(|d| d["stratum"] == stratum)
            .unwrap();
        assert_eq!(d["path"], path, "frozen first-in-stratum selection");
        assert!(
            d["prefix_lengths"]
                .as_array()
                .unwrap()
                .contains(&json!(prefix))
        );
        let mut tokens = parse_ids(&d["tokens"]);
        assert!(tokens.len() >= prefix + CONTINUATION);
        tokens.truncate(prefix + CONTINUATION);
        Document {
            path: path.into(),
            stratum: stratum.into(),
            prefix,
            tokens,
        }
    })
    .into()
}

fn all_documents() -> Vec<Document> {
    let fixture: Value = serde_json::from_slice(FROZEN).unwrap();
    assert_eq!(fixture["name"], "glm53-quality-v1");
    assert_eq!(fixture["continuation"], CONTINUATION);
    let frozen = fixture["documents"].as_array().unwrap();
    assert_eq!(frozen.len(), 22);
    let mut strata = BTreeMap::<String, usize>::new();
    let mut paths = std::collections::BTreeSet::new();
    let documents = frozen
        .iter()
        .map(|d| {
            let stratum = d["stratum"].as_str().unwrap();
            let lengths: Vec<usize> = d["prefix_lengths"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| usize::try_from(n.as_u64().unwrap()).unwrap())
                .collect();
            assert_eq!(
                lengths,
                match stratum {
                    "prose" | "code" => vec![512, 2048],
                    "long" => vec![4352],
                    _ => panic!("unexpected frozen stratum {stratum}"),
                }
            );
            let prefix = *lengths.iter().max().unwrap();
            let path = d["path"].as_str().unwrap().to_owned();
            assert!(paths.insert(path.clone()), "duplicate frozen document");
            *strata.entry(stratum.into()).or_default() += 1;
            let mut tokens = parse_ids(&d["tokens"]);
            assert!(tokens.len() >= prefix + CONTINUATION);
            tokens.truncate(prefix + CONTINUATION);
            Document {
                path,
                stratum: stratum.into(),
                prefix,
                tokens,
            }
        })
        .collect();
    assert_eq!(
        strata,
        BTreeMap::from([("prose".into(), 8), ("code".into(), 8), ("long".into(), 6)])
    );
    documents
}

struct ToolTask {
    id: String,
    context_level: u64,
    prompt: Vec<u32>,
    expected: Value,
    expect: Expect,
}

impl ToolTask {
    fn capacity(&self) -> usize {
        // The 1455-token tasks still need the same 2048-row workspace.
        ROWS.max(self.prompt.len().checked_add(TOOL_CAP).unwrap())
    }
}

fn long_tasks() -> (Vec<ToolTask>, Vec<ToolDefinition>) {
    let fixture: Value = serde_json::from_slice(FROZEN).unwrap();
    assert_eq!(fixture["name"], "glm53-quality-v1");
    assert_eq!(fixture["tool_cap"], TOOL_CAP);
    assert_eq!(fixture["tool_seeds"][0], 11);
    let (tools, definitions) = tool_definitions();
    assert_eq!(fixture["tools"], json!(tools), "frozen tool definitions");
    let frozen_definitions: Vec<_> = fixture["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| ToolDefinition::from_value(v).unwrap())
        .collect();
    let tasks: Vec<_> = fixture["tool_tasks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| t["context_level"].as_u64().unwrap() > 0)
        .map(|t| {
            let id = t["id"].as_str().unwrap();
            let (_, _, context, expect) = TOOL_TASKS.iter().find(|t| t.0 == id).unwrap();
            let context_level = t["context_level"].as_u64().unwrap();
            assert_eq!(context_level, u64::from(*context));
            assert_eq!(t["effort"], "low");
            assert_eq!(t["expected"], expect.describe());
            let prompt = parse_ids(&t["prompt"]);
            assert_eq!(prompt.len() as u64, t["prompt_tokens"].as_u64().unwrap());
            assert!(
                prompt.len() >= 1024,
                "tool prefill must exercise eligible tensor context"
            );
            let messages = [Message::User(t["user"].as_str().unwrap().into())];
            assert_eq!(
                render(&messages, &definitions, Effort::Low),
                render(&messages, &frozen_definitions, Effort::Low),
                "frozen rendered tools"
            );
            ToolTask {
                id: id.into(),
                context_level,
                prompt,
                expected: t["expected"].clone(),
                expect: *expect,
            }
        })
        .collect();
    assert_eq!(
        tasks.iter().map(|t| t.id.as_str()).collect::<Vec<_>>(),
        LONG_TASKS
    );
    (tasks, frozen_definitions)
}

fn token_hash(tokens: &[u32]) -> String {
    let mut hash = Sha256::new();
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn environment() {
    assert!(!cfg!(debug_assertions), "use a release test binary");
    for (key, value) in [
        ("MTL_DEBUG_LAYER", "1"),
        ("QWEN_METAL_LEASE_WAIT", "1"),
        ("QWEN_MATMAT_Q6_K_N64_MIN_N", "1024"),
    ] {
        assert_eq!(
            std::env::var(key).as_deref(),
            Ok(value),
            "requires {key}={value}"
        );
    }
    assert_eq!(crate::metal::mat_mat_q6_k_n64_min_query(), 1024);
    assert!(
        crate::env_flag::read_default_on("QWEN_MATMAT_Q6_K_N64"),
        "production Q6_K N64 route must be enabled"
    );
    assert_ne!(std::env::var("QWEN_EXPERT_TENSOR_TILE").as_deref(), Ok("0"));
    assert!(
        std::env::var_os("GLM53_GGUF").is_some(),
        "GLM53_GGUF must name shard 1 via its standard shard path, not a canonicalized backing blob"
    );
    assert_plain();
}

fn assert_plain() {
    assert!(!dispatch_census_is_active());
    assert_eq!(
        diagnostics_observer_active_counts(),
        [0; 3],
        "no trace, census or allocation observer may be active"
    );
}

struct Report(Option<File>);

impl Report {
    fn new(env: &str, required: bool) -> Self {
        let path = std::env::var_os(env);
        assert!(
            !required || path.is_some(),
            "requires new private {env} JSONL path"
        );
        Self(path.map(|path| {
            File::options()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(path)
                .unwrap()
        }))
    }

    fn emit(&mut self, value: Value) {
        if let Some(out) = &mut self.0 {
            serde_json::to_writer(&mut *out, &value).unwrap();
            writeln!(out).unwrap();
            out.flush().unwrap();
        } else {
            eprintln!("{value}");
        }
    }

    fn sync(&self) {
        if let Some(out) = &self.0 {
            out.sync_all().unwrap();
        }
    }
}

fn binary_hash() -> String {
    let mut file = File::open(std::env::current_exe().unwrap()).unwrap();
    let mut hash = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    format!("{:x}", hash.finalize())
}

fn cpu_terms(vocab: u32) -> [(&'static str, u64); 6] {
    let v = u64::from(vocab);
    [
        (
            "one_retained_64_row_baseline_f32",
            CONTINUATION as u64 * v * 4,
        ),
        ("two_live_logits_forward_or_clone_peak", 2 * v * 4),
        ("two_kl_log_softmax_f64_vectors", 2 * v * 8),
        (
            "baseline_row_headers",
            CONTINUATION as u64 * size_of::<Vec<f32>>() as u64,
        ),
        ("dispatch_census_rows_and_strings", CENSUS_BYTES),
        ("report_hashes_metrics_and_census_aggregation", REPORT_BYTES),
    ]
}

fn fresh<'w>(
    ctx: &MetalContext,
    weights: &'w Glm5NextWeights,
    capacity: usize,
    cpu: u64,
    arm: &str,
    phase: &str,
    out: &mut Report,
) -> Glm5NextSession<'w> {
    let mut s =
        Glm5NextSession::with_prefill_rows_and_cpu_reserve(ctx, weights, capacity, ROWS, cpu)
            .unwrap();
    s.set_packed_lineage(PackedLineage::Fast).unwrap();
    s.set_fast_precision(FastPrecision::Half).unwrap();
    check_session(&s, 0);
    out.emit(
        json!({"event":"session", "arm":arm, "phase":phase, "capacity":capacity,
        "actual_prefill_rows":s.ledger().prefill_rows(), "cpu_reserve_bytes":cpu,
        "ledger":s.ledger().terms(), "observed_allocation_delta":s.observed_allocation_delta(),
        "lineage":"Fast", "precision":"Half"}),
    );
    s
}

fn check_session(s: &Glm5NextSession<'_>, position: usize) {
    assert!(!s.is_poisoned());
    assert_eq!(s.position(), position);
    assert_eq!(s.ledger().prefill_rows(), ROWS as u64);
    assert_eq!(s.packed_lineage(), Some(PackedLineage::Fast));
    assert_eq!(s.fast_precision(), Some(FastPrecision::Half));
}

fn check_logits(logits: &[f32], vocab: usize) {
    assert_eq!(logits.len(), vocab);
    assert!(vocab > 0);
    assert_finite("tensor prefill full-vocab logits", logits);
}

fn chunks(n: usize) -> Vec<usize> {
    (0..n).step_by(ROWS).map(|i| (n - i).min(ROWS)).collect()
}

fn observed_prefill(
    ctx: &MetalContext,
    s: &mut Glm5NextSession<'_>,
    tokens: &[u32],
    enabled: bool,
) -> (Vec<f32>, Vec<usize>) {
    check_session(s, 0);
    let _scope = Scope::new(enabled);
    let mut commits = Vec::with_capacity(tokens.len().div_ceil(ROWS));
    let logits = s
        .prefill_packed_with_observer(ctx, tokens, &mut || Ok(()), &mut |s, count| {
            commits.push(count);
            check_session(s, commits.iter().sum());
        })
        .unwrap();
    assert_eq!(commits, chunks(tokens.len()));
    check_session(s, tokens.len());
    (logits, commits)
}

struct Census;

impl Census {
    fn begin() -> Self {
        assert_plain();
        dispatch_census_begin();
        Self
    }

    fn finish(self) -> Vec<DispatchCensusRow> {
        dispatch_census_take()
    }
}

impl Drop for Census {
    fn drop(&mut self) {
        drop(dispatch_census_take());
    }
}

fn warm(ctx: &MetalContext, weights: &Glm5NextWeights, tokens: &[u32], cpu: u64, out: &mut Report) {
    for (arm, enabled) in ARMS {
        let mut s = fresh(ctx, weights, tokens.len(), cpu, arm, "warm", out);
        let census = Census::begin();
        let (logits, commits) = observed_prefill(ctx, &mut s, tokens, enabled);
        let census = census.finish();
        assert_plain();
        check_logits(&logits, weights.config.vocab_size as usize);
        let heap_bytes = census.capacity() * size_of::<DispatchCensusRow>()
            + census
                .iter()
                .map(|r| r.kernel.capacity() + r.tag.as_ref().map_or(0, String::capacity))
                .sum::<usize>();
        let census_upper_bytes = (heap_bytes as u64).checked_mul(2).unwrap();
        assert!(
            census_upper_bytes <= CENSUS_BYTES,
            "increase the admitted census bound before using this artifact"
        );
        let mut kernels = BTreeMap::<&str, usize>::new();
        for row in &census {
            *kernels.entry(&row.kernel).or_default() += 1;
        }
        assert!(kernels.len() <= 1024);
        let dense_tensor = |r: &&DispatchCensusRow| {
            r.kernel.starts_with("kernel_mat_mat_") && r.kernel.ends_with("_f32_tensor")
        };
        let dense_calls = census.iter().filter(dense_tensor).count();
        let counts = [GgmlType::Q6_K, GgmlType::Q8_0].map(|dtype| {
            let name = crate::metal::mat_mat_tensor_pipeline_name(dtype).unwrap();
            let count = census.iter().filter(|r| r.kernel == name).count();
            if enabled {
                assert!(
                    count >= tokens.len().div_ceil(ROWS),
                    "missing dense {dtype:?} tensor dispatches"
                );
                assert!(
                    census
                        .iter()
                        .filter(|r| r.kernel == name)
                        .all(|r| r.grid_width == (ROWS / 32) as u64
                            && r.grid_depth == 1
                            && [r.threads_width, r.threads_height, r.threads_depth] == [128, 1, 1])
                );
            } else {
                assert_eq!(count, 0);
            }
            (name, count)
        });
        if enabled {
            assert_eq!(dense_calls, counts.iter().map(|(_, n)| n).sum::<usize>());
        } else {
            assert_eq!(
                dense_calls, 0,
                "old arm must not dispatch any dense tensor kernel"
            );
            assert!(
                kernels
                    .get("kernel_mat_mat_q6_K_f32_n64")
                    .copied()
                    .unwrap_or(0)
                    > 0,
                "old arm must use the production Q6_K N64 route"
            );
            assert!(
                kernels
                    .keys()
                    .any(|k| k.starts_with("kernel_mat_mat_q8_0_")),
                "old Q8_0 route missing"
            );
        }
        out.emit(json!({"event":"warm_census", "arm":arm, "prompt_tokens":tokens.len(),
            "prefill_chunks":commits, "dispatches":census.len(), "dense_tensor_calls":dense_calls,
            "dense_tensor_kernel_counts":counts, "kernel_counts":kernels, "census_heap_bytes":heap_bytes,
            "census_upper_including_vec_growth_bytes":census_upper_bytes,
            "census_reserve_bytes":CENSUS_BYTES, "timed":false,
            "scope":"kernel selection/geometry only; no routing-data equality claim"}));
    }
}

fn quality(
    ctx: &MetalContext,
    weights: &Glm5NextWeights,
    documents: &[Document],
    cpu: u64,
    out: &mut Report,
) {
    let long = documents.iter().find(|d| d.stratum == "long").unwrap();
    warm(ctx, weights, &long.tokens[..ROWS], cpu, out);
    let vocab = weights.config.vocab_size as usize;
    for d in documents {
        let n = d.prefix;
        let mut baseline = Vec::<Vec<f32>>::with_capacity(CONTINUATION);
        for (arm, enabled) in ARMS {
            assert_eq!(baseline.len(), if enabled { CONTINUATION } else { 0 });
            out.emit(json!({"event":"document_begin", "arm":arm, "path":d.path,
                "stratum":d.stratum, "prefix":n, "continuation":CONTINUATION}));
            let mut s = fresh(ctx, weights, n + CONTINUATION, cpu, arm, "quality", out);
            assert_plain();
            let started = Instant::now();
            let (mut logits, commits) = observed_prefill(ctx, &mut s, &d.tokens[..n], enabled);
            let prefill_ms = started.elapsed().as_secs_f64() * 1e3;
            let (mut nll_sum, mut kl_sum, mut kl_max) = (0.0, 0.0, 0.0f64);
            let (mut hits, mut agreements) = (0usize, 0usize);
            let mut decode_ms = 0.0;
            for (i, &token) in d.tokens[n..n + CONTINUATION].iter().enumerate() {
                check_session(&s, n + i);
                check_logits(&logits, vocab);
                let nll = -log_prob(&logits, token);
                assert!(nll.is_finite());
                let prediction = argmax(&logits);
                if !enabled {
                    baseline.push(logits.clone());
                }
                let reference = &baseline[i];
                let reference_argmax = argmax(reference);
                let kl = if enabled {
                    kl_divergence(reference, &logits)
                } else {
                    0.0
                };
                assert!(kl.is_finite());
                nll_sum += nll;
                kl_sum += kl;
                kl_max = kl_max.max(kl);
                hits += usize::from(prediction == token as usize);
                agreements += usize::from(prediction == reference_argmax);
                out.emit(json!({"event":"document_token", "arm":arm, "path":d.path,
                    "stratum":d.stratum, "prefix":n, "index":i, "target_position":n+i,
                    "target_token":token, "nll":nll, "top1_hit":prediction == token as usize,
                    "argmax":prediction, "baseline_argmax":reference_argmax,
                    "top1_agrees_baseline":prediction == reference_argmax,
                    "kl_baseline_to_arm":kl, "reference_self":!enabled, "logits_finite":true}));
                if i + 1 < CONTINUATION {
                    let started = Instant::now();
                    logits = s.forward(ctx, token).unwrap();
                    decode_ms += started.elapsed().as_secs_f64() * 1e3;
                }
            }
            check_session(&s, n + CONTINUATION - 1);
            assert!(nll_sum.is_finite() && kl_sum.is_finite());
            out.emit(json!({"event":"document_arm", "arm":arm, "path":d.path,
                "stratum":d.stratum, "prefix":n, "capacity":n+CONTINUATION,
                "continuation_tokens":&d.tokens[n..n+CONTINUATION], "logit_rows":CONTINUATION,
                "vocab":vocab, "mean_nll":nll_sum / CONTINUATION as f64, "top1_hits":hits,
                "top1_baseline_agreements":agreements, "mean_kl_baseline_to_arm":kl_sum / CONTINUATION as f64,
                "max_kl_baseline_to_arm":kl_max, "reference_self":!enabled,
                "kl_scope":"per-token descriptive KL(baseline || arm); baseline self is zero",
                "actual_prefill_rows":s.ledger().prefill_rows(), "prefill_chunks":commits,
                "final_position":s.position(), "decode_calls":CONTINUATION-1,
                "lineage":"Fast", "precision":"Half", "prefill_ms":prefill_ms, "decode_ms":decode_ms,
                "timing_scope":"diagnostic only; validation on; allocation excluded; no census or profiler"}));
        }
    }
}

fn tool_quality(
    ctx: &MetalContext,
    weights: &Glm5NextWeights,
    tokenizer: &NativeTokenizer,
    tasks: &[ToolTask],
    definitions: &[ToolDefinition],
    cpu: u64,
    out: &mut Report,
) {
    let vocab = weights.config.vocab_size as usize;
    for task in tasks {
        let n = task.prompt.len();
        let prompt_hash = token_hash(&task.prompt);
        for generation in [
            Generation::Greedy { cap: TOOL_CAP },
            Generation::Sampled {
                seed: 11,
                cap: TOOL_CAP,
            },
        ] {
            for (arm, enabled) in ARMS {
                out.emit(json!({"event":"task_begin","arm":arm,"id":task.id,
                    "context_level":task.context_level,"expected":task.expected,
                    "generation":generation.describe(),"prompt_tokens":n,"prompt_sha256":prompt_hash,
                    "token_hash_encoding":"u32 little-endian","requested_capacity":n+TOOL_CAP,
                    "capacity":task.capacity(),"workspace_rows":ROWS}));
                let mut s = fresh(ctx, weights, task.capacity(), cpu, arm, "tool_quality", out);
                assert_plain();
                let (mut logits, commits) = observed_prefill(ctx, &mut s, &task.prompt, enabled);
                out.emit(json!({"event":"task_prefill","arm":arm,"id":task.id,
                    "generation":generation.describe(),"prefill_chunks":commits,"position":s.position(),
                    "actual_prefill_rows":s.ledger().prefill_rows()}));
                let mut sampler = Sampler::new(generation.sampling()).unwrap();
                let mut emitted = Vec::with_capacity(TOOL_CAP);
                let mut sampled = Vec::with_capacity(TOOL_CAP);
                let mut forward_calls = 0;
                let stop = loop {
                    check_session(&s, n + forward_calls);
                    check_logits(&logits, vocab);
                    assert_plain();
                    let token = u32::try_from(sampler.sample(&logits).unwrap().token).unwrap();
                    assert!((token as usize) < vocab);
                    let is_stop = chat::CHAT_STOPS.contains(&i32::try_from(token).unwrap());
                    sampled.push(token);
                    out.emit(json!({"event":"task_token","arm":arm,"id":task.id,
                        "generation":generation.describe(),"sample_index":sampled.len()-1,
                        "position":s.position(),"token":token,"is_stop":is_stop,"logits_finite":true}));
                    if is_stop {
                        break Some(token);
                    }
                    emitted.push(token);
                    if emitted.len() == TOOL_CAP {
                        break None;
                    }
                    logits = s.forward(ctx, token).unwrap();
                    forward_calls += 1;
                };
                assert!(!sampled.is_empty() && sampled.len() <= TOOL_CAP);
                assert_eq!(
                    forward_calls,
                    sampled.len() - 1,
                    "last sampled token is not forwarded"
                );
                assert_eq!(emitted.len() + usize::from(stop.is_some()), sampled.len());
                check_session(&s, n + forward_calls);
                let final_position = s.position();
                drop(s);
                drop(logits);
                drop(sampler);
                let text = tokenizer
                    .try_decode(
                        &emitted
                            .iter()
                            .map(|&t| i32::try_from(t).unwrap())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let (reasoning, content, calls, parse_error) = match parse_turn(&text, definitions)
                {
                    Ok((reasoning, content, calls)) => {
                        (Some(reasoning), Some(content), calls, None)
                    }
                    Err(error) => (None, None, Vec::new(), Some(error)),
                };
                out.emit(json!({"event":"task_generation","arm":arm,"id":task.id,
                    "context_level":task.context_level,"expected":task.expected,
                    "generation":generation.describe(),"prompt_tokens":n,"prompt_sha256":prompt_hash,
                    "token_hash_encoding":"u32 little-endian","emitted_ids":emitted,
                    "emitted_ids_exclude_stop":true,"emitted_sha256":token_hash(&emitted),
                    "sampled_ids":sampled,"sampled_ids_include_stop":true,"sampled_sha256":token_hash(&sampled),
                    "emitted_tokens":emitted.len(),"sampled_tokens":sampled.len(),
                    "text":text,"text_sha256":sha256_hex(text.as_bytes()),"reasoning":reasoning,"content":content,
                    "stop":stop,"stop_reason":if stop.is_some() { "chat_stop" } else { "cap" },
                    "parse_error":parse_error,"calls":calls.iter().map(|c|json!({"id":c.id,"name":c.name,"arguments":c.arguments})).collect::<Vec<_>>(),
                    "lenient_first_call_match":call_correct(&calls, task.expect),
                    "strict_scorer":"scripts/reference/glm53/quality_analysis.py::tool_correct on recorded calls/stop/parse_error",
                    "final_position":final_position,"decode_calls":forward_calls,"capacity":task.capacity(),
                    "lineage":"Fast","precision":"Half","workspace_rows":ROWS,
                    "policy":"descriptive tool screen only; parse/task outcomes are recorded, not asserted"}));
            }
        }
    }
}

fn plain_abba(
    ctx: &MetalContext,
    weights: &Glm5NextWeights,
    long: &Document,
    cpu: u64,
    out: &mut Report,
) {
    for n in [2048, 4096] {
        let tokens = &long.tokens[..n];
        warm(ctx, weights, tokens, cpu, out);
        let mut walls = Vec::with_capacity(8);
        for (sample, enabled) in [false, true, true, false, true, false, false, true]
            .into_iter()
            .enumerate()
        {
            let arm = if enabled { ARMS[1].0 } else { ARMS[0].0 };
            let mut s = fresh(ctx, weights, n, cpu, arm, "plain", out);
            assert_plain();
            let _scope = Scope::new(enabled);
            let started = Instant::now();
            let logits = s.prefill_packed(ctx, tokens).unwrap();
            let wall_ms = started.elapsed().as_secs_f64() * 1e3;
            assert_plain();
            check_session(&s, n);
            check_logits(&logits, weights.config.vocab_size as usize);
            assert!(wall_ms.is_finite() && wall_ms > 0.0);
            walls.push(wall_ms);
            out.emit(json!({"event":"plain_sample", "arm":arm, "prompt_tokens":n,
                "sample":sample, "block":if sample < 4 { "ABBA" } else { "BAAB" },
                "wall_ms":wall_ms, "final_position":s.position(), "logits_finite":true,
                "actual_prefill_rows":s.ledger().prefill_rows(), "expected_prefill_chunks":chunks(n),
                "scope":"ordinary prefill_packed wall including GPU wait and logits readback; session allocation, checks, JSON excluded; no profiler, trace or census; Metal validation on; diagnostic only"}));
        }
        out.emit(
            json!({"event":"plain_width", "prompt_tokens":n, "workspace_rows":ROWS,
            "order":["old","tensor","tensor","old","tensor","old","old","tensor"],
            "wall_ms":walls, "samples_per_arm":4, "performance_verdict":null}),
        );
    }
}

fn packet(kind: Packet) {
    let quality_screen = kind != Packet::Plain;
    let extended = kind == Packet::Extended;
    let mut out = Report::new(kind.report_env(), quality_screen);
    let documents = if extended {
        all_documents()
    } else {
        documents()
    };
    let (tasks, definitions) = if extended {
        long_tasks()
    } else {
        (Vec::new(), Vec::new())
    };
    // Preserve the shard symlink name: GGUF uses it to find sibling shards.
    let path = PathBuf::from(std::env::var_os("GLM53_GGUF").unwrap());
    let gguf = GgufFile::open(&path).unwrap();
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let artifact = Glm5NextArtifactLayout::inspect(&gguf).unwrap();
    assert!(artifact.packed_prefill());
    let vocab = artifact.config().vocab_size;
    assert!(vocab > 0);
    for d in &documents {
        assert!(d.tokens.iter().all(|&t| t < vocab));
    }
    for task in &tasks {
        assert!(
            task.prompt
                .iter()
                .all(|&t| t < vocab && i32::try_from(t).is_ok())
        );
    }
    if extended {
        assert!(
            chat::CHAT_STOPS
                .iter()
                .all(|&t| t >= 0 && (t as u32) < vocab)
        );
    }
    let capacity = if quality_screen {
        documents
            .iter()
            .map(|d| d.prefix + CONTINUATION)
            .chain(tasks.iter().map(ToolTask::capacity))
            .max()
            .unwrap()
    } else {
        4096
    };
    assert!(capacity <= artifact.config().context_length as usize);
    let mut terms = Vec::from(cpu_terms(vocab));
    if extended {
        terms.extend([
            (
                "native_tokenizer_construction_retained_and_decode_peak",
                TOKENIZER_BYTES,
            ),
            ("sampler_topk_off_peak_vectors", u64::from(vocab) * 24),
            ("sampler_radix_histogram", 65536 * size_of::<usize>() as u64),
            ("tool_ids_text_parse_and_progress_reports", 8 << 20),
        ]);
    }
    let cpu = terms
        .iter()
        .try_fold(0u64, |sum, (_, n)| sum.checked_add(*n))
        .unwrap();
    let ctx = MetalContext::new().unwrap();
    assert!(
        ctx.tensor_ops_status().is_available(),
        "actual tensor tier required: {:?}",
        ctx.tensor_ops_status()
    );
    assert!(
        crate::metal::moe_grouped_tensor_selected(&ctx),
        "QWEN_EXPERT_TENSOR_TILE must not disable the tensor tier"
    );
    let mut shapes = BTreeMap::<String, usize>::new();
    for t in &gguf.tensors {
        *shapes
            .entry(format!("{:?} {:?}", t.dtype, t.shape))
            .or_default() += 1;
    }
    let mut header = json!({"event":"header", "schema":kind.schema(),
        "model":path, "artifact_layout":artifact_layout(&gguf), "tensor_shape_counts":shapes,
        "config":format!("{:?}", artifact.config()), "binary":std::env::current_exe().unwrap(),
        "binary_sha256":binary_hash(), "fixture_sha256":sha256_hex(FROZEN),
        "packet_sha256":sha256_hex(include_bytes!("tensor_prefill.rs")),
        "candidate_source_sha256":sha256_hex(include_bytes!("../../tensor_prefill.rs")),
        "packed_source_sha256":sha256_hex(include_bytes!("../../packed.rs")),
        "shards":stamps.iter().map(|s|json!({"path":s.path,"shard":s.shard_idx,"bytes":s.size,
            "device":s.device,"inode":s.inode,"mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,
            "ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec})).collect::<Vec<_>>(),
        "documents":documents.iter().map(|d|json!({"path":d.path,"stratum":d.stratum,"prefix":d.prefix,
            "tokens_sha256":sha256_hex(bytemuck::cast_slice(&d.tokens))})).collect::<Vec<_>>(),
        "arms":["old","tensor"], "control":"same-thread RAII Scope(false/true), whole prefill; decode unchanged; no capability/precision/bounds bypass",
        "selection":"explicit old/tensor override, independent of production default", "lineage":"Fast", "precision":"Half", "workspace_rows":ROWS,
        "continuation":CONTINUATION, "plain_prompt_lengths":[2048,4096],
        "cpu_reserve_bytes":cpu, "cpu_terms":terms, "mtl_debug_layer":"1", "metal_lease_wait":"1",
        "q6_k_n64_min_n":1024, "tensor_tier":format!("{:?}",ctx.tensor_ops_status()),
        "token_source":"frozen repository quality-v1 token IDs only; no retokenization or tokenizer compatibility claim",
        "policy":"descriptive only; finite/state/alignment/dispatch assertions, no NLL/KL threshold, quality/noninferiority pass, promotion or performance verdict; all timings diagnostic with validation on; incomplete without final complete event"});
    if extended {
        header["document_selection"] = json!({"method":"all frozen quality-v1 documents in fixture order; maximum frozen prefix per document",
            "documents":22,"strata":{"prose":8,"code":8,"long":6},
            "overlap":"includes the original three bounded cases; not independent replication",
            "bounded_cases":self::documents().iter().map(|d|json!({"path":d.path,"prefix":d.prefix})).collect::<Vec<_>>()});
        header["tool_tasks"] = json!(tasks.iter().map(|t|json!({"id":t.id,"context_level":t.context_level,
            "prompt_tokens":t.prompt.len(),"prompt_sha256":token_hash(&t.prompt),"expected":t.expected,
            "requested_capacity":t.prompt.len()+TOOL_CAP,"capacity":t.capacity()})).collect::<Vec<_>>());
        header["tool_generations"] = json!([
            Generation::Greedy { cap: TOOL_CAP }.describe(),
            Generation::Sampled {
                seed: 11,
                cap: TOOL_CAP
            }
            .describe()
        ]);
        header["tool_definitions"] = json!(tool_definitions().0);
        header["tool_definition_check"] = json!(
            "frozen definitions and their rendered form equal quality.rs presets; no prompt retokenization"
        );
        header["tool_token_hash_encoding"] = json!("u32 little-endian");
        header["decoder"] = json!({"implementation":"NativeTokenizer::from_gguf (source artifact)",
            "cpu_reserve_bytes":TOKENIZER_BYTES,"construction":"after full CPU/GPU admission, before weight load",
            "scope":"decodes generated IDs only; frozen input IDs remain authoritative; no tokenizer compatibility claim"});
        header["plain_prompt_lengths"] = json!([]);
    }
    out.emit(header);
    let (tokenizer, weights) = {
        let _allocation = ctx.begin_allocation_transaction();
        let preflight = preflight_session(&ctx, &gguf, artifact.model(), capacity, ROWS).unwrap();
        assert_eq!(preflight.ledger.prefill_rows(), ROWS as u64);
        // CPU payload also consumes working-set headroom, conservatively
        // preserving its bound when macOS omits the process budget.
        let admission = crate::metal::evaluate_metal_memory_admission(
            preflight.ledger.peak_bytes().checked_add(cpu).unwrap(),
            0,
            ctx.memory_signals(),
            true,
        );
        out.emit(json!({"event":"preload_admission", "capacity":capacity, "workspace_rows":ROWS,
            "full_preflight_peak_bytes":preflight.ledger.peak_bytes(), "ledger":preflight.ledger.terms(),
            "cpu_reserve_bytes":cpu, "admitted":admission.is_admitted(), "reason":admission.reason.as_str(),
            "required_bytes":admission.required_bytes, "allow_zero_process_budget":true,
            "cpu_charged_to_working_set_headroom":true,
            "current_gpu_bytes":admission.signals.current_allocated_bytes,
            "process_limit_remaining_bytes":admission.signals.process_limit_remaining_bytes,
            "working_set_headroom_bytes":admission.working_set_headroom_bytes}));
        assert!(
            admission.is_admitted(),
            "mandatory full model + session + CPU admission refused: {:?}",
            admission.reason
        );
        let tokenizer = extended.then(|| {
            let tokenizer = NativeTokenizer::from_gguf(&gguf).unwrap();
            assert_eq!(tokenizer.n_vocab(), vocab);
            tokenizer
        });
        (tokenizer, Glm5NextWeights::load(&ctx, &gguf).unwrap())
    };
    assert_eq!(weights.config.vocab_size, vocab);
    if quality_screen {
        quality(&ctx, &weights, &documents, cpu, &mut out);
        if extended {
            tool_quality(
                &ctx,
                &weights,
                tokenizer.as_ref().unwrap(),
                &tasks,
                &definitions,
                cpu,
                &mut out,
            );
        }
    } else {
        plain_abba(&ctx, &weights, &documents[2], cpu, &mut out);
    }
    assert_plain();
    assert_eq!(gguf.revalidate_retained_shard_stamps().unwrap(), stamps);
    let mut complete = json!({"event":"complete", "execution_complete":true,
        "documents_per_arm":if quality_screen { documents.len() } else { 0 },
        "quality_rows_per_document_arm":if quality_screen { CONTINUATION } else { 0 },
        "plain_samples":if quality_screen { 0 } else { 16 }, "arms":2,
        "quality_verdict":null, "performance_verdict":null});
    if extended {
        complete["tool_tasks_per_arm"] = json!(tasks.len());
        complete["tool_generations_per_task_arm"] = json!(2);
        complete["tool_generations"] = json!(tasks.len() * 2 * ARMS.len());
    }
    out.emit(complete);
    out.sync();
}

#[test]
#[ignore = "release full-model tensor screen: MTL_DEBUG_LAYER=1, QWEN_METAL_LEASE_WAIT=1, QWEN_MATMAT_Q6_K_N64_MIN_N=1024, tensor tier, GLM53_GGUF and new GLM53_TENSOR_QUALITY_REPORT"]
fn tensor_prefill_quality_screen() {
    environment();
    let _lease = production_lease();
    packet(Packet::Quality);
}

#[test]
#[ignore = "release plain tensor ABBA diagnostic: MTL_DEBUG_LAYER=1, QWEN_METAL_LEASE_WAIT=1, QWEN_MATMAT_Q6_K_N64_MIN_N=1024, tensor tier, GLM53_GGUF; optional new GLM53_TENSOR_TIMING_REPORT"]
fn tensor_prefill_plain_abba() {
    environment();
    let _lease = production_lease();
    packet(Packet::Plain);
}

#[test]
#[ignore = "release extended full-model tensor quality: all 22 frozen documents and T09-T12 greedy/seed11; MTL_DEBUG_LAYER=1, QWEN_METAL_LEASE_WAIT=1, QWEN_MATMAT_Q6_K_N64_MIN_N=1024, tensor tier, GLM53_GGUF and new GLM53_TENSOR_EXTENDED_REPORT"]
fn tensor_prefill_extended_quality() {
    environment();
    let _lease = production_lease();
    packet(Packet::Extended);
}
