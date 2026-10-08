//! Map #12 quality comparison (preregistered 2026-10-08): does the Fast
//! packed-prefill lineage cost answer quality relative to Exact, judged on
//! text and tasks the model did not write?
//!
//! 1. [`quality_cohort_generate`] (CPU) freezes the cohort into
//!    `scripts/reference/glm53/quality-v1.json`: documents from this
//!    repository (prose, code, and long documents read past the sparse
//!    frontier), tokenized as raw text, and tool-call tasks with known
//!    answers. It is committed, with the analysis script and its limits
//!    (`scripts/reference/glm53/quality_analysis.py`), before any GPU run.
//! 2. [`quality_cohort_evaluate`] reads each document's prefix with Exact
//!    (512-row chunks), Fast at 512 rows and Fast at 97 rows, then feeds the
//!    next 64 real tokens one at a time, identically in every arm, recording
//!    the log-likelihood and top-1 hit of each real token and the KL from
//!    Exact. Tool tasks run greedy and seeded release sampling after an Exact
//!    or a Fast prefill and record whether the call is right.

use super::natural::{
    Generation, OBSERVATION, artifact_layout, currency_tool, ids, parse_ids, parse_turn, render,
    session, sha256_hex, weather_tool,
};
use super::*;
use crate::glm5_next_chat::{self as chat, Effort, Message, ToolDefinition};
use crate::sampling::Sampler;
use serde_json::{Value, json};

/// Relative to this crate's manifest directory.
const FIXTURE: &str = "../../scripts/reference/glm53/quality-v1.json";
const CONTINUATION: usize = 64;
/// Tool-task generations stop at a released stop or this many tokens.
const TOOL_CAP: usize = 256;
const TOOL_SEEDS: [u64; 3] = [11, 12, 13];

/// (stratum, repository path, prefix lengths).
const DOCUMENTS: [(&str, &str, &[usize]); 22] = [
    ("prose", "docs/GLM53-FLASH-PLAN.md", &[512, 2048]),
    ("prose", "docs/DEEPSEEK-V41-STRATEGY.md", &[512, 2048]),
    ("prose", "docs/H4-MTP.md", &[512, 2048]),
    ("prose", "docs/CLI-UX.md", &[512, 2048]),
    ("prose", "docs/PLAN.md", &[512, 2048]),
    ("prose", "docs/APPLE-GPU-OPTIMIZATION.md", &[512, 2048]),
    ("prose", "docs/K2-HORIZON-REVIEW.md", &[512, 2048]),
    ("prose", "docs/LENS-COMPARATIVE-PHYSIOLOGY.md", &[512, 2048]),
    ("code", "crates/qwen-llm/src/sampling.rs", &[512, 2048]),
    ("code", "crates/qwen-llm/src/tokenizer.rs", &[512, 2048]),
    ("code", "crates/qwen-llm/src/gguf.rs", &[512, 2048]),
    ("code", "crates/qwen-llm/src/metal/context.rs", &[512, 2048]),
    ("code", "crates/qwen-llm/src/glm5_next.rs", &[512, 2048]),
    ("code", "crates/qwen-cli/src/serve/events.rs", &[512, 2048]),
    ("code", "kernels/mat_vec.metal", &[512, 2048]),
    ("code", "kernels/moe.metal", &[512, 2048]),
    ("long", "docs/SERVE.md", &[4352]),
    ("long", "docs/H5-DFLASH.md", &[4352]),
    ("long", "docs/DEEPSEEK-V4-STRATEGY.md", &[4352]),
    ("long", "docs/ENV.md", &[4352]),
    ("long", "crates/qwen-llm/src/glm5_next_metal.rs", &[4352]),
    ("long", "crates/qwen-cli/src/serve/http.rs", &[4352]),
];

/// Expected call: tool name and argument checks.
#[derive(Clone, Copy)]
enum Expect {
    Weather {
        city: &'static str,
    },
    Currency {
        amount: f64,
        from: &'static str,
        to: &'static str,
    },
}

impl Expect {
    fn describe(self) -> Value {
        match self {
            Self::Weather { city } => json!({"name": "get_weather", "city": city}),
            Self::Currency { amount, from, to } => {
                json!({"name": "convert_currency", "amount": amount, "from": from, "to": to})
            }
        }
    }
}

/// (id, question, padding: 0 none / 1 ~1.5K-token context / 2 ~4.4K, expected).
const TOOL_TASKS: [(&str, &str, u8, Expect); 12] = [
    (
        "T01",
        "What's the weather in Oslo tomorrow? Use the tool.",
        0,
        Expect::Weather { city: "Oslo" },
    ),
    (
        "T02",
        "Convert 250 US dollars to euros using the tool.",
        0,
        Expect::Currency {
            amount: 250.0,
            from: "USD",
            to: "EUR",
        },
    ),
    (
        "T03",
        "Give me the three-day forecast for Lisbon. Use the tool.",
        0,
        Expect::Weather { city: "Lisbon" },
    ),
    (
        "T04",
        "How many Japanese yen is 40 British pounds? Use the tool.",
        0,
        Expect::Currency {
            amount: 40.0,
            from: "GBP",
            to: "JPY",
        },
    ),
    (
        "T05",
        "I'm flying to Reykjavik on Friday. Will it be cold? Check the forecast.",
        0,
        Expect::Weather { city: "Reykjavik" },
    ),
    (
        "T06",
        "Before anything else, check the weather in Kyoto.",
        0,
        Expect::Weather { city: "Kyoto" },
    ),
    (
        "T07",
        "Convert 1000 Swiss francs into US dollars.",
        0,
        Expect::Currency {
            amount: 1000.0,
            from: "CHF",
            to: "USD",
        },
    ),
    (
        "T08",
        "What's the weather like in Nairobi right now? Use the tool.",
        0,
        Expect::Weather { city: "Nairobi" },
    ),
    (
        "T09",
        "After reading the passages above: what's the weather in Montreal? Use the tool.",
        1,
        Expect::Weather { city: "Montreal" },
    ),
    (
        "T10",
        "After reading the passages above: convert 75 euros to Canadian dollars with the tool.",
        1,
        Expect::Currency {
            amount: 75.0,
            from: "EUR",
            to: "CAD",
        },
    ),
    (
        "T11",
        "After reading the passages above: what's the weather in Santiago? Use the tool.",
        2,
        Expect::Weather { city: "Santiago" },
    ),
    (
        "T12",
        "After reading the passages above: convert 500 Australian dollars to yen with the tool.",
        2,
        Expect::Currency {
            amount: 500.0,
            from: "AUD",
            to: "JPY",
        },
    ),
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Normalizes a currency argument to an ISO code where it names a common one.
fn currency_code(value: &str) -> String {
    let v = value.trim().to_ascii_lowercase();
    let code = match v.as_str() {
        "usd" | "us dollar" | "us dollars" | "dollar" | "dollars" | "united states dollar" => "USD",
        "eur" | "euro" | "euros" => "EUR",
        "gbp" | "pound" | "pounds" | "british pound" | "british pounds" | "pound sterling" => "GBP",
        "jpy" | "yen" | "japanese yen" => "JPY",
        "chf" | "swiss franc" | "swiss francs" | "franc" | "francs" => "CHF",
        "cad" | "canadian dollar" | "canadian dollars" => "CAD",
        "aud" | "australian dollar" | "australian dollars" => "AUD",
        _ => return value.trim().to_ascii_uppercase(),
    };
    code.to_string()
}

/// Whether a parsed call is the expected one (first call only).
fn call_correct(calls: &[chat::ToolCall], expect: Expect) -> bool {
    let Some(call) = calls.first() else {
        return false;
    };
    let text = |key: &str| {
        call.arguments
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or("")
    };
    match expect {
        Expect::Weather { city } => {
            call.name == "get_weather"
                && text("city")
                    .to_ascii_lowercase()
                    .contains(&city.to_ascii_lowercase())
        }
        Expect::Currency { amount, from, to } => {
            let got = call.arguments.get("amount").and_then(|v| {
                v.as_f64()
                    .or_else(|| v.as_str().and_then(|s| s.replace(',', "").parse().ok()))
            });
            call.name == "convert_currency"
                && got.is_some_and(|a| (a - amount).abs() < 1e-6)
                && currency_code(text("from")) == from
                && currency_code(text("to")) == to
        }
    }
}

/// The padding passage for a tool task's context level.
fn padding(level: u8, passages: &str) -> String {
    let bytes = match level {
        0 => return String::new(),
        1 => 6_000,
        _ => passages.len(),
    };
    let mut end = bytes.min(passages.len());
    while !passages.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n\n", &passages[..end])
}

fn tool_definitions() -> (Vec<Value>, Vec<ToolDefinition>) {
    let tools = vec![weather_tool(), currency_tool()];
    let definitions = tools
        .iter()
        .map(|t| ToolDefinition::from_value(t).unwrap())
        .collect();
    (tools, definitions)
}

/// Phase 1 (CPU only). Writes the cohort to `GLM53_QUALITY_OUT` with the
/// producer from `GLM53_PRODUCER_COMMIT`.
#[test]
#[ignore = "generator (map #12 quality cohort): CPU only; requires GLM53_GGUF (tokenizer), GLM53_QUALITY_OUT and GLM53_PRODUCER_COMMIT"]
fn quality_cohort_generate() {
    let out = PathBuf::from(std::env::var("GLM53_QUALITY_OUT").expect("GLM53_QUALITY_OUT"));
    let commit = std::env::var("GLM53_PRODUCER_COMMIT").expect("GLM53_PRODUCER_COMMIT");
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let gguf = GgufFile::open(&path).unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let encode = |text: &str| -> Vec<u32> {
        tokenizer
            .encode(text, false)
            .unwrap()
            .into_iter()
            .map(|t| t as u32)
            .collect()
    };
    let mut documents = Vec::new();
    for (stratum, rel, lengths) in DOCUMENTS {
        let text = std::fs::read_to_string(repo_root().join(rel)).unwrap();
        let tokens = encode(&format!("[gMASK]<sop>{text}"));
        let need = lengths.iter().max().unwrap() + CONTINUATION;
        assert!(
            tokens.len() >= need,
            "{rel}: {} tokens < {need}",
            tokens.len()
        );
        documents.push(json!({
            "stratum": stratum, "path": rel, "text_sha256": sha256_hex(text.as_bytes()),
            "prefix_lengths": lengths, "tokens": ids(&tokens[..need]),
        }));
    }
    let passages = long_qualification_text();
    let passages = &passages["[gMASK]<sop>".len()..];
    let (tools, definitions) = tool_definitions();
    let tasks: Vec<Value> = TOOL_TASKS
        .iter()
        .map(|&(id, question, level, expect)| {
            let user = format!("{}{question}", padding(level, passages));
            let messages = vec![Message::User(user.clone())];
            let tokens = encode(&render(&messages, &definitions, Effort::Low));
            json!({"id": id, "context_level": level, "user": user, "effort": "low",
                "expected": expect.describe(), "prompt": ids(&tokens), "prompt_tokens": tokens.len()})
        })
        .collect();
    let fixture = json!({
        "name": "glm53-quality-v1",
        "purpose": "Map #12 quality comparison: Fast vs Exact packed prefill on text and tasks the model did not write; analysed by scripts/reference/glm53/quality_analysis.py with limits fixed before any GPU run.",
        "artifact_layout": artifact_layout(&gguf),
        "producer": {"test": "glm5_next_metal::tests::quality::quality_cohort_generate", "commit": commit,
            "renderer": chat::RENDERER},
        "continuation": CONTINUATION, "tool_cap": TOOL_CAP, "tool_seeds": TOOL_SEEDS,
        "tools": tools, "token_encoding": "space-separated token ids; documents are [gMASK]<sop> + raw text",
        "documents": documents, "tool_tasks": tasks,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&fixture).unwrap() + "\n").unwrap();
    eprintln!("wrote {}", out.display());
}

#[derive(Clone, Copy, Debug)]
enum Arm {
    Exact512,
    Fast512,
    Fast97,
}

impl Arm {
    fn lineage(self) -> PackedLineage {
        match self {
            Self::Exact512 => PackedLineage::Exact,
            Self::Fast512 | Self::Fast97 => PackedLineage::Fast,
        }
    }

    fn rows(self) -> usize {
        match self {
            Self::Exact512 | Self::Fast512 => 512,
            Self::Fast97 => 97,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Exact512 => "exact_512",
            Self::Fast512 => "fast_512",
            Self::Fast97 => "fast_97",
        }
    }
}

/// log p(token) under `logits` (f64 log-softmax).
fn log_prob(logits: &[f32], token: u32) -> f64 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|v| (*v as f64 - max).exp()).sum();
    logits[token as usize] as f64 - max - sum.ln()
}

/// Phase 2. Reads the committed cohort (or `GLM53_QUALITY`) and writes the
/// per-item measurements to `GLM53_QUALITY_REPORT`; the verdict is the
/// analysis script's. `GLM53_EVALUATOR_COMMIT` is recorded.
#[test]
#[ignore = "map #12 quality comparison: loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, GLM53_QUALITY_REPORT, GLM53_EVALUATOR_COMMIT and an idle GPU"]
fn quality_cohort_evaluate() {
    let report_path =
        PathBuf::from(std::env::var("GLM53_QUALITY_REPORT").expect("GLM53_QUALITY_REPORT"));
    let evaluator = std::env::var("GLM53_EVALUATOR_COMMIT").expect("GLM53_EVALUATOR_COMMIT");
    let fixture_path = std::env::var_os("GLM53_QUALITY")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE));
    let fixture_bytes = std::fs::read(&fixture_path).unwrap();
    let fixture: Value = serde_json::from_slice(&fixture_bytes).unwrap();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let gguf = GgufFile::open(&path).unwrap();
    assert_eq!(
        fixture["artifact_layout"],
        artifact_layout(&gguf),
        "different artifact layout"
    );
    assert_eq!(fixture["name"], "glm53-quality-v1");
    let documents = fixture["documents"].as_array().unwrap();
    let tasks = fixture["tool_tasks"].as_array().unwrap();
    assert_eq!(documents.len(), DOCUMENTS.len());
    assert_eq!(tasks.len(), TOOL_TASKS.len());
    let _lease = production_lease();
    let ctx = MetalContext::new().expect("Metal context");
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let arms = [Arm::Exact512, Arm::Fast512, Arm::Fast97];

    let mut items = Vec::new();
    for document in documents {
        let tokens = parse_ids(&document["tokens"]);
        for length in document["prefix_lengths"].as_array().unwrap() {
            let length = length.as_u64().unwrap() as usize;
            let (prefix, continuation) =
                (&tokens[..length], &tokens[length..length + CONTINUATION]);
            let mut exact_logits: Vec<Vec<f32>> = Vec::new();
            let mut results = serde_json::Map::new();
            for arm in arms {
                let started = std::time::Instant::now();
                let mut s = session(
                    &ctx,
                    &weights,
                    length + CONTINUATION + 1,
                    arm.rows(),
                    arm.lineage(),
                );
                let mut logits = s.prefill_packed(&ctx, prefix).unwrap();
                let prefill_ms = started.elapsed().as_secs_f64() * 1e3;
                let (mut nll, mut hits, mut kls) = (Vec::new(), 0usize, Vec::new());
                for (i, &token) in continuation.iter().enumerate() {
                    assert_finite("quality logits", &logits);
                    nll.push(-log_prob(&logits, token));
                    hits += usize::from(argmax(&logits) == token as usize);
                    match arm {
                        Arm::Exact512 => exact_logits.push(logits.clone()),
                        _ => kls.push(kl_divergence(&exact_logits[i], &logits)),
                    }
                    if i + 1 < continuation.len() {
                        logits = s.forward(&ctx, token).unwrap();
                    }
                }
                let mean = nll.iter().sum::<f64>() / nll.len() as f64;
                eprintln!(
                    "{} @{length} {}: mean NLL {mean:.4}, top-1 {hits}/{}, prefill {prefill_ms:.0} ms{}",
                    document["path"].as_str().unwrap(),
                    arm.name(),
                    continuation.len(),
                    if kls.is_empty() {
                        String::new()
                    } else {
                        format!(
                            ", KL(exact||arm) mean {:.3e} worst {:.3e}",
                            kls.iter().sum::<f64>() / kls.len() as f64,
                            kls.iter().cloned().fold(0.0, f64::max)
                        )
                    }
                );
                results.insert(
                    arm.name().into(),
                    json!({"mean_nll": mean, "nll": nll, "top1": hits,
                    "kl_from_exact": kls, "prefill_ms": prefill_ms}),
                );
            }
            items.push(
                json!({"path": document["path"], "stratum": document["stratum"],
                "prefix": length, "continuation": CONTINUATION, "arms": results}),
            );
        }
    }

    let (_, definitions) = tool_definitions();
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let mut task_reports = Vec::new();
    for task in tasks {
        let id = task["id"].as_str().unwrap();
        let (_, _, _, expect) = TOOL_TASKS.iter().find(|t| t.0 == id).copied().unwrap();
        let prompt = parse_ids(&task["prompt"]);
        let long = task["context_level"].as_u64().unwrap() > 0;
        // Short tasks: greedy and three seeds; long tasks greedy and one seed
        // (an Exact prefill per generation is the cost).
        let mut generations = vec![Generation::Greedy { cap: TOOL_CAP }];
        let seeds: &[u64] = if long { &TOOL_SEEDS[..1] } else { &TOOL_SEEDS };
        generations.extend(seeds.iter().map(|&seed| Generation::Sampled {
            seed,
            cap: TOOL_CAP,
        }));
        let mut per_arm = serde_json::Map::new();
        for arm in [Arm::Exact512, Arm::Fast512] {
            let mut rows = Vec::new();
            for generation in &generations {
                let mut s = session(
                    &ctx,
                    &weights,
                    prompt.len() + TOOL_CAP + 1,
                    arm.rows(),
                    arm.lineage(),
                );
                let mut sampler = Sampler::new(generation.sampling()).unwrap();
                let mut logits = s.prefill_packed(&ctx, &prompt).unwrap();
                let mut emitted = Vec::new();
                let stop = loop {
                    let token = sampler.sample(&logits).unwrap().token as u32;
                    if chat::CHAT_STOPS.contains(&(token as i32)) {
                        break Some(token);
                    }
                    emitted.push(token);
                    if emitted.len() == TOOL_CAP {
                        break None;
                    }
                    logits = s.forward(&ctx, token).unwrap();
                };
                let text = {
                    let as_i32: Vec<i32> = emitted.iter().map(|&t| t as i32).collect();
                    tokenizer.decode(&as_i32)
                };
                let parsed = parse_turn(&text, &definitions);
                let (correct, calls, error) = match &parsed {
                    Ok((_, _, calls)) => (
                        stop == Some(OBSERVATION) && call_correct(calls, expect),
                        calls.iter().map(|c| json!({"name": c.name, "arguments": Value::Object(c.arguments.clone())})).collect::<Vec<_>>(),
                        None,
                    ),
                    Err(e) => (false, Vec::new(), Some(e.clone())),
                };
                eprintln!(
                    "{id} {} {:?}: correct {correct}, calls {calls:?}, stop {stop:?}",
                    arm.name(),
                    generation.describe()["mode"]
                );
                rows.push(
                    json!({"generation": generation.describe(), "correct": correct, "calls": calls,
                    "stop": stop, "emitted": emitted.len(), "parse_error": error, "text": text}),
                );
            }
            per_arm.insert(arm.name().into(), json!(rows));
        }
        task_reports.push(json!({"id": id, "context_level": task["context_level"],
            "expected": task["expected"], "arms": per_arm}));
    }

    let report = json!({
        "schema": "glm53.quality_report.v1",
        "fixture": fixture_path.display().to_string(), "fixture_sha256": sha256_hex(&fixture_bytes),
        "evaluator_commit": evaluator, "items": items, "tool_tasks": task_reports,
    });
    std::fs::write(
        &report_path,
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .unwrap();
}
