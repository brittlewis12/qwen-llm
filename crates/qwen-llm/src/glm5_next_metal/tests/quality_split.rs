//! Map #12 / #15 qualification of serve's Fast shared-prefix split
//! (`fast_shared_split_v1`, `serve::backend_glm5_next`) as a new Fast
//! prefill schedule, preregistered as `quality-split-v1` (cx 01a10cc jam,
//! 2026-10-08). A split prefill reads `[0, s)` then `[s, N)`, each in
//! 512-row chunks from its own start; that is serve's miss, and its hit
//! (restore at `s`, then `[s, N)`) is bit-identical to it by construction.
//!
//! 1. [`quality_split_generate`] (CPU) freezes the manifest
//!    `scripts/reference/glm53/quality-split-v1.json` from the committed
//!    quality-v1 cohort and the opencode request shape: every cut of every
//!    arm, every tool task's renderer-recorded shared-prefix cut, and the
//!    agent-shaped cases (full prompt tokens, cuts, expected calls). It is
//!    committed with `quality_split_analysis.py` before any GPU run.
//! 2. [`quality_split_evaluate`] reruns Exact in full and reads every text
//!    item with Exact, Fast 512 (the qualified control) and the split arms,
//!    then feeds the next 64 real tokens identically; runs the tool tasks
//!    with Exact, Fast 512 and the split at their real boundaries; and runs
//!    the agent cases at the deployed geometry. Per position it persists
//!    NLL, top-1 hits, KL from Exact and logit fingerprints (SHA-256 of the
//!    logit bits) so later comparisons need not rerun Exact.

use super::natural::{
    Generation, artifact_layout, ids, parse_ids, parse_turn, session, sha256_hex,
};
use super::quality::{
    CONTINUATION, FIXTURE as QUALITY_FIXTURE, TOOL_CAP, TOOL_SEEDS, log_prob, tool_definitions,
};
use super::*;
use crate::glm5_next_chat::{self as chat, Effort, Message, RenderOptions, ToolDefinition};
use crate::sampling::Sampler;
use serde_json::{Value, json};

/// Relative to this crate's manifest directory.
const SPLIT_FIXTURE: &str = "../../scripts/reference/glm53/quality-split-v1.json";
const OPENCODE_SHAPE: &str = "../../scripts/reference/glm53/opencode-shape-v1.json";
/// Serve's Fast chunk rows.
const ROWS: usize = 512;
/// `fast_split_tail`: the final segment is this many tokens plus the
/// document index mod 4 (a user turn and header; every pool residue).
const TAIL: usize = 17;
/// `fast_split_frontier` (long items only): cuts around the 2,052-token
/// sparse frontier, by long-document index mod 4.
const FRONTIER_CUTS: [usize; 4] = [2050, 2051, 2052, 2053];
/// Agent cases: generation cap and the seeded sample after greedy.
const AGENT_CAP: usize = 512;
const AGENT_SEED: u64 = 11;
/// (id, user turn, expected tool, argument, substring the argument must
/// contain): unambiguous first calls under opencode's instructions.
const AGENT_TASKS: [(&str, &str, &str, &str, &str); 4] = [
    (
        "A1-read",
        "Show me the contents of crates/qwen-cli/src/main.rs.",
        "read",
        "filePath",
        "crates/qwen-cli/src/main.rs",
    ),
    (
        "A2-grep",
        "Search the codebase for the identifier SnapshotCache and list the files that use it.",
        "grep",
        "pattern",
        "SnapshotCache",
    ),
    (
        "A3-bash",
        "Run git log --oneline -5 and tell me what the latest commit is.",
        "bash",
        "command",
        "git log",
    ),
    (
        "A4-glob",
        "Find every Markdown file under the docs directory.",
        "glob",
        "pattern",
        ".md",
    ),
];

fn manifest_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

/// The opencode request shape's instructions and tools as the GLM renderer
/// sees them (as serve converts Responses function tools).
fn opencode_shape(shape: &Value) -> (String, Vec<ToolDefinition>) {
    let instructions = shape["instructions"].as_str().unwrap().to_string();
    let tools = shape["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| {
            ToolDefinition::from_parts(
                tool["name"].as_str().unwrap(),
                tool["description"].as_str(),
                Some(&tool["parameters"]),
            )
            .unwrap()
        })
        .collect();
    (instructions, tools)
}

/// Token positions of a rendered chat's shared-prefix end and generation
/// header start, each verified to be a strict token prefix of the prompt.
fn rendered_cuts(
    encode: &dyn Fn(&str) -> Vec<u32>,
    rendered: &chat::RenderedChat,
    prompt: &[u32],
) -> (usize, usize) {
    let cut = |offset: usize| {
        let ids = encode(&rendered.text[..offset]);
        assert!(
            !ids.is_empty() && ids.len() < prompt.len() && prompt.starts_with(&ids),
            "boundary at byte {offset} is not a strict token prefix"
        );
        ids.len()
    };
    (
        cut(rendered.shared_prefix_end),
        cut(rendered.generation_header_start.unwrap()),
    )
}

/// The largest s < length with s mod ROWS = offset.
fn grid_cut(length: usize, offset: usize) -> usize {
    let s = (length - 1 - offset) / ROWS * ROWS + offset;
    assert!(s < length && s % ROWS == offset);
    s
}

/// Phase 1 (CPU only). Writes the manifest to `GLM53_SPLIT_OUT` with the
/// producer from `GLM53_PRODUCER_COMMIT`.
#[test]
#[ignore = "generator (map #12/#15 split qualification): CPU only; requires GLM53_GGUF (tokenizer), GLM53_SPLIT_OUT and GLM53_PRODUCER_COMMIT"]
fn quality_split_generate() {
    let out = PathBuf::from(std::env::var("GLM53_SPLIT_OUT").expect("GLM53_SPLIT_OUT"));
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
    let quality_bytes = std::fs::read(manifest_path(QUALITY_FIXTURE)).unwrap();
    let quality: Value = serde_json::from_slice(&quality_bytes).unwrap();
    assert_eq!(quality["name"], "glm53-quality-v1");
    assert_eq!(quality["artifact_layout"], artifact_layout(&gguf));
    let shape_bytes = std::fs::read(manifest_path(OPENCODE_SHAPE)).unwrap();
    let shape: Value = serde_json::from_slice(&shape_bytes).unwrap();
    let options = RenderOptions::generate(Effort::Low, false);

    // The agent cases first: their shared prefix fixes the grid offset.
    let (instructions, agent_tools) = opencode_shape(&shape);
    let mut agent = Vec::new();
    let mut shared_cuts = Vec::new();
    for (id, user, tool, argument, contains) in AGENT_TASKS {
        let messages = [
            Message::System(instructions.clone()),
            Message::User(user.into()),
        ];
        let rendered = chat::render_with_boundaries(&messages, &agent_tools, options).unwrap();
        let prompt = encode(&rendered.text);
        let (shared, header) = rendered_cuts(&encode, &rendered, &prompt);
        shared_cuts.push(shared);
        agent.push(json!({"id": id, "user": user,
            "expected": {"name": tool, "argument": argument, "contains": contains},
            "prompt": ids(&prompt), "prompt_tokens": prompt.len(),
            "shared_prefix_tokens": shared, "generation_header_tokens": header}));
    }
    assert!(
        shared_cuts.windows(2).all(|w| w[0] == w[1]),
        "agent cases share one prefix"
    );
    let agent_shared = shared_cuts[0];
    let offset = agent_shared % ROWS;
    eprintln!("agent shared prefix {agent_shared} tokens; in-chunk offset {offset}");

    let mut items = Vec::new();
    let mut long_index = 0;
    for (doc_index, document) in quality["documents"].as_array().unwrap().iter().enumerate() {
        let long = document["stratum"] == "long";
        for length in document["prefix_lengths"].as_array().unwrap() {
            let length = length.as_u64().unwrap() as usize;
            let frontier = long.then(|| FRONTIER_CUTS[long_index % FRONTIER_CUTS.len()]);
            let cuts = json!({
                "fast_split_grid": grid_cut(length, offset),
                "fast_split_tail": length - TAIL - doc_index % 4,
                "fast_split_frontier": frontier,
            });
            items.push(
                json!({"path": document["path"], "stratum": document["stratum"],
                "doc_index": doc_index, "prefix": length, "cuts": cuts}),
            );
        }
        if long {
            long_index += 1;
        }
    }

    let (_, definitions) = tool_definitions();
    let mut tool_tasks = Vec::new();
    for task in quality["tool_tasks"].as_array().unwrap() {
        let messages = [Message::User(task["user"].as_str().unwrap().into())];
        let rendered = chat::render_with_boundaries(&messages, &definitions, options).unwrap();
        let prompt = encode(&rendered.text);
        assert_eq!(
            prompt,
            parse_ids(&task["prompt"]),
            "{}: the re-rendered prompt differs from quality-v1",
            task["id"]
        );
        let (shared, header) = rendered_cuts(&encode, &rendered, &prompt);
        tool_tasks.push(json!({"id": task["id"], "prompt_tokens": prompt.len(),
            "shared_prefix_tokens": shared, "generation_header_tokens": header}));
    }

    let manifest = json!({
        "name": "glm53-quality-split-v1",
        "purpose": "Map #12/#15: qualify serve's Fast shared-prefix split (fast_shared_split_v1) as a Fast prefill schedule against Exact, with quality-v1's limits; analysed by scripts/reference/glm53/quality_split_analysis.py, fixed before any GPU run.",
        "schedule": "prefill [0, s) then [s, N), each in 512-row chunks from its own start (serve's miss; its hit restores at s and is bit-identical)",
        "base": {
            "quality_v1": {"path": "scripts/reference/glm53/quality-v1.json", "sha256": sha256_hex(&quality_bytes)},
            "opencode_shape": {"path": "scripts/reference/glm53/opencode-shape-v1.json", "sha256": sha256_hex(&shape_bytes)},
        },
        "artifact_layout": artifact_layout(&gguf),
        "producer": {"test": "glm5_next_metal::tests::quality_split::quality_split_generate",
            "commit": commit, "renderer": chat::RENDERER},
        "rows": ROWS,
        "agent_in_chunk_offset": offset,
        "arms": {
            "exact_512": "Exact, one prefill (full rerun; the reference)",
            "fast_512": "Fast, one prefill (the qualified control)",
            "fast_split_grid": "Fast split at the largest s < L with s mod 512 = the agent prefix's in-chunk offset",
            "fast_split_tail": format!("Fast split at s = L - {TAIL} - (document index mod 4)"),
            "fast_split_frontier": "long items only: Fast split at s = 2050 + (long-document index mod 4)",
            "tools": "exact_512, fast_512, fast_split at each task's renderer-recorded shared-prefix end",
            "agent": "exact (shared prefix prefilled once, restored per generation; checked once against an unsplit Exact prefill), fast_512 (unsplit), fast_split (shared prefix once, restored per generation: serve's hit)",
        },
        "continuation": CONTINUATION, "tool_cap": TOOL_CAP, "tool_seeds": TOOL_SEEDS,
        "agent_cap": AGENT_CAP, "agent_seed": AGENT_SEED, "agent_effort": "low",
        "token_encoding": "space-separated token ids",
        "text_items": items, "tool_tasks": tool_tasks, "agent_tasks": agent,
    });
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&manifest).unwrap() + "\n",
    )
    .unwrap();
    eprintln!("wrote {}", out.display());
}

/// A text arm: lineage and, for split arms, the cut.
#[derive(Clone, Copy)]
struct TextArm {
    name: &'static str,
    lineage: PackedLineage,
    cut: Option<usize>,
}

/// Prefill `prompt` in a fresh 512-row session of `lineage`, split at `cut`.
fn read_prompt<'w>(
    ctx: &MetalContext,
    weights: &'w Glm5NextWeights,
    capacity: usize,
    lineage: PackedLineage,
    prompt: &[u32],
    cut: Option<usize>,
) -> (Glm5NextSession<'w>, Vec<f32>) {
    let mut s = session(ctx, weights, capacity, ROWS, lineage);
    if let Some(cut) = cut {
        s.prefill_packed(ctx, &prompt[..cut]).unwrap();
    }
    let logits = s.prefill_packed(ctx, &prompt[cut.unwrap_or(0)..]).unwrap();
    (s, logits)
}

fn fingerprint(logits: &[f32]) -> String {
    sha256_hex(bytemuck::cast_slice(logits))
}

/// Chunk starts of a (possibly split) prefill of `length` tokens.
fn chunk_starts(length: usize, cut: Option<usize>) -> Vec<usize> {
    match cut {
        None => (0..length).step_by(ROWS).collect(),
        Some(cut) => (0..cut)
            .step_by(ROWS)
            .chain((cut..length).step_by(ROWS))
            .collect(),
    }
}

/// One generation after `logits` from session `s`: emitted tokens and stop.
fn generate(
    ctx: &MetalContext,
    s: &mut Glm5NextSession<'_>,
    mut logits: Vec<f32>,
    generation: Generation,
    cap: usize,
) -> (Vec<u32>, Option<u32>) {
    let mut sampler = Sampler::new(generation.sampling()).unwrap();
    let mut emitted = Vec::new();
    let stop = loop {
        assert_finite("generation logits", &logits);
        let token = sampler.sample(&logits).unwrap().token as u32;
        if chat::CHAT_STOPS.contains(&(token as i32)) {
            break Some(token);
        }
        emitted.push(token);
        if emitted.len() == cap {
            break None;
        }
        logits = s.forward(ctx, token).unwrap();
    };
    (emitted, stop)
}

/// A generation's record: calls as parsed (scored by the analysis script).
fn generation_row(
    tokenizer: &crate::tokenizer::Tokenizer,
    definitions: &[ToolDefinition],
    generation: Generation,
    emitted: &[u32],
    stop: Option<u32>,
    prompt_end: &[f32],
) -> Value {
    let as_i32: Vec<i32> = emitted.iter().map(|&t| t as i32).collect();
    let text = tokenizer.decode(&as_i32);
    let (calls, error) = match parse_turn(&text, definitions) {
        Ok((_, _, calls)) => (
            calls
                .iter()
                .map(|c| json!({"name": c.name, "arguments": Value::Object(c.arguments.clone())}))
                .collect::<Vec<_>>(),
            None,
        ),
        Err(e) => (Vec::new(), Some(e)),
    };
    json!({"generation": generation.describe(), "calls": calls, "stop": stop,
        "emitted": emitted.len(), "parse_error": error, "text": text,
        "prompt_end_logits_sha256": fingerprint(prompt_end)})
}

/// Phase 2. Reads the committed manifest (or `GLM53_SPLIT`) and writes the
/// measurements to `GLM53_SPLIT_REPORT`; the verdict is the analysis
/// script's. `GLM53_EVALUATOR_COMMIT` is recorded.
#[test]
#[ignore = "map #12/#15 split qualification: loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, GLM53_SPLIT_REPORT, GLM53_EVALUATOR_COMMIT and an idle GPU"]
fn quality_split_evaluate() {
    let report_path =
        PathBuf::from(std::env::var("GLM53_SPLIT_REPORT").expect("GLM53_SPLIT_REPORT"));
    let evaluator = std::env::var("GLM53_EVALUATOR_COMMIT").expect("GLM53_EVALUATOR_COMMIT");
    let manifest_file = std::env::var_os("GLM53_SPLIT")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_path(SPLIT_FIXTURE));
    let manifest_bytes = std::fs::read(&manifest_file).unwrap();
    let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    assert_eq!(manifest["name"], "glm53-quality-split-v1");
    let quality_bytes = std::fs::read(manifest_path(QUALITY_FIXTURE)).unwrap();
    assert_eq!(
        manifest["base"]["quality_v1"]["sha256"].as_str(),
        Some(sha256_hex(&quality_bytes).as_str()),
        "quality-v1 changed since the manifest was frozen"
    );
    let quality: Value = serde_json::from_slice(&quality_bytes).unwrap();
    let shape_bytes = std::fs::read(manifest_path(OPENCODE_SHAPE)).unwrap();
    assert_eq!(
        manifest["base"]["opencode_shape"]["sha256"].as_str(),
        Some(sha256_hex(&shape_bytes).as_str()),
        "the opencode shape changed since the manifest was frozen"
    );
    let shape: Value = serde_json::from_slice(&shape_bytes).unwrap();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let gguf = GgufFile::open(&path).unwrap();
    assert_eq!(manifest["artifact_layout"], artifact_layout(&gguf));
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let documents = quality["documents"].as_array().unwrap();
    let text_items = manifest["text_items"].as_array().unwrap();
    assert_eq!(text_items.len(), 38);
    let _lease = production_lease();
    let ctx = MetalContext::new().expect("Metal context");
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");

    // Text: Exact rerun, Fast 512 control, split arms; 64 teacher-forced
    // tokens each.
    let mut items = Vec::new();
    for item in text_items {
        let document = &documents[item["doc_index"].as_u64().unwrap() as usize];
        assert_eq!(document["path"], item["path"]);
        let tokens = parse_ids(&document["tokens"]);
        let length = item["prefix"].as_u64().unwrap() as usize;
        let (prefix, continuation) = (&tokens[..length], &tokens[length..length + CONTINUATION]);
        let cut = |arm: &str| item["cuts"][arm].as_u64().map(|c| c as usize);
        let mut arms = vec![
            TextArm {
                name: "exact_512",
                lineage: PackedLineage::Exact,
                cut: None,
            },
            TextArm {
                name: "fast_512",
                lineage: PackedLineage::Fast,
                cut: None,
            },
        ];
        for name in ["fast_split_grid", "fast_split_tail", "fast_split_frontier"] {
            if let Some(cut) = cut(name) {
                assert!(cut > 0 && cut < length, "{name} cut {cut} of {length}");
                arms.push(TextArm {
                    name,
                    lineage: PackedLineage::Fast,
                    cut: Some(cut),
                });
            }
        }
        let mut exact_logits: Vec<Vec<f32>> = Vec::new();
        let mut results = serde_json::Map::new();
        for arm in arms {
            let started = std::time::Instant::now();
            let (mut s, mut logits) = read_prompt(
                &ctx,
                &weights,
                length + CONTINUATION + 1,
                arm.lineage,
                prefix,
                arm.cut,
            );
            let prefill_ms = started.elapsed().as_secs_f64() * 1e3;
            let (mut nll, mut hits, mut kls, mut prints) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (i, &token) in continuation.iter().enumerate() {
                assert_finite("quality logits", &logits);
                nll.push(-log_prob(&logits, token));
                hits.push(argmax(&logits) == token as usize);
                prints.push(fingerprint(&logits));
                if arm.name == "exact_512" {
                    exact_logits.push(logits.clone());
                } else {
                    kls.push(kl_divergence(&exact_logits[i], &logits));
                }
                if i + 1 < continuation.len() {
                    logits = s.forward(&ctx, token).unwrap();
                }
            }
            let mean = nll.iter().sum::<f64>() / nll.len() as f64;
            let top1 = hits.iter().filter(|&&h| h).count();
            eprintln!(
                "{} @{length} {}{}: mean NLL {mean:.4}, top-1 {top1}/{CONTINUATION}, prefill {prefill_ms:.0} ms",
                item["path"].as_str().unwrap(),
                arm.name,
                arm.cut.map_or(String::new(), |c| format!(" (cut {c})")),
            );
            results.insert(
                arm.name.into(),
                json!({"cut": arm.cut, "chunk_starts": chunk_starts(length, arm.cut),
                    "mean_nll": mean, "nll": nll, "top1": top1, "hits": hits,
                    "kl_from_exact": kls, "logits_sha256": prints, "prefill_ms": prefill_ms}),
            );
        }
        items.push(json!({"path": item["path"], "stratum": item["stratum"],
            "doc_index": item["doc_index"], "prefix": length, "continuation": CONTINUATION,
            "arms": results}));
    }

    // Tool tasks at their real shared-prefix cuts.
    let (_, definitions) = tool_definitions();
    let split_tasks = manifest["tool_tasks"].as_array().unwrap();
    let mut task_reports = Vec::new();
    for (task, split) in quality["tool_tasks"]
        .as_array()
        .unwrap()
        .iter()
        .zip(split_tasks)
    {
        let id = task["id"].as_str().unwrap();
        assert_eq!(split["id"], task["id"]);
        let prompt = parse_ids(&task["prompt"]);
        let cut = split["shared_prefix_tokens"].as_u64().unwrap() as usize;
        let long = task["context_level"].as_u64().unwrap() > 0;
        let mut generations = vec![Generation::Greedy { cap: TOOL_CAP }];
        let seeds: &[u64] = if long { &TOOL_SEEDS[..1] } else { &TOOL_SEEDS };
        generations.extend(seeds.iter().map(|&seed| Generation::Sampled {
            seed,
            cap: TOOL_CAP,
        }));
        let mut per_arm = serde_json::Map::new();
        for (name, lineage, arm_cut) in [
            ("exact_512", PackedLineage::Exact, None),
            ("fast_512", PackedLineage::Fast, None),
            ("fast_split", PackedLineage::Fast, Some(cut)),
        ] {
            let mut rows = Vec::new();
            for &generation in &generations {
                let (mut s, logits) = read_prompt(
                    &ctx,
                    &weights,
                    prompt.len() + TOOL_CAP + 1,
                    lineage,
                    &prompt,
                    arm_cut,
                );
                let prompt_end = logits.clone();
                let (emitted, stop) = generate(&ctx, &mut s, logits, generation, TOOL_CAP);
                let row = generation_row(
                    &tokenizer,
                    &definitions,
                    generation,
                    &emitted,
                    stop,
                    &prompt_end,
                );
                eprintln!(
                    "{id} {name} {:?}: calls {}, stop {stop:?}",
                    generation.describe()["mode"],
                    row["calls"]
                );
                rows.push(row);
            }
            per_arm.insert(name.into(), json!(rows));
        }
        task_reports.push(json!({"id": id, "context_level": task["context_level"],
            "expected": task["expected"], "cut": cut, "arms": per_arm}));
    }

    // Agent-shaped cases at the deployed geometry.
    let (_, agent_tools) = opencode_shape(&shape);
    let agent_tasks = manifest["agent_tasks"].as_array().unwrap();
    let shared = agent_tasks[0]["shared_prefix_tokens"].as_u64().unwrap() as usize;
    let first_prompt = parse_ids(&agent_tasks[0]["prompt"]);
    let capacity = agent_tasks
        .iter()
        .map(|t| t["prompt_tokens"].as_u64().unwrap() as usize)
        .max()
        .unwrap()
        + AGENT_CAP
        + 1;
    // The shared prefix once per lineage, captured for restores.
    let shared_snapshot = |lineage: PackedLineage| {
        let mut s = session(&ctx, &weights, capacity, ROWS, lineage);
        let started = std::time::Instant::now();
        s.prefill_packed(&ctx, &first_prompt[..shared]).unwrap();
        eprintln!(
            "agent shared prefix ({shared} tokens) {lineage:?}: {:.0} ms",
            started.elapsed().as_secs_f64() * 1e3
        );
        s.capture_snapshot().unwrap()
    };
    let exact_shared = shared_snapshot(PackedLineage::Exact);
    let fast_shared = shared_snapshot(PackedLineage::Fast);
    // Check: Exact restored at the cut equals one unsplit Exact prefill.
    let (_, unsplit) = read_prompt(
        &ctx,
        &weights,
        capacity,
        PackedLineage::Exact,
        &first_prompt,
        None,
    );
    let mut s = session(&ctx, &weights, capacity, ROWS, PackedLineage::Exact);
    s.restore_snapshot(&exact_shared).unwrap();
    let restored = s.prefill_packed(&ctx, &first_prompt[shared..]).unwrap();
    drop(s);
    let exact_restore_bitwise = logit_bits(&[unsplit]) == logit_bits(&[restored]);
    assert!(
        exact_restore_bitwise,
        "Exact restored at the shared cut differs from an unsplit Exact prefill"
    );
    let generations = [
        Generation::Greedy { cap: AGENT_CAP },
        Generation::Sampled {
            seed: AGENT_SEED,
            cap: AGENT_CAP,
        },
    ];
    let mut agent_reports = Vec::new();
    for task in agent_tasks {
        let id = task["id"].as_str().unwrap();
        let prompt = parse_ids(&task["prompt"]);
        assert_eq!(
            task["shared_prefix_tokens"].as_u64(),
            Some(shared as u64),
            "{id}: one shared prefix"
        );
        assert_eq!(&prompt[..shared], &first_prompt[..shared]);
        let mut per_arm = serde_json::Map::new();
        let mut exact_end: Option<Vec<f32>> = None;
        for name in ["exact", "fast_512", "fast_split"] {
            let mut rows = Vec::new();
            for &generation in &generations {
                let (mut s, logits) = match name {
                    "fast_512" => {
                        read_prompt(&ctx, &weights, capacity, PackedLineage::Fast, &prompt, None)
                    }
                    _ => {
                        let (lineage, snapshot) = if name == "exact" {
                            (PackedLineage::Exact, &exact_shared)
                        } else {
                            (PackedLineage::Fast, &fast_shared)
                        };
                        let mut s = session(&ctx, &weights, capacity, ROWS, lineage);
                        s.restore_snapshot(snapshot).unwrap();
                        let logits = s.prefill_packed(&ctx, &prompt[shared..]).unwrap();
                        (s, logits)
                    }
                };
                let prompt_end = logits.clone();
                if name == "exact" && exact_end.is_none() {
                    exact_end = Some(prompt_end.clone());
                }
                let kl = kl_divergence(exact_end.as_ref().unwrap(), &prompt_end);
                let (emitted, stop) = generate(&ctx, &mut s, logits, generation, AGENT_CAP);
                let mut row = generation_row(
                    &tokenizer,
                    &agent_tools,
                    generation,
                    &emitted,
                    stop,
                    &prompt_end,
                );
                row["prompt_end_kl_from_exact"] = json!(kl);
                eprintln!(
                    "{id} {name} {:?}: calls {}, stop {stop:?}, prompt-end KL {kl:.3e}",
                    generation.describe()["mode"],
                    row["calls"]
                );
                rows.push(row);
            }
            per_arm.insert(name.into(), json!(rows));
        }
        agent_reports.push(json!({"id": id, "expected": task["expected"],
            "prompt_tokens": prompt.len(), "shared_prefix_tokens": shared, "arms": per_arm}));
    }

    let report = json!({
        "schema": "glm53.quality_split_report.v1",
        "manifest": manifest_file.display().to_string(), "manifest_sha256": sha256_hex(&manifest_bytes),
        "evaluator_commit": evaluator, "items": items, "tool_tasks": task_reports,
        "agent": {"shared_prefix_tokens": shared, "exact_restore_equals_unsplit": exact_restore_bitwise,
            "tasks": agent_reports},
    });
    std::fs::write(
        &report_path,
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .unwrap();
}

#[test]
fn grid_cuts_keep_the_in_chunk_offset_below_the_prefix() {
    assert_eq!(grid_cut(512, 377), 377);
    assert_eq!(grid_cut(2048, 377), 1913);
    assert_eq!(grid_cut(4352, 377), 3961);
    assert_eq!(grid_cut(378, 377), 377);
    assert_eq!(chunk_starts(1200, Some(700)), vec![0, 512, 700]);
    assert_eq!(chunk_starts(1200, None), vec![0, 512, 1024]);
}
