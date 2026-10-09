//! Map #12 on natural trajectories (design jam and harness review with cx
//! 01a10cc, 2026-10-07). Two phases, so every continuation is frozen before
//! Fast is evaluated:
//!
//! 1. [`reuse_natural_generate`] (Exact lineage only) renders each
//!    preregistered case's first turn, generates the model's own turn
//!    (greedy, or seeded release sampling for the max-effort case), renders
//!    the conversation again with the next turn, and records an Exact greedy
//!    continuation. It writes the fixture (`scripts/reference/glm53/
//!    reuse-natural-v1.json`) and an attempt journal beside it. Rejections
//!    (token cap, grammar, round trip, geometry) are journaled as they
//!    happen; replacements come only from each case's ordered variant list.
//!    Serve's CPU round trip (`qwen-cli`, `serve::render_glm5_next` tests)
//!    checks the fixture before it is committed and before any Fast run.
//! 2. [`reuse_natural_evaluate`] replays the frozen tokens. Exact cold is the
//!    reference R. Exact warm must equal R bitwise (logits, and persistent
//!    state at the join and at the end) at both chunk schedules, and Exact
//!    cold at 97 rows must too on the frontier cases. Fast cold and warm at
//!    512 and 97 rows must meet the Fast policy against R, and Fast warm
//!    must meet the frozen reuse gate against Fast cold at the same schedule.
//!
//! Emitted and consumed tokens are distinct, as in serve: a sampled stop is
//! emitted but never forwarded, and the rendered next turn supplies it. The
//! generated text is split as serve's pre-opened partition splits it (one
//! repeated leading `<think>` is dropped).
//!
//! [`packed_lineage_prefill_cost`] prices the Exact lineage against Fast.

use super::*;
use crate::glm5_next_chat::{
    self as chat, Effort, Message, RenderOptions, ToolCall, ToolDefinition,
};
use crate::sampling::{Sampler, SamplingConfig};
use serde_json::{Value, json};
use std::io::Write as _;

/// Relative to this crate's manifest directory.
const FIXTURE: &str = "../../scripts/reference/glm53/reuse-natural-v1.json";
const CONTINUATION: usize = 32;
const SCHEDULES: [usize; 2] = [512, 97];
/// The committed 2026-10-07 natural-reuse report: the immutable baseline
/// the adopted map #12 policy reports drift changes against (hash-checked).
const BASELINE: &str = "../../docs/bench/2026-10-07-glm53-natural-reuse/report.json";
const BASELINE_SHA256: &str = "1c58578eb981fef7695f1d71ac7b26f08e60bbb713bacddfa30aa66404bf58c1";
/// The schedule the investigation trigger applies to (serve's); other
/// schedules are diagnostic.
const TRIGGER_ROWS: usize = 512;
/// Released `<|observation|>`: a turn that called tools ends with it.
pub(super) const OBSERVATION: u32 = 154_829;
/// The preregistered cohort, in order. A full qualification needs all of it.
const EXPECTED_CASES: [&str; 6] = [
    "H1-short-chat",
    "H2-code-context",
    "H3-tool-dense",
    "H4-tool-across-frontier",
    "H5-long-chat-sparse",
    "H6-max-sampled-tool",
];

#[derive(Clone, Copy, Debug)]
pub(super) enum Generation {
    Greedy {
        cap: usize,
    },
    /// The release preset (`SamplingConfig::glm5_next`) with this seed.
    Sampled {
        seed: u64,
        cap: usize,
    },
}

impl Generation {
    pub(super) fn cap(self) -> usize {
        match self {
            Self::Greedy { cap } | Self::Sampled { cap, .. } => cap,
        }
    }

    pub(super) fn sampling(self) -> SamplingConfig {
        match self {
            Self::Greedy { .. } => SamplingConfig {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                seed: 0,
            },
            Self::Sampled { seed, .. } => SamplingConfig::glm5_next(seed),
        }
    }

    pub(super) fn describe(self) -> Value {
        let s = self.sampling();
        let mode = match self {
            Self::Greedy { .. } => "greedy",
            Self::Sampled { .. } => "sampled",
        };
        json!({"mode": mode, "cap": self.cap(), "temperature": s.temperature,
            "top_k": s.top_k, "top_p": s.top_p, "min_p": s.min_p, "seed": s.seed,
            "sampler": "qwen_llm::sampling v1"})
    }
}

/// What the generated turn must look like, and where the conversation must
/// sit relative to the sparse frontier. These are scoped cohort
/// restrictions, not grammar claims.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Shape {
    /// No tool call; the whole second turn plus continuation stays dense.
    ChatDense,
    /// No tool call; the first prompt already crosses the frontier.
    ChatSparse,
    /// One call; the whole second turn plus continuation stays dense.
    ToolDense,
    /// One call; the consumed history ends below the frontier at a position
    /// that is not pool-aligned, and the results alone cross it.
    ToolAcross,
    /// One call; no geometry requirement (recorded).
    ToolAny,
}

impl Shape {
    fn tools(self) -> bool {
        !matches!(self, Self::ChatDense | Self::ChatSparse)
    }
}

/// A user turn of a leading passage trimmed (at whitespace, from the front)
/// until the rendered first turn fits `frontier - window.1 ..=
/// frontier - window.0` tokens, then the question, which is never trimmed.
struct Pad {
    passage: String,
    question: &'static str,
    window: (usize, usize),
}

/// One preregistered prompt.
struct Variant {
    label: String,
    system: Option<&'static str>,
    user: String,
    pad: Option<Pad>,
    next: String,
    effort: Effort,
    generation: Generation,
}

struct Case {
    id: &'static str,
    shape: Shape,
    tools: Vec<Value>,
    /// Primary first, then the ordered replacements.
    variants: Vec<Variant>,
}

pub(super) fn weather_tool() -> Value {
    json!({"name": "get_weather", "description": "Get the weather forecast for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "days": {"type": "integer", "description": "Forecast days (1-7)"}},
            "required": ["city"]}})
}

pub(super) fn currency_tool() -> Value {
    json!({"name": "convert_currency", "description": "Convert an amount between currencies at today's rate.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}},
            "required": ["amount", "from", "to"]}})
}

/// The first `bytes` bytes of `source` (rounded down to a character
/// boundary), cut at the last line break.
fn lines_prefix(source: &str, bytes: usize) -> &str {
    let mut end = bytes.min(source.len());
    while !source.is_char_boundary(end) {
        end -= 1;
    }
    match source[..end].rfind('\n') {
        Some(line) => &source[..line],
        None => &source[..end],
    }
}

/// The preregistered cohort (natural-v1). Fixed before any generation; a
/// case's later variants are used only when an earlier one is rejected.
fn cohort() -> Vec<Case> {
    let low = |cap| (Effort::Low, Generation::Greedy { cap });
    let variant = |label: &str,
                   user: String,
                   next: &str,
                   (effort, generation): (Effort, Generation)| Variant {
        label: label.to_string(),
        system: None,
        user,
        pad: None,
        next: next.to_string(),
        effort,
        generation,
    };
    let code = lines_prefix(include_str!("../../glm5_next/memory.rs"), 5200);
    let code_b = lines_prefix(include_str!("../../glm5_next/coverage.rs"), 5200);
    let passages = long_qualification_text();
    let passages = passages["[gMASK]<sop>".len()..].to_string();
    let weather_results = "Lisbon forecast. Day 1: 21C, sunny, light wind from the north-west, humidity 55 percent. Day 2: 19C, light rain in the afternoon, wind 20 km/h, humidity 78 percent. ";
    let cases = vec![
        Case {
            id: "H1-short-chat",
            shape: Shape::ChatDense,
            tools: vec![],
            variants: vec![
                variant("bicycle", "In two sentences, why does a moving bicycle stay upright?".into(),
                    "Now explain it to a ten-year-old in one sentence.", low(384)),
                variant("tuple", "What is the difference between a list and a tuple in Python? Answer briefly.".into(),
                    "Give one example where a tuple is the better choice.", low(384)),
                variant("copper", "Name three uses of copper and say briefly why it suits each.".into(),
                    "Which of those uses is the most common today?", low(384)),
            ],
        },
        Case {
            id: "H2-code-context",
            shape: Shape::ChatDense,
            tools: vec![],
            variants: vec![
                variant("memory-ledger", format!("```rust\n{code}\n```\n\nWhat does `decode_scratch_specs` allocate, and which buffer grows with the vocabulary? Be brief."),
                    "Which of these buffers would you share across layers to save memory? One sentence.", low(384)),
                variant("coverage", format!("```rust\n{code_b}\n```\n\nWhat does `coverage` return for a routed expert down bank stored as IQ2_S, and why? Be brief."),
                    "How would you add that case? Answer in three short steps.", low(384)),
            ],
        },
        Case {
            id: "H3-tool-dense",
            shape: Shape::ToolDense,
            tools: vec![weather_tool(), currency_tool()],
            variants: vec![
                variant("lisbon", "What's the weather in Lisbon for the next two days? Use the tool.".into(),
                    weather_results, low(384)),
                variant("yen", "How many euros is 15000 Japanese yen today? Use the tool.".into(),
                    "15000 JPY = 91.73 EUR at 0.0061153 EUR per JPY (rate as of 09:00 UTC).", low(384)),
            ],
        },
        Case {
            id: "H4-tool-across-frontier",
            shape: Shape::ToolAcross,
            tools: vec![weather_tool()],
            variants: [("window-a", (400, 460)), ("window-b", (390, 450)), ("window-c", (410, 470))]
                .into_iter()
                .map(|(label, window)| Variant {
                    label: label.to_string(),
                    system: None,
                    user: String::new(),
                    pad: Some(Pad {
                        passage: passages.clone(),
                        question: "After reading the passages above: what's the weather in Lisbon for the next two days? Use the tool.",
                        window,
                    }),
                    next: weather_results.repeat(12),
                    effort: Effort::Low,
                    generation: Generation::Greedy { cap: 384 },
                })
                .collect(),
        },
        Case {
            id: "H5-long-chat-sparse",
            shape: Shape::ChatSparse,
            tools: vec![],
            variants: vec![
                variant("summarize", format!("{passages}\n\nSummarize the passages above in three sentences."),
                    "Which passage is the most technical, and why? One sentence.", low(384)),
                variant("themes", format!("{passages}\n\nList the main topic of each passage above, one line each."),
                    "Which two passages are most closely related? One sentence.", low(384)),
            ],
        },
        Case {
            id: "H6-max-sampled-tool",
            shape: Shape::ToolAny,
            tools: vec![weather_tool()],
            variants: [20_261_007u64, 20_261_008, 20_261_009]
                .into_iter()
                .map(|seed| Variant {
                    label: format!("kyoto-seed-{seed}"),
                    system: Some("You are a concise travel assistant."),
                    user: "I'm planning a weekend in Kyoto. Check the weather for Saturday and Sunday, then suggest one indoor and one outdoor activity.".into(),
                    pad: None,
                    next: "Kyoto forecast. Saturday: 24C, clear skies, light breeze. Sunday: 17C, rain all day, heavy in the afternoon.".into(),
                    effort: Effort::Max,
                    generation: Generation::Sampled { seed, cap: 1536 },
                })
                .collect(),
        },
    ];
    let ids: Vec<&str> = cases.iter().map(|c| c.id).collect();
    assert_eq!(
        ids, EXPECTED_CASES,
        "the cohort and its expected ids differ"
    );
    cases
}

pub(super) fn render(messages: &[Message], tools: &[ToolDefinition], effort: Effort) -> String {
    let options = RenderOptions::generate(effort, false);
    if tools.is_empty() {
        chat::render(messages, options).unwrap()
    } else {
        chat::render_with_tools(messages, tools, options).unwrap()
    }
}

/// `--messages` document form of a conversation (role/content, reasoning,
/// tool calls and results), as `chat::parse_document` reads it.
fn document(messages: &[Message], tools: &[Value]) -> Value {
    let message = |m: &Message| match m {
        Message::System(text) => json!({"role": "system", "content": text}),
        Message::User(text) => json!({"role": "user", "content": text}),
        Message::Assistant {
            content,
            reasoning,
            calls,
        } => {
            let mut object = json!({"role": "assistant", "content": content});
            if let Some(reasoning) = reasoning {
                object["reasoning_content"] = json!(reasoning);
            }
            if !calls.is_empty() {
                object["tool_calls"] = calls
                    .iter()
                    .map(|c| json!({"id": c.id, "type": "function", "function": {"name": c.name, "arguments": Value::Object(c.arguments.clone())}}))
                    .collect();
            }
            object
        }
        Message::Tool { call_id, content } => {
            json!({"role": "tool", "tool_call_id": call_id, "content": content})
        }
    };
    let mut doc = json!({"messages": messages.iter().map(message).collect::<Vec<_>>()});
    if !tools.is_empty() {
        doc["tools"] = json!(tools);
    }
    doc
}

pub(super) fn ids(tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn parse_ids(value: &Value) -> Vec<u32> {
    value
        .as_str()
        .expect("token ids are a string")
        .split_whitespace()
        .map(|t| t.parse().expect("token id"))
        .collect()
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Layout fingerprint of the opened artifact from metadata only (no weight
/// bytes are read): shard file names and lengths, and a digest of the
/// tensor table (name, dtype, shape, shard, data offset and length of every
/// tensor). It matches the pinned artifact's layout; it does not verify
/// weight bytes or tokenizer metadata.
pub(super) fn artifact_layout(gguf: &GgufFile) -> Value {
    let shards: Vec<Value> = gguf
        .shards
        .iter()
        .map(|s| {
            json!({"file": s.path.file_name().map(|n| n.to_string_lossy().into_owned()),
                "bytes": s.mmap_len()})
        })
        .collect();
    let mut table = String::new();
    for t in &gguf.tensors {
        table.push_str(&format!(
            "{}\t{:?}\t{:?}\t{}\t{}\t{}\n",
            t.name, t.dtype, t.shape, t.shard_idx, t.data_offset, t.n_bytes
        ));
    }
    json!({"shards": shards, "tensors": gguf.tensors.len(),
        "tensor_table_sha256": sha256_hex(table.as_bytes())})
}

/// Exact or Fast session of `capacity` positions with `rows`-row chunks.
pub(super) fn session<'w>(
    ctx: &MetalContext,
    weights: &'w Glm5NextWeights,
    capacity: usize,
    rows: usize,
    lineage: PackedLineage,
) -> Glm5NextSession<'w> {
    let mut session =
        Glm5NextSession::with_prefill_rows(ctx, weights, capacity, rows.min(capacity)).unwrap();
    session.set_packed_lineage(lineage).unwrap();
    session
}

/// The model's own turn after `prompt`: emitted tokens and the stop, if one
/// ended it (the stop is the last emitted token). `Err` on the cap, with the
/// emitted tokens.
fn generate_turn(
    ctx: &MetalContext,
    weights: &Glm5NextWeights,
    prompt: &[u32],
    generation: Generation,
) -> std::result::Result<(Vec<u32>, Option<u32>), Vec<u32>> {
    let cap = generation.cap();
    let mut s = session(
        ctx,
        weights,
        prompt.len() + cap + 1,
        512,
        PackedLineage::Exact,
    );
    let mut sampler = Sampler::new(generation.sampling()).unwrap();
    let mut logits = s.prefill_packed(ctx, prompt).unwrap();
    let mut emitted = Vec::new();
    loop {
        let token = sampler.sample(&logits).unwrap().token as u32;
        emitted.push(token);
        if chat::CHAT_STOPS.contains(&(token as i32)) {
            return Ok((emitted, Some(token)));
        }
        if emitted.len() == cap {
            return Err(emitted);
        }
        logits = s.forward(ctx, token).unwrap();
    }
}

/// Reasoning, content and tool calls of a generated turn's text, split as
/// serve's pre-opened partition splits it: one repeated leading `<think>` is
/// dropped, reasoning ends at the first `</think>`, and the rest is visible
/// text whose tool block starts at the first `<tool_call>`.
pub(super) fn parse_turn(
    text: &str,
    tools: &[ToolDefinition],
) -> std::result::Result<(String, String, Vec<ToolCall>), String> {
    let text = text.strip_prefix(chat::THINK_OPEN).unwrap_or(text);
    let (reasoning, rest) = text
        .split_once(chat::THINK_CLOSE)
        .ok_or("no </think> in the generated turn")?;
    let Some(open) = rest.find(chat::TOOL_CALL_OPEN) else {
        return Ok((reasoning.into(), rest.into(), Vec::new()));
    };
    let calls = chat::parse_tool_calls(&rest[open..], tools)
        .map_err(|e| format!("tool block: {e}"))?
        .into_iter()
        .enumerate()
        .map(|(i, call)| ToolCall {
            id: format!("call_{}", i + 1),
            name: call.name,
            arguments: call.arguments,
        })
        .collect();
    Ok((reasoning.into(), rest[..open].into(), calls))
}

/// Exact greedy continuation of `prompt`: up to [`CONTINUATION`] tokens,
/// ending early at a released stop (which is not part of the continuation).
fn natural_continuation(
    ctx: &MetalContext,
    weights: &Glm5NextWeights,
    prompt: &[u32],
) -> (Vec<u32>, Option<u32>) {
    let mut s = session(
        ctx,
        weights,
        prompt.len() + CONTINUATION + 1,
        512,
        PackedLineage::Exact,
    );
    let mut sampler = Sampler::new(Generation::Greedy { cap: 0 }.sampling()).unwrap();
    let mut logits = s.prefill_packed(ctx, prompt).unwrap();
    let mut tokens = Vec::new();
    while tokens.len() < CONTINUATION {
        let token = sampler.sample(&logits).unwrap().token as u32;
        if chat::CHAT_STOPS.contains(&(token as i32)) {
            return (tokens, Some(token));
        }
        tokens.push(token);
        logits = s.forward(ctx, token).unwrap();
    }
    (tokens, None)
}

/// Trims `pad.passage` from the front at whitespace boundaries until the
/// rendered first turn fits the window: a binary search proposes a start
/// (trimming usually lowers the count), and a linear scan over every start
/// decides when the proposal does not fit. `Err` when no start fits.
/// Deterministic.
fn fit_padding(
    pad: &Pad,
    frontier: usize,
    count: impl Fn(&str) -> usize,
) -> std::result::Result<String, String> {
    let (low, high) = (frontier - pad.window.1, frontier - pad.window.0);
    let user = |start: usize| format!("{}\n\n{}", &pad.passage[start..], pad.question);
    let starts: Vec<usize> = std::iter::once(0)
        .chain(
            pad.passage
                .char_indices()
                .filter(|(_, c)| c.is_whitespace())
                .map(|(i, c)| i + c.len_utf8()),
        )
        .filter(|&i| i < pad.passage.len())
        .collect();
    let (mut lo, mut hi) = (0usize, starts.len() - 1);
    if count(&user(starts[hi])) > high {
        return Err(format!("even the shortest padding exceeds {high} tokens"));
    }
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if count(&user(starts[mid])) <= high {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    // Token counts need not fall monotonically as text is trimmed: accept
    // the search's candidate only if it fits, else scan every start in
    // order (least trimmed first) for the first that fits.
    if (low..=high).contains(&count(&user(starts[lo]))) {
        return Ok(user(starts[lo]));
    }
    starts
        .iter()
        .map(|&start| user(start))
        .find(|candidate| (low..=high).contains(&count(candidate)))
        .ok_or_else(|| format!("no whitespace start fits the padding window {low}..={high}"))
}

/// Phase 1. Writes the fixture to `GLM53_REUSE_NATURAL_OUT` and journals
/// every attempt to `<out>.attempts.jsonl` as it happens; records the
/// producer from `GLM53_PRODUCER_COMMIT` and `GLM53_PRODUCER_DIRTY`. When a
/// case exhausts its variants, an explicitly incomplete fixture is written
/// before the test fails.
#[test]
#[ignore = "generator (map #12 natural trajectories): loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, GLM53_REUSE_NATURAL_OUT, GLM53_PRODUCER_COMMIT and an idle GPU"]
fn reuse_natural_generate() {
    let out =
        PathBuf::from(std::env::var("GLM53_REUSE_NATURAL_OUT").expect("GLM53_REUSE_NATURAL_OUT"));
    let commit = std::env::var("GLM53_PRODUCER_COMMIT").expect("GLM53_PRODUCER_COMMIT");
    let dirty = std::env::var("GLM53_PRODUCER_DIRTY").as_deref() == Ok("1");
    let journal_path = out.with_extension("attempts.jsonl");
    let mut journal = std::fs::File::create(&journal_path).unwrap();
    let _lease = production_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let layout = artifact_layout(&gguf);
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let encode = |text: &str| -> Vec<u32> {
        tokenizer
            .encode(text, false)
            .unwrap()
            .into_iter()
            .map(|t| t as u32)
            .collect()
    };
    let frontier = weights.config.sparse_frontier() as usize;
    let pool = weights.config.indexer_pool as usize;
    let qual: Value = serde_json::from_str(QUAL_V1_MANIFEST).unwrap();
    let write_fixture = |cases: &[Value], complete: bool| {
        let fixture = json!({
            "name": "glm53-reuse-natural-v1",
            "complete": complete,
            "purpose": "Map #12 natural trajectories: the model's own turns (Exact lineage) and Exact greedy continuations, frozen before any Fast evaluation; replayed by glm5_next_metal::tests::natural::reuse_natural_evaluate.",
            "declared_artifact": qual["artifact"],
            "artifact_layout": layout,
            "retained_weight_bytes": weights.retained_bytes,
            "frontier": frontier, "pool": pool, "continuation_max": CONTINUATION,
            "expected_cases": EXPECTED_CASES,
            "producer": {"test": "glm5_next_metal::tests::natural::reuse_natural_generate",
                "commit": commit, "dirty": dirty, "lineage": "exact", "prefill_rows": 512,
                "renderer": chat::RENDERER, "template_sha256": chat::GGUF_TEMPLATE_SHA256,
                "journal": journal_path.file_name().map(|n| n.to_string_lossy().into_owned())},
            "token_encoding": "space-separated token ids",
            "cases": cases,
        });
        std::fs::write(&out, serde_json::to_string_pretty(&fixture).unwrap() + "\n").unwrap();
    };
    let mut cases: Vec<Value> = Vec::new();
    for case in cohort() {
        let definitions: Vec<ToolDefinition> = case
            .tools
            .iter()
            .map(|t| ToolDefinition::from_value(t).unwrap())
            .collect();
        let mut accepted = None;
        for (index, variant) in case.variants.iter().enumerate() {
            let started = std::time::Instant::now();
            let first_messages = |user: String| {
                let mut messages = Vec::new();
                if let Some(system) = variant.system {
                    messages.push(Message::System(system.into()));
                }
                messages.push(Message::User(user));
                messages
            };
            let mut entry = json!({"case": case.id, "variant_index": index, "variant": variant.label,
                "effort": variant.effort.as_str(), "generation": variant.generation.describe()});
            let mut journal_entry = |entry: &mut Value, outcome: &str, detail: Option<String>| {
                entry["outcome"] = json!(outcome);
                if let Some(detail) = detail {
                    eprintln!(
                        "{} {}: rejected ({outcome}): {detail}",
                        case.id, variant.label
                    );
                    entry["detail"] = json!(detail);
                }
                writeln!(journal, "{}", serde_json::to_string(entry).unwrap()).unwrap();
                journal.flush().unwrap();
            };
            let user = match &variant.pad {
                None => Ok(variant.user.clone()),
                Some(pad) => fit_padding(pad, frontier, |user| {
                    encode(&render(
                        &first_messages(user.to_string()),
                        &definitions,
                        variant.effort,
                    ))
                    .len()
                }),
            };
            let user = match user {
                Ok(user) => user,
                Err(detail) => {
                    journal_entry(&mut entry, "geometry", Some(detail));
                    continue;
                }
            };
            let turn1_messages = first_messages(user.clone());
            let turn1 = encode(&render(&turn1_messages, &definitions, variant.effort));
            entry["user_sha256"] = json!(sha256_hex(user.as_bytes()));
            entry["first_turn_tokens"] = json!(turn1.len());
            let (emitted, stop) = match generate_turn(&ctx, &weights, &turn1, variant.generation) {
                Ok(turn) => turn,
                Err(emitted) => {
                    entry["emitted"] = json!(ids(&emitted));
                    journal_entry(
                        &mut entry,
                        "cap",
                        Some(format!(
                            "token cap {} reached without a stop",
                            variant.generation.cap()
                        )),
                    );
                    continue;
                }
            };
            entry["emitted"] = json!(ids(&emitted));
            entry["stop"] = json!(stop);
            // A stop is emitted, never forwarded; everything before it is consumed.
            let consumed: Vec<u32> =
                emitted[..emitted.len() - usize::from(stop.is_some())].to_vec();
            let consumed_i32: Vec<i32> = consumed.iter().map(|&t| t as i32).collect();
            let text = tokenizer.decode(&consumed_i32);
            let (reasoning, content, calls) = match parse_turn(&text, &definitions) {
                Ok(parsed) => parsed,
                Err(detail) => {
                    journal_entry(&mut entry, "grammar", Some(detail));
                    continue;
                }
            };
            let wants_call = case.shape.tools();
            if wants_call && (calls.len() != 1 || stop != Some(OBSERVATION)) {
                journal_entry(
                    &mut entry,
                    "grammar",
                    Some(format!(
                        "{} calls, stop {stop:?}; one call ending at <|observation|> required",
                        calls.len()
                    )),
                );
                continue;
            }
            if !wants_call && !calls.is_empty() {
                journal_entry(
                    &mut entry,
                    "grammar",
                    Some(format!("{} unexpected calls", calls.len())),
                );
                continue;
            }
            let mut turn2_messages = turn1_messages.clone();
            turn2_messages.push(Message::Assistant {
                content,
                reasoning: Some(reasoning),
                calls: calls.clone(),
            });
            turn2_messages.push(if wants_call {
                Message::Tool {
                    call_id: calls[0].id.clone(),
                    content: variant.next.clone(),
                }
            } else {
                Message::User(variant.next.clone())
            });
            let turn2 = encode(&render(&turn2_messages, &definitions, variant.effort));
            let join = turn1.len() + consumed.len();
            let extends = turn2.len() > join
                && turn2[..turn1.len()] == turn1[..]
                && turn2[turn1.len()..join] == consumed[..];
            if !extends {
                let at = turn2
                    .iter()
                    .zip(turn1.iter().chain(&consumed))
                    .position(|(a, b)| a != b);
                journal_entry(
                    &mut entry,
                    "round_trip",
                    Some(format!(
                        "the rendered second turn does not extend the consumed history (first difference at {at:?}; join {join})"
                    )),
                );
                continue;
            }
            let geometry_error = match case.shape {
                Shape::ChatDense | Shape::ToolDense => (turn2.len() + CONTINUATION + 1 >= frontier)
                    .then(|| format!("second turn {} + continuation reaches the frontier {frontier}", turn2.len())),
                Shape::ChatSparse => (turn1.len() <= frontier)
                    .then(|| format!("first turn {} does not cross the frontier {frontier}", turn1.len())),
                Shape::ToolAcross => (!(join < frontier && frontier < turn2.len()) || join.is_multiple_of(pool))
                    .then(|| format!("join {join}, frontier {frontier}, second turn {}; need join < frontier < second turn and join % {pool} != 0", turn2.len())),
                Shape::ToolAny => None,
            };
            if let Some(detail) = geometry_error {
                journal_entry(&mut entry, "geometry", Some(detail));
                continue;
            }
            // The stored document must reproduce these messages.
            let doc = document(&turn2_messages, &case.tools);
            let parsed =
                chat::parse_document(serde_json::to_string(&doc).unwrap().as_bytes()).unwrap();
            assert_eq!(
                parsed.messages, turn2_messages,
                "{}: document round trip",
                case.id
            );
            let (continuation, continuation_stop) = natural_continuation(&ctx, &weights, &turn2);
            let seconds = started.elapsed().as_secs_f64();
            eprintln!(
                "{} {}: accepted; first {} + consumed {} (stop {stop:?}) + suffix {} = {}; continuation {} (stop {continuation_stop:?}); {seconds:.0} s",
                case.id,
                variant.label,
                turn1.len(),
                consumed.len(),
                turn2.len() - join,
                turn2.len(),
                continuation.len()
            );
            journal_entry(&mut entry, "accepted", None);
            accepted = Some(json!({
                "id": case.id, "shape": format!("{:?}", case.shape), "variant": variant.label,
                "variant_index": index, "effort": variant.effort.as_str(),
                "generation": variant.generation.describe(),
                "conversation": doc,
                "turn1": ids(&turn1), "emitted": ids(&emitted),
                "stop": stop, "consumed": consumed.len(),
                "suffix": ids(&turn2[join..]),
                "continuation": ids(&continuation), "continuation_stop": continuation_stop,
                "join": join, "second_turn": turn2.len(),
            }));
            break;
        }
        match accepted {
            Some(record) => cases.push(record),
            None => {
                write_fixture(&cases, false);
                panic!(
                    "{}: every preregistered variant was rejected (see {}); an incomplete fixture was written",
                    case.id,
                    journal_path.display()
                );
            }
        }
    }
    write_fixture(&cases, true);
    eprintln!("wrote {} and {}", out.display(), journal_path.display());
}

/// One frozen case, validated on the CPU.
struct Frozen {
    id: String,
    shape: String,
    turn1: Vec<u32>,
    consumed: Vec<u32>,
    suffix: Vec<u32>,
    continuation: Vec<u32>,
}

impl Frozen {
    fn join(&self) -> usize {
        self.turn1.len() + self.consumed.len()
    }

    fn turn2(&self) -> Vec<u32> {
        [
            self.turn1.as_slice(),
            self.consumed.as_slice(),
            self.suffix.as_slice(),
        ]
        .concat()
    }
}

/// Schema, token accounting and geometry of every case, before GPU work.
fn validate_fixture(fixture: &Value, frontier: usize, pool: usize) -> Vec<Frozen> {
    assert_eq!(fixture["name"], "glm53-reuse-natural-v1");
    assert_eq!(fixture["complete"], true, "the fixture is incomplete");
    assert_eq!(fixture["frontier"].as_u64(), Some(frontier as u64));
    assert_eq!(fixture["pool"].as_u64(), Some(pool as u64));
    let cases = fixture["cases"].as_array().expect("cases");
    let ids: Vec<&str> = cases.iter().map(|c| c["id"].as_str().unwrap()).collect();
    assert_eq!(
        ids, EXPECTED_CASES,
        "the fixture's cases differ from the cohort"
    );
    cases
        .iter()
        .map(|case| {
            let id = case["id"].as_str().unwrap().to_string();
            let emitted = parse_ids(&case["emitted"]);
            let consumed_len = case["consumed"].as_u64().unwrap() as usize;
            let stop = case["stop"].as_u64().map(|s| s as u32);
            match stop {
                Some(stop) => {
                    assert!(
                        chat::CHAT_STOPS.contains(&(stop as i32)),
                        "{id}: stop {stop}"
                    );
                    assert_eq!(
                        emitted.last(),
                        Some(&stop),
                        "{id}: the stop is the last emitted token"
                    );
                    assert_eq!(
                        consumed_len + 1,
                        emitted.len(),
                        "{id}: consumed = emitted - stop"
                    );
                }
                None => panic!("{id}: an accepted turn ends at a stop"),
            }
            let frozen = Frozen {
                shape: case["shape"].as_str().unwrap().to_string(),
                turn1: parse_ids(&case["turn1"]),
                consumed: emitted[..consumed_len].to_vec(),
                suffix: parse_ids(&case["suffix"]),
                continuation: parse_ids(&case["continuation"]),
                id,
            };
            let (id, join, second) = (&frozen.id, frozen.join(), frozen.turn2().len());
            assert_eq!(case["join"].as_u64(), Some(join as u64), "{id}: join");
            assert_eq!(
                case["second_turn"].as_u64(),
                Some(second as u64),
                "{id}: second turn"
            );
            assert!(
                !frozen.suffix.is_empty() && frozen.continuation.len() <= CONTINUATION,
                "{id}"
            );
            assert!(
                frozen
                    .consumed
                    .iter()
                    .chain(&frozen.continuation)
                    .all(|t| !chat::CHAT_STOPS.contains(&(*t as i32))),
                "{id}: a stop inside consumed or continuation tokens"
            );
            match frozen.shape.as_str() {
                "ChatDense" | "ToolDense" => {
                    assert!(second + CONTINUATION + 1 < frontier, "{id}: geometry")
                }
                "ChatSparse" => assert!(frozen.turn1.len() > frontier, "{id}: geometry"),
                "ToolAcross" => assert!(
                    join < frontier && frontier < second && !join.is_multiple_of(pool),
                    "{id}: geometry"
                ),
                "ToolAny" => {}
                other => panic!("{id}: unknown shape {other}"),
            }
            frozen
        })
        .collect()
}

/// Persistent state bits at the prompt end and at the end of a run.
type StatePair = (Vec<Vec<u32>>, Vec<Vec<u32>>);

/// One run's logits at the prompt end and every continuation position, and
/// (Exact runs only) its [`StatePair`].
struct Run {
    logits: Vec<Vec<f32>>,
    state: Option<StatePair>,
    ms: f64,
}

/// Fast drift from the Exact reference (adopted map #12 policy): reported,
/// with the change from the case's committed baseline; at the trigger
/// schedule a worst-position KL above the trigger records an investigation
/// failure.
fn fast_drift(
    label: &str,
    reference: &[Vec<f32>],
    fast: &[Vec<f32>],
    baseline: Option<f64>,
    trigger: bool,
    failures: &mut Vec<String>,
) -> Value {
    let drift = if trigger {
        fast_drift_with_trigger(label, reference, fast, failures)
    } else {
        let drift = Drift::measure(label, reference, fast);
        drift.print(&format!("{label} vs Exact (diagnostic)"));
        drift
    };
    let mut value = drift.json();
    value["trigger"] = json!({"applies": trigger, "tripped": trigger && drift.trips()});
    value["baseline_worst_kl"] = json!(baseline);
    value["change_from_baseline"] = json!(baseline.map(|b| drift.worst_kl - b));
    value
}

/// Phase 2. Reads the fixture (`GLM53_REUSE_NATURAL` or the committed path),
/// optionally a comma-separated case subset (`GLM53_REUSE_NATURAL_CASES`,
/// reported as partial), and writes the report to
/// `GLM53_REUSE_NATURAL_REPORT` with the evaluator revision from
/// `GLM53_EVALUATOR_COMMIT`.
#[test]
#[ignore = "map #12 natural-trajectory qualification: loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, GLM53_REUSE_NATURAL_REPORT, GLM53_EVALUATOR_COMMIT, the committed fixture and an idle GPU"]
fn reuse_natural_evaluate() {
    let report_path = PathBuf::from(
        std::env::var("GLM53_REUSE_NATURAL_REPORT").expect("GLM53_REUSE_NATURAL_REPORT"),
    );
    let evaluator = std::env::var("GLM53_EVALUATOR_COMMIT").expect("GLM53_EVALUATOR_COMMIT");
    let fixture_path = std::env::var_os("GLM53_REUSE_NATURAL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE));
    let fixture_bytes =
        std::fs::read(&fixture_path).unwrap_or_else(|e| panic!("{}: {e}", fixture_path.display()));
    let fixture: Value = serde_json::from_slice(&fixture_bytes).unwrap();
    let subset: Option<Vec<String>> = std::env::var("GLM53_REUSE_NATURAL_CASES")
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect());
    if let Some(subset) = &subset {
        assert!(
            !subset.is_empty() && subset.iter().all(|s| !s.is_empty()),
            "empty case selection"
        );
        for id in subset {
            assert!(EXPECTED_CASES.contains(&id.as_str()), "unknown case {id:?}");
        }
    }
    // CPU checks first: artifact layout, schema, accounting and geometry.
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let gguf = GgufFile::open(&path).unwrap();
    assert_eq!(
        fixture["artifact_layout"],
        artifact_layout(&gguf),
        "the fixture was generated from an artifact with a different layout"
    );
    let config = crate::glm5_next::Glm5NextConfig::from_gguf(&gguf).unwrap();
    let frontier = config.sparse_frontier() as usize;
    let pool = config.indexer_pool as usize;
    let frozen = validate_fixture(&fixture, frontier, pool);
    let _lease = production_lease();
    let ctx = MetalContext::new().expect("Metal context");
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    assert_eq!(
        fixture["retained_weight_bytes"].as_u64(),
        Some(weights.retained_bytes)
    );
    // The immutable baseline for reported changes.
    let baseline_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(BASELINE);
    let baseline_bytes = std::fs::read(&baseline_path)
        .unwrap_or_else(|e| panic!("{}: {e}", baseline_path.display()));
    assert_eq!(
        sha256_hex(&baseline_bytes),
        BASELINE_SHA256,
        "the committed baseline report changed"
    );
    let baseline: Value = serde_json::from_slice(&baseline_bytes).unwrap();
    let baseline_kl = |id: &str, rows: usize, arm: &str| -> Option<f64> {
        baseline["cases"]
            .as_array()?
            .iter()
            .find(|case| case["id"] == id)?["fast"][rows.to_string()][arm]["worst_kl"]
            .as_f64()
    };
    let mut failures = Vec::new();
    let mut reports = Vec::new();
    for case in &frozen {
        let id = case.id.as_str();
        if subset.as_ref().is_some_and(|s| !s.iter().any(|o| o == id)) {
            continue;
        }
        let (turn1, consumed, suffix, continuation) = (
            &case.turn1,
            &case.consumed,
            &case.suffix,
            &case.continuation,
        );
        let (join, turn2) = (case.join(), case.turn2());
        let positions = continuation.len() + 1;
        let capacity = turn2.len() + continuation.len() + 1;
        let frontier_case = matches!(case.shape.as_str(), "ToolAcross" | "ChatSparse");
        eprintln!(
            "{id}: first {} + consumed {} + suffix {} = {} (join {join}, frontier {frontier}); {positions} positions",
            turn1.len(),
            consumed.len(),
            suffix.len(),
            turn2.len()
        );
        let finish = |mut s: Glm5NextSession<'_>,
                      first: Vec<f32>,
                      exact: bool,
                      started: std::time::Instant|
         -> Run {
            let prompt_state = exact.then(|| state_bits(&s));
            let mut logits = vec![first];
            for &token in continuation {
                logits.push(s.forward(&ctx, token).unwrap());
            }
            let ms = started.elapsed().as_secs_f64() * 1e3;
            let state = prompt_state.map(|p| (p, state_bits(&s)));
            Run { logits, state, ms }
        };
        let cold = |lineage: PackedLineage, rows: usize| -> Run {
            let mut s = session(&ctx, &weights, capacity, rows, lineage);
            let started = std::time::Instant::now();
            let first = s.prefill_packed(&ctx, &turn2).unwrap();
            finish(s, first, lineage == PackedLineage::Exact, started)
        };
        let warm = |lineage: PackedLineage, rows: usize| -> Run {
            let mut s = session(&ctx, &weights, capacity, rows, lineage);
            let started = std::time::Instant::now();
            s.prefill_packed(&ctx, turn1).unwrap();
            for &token in consumed {
                s.forward(&ctx, token).unwrap();
            }
            assert_eq!(s.position(), join);
            let first = s.prefill_packed(&ctx, suffix).unwrap();
            finish(s, first, lineage == PackedLineage::Exact, started)
        };
        let starts = |from: usize, to: usize, rows: usize| -> Vec<usize> {
            (from..to).step_by(rows).collect()
        };
        let reference = cold(PackedLineage::Exact, 512);
        eprintln!("  exact cold 512: {:.0} ms", reference.ms);
        let (ref_prompt, ref_end) = reference.state.as_ref().unwrap();
        let mut exact = serde_json::Map::new();
        let mut check_exact = |label: String, run: Run, failures: &mut Vec<String>| {
            let (prompt, end) = run.state.as_ref().unwrap();
            let logits_equal = logit_bits(&run.logits) == logit_bits(&reference.logits);
            let (prompt_equal, end_equal) = (prompt == ref_prompt, end == ref_end);
            eprintln!(
                "  {label}: {:.0} ms; bitwise logits {logits_equal}, state at prompt end {prompt_equal}, at end {end_equal}",
                run.ms
            );
            if !(logits_equal && prompt_equal && end_equal) {
                failures.push(format!("{id} {label}: not bitwise equal to Exact cold 512 (logits {logits_equal}, prompt state {prompt_equal}, end state {end_equal})"));
            }
            exact.insert(label, json!({"ms": run.ms, "logits": logits_equal, "prompt_state": prompt_equal, "end_state": end_equal}));
        };
        for rows in SCHEDULES {
            check_exact(
                format!("exact warm {rows}"),
                warm(PackedLineage::Exact, rows),
                &mut failures,
            );
        }
        if frontier_case {
            check_exact(
                "exact cold 97".into(),
                cold(PackedLineage::Exact, 97),
                &mut failures,
            );
        }
        let mut fast = serde_json::Map::new();
        for rows in SCHEDULES {
            let fast_cold = cold(PackedLineage::Fast, rows);
            let fast_warm = warm(PackedLineage::Fast, rows);
            eprintln!(
                "  fast {rows}: cold {:.0} ms, warm {:.0} ms",
                fast_cold.ms, fast_warm.ms
            );
            let trigger = rows == TRIGGER_ROWS;
            let cold_drift = fast_drift(
                &format!("{id} fast cold {rows}"),
                &reference.logits,
                &fast_cold.logits,
                baseline_kl(id, rows, "cold"),
                trigger,
                &mut failures,
            );
            let warm_drift = fast_drift(
                &format!("{id} fast warm {rows}"),
                &reference.logits,
                &fast_warm.logits,
                baseline_kl(id, rows, "warm"),
                trigger,
                &mut failures,
            );
            let (kl, regret, agree, bitwise) = warm_cold_report(
                &format!("{id} fast warm vs cold {rows}"),
                positions,
                &fast_cold.logits,
                &fast_warm.logits,
            );
            fast.insert(rows.to_string(), json!({
                "cold": cold_drift, "warm": warm_drift,
                "warm_vs_cold": {"worst_kl_either_way": kl, "worst_regret": regret, "top1_agree": agree, "bitwise": bitwise},
                "chunk_starts": {"cold": starts(0, turn2.len(), rows),
                    "warm": [starts(0, turn1.len(), rows), starts(join, turn2.len(), rows)]},
                "ms": {"cold": fast_cold.ms, "warm": fast_warm.ms},
            }));
        }
        reports.push(json!({"id": id, "first": turn1.len(), "consumed": consumed.len(),
            "suffix": suffix.len(), "join": join, "second_turn": turn2.len(), "positions": positions,
            "exact_reference_ms": reference.ms, "exact": exact, "fast": fast}));
    }
    let partial = subset.is_some();
    let report = json!({
        "schema": "glm53.reuse_natural_report.v2",
        "partial": partial,
        "fixture": fixture_path.display().to_string(),
        "fixture_sha256": sha256_hex(&fixture_bytes),
        "fixture_producer": fixture["producer"],
        "evaluator_commit": evaluator,
        "policy": {
            "adopted": "map #12, PERF-LOG 2026-10-08: Fast drift is reported; quality is gated by the preregistered cohort",
            "fast_vs_exact": {"reported": "KL(exact||fast) per position, flips by argmax inequality with both regrets",
                "investigation_trigger": {"worst_kl_exact_fast_above": FAST_DRIFT_TRIGGER_KL,
                    "rows": TRIGGER_ROWS, "arms": ["cold", "warm"],
                    "meaning": "investigate; not a quality verdict"}},
            "warm_vs_cold": "reported",
            "exact": "bitwise logits and persistent state (hard)",
            "baseline": {"path": BASELINE, "sha256": BASELINE_SHA256}},
        "schedules": SCHEDULES, "cases": reports, "failures": failures,
    });
    std::fs::write(
        &report_path,
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .unwrap();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(!reports.is_empty(), "no case was evaluated");
}

/// Exact lineage priced against Fast: a fresh session's prompt prefill wall
/// (allocation excluded; final logits included; weights and both lineages'
/// dense and sparse pipelines warmed first) at 512, 2048 and 4096 tokens,
/// and a 64-token suffix after a reused prefix, in A-B-B-A order per length.
/// Timing, not qualification. Writes JSON to `GLM53_LINEAGE_COST_OUT`.
#[test]
#[ignore = "timing (map #12): loads the 109.5 GiB GLM-5.3 trunk; requires GLM53_GGUF, GLM53_LINEAGE_COST_OUT, no MTL_DEBUG_LAYER and an idle GPU"]
fn packed_lineage_prefill_cost() {
    let out =
        PathBuf::from(std::env::var("GLM53_LINEAGE_COST_OUT").expect("GLM53_LINEAGE_COST_OUT"));
    let _lease = perf_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let tokens: Vec<u32> = tokenizer
        .encode(&long_qualification_text(), false)
        .unwrap()
        .into_iter()
        .map(|t| t as u32)
        .collect();
    const SUFFIX: usize = 64;
    // Warm-up outside the measurements: first GPU use of the weights, and
    // each lineage's dense and sparse pipelines.
    let warm_len = weights.config.sparse_frontier() as usize + 64;
    for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
        let mut s = session(&ctx, &weights, warm_len + 1, 512, lineage);
        s.prefill_packed(&ctx, &tokens[..warm_len]).unwrap();
    }
    let mut rows = Vec::new();
    for length in [512usize, 2048, 4096] {
        assert!(tokens.len() >= length);
        for (slot, lineage) in [
            PackedLineage::Fast,
            PackedLineage::Exact,
            PackedLineage::Exact,
            PackedLineage::Fast,
        ]
        .into_iter()
        .enumerate()
        {
            let mut s = session(&ctx, &weights, length + 1, 512, lineage);
            let started = std::time::Instant::now();
            s.prefill_packed(&ctx, &tokens[..length]).unwrap();
            let fresh_ms = started.elapsed().as_secs_f64() * 1e3;
            drop(s);
            let mut s = session(&ctx, &weights, length + 1, 512, lineage);
            s.prefill_packed(&ctx, &tokens[..length - SUFFIX]).unwrap();
            let started = std::time::Instant::now();
            s.prefill_packed(&ctx, &tokens[length - SUFFIX..length])
                .unwrap();
            let suffix_ms = started.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "{length} tokens, {lineage:?} (slot {slot}): fresh prefill {fresh_ms:.0} ms ({:.1} tok/s); {SUFFIX}-token suffix after {} reused: {suffix_ms:.0} ms",
                length as f64 / (fresh_ms / 1e3),
                length - SUFFIX
            );
            rows.push(json!({"length": length, "lineage": format!("{lineage:?}"), "slot": slot,
                "fresh_prefill_ms": fresh_ms, "fresh_prefill_tok_s": length as f64 / (fresh_ms / 1e3),
                "suffix_tokens": SUFFIX, "suffix_ms": suffix_ms}));
        }
    }
    let document = json!({
        "schema": "glm53.packed_lineage_prefill_cost.v1",
        "method": "per length, Fast-Exact-Exact-Fast; a fresh session per measurement (allocation excluded); prompt prefill including final logits; suffix measured after a same-lineage prefix prefill; 512-row chunks; weights and both lineages' dense and sparse pipelines warmed first",
        "prompt": "long_qualification_text (teacher-forced)",
        "rows": rows,
    });
    std::fs::write(&out, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
}

/// Map #12 attribution (no bounds; diagnostic). For one frozen case
/// (`GLM53_ATTRIBUTION_CASE`, default H2) at 512 rows: Exact cold is the
/// reference; then Fast cold with each stage family alone in its Exact form
/// ("fast except stage"), and with every family Exact except one ("only
/// stage fast"). One-at-a-time substitutions measure sensitivity, not
/// additive shares. Writes JSON to `GLM53_ATTRIBUTION_OUT`.
#[test]
#[ignore = "map #12 attribution: loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, GLM53_ATTRIBUTION_OUT and an idle GPU"]
fn reuse_natural_stage_attribution() {
    use super::super::packed::{ExactStages, Stage};
    let out = PathBuf::from(std::env::var("GLM53_ATTRIBUTION_OUT").expect("GLM53_ATTRIBUTION_OUT"));
    let id = std::env::var("GLM53_ATTRIBUTION_CASE").unwrap_or_else(|_| "H2-code-context".into());
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let fixture: Value = serde_json::from_slice(&std::fs::read(&fixture_path).unwrap()).unwrap();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let gguf = GgufFile::open(&path).unwrap();
    assert_eq!(fixture["artifact_layout"], artifact_layout(&gguf));
    let config = crate::glm5_next::Glm5NextConfig::from_gguf(&gguf).unwrap();
    let frozen = validate_fixture(
        &fixture,
        config.sparse_frontier() as usize,
        config.indexer_pool as usize,
    );
    let case = frozen
        .iter()
        .find(|c| c.id == id)
        .unwrap_or_else(|| panic!("unknown case {id}"));
    let _lease = production_lease();
    let ctx = MetalContext::new().expect("Metal context");
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let turn2 = case.turn2();
    let capacity = turn2.len() + case.continuation.len() + 1;
    let run = |lineage: PackedLineage| -> Vec<Vec<f32>> {
        let mut s = session(&ctx, &weights, capacity, 512, lineage);
        let mut logits = vec![s.prefill_packed(&ctx, &turn2).unwrap()];
        for &token in &case.continuation {
            logits.push(s.forward(&ctx, token).unwrap());
        }
        logits
    };
    let summary = |reference: &[Vec<f32>], other: &[Vec<f32>]| -> Value {
        let kls: Vec<f64> = reference
            .iter()
            .zip(other)
            .map(|(r, o)| kl_divergence(r, o))
            .collect();
        let flips: Vec<Value> = reference
            .iter()
            .zip(other)
            .enumerate()
            .filter_map(|(i, (r, o))| {
                let (a, b) = choice_regret(r, o);
                ((a, b) != (0.0, 0.0)).then(|| json!([i, a, b]))
            })
            .collect();
        let (worst_kl, at) = worst(kls.iter().copied());
        json!({"worst_kl": worst_kl, "worst_at": at, "mean_kl": kls.iter().sum::<f64>() / kls.len() as f64,
            "prompt_end_kl": kls[0], "flips": flips})
    };
    let reference = run(PackedLineage::Exact);
    let fast = summary(&reference, &run(PackedLineage::Fast));
    eprintln!("{id}: fast {fast}");
    let mut rows = Vec::new();
    for stage in Stage::ALL {
        let except = {
            let _scope = ExactStages::set(&[stage]);
            summary(&reference, &run(PackedLineage::Fast))
        };
        let others: Vec<Stage> = Stage::ALL.into_iter().filter(|s| *s != stage).collect();
        let only = {
            let _scope = ExactStages::set(&others);
            summary(&reference, &run(PackedLineage::Fast))
        };
        eprintln!(
            "{stage:?}: fast except it: worst KL {:.3e} mean {:.3e}; only it fast: worst KL {:.3e} mean {:.3e}",
            except["worst_kl"].as_f64().unwrap(),
            except["mean_kl"].as_f64().unwrap(),
            only["worst_kl"].as_f64().unwrap(),
            only["mean_kl"].as_f64().unwrap()
        );
        rows.push(json!({"stage": format!("{stage:?}"), "fast_except_stage": except, "only_stage_fast": only}));
    }
    let all_exact = {
        let _scope = ExactStages::set(&Stage::ALL);
        summary(&reference, &run(PackedLineage::Fast))
    };
    eprintln!("all stages exact inside a Fast session: {all_exact}");
    let document = json!({
        "schema": "glm53.reuse_natural_stage_attribution.v1", "case": id, "rows_per_chunk": 512,
        "reference": "Exact cold", "fast": fast, "stages": rows,
        "all_stages_exact_in_fast_session": all_exact,
        "note": "one-at-a-time substitutions measure sensitivity, not additive shares",
    });
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&document).unwrap() + "\n",
    )
    .unwrap();
}

/// Map #12 precision probe (diagnostic; no bounds). On frozen cases (H2, H5
/// by default; `GLM53_PROBE_CASES`) at 512 rows: Exact, Exact with every
/// quantized-weight matrix input rounded through half first
/// (`packed::RoundExactActivations`: activations as a half-staged kernel
/// sees them, weights still F32-dequantized, decode accumulation), and
/// Fast. If rounded Exact diverges from Exact about as much as Fast does,
/// activation rounding alone is a sufficient source of drift of that size
/// on these cases (not proof it is Fast's source, nor a measure of what
/// F32 staging would leave); if it stays small, look to weight tiles or
/// accumulation. Writes JSON to `GLM53_PROBE_OUT`.
#[test]
#[ignore = "map #12 precision probe: loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, GLM53_PROBE_OUT and an idle GPU"]
fn reuse_natural_activation_rounding_probe() {
    use super::super::packed::RoundExactActivations;
    let out = PathBuf::from(std::env::var("GLM53_PROBE_OUT").expect("GLM53_PROBE_OUT"));
    let wanted: Vec<String> = std::env::var("GLM53_PROBE_CASES")
        .unwrap_or_else(|_| "H2-code-context,H5-long-chat-sparse".into())
        .split(',')
        .map(str::to_string)
        .collect();
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let fixture: Value = serde_json::from_slice(&std::fs::read(&fixture_path).unwrap()).unwrap();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let gguf = GgufFile::open(&path).unwrap();
    assert_eq!(fixture["artifact_layout"], artifact_layout(&gguf));
    let config = crate::glm5_next::Glm5NextConfig::from_gguf(&gguf).unwrap();
    let frozen = validate_fixture(
        &fixture,
        config.sparse_frontier() as usize,
        config.indexer_pool as usize,
    );
    for id in &wanted {
        assert!(frozen.iter().any(|c| &c.id == id), "unknown case {id}");
    }
    let _lease = production_lease();
    let ctx = MetalContext::new().expect("Metal context");
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let summary = |reference: &[Vec<f32>], other: &[Vec<f32>]| -> Value {
        let kls: Vec<f64> = reference
            .iter()
            .zip(other)
            .map(|(r, o)| kl_divergence(r, o))
            .collect();
        let flips: Vec<Value> = reference
            .iter()
            .zip(other)
            .enumerate()
            .filter_map(|(i, (r, o))| {
                let (a, b) = choice_regret(r, o);
                ((a, b) != (0.0, 0.0)).then(|| json!([i, a, b]))
            })
            .collect();
        let (worst_kl, at) = worst(kls.iter().copied());
        json!({"worst_kl": worst_kl, "worst_at": at, "mean_kl": kls.iter().sum::<f64>() / kls.len() as f64,
            "prompt_end_kl": kls[0], "flips": flips})
    };
    let mut cases = Vec::new();
    for case in frozen.iter().filter(|c| wanted.contains(&c.id)) {
        let turn2 = case.turn2();
        let capacity = turn2.len() + case.continuation.len() + 1;
        let run = |lineage: PackedLineage| -> Vec<Vec<f32>> {
            let mut s = session(&ctx, &weights, capacity, 512, lineage);
            let mut logits = vec![s.prefill_packed(&ctx, &turn2).unwrap()];
            for &token in &case.continuation {
                logits.push(s.forward(&ctx, token).unwrap());
            }
            logits
        };
        let exact = run(PackedLineage::Exact);
        let rounded = {
            let _scope = RoundExactActivations::set();
            run(PackedLineage::Exact)
        };
        let fast = run(PackedLineage::Fast);
        let row = json!({
            "id": case.id,
            "rounded_exact_vs_exact": summary(&exact, &rounded),
            "fast_vs_exact": summary(&exact, &fast),
            "fast_vs_rounded_exact": summary(&rounded, &fast),
        });
        eprintln!("{}: {row}", case.id);
        cases.push(row);
    }
    let document = json!({
        "schema": "glm53.activation_rounding_probe.v1", "rows_per_chunk": 512,
        "rounding": "Exact-lineage inputs to quantized-weight matrices (projections, KDA expansions, routed experts gate/up and down) rounded through half on a scratch copy; F32 router unrounded; weights F32-dequantized",
        "cases": cases,
    });
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&document).unwrap() + "\n",
    )
    .unwrap();
}

/// Fast stage families whose dense projections can take an F32-operand
/// tile (map #12 accuracy lane): Q8_0 KDA expansions, KDA beta/f_a/g_a, MLA
/// q_b and kv_a, the indexer and block 11's attention and shared expert;
/// Q6_K KDA q/k/v/o, MLA q_a and wo, the shared expert and the dense FFN.
/// The router and MLA absorption are F32 already.
fn f32_dense_families() -> Vec<super::super::packed::Stage> {
    use super::super::packed::Stage;
    vec![
        Stage::KdaExpand,
        Stage::KdaProjection,
        Stage::MlaProjection,
        Stage::IndexerProjection,
        Stage::SharedExpert,
        Stage::DenseFfn,
    ]
}

/// One F32-operand configuration of Fast: the stage families selected and
/// the tiles they may take.
pub(super) struct F32Selection {
    label: &'static str,
    stages: Vec<super::super::packed::Stage>,
    tiles: &'static [super::super::packed::F32Tile],
    /// Every call of a selected stage runs on F32 operands (a tile, or an
    /// F32 weight's own kernel); none stays half-staged.
    complete: bool,
    description: &'static str,
}

impl F32Selection {
    pub(super) fn scope(&self) -> super::super::packed::F32Stages {
        super::super::packed::F32Stages::with_tiles(&self.stages, self.tiles)
    }
}

/// Q8_0 only; every dense projection (Q8_0 and Q6_K); and every dense
/// projection plus the routed experts ("all": every quantized matrix
/// operand of Fast in F32).
pub(super) fn f32_selections() -> [F32Selection; 3] {
    use super::super::packed::{F32Tile, Stage};
    let mut all = f32_dense_families();
    all.push(Stage::RoutedExperts);
    [
        F32Selection {
            label: "fast_f32_q8",
            stages: f32_dense_families(),
            tiles: &[F32Tile::Q8_0],
            complete: false,
            description: "every Q8_0 dense projection on the F32-operand tile",
        },
        F32Selection {
            label: "fast_f32_dense",
            stages: f32_dense_families(),
            tiles: &[F32Tile::Q8_0, F32Tile::Q6K],
            complete: true,
            description: "every Q8_0 and Q6_K dense projection on F32-operand tiles (routed experts half-staged)",
        },
        F32Selection {
            label: "fast_f32_all",
            stages: all,
            tiles: &F32Tile::ALL,
            complete: true,
            description: "every dense projection and the routed experts (gate/up, SwiGLU input to down) on F32-operand tiles",
        },
    ]
}

/// A selection's census against the half-staged run's on the same tokens
/// and chunking ([`super::super::packed::F32Census`]): the same calls per
/// stage family and weight kind (coverage); in selected stages every call of
/// an enabled tile's kind took the tile and, for a complete selection, no
/// call stayed half-staged; unselected stages kept their paths; each enabled
/// tile ran. Returns the selection's census as JSON.
pub(super) fn assert_f32_census(
    label: &str,
    half: &super::super::packed::CensusCounts,
    selected: &super::super::packed::CensusCounts,
    selection: &F32Selection,
) -> Value {
    use super::super::packed::{CensusPath, F32Tile};
    let totals = |census: &super::super::packed::CensusCounts| {
        let mut totals = std::collections::BTreeMap::new();
        for ((stage, kind, _), n) in census {
            *totals.entry((*stage, kind.clone())).or_insert(0usize) += n;
        }
        totals
    };
    assert_eq!(
        totals(half),
        totals(selected),
        "{label}: call coverage differs"
    );
    assert!(
        half.keys().all(|(_, _, path)| *path != CensusPath::F32Tile),
        "{label}: the half-staged run used an F32 tile"
    );
    let tile_of = |kind: &str| match kind {
        "Q8_0" => Some(F32Tile::Q8_0),
        "Q6_K" => Some(F32Tile::Q6K),
        "experts" => Some(F32Tile::RoutedExperts),
        _ => None,
    };
    for ((stage, kind, path), n) in selected {
        let in_selection = selection.stages.contains(stage);
        let enabled =
            in_selection && tile_of(kind).is_some_and(|tile| selection.tiles.contains(&tile));
        if enabled {
            assert_eq!(
                *path,
                CensusPath::F32Tile,
                "{label}: {n} {stage:?} {kind} calls fell back"
            );
        } else if in_selection {
            assert_ne!(
                *path,
                CensusPath::F32Tile,
                "{label}: {stage:?} {kind} took a disabled tile"
            );
        } else {
            assert_eq!(
                half.get(&(*stage, kind.clone(), *path)),
                Some(n),
                "{label}: unselected {stage:?} {kind} changed path"
            );
        }
        if selection.complete && in_selection {
            assert_ne!(
                *path,
                CensusPath::Half,
                "{label}: {n} {stage:?} {kind} calls half-staged"
            );
        }
    }
    for tile in selection.tiles {
        assert!(
            selected
                .keys()
                .any(|(_, kind, path)| *path == CensusPath::F32Tile && tile_of(kind) == Some(*tile)),
            "{label}: no {tile:?} call ran"
        );
    }
    Value::Object(
        selected
            .iter()
            .map(|((stage, kind, path), n)| (format!("{stage:?} {kind} {path:?}"), json!(n)))
            .collect(),
    )
}

/// Map #12 accuracy probe (diagnostic; no bounds): on every frozen natural
/// case (`GLM53_PROBE_CASES` narrows) at 512 rows, Fast and each F32-operand
/// selection ([`f32_selections`]: Q8_0 only, every dense projection, and
/// every dense projection plus the routed experts), each against Exact on
/// the same frozen continuation: per case worst, mean and prompt-end KL and
/// flips with both regrets; across cases the equal-case mean of
/// per-position mean KL (the development summary) and the median and
/// largest worst-position KL. A census asserts every selected call took its
/// tile. Writes JSON to `GLM53_PROBE_OUT`.
#[test]
#[ignore = "map #12 F32-operand accuracy probe: loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, GLM53_PROBE_OUT and an idle GPU"]
fn reuse_natural_f32_operand_probe() {
    use super::super::packed::F32Census;
    let out = PathBuf::from(std::env::var("GLM53_PROBE_OUT").expect("GLM53_PROBE_OUT"));
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let fixture_bytes = std::fs::read(&fixture_path).unwrap();
    let fixture: Value = serde_json::from_slice(&fixture_bytes).unwrap();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let gguf = GgufFile::open(&path).unwrap();
    assert_eq!(fixture["artifact_layout"], artifact_layout(&gguf));
    let config = crate::glm5_next::Glm5NextConfig::from_gguf(&gguf).unwrap();
    let frozen = validate_fixture(
        &fixture,
        config.sparse_frontier() as usize,
        config.indexer_pool as usize,
    );
    let wanted: Vec<String> = match std::env::var("GLM53_PROBE_CASES") {
        Ok(list) => list.split(',').map(str::to_string).collect(),
        Err(_) => frozen.iter().map(|c| c.id.clone()).collect(),
    };
    for id in &wanted {
        assert!(frozen.iter().any(|c| &c.id == id), "unknown case {id}");
    }
    let _lease = production_lease();
    let ctx = MetalContext::new().expect("Metal context");
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let selections = f32_selections();
    let labels: Vec<&str> = std::iter::once("fast")
        .chain(selections.iter().map(|s| s.label))
        .collect();
    let mut cases = Vec::new();
    let mut per_arm: Vec<Vec<(f64, f64)>> = vec![Vec::new(); labels.len()];
    for case in frozen.iter().filter(|c| wanted.contains(&c.id)) {
        let turn2 = case.turn2();
        let capacity = turn2.len() + case.continuation.len() + 1;
        let run = |lineage: PackedLineage| -> Vec<Vec<f32>> {
            let mut s = session(&ctx, &weights, capacity, 512, lineage);
            let mut logits = vec![s.prefill_packed(&ctx, &turn2).unwrap()];
            for &token in &case.continuation {
                logits.push(s.forward(&ctx, token).unwrap());
            }
            logits
        };
        let exact = run(PackedLineage::Exact);
        let mut row = json!({"id": case.id, "prompt_tokens": turn2.len(),
            "positions": exact.len()});
        let mut fast_bits = None;
        let mut half_census = None;
        for (i, label) in labels.iter().enumerate() {
            let census = F32Census::begin();
            let (logits, census) = match i.checked_sub(1).map(|j| &selections[j]) {
                None => {
                    let logits = run(PackedLineage::Fast);
                    half_census = Some(census.take());
                    (logits, Value::Null)
                }
                Some(selection) => {
                    let _scope = selection.scope();
                    let logits = run(PackedLineage::Fast);
                    let counts = census.take();
                    let label = format!("{} {label}", case.id);
                    let half = half_census
                        .as_ref()
                        .expect("the half-staged arm runs first");
                    (logits, assert_f32_census(&label, half, &counts, selection))
                }
            };
            let drift = Drift::measure(&format!("{} {label}", case.id), &exact, &logits);
            drift.print(&format!("{} {label} vs Exact", case.id));
            let mut value = drift.json();
            value["prompt_end_kl"] = json!(drift.kls[0]);
            value["census"] = census;
            match &fast_bits {
                None => fast_bits = Some(logit_bits(&logits)),
                Some(bits) => value["differs_from_fast"] = json!(logit_bits(&logits) != *bits),
            }
            row[*label] = value;
            per_arm[i].push((drift.mean_kl, drift.worst_kl));
        }
        cases.push(row);
    }
    assert!(!cases.is_empty(), "no case was evaluated");
    let summary: serde_json::Map<String, Value> = labels
        .iter()
        .zip(&per_arm)
        .map(|(label, values)| {
            let mut worst: Vec<f64> = values.iter().map(|v| v.1).collect();
            worst.sort_by(f64::total_cmp);
            let median = if worst.len() % 2 == 1 {
                worst[worst.len() / 2]
            } else {
                (worst[worst.len() / 2 - 1] + worst[worst.len() / 2]) / 2.0
            };
            let mean_of_means = values.iter().map(|v| v.0).sum::<f64>() / values.len() as f64;
            eprintln!(
                "{label}: equal-case mean of mean KL {mean_of_means:.4e}; worst KL median {median:.4e}, max {:.4e}",
                worst[worst.len() - 1]
            );
            (
                label.to_string(),
                json!({"equal_case_mean_of_mean_kl": mean_of_means,
                    "median_worst_kl": median, "max_worst_kl": worst[worst.len() - 1]}),
            )
        })
        .collect();
    let mut arms = serde_json::Map::new();
    arms.insert("fast".into(), json!("half-staged tiles"));
    for selection in &selections {
        arms.insert(
            selection.label.into(),
            json!({"description": selection.description,
                "stages": format!("{:?}", selection.stages), "tiles": format!("{:?}", selection.tiles)}),
        );
    }
    let document = json!({
        "schema": "glm53.f32_operand_probe.v2", "rows_per_chunk": 512,
        "reference": "Exact cold on the same frozen continuation",
        "fixture": {"path": FIXTURE, "sha256": sha256_hex(&fixture_bytes)},
        "artifact_layout": fixture["artifact_layout"],
        "evaluator_commit": std::env::var("GLM53_PROBE_COMMIT").ok(),
        "cases_evaluated": wanted, "all_frozen_cases": wanted.len() == frozen.len(),
        "arms": arms, "summary": summary, "cases": cases,
    });
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&document).unwrap() + "\n",
    )
    .unwrap();
}

/// F32-operand tiles keep Fast's chunk identities: with every dense
/// projection and the routed experts on them, 512- and 128-row chunkings
/// agree bitwise, as do 64 and 97 rows (the pairs the half-staged Fast
/// already holds), at the prompt end and through decode; every selected call
/// took its tile at each chunking; and the selection takes effect (logits
/// differ from half-staged Fast).
#[test]
#[ignore = "map #12 F32-operand chunk identities: loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF and an idle GPU"]
fn fast_f32_operands_keep_chunk_identities() {
    use super::super::packed::F32Census;
    let _lease = production_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let gguf = GgufFile::open(&path).unwrap();
    let ctx = MetalContext::new().expect("Metal context");
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let tokens: Vec<u32> = tokenizer
        .encode(&long_qualification_text(), false)
        .unwrap()
        .into_iter()
        .map(|t| t as u32)
        .collect();
    let (prompt, continuation) = (&tokens[..600], &tokens[600..608]);
    let run = |rows: usize| -> Vec<Vec<f32>> {
        let mut s = session(&ctx, &weights, 700, rows, PackedLineage::Fast);
        let mut logits = vec![s.prefill_packed(&ctx, prompt).unwrap()];
        for &token in continuation {
            logits.push(s.forward(&ctx, token).unwrap());
        }
        logits
    };
    let [_, _, all] = f32_selections();
    let census = F32Census::begin();
    let mut half_512 = None;
    let f32_runs: Vec<(usize, Vec<Vec<f32>>)> = [512, 128, 64, 97]
        .map(|rows| {
            let half = run(rows);
            let half_counts = census.take();
            if rows == 512 {
                half_512 = Some(half);
            }
            let _scope = all.scope();
            let logits = run(rows);
            let counts = census.take();
            let value = assert_f32_census(&format!("{rows} rows"), &half_counts, &counts, &all);
            eprintln!("{rows} rows: census {value}");
            (rows, logits)
        })
        .into();
    let half_512 = half_512.expect("512 rows ran");
    let bits = |rows: usize| logit_bits(&f32_runs.iter().find(|(r, _)| *r == rows).unwrap().1);
    assert_eq!(bits(512), bits(128), "512 vs 128 rows");
    assert_eq!(bits(64), bits(97), "64 vs 97 rows");
    assert_ne!(
        bits(512),
        logit_bits(&half_512),
        "the F32 selection took effect"
    );
    for (rows, logits) in &f32_runs {
        for (i, l) in logits.iter().enumerate() {
            assert_finite(&format!("rows {rows} position {i}"), l);
        }
    }
}

/// Release timing of the F32-operand selections (map #12 accuracy lane):
/// Fast, every dense projection on F32-operand tiles, and those plus the
/// routed experts, in A-B-C-C-B-A order twice after a warm-up of each: a
/// fresh 2,048-token prompt in 512-row chunks; 1-, 17- and 64-token
/// suffixes ending at 2,048 after a reused same-arm prefix (the short spans
/// a reused conversation prefills); and a 64-token suffix 512 tokens past
/// the sparse frontier (sparse attention active). Reports per-arm means to
/// `GLM53_PROBE_OUT`. Diagnostic; no bounds.
#[test]
#[ignore = "diagnostic timing: GLM53_GGUF, GLM53_PROBE_OUT, release build, no MTL_DEBUG_LAYER; loads 109.5 GiB under production lease"]
fn fast_f32_operand_prefill_cost() {
    if cfg!(debug_assertions) {
        panic!("timing requires --release");
    }
    let out = PathBuf::from(std::env::var("GLM53_PROBE_OUT").expect("GLM53_PROBE_OUT"));
    let _lease = super::perf_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let gguf = GgufFile::open(&path).unwrap();
    let ctx = MetalContext::new().expect("Metal context");
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let tokens: Vec<u32> = tokenizer
        .encode(&long_qualification_text(), false)
        .unwrap()
        .into_iter()
        .map(|t| t as u32)
        .collect();
    const PROMPT: usize = 2048;
    // (suffix tokens, end position)
    let frontier = weights.config.sparse_frontier() as usize;
    let suffixes = [
        (1, PROMPT),
        (17, PROMPT),
        (64, PROMPT),
        (64, frontier + 512),
    ];
    assert!(tokens.len() >= frontier + 512);
    let [_, dense, all] = f32_selections();
    let arms: [(&str, Option<&F32Selection>); 3] = [
        ("half", None),
        (dense.label, Some(&dense)),
        (all.label, Some(&all)),
    ];
    // Per measurement: the fresh prompt's ms, then each suffix's.
    let time = |selection: Option<&F32Selection>| -> Vec<f64> {
        let _scope = selection.map(F32Selection::scope);
        let mut s = session(&ctx, &weights, PROMPT + 8, 512, PackedLineage::Fast);
        let started = std::time::Instant::now();
        let logits = s.prefill_packed(&ctx, &tokens[..PROMPT]).unwrap();
        let mut spans = vec![started.elapsed().as_secs_f64() * 1e3];
        assert_finite("timing logits", &logits);
        drop(s);
        for (n, end) in suffixes {
            let mut s = session(&ctx, &weights, end + 8, 512, PackedLineage::Fast);
            s.prefill_packed(&ctx, &tokens[..end - n]).unwrap();
            let started = std::time::Instant::now();
            let logits = s.prefill_packed(&ctx, &tokens[end - n..end]).unwrap();
            spans.push(started.elapsed().as_secs_f64() * 1e3);
            assert_finite("suffix logits", &logits);
        }
        spans
    };
    for (_, selection) in &arms {
        time(*selection);
    }
    let mut runs: Vec<Vec<Vec<f64>>> = vec![Vec::new(); arms.len()];
    for _ in 0..2 {
        for i in [0, 1, 2, 2, 1, 0] {
            runs[i].push(time(arms[i].1));
        }
    }
    let labels = std::iter::once(format!("fresh {PROMPT}"))
        .chain(suffixes.map(|(n, end)| format!("{n}-token suffix after {}", end - n)));
    let rows: Vec<Value> = labels
        .enumerate()
        .map(|(i, span)| {
            let mut row = json!({"span": span});
            let half_mean = runs[0].iter().map(|t| t[i]).sum::<f64>() / runs[0].len() as f64;
            for ((label, _), arm) in arms.iter().zip(&runs) {
                let ms: Vec<f64> = arm.iter().map(|t| t[i]).collect();
                let mean = ms.iter().sum::<f64>() / ms.len() as f64;
                row[*label] = json!({"ms": ms, "mean_ms": mean,
                    "relative_to_half": mean / half_mean - 1.0});
            }
            row
        })
        .collect();
    let document = json!({
        "schema": "glm53.f32_operand_cost.v1", "rows_per_chunk": 512,
        "order": "warm-up A,B,C; then (A,B,C,C,B,A) x 2; a fresh session per measurement (allocation excluded)",
        "arms": {"half": "Fast", "fast_f32_dense": dense.description, "fast_f32_all": all.description},
        "rows": rows,
    });
    eprintln!("{document}");
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&document).unwrap() + "\n",
    )
    .unwrap();
}

/// Map #12 diagnostic (no bounds): does the drift left under the all-F32
/// selection coincide with discrete routing changes? On frozen cases whose
/// second-turn prompt fits one 512-row chunk (`GLM53_PROBE_CASES`; default
/// H1, H3, H6), the prompt is prefilled under Exact, Fast and the all-F32
/// selection, and the expert set each MoE block routed every prompt row to
/// is compared with Exact's: rows changed per block, the first block with a
/// change, and the prompt-end KL. Writes JSON to `GLM53_PROBE_OUT`.
#[test]
#[ignore = "map #12 route divergence: loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, GLM53_PROBE_OUT and an idle GPU"]
fn reuse_natural_f32_route_divergence() {
    let out = PathBuf::from(std::env::var("GLM53_PROBE_OUT").expect("GLM53_PROBE_OUT"));
    let wanted: Vec<String> = std::env::var("GLM53_PROBE_CASES")
        .unwrap_or_else(|_| "H1-short-chat,H3-tool-dense,H6-max-sampled-tool".into())
        .split(',')
        .map(str::to_string)
        .collect();
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let fixture: Value = serde_json::from_slice(&std::fs::read(&fixture_path).unwrap()).unwrap();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let gguf = GgufFile::open(&path).unwrap();
    assert_eq!(fixture["artifact_layout"], artifact_layout(&gguf));
    let config = crate::glm5_next::Glm5NextConfig::from_gguf(&gguf).unwrap();
    let frozen = validate_fixture(
        &fixture,
        config.sparse_frontier() as usize,
        config.indexer_pool as usize,
    );
    for id in &wanted {
        assert!(frozen.iter().any(|c| &c.id == id), "unknown case {id}");
    }
    let _lease = production_lease();
    let ctx = MetalContext::new().expect("Metal context");
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let top_k = weights.config.expert_used_count as usize;
    let [_, _, all] = f32_selections();
    let mut cases = Vec::new();
    for case in frozen.iter().filter(|c| wanted.contains(&c.id)) {
        let turn2 = case.turn2();
        assert!(turn2.len() <= 512, "{}: one 512-row chunk", case.id);
        let run = |lineage: PackedLineage| -> (Vec<f32>, Vec<(usize, Vec<i32>)>) {
            let mut s = session(&ctx, &weights, turn2.len() + 1, 512, lineage);
            let logits = s.prefill_packed(&ctx, &turn2).unwrap();
            let routes = s
                .packed
                .as_ref()
                .expect("packed scratch")
                .route_ids()
                .into_iter()
                .map(|(block, ids)| {
                    let ids = super::super::read_i32(ids).unwrap();
                    (block, ids[..turn2.len() * top_k].to_vec())
                })
                .collect();
            (logits, routes)
        };
        let (exact_logits, exact_routes) = run(PackedLineage::Exact);
        let fast = run(PackedLineage::Fast);
        let f32_all = {
            let _scope = all.scope();
            run(PackedLineage::Fast)
        };
        let compare = |(logits, routes): &(Vec<f32>, Vec<(usize, Vec<i32>)>)| -> Value {
            let mut first = None;
            let mut total = 0usize;
            let per_block: Vec<Value> = exact_routes
                .iter()
                .zip(routes)
                .map(|((block, a), (other, b))| {
                    assert_eq!(block, other);
                    let changed = a
                        .chunks_exact(top_k)
                        .zip(b.chunks_exact(top_k))
                        .filter(|(a, b)| {
                            let (mut a, mut b) = (a.to_vec(), b.to_vec());
                            a.sort_unstable();
                            b.sort_unstable();
                            a != b
                        })
                        .count();
                    if changed > 0 && first.is_none() {
                        first = Some(*block);
                    }
                    total += changed;
                    json!([block, changed])
                })
                .collect();
            json!({"prompt_end_kl": kl_divergence(&exact_logits, logits),
                "route_set_rows_changed": total, "first_block_changed": first,
                "rows_changed_by_block": per_block})
        };
        let row = json!({"id": case.id, "prompt_tokens": turn2.len(),
            "moe_blocks": exact_routes.len(),
            "fast": compare(&fast), "fast_f32_all": compare(&f32_all)});
        eprintln!(
            "{}: Fast {} rows changed (first block {}), prompt-end KL {:.3e}; all-F32 {} rows changed (first block {}), prompt-end KL {:.3e}",
            case.id,
            row["fast"]["route_set_rows_changed"],
            row["fast"]["first_block_changed"],
            row["fast"]["prompt_end_kl"].as_f64().unwrap(),
            row["fast_f32_all"]["route_set_rows_changed"],
            row["fast_f32_all"]["first_block_changed"],
            row["fast_f32_all"]["prompt_end_kl"].as_f64().unwrap()
        );
        cases.push(row);
    }
    let document = json!({
        "schema": "glm53.f32_route_divergence.v1", "rows_per_chunk": 512,
        "reference": "Exact prompt prefill", "selection": all.description,
        "evaluator_commit": std::env::var("GLM53_PROBE_COMMIT").ok(),
        "cases": cases,
    });
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&document).unwrap() + "\n",
    )
    .unwrap();
}
