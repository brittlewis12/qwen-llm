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
/// The Fast policy of `packed_prefill_matches_serial_on_long_context`
/// (`LOGIT_KL`, `TOP1_REGRET`): KL(reference || fast) per position, and a
/// top-1 flip only to a near tie, both regrets within the bound.
const FAST_KL: f64 = 2e-2;
const FAST_REGRET: f32 = 0.2;
/// Released `<|observation|>`: a turn that called tools ends with it.
const OBSERVATION: u32 = 154_829;
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
enum Generation {
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
    fn cap(self) -> usize {
        match self {
            Self::Greedy { cap } | Self::Sampled { cap, .. } => cap,
        }
    }

    fn sampling(self) -> SamplingConfig {
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

    fn describe(self) -> Value {
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

fn weather_tool() -> Value {
    json!({"name": "get_weather", "description": "Get the weather forecast for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "days": {"type": "integer", "description": "Forecast days (1-7)"}},
            "required": ["city"]}})
}

fn currency_tool() -> Value {
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

fn render(messages: &[Message], tools: &[ToolDefinition], effort: Effort) -> String {
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

fn ids(tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_ids(value: &Value) -> Vec<u32> {
    value
        .as_str()
        .expect("token ids are a string")
        .split_whitespace()
        .map(|t| t.parse().expect("token id"))
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
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
fn artifact_layout(gguf: &GgufFile) -> Value {
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
fn session<'w>(
    ctx: &MetalContext,
    weights: &'w Glm5NextWeights,
    capacity: usize,
    rows: usize,
    lineage: PackedLineage,
) -> Glm5NextSession<'w> {
    let mut session =
        Glm5NextSession::with_prefill_rows(ctx, weights, capacity, rows.min(capacity)).unwrap();
    session.set_packed_lineage(lineage);
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
fn parse_turn(
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

/// Fast policy against the Exact reference; records violations.
fn fast_policy(
    label: &str,
    reference: &[Vec<f32>],
    fast: &[Vec<f32>],
    failures: &mut Vec<String>,
) -> Value {
    assert_eq!(reference.len(), fast.len(), "{label}: position count");
    let mut kls = Vec::with_capacity(fast.len());
    let mut flips = Vec::new();
    let mut pass = true;
    for (position, (r, f)) in reference.iter().zip(fast).enumerate() {
        assert_finite(&format!("{label} reference {position}"), r);
        assert_finite(&format!("{label} fast {position}"), f);
        let kl = kl_divergence(r, f);
        let (r0, r1) = choice_regret(r, f);
        if !within(kl, FAST_KL) || !(r0 <= FAST_REGRET && r1 <= FAST_REGRET) {
            pass = false;
            failures.push(format!("{label} position {position}: KL(exact||fast) {kl:.3e} (bound {FAST_KL:e}), regret {r0:.3}/{r1:.3} (bound {FAST_REGRET})"));
        }
        if (r0, r1) != (0.0, 0.0) {
            flips.push(json!([position, r0, r1]));
        }
        kls.push(kl);
    }
    let (worst_kl, at) = worst(kls.iter().copied());
    let mean = kls.iter().sum::<f64>() / kls.len() as f64;
    eprintln!(
        "  {label}: worst KL(exact||fast) {worst_kl:.3e} at {at}, mean {mean:.3e}, flips {flips:?}, pass {pass}"
    );
    json!({"worst_kl": worst_kl, "worst_at": at, "mean_kl": mean, "flips": flips, "kl": kls, "pass": pass})
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
            let cold_policy = fast_policy(
                &format!("{id} fast cold {rows}"),
                &reference.logits,
                &fast_cold.logits,
                &mut failures,
            );
            let warm_policy = fast_policy(
                &format!("{id} fast warm {rows}"),
                &reference.logits,
                &fast_warm.logits,
                &mut failures,
            );
            let (kl, regret, agree, bitwise) = reuse_gate(
                &format!("{id} fast warm vs cold {rows}"),
                positions,
                &fast_cold.logits,
                &fast_warm.logits,
                &mut failures,
            );
            fast.insert(rows.to_string(), json!({
                "cold": cold_policy, "warm": warm_policy,
                "reuse_gate": {"worst_kl": kl, "worst_regret": regret, "top1_agree": agree, "bitwise": bitwise},
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
        "schema": "glm53.reuse_natural_report.v1",
        "partial": partial,
        "fixture": fixture_path.display().to_string(),
        "fixture_sha256": sha256_hex(&fixture_bytes),
        "fixture_producer": fixture["producer"],
        "evaluator_commit": evaluator,
        "policy": {"fast_vs_exact": {"kl_reference_fast": FAST_KL, "regret": FAST_REGRET},
            "reuse_gate": {"kl_both_ways": 2e-2, "regret": 0.2},
            "exact": "bitwise logits and persistent state"},
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
