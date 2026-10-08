//! Map #12 on natural trajectories (design jam with cx 01a10cc, 2026-10-07).
//! Two phases, so every continuation is frozen before Fast is evaluated:
//!
//! 1. [`reuse_natural_generate`] (Exact lineage only) renders each
//!    preregistered case's first turn, generates the model's own turn
//!    (greedy, or seeded release sampling for the max-effort case), renders
//!    the conversation again with the next turn, and records an Exact greedy
//!    continuation. It writes the fixture (`scripts/reference/glm53/
//!    reuse-natural-v1.json`), which is committed before phase 2 runs.
//!    Rejections (token cap, grammar, round trip, geometry) are recorded;
//!    replacements come only from each case's ordered variant list.
//! 2. [`reuse_natural_evaluate`] replays the frozen tokens. Exact cold is the
//!    reference R. Exact warm must equal R bitwise (logits, and persistent
//!    state at the join and at the end) at both chunk schedules, and Exact
//!    cold at 97 rows must too on the frontier cases. Fast cold and warm at
//!    512 and 97 rows must meet the Fast policy against R, and Fast warm
//!    must meet the frozen reuse gate against Fast cold at the same schedule.
//!
//! Emitted and consumed tokens are distinct, as in serve: a sampled stop is
//! emitted but never forwarded, and the rendered next turn supplies it.
//!
//! [`packed_lineage_prefill_cost`] prices the Exact lineage against Fast.

use super::*;
use crate::glm5_next_chat::{
    self as chat, Effort, Message, RenderOptions, ToolCall, ToolDefinition,
};
use crate::sampling::{Sampler, SamplingConfig};
use serde_json::{Value, json};

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
        json!({"mode": match self { Self::Greedy { .. } => "greedy", Self::Sampled { .. } => "sampled" },
            "cap": self.cap(), "temperature": s.temperature, "top_k": s.top_k,
            "top_p": s.top_p, "min_p": s.min_p, "seed": s.seed, "sampler": "qwen_llm::sampling v1"})
    }
}

/// What the generated turn must look like, and where the conversation must sit
/// relative to the sparse frontier.
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

/// One preregistered prompt. `pad` fits the first turn into a token window
/// below the frontier (`frontier - hi ..= frontier - lo`) by trimming a
/// leading passage.
struct Variant {
    label: &'static str,
    system: Option<&'static str>,
    user: String,
    pad: Option<(usize, usize)>,
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

/// The first `chars` characters of `source`, cut at a line boundary.
fn lines_prefix(source: &str, chars: usize) -> &str {
    match source[..chars.min(source.len())].rfind('\n') {
        Some(end) => &source[..end],
        None => source,
    }
}

/// The preregistered cohort (natural-v1). Fixed before any generation; a
/// case's later variants are used only when an earlier one is rejected.
fn cohort() -> Vec<Case> {
    let low = |cap| (Effort::Low, Generation::Greedy { cap });
    let variant =
        |label, user: String, next: &str, (effort, generation): (Effort, Generation)| Variant {
            label,
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
    let passages = &passages["[gMASK]<sop>".len()..];
    let weather_results = "Lisbon forecast. Day 1: 21C, sunny, light wind from the north-west, humidity 55 percent. Day 2: 19C, light rain in the afternoon, wind 20 km/h, humidity 78 percent. ";
    vec![
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
                .map(|(label, pad)| Variant {
                    label,
                    system: None,
                    user: format!("{passages}\n\nAfter reading the passages above: what's the weather in Lisbon for the next two days? Use the tool."),
                    pad: Some(pad),
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
                    label: "kyoto",
                    system: Some("You are a concise travel assistant."),
                    user: "I'm planning a weekend in Kyoto. Check the weather for Saturday and Sunday, then suggest one indoor and one outdoor activity.".into(),
                    pad: None,
                    next: "Kyoto forecast. Saturday: 24C, clear skies, light breeze. Sunday: 17C, rain all day, heavy in the afternoon.".into(),
                    effort: Effort::Max,
                    generation: Generation::Sampled { seed, cap: 1536 },
                })
                .collect(),
        },
    ]
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
        .unwrap()
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect()
}

/// Exact session of `capacity` positions with `rows`-row chunks.
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

/// The model's own turn after `prompt`: emitted tokens, the stop (if any),
/// and how many emitted tokens were consumed (forwarded). `Err` on the cap.
fn generate_turn(
    ctx: &MetalContext,
    weights: &Glm5NextWeights,
    prompt: &[u32],
    generation: Generation,
) -> std::result::Result<(Vec<u32>, Option<u32>), String> {
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
            return Err(format!("token cap {cap} reached without a stop"));
        }
        logits = s.forward(ctx, token).unwrap();
    }
}

/// Reasoning, content and tool calls of a generated turn's text.
fn parse_turn(
    text: &str,
    tools: &[ToolDefinition],
) -> std::result::Result<(String, String, Vec<ToolCall>), String> {
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

/// Phase 1. Writes the frozen fixture to `GLM53_REUSE_NATURAL_OUT`; records
/// the producer from `GLM53_PRODUCER_COMMIT` and `GLM53_PRODUCER_DIRTY`.
#[test]
#[ignore = "generator (map #12 natural trajectories): loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, GLM53_REUSE_NATURAL_OUT, GLM53_PRODUCER_COMMIT and an idle GPU"]
fn reuse_natural_generate() {
    let out =
        PathBuf::from(std::env::var("GLM53_REUSE_NATURAL_OUT").expect("GLM53_REUSE_NATURAL_OUT"));
    let commit = std::env::var("GLM53_PRODUCER_COMMIT").expect("GLM53_PRODUCER_COMMIT");
    let dirty = std::env::var("GLM53_PRODUCER_DIRTY").as_deref() == Ok("1");
    let _lease = production_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
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
    let mut cases = Vec::new();
    for case in cohort() {
        let definitions: Vec<ToolDefinition> = case
            .tools
            .iter()
            .map(|t| ToolDefinition::from_value(t).unwrap())
            .collect();
        let mut attempts = Vec::new();
        let mut accepted = None;
        for variant in &case.variants {
            let started = std::time::Instant::now();
            let first_messages = |user: String| {
                let mut messages = Vec::new();
                if let Some(system) = variant.system {
                    messages.push(Message::System(system.into()));
                }
                messages.push(Message::User(user));
                messages
            };
            // Fit the first turn into its window by trimming leading text.
            let mut user = variant.user.clone();
            if let Some((lo, hi)) = variant.pad {
                let (low, high) = (frontier - hi, frontier - lo);
                let mut fitted = None;
                for _ in 0..48 {
                    let n = encode(&render(
                        &first_messages(user.clone()),
                        &definitions,
                        variant.effort,
                    ))
                    .len();
                    if (low..=high).contains(&n) {
                        fitted = Some(user.clone());
                        break;
                    }
                    let excess = n as isize - ((low + high) / 2) as isize;
                    let mut cut = (excess * 3).clamp(-(user.len() as isize), user.len() as isize);
                    if cut < 0 {
                        cut = 0; // never longer than the preregistered text
                    }
                    let mut start = cut as usize;
                    while !user.is_char_boundary(start) {
                        start += 1;
                    }
                    user = user[start..].to_string();
                }
                user = fitted.unwrap_or_else(|| {
                    panic!("{} {}: padding did not converge", case.id, variant.label)
                });
            }
            let turn1_messages = first_messages(user);
            let turn1 = encode(&render(&turn1_messages, &definitions, variant.effort));
            let reject = |attempts: &mut Vec<Value>, outcome: &str, detail: String| {
                eprintln!(
                    "{} {}: rejected ({outcome}): {detail}",
                    case.id, variant.label
                );
                attempts
                    .push(json!({"variant": variant.label, "outcome": outcome, "detail": detail}));
            };
            let (emitted, stop) = match generate_turn(&ctx, &weights, &turn1, variant.generation) {
                Ok(turn) => turn,
                Err(detail) => {
                    reject(&mut attempts, "cap", detail);
                    continue;
                }
            };
            // A stop is emitted, never forwarded; everything before it is consumed.
            let consumed: Vec<u32> = match stop {
                Some(_) => emitted[..emitted.len() - 1].to_vec(),
                None => emitted.clone(),
            };
            let consumed_i32: Vec<i32> = consumed.iter().map(|&t| t as i32).collect();
            let text = tokenizer.decode(&consumed_i32);
            let (reasoning, content, calls) = match parse_turn(&text, &definitions) {
                Ok(parsed) => parsed,
                Err(detail) => {
                    reject(&mut attempts, "grammar", detail);
                    continue;
                }
            };
            let wants_call = case.shape.tools();
            if wants_call && (calls.len() != 1 || stop != Some(OBSERVATION)) {
                reject(
                    &mut attempts,
                    "grammar",
                    format!(
                        "{} calls, stop {stop:?}; one call ending at <|observation|> required",
                        calls.len()
                    ),
                );
                continue;
            }
            if !wants_call && !calls.is_empty() {
                reject(
                    &mut attempts,
                    "grammar",
                    format!("{} unexpected calls", calls.len()),
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
                reject(
                    &mut attempts,
                    "round_trip",
                    format!(
                        "the rendered second turn does not extend the consumed history (first difference at {at:?}; join {join})"
                    ),
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
                reject(&mut attempts, "geometry", detail);
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
            attempts.push(json!({"variant": variant.label, "outcome": "accepted"}));
            accepted = Some(json!({
                "id": case.id, "shape": format!("{:?}", case.shape), "variant": variant.label,
                "effort": variant.effort.as_str(), "generation": variant.generation.describe(),
                "conversation": doc,
                "turn1": ids(&turn1), "emitted": ids(&emitted),
                "stop": stop, "consumed": consumed.len(),
                "suffix": ids(&turn2[join..]),
                "continuation": ids(&continuation), "continuation_stop": continuation_stop,
                "join": join, "second_turn": turn2.len(),
            }));
            break;
        }
        let mut record = accepted.unwrap_or_else(|| {
            panic!(
                "{}: every preregistered variant was rejected: {attempts:?}",
                case.id
            )
        });
        record["attempts"] = json!(attempts);
        cases.push(record);
    }
    let qual: Value = serde_json::from_str(QUAL_V1_MANIFEST).unwrap();
    let fixture = json!({
        "name": "glm53-reuse-natural-v1",
        "purpose": "Map #12 natural trajectories: the model's own turns (Exact lineage) and Exact greedy continuations, frozen before any Fast evaluation; replayed by glm5_next_metal::tests::natural::reuse_natural_evaluate.",
        "artifact": qual["artifact"],
        "retained_weight_bytes": weights.retained_bytes,
        "frontier": frontier, "pool": pool, "continuation_max": CONTINUATION,
        "producer": {"test": "glm5_next_metal::tests::natural::reuse_natural_generate",
            "commit": commit, "dirty": dirty, "lineage": "exact", "prefill_rows": 512,
            "renderer": chat::RENDERER, "template_sha256": chat::GGUF_TEMPLATE_SHA256},
        "token_encoding": "space-separated token ids",
        "cases": cases,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&fixture).unwrap() + "\n").unwrap();
    eprintln!("wrote {}", out.display());
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
    for (position, (r, f)) in reference.iter().zip(fast).enumerate() {
        assert_finite(&format!("{label} reference {position}"), r);
        assert_finite(&format!("{label} fast {position}"), f);
        let kl = kl_divergence(r, f);
        let (r0, r1) = choice_regret(r, f);
        if !within(kl, FAST_KL) || !(r0 <= FAST_REGRET && r1 <= FAST_REGRET) {
            failures.push(format!("{label} position {position}: KL(exact||fast) {kl:.3e} (bound {FAST_KL:e}), regret {r0:.3}/{r1:.3} (bound {FAST_REGRET})"));
        }
        if (r0, r1) != (0.0, 0.0) {
            flips.push(json!([position, r0, r1]));
        }
        kls.push(kl);
    }
    let (worst_kl, at) = worst(kls.iter().copied());
    let mean = kls.iter().sum::<f64>() / kls.len() as f64;
    let pass = within(worst_kl, FAST_KL)
        && flips.iter().all(|f| {
            f[1].as_f64().unwrap() <= f64::from(FAST_REGRET)
                && f[2].as_f64().unwrap() <= f64::from(FAST_REGRET)
        });
    eprintln!(
        "  {label}: worst KL(exact||fast) {worst_kl:.3e} at {at}, mean {mean:.3e}, flips {flips:?}, pass {pass}"
    );
    json!({"worst_kl": worst_kl, "worst_at": at, "mean_kl": mean, "flips": flips, "kl": kls, "pass": pass})
}

/// Phase 2. Reads the fixture (`GLM53_REUSE_NATURAL` or the committed path)
/// and writes the report to `GLM53_REUSE_NATURAL_REPORT`.
#[test]
#[ignore = "map #12 natural-trajectory qualification: loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, GLM53_REUSE_NATURAL_REPORT, the committed fixture and an idle GPU"]
fn reuse_natural_evaluate() {
    let report_path = PathBuf::from(
        std::env::var("GLM53_REUSE_NATURAL_REPORT").expect("GLM53_REUSE_NATURAL_REPORT"),
    );
    let fixture_path = std::env::var_os("GLM53_REUSE_NATURAL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE));
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(&fixture_path).unwrap_or_else(|e| panic!("{}: {e}", fixture_path.display())),
    )
    .unwrap();
    let only: Option<Vec<String>> = std::env::var("GLM53_REUSE_NATURAL_CASES")
        .ok()
        .map(|v| v.split(',').map(str::to_string).collect());
    let _lease = production_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    assert_eq!(
        fixture["retained_weight_bytes"].as_u64(),
        Some(weights.retained_bytes),
        "the fixture was generated from different weights"
    );
    let frontier = weights.config.sparse_frontier() as usize;
    assert_eq!(fixture["frontier"].as_u64(), Some(frontier as u64));
    let mut failures = Vec::new();
    let mut reports = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        if only
            .as_ref()
            .is_some_and(|only| !only.iter().any(|o| o == id))
        {
            continue;
        }
        let turn1 = parse_ids(&case["turn1"]);
        let emitted = parse_ids(&case["emitted"]);
        let consumed = &emitted[..case["consumed"].as_u64().unwrap() as usize];
        let suffix = parse_ids(&case["suffix"]);
        let continuation = parse_ids(&case["continuation"]);
        let join = turn1.len() + consumed.len();
        let turn2: Vec<u32> = [turn1.as_slice(), consumed, suffix.as_slice()].concat();
        let positions = continuation.len() + 1;
        let capacity = turn2.len() + continuation.len() + 1;
        let frontier_case = matches!(case["shape"].as_str(), Some("ToolAcross" | "ChatSparse"));
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
            for &token in &continuation {
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
            s.prefill_packed(&ctx, &turn1).unwrap();
            for &token in consumed {
                s.forward(&ctx, token).unwrap();
            }
            assert_eq!(s.position(), join);
            let first = s.prefill_packed(&ctx, &suffix).unwrap();
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
        reports.push(json!({"id": id, "variant": case["variant"], "first": turn1.len(), "consumed": consumed.len(),
            "suffix": suffix.len(), "join": join, "second_turn": turn2.len(), "positions": positions,
            "exact_reference_ms": reference.ms, "exact": exact, "fast": fast}));
    }
    let report = json!({
        "schema": "glm53.reuse_natural_report.v1",
        "fixture": fixture_path.display().to_string(),
        "fixture_producer": fixture["producer"],
        "policy": {"fast_vs_exact": {"kl_reference_fast": FAST_KL, "regret": FAST_REGRET},
            "reuse_gate": {"kl_both_ways": 2e-2, "regret": 0.2}, "exact": "bitwise logits and persistent state"},
        "schedules": SCHEDULES, "cases": reports, "failures": failures,
    });
    std::fs::write(
        &report_path,
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .unwrap();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Exact lineage priced against Fast: prompt prefill wall (allocation
/// excluded; final logits included) at 512, 2048 and 4096 tokens, and a
/// 64-token suffix after a reused prefix, in A-B-B-A order per length.
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
    // Warm-up outside the measurement: first GPU use of the weights.
    {
        let mut s = session(&ctx, &weights, 1025, 512, PackedLineage::Fast);
        s.prefill_packed(&ctx, &tokens[..1024]).unwrap();
    }
    let mut rows = Vec::new();
    for length in [512usize, 2048, 4096] {
        assert!(tokens.len() >= length);
        for (order, lineage) in [
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
            let cold_ms = started.elapsed().as_secs_f64() * 1e3;
            drop(s);
            let mut s = session(&ctx, &weights, length + 1, 512, lineage);
            s.prefill_packed(&ctx, &tokens[..length - SUFFIX]).unwrap();
            let started = std::time::Instant::now();
            s.prefill_packed(&ctx, &tokens[length - SUFFIX..length])
                .unwrap();
            let suffix_ms = started.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "{length} tokens, {lineage:?} (slot {order}): cold {cold_ms:.0} ms ({:.1} tok/s); {SUFFIX}-token suffix after {} reused: {suffix_ms:.0} ms",
                length as f64 / (cold_ms / 1e3),
                length - SUFFIX
            );
            rows.push(
                json!({"length": length, "lineage": format!("{lineage:?}"), "slot": order,
                "cold_ms": cold_ms, "cold_tok_s": length as f64 / (cold_ms / 1e3),
                "suffix_tokens": SUFFIX, "suffix_ms": suffix_ms}),
            );
        }
    }
    let document = json!({
        "schema": "glm53.packed_lineage_prefill_cost.v1",
        "method": "per length, Fast-Exact-Exact-Fast; fresh session per measurement (allocation excluded); prompt prefill including final logits; suffix measured after a same-lineage prefix prefill; 512-row chunks; one warm-up prefill before all rows",
        "prompt": "long_qualification_text (teacher-forced)",
        "rows": rows,
    });
    std::fs::write(&out, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
}
