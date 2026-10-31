//! Extended routed-only qualification. All reports are private, descriptive,
//! complete-row protocols; no quality threshold or default promotion.
use super::*;
use crate::glm5_next::memory;
use crate::glm5_next_metal::tensor_q5_down;
use crate::tensor::TensorDesc;
use unicode_normalization::UnicodeNormalization;

#[path = "routed_latency.rs"]
mod latency;

const PIN: &[u8] =
    include_bytes!("../../../../../../docs/bench/2026-10-16-glm-quality-comparison/stock-q4.json");
const LABELS: [&str; 2] = ["CURRENT", "TENSOR_GATE_UP"];
const QUALITY_CAPACITY: usize = 5264;
const LATENCY_CAPACITY: usize = 4416;
const ORDER: [bool; 8] = [false, true, true, false, true, false, false, true];

fn selected<T>(candidate: bool, body: impl FnOnce() -> T) -> T {
    let _dense = tensor_prefill::Scope::new(true);
    let _wide = tensor_prefill::Q8WideScope::new(true);
    let _q5 = tensor_q5_down::Scope::new(true);
    let _kda = memory::KdaHalfInputsScope::new(false);
    let _routed = memory::TensorGateUpScope::new(candidate);
    body()
}

fn environment_for(validation: bool) {
    assert!(
        !cfg!(debug_assertions),
        "canonical release test binary required"
    );
    assert_eq!(
        std::env::var("MTL_DEBUG_LAYER").as_deref(),
        Ok(if validation { "1" } else { "0" })
    );
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    assert_eq!(
        std::env::var("QWEN_MATMAT_Q6_K_N64_MIN_N").as_deref(),
        Ok("1024")
    );
    assert_eq!(crate::metal::mat_mat_q6_k_n64_min_query(), 1024);
    for key in [
        "QWEN_EXPERT_TENSOR_TILE",
        "QWEN_MATMAT_Q6_K_N64",
        "QWEN_GLM53_DENSE_Q8_WIDE",
        "QWEN_GLM53_TENSOR_Q5_DOWN",
    ] {
        assert!(
            crate::env_flag::read_default_on(key),
            "qualified control {key} must remain ON"
        );
    }
    assert!(crate::env_flag::read_default_off(
        "QWEN_GLM53_TENSOR_GATE_UP"
    ));
    assert!(!crate::env_flag::read_default_off(
        "QWEN_GLM53_KDA_HALF_INPUTS"
    ));
    assert!(std::env::var_os("GLM53_GGUF").is_some());
    assert_plain();
}

fn kda_pins() -> Vec<(String, GgmlType, Vec<u64>)> {
    let mut pins = Vec::new();
    for block in (0..45).filter(|block| block % 4 != 3) {
        for (suffix, dtype, shape) in [
            ("attn_q.weight", GgmlType::Q8_0, vec![4096, 8192]),
            ("attn_k.weight", GgmlType::Q8_0, vec![4096, 8192]),
            ("attn_v.weight", GgmlType::Q8_0, vec![4096, 8192]),
            ("ssm_f_a.weight", GgmlType::Q8_0, vec![4096, 128]),
            ("ssm_beta.weight", GgmlType::Q8_0, vec![4096, 64]),
            ("ssm_g_a.weight", GgmlType::Q8_0, vec![4096, 128]),
            ("ssm_f_b.weight", GgmlType::Q8_0, vec![128, 8192]),
            ("ssm_g_b.weight", GgmlType::Q8_0, vec![128, 8192]),
            ("attn_output.weight", GgmlType::Q8_0, vec![8192, 4096]),
        ] {
            pins.push((format!("blk.{block}.{suffix}"), dtype, shape));
        }
    }
    pins
}

pub(super) fn assert_pins(tensors: &[TensorDesc]) {
    super::kda_half_inputs::assert_metadata_pins(tensors);
    for (name, dtype, shape) in kda_pins() {
        let t = tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("missing KDA role {name}"));
        assert_eq!(t.dtype, dtype, "{name}");
        assert_eq!(t.shape, shape, "{name}");
    }
}

fn reserve(vocab: u32, quality: bool) -> Vec<(&'static str, u64)> {
    let v = u64::from(vocab);
    let mut terms = if quality {
        cpu_terms(vocab, false)
    } else {
        vec![(
            "eight_retained_logits_plus_two_live_readback_rows",
            10 * v * 4,
        )]
    };
    terms.extend([
        ("fixture_metadata_and_session_records", 16 << 20),
        ("private_reports_hashing_and_metrics", REPORT_BYTES),
    ]);
    if quality {
        terms.extend([
            (
                "tokenizer_construction_retained_and_decode",
                TOKENIZER_BYTES,
            ),
            ("sampler_topk_off_vectors", v * 24),
            ("sampler_radix_histogram", 65536 * size_of::<usize>() as u64),
            ("two_tool_sequences_text_parsing_serialization", 64 << 20),
        ]);
    }
    terms
}

fn stamps(gguf: &GgufFile) -> Value {
    json!(gguf.revalidate_retained_shard_stamps().unwrap().iter().map(|s| json!({
        "name":s.path.file_name().unwrap(),"shard":s.shard_idx,"bytes":s.size,"device":s.device,"inode":s.inode,
        "mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,"ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec})).collect::<Vec<_>>())
}

struct Loaded {
    ctx: MetalContext,
    gguf: GgufFile,
    weights: Glm5NextWeights,
    stamps: Value,
    cpu: u64,
    ledger: Glm5NextMemoryLedger,
    capacity: usize,
    tokenizer: Option<NativeTokenizer>,
}

impl Loaded {
    fn load(quality: bool, out: &mut Report) -> Self {
        let gguf = GgufFile::open(std::env::var_os("GLM53_GGUF").unwrap()).unwrap();
        let pin: Value = serde_json::from_slice(PIN).unwrap();
        assert_eq!((gguf.shards.len(), gguf.tensors.len()), (6, 1412));
        assert_eq!(pin["shards"].as_array().unwrap().len(), 6);
        for (shard, expected) in gguf.shards.iter().zip(pin["shards"].as_array().unwrap()) {
            assert_eq!(
                shard.path.file_name().unwrap().to_str().unwrap(),
                expected["name"].as_str().unwrap()
            );
            assert_eq!(shard.mmap_len() as u64, expected["size"].as_u64().unwrap());
        }
        let primary = sha256_hex(gguf.primary_shard().mmap_bytes());
        assert_eq!(primary, pin["shards"][0]["sha256"].as_str().unwrap());
        let artifact = Glm5NextArtifactLayout::inspect(&gguf).unwrap();
        assert!(artifact.packed_prefill());
        assert_eq!(
            (
                artifact.config().blocks.len(),
                artifact.config().leading_dense_block_count
            ),
            (45, 3)
        );
        assert_pins(&gguf.tensors);
        let stamps = stamps(&gguf);
        let terms = reserve(artifact.config().vocab_size, quality);
        let cpu = terms
            .iter()
            .try_fold(0u64, |n, (_, b)| n.checked_add(*b))
            .unwrap();
        let capacity = if quality {
            QUALITY_CAPACITY
        } else {
            LATENCY_CAPACITY
        };
        let ctx = MetalContext::new().unwrap();
        assert!(ctx.device.name().to_string().contains("M5"));
        assert!(crate::metal::moe_grouped_tensor_selected(&ctx));
        out.emit(json!({"event":"header","schema":if quality {"glm53.routed_extended_quality.v1"} else {"glm53.routed_heated_latency.v1"},
            "binary_sha256":binary_hash(),"fixture_sha256":sha256_hex(FROZEN),"pin_sha256":sha256_hex(PIN),"primary_sha256":primary,
            "probe_sha256":sha256_hex(include_bytes!("routed_qualification.rs")),"latency_sha256":sha256_hex(include_bytes!("routed_latency.rs")),
            "packed_sha256":sha256_hex(include_bytes!("../../packed.rs")),"adapter_sha256":sha256_hex(include_bytes!("../../../metal/expert_tensor_gate_up.rs")),
            "strict_scorer_sha256":sha256_hex(include_bytes!("../../../../../../scripts/reference/glm53/quality_analysis.py")),
            "artifact_layout":artifact_layout(&gguf),"retained_stamps":stamps,"identity_scope":"named mixed expert/MLA and KDA fanout pins; primary metadata SHA, shard names/sizes and retained stamps; weight payload hashes not reverified",
            "arms":LABELS,"q8_wide":true,"q5_down":true,"kda_execution":false,"global_routed_storage":true,
            "capacity":capacity,"workspace_rows":ROWS,"half_bytes_each_arm":16<<20,"cpu_terms":terms,"cpu_reserve_bytes":cpu,
            "validation":quality,"external_outer_limit_seconds":if quality {900} else {240},
            "limit_scope":"external runner must enforce the fresh wall limit; synchronous GPU calls cannot be interrupted by between-call checks",
            "policy":"descriptive only; no automatic thresholds, quality verdict or promotion; incomplete without complete event"}));
        let (weights, ledger, tokenizer) = {
            let _allocation = ctx.begin_allocation_transaction();
            let plan = preflight_session(&ctx, &gguf, artifact.model(), capacity, ROWS).unwrap();
            assert!(plan.ledger.scratch_options().tensor_gate_up);
            assert!(!plan.ledger.scratch_options().kda_half_inputs);
            let half = memory::packed_normalized_half_specs(
                artifact.config(),
                ROWS as u64,
                plan.ledger.scratch_options(),
            );
            assert_eq!(
                (half.len(), half[0].name, half[0].bytes()),
                (1, "normalized_half", Some(16 << 20))
            );
            let admission = crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(
                plan.ledger.peak_bytes(),
                cpu,
                0,
                ctx.memory_signals(),
                true,
            );
            out.emit(json!({"event":"preload_admission","ledger":plan.ledger.terms(),"cpu_reserve_bytes":cpu,"cpu_availability_checked":true,"required_bytes":admission.required_bytes,"admitted":admission.is_admitted(),"reason":admission.reason.as_str()}));
            assert!(
                admission.is_admitted(),
                "full GPU+CPU admission refused: {admission:?}"
            );
            let tokenizer = quality.then(|| {
                let tokenizer = NativeTokenizer::from_gguf(&gguf).unwrap();
                assert_eq!(tokenizer.n_vocab(), artifact.config().vocab_size);
                // Lossy UTF-8 can expand threefold; both retained sequences,
                // parse trees and JSON copies share the conservative heap bound.
                let decode_upper = tokenizer.max_decoded_piece_bytes().checked_mul(TOOL_CAP).and_then(|n|n.checked_mul(3)).and_then(|n|n.checked_mul(64)).unwrap();
                assert!(decode_upper <= 64<<20, "increase and readmit the two-sequence parse/serialization reserve before loading weights");
                tokenizer
            });
            (
                Glm5NextWeights::load(&ctx, &gguf).unwrap(),
                plan.ledger,
                tokenizer,
            )
        };
        Self {
            ctx,
            gguf,
            weights,
            stamps,
            cpu,
            ledger,
            capacity,
            tokenizer,
        }
    }

    fn fresh(&self) -> Glm5NextSession<'_> {
        let mut s = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
            &self.ctx,
            &self.weights,
            self.capacity,
            ROWS,
            self.cpu,
        )
        .unwrap();
        s.set_packed_lineage(PackedLineage::Fast).unwrap();
        s.set_fast_precision(FastPrecision::Half).unwrap();
        s
    }

    fn prefill(&self, s: &mut Glm5NextSession<'_>, tokens: &[u32], candidate: bool) -> Vec<f32> {
        check_session(s, 0);
        assert_eq!(*s.ledger(), self.ledger);
        let mut commits = Vec::new();
        let logits = selected(candidate, || {
            s.prefill_packed_with_observer(&self.ctx, tokens, &mut || Ok(()), &mut |s, n| {
                commits.push(n);
                check_session(s, commits.iter().sum());
            })
            .unwrap()
        });
        assert_eq!(commits, chunks(tokens.len()));
        check_session(s, tokens.len());
        check_logits(&logits, self.weights.config.vocab_size as usize);
        logits
    }

    fn finish(&self) {
        assert_plain();
        assert_eq!(stamps(&self.gguf), self.stamps);
    }
}

#[derive(Default)]
struct Totals {
    rows: usize,
    nll: f64,
    kl: f64,
    max_kl: f64,
    hits: usize,
    agreements: usize,
}

impl Totals {
    fn token(&mut self, logits: &[f32], reference: &[f32], target: u32) {
        let nll = -log_prob(logits, target);
        let kl = kl_divergence(reference, logits);
        assert!(nll.is_finite() && kl.is_finite());
        self.rows += 1;
        self.nll += nll;
        self.kl += kl;
        self.max_kl = self.max_kl.max(kl);
        self.hits += usize::from(argmax(logits) == target as usize);
        self.agreements += usize::from(argmax(logits) == argmax(reference));
    }

    fn add(&mut self, other: &Self) {
        self.rows += other.rows;
        self.nll += other.nll;
        self.kl += other.kl;
        self.max_kl = self.max_kl.max(other.max_kl);
        self.hits += other.hits;
        self.agreements += other.agreements;
    }

    fn describe(&self, expected: usize) -> Value {
        assert_eq!(self.rows, expected);
        assert!(self.nll.is_finite() && self.kl.is_finite() && self.max_kl.is_finite());
        json!({"rows":self.rows,"mean_nll":self.nll/self.rows as f64,"mean_kl_current_to_arm":self.kl/self.rows as f64,
            "max_kl_current_to_arm":self.max_kl,"top1_correct":self.hits,"top1_agrees_current":self.agreements,"all_finite":true})
    }
}

fn currency(value: &Value) -> Option<&'static str> {
    match value.as_str()?.trim().to_ascii_lowercase().as_str() {
        "usd" | "us dollar" | "us dollars" | "united states dollar" | "united states dollars" => {
            Some("USD")
        }
        "eur" | "euro" | "euros" => Some("EUR"),
        "gbp" | "british pound" | "british pounds" | "pound sterling" => Some("GBP"),
        "jpy" | "yen" | "japanese yen" => Some("JPY"),
        "chf" | "swiss franc" | "swiss francs" => Some("CHF"),
        "cad" | "canadian dollar" | "canadian dollars" => Some("CAD"),
        "aud" | "australian dollar" | "australian dollars" => Some("AUD"),
        _ => None,
    }
}

fn strict_correct(
    calls: &[chat::ToolCall],
    stop: Option<u32>,
    parse_error: bool,
    expected: Expect,
) -> bool {
    if parse_error || stop != Some(OBSERVATION) || calls.len() != 1 {
        return false;
    }
    let call = &calls[0];
    let value = |key: &str| call.arguments.get(key).unwrap_or(&Value::Null);
    match expected {
        Expect::Weather { city } => {
            let got: String = value("city")
                .as_str()
                .unwrap_or("")
                .nfkd()
                .filter(char::is_ascii)
                .collect();
            let normalized = got.split(',').next().unwrap().trim().to_ascii_lowercase();
            let days = value("days");
            call.name == "get_weather"
                && normalized == city.to_ascii_lowercase()
                && (days.is_null() || days.as_i64().is_some_and(|n| (1..=7).contains(&n)))
        }
        Expect::Currency { amount, from, to } => {
            call.name == "convert_currency"
                && value("amount").as_f64().is_some_and(|got| {
                    got.is_finite()
                        && (got - amount).abs() <= 1e-9f64.max(1e-9 * got.abs().max(amount.abs()))
                })
                && currency(value("from")) == Some(from)
                && currency(value("to")) == Some(to)
        }
    }
}

struct Sequence {
    sampled: Vec<u32>,
    emitted: Vec<u32>,
    text: String,
    serialized: Vec<u8>,
    strict: bool,
    stop: Option<u32>,
}

fn generate(
    loaded: &Loaded,
    tokenizer: &NativeTokenizer,
    definitions: &[ToolDefinition],
    task: &ToolTask,
    generation: Generation,
    candidate: bool,
) -> Sequence {
    let mut s = loaded.fresh();
    let mut logits = loaded.prefill(&mut s, &task.prompt, candidate);
    let mut sampler = Sampler::new(generation.sampling()).unwrap();
    let mut sampled = Vec::with_capacity(TOOL_CAP);
    let mut emitted = Vec::with_capacity(TOOL_CAP);
    let stop = loop {
        check_session(&s, task.prompt.len() + sampled.len());
        check_logits(&logits, loaded.weights.config.vocab_size as usize);
        let token = u32::try_from(sampler.sample(&logits).unwrap().token).unwrap();
        assert!(token < loaded.weights.config.vocab_size);
        sampled.push(token);
        if chat::CHAT_STOPS.contains(&i32::try_from(token).unwrap()) {
            break Some(token);
        }
        emitted.push(token);
        if sampled.len() == TOOL_CAP {
            break None;
        }
        logits = s.forward(&loaded.ctx, token).unwrap();
    };
    check_session(&s, task.prompt.len() + sampled.len() - 1);
    drop(s);
    drop(logits);
    drop(sampler);
    let text = tokenizer
        .try_decode(
            &emitted
                .iter()
                .map(|&id| i32::try_from(id).unwrap())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let (calls, parse_error) = match parse_turn(&text, definitions) {
        Ok((_, _, calls)) => (calls, false),
        Err(_) => (Vec::new(), true),
    };
    let strict = strict_correct(&calls, stop, parse_error, task.expect);
    let serialized = serde_json::to_vec(&json!({"parse_error":parse_error,"stop":stop,
        "calls":calls.iter().map(|c|json!({"id":c.id,"name":c.name,"arguments":c.arguments})).collect::<Vec<_>>()})).unwrap();
    Sequence {
        sampled,
        emitted,
        text,
        serialized,
        strict,
        stop,
    }
}

#[test]
#[ignore = "release M5, validation ON, production lease, fresh GLM53_ROUTED_EXTENDED_REPORT; 22 documents and eight strict tool sequences per arm"]
fn m5_routed_extended_quality_and_strict_tools() {
    environment_for(true);
    let _lease = production_lease();
    let mut out = Report::new("GLM53_ROUTED_EXTENDED_REPORT", true);
    let docs = all_documents();
    let (tasks, definitions) = long_tasks();
    assert_eq!((docs.len(), tasks.len(), CONTINUATION), (22, 4, 64));
    let loaded = Loaded::load(true, &mut out);
    let vocab = loaded.weights.config.vocab_size as usize;
    assert!(
        docs.iter()
            .flat_map(|d| &d.tokens)
            .chain(tasks.iter().flat_map(|t| &t.prompt))
            .all(|&id| (id as usize) < vocab)
    );
    let tokenizer = loaded.tokenizer.as_ref().unwrap();
    out.emit(json!({"event":"protocol","documents":22,"targets_per_arm":1408,"quality_rows":2816,"tool_sequences_per_arm":8,"tool_generations":16,
        "tool_prompt_lengths":tasks.iter().map(|t|t.prompt.len()).collect::<Vec<_>>(),"seed":11,"retained_baseline_rows":64,"census_calls":0,"profile_calls":0,
        "scope":"same frozen IDs and sampling seeds in both arms; only routed TLS differs during prefill; ordinary decode outside all scopes; no raw IDs, prompts or private paths emitted"}));
    let mut aggregates: [Totals; 2] = std::array::from_fn(|_| Totals::default());
    let mut document_arms = 0;
    for (case, d) in docs.iter().enumerate() {
        let mut baseline = Vec::with_capacity(CONTINUATION);
        for candidate in [false, true] {
            assert_plain();
            let mut s = loaded.fresh();
            let mut logits = loaded.prefill(&mut s, &d.tokens[..d.prefix], candidate);
            let mut totals = Totals::default();
            for (i, &target) in d.tokens[d.prefix..].iter().enumerate() {
                check_session(&s, d.prefix + i);
                check_logits(&logits, vocab);
                if !candidate {
                    baseline.push(logits.clone());
                }
                totals.token(&logits, &baseline[i], target);
                if i + 1 < CONTINUATION {
                    logits = s.forward(&loaded.ctx, target).unwrap();
                }
            }
            check_session(&s, d.prefix + CONTINUATION - 1);
            assert_eq!(baseline.len(), CONTINUATION);
            out.emit(json!({"event":"document_arm","case":case,"stratum":d.stratum,"prefix":d.prefix,"tokens_sha256":token_hash(&d.tokens),
                "arm":LABELS[usize::from(candidate)],"metrics":totals.describe(CONTINUATION),"final_position":s.position(),"prefill_chunks":chunks(d.prefix)}));
            aggregates[usize::from(candidate)].add(&totals);
            document_arms += 1;
        }
    }
    let mut tools = 0;
    let mut correct = [0usize; 2];
    let mut equal_text = 0;
    let mut equal_serialized = 0;
    for task in &tasks {
        for generation in [
            Generation::Greedy { cap: TOOL_CAP },
            Generation::Sampled {
                seed: 11,
                cap: TOOL_CAP,
            },
        ] {
            let a = generate(&loaded, &tokenizer, &definitions, task, generation, false);
            let b = generate(&loaded, &tokenizer, &definitions, task, generation, true);
            for (index, sequence) in [&a, &b].into_iter().enumerate() {
                assert!(!sequence.sampled.is_empty() && sequence.sampled.len() <= TOOL_CAP);
                assert_eq!(
                    sequence.sampled.len(),
                    sequence.emitted.len() + usize::from(sequence.stop.is_some())
                );
                correct[index] += usize::from(sequence.strict);
                out.emit(json!({"event":"tool_sequence","id":task.id,"arm":LABELS[index],"generation":generation.describe(),"strict_correct":sequence.strict,
                    "sampled_tokens":sequence.sampled.len(),"emitted_tokens":sequence.emitted.len(),"sampled_sha256":token_hash(&sequence.sampled),
                    "text_sha256":sha256_hex(sequence.text.as_bytes()),"serialized_sha256":sha256_hex(&sequence.serialized),"serialized":serde_json::from_slice::<Value>(&sequence.serialized).unwrap()}));
                tools += 1;
            }
            equal_text += usize::from(a.text.as_bytes() == b.text.as_bytes());
            equal_serialized += usize::from(a.serialized == b.serialized);
            out.emit(json!({"event":"tool_comparison","id":task.id,"generation":generation.describe(),"sampled_ids_equal":a.sampled==b.sampled,
                "decoded_bytes_equal":a.text.as_bytes()==b.text.as_bytes(),"serialized_bytes_equal":a.serialized==b.serialized}));
        }
    }
    assert_eq!((document_arms, tools), (44, 16));
    let metrics = [aggregates[0].describe(1408), aggregates[1].describe(1408)];
    loaded.finish();
    out.emit(json!({"event":"complete","execution_complete":true,"document_arms":44,"quality_rows":2816,"metrics":metrics,"tool_generations":16,
        "strict_tool_correct_per_arm":correct,"tool_byte_equal_sequences":equal_text,"tool_serialized_equal_sequences":equal_serialized,"sessions":60,
        "retained_stamps_unchanged":true,"observers_clear":true,"quality_verdict":null,"performance_verdict":null}));
    out.sync();
}

#[test]
fn cpu_routed_extended_protocol_counts_and_reserves() {
    let docs = all_documents();
    let (tasks, _) = long_tasks();
    assert_eq!((docs.len() * CONTINUATION, tasks.len() * 2), (1408, 8));
    assert_eq!(
        docs.iter()
            .map(|d| d.prefix + CONTINUATION)
            .chain(tasks.iter().map(ToolTask::capacity))
            .max(),
        Some(QUALITY_CAPACITY)
    );
    assert_eq!(ORDER.iter().filter(|&&v| v).count(), 4);
    for quality in [false, true] {
        let terms = reserve(154880, quality);
        let bytes: u64 = terms.iter().map(|(_, n)| n).sum();
        assert!(
            bytes
                > if quality {
                    64 * 154880 * 4 + TOKENIZER_BYTES
                } else {
                    8 * 154880 * 4
                }
        );
        assert!(bytes < 512 << 20);
    }
}

#[test]
fn cpu_routed_strict_tool_scorer_rejects_lenient_cases() {
    let weather = Expect::Weather { city: "Montreal" };
    let call = |name: &str, args: Value| chat::ToolCall {
        id: "call_1".into(),
        name: name.into(),
        arguments: args.as_object().unwrap().clone(),
    };
    let good = call(
        "get_weather",
        json!({"city":"Montr\u{e9}al, Canada","days":3}),
    );
    assert!(strict_correct(
        &[good.clone()],
        Some(OBSERVATION),
        false,
        weather
    ));
    for args in [
        json!({"city":"Greater Montreal"}),
        json!({"city":"Montreal","days":true}),
        json!({"city":"Montreal","days":8}),
        json!({"city":"Montreal","days":3.0}),
    ] {
        assert!(!strict_correct(
            &[call("get_weather", args)],
            Some(OBSERVATION),
            false,
            weather
        ));
    }
    assert!(!strict_correct(
        &[good.clone(), good.clone()],
        Some(OBSERVATION),
        false,
        weather
    ));
    assert!(!strict_correct(&[good.clone()], None, false, weather));
    assert!(!strict_correct(&[good], Some(OBSERVATION), true, weather));
    let expected = Expect::Currency {
        amount: 75.0,
        from: "EUR",
        to: "CAD",
    };
    assert!(strict_correct(
        &[call(
            "convert_currency",
            json!({"amount":75,"from":"euros","to":"Canadian dollars"})
        )],
        Some(OBSERVATION),
        false,
        expected
    ));
    for amount in [json!("75"), json!(true), json!(76)] {
        assert!(!strict_correct(
            &[call(
                "convert_currency",
                json!({"amount":amount,"from":"EUR","to":"CAD"})
            )],
            Some(OBSERVATION),
            false,
            expected
        ));
    }
}

#[test]
fn cpu_routed_kda_projection_pins_include_q8_ranks_output_and_reject_drift() {
    let mut pins = super::kda_half_inputs::metadata_pins();
    pins.extend(kda_pins());
    assert_eq!((kda_pins().len(), pins.len()), (306, 456));
    assert!(
        kda_pins()
            .iter()
            .all(|(_, dtype, _)| *dtype == GgmlType::Q8_0)
    );
    let tensors: Vec<_> = pins
        .into_iter()
        .map(|(name, dtype, shape)| TensorDesc {
            name,
            dtype,
            shape,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: 0,
        })
        .collect();
    assert_pins(&tensors);
    for suffix in [
        "attn_q.weight",
        "ssm_f_b.weight",
        "ssm_g_b.weight",
        "attn_output.weight",
    ] {
        for dtype in [GgmlType::Q5_K, GgmlType::Q6_K] {
            let mut bad = tensors.clone();
            let index = bad
                .iter()
                .position(|t| t.name == format!("blk.0.{suffix}"))
                .unwrap();
            bad[index].dtype = dtype;
            assert!(std::panic::catch_unwind(|| assert_pins(&bad)).is_err());
        }
    }
}

#[test]
fn cpu_routed_preload_admission_checks_cpu_availability_separately() {
    use crate::metal::{HostMemoryBudget, MetalMemoryAdmissionReason, MetalMemorySignals};
    const GIB: u64 = 1 << 30;
    let cpu = GIB;
    let mut signals = MetalMemorySignals {
        recommended_max_bytes: 16 * GIB,
        current_allocated_bytes: GIB,
        process_limit_remaining_bytes: Some(20 * GIB),
        host_budget: Some(HostMemoryBudget {
            physical_bytes: 20 * GIB,
            conservative_available_bytes: cpu + 32,
            external_gap_bytes: GIB,
            private_shadow_bytes: 32,
        }),
    };
    let decide =
        |s| crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(10 * GIB, cpu, 0, s, true);
    assert!(decide(signals).is_admitted());
    assert_eq!(decide(signals).cpu_required_bytes, Some(cpu));
    assert_eq!(decide(signals).required_bytes, Some(10 * GIB + cpu));
    signals
        .host_budget
        .as_mut()
        .unwrap()
        .conservative_available_bytes -= 1;
    assert_eq!(
        decide(signals).reason,
        MetalMemoryAdmissionReason::HostAvailabilityInsufficient
    );
    assert!(
        crate::metal::evaluate_metal_memory_admission(10 * GIB + cpu, 0, signals, true)
            .is_admitted(),
        "combined Metal-only demand misses the CPU availability refusal"
    );
}

#[test]
fn cpu_routed_metrics_require_every_finite_row_before_completion() {
    let logits = [0.0, 1.0];
    let mut metrics = Totals::default();
    metrics.token(&logits, &logits, 1);
    assert_eq!(metrics.describe(1)["top1_correct"], 1);
    assert!(std::panic::catch_unwind(|| metrics.describe(2)).is_err());
    assert!(
        std::panic::catch_unwind(|| {
            let mut bad = Totals::default();
            bad.token(&[f32::NAN, 0.0], &logits, 1);
        })
        .is_err()
    );
}
