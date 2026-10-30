//! Separate actual-Q4 transfer screens, not the stock four-shard qualification.
use super::*;
use crate::glm5_next_metal::tensor_q5_down;

const ARMS: [(&str, bool, bool); 3] = [
    ("CURRENT", false, false),
    ("Q8WIDE", true, false),
    ("Q5DOWN", false, true),
];
const Q4_PIN: &[u8] =
    include_bytes!("../../../../../../docs/bench/2026-10-16-glm-quality-comparison/stock-q4.json");
const Q6_DOWN_BLOCKS: [usize; 3] = [11, 12, 44];
const Q5_DOWN_BANKS: usize = 39;
const Q5_GATE_UP_BLOCKS: [usize; 1] = [11];
const Q8_MLA_BLOCKS: [usize; 11] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43];
const EXTENDED_REPORT: &str = "GLM53_M5_TRANSFER_EXTENDED_REPORT";

fn extended_cpu_terms(vocab: u32) -> Vec<(&'static str, u64)> {
    let mut terms = cpu_terms(vocab, false);
    terms.extend([
        (
            "native_tokenizer_construction_retained_and_decode_peak",
            TOKENIZER_BYTES,
        ),
        ("sampler_topk_off_peak_vectors", u64::from(vocab) * 24),
        ("sampler_radix_histogram", 65536 * size_of::<usize>() as u64),
        ("tool_ids_text_parse_and_progress_reports", 8 << 20),
    ]);
    terms
}

fn extended_capacity(docs: &[Document], tasks: &[ToolTask]) -> usize {
    docs.iter()
        .map(|d| d.prefix + CONTINUATION)
        .chain(tasks.iter().map(ToolTask::capacity))
        .max()
        .unwrap()
}

#[test]
fn cpu_extended_frozen_protocol_counts_capacity_and_allowances() {
    let docs = all_documents();
    let (tasks, _) = long_tasks();
    assert_eq!(
        ARMS,
        [
            ("CURRENT", false, false),
            ("Q8WIDE", true, false),
            ("Q5DOWN", false, true)
        ]
    );
    assert_eq!(CONTINUATION, 64);
    assert_eq!(TOOL_CAP, 256);
    assert_eq!(docs.len() * CONTINUATION, 1408);
    assert_eq!(docs.len() * CONTINUATION * ARMS.len(), 4224);
    assert_eq!(docs.len() * ARMS.len(), 66);
    assert_eq!(tasks.len() * 2 * ARMS.len(), 24);
    assert_eq!(extended_capacity(&docs, &tasks), 5264);
    assert_eq!(
        tasks.iter().map(|t| t.prompt.len()).collect::<Vec<_>>(),
        [1455, 1455, 5008, 5008]
    );
    assert_eq!(chunks(tasks[0].prompt.len()), [1455]);
    assert_eq!(chunks(tasks[2].prompt.len()), [2048, 2048, 912]);
    for d in &docs {
        assert_eq!(d.tokens.len(), d.prefix + CONTINUATION);
    }
    let vocab = 154880;
    let base = cpu_terms(vocab, false);
    let terms = extended_cpu_terms(vocab);
    assert_eq!(&terms[..base.len()], base.as_slice());
    assert_eq!(terms.len(), base.len() + 4);
    assert_eq!(terms[0].1, 64 * u64::from(vocab) * 4);
    assert_eq!(terms[base.len()].1, TOKENIZER_BYTES);
    assert_eq!(
        terms.iter().map(|(_, n)| n).sum::<u64>() - base.iter().map(|(_, n)| n).sum::<u64>(),
        TOKENIZER_BYTES + u64::from(vocab) * 24 + 65536 * size_of::<usize>() as u64 + (8 << 20)
    );
    assert_ne!(EXTENDED_REPORT, "GLM53_M5_TRANSFER_REPORT");
}

fn down_dtype(block: usize) -> GgmlType {
    if Q6_DOWN_BLOCKS.contains(&block) {
        GgmlType::Q6_K
    } else {
        GgmlType::Q5_K
    }
}

fn gate_up_dtype(block: usize) -> GgmlType {
    if Q5_GATE_UP_BLOCKS.contains(&block) {
        GgmlType::Q5_K
    } else {
        GgmlType::Q4_K
    }
}

#[test]
fn cpu_main_down_inventory_keeps_exact_q6_exceptions_and_39_q5_banks() {
    let blocks = |dtype| {
        (3..45)
            .filter(|&block| down_dtype(block) == dtype)
            .collect::<Vec<_>>()
    };
    assert_eq!(blocks(GgmlType::Q6_K), [11, 12, 44]);
    assert_eq!(blocks(GgmlType::Q5_K).len(), Q5_DOWN_BANKS);
    assert_eq!(Q5_DOWN_BANKS + Q6_DOWN_BLOCKS.len(), 42);
}

#[test]
fn cpu_main_gate_up_inventory_keeps_exact_block11_q5_pair_and_82_q4_banks() {
    let q5_blocks = (3..45)
        .filter(|&block| gate_up_dtype(block) == GgmlType::Q5_K)
        .collect::<Vec<_>>();
    assert_eq!(q5_blocks, [11]);
    assert_eq!(
        (3..45)
            .filter(|&block| gate_up_dtype(block) == GgmlType::Q4_K)
            .count()
            * 2,
        82
    );
    assert_eq!(q5_blocks.len() * 2, 2);
    for block in 3..45 {
        if down_dtype(block) == GgmlType::Q5_K {
            assert_eq!(gate_up_dtype(block), GgmlType::Q4_K);
        }
    }
    assert_eq!(gate_up_dtype(45), GgmlType::Q4_K);
    assert_eq!(down_dtype(45), GgmlType::Q5_K);
}

#[test]
fn cpu_q8_mla_inventory_has_11_executed_pairs_and_one_unused_mtp_pair() {
    assert_eq!(
        Q8_MLA_BLOCKS.as_slice(),
        (3..45).step_by(4).collect::<Vec<_>>()
    );
    assert_eq!(Q8_MLA_BLOCKS.len() * 2, 22);
    assert_eq!((Q8_MLA_BLOCKS.len() + 1) * 2, 24);
    assert!(!Q8_MLA_BLOCKS.contains(&45));
}

fn selected<R>(arm: usize, body: impl FnOnce() -> R) -> R {
    let _dense = tensor_prefill::Scope::new(true);
    let _down = tensor_down::Scope::new(true);
    let _wide = tensor_prefill::Q8WideScope::new(ARMS[arm].1);
    let _q5 = tensor_q5_down::Scope::new(ARMS[arm].2);
    body()
}

fn difference(reference: &[f32], logits: &[f32]) -> Value {
    let max_abs = max_abs_diff(logits, reference);
    let kl = kl_divergence(reference, logits);
    assert!(max_abs.is_finite() && kl.is_finite());
    json!({"changed_bits_components":reference.iter().zip(logits)
        .filter(|(a,b)| a.to_bits() != b.to_bits()).count(),
        "max_abs":max_abs,"kl_reference_to_arm":kl,"compared_vocab":reference.len(),
        "all_finite":true})
}

fn profile(ctx: &MetalContext, w: &Glm5NextWeights, tokens: &[u32], cpu: u64, out: &mut Report) {
    for (arm, &(label, wide, q5)) in ARMS.iter().enumerate() {
        let mut reference = None;
        let mut plain = Vec::new();
        for phase in ["warm_census", "plain1", "plain2", "profiled"] {
            let mut s = fresh(ctx, w, ROWS, cpu, label, phase, out);
            assert_plain();
            let census = (phase == "warm_census").then(Census::begin);
            let (logits, wall_ms, report) = selected(arm, || {
                let start = Instant::now();
                let (logits, report) = if phase == "profiled" {
                    let (logits, report) = s.prefill_chunk_stage_profiled(ctx, tokens).unwrap();
                    (logits, Some(report))
                } else {
                    (s.prefill_packed(ctx, tokens).unwrap(), None)
                };
                (logits, start.elapsed().as_secs_f64() * 1e3, report)
            });
            let census = census.map(Census::finish);
            assert_plain();
            check_session(&s, ROWS);
            check_logits(&logits, w.config.vocab_size as usize);
            assert!(wall_ms.is_finite() && wall_ms > 0.0);
            if let Some(census) = census {
                let heap = census.capacity() * size_of::<DispatchCensusRow>()
                    + census
                        .iter()
                        .map(|r| r.kernel.capacity() + r.tag.as_ref().map_or(0, String::capacity))
                        .sum::<usize>();
                assert!((heap as u64).checked_mul(2).unwrap() <= CENSUS_BYTES);
                let mut kernels = BTreeMap::<&str, usize>::new();
                for row in &census {
                    *kernels.entry(&row.kernel).or_default() += 1;
                }
                let count = |name: &str| kernels.get(name).copied().unwrap_or(0);
                let wide_calls = count("kernel_mat_mat_q8_0_f32_tensor_wide");
                assert_eq!(wide_calls > 0, wide);
                if !wide {
                    assert!(count("kernel_mat_mat_q8_0_f32_tensor") > 0);
                }
                assert_eq!(
                    count("kernel_moe_down_q5_K_f32_grouped_slots_tensor"),
                    if q5 { Q5_DOWN_BANKS } else { 0 }
                );
                out.emit(json!({"event":"census","arm":label,"prompt_tokens":ROWS,
                    "expected_chunks":chunks(tokens.len()),"dispatches":census.len(),"kernel_counts":kernels,
                    "census_heap_bytes":heap,"census_upper_bytes":heap*2,"timed":false}));
            }
            let delta = reference
                .as_ref()
                .map(|r: &Vec<f32>| difference(r, &logits));
            out.emit(
                json!({"event":"profile_run","arm":label,"phase":phase,"wall_ms":wall_ms,
                "vs_same_arm_warm":delta,"logits_sha256":sha256_hex(bytemuck::cast_slice(&logits)),
                "ordinary_plain":phase.starts_with("plain"),"logits_finite":true}),
            );
            if phase == "warm_census" {
                reference = Some(logits);
            }
            if phase.starts_with("plain") {
                plain.push(wall_ms);
            }
            if let Some(report) = report {
                let mut stages = BTreeMap::<&str, f64>::new();
                for span in &report.spans {
                    *stages.entry(span.stage.as_str()).or_default() += span.gpu_ms;
                }
                out.emit(json!({"event":"profile","arm":label,"plain_wall_ms":plain,
                    "profiled_wall_ms":wall_ms,"profiled_over_plain_mean":wall_ms/((plain[0]+plain[1])*0.5),
                    "command_gpu_ms":report.command_gpu_ms,"scaled_span_sum_ms":report.span_sum_ms,
                    "stages_ms":stages,"warning":"observer diagnostic: encoder splitting and sample resolution perturb execution; scaled spans are not additive ordinary-wall shares or independent leaf costs"}));
            }
        }
    }
}

fn plain(ctx: &MetalContext, w: &Glm5NextWeights, long: &Document, cpu: u64, out: &mut Report) {
    for n in [2048, 4096] {
        for candidate in [1, 2] {
            for (sample, arm) in [0, candidate, candidate, 0, candidate, 0, 0, candidate]
                .into_iter()
                .enumerate()
            {
                let mut s = fresh(ctx, w, n, cpu, ARMS[arm].0, "plain_abba", out);
                assert_plain();
                let (logits, wall_ms) = selected(arm, || {
                    let start = Instant::now();
                    let logits = s.prefill_packed(ctx, &long.tokens[..n]).unwrap();
                    (logits, start.elapsed().as_secs_f64() * 1e3)
                });
                assert_plain();
                check_session(&s, n);
                check_logits(&logits, w.config.vocab_size as usize);
                assert!(wall_ms.is_finite() && wall_ms > 0.0);
                out.emit(json!({"event":"plain_sample","candidate":ARMS[candidate].0,"arm":ARMS[arm].0,
                    "prompt_tokens":n,"sample":sample,"block":if sample < 4 { "ABBA" } else { "BAAB" },
                    "wall_ms":wall_ms,"expected_prefill_chunks":chunks(n),"final_position":s.position(),
                    "actual_prefill_rows":s.ledger().prefill_rows(),"logits_finite":true}));
            }
        }
    }
}

fn quality(
    ctx: &MetalContext,
    w: &Glm5NextWeights,
    docs: &[Document],
    cpu: u64,
    out: &mut Report,
) -> (usize, usize) {
    let (mut document_arms, mut rows) = (0, 0);
    for d in docs {
        let mut baseline = Vec::<Vec<f32>>::with_capacity(CONTINUATION);
        for (arm, &(label, _, _)) in ARMS.iter().enumerate() {
            let mut s = fresh(ctx, w, d.prefix + CONTINUATION, cpu, label, "quality", out);
            assert_plain();
            let (mut logits, commits) = selected(arm, || {
                observed_prefill(ctx, &mut s, &d.tokens[..d.prefix], DENSE_AND_DOWN)
            });
            for (i, &target) in d.tokens[d.prefix..].iter().enumerate() {
                check_session(&s, d.prefix + i);
                check_logits(&logits, w.config.vocab_size as usize);
                if arm == 0 {
                    baseline.push(logits.clone());
                }
                let reference = &baseline[i];
                let nll = -log_prob(&logits, target);
                let reference_nll = -log_prob(reference, target);
                assert!(nll.is_finite() && reference_nll.is_finite());
                let prediction = argmax(&logits);
                let reference_top1 = argmax(reference);
                out.emit(json!({"event":"quality_token","arm":label,"path":d.path,"stratum":d.stratum,
                    "prefix":d.prefix,"index":i,"target_position":d.prefix+i,"target":target,
                    "difference":difference(reference, &logits),"nll":nll,"current_nll":reference_nll,
                    "nll_delta":nll-reference_nll,"top1":prediction,"current_top1":reference_top1,
                    "top1_hit":prediction == target as usize,"current_top1_hit":reference_top1 == target as usize,
                    "top1_agrees_current":prediction == reference_top1}));
                rows += 1;
                if i + 1 < CONTINUATION {
                    logits = s.forward(ctx, target).unwrap();
                }
            }
            assert_plain();
            assert_eq!(baseline.len(), CONTINUATION);
            check_session(&s, d.prefix + CONTINUATION - 1);
            out.emit(
                json!({"event":"document_arm","arm":label,"path":d.path,"prefix":d.prefix,
                "logit_rows":CONTINUATION,"vocab":w.config.vocab_size,"prefill_chunks":commits,
                "decode_calls":CONTINUATION-1,"final_position":s.position()}),
            );
            document_arms += 1;
        }
    }
    (document_arms, rows)
}

#[derive(Default)]
struct ToolCounts {
    generations: usize,
    tokens: usize,
    stops: usize,
    caps: usize,
}

fn transfer_tool_quality(
    ctx: &MetalContext,
    weights: &Glm5NextWeights,
    tokenizer: &NativeTokenizer,
    tasks: &[ToolTask],
    definitions: &[ToolDefinition],
    cpu: u64,
    out: &mut Report,
) -> ToolCounts {
    let mut counts = ToolCounts::default();
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
            for (arm, &(label, _, _)) in ARMS.iter().enumerate() {
                out.emit(json!({"event":"task_begin","arm":label,"id":task.id,
                    "context_level":task.context_level,"expected":task.expected,
                    "generation":generation.describe(),"prompt_tokens":n,"prompt_sha256":prompt_hash,
                    "token_hash_encoding":"u32 little-endian","requested_capacity":n+TOOL_CAP,
                    "capacity":task.capacity(),"workspace_rows":ROWS}));
                let mut s = fresh(
                    ctx,
                    weights,
                    task.capacity(),
                    cpu,
                    label,
                    "tool_quality",
                    out,
                );
                assert_plain();
                let (mut logits, commits) = selected(arm, || {
                    observed_prefill(ctx, &mut s, &task.prompt, DENSE_AND_DOWN)
                });
                assert_plain();
                out.emit(json!({"event":"task_prefill","arm":label,"id":task.id,
                    "generation":generation.describe(),"prefill_chunks":commits,"position":s.position(),
                    "actual_prefill_rows":s.ledger().prefill_rows(),
                    "q5_eligible_chunks":commits.iter().filter(|&&n| matches!(n, 1024 | 2048)).count(),
                    "scope":"eligibility from committed chunk lengths, not a dispatch census; no dispatch-count claim"}));
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
                    out.emit(json!({"event":"task_token","arm":label,"id":task.id,
                        "generation":generation.describe(),"sample_index":sampled.len()-1,
                        "position":s.position(),"token":token,"is_stop":is_stop,"logits_finite":true}));
                    counts.tokens += 1;
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
                out.emit(json!({"event":"task_generation","arm":label,"id":task.id,
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
                counts.generations += 1;
                counts.stops += usize::from(stop.is_some());
                counts.caps += usize::from(stop.is_none());
            }
        }
    }
    counts
}

#[test]
#[ignore = "release actual six-shard UD-Q4_K_XL M5 transfer screen; GLM53_GGUF, fresh GLM53_M5_TRANSFER_REPORT; MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1 QWEN_MATMAT_Q6_K_N64_MIN_N=1024"]
fn tensor_m5_transfer_screen() {
    transfer_screen(false);
}

#[test]
#[ignore = "release actual six-shard UD-Q4_K_XL M5 extended quality ONLY: all 22 frozen documents and T09-T12 greedy/seed11; GLM53_GGUF, fresh private GLM53_M5_TRANSFER_EXTENDED_REPORT; MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1 QWEN_MATMAT_Q6_K_N64_MIN_N=1024; normal full admission and production lease required"]
fn tensor_m5_transfer_extended_quality() {
    transfer_screen(true);
}

fn transfer_screen(extended: bool) {
    environment();
    let _lease = production_lease();
    let mut out = Report::new(
        if extended {
            EXTENDED_REPORT
        } else {
            "GLM53_M5_TRANSFER_REPORT"
        },
        true,
    );
    let docs = if extended {
        all_documents()
    } else {
        documents()
    };
    let (tasks, definitions) = if extended {
        long_tasks()
    } else {
        (Vec::new(), Vec::new())
    };
    let path = PathBuf::from(std::env::var_os("GLM53_GGUF").unwrap());
    let gguf = GgufFile::open(&path).unwrap();
    let pin: Value = serde_json::from_slice(Q4_PIN).unwrap();
    assert_eq!(gguf.shards.len(), 6);
    assert_eq!(gguf.tensors.len(), 1412);
    for (shard, expected) in gguf.shards.iter().zip(pin["shards"].as_array().unwrap()) {
        assert_eq!(
            shard.path.file_name().unwrap().to_str().unwrap(),
            expected["name"].as_str().unwrap()
        );
        assert_eq!(shard.mmap_len() as u64, expected["size"].as_u64().unwrap());
    }
    let header_sha = sha256_hex(gguf.primary_shard().mmap_bytes());
    assert_eq!(header_sha, pin["shards"][0]["sha256"].as_str().unwrap());
    let layout = artifact_layout(&gguf);
    let fixture: Value = serde_json::from_slice(FROZEN).unwrap();
    assert_ne!(layout, fixture["artifact_layout"]);
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let artifact = Glm5NextArtifactLayout::inspect(&gguf).unwrap();
    assert!(artifact.packed_prefill());
    let config = artifact.config();
    assert_eq!(config.blocks.len(), 45);
    assert_eq!(config.leading_dense_block_count, 3);
    for block in 3..45 {
        for (suffix, dtype, shape) in [
            ("down", down_dtype(block), [2048, 4096, 288]),
            ("gate", gate_up_dtype(block), [4096, 2048, 288]),
            ("up", gate_up_dtype(block), [4096, 2048, 288]),
        ] {
            let t = gguf
                .tensors
                .iter()
                .find(|t| t.name == format!("blk.{block}.ffn_{suffix}_exps.weight"))
                .unwrap();
            assert_eq!(t.dtype, dtype);
            assert_eq!(t.shape, shape);
        }
    }
    for block in Q8_MLA_BLOCKS.into_iter().chain([45]) {
        for (suffix, shape) in [("k", [256, 512, 64]), ("v", [512, 256, 64])] {
            let t = gguf
                .tensors
                .iter()
                .find(|t| t.name == format!("blk.{block}.attn_{suffix}_b.weight"))
                .unwrap();
            assert_eq!(t.dtype, GgmlType::Q8_0);
            assert_eq!(t.shape, shape);
        }
    }
    assert_eq!(gguf.tensors.iter().filter(|t| t.name.ends_with(".attn_k_b.weight") || t.name.ends_with(".attn_v_b.weight")).count(), 24);
    let vocab = config.vocab_size;
    assert!(docs.iter().all(|d| d.tokens.iter().all(|&t| t < vocab)));
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
    let capacity = extended_capacity(&docs, &tasks);
    assert_eq!(capacity, if extended { 5264 } else { 4416 });
    assert!(capacity <= config.context_length as usize);
    let terms = if extended {
        extended_cpu_terms(vocab)
    } else {
        cpu_terms(vocab, false)
    };
    let cpu = terms
        .iter()
        .try_fold(0u64, |n, (_, bytes)| n.checked_add(*bytes))
        .unwrap();
    let ctx = MetalContext::new().unwrap();
    assert!(
        ctx.tensor_ops_status().is_available(),
        "actual M5 tensor tier required"
    );
    assert!(crate::metal::moe_grouped_tensor_selected(&ctx));
    let mut shapes = BTreeMap::<String, usize>::new();
    for t in &gguf.tensors {
        *shapes
            .entry(format!("{:?} {:?}", t.dtype, t.shape))
            .or_default() += 1;
    }
    let mut header = json!({"event":"header","schema":"glm53.m5_transfers.v1","model":path,
        "artifact_layout":layout,"stock_fixture_artifact_layout":fixture["artifact_layout"],
        "artifact_layout_matches_frozen_fixture":false,"artifact_pin":pin,"metadata_shard_sha256":header_sha,
        "artifact_identity_scope":"pinned shard names/sizes and metadata-only primary SHA; actual tensor-table digest and retained stamps recorded; weight payload hashes not verified",
        "shards":stamps.iter().map(|s|json!({"path":s.path,"shard":s.shard_idx,"bytes":s.size,
            "device":s.device,"inode":s.inode,"mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,
            "ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec})).collect::<Vec<_>>(),
        "config":format!("{config:?}"),"tensor_shape_counts":shapes,
        "executed_expert_inventory":{"main_blocks":45,"routed_blocks":42,"q5_down":Q5_DOWN_BANKS,
            "q6_down":3,"q6_down_blocks":Q6_DOWN_BLOCKS,"q4_gate_up":82,
            "q5_gate_up":2,"q5_gate_up_blocks":Q5_GATE_UP_BLOCKS},
        "binary":std::env::current_exe().unwrap(),"binary_sha256":binary_hash(),
        "fixture_sha256":sha256_hex(FROZEN),"probe_sha256":sha256_hex(include_bytes!("tensor_m5_transfers.rs")),
        "dense_source_sha256":sha256_hex(include_bytes!("../../tensor_prefill.rs")),
        "q5_source_sha256":sha256_hex(include_bytes!("../../tensor_q5_down.rs")),
        "packed_source_sha256":sha256_hex(include_bytes!("../../packed.rs"))});
    let Value::Object(protocol) = json!({"arms":ARMS,"arm_columns":["label","q8_wide","q5_down"],"dense_scope":true,"down_scope":true,
        "scope":"independent thread-local overrides during whole prefill only; no combined candidate; ordinary forward continuation unchanged",
        "documents":docs.iter().map(|d|json!({"path":d.path,"stratum":d.stratum,"prefix":d.prefix,
            "tokens":d.tokens,"tokens_sha256":token_hash(&d.tokens)})).collect::<Vec<_>>(),
        "token_source":"frozen first-in-stratum quality-v1 IDs, u32 little-endian hashes; no retokenization or stock-artifact equivalence claim",
        "capacity":capacity,"workspace_rows":ROWS,"cpu_reserve_bytes":cpu,"cpu_terms":terms,
        "retained_baselines":1,"baseline_rows":CONTINUATION,"vocab":vocab,"targets_per_arm":192,
        "lineage":"Fast","precision":"Half","tensor_tier":format!("{:?}",ctx.tensor_ops_status()),
        "environment": (["MTL_DEBUG_LAYER","QWEN_METAL_LEASE_WAIT","QWEN_MATMAT_Q6_K_N64_MIN_N",
            "QWEN_MATMAT_Q6_K_N64","QWEN_EXPERT_TENSOR_TILE","QWEN_GLM53_DENSE_Q8_WIDE","QWEN_GLM53_TENSOR_Q5_DOWN"]
            .map(|key| (key,std::env::var(key).ok()))),
        "plain_prompts":"first frozen long document, prefixes 2048/4096; ABBA then BAAB CURRENT vs each candidate",
        "timing":"diagnostic only, validation ON; ordinary prefill includes GPU wait and normal logits readback, excludes allocation/checks/JSON; no census/profiler in plain samples; parent qwen-bench same binary without DEBUG supplies canonical performance evidence",
        "policy":"descriptive full-vocab arithmetic/NLL/KL/top1 screen, no equality/quality threshold or promotion; incomplete without complete event"})
    else {
        unreachable!()
    };
    header.as_object_mut().unwrap().extend(protocol);
    header["q8_mla_inventory"] = json!({"main_blocks":Q8_MLA_BLOCKS,"executed_banks":22,
        "serialized_pairs":12,"unused_mtp_block":45,"dtype":"Q8_0",
        "attn_k_b_shape":[256,512,64],"attn_v_b_shape":[512,256,64]});
    if extended {
        header["schema"] = json!("glm53.m5_transfers_extended_quality.v1");
        header["model"] = json!(path.file_name().unwrap());
        header["binary"] = json!(std::env::current_exe().unwrap().file_name().unwrap());
        header["shards"] = json!(
            stamps
                .iter()
                .map(|s| json!({"name":s.path.file_name().unwrap(),
            "shard":s.shard_idx,"bytes":s.size,"device":s.device,"inode":s.inode,
            "mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,
            "ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec}))
                .collect::<Vec<_>>()
        );
        header["artifact_pin_sha256"] = json!(sha256_hex(Q4_PIN));
        header["parent_packet_sha256"] = json!(sha256_hex(include_bytes!("tensor_prefill.rs")));
        header["tool_policy_source_sha256"] = json!(sha256_hex(include_bytes!("../quality.rs")));
        header["generation_source_sha256"] = json!(sha256_hex(include_bytes!("../natural.rs")));
        header["sampler_source_sha256"] = json!(sha256_hex(include_bytes!("../../../sampling.rs")));
        header["tokenizer_source_sha256"] =
            json!(sha256_hex(include_bytes!("../../../tokenizer.rs")));
        header["down_source_sha256"] = json!(sha256_hex(include_bytes!("../../tensor_down.rs")));
        header["down_adapter_source_sha256"] = json!(sha256_hex(include_bytes!(
            "../../../metal/expert_tensor_down.rs"
        )));
        header["token_source"] = json!(
            "all frozen quality-v1 documents in fixture order, maximum frozen prefix; T09-T12 frozen prompt IDs; u32 little-endian hashes; no retokenization or stock-artifact/tokenizer equivalence claim"
        );
        header["document_selection"] = json!({"documents":22,"strata":{"prose":8,"code":8,"long":6},
            "prefixes":{"prose":2048,"code":2048,"long":4352},
            "overlap":"includes existing three-document screen; not independent replication"});
        header["arm_selection"] = json!({"CURRENT":{"dense":true,"iq3_down":true,"q8_wide":false,"q5_down":false},
            "Q8WIDE":{"dense":true,"iq3_down":true,"q8_wide":true,"q5_down":false},
            "Q5DOWN":{"dense":true,"iq3_down":true,"q8_wide":false,"q5_down":true}});
        header["arithmetic"] = json!({"reference":"CURRENT, same frozen document and target position",
            "metrics":["changed_bits_components","max_abs","KL(CURRENT || arm)","NLL","NLL-CURRENT_NLL","top1","top1_hit","top1_agrees_current"],
            "full_vocab":vocab,"continuation_targets":CONTINUATION,"teacher_forcing":true,
            "baseline_lifetime":"one document at a time, 64 full-vocabulary f32 rows; candidates do not replace CURRENT",
            "prefill":"Fast/Half, F32 storage with Half tensor operands and F32 accumulation; reduction order may differ",
            "decode":"normal forward outside all prefill override scopes"});
        header["targets_per_arm"] = json!(1408);
        header["tool_tasks"] = json!(tasks.iter().map(|t|json!({"id":t.id,"context_level":t.context_level,
            "prompt_tokens":t.prompt.len(),"prompt_ids":t.prompt,"prompt_sha256":token_hash(&t.prompt),
            "expected":t.expected,"requested_capacity":t.prompt.len()+TOOL_CAP,"capacity":t.capacity()})).collect::<Vec<_>>());
        header["tool_generations"] = json!([
            Generation::Greedy { cap: TOOL_CAP }.describe(),
            Generation::Sampled {
                seed: 11,
                cap: TOOL_CAP
            }
            .describe()
        ]);
        header["tool_definitions"] = json!(tool_definitions().0);
        header["tool_policy"] = json!({"source":"parent tool_quality",
            "prompt_check":"long_tasks verifies frozen definitions/rendering/expectations; frozen IDs authoritative",
            "chat_stops":chat::CHAT_STOPS,"cap":TOOL_CAP,"stop_before_emitting":true,
            "last_sample_not_forwarded":true,"decode":"NativeTokenizer::from_gguf, generated IDs only",
            "parse":"parse_turn with frozen definitions","correctness":"record calls, parse_error, stop, call_correct; no outcome assertions",
            "scope":"T09/T10: 1455-row prefixes, Q5 down ineligible; T11/T12: 2048/2048/912 chunks, two Q5-eligible chunks; Q8 eligibility is separate; no census or dispatch-count claim"});
        header["decoder"] = json!({"implementation":"NativeTokenizer::from_gguf (actual Q4 artifact)",
            "cpu_reserve_bytes":TOKENIZER_BYTES,"construction":"after full CPU/GPU admission, before weight load"});
        header["memory_scope"] = json!(
            "normal full-model/session preflight at max(document prefix+64, tool capacity)=5264, workspace 2048; same CPU reserve on every fresh session; one retained full-vocab CURRENT baseline plus live logits/KL/report allowances, tokenizer construction/retention/decode, sampler and tool reports; inherited unused census allowance retained conservatively; no admission exception or unpriced expansion"
        );
        header["plain_prompts"] = json!([]);
        header["timing"] = json!(
            "quality and tools only; no plain samples, warm census, profiler, timing or performance verdict"
        );
        header["completion_checks"] = json!({"document_arms":66,"quality_rows":4224,"targets_per_arm":1408,
            "tool_tasks_per_arm":4,"tool_generations_per_task_arm":2,"tool_generations":24,
            "tool_token_rows":"observed, 1..256 per generation; stops+caps=24",
            "sessions":90,"plain_samples":0,"profiled_calls":0,"census_calls":0,
            "state":"fresh/ledger/position/logit checks throughout, final observers clear and retained shard stamps equal; complete event required"});
        header["policy"] = json!(
            "descriptive full-vocab quality and tool outcomes only; no posthoc thresholds, equality/noninferiority claim, promotion or verdict; private mode-0600 fresh report contains frozen IDs and generated text, do not publish raw inputs/outputs or private paths"
        );
    }
    out.emit(header);
    let (tokenizer, weights) = {
        let _allocation = ctx.begin_allocation_transaction();
        let preflight = preflight_session(&ctx, &gguf, artifact.model(), capacity, ROWS).unwrap();
        assert_eq!(preflight.ledger.prefill_rows(), ROWS as u64);
        let admission = crate::metal::evaluate_metal_memory_admission(
            preflight.ledger.peak_bytes().checked_add(cpu).unwrap(),
            0,
            ctx.memory_signals(),
            true,
        );
        out.emit(json!({"event":"preload_admission","capacity":capacity,"workspace_rows":ROWS,
            "full_preflight_peak_bytes":preflight.ledger.peak_bytes(),"ledger":preflight.ledger.terms(),
            "cpu_reserve_bytes":cpu,"admitted":admission.is_admitted(),"reason":admission.reason.as_str(),
            "required_bytes":admission.required_bytes,"allow_zero_process_budget":true,
            "cpu_charged_to_nonzero_process_budget":true,"cpu_charged_to_working_set_headroom":true,
            "current_gpu_bytes":admission.signals.current_allocated_bytes,
            "process_limit_remaining_bytes":admission.signals.process_limit_remaining_bytes,
            "working_set_headroom_bytes":admission.working_set_headroom_bytes}));
        assert!(
            admission.is_admitted(),
            "full model + session + CPU refused: {:?}",
            admission.reason
        );
        let tokenizer = extended.then(|| {
            let tokenizer = NativeTokenizer::from_gguf(&gguf).unwrap();
            assert_eq!(tokenizer.n_vocab(), vocab);
            tokenizer
        });
        (tokenizer, Glm5NextWeights::load(&ctx, &gguf).unwrap())
    };
    assert_eq!(weights.config, *config);
    if !extended {
        profile(&ctx, &weights, &docs[0].tokens[..ROWS], cpu, &mut out);
        plain(&ctx, &weights, &docs[2], cpu, &mut out);
    }
    let (document_arms, quality_rows) = quality(&ctx, &weights, &docs, cpu, &mut out);
    assert_eq!(document_arms, docs.len() * ARMS.len());
    assert_eq!(quality_rows, document_arms * CONTINUATION);
    let tools = extended.then(|| {
        transfer_tool_quality(
            &ctx,
            &weights,
            tokenizer.as_ref().unwrap(),
            &tasks,
            &definitions,
            cpu,
            &mut out,
        )
    });
    assert_plain();
    assert_eq!(gguf.revalidate_retained_shard_stamps().unwrap(), stamps);
    if let Some(tools) = tools {
        assert_eq!(document_arms, 66);
        assert_eq!(quality_rows, 4224);
        assert_eq!(tools.generations, 24);
        assert_eq!(tools.stops + tools.caps, tools.generations);
        assert!((tools.generations..=tools.generations * TOOL_CAP).contains(&tools.tokens));
        out.emit(json!({"event":"complete","execution_complete":true,"arms":3,"documents_per_arm":22,
            "document_arms":document_arms,"quality_rows_per_document_arm":CONTINUATION,
            "targets_per_arm":1408,"quality_rows":quality_rows,"tool_tasks_per_arm":4,
            "tool_generations_per_task_arm":2,"tool_generations":tools.generations,
            "tool_token_rows":tools.tokens,"tool_chat_stops":tools.stops,"tool_caps":tools.caps,
            "sessions":document_arms+tools.generations,"plain_samples":0,"profiled_calls":0,"census_calls":0,
            "retained_shard_stamps_unchanged":true,"observers_clear":true,
            "quality_verdict":null,"performance_verdict":null}));
    } else {
        out.emit(
            json!({"event":"complete","execution_complete":true,"arms":3,"documents_per_arm":3,
        "targets_per_arm":192,"quality_rows":576,"plain_samples":32,"profiled_calls":3,
        "quality_verdict":null,"performance_verdict":null}),
        );
    }
    out.sync();
}
