//! Fixed quality-v1 subset; forced Half scalar/MMA prefill, unchanged decode.
//! GLM53_GGUF=Maya GLM53_LATENT_QUALITY_OUT=new.jsonl; production perf lease.
//! No historical quality verdict, timing claim, or production promotion.
use super::*;
use crate::metal::{
    dispatch_census_begin, dispatch_census_take, with_grouped_projection_scalar_baseline_override,
};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Write},
};

const FROZEN: &[u8] = include_bytes!("../../../../../../scripts/reference/glm53/quality-v1.json");
const ROWS: usize = 2048;
const TASKS: [&str; 4] = ["T03", "T10", "T11", "T12"];
const ARMS: [(&str, bool); 2] = [("half_scalar", true), ("half_mma", false)];
const UD_SHA: &str = "632f4fabe03ea73d20c299c8ef00c4a16e47dd9ab3f1750a8585b92eb70dfa74";

fn selection(f: &Value) -> (Vec<(&Value, usize)>, Vec<&Value>) {
    assert_eq!(f["name"], "glm53-quality-v1");
    assert_eq!(f["continuation"], CONTINUATION);
    assert_eq!(f["tool_cap"], TOOL_CAP);
    let docs = [("prose", 2048), ("code", 2048), ("long", 4352)].map(|(stratum, n)| {
        let d = f["documents"]
            .as_array()
            .unwrap()
            .iter()
            .find(|d| d["stratum"] == stratum)
            .unwrap();
        assert!(d["prefix_lengths"].as_array().unwrap().contains(&json!(n)));
        assert!(parse_ids(&d["tokens"]).len() >= n + CONTINUATION);
        (d, n)
    });
    let tasks = TASKS.map(|id| {
        f["tool_tasks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["id"] == id)
            .unwrap()
    });
    (docs.into(), tasks.into())
}
fn emit(out: &mut File, value: Value) {
    serde_json::to_writer(&mut *out, &value).unwrap();
    writeln!(out).unwrap();
    out.flush().unwrap(); // Process-exit evidence; no per-event filesystem sync.
}
fn take<'a>(s: &mut &'a [u8], n: usize) -> &'a [u8] {
    let (a, b) = s.split_at_checked(n).expect("bounded GGUF header");
    *s = b;
    a
}
fn u32le(s: &mut &[u8]) -> u32 {
    u32::from_le_bytes(take(s, 4).try_into().unwrap())
}
fn u64le(s: &mut &[u8]) -> usize {
    usize::try_from(u64::from_le_bytes(take(s, 8).try_into().unwrap())).unwrap()
}
fn skip(s: &mut &[u8], ty: u32) {
    match ty {
        8 => {
            let n = u64le(s);
            take(s, n);
        }
        9 => {
            let t = u32le(s);
            assert_ne!(t, 9);
            let n = u64le(s);
            assert!(n <= 1_000_000);
            for _ in 0..n {
                skip(s, t);
            }
        }
        _ => {
            let n = match ty {
                0 | 1 | 7 => 1,
                2 | 3 => 2,
                4 | 5 | 6 => 4,
                10 | 11 | 12 => 8,
                _ => panic!("GGUF type"),
            };
            take(s, n);
        }
    }
}
// Hash full raw typed metadata values, including every array element; the
// general GGUF JSON view truncates long arrays. No tensor payload is accessed.
fn tokenizer_fields(mut s: &[u8]) -> BTreeMap<String, String> {
    assert_eq!(take(&mut s, 4), b"GGUF");
    assert_eq!(u32le(&mut s), 3);
    let _tensors = u64le(&mut s);
    let count = u64le(&mut s);
    let mut fields = BTreeMap::new();
    for _ in 0..count {
        let n = u64le(&mut s);
        let key = std::str::from_utf8(take(&mut s, n)).unwrap();
        let start = s;
        let ty = u32le(&mut s);
        skip(&mut s, ty);
        if key.starts_with("tokenizer.") {
            assert!(
                fields
                    .insert(key.into(), sha256_hex(&start[..start.len() - s.len()]))
                    .is_none()
            );
        }
    }
    for key in [
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.pre",
    ] {
        assert!(fields.contains_key(key));
    }
    fields
}
fn binary_hash(path: &std::path::Path) -> String {
    let mut file = File::open(path).unwrap();
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
fn fresh<'w>(
    ctx: &MetalContext,
    w: &'w Glm5NextWeights,
    prefix: usize,
    extra: usize,
    out: &mut File,
) -> Glm5NextSession<'w> {
    // Existing fixture/tokenizer allocations are resident before admission.
    // Two returned logits rows; sampler-v1 (top-k off): max(keys+Candidates,
    // Candidates+F64 weights)=24*vocab, plus its 65536-usize radix histogram.
    // Text decoding/parsing runs after the GPU session is dropped.
    let terms = [
        2 * u64::from(w.config.vocab_size) * 4,
        u64::from(w.config.vocab_size) * 24,
        (65536 * std::mem::size_of::<usize>()) as u64,
        (prefix + extra) as u64 * 4,
        TOOL_CAP as u64 * 8,
        CONTINUATION as u64 * 9,
    ];
    let cpu = terms.iter().sum();
    let capacity = ROWS.max(prefix + extra);
    emit(
        out,
        json!({"event":"session_begin","prefix":prefix,"capacity":capacity,"workspace_rows":ROWS,
        "prefill_chunks":(0..prefix).step_by(ROWS).map(|i|(prefix-i).min(ROWS)).collect::<Vec<_>>(),
        "cpu_reserve_bytes":cpu,"cpu_terms":terms,"cpu_names":["two_logits","sampler_peak_vectors","radix_histogram","ids","emitted_ids","nll_hits"],
        "current_gpu_bytes":ctx.current_allocated_size()}),
    );
    let s =
        Glm5NextSession::with_prefill_rows_and_cpu_reserve(ctx, w, capacity, ROWS, cpu).unwrap();
    assert_eq!(s.packed_lineage(), Some(PackedLineage::Fast));
    assert_eq!(s.fast_precision(), Some(FastPrecision::Half));
    s
}
fn prefill(
    ctx: &MetalContext,
    s: &mut Glm5NextSession<'_>,
    ids: &[u32],
    baseline: bool,
) -> Vec<f32> {
    // Scope covers ALL chunks, including 2048+2048+256 for the long document.
    with_grouped_projection_scalar_baseline_override(baseline, || s.prefill_packed(ctx, ids))
        .unwrap()
}

#[test]
#[ignore = "bounded Maya Half latent quality subset: GLM53_GGUF, new GLM53_LATENT_QUALITY_OUT, release, no MTL_DEBUG_LAYER"]
fn half_latent_mma_quality() {
    assert!(!cfg!(debug_assertions));
    let _lease = perf_lease();
    let fixture: Value = serde_json::from_slice(FROZEN).unwrap();
    let (documents, tasks) = selection(&fixture); // fixed before any model output
    let path = PathBuf::from(std::env::var_os("GLM53_GGUF").expect("Maya GGUF"));
    let gguf = GgufFile::open(&path).unwrap();
    let ud_path = std::env::var_os("GLM53_LATENT_QUALITY_UD_HEADER")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS
                .required()
                .clone()
        });
    let ud_file = File::open(&ud_path).unwrap();
    assert_eq!(
        ud_file.metadata().unwrap().len(),
        9429984,
        "metadata-only UD shard"
    );
    let mut ud = Vec::with_capacity(9429984);
    ud_file.take(9429984).read_to_end(&mut ud).unwrap();
    assert_eq!(ud.len(), 9429984);
    assert_eq!(sha256_hex(&ud), UD_SHA, "pinned UD metadata-only shard");
    assert_eq!(&ud[8..16], &0u64.to_le_bytes());
    let fields = tokenizer_fields(&ud);
    assert_eq!(
        fields,
        tokenizer_fields(
            &gguf.primary_shard().mmap_bytes()[..gguf.primary_shard().tensor_data_start as usize]
        ),
        "frozen UD token IDs require identical complete tokenizer metadata"
    );
    drop(ud);
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let definitions: Vec<_> = fixture["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| ToolDefinition::from_value(v).unwrap())
        .collect();
    for t in &tasks {
        let prompt = render(
            &[Message::User(t["user"].as_str().unwrap().into())],
            &definitions,
            Effort::Low,
        );
        let encoded: Vec<u32> = tokenizer
            .encode(&prompt, false)
            .unwrap()
            .into_iter()
            .map(|x| x as u32)
            .collect();
        assert_eq!(
            encoded,
            parse_ids(&t["prompt"]),
            "frozen task prompt/template"
        );
    }
    let mut out = File::options()
        .write(true)
        .create_new(true)
        .open(std::env::var_os("GLM53_LATENT_QUALITY_OUT").expect("new JSONL path"))
        .unwrap();
    let exe = std::env::current_exe().unwrap();
    let stamps: Vec<_> = gguf.revalidate_retained_shard_stamps().unwrap().into_iter().map(|s|json!({"path":s.path,"shard":s.shard_idx,"device":s.device,"inode":s.inode,"bytes":s.size,"mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,"ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec})).collect();
    emit(
        &mut out,
        json!({"event":"header","schema":"glm53.latent_quality.v1","fixture_sha256":sha256_hex(FROZEN),
        "model":path,"artifact_layout":artifact_layout(&gguf),"shards":stamps,"binary":exe,"binary_sha256":binary_hash(&exe),
        "tokenizer_compatibility":{"verified":true,"ud_header":ud_path,"ud_header_sha256":UD_SHA,"method":"all tokenizer.* raw typed GGUF values, complete arrays","fields":fields},
        "documents":documents.iter().map(|(d,n)|json!({"path":d["path"],"stratum":d["stratum"],"prefix":n})).collect::<Vec<_>>(),
        "tasks":TASKS,"continuation":CONTINUATION,"tool_cap":TOOL_CAP,"seeds":[11],"arms":["half_scalar","half_mma"],"lineage":"Fast","precision":"Half","workspace_rows":ROWS,
        "policy":"override around entire prefill in both arms; fresh sessions; decode unchanged; no promotion, verdict, or timing claim",
        "source_binding":{"packet":sha256_hex(include_bytes!("latent_mma.rs")),"quality":sha256_hex(include_bytes!("../quality.rs")),"natural":sha256_hex(include_bytes!("../natural.rs")),
            "packed":sha256_hex(include_bytes!("../../packed.rs")),"session":sha256_hex(include_bytes!("../../../glm5_next_metal.rs")),"latent":sha256_hex(include_bytes!("../../../metal/latent.rs")),
            "sampling":sha256_hex(include_bytes!("../../../sampling.rs")),"tokenizer":sha256_hex(include_bytes!("../../../tokenizer.rs")),
            "mma_shader":sha256_hex(include_bytes!("../../../../../../kernels/latent_grouped_mma.metal")),"metallib":sha256_hex(crate::KERNELS_METALLIB)}}),
    );
    let ctx = MetalContext::new().unwrap();
    let w = Glm5NextWeights::load(&ctx, &gguf).unwrap(); // one resident model
    let warm_ids = parse_ids(&documents[0].0["tokens"]);
    let mut other_signature = None;
    for (arm, baseline) in ARMS {
        let mut s = fresh(&ctx, &w, ROWS, CONTINUATION, &mut out);
        dispatch_census_begin();
        let logits = prefill(&ctx, &mut s, &warm_ids[..ROWS], baseline);
        let census = dispatch_census_take();
        assert_finite("warm", &logits);
        let latent = |r: &&crate::metal::DispatchCensusRow| {
            r.grid_depth == u64::from(w.config.head_count)
                && r.kernel.starts_with("kernel_latent_grouped_q6_K_f32")
        };
        let q6: Vec<_> = census.iter().filter(latent).collect();
        assert_eq!(q6.len(), 22);
        let expected = if baseline {
            "kernel_latent_grouped_q6_K_f32"
        } else {
            "kernel_latent_grouped_q6_K_f32_r2c16k64"
        };
        assert!(q6.iter().all(|r| r.kernel == expected));
        assert!(q6.iter().all(|r| if baseline {
            r.grid_height == 512
        } else {
            r.grid_width == 16
        }));
        let other: Vec<_> = census
            .iter()
            .filter(|r| !latent(r))
            .map(|r| {
                (
                    &r.kernel,
                    [r.grid_width, r.grid_height, r.grid_depth],
                    [r.threads_width, r.threads_height, r.threads_depth],
                )
            })
            .collect();
        let signature = sha256_hex(&serde_json::to_vec(&other).unwrap());
        if let Some(a) = &other_signature {
            assert_eq!(a, &signature, "non-latent dispatch changed");
        } else {
            other_signature = Some(signature);
        }
        emit(
            &mut out,
            json!({"event":"warm_census","arm":arm,"q6_latent_calls":q6.len(),"kernel":expected,"other_dispatches":census.len()-q6.len(),
            "census_cpu_bytes":census.capacity()*std::mem::size_of::<crate::metal::DispatchCensusRow>()+census.iter().map(|r|r.kernel.capacity()+r.tag.as_ref().map_or(0,String::capacity)).sum::<usize>(),
            "other_dispatch_signature_sha256":other_signature,
            "scope":"dispatch selection/geometry only, no routing-data equality claim","measured":false}),
        );
    }
    for (document, n) in documents {
        let tokens = parse_ids(&document["tokens"]);
        for (arm, baseline) in ARMS {
            emit(
                &mut out,
                json!({"event":"document_begin","arm":arm,"path":document["path"],"prefix":n}),
            );
            let mut s = fresh(&ctx, &w, n, CONTINUATION, &mut out);
            let mut logits = prefill(&ctx, &mut s, &tokens[..n], baseline);
            let (mut nll, mut hits) = (
                Vec::with_capacity(CONTINUATION),
                Vec::with_capacity(CONTINUATION),
            );
            for (i, &token) in tokens[n..n + CONTINUATION].iter().enumerate() {
                assert_finite("document", &logits);
                let loss = -log_prob(&logits, token);
                assert!(loss.is_finite());
                let hit = argmax(&logits) == token as usize;
                nll.push(loss);
                hits.push(hit);
                emit(
                    &mut out,
                    json!({"event":"document_token","arm":arm,"path":document["path"],"prefix":n,"index":i,"token":token,"nll":loss,"hit":hit}),
                );
                if i + 1 < CONTINUATION {
                    logits = s.forward(&ctx, token).unwrap();
                }
            }
            drop(s);
            emit(
                &mut out,
                json!({"event":"document","arm":arm,"path":document["path"],"stratum":document["stratum"],"prefix":n,"continuation_tokens":&tokens[n..n+CONTINUATION],"nll":nll,"hits":hits}),
            );
        }
    }
    for task in tasks {
        let prompt = parse_ids(&task["prompt"]);
        for generation in [
            Generation::Greedy { cap: TOOL_CAP },
            Generation::Sampled {
                seed: 11,
                cap: TOOL_CAP,
            },
        ] {
            for (arm, baseline) in ARMS {
                emit(
                    &mut out,
                    json!({"event":"task_begin","arm":arm,"id":task["id"],"generation":generation.describe(),"prompt_tokens":prompt.len()}),
                );
                let mut s = fresh(&ctx, &w, prompt.len(), TOOL_CAP, &mut out);
                let mut logits = prefill(&ctx, &mut s, &prompt, baseline);
                let mut sampler = Sampler::new(generation.sampling()).unwrap();
                let mut emitted = Vec::with_capacity(TOOL_CAP);
                let stop = loop {
                    assert_finite("task", &logits);
                    let token = sampler.sample(&logits).unwrap().token as u32;
                    emit(
                        &mut out,
                        json!({"event":"task_token","arm":arm,"id":task["id"],"generation":generation.describe(),"index":emitted.len(),"token":token}),
                    );
                    if chat::CHAT_STOPS.contains(&(token as i32)) {
                        break Some(token);
                    }
                    emitted.push(token);
                    if emitted.len() == TOOL_CAP {
                        break None;
                    }
                    logits = s.forward(&ctx, token).unwrap();
                };
                drop(s);
                drop(logits);
                let text = tokenizer.decode(&emitted.iter().map(|&t| t as i32).collect::<Vec<_>>());
                let parsed = parse_turn(&text, &definitions);
                let (calls, error) = match parsed {
                    Ok((_, _, calls)) => (
                        calls
                            .into_iter()
                            .map(|c| json!({"name":c.name,"arguments":c.arguments}))
                            .collect::<Vec<_>>(),
                        None,
                    ),
                    Err(e) => (vec![], Some(e)),
                };
                emit(
                    &mut out,
                    json!({"event":"task","arm":arm,"id":task["id"],"expected":task["expected"],"generation":generation.describe(),"emitted":emitted.len(),"emitted_ids":emitted,"text":text,"calls":calls,"stop":stop,"parse_error":error}),
                );
            }
        }
    }
    emit(
        &mut out,
        json!({"event":"complete","documents_per_arm":3,"task_generations_per_arm":8}),
    );
}

#[test]
fn frozen_subset_is_three_documents_and_four_tasks() {
    let f: Value = serde_json::from_slice(FROZEN).unwrap();
    let (docs, tasks) = selection(&f);
    assert_eq!(
        docs.iter()
            .map(|(d, n)| (d["path"].as_str().unwrap(), *n))
            .collect::<Vec<_>>(),
        [
            ("docs/GLM53-FLASH-PLAN.md", 2048),
            ("crates/qwen-llm/src/sampling.rs", 2048),
            ("docs/SERVE.md", 4352)
        ]
    );
    assert_eq!(
        tasks
            .iter()
            .map(|t| t["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        TASKS
    );
    for task in tasks {
        assert_eq!(
            parse_ids(&task["prompt"]).len(),
            task["prompt_tokens"].as_u64().unwrap() as usize
        );
        let expected = TOOL_TASKS.iter().find(|t| task["id"] == t.0).unwrap().3;
        assert_eq!(task["expected"], expected.describe());
    }
}
