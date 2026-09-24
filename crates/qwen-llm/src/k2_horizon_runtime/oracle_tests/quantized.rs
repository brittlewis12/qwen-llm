//! Same-artifact integration screen, not a quantization-quality comparison or holdout.
use super::*;

const Q4_SHA256: &str = "eb89c15a0ae9712be2ee462bf43802de14200f20f93b73da6eb68c2ebdd28e4e";
const MAX_ABS: f64 = 0.005;
const RMSE: f64 = 0.001;
const REGRET: f64 = 0.001;

fn compare(actual: &[f32], reference: &[f32]) -> serde_json::Value {
    let logits = error_metrics(actual, reference);
    let a = logits["actual_top1"].as_u64().unwrap() as usize;
    let r = logits["reference_top1"].as_u64().unwrap() as usize;
    let native_regret = f64::from(actual[a]) - f64::from(actual[r]);
    let reference_regret = f64::from(reference[r]) - f64::from(reference[a]);
    let pass = logits["max_abs"].as_f64().unwrap() < MAX_ABS
        && logits["rmse"].as_f64().unwrap() < RMSE
        && native_regret < REGRET
        && reference_regret < REGRET;
    json!({"logits":logits, "native_regret":native_regret,
        "reference_regret":reference_regret, "pass":pass})
}

#[test]
fn same_artifact_screen_accepts_only_bounded_numerics_and_reciprocal_near_ties() {
    assert_eq!(compare(&[0., 0.0005], &[0.0005, 0.])["pass"], true);
    let below = f32::from_bits(0.001_f32.to_bits() - 1);
    assert_eq!(compare(&[0., below], &[below, 0.])["pass"], true);
    assert_eq!(compare(&[0., 0.001], &[0.001, 0.])["pass"], false);
    let reference = vec![1.; 128];
    let mut actual = reference.clone();
    actual[0] += 0.006;
    let isolated = compare(&actual, &reference);
    assert!(isolated["logits"]["rmse"].as_f64().unwrap() < RMSE);
    assert_eq!(isolated["pass"], false);
    for (a, r) in [
        (vec![0., 0.002], vec![0.0001, 0.]),
        (vec![0., 0.0001], vec![0.002, 0.]),
        (vec![1., 2.], vec![1.01, 2.01]),
    ] {
        assert_eq!(compare(&a, &r)["pass"], false);
    }
    for (a, r) in [(vec![f32::NAN], vec![1.]), (vec![1.], vec![])] {
        assert!(std::panic::catch_unwind(|| compare(&a, &r)).is_err());
    }
}

fn artifact(source: &GgufFile, path: &Path) -> serde_json::Value {
    assert_eq!(file_sha256(path), Q4_SHA256);
    let prepared = K2PreparedArtifact::inspect(source).unwrap();
    assert_eq!(prepared.generation_stops().unwrap(), [1]);
    let tokenizer = format!(
        "{:016x}",
        crate::runtime::tokenizer_metadata_identity(source)
    );
    assert_eq!(tokenizer, "51ebd8140ea2abd9");
    let identity =
        crate::checkpoint_identity::verified_checkpoint_content_identity(source).unwrap();
    let content = identity
        .content_id
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let tensors = source
        .tensors
        .iter()
        .map(|t| {
            json!({
                "name":t.name,"dtype":format!("{:?}",t.dtype),"shape":t.shape,"bytes":t.n_bytes
            })
        })
        .collect::<Vec<_>>();
    let template = source.get_str("tokenizer.chat_template").unwrap();
    json!({"model_sha256":Q4_SHA256,"content_blake3":content,
        "tokenizer_metadata_id":tokenizer,"tensors":tensors,
        "gguf_template_sha256":format!("{:x}", Sha256::digest(template.as_bytes())),
        "generation_stops":[1], "chat_authorized":false})
}

#[test]
#[ignore = "CPU only: pinned final Q4_K_M census, retained content and generation admission"]
fn cpu_final_q4_artifact_preflight() {
    let path = PathBuf::from(std::env::var("K2_GGUF").unwrap());
    let source = GgufFile::open(&path).unwrap();
    let report = artifact(&source, &path);
    eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn reference_memory_preflight(model_bytes: u64) -> serde_json::Value {
    let physical = crate::metal::host_physical_memory_bytes().unwrap();
    let wired = crate::cache_probe::wired_memory_bytes().unwrap();
    assert!(
        !crate::metal::host_wired_memory_is_unsafe(wired, physical),
        "unsafe wired memory before oracle child"
    );
    let available = crate::cache_probe::available_memory_bytes().unwrap();
    // This small singleton oracle is not the native allocator. Conservatively
    // reserve two weight copies plus 2 GiB for its 256-slot KV, graph and host I/O.
    let required = model_bytes
        .checked_mul(2)
        .unwrap()
        .checked_add(2 << 30)
        .unwrap();
    assert!(
        available > required,
        "insufficient memory for standalone oracle: {available} <= {required}"
    );
    json!({"physical":physical,"wired":wired,"available":available,"required":required})
}

fn embedding_check(model: &K2LoadedModel<'_>, source: &GgufFile) -> serde_json::Value {
    let ctx = model.ctx;
    let desc = source
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .unwrap();
    let ids = [0_i32, 42, 250019, 250623];
    let input =
        MetalTensor::from_bytes(ctx, bytemuck::cast_slice(&ids), vec![4], GgmlType::I32).unwrap();
    let output = MetalTensor::zeros_f32(ctx, vec![4096, 4]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encode_get_rows_f32(
        ctx,
        &encoder,
        &model.weights.embedding,
        &input,
        &output,
        4,
        4096,
    )
    .unwrap();
    encoder.end();
    command.commit();
    crate::metal::wait_unchecked(&command);
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    let row_bytes = desc.n_bytes as usize / 250624;
    let mut row_desc = desc.clone();
    row_desc.shape = vec![4096];
    row_desc.n_bytes = row_bytes as u64;
    let bytes = source.slice(desc);
    let mut maximum = 0_f32;
    for (row, id) in ids.iter().enumerate() {
        let start = *id as usize * row_bytes;
        let expected =
            crate::codec::dequant_to_f32(&row_desc, &bytes[start..start + row_bytes]).unwrap();
        for (&actual, expected) in read_f32(&output)[row * 4096..(row + 1) * 4096]
            .iter()
            .zip(expected)
        {
            assert!(actual.is_finite() && expected.is_finite());
            maximum = maximum.max((actual - expected).abs());
        }
    }
    json!({"ids":ids,"dtype":format!("{:?}",desc.dtype),"max_abs":maximum,"pass":maximum < 1e-6})
}

#[test]
#[ignore = "GPU final Q4_K_M same-artifact screen; pinned oracle, production lease and memory gates"]
fn gpu_final_q4_matches_same_artifact_oracle() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let path = PathBuf::from(std::env::var("K2_GGUF").unwrap());
    let binary = PathBuf::from(std::env::var("K2_LLAMA_ORACLE").unwrap());
    let identity = Command::new(&binary).arg("--identity").output().unwrap();
    assert!(identity.status.success());
    assert_eq!(
        String::from_utf8(identity.stdout).unwrap(),
        reference_identity()
    );
    let source = GgufFile::open(&path).unwrap();
    let stamps = source.revalidate_retained_shard_stamps().unwrap();
    let artifact = artifact(&source, &path);
    let tokenizer = NativeTokenizer::from_gguf(&source).unwrap();
    let encode = |text| {
        tokenizer
            .encode(text, true)
            .unwrap()
            .into_iter()
            .map(|id| u32::try_from(id).unwrap())
            .collect::<Vec<_>>()
    };
    let text = encode("The capital of France is");
    assert_eq!(text.len(), 6);
    let mut code = encode(
        "fn fibonacci(n: u32) -> u32 {\n    if n < 2 { n } else { fibonacci(n - 1) + fibonacci(n - 2) }\n}\n",
    );
    assert!(code.len() >= 32);
    code.truncate(32);
    let cases = [(0, vec![0, 42, 17, 19]), (37, text), (128, code)];
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/profiles")
        .join(format!("k2-q4-screen-{}-{stamp}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    fs::write(directory.join("manifest.json"), serde_json::to_vec_pretty(&json!({
        "schema":"k2.same-artifact-screen.v1","artifact":artifact,"inputs":cases,
        "scope":"42 reused short rows; not holdout, quantization fidelity or long-context qualification",
        "bos":"native tokenizer inserts once for text/code; explicit IDs for raw case; oracle inserts none",
        "limits_exclusive":{"max_abs":MAX_ABS,"rmse":RMSE,"reciprocal_regret":REGRET,"embedding_max_abs":1e-6},
        "reference":reference_identity(),"reference_binary_sha256":file_sha256(&binary),
        "native_metallib_sha256":native_kernel_identity(),"native_sources":native_source_identity(),
        "native_capacity":32,"reference_capacity":256,"cache":"F16","metal_api_validation":true
    })).unwrap()).unwrap();
    eprintln!("Q4 screen artifacts: {}", directory.display());
    let mut references = Vec::new();
    for (base, tokens) in &cases {
        let memory = reference_memory_preflight(fs::metadata(&path).unwrap().len());
        fs::write(
            directory.join(format!("memory-{base}.json")),
            serde_json::to_vec_pretty(&memory).unwrap(),
        )
        .unwrap();
        references.push(oracle_rows(&binary, &path, &directory, *base, tokens));
    }
    assert_eq!(source.revalidate_retained_shard_stamps().unwrap(), stamps);
    let ctx = MetalContext::new().unwrap();
    let model = K2LoadedModel::load(&ctx, &source, 32).unwrap();
    // Single-token appends below use the serial graph in every mode.
    assert_eq!(model.prefill, PrefillMode::General { chunk: 256 });
    let embedding = embedding_check(&model, &source);
    let mut reports = Vec::new();
    for ((base, tokens), rows) in cases.iter().zip(references) {
        let mut session = model.create_session(*base).unwrap();
        for (step, (&token, reference)) in tokens.iter().zip(rows).enumerate() {
            let actual = session.append(&[token]).unwrap();
            let metrics = compare(&actual, &reference);
            fs::write(
                directory.join(format!("native-{base}-{step}.f32")),
                bytemuck::cast_slice(&actual),
            )
            .unwrap();
            eprintln!("base={base} step={step} {metrics}");
            reports.push(json!({"position":*base as usize+step,"token":token,"metrics":metrics}));
        }
    }
    assert_eq!(reports.len(), 42);
    let pass =
        embedding["pass"] == true && reports.iter().all(|row| row["metrics"]["pass"] == true);
    fs::write(
        directory.join("verdict.json"),
        serde_json::to_vec_pretty(&json!({
            "pass":pass,"embedding":embedding,"rows":reports
        }))
        .unwrap(),
    )
    .unwrap();
    assert!(
        pass,
        "same-Q4 integration screen failed; retained evidence at {}",
        directory.display()
    );
}
