use super::*;

pub(super) fn argmax(values: &[f32]) -> u32 {
    assert!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32
}

fn row_metrics(actual: &[f32], reference: &[f32]) -> serde_json::Value {
    let mut logits = error_metrics(actual, reference);
    logits["actual_top1"] = json!(argmax(actual));
    logits["reference_top1"] = json!(argmax(reference));
    json!({"logits":logits,"distribution":probability::metrics(actual,reference)})
}

fn row_failures(p: &serde_json::Value, metrics: &serde_json::Value) -> Vec<&'static str> {
    let mut failed = Vec::new();
    if metrics["logits"]["actual_top1"] != metrics["logits"]["reference_top1"] {
        failed.push("top1");
    }
    failed.extend(numerical_failures(p, metrics));
    failed
}

pub(super) fn numerical_failures(
    p: &serde_json::Value,
    metrics: &serde_json::Value,
) -> Vec<&'static str> {
    let mut failed = Vec::new();
    for (section, key, limit) in [
        ("logits", "max_abs", "max_abs_exclusive"),
        ("logits", "rmse", "rmse_exclusive"),
        (
            "distribution",
            "kl_reference_to_actual",
            "kl_reference_to_actual_exclusive",
        ),
        (
            "distribution",
            "total_variation",
            "total_variation_exclusive",
        ),
        ("distribution", "centered_rmse", "centered_rmse_exclusive"),
    ] {
        if !metrics[section][key]
            .as_f64()
            .is_some_and(|v| v.is_finite() && v < p[section][limit].as_f64().unwrap())
        {
            failed.push(key);
        }
    }
    if !metrics["logits"]["cosine"]
        .as_f64()
        .is_some_and(|v| v.is_finite() && v > p["logits"]["cosine_exclusive_min"].as_f64().unwrap())
    {
        failed.push("cosine");
    }
    failed
}

pub(super) fn exact_trajectory_predictor(p: &serde_json::Value, visible: usize) -> bool {
    assert!((1..=p["capacity"].as_u64().unwrap() as usize).contains(&visible));
    visible >= p["continuation"]["prefix_length"].as_u64().unwrap() as usize
}

fn generated_ids(bytes: &[u8], base: u32, prefix: &[u32]) -> Vec<u32> {
    assert_eq!(prefix.len(), 241);
    assert_eq!(bytes.len(), 16 + 256 * 4);
    assert_eq!(&bytes[..8], b"K2TOK001");
    let word = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    assert_eq!(word(8), base);
    assert_eq!(word(12), 256);
    let ids = (0..256).map(|i| word(16 + i * 4)).collect::<Vec<_>>();
    assert!(ids.iter().all(|&id| id < 250624));
    assert_eq!(&ids[..241], prefix);
    ids
}

fn bitwise_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

struct ReferenceCase {
    logits: PathBuf,
    captures: Option<Vec<Vec<f32>>>,
    greedy: Option<(PathBuf, Vec<u32>)>,
}

fn prepare(
    binary: &Path,
    model: &Path,
    directory: &Path,
    cases: &[(String, u32, Vec<u32>, bool)],
) -> Vec<ReferenceCase> {
    let mut files = Vec::new();
    let mut evidence = Vec::new();
    for (name, base, tokens, selected) in cases {
        let root = directory.join(format!("{name}-{base}"));
        let ordinary_dir = root.join("ordinary");
        fs::create_dir_all(&ordinary_dir).unwrap();
        let logits = run_oracle(binary, model, &ordinary_dir, *base, tokens);
        let mut captures = None;
        let mut greedy = None;
        let mut record = json!({"corpus":name,"base":base,"ordinary":logits,"ordinary_sha256":file_sha256(&logits)});
        if *selected {
            let capture_dir = root.join("captured");
            fs::create_dir(&capture_dir).unwrap();
            let traced = run_oracle_mode(binary, model, &capture_dir, *base, tokens, true);
            let mut a = OracleRows::new(
                BufReader::new(fs::File::open(&logits).unwrap()),
                *base,
                tokens,
                250624,
            );
            let mut b = OracleRows::new(
                BufReader::new(fs::File::open(&traced).unwrap()),
                *base,
                tokens,
                250624,
            );
            for _ in tokens {
                assert!(
                    bitwise_equal(&a.row(), &b.row()),
                    "{name} traced reference changed logits"
                );
            }
            a.finish();
            b.finish();
            let layers_path = traced.with_extension("f32.layers");
            captures = Some(decode_layers(
                &fs::read(&layers_path).unwrap(),
                *base + 255,
                tokens[255],
            ));
            record["capture_sha256"] = json!(file_sha256(&layers_path));
            record["capture_noninterference_bitwise"] = json!(true);
            let greedy_dir = root.join("greedy");
            fs::create_dir(&greedy_dir).unwrap();
            let trajectory = run_oracle_command(
                binary,
                model,
                &greedy_dir,
                *base,
                &tokens[..241],
                ReferenceRun::Greedy15,
            );
            let token_path = trajectory.with_extension("f32.tokens");
            let ids = generated_ids(&fs::read(&token_path).unwrap(), *base, &tokens[..241]);
            record["greedy_sha256"] = json!(file_sha256(&trajectory));
            record["greedy_tokens_sha256"] = json!(file_sha256(&token_path));
            record["greedy_tokens"] = json!(ids);
            greedy = Some((trajectory, ids));
        }
        files.push(ReferenceCase {
            logits,
            captures,
            greedy,
        });
        evidence.push(record);
        fs::write(
            directory.join("references.json"),
            serde_json::to_vec_pretty(&evidence).unwrap(),
        )
        .unwrap();
    }
    files
}

fn save(
    directory: &Path,
    rows: &[serde_json::Value],
    captures: &[serde_json::Value],
) -> Vec<serde_json::Value> {
    let failures = rows
        .iter()
        .chain(captures)
        .filter(|r| !r["failed_gates"].as_array().unwrap().is_empty())
        .cloned()
        .collect::<Vec<_>>();
    for (file, value) in [
        ("metrics.json", json!({"rows":rows,"captures":captures})),
        ("failures.json", json!(failures)),
    ] {
        fs::write(
            directory.join(file),
            serde_json::to_vec_pretty(&value).unwrap(),
        )
        .unwrap();
    }
    failures
}

#[test]
#[ignore = "GPU frozen guarded-256 candidate envelope; production lease; about 5 GiB evidence; no automatic promotion"]
fn gpu_frozen_guarded_256_holdout() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    assert!(crate::metal::mat_vec_q8_0_lcpp_enabled());
    let p = policy();
    let path = PathBuf::from(std::env::var("K2_GGUF").expect("K2_GGUF"));
    let binary = PathBuf::from(std::env::var("K2_LLAMA_ORACLE").expect("K2_LLAMA_ORACLE"));
    let identity = Command::new(&binary).arg("--identity").output().unwrap();
    assert!(identity.status.success());
    assert_eq!(
        String::from_utf8(identity.stdout).unwrap(),
        reference_identity()
    );
    let source = GgufFile::open(&path).unwrap();
    let cases = inputs(&source);
    let stamps = source.revalidate_retained_shard_stamps().unwrap();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/profiles")
        .join(format!("k2-holdout-v1-{}-{stamp}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    fs::write(directory.join("manifest.json"),serde_json::to_vec_pretty(&json!({
        "policy":p,"policy_sha256":POLICY_SHA256,"fixture_freeze_commit":"990bb56d",
        "model":path,"inputs":cases,"native_capacity":256,"native_cache":"F16","native_attention":"materialized",
        "native_q8_matvec":"lcpp","native_metallib_sha256":native_kernel_identity(),"native_sources":native_source_identity(),
        "reference":reference_identity().trim(),"reference_binary_sha256":file_sha256(&binary),
        "reference_cache":"F16","reference_allocated_capacity":256,
        "expected_teacher_forced_rows":3072,"expected_trajectory_rows":1024,"expected_capture_sites":16,
        "public_cap_promoted":false,"metal_api_validation":std::env::var("MTL_DEBUG_LAYER").ok(),
    })).unwrap()).unwrap();
    eprintln!("frozen holdout artifacts: {}", directory.display());
    let files = prepare(&binary, &path, &directory, &cases);
    assert_eq!(source.revalidate_retained_shard_stamps().unwrap(), stamps);
    let ctx = MetalContext::new().unwrap();
    let model = K2LoadedModel::load_unqualified(&ctx, &source, 256).unwrap();
    let boundaries = p["boundaries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    let sites = p["captures"]["layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap() as u32)
        .collect::<Vec<_>>();
    let mut reports = Vec::new();
    let mut capture_reports = Vec::new();
    for ((name, base, tokens, _), reference) in cases.iter().zip(files) {
        let mut rows = OracleRows::new(
            BufReader::new(fs::File::open(reference.logits).unwrap()),
            *base,
            tokens,
            250624,
        );
        let mut session = model.create_session(*base).unwrap();
        let mut checkpoints = Vec::new();
        for (index, &token) in tokens.iter().enumerate() {
            let expected = rows.row();
            let actual = if index == 255 && reference.captures.is_some() {
                let captured = session.append_with_captures(&[token], &sites).unwrap();
                assert_eq!(captured.post_block_layers, sites);
                assert_eq!(captured.residuals.len(), sites.len() * 4096);
                for (&layer, a) in sites.iter().zip(captured.residuals.chunks_exact(4096)) {
                    let b = &reference.captures.as_ref().unwrap()[layer as usize];
                    let metrics = error_metrics(a, b);
                    let relative_l2 = (a
                        .iter()
                        .zip(b)
                        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                        .sum::<f64>()
                        / b.iter().map(|&b| f64::from(b).powi(2)).sum::<f64>())
                    .sqrt();
                    let mut failed = Vec::new();
                    if !relative_l2.is_finite()
                        || relative_l2 >= p["captures"]["relative_l2_exclusive"].as_f64().unwrap()
                    {
                        failed.push("relative_l2");
                    }
                    if !metrics["cosine"].as_f64().is_some_and(|v| {
                        v > p["captures"]["cosine_exclusive_min"].as_f64().unwrap()
                    }) {
                        failed.push("cosine");
                    }
                    capture_reports.push(
                        json!({"corpus":name,"base":base,"layer":layer,"visible_length":256,
                        "metrics":metrics,"relative_l2":relative_l2,"failed_gates":failed}),
                    );
                }
                captured.logits
            } else {
                session.append(&[token]).unwrap()
            };
            let metrics = row_metrics(&actual, &expected);
            let failed = row_failures(&p, &metrics);
            reports.push(
                json!({"corpus":name,"base":base,"lane":"teacher_forced","visible_length":index+1,
                "token":token,"metrics":metrics,"failed_gates":failed}),
            );
            if boundaries.contains(&(index + 1)) {
                checkpoints.push(actual);
            }
        }
        rows.finish();
        assert_eq!(session.committed_len(), 256);
        assert!(session.append(&[42]).is_err());
        assert!(!session.is_poisoned());
        drop(session);
        let mut split = model.create_session(*base).unwrap();
        let mut start = 0;
        for (&end, expected) in boundaries.iter().zip(&checkpoints) {
            assert!(
                bitwise_equal(&split.append(&tokens[start..end]).unwrap(), expected),
                "{name} split{end}"
            );
            start = end;
        }
        drop(split);
        let mut whole = model.create_session(*base).unwrap();
        assert!(
            bitwise_equal(&whole.append(tokens).unwrap(), checkpoints.last().unwrap()),
            "{name} whole"
        );
        drop(whole);
        if let Some((file, ids)) = reference.greedy {
            let mut rows = OracleRows::new(
                BufReader::new(fs::File::open(file).unwrap()),
                *base,
                &ids,
                250624,
            );
            let mut session = model.create_session(*base).unwrap();
            let mut previous = None;
            for (index, &token) in ids.iter().enumerate() {
                let expected = rows.row();
                if index >= 241 {
                    assert_eq!(
                        previous,
                        Some(token),
                        "reference is not the declared greedy trajectory"
                    );
                }
                previous = Some(argmax(&expected));
                let actual = session.append(&[token]).unwrap();
                let metrics = row_metrics(&actual, &expected);
                let failed = row_failures(&p, &metrics);
                reports.push(json!({"corpus":name,"base":base,"lane":"reference_argmax_trajectory","visible_length":index+1,
                    "token":token,"metrics":metrics,"failed_gates":failed}));
            }
            rows.finish();
        }
        let failures = save(&directory, &reports, &capture_reports);
        eprintln!(
            "{name} base={base}: split/whole controls passed; {} failed row/site records so far",
            failures.len()
        );
    }
    assert_eq!(reports.len(), 4096);
    assert_eq!(capture_reports.len(), 16);
    assert_eq!(source.revalidate_retained_shard_stamps().unwrap(), stamps);
    let failures = save(&directory, &reports, &capture_reports);
    fs::write(directory.join("summary.json"),serde_json::to_vec_pretty(&json!({
        "policy_sha256":POLICY_SHA256,"candidate_envelope_passed":failures.is_empty(),"public_cap_promoted":false,
        "rows":reports.len(),"capture_sites":capture_reports.len(),"failed_records":failures.len(),"bitwise_partition_controls":12,
    })).unwrap()).unwrap();
    assert!(
        failures.is_empty(),
        "{} frozen holdout row/site records failed; inspect {}",
        failures.len(),
        directory.display()
    );
}

#[test]
fn frozen_row_gates_reject_boundary_values_and_nonfinite_metrics() {
    let p = policy();
    let good = json!({"logits":{"actual_top1":7,"reference_top1":7,"max_abs":0.,"rmse":0.,"cosine":1.},
        "distribution":{"kl_reference_to_actual":0.,"total_variation":0.,"centered_rmse":0.}});
    assert!(row_failures(&p, &good).is_empty());
    for (section, key, bound) in [
        ("logits", "max_abs", 0.5),
        ("logits", "rmse", 0.05),
        ("logits", "cosine", 0.9995),
        ("distribution", "kl_reference_to_actual", 0.0001),
        ("distribution", "total_variation", 0.002),
        ("distribution", "centered_rmse", 0.03),
    ] {
        let mut bad = good.clone();
        bad[section][key] = json!(bound);
        assert!(row_failures(&p, &bad).contains(&key));
        bad[section][key] = serde_json::Value::Null;
        assert!(row_failures(&p, &bad).contains(&key));
    }
    let mut bad = good;
    bad["logits"]["actual_top1"] = json!(9);
    assert_eq!(row_failures(&p, &bad), vec!["top1"]);
    assert_eq!(argmax(&[1., 2., 2.]), 2);
    assert_eq!(argmax(&[0., -0.]), 1);
}

#[test]
fn trajectory_protocol_binds_prefix_positions_and_exact_extent() {
    let prefix = vec![42; 241];
    let mut bytes = b"K2TOK001".to_vec();
    for word in [37u32, 256].into_iter().chain(std::iter::repeat_n(42, 256)) {
        bytes.extend(word.to_le_bytes());
    }
    assert_eq!(generated_ids(&bytes, 37, &prefix), vec![42; 256]);
    for offset in [0, 8, 12, 16] {
        let mut bad = bytes.clone();
        bad[offset] ^= 1;
        assert!(std::panic::catch_unwind(|| generated_ids(&bad, 37, &prefix)).is_err());
    }
    assert!(
        std::panic::catch_unwind(|| generated_ids(&bytes[..bytes.len() - 1], 37, &prefix)).is_err()
    );
    bytes[16 + 255 * 4..].copy_from_slice(&250624u32.to_le_bytes());
    assert!(std::panic::catch_unwind(|| generated_ids(&bytes, 37, &prefix)).is_err());
}
