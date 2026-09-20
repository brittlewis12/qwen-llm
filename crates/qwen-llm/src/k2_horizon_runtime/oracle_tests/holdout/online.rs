//! Native integration regression on previously seen fixtures, not another holdout.
use super::*;
use std::io::{BufWriter, Write};

struct Control {
    name: String,
    base: u32,
    greedy: bool,
    tokens: Vec<u32>,
    file: PathBuf,
    sha256: String,
    captures: Option<Vec<f32>>,
}

pub(super) fn write_header(out: &mut impl Write, base: u32, count: u32, vocab: u32) {
    // Reuse the checked row protocol; manifest provenance explicitly says native.
    out.write_all(b"K2REF001").unwrap();
    for word in [vocab, count, base, 16] {
        out.write_all(&word.to_le_bytes()).unwrap();
    }
}

pub(super) fn write_row(out: &mut impl Write, position: u32, token: u32, values: &[f32]) {
    assert!(values.iter().all(|v| v.is_finite()));
    out.write_all(&position.to_le_bytes()).unwrap();
    out.write_all(&token.to_le_bytes()).unwrap();
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    out.write_all(&bytes).unwrap();
}

fn save_json(directory: &Path, name: &str, value: &serde_json::Value) {
    fs::write(
        directory.join(name),
        serde_json::to_vec_pretty(value).unwrap(),
    )
    .unwrap();
}

fn controls(
    model: &K2LoadedModel<'_>,
    cases: &[(String, u32, Vec<u32>, bool)],
    sites: &[u32],
    directory: &Path,
) -> Vec<Control> {
    assert_eq!(model.attention, AttentionBackend::Materialized);
    let mut controls = Vec::new();
    let mut records = Vec::new();
    for (name, base, input, _) in cases {
        for greedy in [false, true] {
            let file = directory.join(format!("{name}-{base}-{greedy}.native-f32"));
            let mut writer = BufWriter::new(fs::File::create_new(&file).unwrap());
            write_header(&mut writer, *base, 256, 250624);
            let mut session = model.create_session(*base).unwrap();
            let mut tokens = Vec::new();
            let mut next = None;
            let mut captures = None;
            for index in 0..256 {
                let token = if greedy && index >= 241 {
                    next.unwrap()
                } else {
                    input[index]
                };
                tokens.push(token);
                let values = if !greedy && index == 255 {
                    let result = session.append_with_captures(&[token], sites).unwrap();
                    assert_eq!(result.absolute_position, base + 255);
                    assert_eq!(result.post_block_layers, sites);
                    captures = Some(result.residuals);
                    result.logits
                } else {
                    session.append(&[token]).unwrap()
                };
                assert_eq!(values.len(), 250624);
                next = Some(runner::argmax(&values));
                write_row(&mut writer, base + index as u32, token, &values);
            }
            writer.flush().unwrap();
            drop(writer);
            assert_eq!(session.committed_len(), 256);
            assert!(session.append(&[42]).is_err());
            assert_eq!(session.committed_len(), 256);
            assert!(!session.is_poisoned());
            let sha256 = file_sha256(&file);
            let capture_record = captures.as_ref().map(|values| {
                assert_eq!(values.len(), sites.len() * 4096);
                assert!(values.iter().all(|v| v.is_finite()));
                let path = file.with_extension("captures.f32le");
                fs::write(
                    &path,
                    values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                json!({"path":path,"sha256":file_sha256(&path),"layers":sites,
                    "position":base+255,"token":tokens[255],"hidden":4096})
            });
            records.push(
                json!({"corpus":name,"base":base,"greedy":greedy,"tokens":tokens,
                "file":file,"sha256":sha256,"captures":capture_record,
                "backend":"native_materialized","capacity":256,"vocab":250624,"kv":"F16"}),
            );
            controls.push(Control {
                name: name.clone(),
                base: *base,
                greedy,
                tokens,
                file,
                sha256,
                captures,
            });
            save_json(directory, "controls.json", &json!(records));
        }
    }
    controls
}

pub(super) fn capture_metrics(
    p: &serde_json::Value,
    actual: &[f32],
    expected: &[f32],
) -> serde_json::Value {
    let metrics = error_metrics(actual, expected);
    let relative_l2 = (actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        / expected.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>())
    .sqrt();
    let mut failed = Vec::new();
    if !relative_l2.is_finite()
        || relative_l2 >= p["captures"]["relative_l2_exclusive"].as_f64().unwrap()
    {
        failed.push("relative_l2");
    }
    if !metrics["cosine"].as_f64().is_some_and(|v| {
        v.is_finite() && v > p["captures"]["cosine_exclusive_min"].as_f64().unwrap()
    }) {
        failed.push("cosine");
    }
    json!({"metrics":metrics,"relative_l2":relative_l2,"failed_gates":failed})
}

#[test]
#[ignore = "GPU native-only experimental attention regression; production lease; about 2 GiB evidence; no default promotion"]
fn gpu_online_runtime_matches_materialized_256_controls() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    assert!(crate::metal::mat_vec_q8_0_lcpp_enabled());
    let path = PathBuf::from(std::env::var("K2_GGUF").expect("K2_GGUF"));
    let source = GgufFile::open(&path).unwrap();
    let p = v2::policy();
    let cases = inputs_for(&source, &p)
        .into_iter()
        .filter(|case| case.3)
        .collect::<Vec<_>>();
    assert_eq!(cases.len(), 4);
    let stamps = source.revalidate_retained_shard_stamps().unwrap();
    let boundaries = p["boundaries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    let sites = p["captures"]["layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect::<Vec<_>>();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/profiles")
        .join(format!(
            "k2-online-integration-{}-{stamp}",
            std::process::id()
        ));
    fs::create_dir(&directory).unwrap();
    save_json(
        &directory,
        "manifest.json",
        &json!({
            "claim":"native_backend_integration_regression_not_independent_holdout",
            "policy":p,"policy_sha256":v2::SHA256,"fixture_freeze_commit":"300c8dcb",
            "model":path,"inputs":cases,"capacity":256,"cache":"F16","vocab":250624,
            "control":"native_materialized","candidate":"native_online_experimental",
            "trajectory":"materialized-derived, EOS-inclusive, exact predictors 241..256",
            "native_test_binary_sha256":file_sha256(&std::env::current_exe().unwrap()),
            "native_metallib_sha256":native_kernel_identity(),"native_sources":native_source_identity(),
            "host_build_note":std::env::var("K2_HOST_BUILD_NOTE").ok(),
            "expected_rows":2048,"expected_capture_sites":16,"expected_exact_predictors":64,
            "metal_api_validation":"1","public_default_changed":false,"performance_claim":false,
        }),
    );
    eprintln!(
        "native online integration artifacts: {}",
        directory.display()
    );
    let ctx = MetalContext::new().unwrap();
    let control = {
        let model = K2LoadedModel::load_with_attention_unqualified(
            &ctx,
            &source,
            256,
            AttentionBackend::Materialized,
        )
        .unwrap();
        controls(&model, &cases, &sites, &directory)
    };
    assert_eq!(source.revalidate_retained_shard_stamps().unwrap(), stamps);
    // The materialized model and all its sessions are gone before candidate load.
    let model = K2LoadedModel::load_with_attention_unqualified(
        &ctx,
        &source,
        256,
        AttentionBackend::Online,
    )
    .unwrap();
    assert_eq!(model.attention, AttentionBackend::Online);
    let mut reports = Vec::new();
    let mut captures = Vec::new();
    let mut partitions = 0;
    for control in control {
        assert_eq!(file_sha256(&control.file), control.sha256);
        let mut reader = OracleRows::new(
            BufReader::new(fs::File::open(&control.file).unwrap()),
            control.base,
            &control.tokens,
            250624,
        );
        let mut session = model.create_session(control.base).unwrap();
        let mut checkpoints = Vec::new();
        let mut previous = None;
        for (index, &token) in control.tokens.iter().enumerate() {
            let expected = reader.row();
            if control.greedy && index >= 241 {
                assert_eq!(previous, Some(token), "control trajectory token drift");
            }
            previous = Some(runner::argmax(&expected));
            let actual = if index == 255 && !control.greedy {
                let result = session.append_with_captures(&[token], &sites).unwrap();
                assert_eq!(result.absolute_position, control.base + 255);
                assert_eq!(result.post_block_layers, sites);
                assert_eq!(result.residuals.len(), sites.len() * 4096);
                for ((&layer, a), b) in sites
                    .iter()
                    .zip(result.residuals.chunks_exact(4096))
                    .zip(control.captures.as_ref().unwrap().chunks_exact(4096))
                {
                    let mut metrics = capture_metrics(&p, a, b);
                    metrics["corpus"] = json!(control.name);
                    metrics["base"] = json!(control.base);
                    metrics["layer"] = json!(layer);
                    captures.push(metrics);
                }
                result.logits
            } else {
                session.append(&[token]).unwrap()
            };
            let metrics = runner::row_metrics(&actual, &expected);
            let exact = control.greedy && runner::exact_trajectory_predictor(&p, index + 1);
            let failed = v2::row_failures(&p, &metrics, exact);
            let mismatch = metrics["logits"]["actual_top1"] != metrics["logits"]["reference_top1"];
            reports.push(
                json!({"corpus":control.name,"base":control.base,"greedy":control.greedy,
                "visible_length":index+1,"token":token,"metrics":metrics,"failed_gates":failed,
                "trajectory_exact_required":exact,"exact_top1_mismatch":mismatch,
                "accepted_ranking_indeterminate":mismatch && failed.is_empty()}),
            );
            if !control.greedy && boundaries.contains(&(index + 1)) {
                checkpoints.push(actual);
            }
        }
        reader.finish();
        assert_eq!(session.committed_len(), 256);
        assert!(session.append(&[42]).is_err());
        assert_eq!(session.committed_len(), 256);
        assert!(!session.is_poisoned());
        drop(session);
        save_json(
            &directory,
            "metrics.json",
            &json!({"rows":reports,"captures":captures}),
        );
        if !control.greedy {
            let mut split = model.create_session(control.base).unwrap();
            let mut start = 0;
            for (&end, expected) in boundaries.iter().zip(&checkpoints) {
                assert!(runner::bitwise_equal(
                    &split.append(&control.tokens[start..end]).unwrap(),
                    expected
                ));
                start = end;
            }
            drop(split);
            let mut whole = model.create_session(control.base).unwrap();
            assert!(runner::bitwise_equal(
                &whole.append(&control.tokens).unwrap(),
                checkpoints.last().unwrap()
            ));
            partitions += 1;
        }
        eprintln!(
            "{} base={} greedy={}: regression recorded",
            control.name, control.base, control.greedy
        );
    }
    assert_eq!(source.revalidate_retained_shard_stamps().unwrap(), stamps);
    assert_eq!(reports.len(), 2048);
    assert_eq!(captures.len(), 16);
    assert_eq!(partitions, 4);
    let failures = reports
        .iter()
        .chain(&captures)
        .filter(|r| !r["failed_gates"].as_array().unwrap().is_empty())
        .collect::<Vec<_>>();
    save_json(&directory, "failures.json", &json!(failures));
    let exact = reports
        .iter()
        .filter(|r| r["trajectory_exact_required"] == true)
        .count();
    assert_eq!(exact, 64);
    save_json(
        &directory,
        "summary.json",
        &json!({
            "passed":failures.is_empty(),"rows":reports.len(),"capture_sites":captures.len(),
            "failed_records":failures.len(),"bitwise_partition_controls":partitions,
            "exact_trajectory_predictors":exact,"nonoverlapping_model_lifetimes":true,
            "exact_top1_mismatch_records":reports.iter().filter(|r| r["exact_top1_mismatch"] == true).count(),
            "accepted_ranking_indeterminate_records":reports.iter().filter(|r| r["accepted_ranking_indeterminate"] == true).count(),
            "public_default_changed":false,"independent_oracle_qualification":false,"performance_claim":false,
        }),
    );
    assert!(
        failures.is_empty(),
        "{} failed records; inspect {}",
        failures.len(),
        directory.display()
    );
}

#[test]
fn native_control_writer_roundtrips_coordinates_and_finite_rows() {
    let mut bytes = Vec::new();
    write_header(&mut bytes, 37, 2, 3);
    write_row(&mut bytes, 37, 42, &[0., -0., 2.]);
    write_row(&mut bytes, 38, 17, &[-2., 4., 1.]);
    let decoded = decode_rows(&bytes, 37, &[42, 17], 3);
    assert!(runner::bitwise_equal(&decoded[0], &[0., -0., 2.]));
    assert_eq!(decoded[1], [-2., 4., 1.]);
    assert!(std::panic::catch_unwind(|| write_row(&mut Vec::new(), 0, 0, &[f32::NAN])).is_err());
}
