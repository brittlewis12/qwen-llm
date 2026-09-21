use super::*;

fn cache_digest(session: &K2Session<'_, '_>) -> String {
    let tensor = &session.buffers.cache;
    assert_eq!(tensor.offset, 0);
    let bytes = unsafe {
        std::slice::from_raw_parts(
            tensor.buffer.contents().as_ptr().cast::<u8>(),
            tensor.n_bytes() as usize,
        )
    };
    format!("{:x}", Sha256::digest(bytes))
}

fn poison_empty_cache(session: &K2Session<'_, '_>) {
    assert_eq!(session.committed_len(), 0);
    let tensor = &session.buffers.cache;
    let values = unsafe {
        std::slice::from_raw_parts_mut(
            tensor.buffer.contents().as_ptr().cast::<half::f16>(),
            tensor.n_elements() as usize,
        )
    };
    values.fill(half::f16::NAN);
}

fn equal_capture(a: &K2CapturedForward, b: &K2CapturedForward) -> bool {
    a.absolute_position == b.absolute_position
        && a.post_block_layers == b.post_block_layers
        && runner::bitwise_equal(&a.logits, &b.logits)
        && runner::bitwise_equal(&a.residuals, &b.residuals)
}

fn save(directory: &Path, reports: &[serde_json::Value]) {
    fs::write(
        directory.join("checks.json"),
        serde_json::to_vec_pretty(reports).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "GPU experimental packed-Q8 exact regression; production lease; pinned K2_GGUF; no public promotion"]
fn gpu_packed_q8_matches_serial_captures_cache_partitions_and_interventions() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    assert!(crate::metal::mat_vec_q8_0_lcpp_enabled());
    let source = GgufFile::open(std::env::var("K2_GGUF").expect("K2_GGUF")).unwrap();
    let p = v2::policy();
    let cases = inputs_for(&source, &p)
        .into_iter()
        .filter(|c| c.3)
        .collect::<Vec<_>>();
    assert_eq!(cases.len(), 4);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/profiles")
        .join(format!("k2-packed-q8-{}-{stamp}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    let boundaries = [
        1usize, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256,
    ];
    let sites = [0, 11, 23, 35];
    fs::write(directory.join("manifest.json"), serde_json::to_vec_pretty(&json!({
        "claim":"native_bitwise_packed_regression_not_new_holdout_or_speed_evidence",
        "model_sha256":p["model_sha256"],"tokenizer_metadata_id":p["tokenizer_metadata_id"],
        "fixture_policy_sha256":v2::SHA256,"inputs":cases,"boundaries":boundaries,
        "capture_sites":sites,"chunk_max":32,"capacity":256,"cache":"F16",
        "native_metallib_sha256":native_kernel_identity(),"native_sources":native_source_identity(),
        "native_test_binary_sha256":file_sha256(&std::env::current_exe().unwrap()),
        "metal_api_validation":"1","public_prefill_changed":false,
        "gates":"bitwise logits, captures, full cache including poisoned future; no tolerance",
    })).unwrap()).unwrap();
    eprintln!("packed Q8 artifacts: {}", directory.display());
    let ctx = MetalContext::new().unwrap();
    let mut model = K2LoadedModel::load_unqualified(&ctx, &source, 256).unwrap();
    let mut reports = Vec::new();
    for (name, base, tokens, _) in &cases {
        model.prefill = PrefillMode::Serial;
        let mut session = model.create_session(*base).unwrap();
        poison_empty_cache(&session);
        let mut controls = Vec::new();
        for (index, &id) in tokens.iter().enumerate() {
            if boundaries.contains(&(index + 1)) {
                let capture = session.append_with_captures(&[id], &sites).unwrap();
                controls.push((index + 1, capture, cache_digest(&session)));
            } else {
                session.append(&[id]).unwrap();
            }
        }
        drop(session);
        model.prefill = PrefillMode::BatchQ8;
        for ends in [vec![256], boundaries.to_vec(), vec![2, 33, 65, 128, 256]] {
            let mut session = model.create_session(*base).unwrap();
            poison_empty_cache(&session);
            let initial = cache_digest(&session);
            let mut invalid = tokens[..65].to_vec();
            invalid[64] = 250624;
            assert!(session.append(&invalid).is_err());
            assert!(session.append(&[0; 257]).is_err());
            assert_eq!(session.committed_len(), 0);
            assert!(!session.is_poisoned());
            assert_eq!(initial, cache_digest(&session));
            let mut start = 0;
            for &end in &ends {
                let actual = session
                    .append_with_captures(&tokens[start..end], &sites)
                    .unwrap();
                let (_, expected, expected_cache) =
                    controls.iter().find(|(n, _, _)| *n == end).unwrap();
                let cache = cache_digest(&session);
                let passed = equal_capture(&actual, expected) && cache == *expected_cache;
                reports.push(
                    json!({"corpus":name,"base":base,"partition":ends,"visible_length":end,
                    "passed":passed,"cache_sha256":cache,"control_cache_sha256":expected_cache,
                    "logits":error_metrics(&actual.logits,&expected.logits),
                    "captures":error_metrics(&actual.residuals,&expected.residuals)}),
                );
                save(&directory, &reports);
                assert!(
                    passed,
                    "{name} base={base} end={end}; inspect {}",
                    directory.display()
                );
                assert_eq!(session.committed_len(), end as u32);
                let readout = session.readout(&actual.residuals[3 * 4096..]).unwrap();
                assert!(runner::bitwise_equal(&readout, &actual.logits));
                assert_eq!(cache_digest(&session), cache);
                assert_eq!(session.committed_len(), end as u32);
                start = end;
            }
            assert!(session.append(&[42]).is_err());
            assert_eq!(session.committed_len(), 256);
            assert!(!session.is_poisoned());
        }
        eprintln!("{name} base={base}: packed partitions/captures/cache/readout exact");
    }
    let (_, base, tokens, _) = &cases[0];
    let direction = (0..4096)
        .map(|i| (i % 19) as f32 * 0.0001 - 0.0009)
        .collect::<Vec<_>>();
    let operations = [
        K2Intervention {
            post_block_layer: 0,
            coefficient: 0.125,
            kind: K2InterventionKind::Fixed {
                direction: &direction,
            },
        },
        K2Intervention {
            post_block_layer: 0,
            coefficient: 0.5,
            kind: K2InterventionKind::Projection {
                direction: &direction,
            },
        },
        K2Intervention {
            post_block_layer: 35,
            coefficient: -0.25,
            kind: K2InterventionKind::Fixed {
                direction: &direction,
            },
        },
    ];
    model.prefill = PrefillMode::Serial;
    let mut serial = model.create_session(*base).unwrap();
    poison_empty_cache(&serial);
    let control = serial
        .append_with_interventions(&tokens[..65], &sites, &operations)
        .unwrap();
    let control_cache = cache_digest(&serial);
    let continuation = serial.append(&tokens[65..68]).unwrap();
    let continuation_cache = cache_digest(&serial);
    drop(serial);
    model.prefill = PrefillMode::BatchQ8;
    let mut packed = model.create_session(*base).unwrap();
    poison_empty_cache(&packed);
    let actual = packed
        .append_with_interventions(&tokens[..65], &sites, &operations)
        .unwrap();
    assert!(equal_capture(&actual, &control));
    assert_eq!(cache_digest(&packed), control_cache);
    assert!(runner::bitwise_equal(
        &packed.append(&tokens[65..68]).unwrap(),
        &continuation
    ));
    assert_eq!(cache_digest(&packed), continuation_cache);
    drop(packed);
    let mut failure = model.create_session(*base).unwrap();
    let huge = vec![f32::MAX; 4096];
    let error = failure
        .append_with_interventions(
            &tokens[..65],
            &sites,
            &[K2Intervention {
                post_block_layer: 35,
                coefficient: f32::MAX,
                kind: K2InterventionKind::Fixed { direction: &huge },
            }],
        )
        .unwrap_err();
    assert!(error.to_string().contains("nonfinite"));
    assert_eq!(failure.committed_len(), 0);
    assert!(failure.is_poisoned());
    assert!(matches!(
        failure.append(&[0]),
        Err(K2RuntimeError::Poisoned)
    ));
    drop(failure);
    let mut fresh = model.create_session(*base).unwrap();
    let result = fresh.append_with_captures(&tokens[..2], &sites).unwrap();
    assert!(result.logits.iter().all(|v| v.is_finite()));
    assert_eq!(fresh.committed_len(), 2);
    assert_eq!(reports.len(), 4 * (1 + boundaries.len() + 5));
    fs::write(
        directory.join("summary.json"),
        serde_json::to_vec_pretty(&json!({
            "passed":true,"bitwise_capture_logit_cache_checkpoints":reports.len(),"corpora":4,
            "partitions_per_corpus":3,"readout_preserves_prefix_and_cache":true,
            "ordered_last_token_interventions_and_continuation":true,
            "late_submitted_nonfinite_poisons_without_prefix_commit":true,
            "pre_gpu_invalid_late_id_and_capacity_refusal":true,"public_prefill_changed":false,
            "performance_claim":false,
        }))
        .unwrap(),
    )
    .unwrap();
}
