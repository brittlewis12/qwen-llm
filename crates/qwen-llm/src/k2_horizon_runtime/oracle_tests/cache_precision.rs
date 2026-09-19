use super::*;

fn comparison(actual: &[f32], reference: &[f32]) -> serde_json::Value {
    json!({"logits":error_metrics(actual,reference),"distribution":probability::metrics(actual,reference)})
}

#[test]
fn oracle_cache_precision_is_bound_not_inferred() {
    for cache in [ReferenceCache::F16, ReferenceCache::F32] {
        let mut bytes = b"K2REF001".to_vec();
        for word in [
            2,
            1,
            37,
            cache.bits(),
            37,
            42,
            1_f32.to_bits(),
            (-2_f32).to_bits(),
        ] {
            bytes.extend(word.to_le_bytes());
        }
        let mut rows = OracleRows::with_cache(Cursor::new(&bytes), 37, &[42], 2, cache);
        assert_eq!(rows.row(), vec![1., -2.]);
        rows.finish();
        let wrong = match cache {
            ReferenceCache::F16 => ReferenceCache::F32,
            ReferenceCache::F32 => ReferenceCache::F16,
        };
        assert!(
            std::panic::catch_unwind(|| OracleRows::with_cache(
                Cursor::new(&bytes),
                37,
                &[42],
                2,
                wrong
            ))
            .is_err()
        );
        if cache == ReferenceCache::F32 {
            assert!(std::panic::catch_unwind(|| decode_rows(&bytes, 37, &[42], 2)).is_err());
        }
    }
}

#[test]
#[ignore = "GPU cache/backend-sensitivity diagnostic, not qualification; production lease; about 1.5 GiB evidence"]
fn gpu_f16_native_and_ifm_against_f32_cache_control() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    assert!(crate::metal::mat_vec_q8_0_lcpp_enabled());
    let path = PathBuf::from(std::env::var("K2_GGUF").expect("K2_GGUF"));
    let binary = PathBuf::from(std::env::var("K2_LLAMA_ORACLE").expect("K2_LLAMA_ORACLE"));
    let identity = Command::new(&binary).arg("--identity").output().unwrap();
    assert!(identity.status.success());
    assert_eq!(
        String::from_utf8(identity.stdout).unwrap(),
        reference_identity()
    );
    let source = GgufFile::open(&path).unwrap();
    let stamps = source.revalidate_retained_shard_stamps().unwrap();
    let model_sha256 = file_sha256(&path);
    assert_eq!(
        model_sha256,
        "5a98a289aba5c8c99ef05c9261287f19e8f586fd679bab45b632eb86b47809bf"
    );
    let cases = extended_inputs(&NativeTokenizer::from_gguf(&source).unwrap());
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/profiles")
        .join(format!("k2-cache-precision-{}-{stamp}", std::process::id()));
    let f16_dir = directory.join("ifm-f16");
    let f32_dir = directory.join("ifm-f32");
    fs::create_dir_all(&f16_dir).unwrap();
    fs::create_dir(&f32_dir).unwrap();
    fs::write(directory.join("manifest.json"),serde_json::to_vec_pretty(&json!({
        "qualification":false,"purpose":"cache/backend precision sensitivity, not an exact ground-truth oracle",
        "reference":reference_identity().trim(),"reference_binary_sha256":file_sha256(&binary),
        "model":path,"model_sha256":model_sha256,"inputs":cases,
        "native_capacity":256,"native_cache":"F16","native_attention":"materialized","native_q8_matvec":"lcpp",
        "reference_modes":[{"cache":"F16","flag":null},{"cache":"F32","flag":"--f32-kv"}],
        "requested_reference_capacity":256,"reference_allocated_capacity":256,"reference_flash_attention":false,
        "native_metallib_sha256":native_kernel_identity(),"native_sources":native_source_identity(),
        "probability_metrics":"F64 stable log-softmax; KL(reference||actual); TV=0.5*sum(abs(P-Q)); RMSE after centering logit differences",
        "pair_order":"native_f16_vs_ifm_f16; native_f16_vs_ifm_f32; ifm_f16_vs_ifm_f32 (actual_vs_reference)",
        "caveat":"cache dtype may change backend kernels/reductions; F32 cache is not full-precision weights or HF ground truth",
        "metal_api_validation":std::env::var("MTL_DEBUG_LAYER").ok(),
    })).unwrap()).unwrap();
    eprintln!(
        "cache precision diagnostic artifacts: {}",
        directory.display()
    );
    let outputs = cases
        .iter()
        .map(|(_, base, tokens)| {
            let f16 = run_oracle_cache(
                &binary,
                &path,
                &f16_dir,
                *base,
                tokens,
                false,
                ReferenceCache::F16,
            );
            let f32 = run_oracle_cache(
                &binary,
                &path,
                &f32_dir,
                *base,
                tokens,
                false,
                ReferenceCache::F32,
            );
            (f16, f32)
        })
        .collect::<Vec<_>>();
    assert_eq!(source.revalidate_retained_shard_stamps().unwrap(), stamps);
    let ctx = MetalContext::new().unwrap();
    let model = K2LoadedModel::load_unqualified(&ctx, &source, 256).unwrap();
    let mut reports = Vec::new();
    for ((name, base, tokens), (f16, f32)) in cases.iter().zip(outputs) {
        let mut f16 = OracleRows::with_cache(
            BufReader::new(fs::File::open(f16).unwrap()),
            *base,
            tokens,
            250624,
            ReferenceCache::F16,
        );
        let mut f32 = OracleRows::with_cache(
            BufReader::new(fs::File::open(f32).unwrap()),
            *base,
            tokens,
            250624,
            ReferenceCache::F32,
        );
        let mut session = model.create_session(*base).unwrap();
        for (index, &token) in tokens.iter().enumerate() {
            let half = f16.row();
            let full = f32.row();
            let actual = session.append(&[token]).unwrap();
            reports.push(
                json!({"corpus":name,"base":base,"visible_length":index+1,"token":token,
                "native_f16_vs_ifm_f16":comparison(&actual,&half),
                "native_f16_vs_ifm_f32":comparison(&actual,&full),
                "ifm_f16_vs_ifm_f32":comparison(&half,&full)}),
            );
        }
        f16.finish();
        f32.finish();
        eprintln!("{name} base={base}: 256 cache-control rows compared (diagnostic only)");
        fs::write(
            directory.join("metrics.json"),
            serde_json::to_vec_pretty(&json!({"qualification":false,"cases":reports})).unwrap(),
        )
        .unwrap();
    }
    assert_eq!(reports.len(), 768);
    assert_eq!(source.revalidate_retained_shard_stamps().unwrap(), stamps);
}
