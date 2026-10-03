use super::*;
use crate::linear_transport::{
    cpu_fixture::write_cpu_gguf,
    tests::{Fixture, fixture, modify},
};
use std::sync::Arc;

pub(crate) fn fitted_fixture() -> (Fixture, Arc<Registry>) {
    let f = fixture("synthetic-cpu-fit", 2, 12);
    let model = f.0.join("model.gguf");
    write_cpu_gguf(&model, "qwen35", 2, "cpu-test", false);
    let config = f.0.join("config.json");
    std::fs::write(
        &config,
        serde_json::to_vec(&json!({"schema_version":1,"assets":[
        {"alias":"fit","path":".","allow_unvalidated_transfer":true}]}))
        .unwrap(),
    )
    .unwrap();
    let gguf = GgufFile::open(&model).unwrap();
    let registry = Registry::open(&config, &gguf, &mut || Ok(())).unwrap();
    (f, Arc::new(registry))
}

#[test]
fn registry_binds_before_metal_and_rehashes_retained_selected_matrices() {
    use std::io::{Seek, SeekFrom, Write};
    let (f, registry) = fitted_fixture();
    let metadata = registry.asset("fit").unwrap();
    assert_eq!(metadata["target_layer"], 1);
    assert_eq!(metadata["direction_rows"], json!([]));
    assert_eq!(metadata["binding"]["binding_phase"], "cpu_before_metal");
    assert_eq!(
        metadata["binding"]["status"],
        "source_deployment_equivalence_unverified"
    );
    let keys = BTreeSet::from([MatrixKey {
        alias: "fit".into(),
        layer: 0,
    }]);
    assert_eq!(registry.matrix_bytes(&keys).unwrap(), 8);
    let first = registry.stage(&keys, 8, || Ok(())).unwrap();
    let payload = f.0.join("transport.f16le");
    let retained = f.0.join("retained.f16le");
    std::fs::rename(&payload, &retained).unwrap();
    std::fs::write(&payload, vec![0; 16]).unwrap();
    assert_eq!(
        registry.stage(&keys, 8, || Ok(())).unwrap().matrices,
        first.matrices
    );
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(retained)
        .unwrap();
    file.seek(SeekFrom::Start(8)).unwrap();
    file.write_all(&17u16.to_le_bytes()).unwrap();
    assert!(registry.stage(&keys, 8, || Ok(())).is_err());
    assert!(registry.stage(&keys, 7, || Ok(())).is_err());
}

#[test]
fn geometry_mismatch_precedes_payload_access_without_restricting_ordinary_target() {
    let (f, _) = fitted_fixture();
    let model = GgufFile::open(&f.0.join("model.gguf")).unwrap();
    let config = f.0.join("config.json");
    modify(&f, |m| m["model"]["hidden_size"] = json!(3));
    std::fs::remove_file(f.0.join("transport.f16le")).unwrap();
    let error = Registry::open(&config, &model, &mut || Ok(()))
        .err()
        .unwrap();
    assert!(format!("{error:#}").contains("deployment geometry mismatch"));
}

#[test]
fn config_refuses_duplicates_reserved_aliases_and_implicit_transfer() {
    let (f, _) = fitted_fixture();
    let model = GgufFile::open(&f.0.join("model.gguf")).unwrap();
    let config = f.0.join("config.json");
    for text in [
        r#"{"schema_version":1,"schema_version":1,"assets":[]}"#,
        r#"{"schema_version":1,"assets":[{"alias":"plain","path":"."}]}"#,
        r#"{"schema_version":1,"assets":[{"alias":"fit","path":"."}]}"#,
        r#"{"schema_version":1,"assets":[{"alias":"fit","path":"."},{"alias":"fit","path":"."}]}"#,
    ] {
        std::fs::write(&config, text).unwrap();
        assert!(Registry::open(&config, &model, &mut || Ok(())).is_err());
    }
}

#[test]
fn exact_binding_cannot_be_overridden_and_cancellation_never_publishes_registry() {
    let (f, _) = fitted_fixture();
    let model = GgufFile::open(&f.0.join("model.gguf")).unwrap();
    let content =
        qwen_llm::checkpoint_identity::verified_checkpoint_content_identity(&model).unwrap();
    let content = content
        .content_id
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let tokenizer = format!(
        "{:016x}",
        qwen_llm::runtime::tokenizer_metadata_identity(&model)
    );
    modify(
        &f,
        |m| {
            m["model"]["exact_binding"] =
                json!({"gguf_content_blake3":content,"tokenizer_metadata_id":tokenizer})
        },
    );
    let config = f.0.join("config.json");
    let mut checks = 0;
    let registry = Registry::open(&config, &model, &mut || {
        checks += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(
        registry.asset("fit").unwrap()["transfer"],
        "exact_deployment_binding_matched"
    );
    assert_eq!(
        registry.asset("fit").unwrap()["binding"]["content_bytes_hashed"],
        std::fs::metadata(f.0.join("model.gguf")).unwrap().len()
    );
    for stop in [1, checks / 2, checks] {
        let mut count = 0;
        let result = Registry::open(&config, &model, &mut || {
            count += 1;
            ensure!(count != stop, "cancel registry");
            Ok(())
        });
        assert!(result.is_err());
    }
    for key in ["gguf_content_blake3", "tokenizer_metadata_id"] {
        modify(&f, |m| {
            m["model"]["exact_binding"][key] =
                json!("0".repeat(if key == "gguf_content_blake3" { 64 } else { 16 }))
        });
        let error = Registry::open(&config, &model, &mut || Ok(()))
            .err()
            .unwrap();
        assert!(format!("{error:#}").contains("deployment exact binding mismatch"));
        modify(&f, |m| {
            m["model"]["exact_binding"][key] = json!(if key == "gguf_content_blake3" {
                &content
            } else {
                &tokenizer
            })
        });
    }
}
