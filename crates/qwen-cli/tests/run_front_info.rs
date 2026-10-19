use serde_json::{Value, json};
use std::path::PathBuf;
use std::process::{Command, Output};

const BASELINES: &str = include_str!("fixtures/run-front-info-cli-v1.json");

fn invoke(model: &std::path::Path, json_output: bool) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_qwen"));
    command.args(["info", "--model", model.to_str().unwrap()]);
    if json_output {
        command.arg("--json");
    }
    command.output().expect("start qwen info")
}

fn fixture_path(architecture: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/run-front-models")
        .join(format!("{architecture}.gguf"))
}

fn normalize_model_json(mut value: Value) -> Value {
    value["model"] = json!("<model>");
    if let Some(storage) = value
        .get_mut("weight_storage")
        .and_then(Value::as_object_mut)
    {
        for key in [
            "mode",
            "copied_bytes_estimate",
            "recommended_working_set_bytes",
            "reason",
        ] {
            if storage.contains_key(key) {
                storage.insert(key.to_owned(), json!("<device-derived>"));
            }
        }
    }
    value
}

#[test]
fn every_family_qwen_info_json_and_text_matches_synthetic_cli_baseline() {
    let baselines: Value = serde_json::from_str(BASELINES).unwrap();
    for (architecture, expected) in baselines.as_object().unwrap() {
        let model = fixture_path(architecture);
        let json_output = invoke(&model, true);
        assert!(
            json_output.status.success(),
            "qwen info --json failed for {architecture}: {}",
            String::from_utf8_lossy(&json_output.stderr)
        );
        let actual_json: Value = serde_json::from_slice(&json_output.stdout).unwrap();
        assert_eq!(
            normalize_model_json(actual_json),
            expected["json"],
            "{architecture}"
        );

        let text_output = invoke(&model, false);
        assert!(
            text_output.status.success(),
            "qwen info failed for {architecture}: {}",
            String::from_utf8_lossy(&text_output.stderr)
        );
        let actual_text = String::from_utf8(text_output.stdout)
            .unwrap()
            .replace(model.to_str().unwrap(), "<model>");
        assert_eq!(
            actual_text,
            expected["text_utf8"].as_str().unwrap(),
            "{architecture}"
        );
    }
}
