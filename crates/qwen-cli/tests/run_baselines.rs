use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const BASELINE: &str = include_str!("fixtures/run-baselines/k2-real-v1.json");

fn run(command: &mut Command) -> Output {
    command.output().expect("start baseline subprocess")
}

fn token_ids(model: &Path, text: &str, add_special: bool) -> Vec<i32> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_qwen-tok"));
    command.args(["--model", model.to_str().unwrap(), "--text", text, "--ids"]);
    if add_special {
        command.arg("--add-special");
    }
    let output = run(&mut command);
    assert!(
        output.status.success(),
        "qwen-tok failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .skip(1)
        .map(|line| line.parse().expect("token ID"))
        .collect()
}

#[test]
#[ignore = "model witness: k2-horizon-7b-q8-0"]
fn k2_real_run_matches_pre_refactor_baseline() {
    let baseline: Value = serde_json::from_str(BASELINE).unwrap();
    let model = qwen_llm::test_fixtures::catalog::K2_HORIZON_7B_Q8_0.required();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let stats_path = std::env::temp_dir().join(format!(
        "qwen-k2-baseline-{}-{unique}.jsonl",
        std::process::id()
    ));

    let mut command = Command::new(env!("CARGO_BIN_EXE_qwen"));
    command.args([
        "run",
        "--model",
        model.to_str().unwrap(),
        "--raw-prompt",
        baseline["rendered_prompt_utf8"].as_str().unwrap(),
        "--max-tokens",
        "5",
        "--seed",
        "314159",
        "--temp",
        "0",
        "--request-stats-jsonl",
        stats_path.to_str().unwrap(),
    ]);
    let output = run(&mut command);
    assert!(
        output.status.success(),
        "qwen failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        output.stdout,
        baseline["stdout_utf8"].as_str().unwrap().as_bytes()
    );
    assert_eq!(
        token_ids(
            &model,
            baseline["rendered_prompt_utf8"].as_str().unwrap(),
            true
        ),
        serde_json::from_value::<Vec<i32>>(baseline["prompt_token_ids"].clone()).unwrap()
    );
    let generated_text = String::from_utf8(output.stdout.clone()).unwrap();
    let generated_text = if baseline["stdout_has_final_newline"] == true {
        generated_text.strip_suffix('\n').unwrap()
    } else {
        &generated_text
    };
    assert_eq!(
        token_ids(&model, generated_text, false),
        serde_json::from_value::<Vec<i32>>(baseline["generated_token_ids"].clone()).unwrap()
    );

    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("stop_reason=token_limit"));
    assert!(stderr.contains("prompt_tokens=8 generated_tokens=5 transitions=4"));
    assert!(stderr.contains("weights_first_use=true"));
    assert_eq!(
        stderr.lines().count(),
        baseline["stderr_shape"].as_array().unwrap().len()
    );

    let record_line = fs::read_to_string(&stats_path).unwrap();
    let _ = fs::remove_file(stats_path);
    let record: Value = serde_json::from_str(record_line.trim()).unwrap();
    let mut keys = record
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    keys.sort_unstable();
    assert_eq!(
        keys,
        baseline["stats_jsonl_keys"]
            .as_array()
            .unwrap()
            .iter()
            .map(Value::as_str)
            .map(Option::unwrap)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        record["model"]["family"],
        baseline["stats_jsonl_template_label"]
    );
    assert_eq!(record["finish"]["reason"], baseline["stop_reason"]);
    assert_eq!(record["usage"]["input_tokens"], 8);
    assert_eq!(record["usage"]["output_tokens"], 5);
    assert_eq!(
        record["output_fingerprint"]["value"],
        baseline["stats_jsonl_output_fingerprint"]
    );
    let mut timing_keys = record["timing_ms"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    timing_keys.sort_unstable();
    assert_eq!(
        timing_keys,
        baseline["stats_jsonl_timing_fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(Value::as_str)
            .map(Option::unwrap)
            .collect::<Vec<_>>()
    );
}
