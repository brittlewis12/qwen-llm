use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const QWEN: &str = env!("CARGO_BIN_EXE_qwen");
const MISSING_MODEL: &str = "/qwen-cli-ux-missing-model.gguf";

fn run_with_stdin_held_open(args: &[&str]) -> Output {
    let mut child = Command::new(QWEN)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn qwen");
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match child.try_wait().expect("poll qwen") {
            Some(_) => break,
            None if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
            None => {
                let pid = child.id();
                child.kill().expect("kill exact timed-out qwen child");
                let _ = child.wait();
                panic!("qwen child {pid} blocked on stdin for args {args:?}");
            }
        }
    }
    drop(child.stdin.take());
    child.wait_with_output().expect("collect qwen output")
}

#[test]
fn bare_qwen_points_to_the_modern_front_door() {
    let output = Command::new(QWEN).output().expect("run bare qwen");
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("qwen run -m MODEL --user 'Explain this'"));
    assert!(!stderr.contains("-p <prompt>"));
}

#[test]
fn cheap_validation_precedes_modern_stdin_acquisition() {
    for input in ["--user", "--messages"] {
        let output = run_with_stdin_held_open(&[
            "run",
            "-m",
            MISSING_MODEL,
            input,
            "-",
            "--max-tokens",
            "0",
        ]);
        assert!(!output.status.success());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("--tokens must be >= 1"), "{stderr}");
        assert!(!stderr.contains("read empty stdin"), "{stderr}");
    }
}

#[test]
fn model_detection_precedes_modern_stdin_acquisition() {
    let output = run_with_stdin_held_open(&["run", "-m", MISSING_MODEL, "--user", "-"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("open model"), "{stderr}");
    assert!(!stderr.contains("read empty stdin"), "{stderr}");
}

#[test]
fn legacy_messages_dash_does_not_gain_modern_stdin_behavior() {
    let output = run_with_stdin_held_open(&["-m", MISSING_MODEL, "--messages", "-"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("open model"), "{stderr}");
    assert!(!stderr.contains("read --messages -"), "{stderr}");
}
