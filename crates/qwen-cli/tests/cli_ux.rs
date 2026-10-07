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

/// A header-only GGUF naming `architecture`: enough for family detection,
/// nothing a lane could load. Removed when dropped.
struct HeaderOnlyGguf(std::path::PathBuf);

impl HeaderOnlyGguf {
    fn new(architecture: &str) -> Self {
        Self::with_u64(architecture, &[])
    }

    /// Also writes `<architecture>.<key> = value` (GGUF uint64) entries.
    fn with_u64(architecture: &str, entries: &[(&str, u64)]) -> Self {
        let string = |bytes: &mut Vec<u8>, text: &str| {
            bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
            bytes.extend_from_slice(text.as_bytes());
        };
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(1 + entries.len() as u64).to_le_bytes());
        string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes());
        string(&mut bytes, architecture);
        for (key, value) in entries {
            string(&mut bytes, &format!("{architecture}.{key}"));
            bytes.extend_from_slice(&10u32.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        let path = std::env::temp_dir().join(format!(
            "qwen-cli-ux-{architecture}-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        Self(path)
    }
}

impl Drop for HeaderOnlyGguf {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Drafter admission is settled once, before family dispatch, in run and
/// serve: every family without speculation is refused with the shared
/// message, and the drafter path is never opened (it does not exist here).
#[test]
fn every_family_without_speculation_refuses_drafter_before_dispatch() {
    const DRAFTER: &str = "/qwen-cli-ux-missing-drafter.gguf";
    for (architecture, display) in [
        ("muse-glimmer", "Muse Glimmer"),
        ("k2-horizon", "K2 Horizon"),
        ("glm5-next", "GLM-5.3-Flash"),
        ("qwen4exp", "Qwen3.8-Flash-Next"),
        ("deepseek4", "DeepSeek V4"),
    ] {
        let model = HeaderOnlyGguf::new(architecture);
        let path = model.0.to_str().unwrap();
        for (lane, args) in [
            (
                "run",
                vec!["run", "-m", path, "--user", "hi", "--drafter", DRAFTER],
            ),
            (
                "serve",
                vec![
                    "serve",
                    "-m",
                    path,
                    "--drafter",
                    DRAFTER,
                    "--max-context-tokens",
                    "64",
                    "--max-tokens",
                    "8",
                    "--addr",
                    "127.0.0.1:0",
                ],
            ),
        ] {
            let output = run_with_stdin_held_open(&args);
            assert!(!output.status.success(), "{architecture} {lane}");
            let stderr = String::from_utf8(output.stderr).unwrap();
            assert!(
                stderr.contains(&format!("--drafter is not supported for {display} {lane}")),
                "{architecture} {lane}: {stderr}"
            );
            assert!(
                !stderr.contains("open drafter"),
                "{architecture} {lane}: {stderr}"
            );
        }
    }
}

/// Text `qwen info` reports each family as itself: its own name and its own
/// `<architecture>.*` metadata, never the Qwen hybrid block split or
/// `qwen35.*` keys (MoE reports `qwen35moe.*`). The header and metadata do
/// not depend on capability checks, so a bare header still reports them.
#[test]
fn text_info_describes_each_family_as_itself() {
    for (architecture, family) in [
        ("qwen35", "Qwen (qwen)"),
        ("qwen35moe", "Qwen MoE (qwen)"),
        ("qwen4exp", "Qwen3.8-Flash-Next (qwen4exp)"),
        ("muse-glimmer", "Muse Glimmer (muse_glimmer)"),
        ("k2-horizon", "K2 Horizon (k2_horizon)"),
        ("glm5-next", "GLM-5.3-Flash (glm5_next)"),
        ("glm5next", "GLM-5.3-Flash (glm5_next)"),
        ("deepseek4", "DeepSeek V4 (deepseek_v4)"),
        ("not-a-family", "unrecognised"),
    ] {
        let model = HeaderOnlyGguf::with_u64(architecture, &[("block_count", 4)]);
        let output = Command::new(QWEN)
            .args(["info", "-m", model.0.to_str().unwrap()])
            .output()
            .expect("run qwen info");
        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(output.status.success(), "{architecture}: {stderr}");
        assert!(
            stdout.contains(&format!("family: {family}")),
            "{architecture}: {stdout}"
        );
        assert!(
            stdout.contains(&format!("{architecture}.block_count = 4")),
            "{architecture}: {stdout}"
        );
        assert!(stdout.contains("capabilities"), "{architecture}: {stdout}");
        if !architecture.starts_with("qwen35") {
            assert!(!stdout.contains("GDN"), "{architecture}: {stdout}");
            assert!(!stdout.contains("qwen35."), "{architecture}: {stdout}");
        }
    }
}
