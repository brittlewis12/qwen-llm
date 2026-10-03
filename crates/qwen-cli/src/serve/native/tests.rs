use super::*;
use crate::serve::jobs::state::{GenerationState, JobState, StopReason};
use crate::serve::lens_http::input::Request;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) struct Fixture {
    pub(crate) root: PathBuf,
    pub(crate) store: Arc<JobStore>,
    pub(crate) profile: Arc<Profile>,
}
impl Fixture {
    pub(crate) fn new() -> Self {
        Self::with_extra_tokens(&[])
    }
    pub(crate) fn with_extra_tokens(extra_tokens: &[String]) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "qwen-native-baseline-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let model = root.join("tokenizer.gguf");
        write_byte_tokenizer(&model, extra_tokens);
        let gguf = qwen_llm::gguf::GgufFile::open(&model).unwrap();
        let tokenizer = Arc::new(Tokenizer::from_gguf(&gguf).unwrap());
        let store = Arc::new(
            JobStore::open(
                &root.join("jobs"),
                super::super::jobs::store::Limits::default(),
            )
            .unwrap(),
        );
        let profile = Arc::new(Profile {
            model_id: "test".into(),
            identity: "cpu-fixture".into(),
            protocol: QwenPromptTemplate::Qwen36,
            tokenizer,
            layers: 2,
            hidden: 2,
            context: 4096,
            max_tokens: 16,
            no_thinking_supported: true,
            plain_readouts: false,
            registry: None,
        });
        Self {
            root,
            store,
            profile,
        }
    }
    pub(crate) fn request(&self, key: &str) -> Value {
        json!({"schema_version":1,"idempotency_key":key,
            "input":{"kind":"messages","messages":[{"role":"user","content":"Name an animal."}],"generation_mode":"thinking","assistant_prefill":{"channel":"final","text":"Answer: "}},
            "generation":{"max_new_tokens":3,"sampling":{"temperature":0.0,"top_k":0,"top_p":1.0,"min_p":0.0,"seed":7}},
            "preconditions":{"model_identity":"cpu-fixture","asset_identities":{}}})
    }
    pub(crate) fn prepare(
        &self,
        key: &str,
    ) -> (String, Prepared, crate::ordinary_executor::ExecutionControl) {
        let request = self.request(key);
        let prepared = self
            .profile
            .prepare(&Request::parse(&request).unwrap())
            .ok()
            .unwrap();
        let accepted = self.store.accept(key, &request, false).unwrap();
        let id = accepted.status.id;
        let control = self.store.control(&id).unwrap();
        (id, prepared, control)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

// Real native tokenizer implementation over a synthetic byte vocabulary, not
// evidence about a released model's tokenizer or numerical generation.
fn write_byte_tokenizer(path: &std::path::Path, extra_tokens: &[String]) {
    fn string(bytes: &mut Vec<u8>, text: &str) {
        bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
        bytes.extend_from_slice(text.as_bytes());
    }
    let mut entries = Vec::new();
    for (key, value) in [
        ("general.architecture", "qwen35"),
        ("tokenizer.ggml.model", "gpt2"),
        ("tokenizer.ggml.pre", "qwen35"),
    ] {
        let mut bytes = Vec::new();
        string(&mut bytes, key);
        bytes.extend_from_slice(&8u32.to_le_bytes());
        string(&mut bytes, value);
        entries.push(bytes);
    }
    let mut extra = 0u32;
    let mut tokens = (0u32..256)
        .map(|byte| {
            let code = if (33..=126).contains(&byte)
                || (161..=172).contains(&byte)
                || (174..=255).contains(&byte)
            {
                byte
            } else {
                let code = 256 + extra;
                extra += 1;
                code
            };
            char::from_u32(code).unwrap().to_string()
        })
        .collect::<Vec<_>>();
    tokens.extend(
        [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<think>",
            "</think>",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    tokens.extend(extra_tokens.iter().cloned());
    for (key, values) in [
        ("tokenizer.ggml.tokens", tokens.as_slice()),
        ("tokenizer.ggml.merges", &[][..]),
    ] {
        let mut bytes = Vec::new();
        string(&mut bytes, key);
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            string(&mut bytes, value);
        }
        entries.push(bytes);
    }
    let mut types = Vec::new();
    string(&mut types, "tokenizer.ggml.token_type");
    types.extend_from_slice(&9u32.to_le_bytes());
    types.extend_from_slice(&5u32.to_le_bytes());
    types.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for index in 0..tokens.len() {
        types.extend_from_slice(
            &(if (256..261).contains(&index) {
                3i32
            } else {
                1i32
            })
            .to_le_bytes(),
        );
    }
    entries.push(types);
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for entry in entries {
        bytes.extend(entry);
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    std::fs::write(path, bytes).unwrap();
}

fn logits() -> Vec<f32> {
    let mut values = vec![0.0; 261];
    values[b'x' as usize] = 10.0;
    values
}

#[test]
fn plain_discovery_preconditions_and_admission_agree_without_enabling_other_diagnostics() {
    use crate::serve::lens_http::Admission;
    let mut fixture = Fixture::new();
    let make = |profile: Arc<Profile>| {
        let (sender, _) = std::sync::mpsc::sync_channel(1);
        NativeAdmission {
            profile,
            store: Arc::clone(&fixture.store),
            sender,
            gate: Default::default(),
            activity: Default::default(),
        }
    };
    assert_eq!(
        make(Arc::clone(&fixture.profile)).capabilities()["execution"]["baseline_only"],
        true
    );
    assert_eq!(
        make(Arc::clone(&fixture.profile)).assets()["assets"],
        json!([])
    );
    Arc::get_mut(&mut fixture.profile).unwrap().plain_readouts = true;
    let admission = make(Arc::clone(&fixture.profile));
    assert_eq!(
        admission.capabilities()["readout_modes"],
        json!(["full_vocabulary"])
    );
    assert_eq!(admission.capabilities()["operations"], json!([]));
    assert_eq!(admission.assets()["assets"][0]["direction_rows"], json!([]));
    let mut request = fixture.request("qualified-plain");
    request["diagnostics"] = json!({"directions":[],"operations":[],"readouts":[{"id":"r","lens":"plain","mode":"full_vocabulary","top_k":2,
        "scope":{"layers":{"kind":"all"},"decode":{"kind":"all"}}}]});
    assert!(
        fixture
            .profile
            .prepare(&Request::parse(&request).unwrap())
            .is_err()
    );
    request["preconditions"]["asset_identities"] = json!({"plain":"wrong-model"});
    assert_eq!(
        fixture
            .profile
            .prepare(&Request::parse(&request).unwrap())
            .err()
            .unwrap()
            .status,
        412
    );
    request["preconditions"]["asset_identities"] = json!({"plain":"cpu-fixture"});
    let prepared = fixture
        .profile
        .prepare(&Request::parse(&request).unwrap())
        .ok()
        .unwrap();
    let record: Value = serde_json::from_slice(&prepared.record).unwrap();
    assert_eq!(record["asset_identities"]["plain"], "cpu-fixture");
    assert_eq!(record["resolved_scopes"][0]["kind"], "readout");
    for field in ["directions", "operations", "residual_pairs"] {
        let mut unsupported = request.clone();
        unsupported["diagnostics"][field] = json!([{}]);
        assert!(
            fixture
                .profile
                .prepare(&Request::parse(&unsupported).unwrap())
                .is_err()
        );
    }
    request["diagnostics"]["readouts"][0]["retain"] = json!("scores_and_residual");
    let retained = fixture
        .profile
        .prepare(&Request::parse(&request).unwrap())
        .ok()
        .unwrap();
    assert!(retained.readouts.archive_bytes > 0);
}

#[test]
fn baseline_writer_retains_prompt_terminal_sample_and_successful_consumption() {
    for stop in [false, true] {
        let fixture = Fixture::new();
        let (id, prepared, control) = fixture.prepare("baseline");
        let writer = writer::Writer::spawn(
            Arc::clone(&fixture.store),
            id.clone(),
            control,
            &prepared,
            Default::default(),
        )
        .unwrap();
        writer.wait_ready().unwrap();
        let mut forwards = Vec::new();
        let outcome = execute::run_tokens(
            &prepared,
            writer.sink(),
            if stop { &[120] } else { &[] },
            |token, position, tail| {
                forwards.push((token, position, tail));
                Ok(logits())
            },
            |_| Ok(b"x".to_vec()),
        );
        let samples = if stop { 1 } else { 3 };
        assert_eq!(outcome.counters.sampled_tokens, samples);
        assert_eq!(outcome.counters.consumed_generated_tokens, samples - 1);
        assert_eq!(forwards.len(), prepared.prompt.len() + samples as usize - 1);
        assert!(
            forwards
                .iter()
                .enumerate()
                .all(|(index, (_, position, tail))| *position as usize == index
                    && *tail == (index + 1 >= prepared.prompt.len()))
        );
        writer.finish(outcome).unwrap();
        let status = fixture.store.status(&id).unwrap();
        assert_eq!(status.state, JobState::Completed);
        assert_eq!(status.generation.state, GenerationState::Completed);
        let page = serde_json::to_value(fixture.store.result(&id, None, 64).unwrap()).unwrap();
        let records = page["records"].as_array().unwrap();
        assert_eq!(records[0]["kind"], "prepared_input");
        assert_eq!(records[0]["assistant_prefill"]["text"], "Answer: ");
        let tokens = records
            .iter()
            .filter(|record| record["kind"] == "sampled_token")
            .collect::<Vec<_>>();
        assert_eq!(tokens.len(), samples as usize);
        assert_eq!(tokens.last().unwrap()["consumed"], false);
        assert_eq!(records.last().unwrap()["kind"], "generation_terminal");
        assert_eq!(page["complete"], true);
    }
}

#[test]
fn explicit_cancellation_and_server_stop_have_distinct_durable_outcomes() {
    for stop_server in [false, true] {
        let fixture = Fixture::new();
        let (id, prepared, control) = fixture.prepare("cancel");
        let gate = control::ExecutionGate::default();
        let writer = writer::Writer::spawn(
            Arc::clone(&fixture.store),
            id.clone(),
            control.clone(),
            &prepared,
            gate.clone(),
        )
        .unwrap();
        writer.wait_ready().unwrap();
        let outcome = execute::run_tokens(
            &prepared,
            writer.sink(),
            &[],
            |_, position, _| {
                assert_eq!(position, 0);
                if stop_server {
                    gate.close();
                } else {
                    control.cancel();
                }
                Ok(logits())
            },
            |_| panic!("cancelled during prefill"),
        );
        assert_eq!(outcome.counters.consumed_prompt_tokens, 1);
        assert_eq!(outcome.counters.sampled_tokens, 0);
        assert_eq!(
            outcome.reason,
            if stop_server {
                StopReason::ServerRestart
            } else {
                StopReason::Cancelled
            }
        );
        writer.finish(outcome).unwrap();
        assert_eq!(
            fixture.store.status(&id).unwrap().state,
            if stop_server {
                JobState::Interrupted
            } else {
                JobState::Cancelled
            }
        );
    }
}

#[test]
fn failed_forward_and_unavailable_piece_keep_sample_identity_without_claiming_consumption() {
    for piece_fails in [false, true] {
        let fixture = Fixture::new();
        let (id, prepared, control) = fixture.prepare("failed");
        let writer = writer::Writer::spawn(
            Arc::clone(&fixture.store),
            id.clone(),
            control,
            &prepared,
            Default::default(),
        )
        .unwrap();
        writer.wait_ready().unwrap();
        let outcome = execute::run_tokens(
            &prepared,
            writer.sink(),
            &[],
            |_, position, _| {
                ensure!(
                    (position as usize) < prepared.prompt.len(),
                    "injected failed forward"
                );
                Ok(logits())
            },
            |_| {
                ensure!(!piece_fails, "injected piece decode failure");
                Ok(b"x".to_vec())
            },
        );
        assert_eq!(outcome.reason, StopReason::ExecutionError);
        assert_eq!(outcome.counters.sampled_tokens, 1);
        assert_eq!(outcome.counters.consumed_generated_tokens, 0);
        writer.finish(outcome).unwrap();
        let page = serde_json::to_value(fixture.store.result(&id, None, 64).unwrap()).unwrap();
        let unresolved = page["records"]
            .as_array()
            .unwrap()
            .iter()
            .find(|record| record["kind"] == "unresolved_sample")
            .unwrap();
        assert_eq!(unresolved["token_id"], 120);
        assert_eq!(
            unresolved["reason"],
            if piece_fails {
                "token_piece_unavailable_not_consumed"
            } else {
                "forward_failed_consumption_unknown"
            }
        );
    }
}

#[test]
fn observer_failure_keeps_successful_consumption_and_does_not_fabricate_failed_forward() {
    use execute::TokenEngine;
    for fail_forward in [false, true] {
        let fixture = Fixture::new();
        let (id, prepared, control) = fixture.prepare("observe-failure");
        let writer = writer::Writer::spawn(
            Arc::clone(&fixture.store),
            id.clone(),
            control,
            &prepared,
            Default::default(),
        )
        .unwrap();
        writer.wait_ready().unwrap();
        struct Engine {
            prompt: usize,
            fail_forward: bool,
            observed: usize,
        }
        impl TokenEngine for Engine {
            fn forward(&mut self, _: i32, position: u32, _: bool) -> anyhow::Result<Vec<f32>> {
                ensure!(
                    !self.fail_forward || (position as usize) < self.prompt,
                    "failed forward"
                );
                Ok(logits())
            }
            fn observe(
                &mut self,
                _: i32,
                position: u32,
                _: &[f32],
                counters: &Counters,
                _: &Sink,
            ) -> anyhow::Result<()> {
                self.observed += 1;
                if position as usize >= self.prompt {
                    assert_eq!(counters.consumed_generated_tokens, 1);
                    anyhow::bail!("failed observation after successful consumption");
                }
                Ok(())
            }
        }
        let mut engine = Engine {
            prompt: prepared.prompt.len(),
            fail_forward,
            observed: 0,
        };
        let outcome = execute::run_engine(&prepared, writer.sink(), &[], &mut engine, |_| {
            Ok(b"x".to_vec())
        });
        assert_eq!(outcome.counters.sampled_tokens, 1);
        assert_eq!(
            outcome.counters.consumed_generated_tokens,
            u64::from(!fail_forward)
        );
        assert_eq!(
            engine.observed,
            prepared.prompt.len() + usize::from(!fail_forward)
        );
        writer.finish(outcome).unwrap();
        let page = serde_json::to_value(fixture.store.result(&id, None, 64).unwrap()).unwrap();
        let sample = page["records"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["kind"] == "sampled_token" || r["kind"] == "unresolved_sample")
            .unwrap();
        assert_eq!(
            sample["kind"],
            if fail_forward {
                "unresolved_sample"
            } else {
                "sampled_token"
            }
        );
        if !fail_forward {
            assert_eq!(sample["consumed"], true);
            assert_eq!(
                fixture
                    .store
                    .status(&id)
                    .unwrap()
                    .result
                    .error
                    .unwrap()
                    .code,
                "readout_failed"
            );
        }
    }
}
