//! Native messages input adapter; rendering/tokenization are shared with Lens CLI.

use crate::lens_input::{
    LensMessageMode, PreparedLensGenerationInput, prepare_qwen_generation_messages_bytes,
};
use crate::model_request::prefill::AssistantPrefill;
use crate::prompt_template::QwenPromptTemplate;
use anyhow::{Context, Result, ensure};
use qwen_llm::sampling::{Sampler, SamplingConfig};
use qwen_llm::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    pub(crate) schema_version: u32,
    pub(crate) idempotency_key: String,
    pub(crate) input: Input,
    pub(crate) generation: Generation,
    #[serde(default)]
    pub(crate) diagnostics: Option<Diagnostics>,
    #[serde(default)]
    pub(crate) preconditions: Option<Preconditions>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Preconditions {
    pub(crate) model_identity: String,
    pub(crate) asset_identities: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Input {
    Messages {
        messages: Vec<Message>,
        generation_mode: LensMessageMode,
        assistant_prefill: Option<AssistantPrefill>,
    },
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Message {
    role: Role,
    content: String,
    #[serde(
        default,
        rename(serialize = "reasoning_content", deserialize = "reasoning"),
        skip_serializing_if = "Option::is_none"
    )]
    reasoning: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Generation {
    pub(crate) max_new_tokens: usize,
    pub(crate) sampling: Sampling,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Sampling {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    seed: u64,
}

impl Sampling {
    pub(crate) fn config(&self) -> SamplingConfig {
        SamplingConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            seed: self.seed,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Diagnostics {
    pub(crate) directions: Vec<Value>,
    pub(crate) operations: Vec<Value>,
    pub(crate) readouts: Vec<Value>,
    #[serde(default)]
    pub(crate) residual_pairs: Vec<Value>,
}

impl Request {
    pub(crate) fn parse(value: &Value) -> Result<Self> {
        // Validate before f64 -> f32 conversion: a tiny negative temperature
        // must not round to -0 and silently become greedy sampling.
        for (field, low, high, inclusive_low) in [
            ("temperature", 0.0, f64::from(f32::MAX), true),
            ("top_p", 0.0, 1.0, false),
            ("min_p", 0.0, 1.0, true),
        ] {
            let number = value
                .pointer(&format!("/generation/sampling/{field}"))
                .and_then(Value::as_f64)
                .with_context(|| format!("sampling.{field} must be a number"))?;
            ensure!(
                number.is_finite()
                    && number <= high
                    && if inclusive_low {
                        number >= low
                    } else {
                        number > low
                    },
                "sampling.{field} is outside its native range"
            );
        }
        let request: Self =
            serde_json::from_value(value.clone()).context("parse native Lens request")?;
        if let Some(value) = value.get("preconditions") {
            ensure!(
                value.is_object(),
                "preconditions must be an object when present"
            );
        }
        if let Some(expected) = &request.preconditions {
            ensure!(
                !expected.model_identity.is_empty() && expected.model_identity.len() <= 256,
                "precondition model identity must contain 1..256 bytes"
            );
            ensure!(
                expected.asset_identities.len() <= 65,
                "too many precondition asset identities"
            );
            for (alias, identity) in &expected.asset_identities {
                ensure!(
                    !alias.is_empty()
                        && alias.len() <= 64
                        && alias
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
                        && !identity.is_empty()
                        && identity.len() <= 256,
                    "invalid precondition alias or identity"
                );
            }
        }
        let temperature = value
            .pointer("/generation/sampling/temperature")
            .and_then(Value::as_number)
            .context("sampling.temperature must be a number")?;
        // Arbitrary-precision JSON retains intent even when f64 underflows.
        let spelling = temperature.to_string();
        let nonzero = spelling
            .split(['e', 'E'])
            .next()
            .unwrap_or_default()
            .bytes()
            .any(|byte| matches!(byte, b'1'..=b'9'));
        if nonzero {
            ensure!(
                request.generation.sampling.temperature > 0.0,
                "nonzero temperature must remain positive in native f32 sampling; use zero explicitly for greedy"
            );
        }
        ensure!(request.schema_version == 1, "schema_version must be 1");
        ensure!(
            !request.idempotency_key.is_empty() && request.idempotency_key.len() <= 256,
            "invalid idempotency key"
        );
        ensure!(
            request.generation.max_new_tokens > 0,
            "max_new_tokens must be positive"
        );
        Sampler::new(request.generation.sampling.config()).context("validate native sampler")?;
        let Input::Messages {
            messages,
            assistant_prefill,
            ..
        } = &request.input;
        for message in messages {
            ensure!(
                message.role == Role::Assistant || message.reasoning.is_none(),
                "reasoning is assistant-only"
            );
        }
        crate::messages::parse_strict_ordinary_chat_input(
            &serde_json::to_string(messages)?,
            "native Lens messages",
        )?;
        if let Some(prefill) = assistant_prefill {
            prefill.validate_qwen()?;
        }
        Ok(request)
    }

    pub(crate) fn prefill(&self) -> Option<&AssistantPrefill> {
        let Input::Messages {
            assistant_prefill, ..
        } = &self.input;
        assistant_prefill.as_ref()
    }

    /// The caller binds protocol and context to the resident model; neither is
    /// client-selected. Diagnostic plans are bound separately before acceptance.
    pub(crate) fn prepare_generation(
        &self,
        protocol: QwenPromptTemplate,
        tokenizer: &Tokenizer,
        max_context: usize,
        max_new_tokens: usize,
    ) -> Result<PreparedLensGenerationInput> {
        ensure!(
            self.generation.max_new_tokens <= max_new_tokens,
            "max_new_tokens exceeds the server limit"
        );
        ensure!(
            matches!(
                protocol,
                QwenPromptTemplate::Qwen36 | QwenPromptTemplate::Qwen38
            ),
            "native generation requires a qualified Qwen3.6 or Qwen3.8 template"
        );
        let Input::Messages {
            messages,
            generation_mode,
            assistant_prefill,
        } = &self.input;
        let prepared = prepare_qwen_generation_messages_bytes(
            &serde_json::to_vec(messages)?,
            "native Lens messages",
            Some(*generation_mode),
            assistant_prefill.as_ref(),
            protocol,
            tokenizer,
        )?;
        let forwards = prepared
            .input
            .token_ids
            .len()
            .checked_add(self.generation.max_new_tokens - 1)
            .context("native context count overflow")?;
        ensure!(
            !prepared.input.token_ids.is_empty() && forwards <= max_context,
            "prompt plus generation exceeds the model/server context limit"
        );
        Ok(prepared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request() -> Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/lens_http_v1/request.json"
        ))
        .unwrap()
    }

    #[test]
    fn fixture_retains_sampling_prefix_and_ordered_diagnostics() {
        let value = request();
        let parsed = Request::parse(&value).unwrap();
        assert_eq!(
            parsed.generation.max_new_tokens,
            value["generation"]["max_new_tokens"].as_u64().unwrap() as usize
        );
        let sampler = parsed.generation.sampling.config();
        assert_eq!(
            sampler.temperature.to_bits(),
            (value["generation"]["sampling"]["temperature"]
                .as_f64()
                .unwrap() as f32)
                .to_bits()
        );
        assert_eq!(
            sampler.top_p.to_bits(),
            (value["generation"]["sampling"]["top_p"].as_f64().unwrap() as f32).to_bits()
        );
        assert_eq!(
            sampler.min_p.to_bits(),
            (value["generation"]["sampling"]["min_p"].as_f64().unwrap() as f32).to_bits()
        );
        assert_eq!(
            sampler.seed,
            value["generation"]["sampling"]["seed"].as_u64().unwrap()
        );
        assert_eq!(
            parsed.diagnostics.unwrap().operations,
            value["diagnostics"]["operations"]
                .as_array()
                .unwrap()
                .clone()
        );
    }

    #[test]
    fn validation_rejects_unknown_paths_roles_fields_and_sampling_coercion() {
        for (pointer, value) in [
            ("/schema_version", json!(2)),
            ("/generation/max_new_tokens", json!(0)),
            ("/generation/sampling/temperature", json!(-1e-50)),
            ("/generation/sampling/temperature", json!(1e-50)),
            ("/generation/sampling/top_p", json!(0)),
            ("/generation/sampling/top_p", json!(1.000000001)),
            ("/generation/sampling/min_p", json!(1.000000001)),
            ("/input/kind", json!("raw")),
            ("/input/messages/0/role", json!("tool")),
        ] {
            let mut request = request();
            *request.pointer_mut(pointer).unwrap() = value;
            assert!(Request::parse(&request).is_err(), "{pointer}");
        }
        for pointer in [
            "",
            "/input",
            "/generation",
            "/generation/sampling",
            "/diagnostics",
            "/input/messages/0",
        ] {
            let mut request = request();
            request.pointer_mut(pointer).unwrap()["model_path"] = json!("/tmp/model.gguf");
            assert!(Request::parse(&request).is_err(), "{pointer}");
        }
    }

    #[test]
    fn validation_uses_shared_conversation_grammar_and_keeps_full_u64_seed() {
        let mut value = request();
        value["input"]["messages"] = json!([{"role":"assistant","content":"not a user turn"}]);
        assert!(Request::parse(&value).is_err());
        let mut value = request();
        value["input"]["messages"][0]["reasoning"] = json!("user cannot own reasoning");
        assert!(Request::parse(&value).is_err());
        let mut value = request();
        value["generation"]["sampling"]["seed"] = json!(u64::MAX);
        assert_eq!(
            Request::parse(&value).unwrap().generation.sampling.seed,
            u64::MAX
        );
        value["input"]["messages"] = json!([{"role":"user","content":"first"}, {"role":"assistant","content":"answer","reasoning":"retained reasoning"}, {"role":"user","content":"next"}]);
        assert!(Request::parse(&value).is_ok());
    }

    #[test]
    fn temperature_keeps_authored_greedy_intent_before_f64_underflow() {
        for spelling in ["1e-500", "-1e-500", "1e-50", "-1e-50"] {
            let mut value = request();
            value["generation"]["sampling"]["temperature"] =
                serde_json::from_str(spelling).unwrap();
            assert!(Request::parse(&value).is_err(), "{spelling}");
        }
        for spelling in ["0", "-0.0", "0e-500", "-0e+500", "1e-45"] {
            let mut value = request();
            value["generation"]["sampling"]["temperature"] =
                serde_json::from_str(spelling).unwrap();
            assert!(Request::parse(&value).is_ok(), "{spelling}");
        }
    }

    #[test]
    #[ignore = "CPU-only tokenizer/template parity; requires QWEN_PREFILL_GGUF"]
    fn native_and_cli_prefill_token_parity_cpu_only() {
        use crate::lens_input::{LensInputSpec, prepare_qwen_model_generation_input};
        use crate::model_request::prefill::{AssistantPrefill, AssistantPrefillChannel};
        use crate::prompt_template::{ModelPromptTemplate, resolve_model_prompt_template};
        let path = std::env::var("QWEN_PREFILL_GGUF").expect("set QWEN_PREFILL_GGUF");
        let gguf = qwen_llm::gguf::GgufFile::open(path).unwrap();
        let family = qwen_llm::model_family::ModelFamily::detect(&gguf).unwrap();
        let ModelPromptTemplate::Qwen(protocol) = resolve_model_prompt_template(&gguf).unwrap()
        else {
            panic!("Qwen required");
        };
        let tokenizer = std::sync::Arc::new(Tokenizer::from_gguf(&gguf).unwrap());
        let arch = qwen_llm::loader::Model::from_gguf(&gguf).unwrap().arch;
        let profile = crate::serve::native::Profile {
            model_id: "CPU parity".into(),
            identity: "CPU parity".into(),
            protocol,
            tokenizer: tokenizer.clone(),
            layers: arch.n_layer,
            hidden: arch.hidden_size as usize,
            context: 8192,
            max_tokens: 4096,
            no_thinking_supported: true,
            plain_readouts: false,
            registry: None,
        };
        for mode in [LensMessageMode::Thinking, LensMessageMode::NoThinking] {
            for prefill in [
                None,
                Some(AssistantPrefill {
                    channel: AssistantPrefillChannel::Final,
                    text: "  Answer:\n".into(),
                }),
                Some(AssistantPrefill {
                    channel: AssistantPrefillChannel::Reasoning,
                    text: "  Let me consider\n".into(),
                }),
            ] {
                let mut authored = request();
                authored.as_object_mut().unwrap().remove("preconditions");
                authored["input"]["messages"] = json!([{"role":"system","content":"Be precise."},{"role":"user","content":"Explain this."}]);
                authored["input"]["generation_mode"] = serde_json::to_value(mode).unwrap();
                authored["input"]["assistant_prefill"] = serde_json::to_value(&prefill).unwrap();
                authored["diagnostics"] = json!({"directions":[],"operations":[],"readouts":[]});
                let native_request = Request::parse(&authored).unwrap();
                let native = native_request.prepare_generation(protocol, &tokenizer, 8192, 4096);
                let spec = LensInputSpec {
                    prompt: None,
                    token_ids: None,
                    user: Some("Explain this."),
                    system: Some("Be precise."),
                    messages: None,
                    open_responses: None,
                    no_special_tokens: false,
                    message_mode: Some(mode),
                };
                let cli = if let Some(prefill) = prefill.as_ref() {
                    prepare_qwen_model_generation_input(spec, prefill, family, &gguf, &tokenizer)
                        .map(|(input, record)| (input, Some(serde_json::to_value(record).unwrap())))
                } else {
                    crate::lens_input::prepare_qwen_model_input(spec, family, &gguf, &tokenizer)
                        .map(|input| (input, None))
                };
                if mode == LensMessageMode::NoThinking
                    && prefill.as_ref().is_some_and(|prefill| {
                        prefill.channel == AssistantPrefillChannel::Reasoning
                    })
                {
                    assert!(native.is_err() && cli.is_err());
                    continue;
                }
                let native = native.unwrap();
                let cli = cli.unwrap();
                assert_eq!(native.input.token_ids, cli.0.token_ids);
                assert_eq!(native.input.rendering.spans, cli.0.rendering.spans);
                let record = serde_json::to_value(native.record(native_request.prefill())).unwrap();
                if let Some(cli_record) = cli.1 {
                    assert_eq!(record, cli_record);
                }
                assert_eq!(
                    record["prompt_text"].as_str().unwrap().as_bytes(),
                    serde_json::from_value::<Vec<u8>>(record["prompt_bytes"].clone()).unwrap()
                );
                assert_eq!(
                    record["token_ids"],
                    serde_json::to_value(
                        tokenizer
                            .encode(record["prompt_text"].as_str().unwrap(), false)
                            .unwrap()
                    )
                    .unwrap()
                );
                assert!(
                    native_request
                        .prepare_generation(protocol, &tokenizer, 1, 4096)
                        .is_err()
                );
                assert!(
                    native_request
                        .prepare_generation(protocol, &tokenizer, 8192, 1)
                        .is_err()
                );
                authored["diagnostics"]["operations"] = json!([{"id":"must_not_be_ignored"}]);
                assert!(
                    profile
                        .prepare(&Request::parse(&authored).unwrap())
                        .is_err()
                );
            }
        }
    }
}
