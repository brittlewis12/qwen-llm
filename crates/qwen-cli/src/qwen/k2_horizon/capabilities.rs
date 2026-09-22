use super::*;
use qwen_llm::k2_horizon_runtime::K2AdmissionError;
use serde_json::{Value, json};

fn rejected(error: &K2AdmissionError) -> Value {
    json!({"status":"rejected", "code":error.code(), "message":error.to_string()})
}

pub(crate) fn project(gguf: &GgufFile) -> Result<Value> {
    let (core, generation) = match K2PreparedArtifact::inspect(gguf) {
        Ok(artifact) => (
            json!({"status":"passed"}),
            match artifact.generation_stops() {
                Ok(_) => json!({"status":"passed"}),
                Err(error) => rejected(&error),
            },
        ),
        Err(error) => (
            rejected(&error),
            json!({"status":"not_evaluated", "code":"k2_core_rejected"}),
        ),
    };
    let chat = if generation["status"] == "passed" {
        Some(chat_projection(gguf)?)
    } else {
        None
    };
    shutdown::checkpoint()?;
    Ok(project_reports(core, generation, chat))
}

fn project_reports(core: Value, generation: Value, chat: Option<Value>) -> Value {
    let mut execution = family_implementation();
    for lane in ["run", "serve", "bench", "lens"] {
        let admission = if lane == "lens" || core["status"] != "passed" {
            &core
        } else {
            &generation
        };
        let implementation = execution[lane]["status"].take();
        execution[lane]["implementation_status"] = implementation;
        execution[lane]["status"] = json!(if admission["status"] == "passed" {
            "conditional"
        } else {
            "unsupported"
        });
        execution[lane]["artifact_admission"] = admission.clone();
    }
    execution["artifact"] = json!({"core":core,"generation":generation,
        "scope":"cpu_layout_tokenizer_and_stop_policy_not_device_or_numerical_qualification"});
    execution["request_device"] = json!({"status":"not_evaluated",
        "requires":["request_options_and_token_budget","actual_device_buffer_limits","live_memory_admission"]});
    let can_generate = execution["artifact"]["generation"]["status"] == "passed";
    let verified_chat = can_generate
        && chat
            .as_ref()
            .is_some_and(|c| c["template"]["status"] == "identified");
    execution["serve"]["chat"] = json!(verified_chat);
    execution["serve"]["tools"] = json!(verified_chat);
    execution["serve"]["input"] = json!(if verified_chat {
        "raw_string_or_verified_chat_items"
    } else {
        "raw_string_only"
    });
    execution["run"]["scope"] = json!(if verified_chat {
        "raw_or_verified_chat_and_tools"
    } else {
        "raw_only"
    });
    let unsupported = json!({"status":"unsupported", "code":"chat_profile_unverified",
        "message":"K2 chat requires admitted generation and a verified checkpoint profile"});
    let raw = if can_generate {
        json!({"status":"supported"})
    } else {
        let report = &execution["run"]["artifact_admission"];
        json!({"status":"unsupported", "code":report["code"], "message":report["message"]})
    };
    let mut result = json!({
        "execution":execution,
        "input":{"raw":raw,"user":unsupported,"messages":unsupported,
            "tools":{"status":"unsupported","code":"chat_profile_unverified","message":"K2 tools require a verified checkpoint profile"}},
        "reasoning":unsupported,
        "template":{"status":"not_evaluated","rendered_as":null,"code":"k2_generation_not_admitted"}
    });
    if can_generate && let Some(chat) = chat {
        for (key, value) in chat.as_object().expect("K2 chat projection") {
            result[key] = value.clone();
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn k2_artifact_projection_never_promotes_family_support_to_device_admission() {
        let passed = json!({"status":"passed"});
        for code in [
            "k2_configuration",
            "k2_tensor_inventory",
            "k2_embedding_storage",
            "k2_retained_storage",
            "k2_tokenizer",
        ] {
            let p = project_reports(
                json!({"status":"rejected","code":code,"message":"fixture"}),
                json!({"status":"not_evaluated"}),
                None,
            );
            for lane in ["run", "serve", "bench", "lens"] {
                assert_eq!(p["execution"][lane]["status"], "unsupported");
                assert_eq!(p["execution"][lane]["artifact_admission"]["code"], code);
            }
            assert_eq!(p["input"]["raw"]["status"], "unsupported");
            assert_eq!(p["template"]["status"], "not_evaluated");
            assert_eq!(p["execution"]["request_device"]["status"], "not_evaluated");
        }
        let stops = project_reports(
            passed.clone(),
            json!({"status":"rejected","code":"k2_generation_stops","message":"fixture"}),
            None,
        );
        assert_eq!(stops["execution"]["lens"]["status"], "conditional");
        assert_eq!(stops["execution"]["run"]["status"], "unsupported");
        assert_eq!(stops["template"]["status"], "not_evaluated");
        let raw = project_reports(
            passed.clone(),
            passed.clone(),
            Some(json!({"template":{"status":"unverified"}})),
        );
        assert_eq!(raw["input"]["raw"]["status"], "supported");
        assert_eq!(raw["input"]["user"]["status"], "unsupported");
        assert_eq!(raw["execution"]["serve"]["chat"], false);
        assert_eq!(raw["execution"]["run"]["status"], "conditional");
        assert_eq!(
            raw["execution"]["run"]["implementation_status"],
            "supported"
        );
        let chat = project_reports(
            passed.clone(),
            passed,
            Some(
                json!({"template":{"status":"identified"},"input":{"user":{"status":"supported"}}}),
            ),
        );
        assert_eq!(chat["execution"]["serve"]["chat"], true);
        assert_eq!(
            chat["execution"]["request_device"]["status"],
            "not_evaluated"
        );
        assert_eq!(chat["execution"]["local_fitting"]["status"], "unsupported");
    }
}
