//! Stable JSON projection and text summaries for `qwen info`.

use qwen_llm::gguf::GgufFile;
use qwen_llm::model_family::ModelFamily;
use serde_json::{Value, json};

/// Family- and header-derived values collected by the CLI before assembly.
pub struct ProjectionInput {
    pub model: String,
    pub architecture: Option<String>,
    pub family: Option<&'static str>,
    pub drafter_run: Value,
    pub drafter_serve: Value,
    pub prompt_lookup: Value,
    pub sampling: Value,
    pub weight_storage: Value,
    pub capabilities: Value,
    pub os_build: Option<String>,
}

/// Assemble the versioned `qwen info --json` record.
pub fn assemble_projection(input: ProjectionInput) -> Value {
    json!({
        "version": "qwen_info_v1",
        "model": input.model,
        "architecture": input.architecture,
        "family": input.family,
        "drafter": {
            "run": input.drafter_run,
            "serve": input.drafter_serve,
        },
        "prompt_lookup": { "run": input.prompt_lookup },
        "sampling": input.sampling,
        "weight_storage": input.weight_storage,
        "os_build": input.os_build,
        "capabilities": input.capabilities,
    })
}

/// One line per capability for the text view of `qwen info`.
pub fn capability_lines(projection: &Value) -> Vec<String> {
    fn status(value: &Value) -> String {
        let field = |key: &str| value.get(key).and_then(Value::as_str);
        match (field("status"), field("code")) {
            (Some(status), Some(code)) => format!("{status} ({code})"),
            (Some(status), None) => status.to_owned(),
            (None, _) => "-".to_owned(),
        }
    }
    let mut lines = Vec::new();
    for (label, path) in [
        ("drafter run", "/drafter/run"),
        ("drafter serve", "/drafter/serve"),
        ("prompt lookup", "/prompt_lookup/run"),
    ] {
        if let Some(value) = projection.pointer(path) {
            lines.push(format!("{label}: {}", status(value)));
        }
    }
    if let Some(sampling) = projection.get("sampling").filter(|value| value.is_object()) {
        lines.push(format!(
            "sampling: temperature {} top-k {} top-p {} min-p {} ({})",
            sampling["temperature"],
            sampling["top_k"],
            sampling["top_p"],
            sampling["min_p"],
            sampling["identity"].as_str().unwrap_or("-"),
        ));
    }
    if let Some(os_build) = projection.get("os_build") {
        lines.push(format!(
            "os_build: {}",
            os_build.as_str().unwrap_or("unavailable")
        ));
    }
    let capabilities = &projection["capabilities"];
    if let Some(reasoning) = capabilities.get("reasoning") {
        let levels = reasoning
            .get("levels")
            .and_then(Value::as_array)
            .map(|levels| {
                levels
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join("|")
            });
        lines.push(match levels.filter(|levels| !levels.is_empty()) {
            Some(levels) => format!(
                "reasoning: levels {levels}; no-thinking {}",
                reasoning
                    .get("no_thinking")
                    .map_or_else(|| "-".to_owned(), status)
            ),
            None if reasoning.get("status").is_some() => {
                format!("reasoning: {}", status(reasoning))
            }
            None => format!(
                "reasoning: no effort levels; no-thinking {}",
                reasoning
                    .get("no_thinking")
                    .map_or_else(|| "-".to_owned(), status)
            ),
        });
    }
    if let Some(input) = capabilities.get("input").and_then(Value::as_object) {
        let forms: Vec<String> = input
            .iter()
            .filter(|(_, value)| value.is_object())
            .map(|(form, value)| format!("{form} {}", status(value)))
            .collect();
        if !forms.is_empty() {
            lines.push(format!("input: {}", forms.join(", ")));
        }
    }
    if let Some(template) = capabilities.get("template") {
        let rendered = template
            .get("rendered_as")
            .and_then(Value::as_str)
            .map_or(String::new(), |name| format!(" as {name}"));
        lines.push(format!("template: {}{rendered}", status(template)));
        if let Some(contract) = template.get("rendering_contract") {
            lines.push(format!(
                "rendering contract: {} ({})",
                status(contract),
                contract["name"].as_str().unwrap_or("-")
            ));
        }
    }
    if let Some(execution) = capabilities.get("execution") {
        if execution.get("status").is_some() {
            lines.push(format!("execution: {}", status(execution)));
        } else if let Some(lanes) = execution.as_object() {
            for (lane, value) in lanes {
                lines.push(format!("{lane}: {}", status(value)));
            }
        }
    }
    lines
}

pub fn weight_storage_projection(
    family: Option<ModelFamily>,
    gguf: &GgufFile,
) -> serde_json::Value {
    if !matches!(family, Some(ModelFamily::Qwen35 | ModelFamily::Qwen35Moe)) {
        return serde_json::Value::Null;
    }
    let bound = match qwen_llm::loader::Model::from_gguf(gguf) {
        Ok(model) => model,
        Err(error) => {
            return serde_json::json!({ "status": "unavailable", "reason": error.to_string() });
        }
    };
    let storage_configuration =
        match qwen_llm::metal_forward::resolve_gguf_storage_load_configuration() {
            Ok(configuration) => configuration,
            Err(error) => {
                return serde_json::json!({ "status": "unavailable", "reason": error.to_string() });
            }
        };
    let requests =
        match qwen_llm::metal_forward::model_weight_storage_requests_for_runtime(gguf, &bound) {
            Ok(requests) => requests,
            Err(error) => {
                return serde_json::json!({ "status": "unavailable", "reason": error.to_string() });
            }
        };
    let device = match qwen_llm::metal::probe_metal_device() {
        Ok(device) => device,
        Err(error) => {
            return serde_json::json!({ "status": "unavailable", "reason": error.to_string() });
        }
    };
    let copied = requests.iter().try_fold(0u64, |sum, request| {
        device
            .shared_buffer_size_and_align(request.resident_bytes)
            .map(|price| sum.saturating_add(price.size))
    });
    let mut copied = match copied {
        Ok(bytes) => bytes,
        Err(error) => {
            return serde_json::json!({ "status": "unavailable", "reason": error.to_string() });
        }
    };
    let derived = qwen_llm::metal_forward::fused_qkv_derived_weight_sizes(
        &bound,
        qwen_llm::metal_forward::fused_qkv_g8_enabled_for_runtime(),
    )
    .and_then(|sizes| {
        sizes.into_iter().try_fold(0u64, |sum, bytes| {
            device
                .shared_buffer_size_and_align(bytes)
                .map(|price| sum.saturating_add(price.size))
                .map_err(qwen_llm::metal_forward::MfError::Metal)
        })
    });
    match derived {
        Ok(bytes) => copied = copied.saturating_add(bytes),
        Err(error) => {
            return serde_json::json!({ "status": "unavailable", "reason": error.to_string() });
        }
    }
    let request = storage_configuration.storage_request();
    let signals = device.memory_signals();
    let mapping_unsupported = qwen_llm::metal_forward::runtime_retained_mapping_unsupported_reason(
        gguf,
        &bound,
        usize::try_from(device.max_buffer_length()).unwrap_or(usize::MAX),
    );
    let decision = storage_decision(
        copied,
        0,
        None,
        signals,
        device.device_facts().unified_memory,
        request,
        mapping_unsupported,
    );
    weight_storage_report(
        if decision.map_weights {
            "mapped"
        } else {
            "copied"
        },
        copied,
        signals.recommended_max_bytes,
        decision.reason,
    )
}

pub fn storage_decision(
    copied_bytes: u64,
    auxiliary_weight_bytes: u64,
    session_reserve_bytes: Option<u64>,
    signals: qwen_llm::metal::MetalMemorySignals,
    unified_memory: bool,
    request: qwen_llm::metal_forward::GgufStorageRequest,
    mapping_unsupported_reason: Option<&'static str>,
) -> qwen_llm::metal_forward::GgufNoCopyDecision {
    qwen_llm::metal_forward::decide_gguf_storage_mode_with_signals(
        copied_bytes,
        auxiliary_weight_bytes,
        session_reserve_bytes,
        signals,
        unified_memory,
        request,
        mapping_unsupported_reason,
    )
}

pub fn weight_storage_report(
    mode: &str,
    copied_bytes: u64,
    working_set_bytes: u64,
    reason: &str,
) -> serde_json::Value {
    serde_json::json!({
        "status": if mode == "unavailable" { "unavailable" } else { "available" },
        "mode": mode,
        "load_intent": "reusable",
        "copied_bytes_estimate": copied_bytes,
        "recommended_working_set_bytes": working_set_bytes,
        "reason": reason,
        "margin_policy": "storage-performance heuristic fitted to measurements; not a memory-admission guarantee (76% copied decoded faster, 97% copied loaded 4-6x slower with heavy paging)",
    })
}
