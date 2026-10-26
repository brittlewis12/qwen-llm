//! Whole-process counters during native consumed-token steps, not GPU or weight-only IO.
use qwen_llm::pid_metrics::PidSnapshot;
use serde_json::{Value, json};

#[derive(Default)]
struct Metric {
    count: u64,
    total: u64,
    max: u64,
}

impl Metric {
    fn record(&mut self, value: Option<u64>) {
        if let Some(value) = value {
            self.count += 1;
            self.total += value;
            self.max = self.max.max(value);
        }
    }

    fn json(&self, attempts: u64) -> Value {
        // Partial observations must not masquerade as a complete token total/mean.
        let complete = self.count > 0 && self.count == attempts;
        json!({
            "count": self.count,
            "total": complete.then_some(self.total),
            "max": complete.then_some(self.max),
            "per_token_mean": complete.then(|| self.total as f64 / self.count as f64),
        })
    }
}

#[derive(Default)]
pub(super) struct Observations {
    attempted_tokens: u64,
    successful_tokens: u64,
    pageins: Metric,
    disk_read_bytes: Metric,
}

#[derive(Default)]
pub(super) struct ChunkObservations {
    calls: Observations,
    attempted_tokens: u64,
    successful_tokens: u64,
}

impl ChunkObservations {
    pub(super) fn json(&self) -> Value {
        let mut value = self.calls.json();
        value["scope"] = json!("whole_process_native_prefill_call");
        value["attempted_calls"] = json!(self.calls.attempted_tokens);
        value["successful_calls"] = json!(self.calls.successful_tokens);
        value["attempted_tokens"] = json!(self.attempted_tokens);
        value["successful_tokens"] = json!(self.successful_tokens);
        for key in ["process_pageins", "process_disk_read_bytes"] {
            let metric = value[key].as_object_mut().unwrap();
            let max = metric.remove("max").unwrap();
            metric.insert("max_call".into(), max);
            metric.insert(
                "per_token_mean".into(),
                json!(
                    metric["total"]
                        .as_u64()
                        .filter(|_| self.attempted_tokens > 0)
                        .map(|total| total as f64 / self.attempted_tokens as f64)
                ),
            );
        }
        value
    }
}

pub(super) fn observe_chunk<T, E>(
    observations: Option<&mut ChunkObservations>,
    position: usize,
    rows: usize,
    snapshot: impl FnMut() -> Option<PidSnapshot>,
    prefill: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    let Some(observations) = observations else {
        return prefill();
    };
    let result = observe_call(
        Some(&mut observations.calls),
        position,
        rows,
        "whole_process_native_prefill_call",
        snapshot,
        prefill,
    );
    observations.attempted_tokens += rows as u64;
    if result.is_ok() {
        observations.successful_tokens += rows as u64;
    }
    result
}

impl Observations {
    pub(super) fn json(&self) -> Value {
        json!({
            "scope": "whole_process_native_step",
            "attempted_tokens": self.attempted_tokens,
            "successful_tokens": self.successful_tokens,
            "process_pageins": self.pageins.json(self.attempted_tokens),
            "process_disk_read_bytes": self.disk_read_bytes.json(self.attempted_tokens),
        })
    }
}

pub(super) fn observe<T, E>(
    observations: Option<&mut Observations>,
    position: usize,
    snapshot: impl FnMut() -> Option<PidSnapshot>,
    step: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    observe_call(
        observations,
        position,
        1,
        "whole_process_native_step",
        snapshot,
        step,
    )
}

fn observe_call<T, E>(
    observations: Option<&mut Observations>,
    position: usize,
    rows: usize,
    scope: &str,
    mut snapshot: impl FnMut() -> Option<PidSnapshot>,
    step: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    let Some(observations) = observations else {
        return step();
    };
    let before = snapshot();
    let result = step();
    let after = snapshot();
    let pair = before.zip(after);
    let pageins = pair.and_then(|(b, a)| a.pageins.checked_sub(b.pageins));
    let disk_read_bytes =
        pair.and_then(|(b, a)| a.diskio_bytesread.checked_sub(b.diskio_bytesread));
    observations.attempted_tokens += 1;
    observations.successful_tokens += u64::from(result.is_ok());
    observations.pageins.record(pageins);
    observations.disk_read_bytes.record(disk_read_bytes);
    tracing::info!(target: "qwen_diag",
        "serve forward io: family=deepseek_v41 scope={scope} position={position} rows={rows} success={} process_pageins={} process_disk_read_bytes={}",
        result.is_ok(),
        pageins.map_or_else(|| "null".into(), |n| n.to_string()),
        disk_read_bytes.map_or_else(|| "null".into(), |n| n.to_string()));
    result
}
