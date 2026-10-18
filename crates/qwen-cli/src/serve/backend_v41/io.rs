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
        "serve token io: family=deepseek_v41 scope=whole_process_native_step position={position} success={} process_pageins={} process_disk_read_bytes={}",
        result.is_ok(),
        pageins.map_or_else(|| "null".into(), |n| n.to_string()),
        disk_read_bytes.map_or_else(|| "null".into(), |n| n.to_string()));
    result
}
