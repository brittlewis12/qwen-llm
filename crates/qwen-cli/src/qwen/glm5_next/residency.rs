//! Opt-in phase observations. Probes never read tensor bytes or add mappings.

use qwen_llm::{gguf::GgufFile, metal::MetalContext};
use serde_json::json;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

qwen_llm::env_flag!(
    /// Log GLM run phase boundaries, sampled existing GGUF mapping cache residency, process I/O and Metal allocation. Observations only; no warming, eviction, locking or admission changes. Disabled by default.
    default_off enabled,
    "QWEN_GLM53_RESIDENCY_TRACE"
);

pub(crate) fn sample(ctx: &MetalContext, gguf: &GgufFile, phase: &str) {
    if !enabled() {
        return;
    }
    let started = Instant::now();
    let process = qwen_llm::pid_metrics::PidSnapshot::now().ok().map(|p| {
        json!({"pageins":p.pageins,"disk_read_bytes":p.diskio_bytesread,
            "rss_bytes":p.resident_size,"footprint_bytes":p.phys_footprint})
    });
    let shards: Vec<_> = gguf
        .shard_mapped_lengths()
        .into_iter()
        .enumerate()
        .map(|(i, len)| {
            let probe = gguf.try_shard_range(i, 0, len).ok().and_then(|bytes| {
                qwen_llm::cache_probe::probe_mapped_range_residency_sampled(bytes).ok()
            });
            json!({"shard":i,"mapped_bytes":len,
            "sample_pages":probe.as_ref().map(|p|p.total_pages),
            "sample_resident_pages":probe.as_ref().map(|p|p.resident_pages)})
        })
        .collect();
    eprintln!(
        "glm residency: {}",
        json!({"phase":phase,
        "unix_seconds":SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64(),
        "process":process,"metal_allocated_bytes":ctx.current_allocated_size(),
        "shards":shards,"probe_ms":started.elapsed().as_secs_f64()*1e3})
    );
}
