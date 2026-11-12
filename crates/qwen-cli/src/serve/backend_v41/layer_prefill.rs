//! Active prefill accounting excludes peer decode service and survives disposal.
use super::*;
pub(super) fn parse(key: &str, default: usize, maximum: usize) -> Result<usize> {
    let n = std::env::var(key)
        .ok()
        .map_or(Ok(default), |s| s.parse::<usize>())
        .with_context(|| format!("{key} must be 0..={maximum}"))?;
    ensure!(n <= maximum, "{key} must be 0..={maximum}");
    Ok(n)
}
#[derive(Default)]
pub(super) struct Work {
    pub armed: bool,
    pub finished: bool,
    pub elapsed_ms: f64,
    pub peer_rounds: usize,
    pub matched: usize,
    pub prompt: usize,
    pub reuse_source: &'static str,
}
impl Work {
    pub fn note(&mut self, ms: f64) {
        self.elapsed_ms += ms.max(0.0);
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        if self.armed && !self.finished && self.elapsed_ms > 0.0 {
            tracing::info!(target:"qwen_diag", "serve prefill: family=deepseek_v41 complete=false cancelled=true phase=layer prefill_ms={:.3} reused_tokens={} prefill_tokens={} reused_source={} layer_rounds={}",
                self.elapsed_ms, self.matched, self.prompt.saturating_sub(self.matched), self.reuse_source, self.peer_rounds);
        }
    }
}
