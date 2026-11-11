//! Accounting follows the request, including cancellation by Driver before its
//! next prefill callback. A partially advanced session is dropped with the run;
//! no idle/live state or snapshot is published from it.
#[derive(Default)]
pub(super) struct Work {
    pub armed: bool,
    pub finished: bool,
    pub elapsed_ms: f64,
    pub gpu_ms: f64,
    pub commands: usize,
    pub chunk_end: Option<usize>,
    pub matched: usize,
    pub prompt: usize,
    pub reuse_source: &'static str,
}
impl Work {
    pub fn note(&mut self, ms: f64) {
        self.elapsed_ms += ms.max(0.0);
    }
    pub fn partial(&self) -> Option<(f64, usize)> {
        (self.armed && !self.finished && self.elapsed_ms > 0.0)
            .then_some((self.elapsed_ms, self.prompt.saturating_sub(self.matched)))
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        if let Some((prefill_ms, rows)) = self.partial() {
            tracing::info!(target:"qwen_diag",
                "serve prefill: family=glm5_next complete=false cancelled=true phase=layer prefill_ms={prefill_ms:.3} reused_tokens={} prefill_tokens={rows} reused_source={} layer_slices={} layer_gpu_ms={:.3}",
                self.matched, self.reuse_source, self.commands, self.gpu_ms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_layer_cancellation_accounts_only_work_and_never_claims_completion() {
        let mut w = Work {
            armed: true,
            matched: 48000,
            prompt: 49509,
            ..Work::default()
        };
        w.note(100.0);
        // Peer decode time is deliberately not charged to prefill.
        w.note(75.0);
        assert_eq!(w.partial(), Some((175.0, 1509)));
        w.finished = true;
        assert_eq!(w.partial(), None);
    }
    #[test]
    fn rollback_and_cancel_before_execution_do_not_emit_duplicate_prefill_records() {
        let mut w = Work::default();
        w.note(100.0);
        assert!(w.partial().is_none());
        let w = Work {
            armed: true,
            ..Work::default()
        };
        assert!(w.partial().is_none());
    }
}
