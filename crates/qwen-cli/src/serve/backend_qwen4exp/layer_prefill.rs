//! Selective layer scheduling over the native packed row plan.
use anyhow::Context as _;

/// Nominal GPU milliseconds per Flash-Next layer slice with ready peers (default 0, opt-in 200). Zero restores the original native commands and decode-percentage policy.
const MS: &str = "QWEN_SERVE_FN_LAYER_SLICE_MS";
/// Slice only Flash-Next appends whose predicted original prefill wall exceeds this many milliseconds (default 500).
const MIN_MS: &str = "QWEN_SERVE_FN_LAYER_SLICE_MIN_MS";
pub(super) const LAYERS: usize = 48;

#[derive(Clone, Copy)]
pub(super) struct Policy {
    pub ms: usize,
    minimum: usize,
}
fn parse(key: &str, value: Option<&str>, default: usize) -> anyhow::Result<usize> {
    let n = value
        .map_or(Ok(default), str::parse::<usize>)
        .with_context(|| format!("{key} must be 0..=10000"))?;
    anyhow::ensure!(n <= 10000, "{key} must be 0..=10000");
    Ok(n)
}
impl Policy {
    pub fn for_layers(self, layers: usize) -> Self {
        Self {
            ms: if layers == LAYERS { self.ms } else { 0 },
            ..self
        }
    }
    pub fn from_env() -> anyhow::Result<Self> {
        Ok(Self {
            ms: parse(MS, std::env::var(MS).ok().as_deref(), 0)?,
            minimum: parse(MIN_MS, std::env::var(MIN_MS).ok().as_deref(), 500)?,
        })
    }
    pub fn selects(&self, commands: &std::collections::VecDeque<std::ops::Range<usize>>) -> bool {
        self.ms > 0
            && commands.iter().map(|c| predicted_ms(c.len())).sum::<f64>() > self.minimum as f64
    }
    pub fn end(&self, rows: usize, next: usize, peers: bool) -> usize {
        if !peers {
            return LAYERS;
        }
        // Preserve each original row command and the fused first two layers.
        // Quantize to the 1/2/4/8 partitions covered by the native regression.
        let slices = ((predicted_ms(rows) / self.ms.max(1) as f64).ceil() as usize)
            .clamp(1, 8)
            .next_power_of_two();
        (next + LAYERS / slices).min(LAYERS)
    }
}
fn predicted_ms(rows: usize) -> f64 {
    // Scalar commands already yield at a token. They cannot benefit from a
    // layer cut and must not pay a packed weight-streaming floor in admission.
    if rows < 2 {
        0.0
    } else {
        225.0 + 0.42 * rows as f64
    }
}

#[derive(Default)]
pub(super) struct Work {
    pub armed: bool,
    pub finished: bool,
    pub next: usize,
    pub elapsed_ms: f64,
    pub gpu_ms: f64,
    pub commands: usize,
    pub matched: usize,
    pub prompt: usize,
    pub reuse_source: &'static str,
}
impl Work {
    pub fn note(&mut self, ms: f64) {
        self.elapsed_ms += ms.max(0.0);
    }
    fn partial(&self) -> bool {
        !self.finished && self.elapsed_ms > 0.0
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        if self.partial() {
            tracing::info!(target:"qwen_diag", "serve prefill: family=qwen4exp complete=false cancelled=true phase=layer prefill_ms={:.3} reused_tokens={} prefill_tokens={} reused_source={} layer_slices={} layer_gpu_ms={:.3}", self.elapsed_ms,self.matched,self.prompt.saturating_sub(self.matched),self.reuse_source,self.commands,self.gpu_ms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rollback_and_selective_policy_preserve_small_appends() {
        assert_eq!(parse(MS, None, 0).unwrap(), 0);
        for value in ["-1", "10001", "bad", ""] {
            assert!(parse(MS, Some(value), 0).is_err());
        }
        let off = Policy {
            ms: 0,
            minimum: 500,
        };
        let on = Policy {
            ms: 200,
            minimum: 500,
        };
        assert_eq!(on.for_layers(48).ms, 200);
        for layers in [2, 12, 47, 49, 96] {
            assert_eq!(on.for_layers(layers).ms, 0);
        }
        for rows in [1, 32, 342] {
            let commands = std::collections::VecDeque::from([49152..49152 + rows]);
            assert!(!on.selects(&commands));
        }
        for rows in [1383, 1798, 2048] {
            let commands = std::collections::VecDeque::from([49152..49152 + rows]);
            assert!(on.selects(&commands));
            assert!(!off.selects(&commands));
        }
        let scalar_only = (2048..2080).map(|p| p..p + 1).collect();
        assert!(!on.selects(&scalar_only));
    }
    #[test]
    fn cold_and_warm_slices_preserve_the_row_plan_and_fused_ple_boundary() {
        for target in [1, 100, 200, 10000] {
            let p = Policy {
                ms: target,
                minimum: 500,
            };
            for rows in [2, 32, 342, 1383, 2048] {
                let mut next = 0;
                let mut cuts = Vec::new();
                while next < LAYERS {
                    let end = p.end(rows, next, true);
                    assert!(end > next && end >= 2 && end <= LAYERS);
                    cuts.push(end);
                    next = end;
                }
                assert!([1, 2, 4, 8].contains(&cuts.len()));
                assert_eq!(p.end(rows, 6, false), LAYERS);
            }
        }
    }
    #[test]
    fn cancellation_accounts_only_active_work_until_completion() {
        let p = Policy {
            ms: 200,
            minimum: 500,
        };
        let mut work = Work {
            armed: true,
            matched: 49152,
            prompt: 51200,
            ..Work::default()
        };
        work.next = p.end(2048, 0, true);
        work.note(150.0);
        // A peer round consumes wall time, but does not call note or commit
        // the appender. Cancellation disposes this work and its private state.
        assert!(work.next < LAYERS && work.partial());
        assert_eq!(work.elapsed_ms, 150.0);
        work.finished = true;
        assert!(!work.partial());
        let empty = Work::default();
        assert!(!empty.partial());
    }
}
