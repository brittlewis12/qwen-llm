//! Full prefill: one submitted prefix at a time, joined before the next is
//! encoded. CPU Engram reads can run while that prefix executes. No worker,
//! extra decoded-row queue, or additional live scratch/command is required.
use super::*;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLDevice;

crate::env_flag!(
    /// Full append Engram/GPU overlap on M5 Ultra for 512-2048 command rows at depth 32K+ (512 rows also at 8K+), with the 40-layer 5120/2304/384/top-6 layout and Engram consumers at layers 1/14. Enabled by default; 0 retains preparation before one GPU command for A/B or rollback. Logical chunk rows need not align to a tensor tile.
    default_on enabled,
    "QWEN_V41_ENGRAM_OVERLAP"
);

#[cfg(test)]
thread_local! {
    static OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static SUBMITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static DRAINS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn selected(model: &Forward<'_>, start: usize, rows: usize) -> bool {
    #[cfg(test)]
    if let Some(value) = OVERRIDE.with(|x| x.get()) {
        return value;
    }
    let c = &model.config;
    ((rows == 512 && start >= 8192) || ((513..=2048).contains(&rows) && start >= 32768))
        && [
            c.hidden,
            c.intermediate,
            c.experts,
            c.topk,
            model.layers.len(),
        ] == [5120, 2304, 384, 6, 40]
        && model.engram_hash.layers.iter().map(|x| x.layer).eq([1, 14])
        && model.ctx.device.name().to_string() == "Apple M5 Ultra"
        && enabled()
}

/// Dropping on an IO, encode, cancellation or panic path must retire submitted
/// work before the caller releases chunk scratch or returns the process lease.
/// A successful join consumes the command before another encoder is created.
pub(super) struct Submitted {
    command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    // [request position, segment layers, rows, first layer].
    dimensions: [usize; 4],
    encode_ms: Option<f64>,
}

impl Submitted {
    pub(super) fn commit(
        command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        dimensions: [usize; 4],
        started: Option<std::time::Instant>,
    ) -> Self {
        let encode_ms = started.map(|s| s.elapsed().as_secs_f64() * 1000.0);
        command.commit();
        #[cfg(test)]
        SUBMITS.with(|x| x.set(x.get() + 1));
        Self {
            command: Some(command),
            dimensions,
            encode_ms,
        }
    }

    pub(super) fn wait(mut self) -> Result<()> {
        self.retire()
    }

    fn retire(&mut self) -> Result<()> {
        let Some(command) = self.command.take() else {
            return Ok(());
        };
        let started = std::time::Instant::now();
        let result = metal::wait_completed(&command).map_err(|e| e.to_string());
        #[cfg(test)]
        DRAINS.with(|x| x.set(x.get() + 1));
        if let Some(encode_ms) = self.encode_ms {
            profile::submitted_with_driver_scheduling(
                "prefill_chunk_rows",
                self.dimensions,
                encode_ms,
                started.elapsed().as_secs_f64() * 1000.0,
                result.is_ok(),
                command.GPUStartTime(),
                command.GPUEndTime(),
                || (command.kernelStartTime(), command.kernelEndTime()),
            );
        }
        result
    }
}

impl Drop for Submitted {
    fn drop(&mut self) {
        let _ = self.retire();
    }
}

#[cfg(test)]
mod tests;
