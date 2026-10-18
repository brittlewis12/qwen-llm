//! Request-owned high-water estimate for this synchronous, single-token graph.
//! Not a weight budget or an OS/Metal-driver memory guarantee. Tracing callbacks
//! that retain output, caller-retained logits, shader compilation, and concurrent
//! sequences require separate admission. Metal shared storage is counted once.
use super::{Config, Forward, Result, engram, op};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestMemoryEstimate {
    pub cpu_persistent_bytes: u64,
    pub cpu_transient_bytes: u64,
    pub cpu_bytes: u64,
    pub metal_bytes: u64,
    pub total_bytes: u64,
}

fn add(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(0u64, |a, &b| {
        a.checked_add(b)
            .ok_or_else(|| "request memory overflow".into())
    })
}
fn mul(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(1u64, |a, &b| {
        a.checked_mul(b)
            .ok_or_else(|| "request memory overflow".into())
    })
}

// Vec<Vec<f32>>: 2x element and outer capacity, minimum outer capacity four,
// 24-byte Vec headers, and 64 bytes per row for allocation/alignment overhead.
// The extra outer header is charged even when there are no rows yet.
fn rows(count: u64, width: u64) -> Result<u64> {
    let capacity = if count == 0 {
        0
    } else {
        mul(&[2, count])?.max(4)
    };
    add(&[
        24,
        mul(&[capacity, 24])?,
        mul(&[count, add(&[mul(&[2, width, 4])?, 64])?])?,
    ])
}

impl Config {
    /// GPU-owned caches sized to the requested capacity; includes optional trace
    /// snapshots and conservative CPU Engram/embedding/logit transient storage.
    pub fn estimate_request_memory_with_gpu_step(
        &self,
        positions: usize,
    ) -> Result<RequestMemoryEstimate> {
        op::require(
            positions > 0 && positions as u64 <= engram::MAX_POSITION,
            "invalid GPU step capacity",
        )?;
        let mut estimate = self.estimate_request_memory(1)?;
        estimate.metal_bytes = add(&[
            estimate.metal_bytes,
            super::gpu_step::bytes(self, positions)?,
        ])?;
        estimate.total_bytes = add(&[estimate.cpu_bytes, estimate.metal_bytes])?;
        Ok(estimate)
    }

    /// Legacy conservative request estimate plus GPU FFN workspace and retained
    /// constants. Counts shared storage once, with allocation/descriptor slack.
    pub fn estimate_request_memory_with_gpu_ffn(
        &self,
        positions: usize,
    ) -> Result<RequestMemoryEstimate> {
        let mut estimate = self.estimate_request_memory(positions)?;
        let bytes = super::ffn_gpu::bytes(self)?;
        estimate.metal_bytes = add(&[estimate.metal_bytes, bytes])?;
        estimate.total_bytes = add(&[estimate.total_bytes, bytes])?;
        Ok(estimate)
    }

    pub fn estimate_request_memory_with_expert_staging(
        &self,
        positions: usize,
    ) -> Result<RequestMemoryEstimate> {
        let mut estimate = self.estimate_request_memory(positions)?;
        let bytes = super::staging::bytes(self)?;
        estimate.metal_bytes = add(&[estimate.metal_bytes, bytes])?;
        estimate.total_bytes = add(&[estimate.total_bytes, bytes])?;
        Ok(estimate)
    }

    /// Estimate peak additional memory for ONE sequence through `positions`
    /// total token forwards (prompt plus tokens actually fed back). This is NOT
    /// the configured max_context. For a continuing sequence include its already
    /// cached positions. Resident matrices/banks and bound model vectors are
    /// excluded; the caller must budget those separately.
    ///
    /// The bound sums disjoint transient phases deliberately: two window copies
    /// for rollback/attention, selected rows, growing Vec headers, sort scratch,
    /// candidate masks, projection/readback vectors, MoE slots, and head logits.
    /// Each side adds 16 MiB for small allocations, command objects and page slack.
    pub fn estimate_request_memory(&self, positions: usize) -> Result<RequestMemoryEstimate> {
        op::require(
            positions > 0 && positions as u64 <= engram::MAX_POSITION,
            "request positions outside 1..=1048576",
        )?;
        for dimension in [
            self.hidden,
            self.intermediate,
            self.vocab,
            self.heads,
            self.head_dim,
            self.query_rank,
            self.output_rank,
            self.groups,
            self.experts,
            self.topk,
            self.index_heads,
            self.index_dim,
            self.index_topk,
            self.window,
            self.candidate_block,
            self.candidate_topk,
        ] {
            op::require(dimension > 0, "zero request memory dimension")?;
        }
        op::require(
            self.heads % self.groups == 0 && self.topk <= self.experts && self.topk <= 16,
            "invalid request memory geometry",
        )?;
        let n = positions as u64;
        let h = self.hidden as u64;
        let inter = self.intermediate as u64;
        let vocab = self.vocab as u64;
        let hd = self.head_dim as u64;
        let id = self.index_dim as u64;
        let qr = self.query_rank as u64;
        let orank = self.output_rank as u64;
        let groups = self.groups as u64;
        let heads = self.heads as u64;
        let ih = self.index_heads as u64;
        let experts = self.experts as u64;
        let k = self.topk as u64;
        let window = n.min(self.window as u64);
        let selected = n.min(self.index_topk as u64);
        let query = mul(&[heads, hd])?;
        let index_query = mul(&[ih, id])?;
        let low = mul(&[groups, orank])?;
        let cpu_persistent_bytes = add(&[
            // Push occurs before window eviction; budget the spare row at every layer.
            mul(&[40, rows(add(&[window, 1])?, hd)?])?,
            mul(&[3, rows(n / 2, hd)?])?,
            mul(&[3, rows(n / 2, id)?])?,
            rows(n, hd)?,
            rows(n, id)?,
            // All four owners retain value/score outer vectors after pooling.
            mul(&[8, rows(2, hd)?])?,
            // topk truncates length but RETAINS the full n-position index capacity.
            mul(&[2, n, 8])?,
            mul(&[2, n])?,
            40 * 1024 + 4096,
        ])?;
        let cpu_transient_bytes = add(&[
            mul(&[3, rows(add(&[window, selected])?, hd)?])?,
            mul(&[4, rows(2, hd)?])?,
            // Scores, stable-sort index/scratch, candidate block sort and masks,
            // plus old outer Vec header allocations during owner-cache growth.
            mul(&[n, 160])?,
            mul(&[
                4,
                add(&[
                    mul(&[40, h])?,
                    mul(&[4, qr])?,
                    mul(&[4, query])?,
                    mul(&[4, index_query])?,
                    mul(&[4, low])?,
                    mul(&[8, experts])?,
                    mul(&[6, inter])?,
                    mul(&[4, k, inter])?,
                    mul(&[4, k, h])?,
                    mul(&[2, vocab])?,
                    4 * 6144,
                ])?,
            ])?,
            mul(&[experts, 8, 8])?,
            16 * 1024 * 1024,
        ])?;
        let dense_scratch = [
            add(&[h, vocab])?,
            add(&[6144, mul(&[5, h])?])?,
            add(&[h, qr])?,
            add(&[qr, query])?,
            add(&[h, hd])?,
            add(&[query, low])?,
            add(&[low, h])?,
            add(&[h, inter])?,
            add(&[h, experts])?,
            add(&[qr, index_query])?,
            add(&[h, ih])?,
            add(&[hd, id])?,
            add(&[mul(&[4, h])?, 24])?,
        ]
        .into_iter()
        .max()
        .unwrap();
        let metal_bytes = add(&[
            // Input, unweighted/weighted SwiGLU slots and native down slots
            // stay live while the shared expert invokes dense projections.
            mul(&[
                4,
                add(&[h, mul(&[2, k, inter])?, mul(&[k, h])?, dense_scratch])?,
            ])?,
            mul(&[4, add(&[k, 1])?])?,
            16 * 1024 * 1024,
        ])?;
        let cpu_bytes = add(&[cpu_persistent_bytes, cpu_transient_bytes])?;
        Ok(RequestMemoryEstimate {
            cpu_persistent_bytes,
            cpu_transient_bytes,
            cpu_bytes,
            metal_bytes,
            total_bytes: add(&[cpu_bytes, metal_bytes])?,
        })
    }
}

impl Forward<'_> {
    pub fn estimate_request_memory_with_gpu_step(
        &self,
        positions: usize,
    ) -> Result<RequestMemoryEstimate> {
        self.config.estimate_request_memory_with_gpu_step(positions)
    }

    /// Includes reusable active-bank storage, counted once as Metal shared bytes.
    /// Legacy estimates exclude it. No tensor-sized CPU staging allocation exists.
    pub fn estimate_request_memory_with_expert_staging(
        &self,
        positions: usize,
    ) -> Result<RequestMemoryEstimate> {
        self.config
            .estimate_request_memory_with_expert_staging(positions)
    }

    pub fn estimated_request_bytes(&self, positions: usize) -> Result<u64> {
        Ok(self.config.estimate_request_memory(positions)?.total_bytes)
    }

    pub fn estimate_request_memory(&self, positions: usize) -> Result<RequestMemoryEstimate> {
        self.config.estimate_request_memory(positions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_model_request_estimates_are_monotone_and_split_without_double_counting() {
        let config = Config::default();
        let mut previous = 0;
        for n in [1, 2, 3, 128, 4096, 8192, 1048576] {
            let e = config.estimate_request_memory(n).unwrap();
            assert!(e.total_bytes >= previous);
            assert_eq!(e.total_bytes, e.cpu_bytes + e.metal_bytes);
            assert_eq!(e.cpu_bytes, e.cpu_persistent_bytes + e.cpu_transient_bytes);
            let live_rows = 3 * (n / 2) + n;
            let logical_kv = live_rows as u64 * (512 + 128) * 4;
            assert!(e.cpu_persistent_bytes > logical_kv);
            println!(
                "positions={n} cpu={} metal={} total={} persistent={}",
                e.cpu_bytes, e.metal_bytes, e.total_bytes, e.cpu_persistent_bytes
            );
            previous = e.total_bytes;
        }
        for (n, expected) in [
            (1, 38_348_310),
            (128, 64_658_692),
            (4096, 123_225_092),
            (8192, 178_676_740),
        ] {
            assert_eq!(
                config.estimate_request_memory(n).unwrap().total_bytes,
                expected
            );
        }
    }

    #[test]
    fn grouped_buffers_bound_when_they_dominate_toy_scratch() {
        let c = Config {
            hidden: 8,
            intermediate: 8,
            vocab: 8,
            heads: 8,
            head_dim: 4096,
            query_rank: 1,
            output_rank: 1024,
            groups: 8,
            experts: 1,
            topk: 1,
            index_heads: 1,
            index_dim: 1,
            ..Config::default()
        };
        let expected = 16 * 1024 * 1024 + 4 * (8 + 2 * 8 + 8 + 8 * 4096 + 8 * 1024) + 4 * 2;
        assert_eq!(c.estimate_request_memory(1).unwrap().metal_bytes, expected);
    }

    #[test]
    fn request_estimate_checks_bounds_and_all_arithmetic() {
        let c = Config::default();
        assert!(c.estimate_request_memory(0).is_err());
        assert!(c.estimate_request_memory(1048577).is_err());
        assert!(
            Config {
                groups: 0,
                ..c.clone()
            }
            .estimate_request_memory(1)
            .is_err()
        );
        assert!(
            Config {
                hidden: usize::MAX,
                ..c.clone()
            }
            .estimate_request_memory(1)
            .is_err()
        );
        assert!(
            Config {
                head_dim: usize::MAX,
                ..c.clone()
            }
            .estimate_request_memory(1)
            .is_err()
        );
        assert!(
            Config {
                index_dim: usize::MAX,
                ..c.clone()
            }
            .estimate_request_memory(1)
            .is_err()
        );
        assert!(
            Config {
                experts: usize::MAX,
                ..c.clone()
            }
            .estimate_request_memory(1)
            .is_err()
        );
        let toy = Config {
            hidden: 32,
            intermediate: 32,
            vocab: 8,
            heads: 2,
            head_dim: 4,
            query_rank: 6,
            output_rank: 3,
            groups: 2,
            experts: 8,
            index_heads: 2,
            index_dim: 4,
            window: 3,
            ..c
        };
        assert!(toy.estimate_request_memory(3).unwrap().total_bytes > 32 * 1024 * 1024);
    }
}
