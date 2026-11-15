//! V4.1 Half index scorer: exact K128/head arithmetic, shared key panels.
use super::checks::{bad_shape, check_disjoint, check_tensor, require_serial};
use super::*;
crate::env_flag!(
    /// Default-off M5 V4.1 Half index query reuse. Shares key panels across queries with incumbent K128/head arithmetic and reuses selection-pair scratch. Zero restores the incumbent; F32 MID/short/compact paths are unchanged.
    default_off v41_root_index_reuse_env,
    "QWEN_V41_ROOT_INDEX_REUSE"
);
#[cfg(any(test, feature = "perf-ablations"))]
thread_local! {static OVERRIDE:std::cell::Cell<Option<bool>>=const{std::cell::Cell::new(None)};}
#[cfg(any(test, feature = "perf-ablations"))]
pub fn with_v41_root_index_reuse<R>(on: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            OVERRIDE.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(OVERRIDE.with(|s| s.replace(Some(on))));
    f()
}
fn requested() -> bool {
    #[cfg(any(test, feature = "perf-ablations"))]
    if let Some(on) = OVERRIDE.with(|s| s.get()) {
        return on;
    }
    v41_root_index_reuse_env()
}
#[cfg(feature = "perf-ablations")]
static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "perf-ablations")]
pub fn take_v41_root_index_reuse_calls() -> u64 {
    CALLS.swap(0, std::sync::atomic::Ordering::Relaxed)
}
/// Production passes this only on the existing Half tensor branch. `scratch`
/// aliases the admitted selection-pair slab: scoring and selection are serial
/// and have disjoint lifetimes. No allocation or session-memory increase.
pub struct V41RootIndexReuse<'a> {
    pub query: &'a MetalTensor,
    pub weights: &'a MetalTensor,
    pub keys: &'a MetalTensor,
    pub scores: &'a MetalTensor,
    pub scratch: &'a MetalTensor,
}
/// Returns false before encoding for disabled/unsupported calls. Explicit
/// group sizes are useful for perf-only comparisons; production uses four.
pub fn try_encode_v41_root_index_reuse(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    b: &V41RootIndexReuse<'_>,
    start: usize,
    ratio: usize,
    first: usize,
    n: usize,
    stride: usize,
    group: usize,
) -> Result<bool, MetalError> {
    if !requested()
        || !ctx.device_facts().gpu_families.apple10
        || !ctx.tensor_ops_status().is_available()
    {
        return Ok(false);
    }
    if !geometry(start, ratio, first, n, stride, group) {
        return Ok(false);
    }
    const K: &str = "v41_root_index_reuse";
    require_serial(K, enc)?;
    let count = n * 4096;
    if b.scratch.dtype != GgmlType::I32 || b.scratch.n_bytes() < (count * 2) as u64 {
        return Ok(false);
    }
    let end = first + n;
    let maxkeys = (start + end) / ratio;
    for (t, ty, name, needed) in [
        (b.query, GgmlType::F32, "query", end * 4096),
        (b.weights, GgmlType::F32, "weights", end * 32),
        (b.keys, GgmlType::F32, "keys", maxkeys * 128),
        (b.scores, GgmlType::F32, "scores", n * stride),
    ] {
        if t.n_elements() < (needed as u64) {
            return Err(bad_shape(K, format!("{name} extent")));
        }
        check_tensor(K, t, ty, &t.shape, std::ptr::eq(t, b.scores), name)?;
    }
    check_tensor(
        K,
        b.scratch,
        GgmlType::I32,
        &b.scratch.shape,
        true,
        "scratch",
    )?;
    check_disjoint(
        K,
        b.scores,
        &[
            (b.query, "query"),
            (b.weights, "weights"),
            (b.keys, "keys"),
            (b.scratch, "scratch"),
        ],
    )?;
    check_disjoint(
        K,
        b.scratch,
        &[(b.query, "query"), (b.weights, "weights"), (b.keys, "keys")],
    )?;
    let pso = match ctx.tensor_pipeline("kernel_deepseek_v41_attention_index_scores_reuse_serial") {
        Ok(p) => p,
        Err(MetalError::NoFunction(_)) => return Ok(false),
        Err(e) => return Err(e),
    };
    if pso.threadExecutionWidth() != 32
        || pso.maxTotalThreadsPerThreadgroup() < 128
        || ctx.device.maxThreadgroupMemoryLength() < 32768
    {
        return Ok(false);
    }
    let q = b
        .query
        .view_subrange((first * 4096) as u64, vec![count as u64]);
    let w = b
        .weights
        .view_subrange((first * 32) as u64, vec![(n * 32) as u64]);
    let mut half = b.scratch.view_subrange(0, vec![(count / 2) as u64]);
    half.dtype = GgmlType::F16;
    half.shape = vec![count as u64];
    encode_scatter_offset_f32_to_f16(ctx, enc, &q, &half, 0, count)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &[(start + first) as u32, ratio as u32, 512u32, maxkeys as u32],
    );
    enc.set_bytes(1, &[128u32, 32, 4096, 32]);
    enc.set_bytes(2, &[stride as u32, n as u32, group as u32, 0]);
    for (i, t) in [(3, &half), (4, &w), (5, b.keys), (6, b.scores)] {
        enc.set_tensor(i, t)
    }
    enc.dispatch(
        MTLSize {
            width: 1,
            height: n.div_ceil(group),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    #[cfg(feature = "perf-ablations")]
    CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(true)
}
fn geometry(
    start: usize,
    ratio: usize,
    first: usize,
    n: usize,
    stride: usize,
    group: usize,
) -> bool {
    ratio > 0
        && n > 0
        && n <= 512
        && matches!(group, 2 | 4 | 8 | 16)
        && start
            .checked_add(first)
            .and_then(|s| s.checked_add(n))
            .is_some_and(|end| end <= u32::MAX as usize && end / ratio <= stride)
        && first
            .checked_add(n)
            .and_then(|end| end.checked_mul(4096))
            .is_some_and(|v| v <= u32::MAX as usize)
        && stride
            .checked_mul(n)
            .is_some_and(|v| v <= u32::MAX as usize)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_bad_bounds_and_restores_scoped_controls() {
        assert!(!geometry(usize::MAX, 2, 0, 512, 16384, 4));
        assert!(!geometry(0, 0, 0, 512, 16384, 4));
        assert!(!geometry(0, 2, 0, 513, 16384, 4));
        assert!(!geometry(32768, 2, 0, 512, 16384, 4));
        assert!(geometry(28672, 2, 3584, 512, 16384, 4));
        with_v41_root_index_reuse(false, || {
            assert!(!requested());
            with_v41_root_index_reuse(true, || assert!(requested()));
            assert!(!requested());
        });
    }
}

/// Perf witness using the incumbent bindings and grid verbatim.
#[cfg(feature = "perf-ablations")]
pub fn encode_v41_root_index_incumbent(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    b: &V41RootIndexReuse<'_>,
    start: usize,
    ratio: usize,
    first: usize,
    n: usize,
    stride: usize,
) -> Result<(), MetalError> {
    let p = ctx.tensor_pipeline("kernel_deepseek_v41_attention_index_scores_tensor")?;
    enc.set_pipeline(&p);
    enc.set_bytes(
        0,
        &[
            start as u32,
            ratio as u32,
            512u32,
            ((start + first + n) / ratio) as u32,
        ],
    );
    enc.set_bytes(1, &[128u32, 32, 4096, 32]);
    enc.set_bytes(2, &[stride as u32, n as u32, 0, 0]);
    for (i, t) in [(3, b.query), (4, b.weights), (5, b.keys), (6, b.scores)] {
        enc.set_tensor(i, t)
    }
    enc.set_bytes(7, &(first as u32));
    enc.dispatch(
        MTLSize {
            width: 1,
            height: n,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
