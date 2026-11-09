//! Independent-batch projection geometry and MLA scratch lifetime policy.
use super::*;

crate::env_flag!(
    /// Default-on M5 Fast/Half independent-batch Q6 tiles, cooperative F32 routers and direct packed MLA views. Zero restores the previous projection/copy policies. Single-token, Exact/F32 and prefill policies are unchanged.
    default_on enabled,
    "QWEN_GLM53_BATCH_PROJECTIONS"
);

pub(super) const Q6: u8 = 1;
pub(super) const ROUTER: u8 = 2;
pub(super) const MLA: u8 = 4;
pub(super) const SELECT: u8 = 8;
pub(super) const KDA: u8 = 16;
crate::env_flag!(
    /// Default-on M5 Fast/Half independent decode KDA grid over two to four members, with separate state buffers and unchanged arithmetic. Zero restores per-member dispatches; ordinary decode, prefill and Exact/F32 policies are unchanged.
    default_on kda_enabled,
    "QWEN_GLM53_BATCH_KDA"
);
#[cfg(test)]
thread_local! { static OVERRIDE:std::cell::Cell<Option<u8>>=const {std::cell::Cell::new(None)}; }
pub(super) fn features() -> u8 {
    #[cfg(test)]
    if let Some(v) = OVERRIDE.with(|o| o.get()) {
        return v;
    }
    (if kda_enabled() { KDA } else { 0 })
        | (if enabled() { Q6 | ROUTER | MLA } else { 0 })
        | (if super::selection::requested() {
            SELECT
        } else {
            0
        })
}
#[cfg(test)]
pub(super) fn with_features<T>(v: u8, f: impl FnOnce() -> T) -> T {
    assert_eq!(v & !31, 0);
    struct Reset(Option<u8>);
    impl Drop for Reset {
        fn drop(&mut self) {
            OVERRIDE.with(|o| o.set(self.0));
        }
    }
    let _reset = Reset(OVERRIDE.with(|o| o.replace(Some(v))));
    f()
}

/// Keep the incumbent K split, changing only independent output tiles.
pub(super) fn q6_tile(k: usize, m: usize, rows: usize) -> Option<(usize, usize)> {
    if !(2..=4).contains(&rows) {
        return None;
    }
    match (k, m) {
        (2048 | 8192 | 16384, 4096) => Some((1, 16)),
        (4096, 8192) => Some((2, 8)),
        _ => None,
    }
}
#[allow(clippy::too_many_arguments)]
pub(super) fn matmat(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    rows: usize,
) -> Result<bool> {
    let f = features();
    if f & Q6 != 0 && w.dtype == GgmlType::Q6_K {
        if let Some((nt, nsg)) = q6_tile(k, m, rows) {
            crate::metal::encode_glm_batch_q6_tile(ctx, enc, w, x, y, k, m, rows, nt, nsg)?;
            return Ok(true);
        }
    }
    if f & ROUTER != 0
        && w.dtype == GgmlType::F32
        && (k, m) == (4096, 288)
        && (2..=4).contains(&rows)
        && crate::metal::mat_vec_f32_cooperative_enabled()
    {
        return Ok(crate::metal::try_encode_mat_vec_f32_members(
            ctx, enc, w, x, y, k, m, rows,
        )?);
    }
    Ok(false)
}

#[cfg(test)]
thread_local! {static TIMING: std::cell::Cell<(f64,f64)>=const {std::cell::Cell::new((0.0,0.0))};}
#[cfg(test)]
pub(super) fn record_timing(encode_ms: f64, gpu_ms: f64) {
    TIMING.with(|v| v.set((encode_ms, gpu_ms)));
}
#[cfg(test)]
pub(super) fn timing() -> (f64, f64) {
    TIMING.with(|v| v.get())
}

crate::env_flag!(
    /// Default-on causal dependency waves for M5 Fast/Half independent decode batches. No arithmetic fusion; zero restores serial dispatch ordering. Singleton, prefill and Exact/F32 routes are unchanged.
    default_on waves_requested,
    "QWEN_GLM53_BATCH_WAVES"
);
#[cfg(feature = "perf-ablations")]
thread_local! {static WAVES:std::cell::Cell<Option<bool>>=const{std::cell::Cell::new(None)};}
pub(super) fn waves() -> bool {
    #[cfg(feature = "perf-ablations")]
    if let Some(v) = WAVES.with(|c| c.get()) {
        return v;
    }
    waves_requested()
}
#[cfg(feature = "perf-ablations")]
pub(super) fn with_batch_waves<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    struct Reset(Option<bool>);
    impl Drop for Reset {
        fn drop(&mut self) {
            WAVES.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(WAVES.with(|c| c.replace(Some(enabled))));
    f()
}
