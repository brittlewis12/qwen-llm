//! Policy shared by the bounded short-append routes. No memory-plan changes.
crate::env_flag!(
    /// Default-on M5 Fast/Half tensor dense/gate-up/down and KDA value4 routes at exactly 512 command rows. Zero restores the prior short-append kernels; existing enclosing precision, tier and per-stage rollbacks still apply. No added GPU allocation.
    default_on requested,
    "QWEN_GLM53_SHORT_PREFILL"
);

pub(super) fn enabled() -> bool {
    requested()
}
