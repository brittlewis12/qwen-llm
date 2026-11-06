//! Policy shared by the bounded short-append routes. No memory-plan changes.
crate::env_flag!(
    /// Default-on M5 Fast/Half tensor dense/gate-up/down and KDA value4 routes at exactly 512 command rows, plus the existing wide Q6_K tensor tile at 512/1024 rows. Zero restores prior short-append kernels; existing enclosing precision, tier and per-stage rollbacks still apply. Exact/F32 and other devices retain their incumbent routes. No added GPU allocation.
    default_on requested,
    "QWEN_GLM53_SHORT_PREFILL"
);

pub(super) fn enabled() -> bool {
    requested()
}
