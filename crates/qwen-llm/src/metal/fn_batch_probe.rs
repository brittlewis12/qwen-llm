//! Research-only whole-round controls. Held stages intentionally invalidate outputs.
use std::{cell::RefCell, collections::BTreeMap};
#[derive(Clone, Default)]
struct Settings {
    other: Option<bool>,

    group: String,
    skipped: BTreeMap<String, usize>,
    observed: BTreeMap<String, usize>,
    causal: bool,
}
thread_local! { static CANDIDATE: RefCell<String> = RefCell::new("formats".into()); }
thread_local! { static ACTIVE: RefCell<Settings> = RefCell::new(Settings::default()); }
pub struct Scope(Settings);
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|a| *a.borrow_mut() = self.0.clone());
    }
}
pub fn scope(other: Option<bool>, group: &str, causal: bool) -> Scope {
    Scope(ACTIVE.with(|a| {
        a.replace(Settings {
            other,
            group: group.into(),
            causal,
            ..Default::default()
        })
    }))
}
pub fn skipped() -> BTreeMap<String, usize> {
    ACTIVE.with(|a| a.borrow().skipped.clone())
}
pub(crate) fn observe(name: &str, count: usize) {
    ACTIVE.with(|a| *a.borrow_mut().observed.entry(name.into()).or_default() += count);
}
pub fn observed(name: &str) -> usize {
    ACTIVE.with(|a| a.borrow().observed.get(name).copied().unwrap_or(0))
}
pub fn candidate(name: &str) {
    CANDIDATE.with(|v| *v.borrow_mut() = name.into());
}
pub(crate) fn axis(name: &str, default: bool) -> bool {
    if CANDIDATE.with(|v| v.borrow().as_str() == name) {
        ACTIVE.with(|a| a.borrow().other.unwrap_or(default))
    } else {
        default
    }
}
pub(crate) fn other(default: bool) -> bool {
    axis("formats", default)
}
pub(crate) fn causal() -> bool {
    ACTIVE.with(|a| a.borrow().causal)
}
pub(crate) fn skip(name: &str) -> bool {
    ACTIVE.with(|a| {
        let mut a = a.borrow_mut();
        let yes = match a.group.as_str() {
            "hc_inject" => name == "kernel_qwen4exp_hc_inject_f32",
            "gdn_prep" => name.contains("gdn_controls"),
            "qsa_norm" => {
                name == "kernel_qwen4exp_qsa_norm_rope_f32"
                    || name == "kernel_qwen4exp_qsa_qgate_norm_rope_f32"
            }
            "qsa_publish" => {
                name.contains("qsa_write_pending")
                    || name.contains("qsa_pool_publish")
                    || name.contains("qsa_publish_kv")
            }
            "qsa_prepare" => {
                name.contains("qsa_norm_rope")
                    || name.contains("qsa_qgate_norm_rope")
                    || name.contains("qsa_write_pending")
                    || name.contains("qsa_pool_publish")
                    || name.contains("qsa_publish_kv")
            }
            "copies" => name.contains("copy_offset") || name.contains("copy_f32"),
            "hc" => name.contains("hc_"),
            "gdn" => name.contains("gdn") || name.contains("gated_delta"),
            "qsa" => name.contains("qsa") || name.contains("qwen_sparse"),
            "routing" => name.contains("topk") || name.contains("router"),
            "experts" => name.contains("expert_native"),
            "transport" => {
                name.contains("fn_batch_gather_columns")
                    || name.contains("fn_batch_scatter_columns")
            }
            "norms" => {
                name.contains("rms_norm") || name.contains("l2_norm") || name.contains("rope")
            }
            // B=1 family holds shared by Flash-Next and GLM solo probes.
            "dense" => {
                name.contains("mat_vec_q6_K")
                    || name.contains("mat_vec_q5_K")
                    || name.contains("mat_vec_q4_K")
                    || name.contains("mat_vec_q8_0")
                    || name.contains("mat_vec_iq")
                    || name.contains("mat_vec_bf16")
                    || name.contains("mat_vec_f32_f32_lcpp")
            }
            "head" => name.contains("mat_vec_q6_K_f32") || name.contains("lm_head"),
            "attention" => {
                name.contains("qsa_")
                    || name.contains("select_top_k")
                    || name.contains("expand_ids")
                    || name.contains("selected_attention")
                    || name.contains("latent_attention")
                    || name.contains("attention")
            }
            // QSA compute only; leaves norm/publish/selection status intact.
            "qsa_core" => {
                name.contains("qsa_split")
                    || name.contains("qsa_decode_logits")
                    || name.contains("qsa_expand_ids")
            }
            "routed" => {
                name.contains("expert_native")
                    || name.contains("qwen4exp_expert_compat")
                    || name.contains("moe_swiglu")
                    || name.contains("moe_down")
                    || name.contains("glm_batch_gate_up")
                    || name.contains("glm_batch_down")
                    || name.contains("weighted_sum")
                    || name.contains("finalizer")
            }
            "q6" => name.contains("q6_K") || name.contains("q6_k"),
            "q8" => name.contains("q8_0") || name.contains("q8_"),
            "index" => {
                name.contains("indexer")
                    || name.contains("index_score")
                    || name.contains("select_top_k")
                    || name.contains("expand_selection")
                    || name.contains("scatter_offset")
            }
            "kda" => name.contains("kda"),
            "hc" => name.contains("hc_") || name.contains("mhc"),
            "router" => name.contains("route_learned"),
            "pointwise" => {
                name.contains("rms_norm")
                    || name.contains("l2_norm")
                    || name.contains("swiglu")
                    || name.contains("silu")
                    || name.contains("argmax")
                    || name.contains("copy")
                    || name.contains("fill")
                    || name.contains("add")
                    || name.contains("scale")
                    || name.contains("rope")
            }
            _ => false,
        };
        if yes {
            *a.skipped.entry(name.into()).or_default() += 1;
        }
        yes
    })
}
