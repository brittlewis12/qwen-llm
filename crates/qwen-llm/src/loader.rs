//! Model loader: bind every Qwen3.5 weight tensor by its canonical
//! llama.cpp GGUF name and validate the shape against the [`Arch`].
//!
//! Tensor-name conventions (from `~/code/llama.cpp/src/models/qwen35.cpp`
//! and `convert_hf_to_gguf.py`'s `_LinearAttentionVReorderBase`):
//!
//! * `token_embd.weight`  : embedding matrix
//! * `output_norm.weight` : final pre-`lm_head` RMSNorm
//! * `output.weight`      : `lm_head` (untied; `tie_word_embeddings=false`)
//!
//! Per-layer:
//!
//! * `blk.{i}.attn_norm.weight`             : pre-mixer RMSNorm  (both kinds)
//! * `blk.{i}.post_attention_norm.weight`   : pre-FFN RMSNorm    (both kinds)
//! * `blk.{i}.ffn_{gate,up,down}.weight`    : SwiGLU FFN         (both kinds)
//!
//! GDN-only:
//!
//! * `blk.{i}.attn_qkv.weight`     : combined Q/K/V projection
//! * `blk.{i}.attn_gate.weight`    : `in_proj_z` (output gate)
//! * `blk.{i}.ssm_beta.weight`     : β-projection — sigmoid'd into per-token β
//!                                   (this is named `ssm_beta` in GGUF and `wq_beta`-like
//!                                   in HF — `beta = sigmoid(ssm_beta @ x)`)
//! * `blk.{i}.ssm_alpha.weight`    : α-projection — `softplus(ssm_alpha @ x + ssm_dt) * (-A_log.exp())`
//!                                   produces the gate
//! * `blk.{i}.ssm_a`               : already-`-A_log.exp()`-baked at convert time
//!                                   (multiplied with softplus(α + dt) to make `g`)
//! * `blk.{i}.ssm_dt.bias`         : per-head time-step bias
//! * `blk.{i}.ssm_conv1d.weight`   : depthwise conv1d kernel
//! * `blk.{i}.ssm_norm.weight`     : RMSNormGated norm weight
//! * `blk.{i}.ssm_out.weight`      : `out_proj` (back to hidden_size)
//!
//! Reference: `~/code/llama.cpp/src/llama-model.cpp` case `LLM_ARCH_QWEN35`
//! (tensor creation) and `~/code/llama.cpp/src/models/qwen35.cpp`
//! `build_layer_attn_linear` (forward).
//!
//! Full-attn-only:
//!
//! * `blk.{i}.attn_q.weight`       : Q proj (note: Q has the gated-attention
//!                                   trick — output is 2× hidden, second half
//!                                   is the sigmoid output gate)
//! * `blk.{i}.attn_k.weight`, `attn_v.weight`, `attn_output.weight`
//! * `blk.{i}.attn_q_norm.weight`, `attn_k_norm.weight` : per-head RMSNorms
//!
//! Shapes follow GGUF convention: matrix `M[in, out]` is stored as
//! `shape=[in, out]` and applied as `y = x @ W` with `W: [in, out]`.

use crate::gguf::GgufFile;
use crate::model::{Arch, ArchKind, LayerKind};
use crate::tensor::{GgmlType, TensorDesc};
use serde_json::Value;
use std::collections::BTreeMap;

/// GGUF metadata prefix used by Prism/Bonsai rotated-basis artifacts.
pub const PRISM_HADAMARD_METADATA_PREFIX: &str = "prism.hadamard.";

pub(crate) fn prism_hadamard_metadata_key_in(metadata: &BTreeMap<String, Value>) -> Option<&str> {
    metadata
        .keys()
        .find(|key| key.starts_with(PRISM_HADAMARD_METADATA_PREFIX))
        .map(String::as_str)
}

/// Return the first Prism rotated-basis metadata key, if this GGUF declares
/// the execution contract that the ordinary Qwen runtime does not implement.
pub fn prism_hadamard_metadata_key(gguf: &GgufFile) -> Option<&str> {
    prism_hadamard_metadata_key_in(gguf.model.metadata())
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("missing required tensor: {0}")]
    Missing(String),
    #[error("shape mismatch for {name}: expected {expected:?}, got {got:?}")]
    Shape {
        name: String,
        expected: Vec<u64>,
        got: Vec<u64>,
    },
    #[error("architecture not supported: {0:?}")]
    UnsupportedArch(Option<String>),
    #[error(
        "DeepSeek V4 does not bind through the Qwen model runtime; use the native DeepSeek V4 execution path"
    )]
    DeepSeek4RequiresNativeRuntime,
    #[error(
        "metadata key {key:?} declares a Prism rotated-basis execution contract that is not implemented"
    )]
    PrismBasisUnsupported { key: String },
    #[error("metadata key {0:?} missing or wrong type")]
    BadMetadata(&'static str),
    #[error("metadata says {key:?} = {got}, but Arch expects {expected}")]
    ArchMismatch {
        key: &'static str,
        got: u64,
        expected: u64,
    },
    #[error("unexpected layer kind for blk.{idx}: expected {expected:?} based on pattern")]
    LayerKind { idx: u32, expected: LayerKind },
    #[error("metadata key {key:?} = {got} is out of range (must fit in u32)")]
    MetadataOverflow { key: &'static str, got: u64 },
    #[error("metadata key {key:?} = {got} is out of range (must fit in i32)")]
    MetadataI32Overflow { key: &'static str, got: u64 },
    #[error("metadata key {key:?} = {got} exceeds safety cap {cap}")]
    MetadataCap {
        key: &'static str,
        got: u64,
        cap: u64,
    },
    #[error("metadata key {key:?} = {got} is inconsistent: must be <= {bound_key:?} = {bound}")]
    MetadataInconsistent {
        key: &'static str,
        got: u64,
        bound_key: &'static str,
        bound: u64,
    },
    #[error("metadata-derived dimension overflows u64: {0}")]
    DimensionOverflow(&'static str),
    #[error("metadata key {key:?} = {numerator} is not divisible by {divisor_key:?} = {divisor}")]
    MetadataNotDivisible {
        key: &'static str,
        numerator: u64,
        divisor_key: &'static str,
        divisor: u64,
    },
}

const MAX_ARCH_LAYERS: u32 = 4096;
const MAX_ARCH_DIM: u32 = 1 << 20;
const MAX_DFLASH_BLOCK_SIZE: u32 = 4096;
const MAX_KERNEL_DIM: u64 = 1 << 20;
const MAX_GDN_STATE_ELEMS: u64 = 1 << 24;

/// Strict narrowing of GGUF u64 metadata to u32. Returns a typed error
/// instead of silently truncating, which `as u32` would do on a malformed
/// GGUF claiming e.g. `block_count = u64::MAX`.
fn u64_to_u32(key: &'static str, got: u64) -> Result<u32, LoadError> {
    u32::try_from(got).map_err(|_| LoadError::MetadataOverflow { key, got })
}

fn u64_to_i32(key: &'static str, got: u64) -> Result<i32, LoadError> {
    i32::try_from(got).map_err(|_| LoadError::MetadataI32Overflow { key, got })
}

fn ensure_cap(key: &'static str, got: u32, cap: u32) -> Result<(), LoadError> {
    if got > cap {
        return Err(LoadError::MetadataCap {
            key,
            got: got as u64,
            cap: cap as u64,
        });
    }
    Ok(())
}

fn ensure_nonzero(key: &'static str, got: u32) -> Result<(), LoadError> {
    if got == 0 {
        return Err(LoadError::BadMetadata(key));
    }
    Ok(())
}

fn ensure_divisible(
    key: &'static str,
    numerator: u32,
    divisor_key: &'static str,
    divisor: u32,
) -> Result<(), LoadError> {
    if !numerator.is_multiple_of(divisor) {
        return Err(LoadError::MetadataNotDivisible {
            key,
            numerator: numerator as u64,
            divisor_key,
            divisor: divisor as u64,
        });
    }
    Ok(())
}

fn checked_mul_dim(a: u64, b: u64, label: &'static str) -> Result<u64, LoadError> {
    a.checked_mul(b).ok_or(LoadError::DimensionOverflow(label))
}

fn checked_add_dim(a: u64, b: u64, label: &'static str) -> Result<u64, LoadError> {
    a.checked_add(b).ok_or(LoadError::DimensionOverflow(label))
}

fn checked_double_dim(a: u64, label: &'static str) -> Result<u64, LoadError> {
    checked_mul_dim(a, 2, label)
}

/// Per-block weight references for a GDN layer.
///
/// Field names follow the *forward-pass role*, not the GGUF tensor name —
/// the mapping is documented at the module level. In particular, the
/// β-source projection is the GGUF tensor `ssm_beta.weight` and the
/// α-source (which feeds softplus → multiplies with A_log → forms the
/// gate `g`) is `ssm_alpha.weight`. These are easy to invert; double-check
/// against `~/code/llama.cpp/src/models/qwen35.cpp` if in doubt.
#[derive(Clone)]
pub struct GdnBlock<'a> {
    // shared with attn-block layout
    pub attn_norm: &'a TensorDesc,
    pub post_attention_norm: &'a TensorDesc,
    pub ffn_gate: &'a TensorDesc,
    pub ffn_up: &'a TensorDesc,
    pub ffn_down: &'a TensorDesc,
    // GDN-specific (GGUF name → forward-pass role)
    pub in_proj_qkv: &'a TensorDesc, // attn_qkv         — combined QKV input projection
    pub in_proj_z: &'a TensorDesc,   // attn_gate        — z (gate) input projection
    pub beta_proj: &'a TensorDesc,   // ssm_beta.weight  — β = sigmoid(beta_proj @ x)
    pub alpha_proj: &'a TensorDesc,  // ssm_alpha.weight — softplus(alpha_proj @ x + dt) * a_log
    pub a_log: &'a TensorDesc,       // ssm_a            — already negated/exp'd at convert
    pub dt_bias: &'a TensorDesc,     // ssm_dt.bias
    pub conv1d: &'a TensorDesc,      // ssm_conv1d.weight
    pub norm: &'a TensorDesc,        // ssm_norm.weight  — RMSNorm before silu(z) gate
    pub out_proj: &'a TensorDesc,    // ssm_out.weight   — back to hidden_size
    /// Present on qwen35moe variants. When `Some`, `ffn_gate/up/down` refer
    /// to the shared-expert branch and routed experts live in `ffn_moe`.
    pub ffn_moe: Option<MoeFfn<'a>>,
}

/// Per-block weight references for a full-attention layer.
#[derive(Clone)]
pub struct AttnBlock<'a> {
    pub attn_norm: &'a TensorDesc,
    pub post_attention_norm: &'a TensorDesc,
    pub ffn_gate: &'a TensorDesc,
    pub ffn_up: &'a TensorDesc,
    pub ffn_down: &'a TensorDesc,
    pub q: &'a TensorDesc, // attn_q.weight (output is 2× hidden — gated)
    pub k: &'a TensorDesc,
    pub v: &'a TensorDesc,
    pub o: &'a TensorDesc, // attn_output.weight
    pub q_norm: &'a TensorDesc,
    pub k_norm: &'a TensorDesc,
    /// Present on qwen35moe variants. When `Some`, `ffn_gate/up/down` refer
    /// to the shared-expert branch and routed experts live in `ffn_moe`.
    pub ffn_moe: Option<MoeFfn<'a>>,
}

/// Routed/shared expert FFN tensors for qwen35moe blocks.
#[derive(Clone)]
pub struct MoeFfn<'a> {
    /// Router logits: `[H, n_expert]`.
    pub gate_inp: &'a TensorDesc,
    /// Routed expert gate projection bank: `[H, F_exp, n_expert]`.
    pub gate_exps: &'a TensorDesc,
    /// Routed expert up projection bank: `[H, F_exp, n_expert]`.
    pub up_exps: &'a TensorDesc,
    /// Routed expert down projection bank: `[F_exp, H, n_expert]`.
    pub down_exps: &'a TensorDesc,
    /// Shared-expert scalar gate input. Stored as `[H]` or `[H, 1]`.
    pub gate_inp_shexp: &'a TensorDesc,
}

#[derive(Clone)]
pub enum Block<'a> {
    Gdn(GdnBlock<'a>),
    Attn(AttnBlock<'a>),
}

/// Multi-Token-Prediction (NEXTN) head. Present when the GGUF contains
/// the `blk.{n_layer}.nextn.*` tensor set. The MTP head is structurally
/// a normal full-attention block at block index `n_layer` (one past the
/// last base layer) plus four adornment tensors (`eh_proj`, `enorm`,
/// `hnorm`, `shared_head_norm`). See `docs/H4-MTP.md` §1.1 for the
/// per-tensor shape table.
///
/// Note: `nextn.embed_tokens` and `nextn.shared_head_head` are
/// `TENSOR_NOT_REQUIRED` in the converter — Qwen3.5/3.6 ties them to
/// the main model's `token_embd` and `output` weights respectively.
/// We don't store separate references; consumers read from
/// `Model.token_embd` / `Model.lm_head`.
#[derive(Clone)]
pub struct MtpHead<'a> {
    /// Block index in the GGUF (typically `arch.n_layer`, e.g. 64 for 27B
    /// or 24 for 0.8B). All `attn` and adornment tensors live under
    /// `blk.{block_idx}.*`.
    pub block_idx: u32,
    /// Standard full-attention block at `blk.{block_idx}.*` — same
    /// tensor names as a regular attn layer (`attn_norm`, `attn_q`,
    /// `attn_k`, `attn_v`, `attn_output`, `attn_q_norm`, `attn_k_norm`,
    /// `post_attention_norm`, `ffn_gate`, `ffn_up`, `ffn_down`).
    pub attn: AttnBlock<'a>,
    /// `blk.{block_idx}.nextn.eh_proj.weight` — `[2*H, H]`. Projects
    /// the concatenated `[RMSNorm(embed(t_{i+1})), RMSNorm(h_i)]` down
    /// to `H`. Concat order is `[embed, hidden]` (vLLM canonical).
    pub eh_proj: &'a TensorDesc,
    /// `blk.{block_idx}.nextn.enorm.weight` — `[H]`. RMSNorm gain
    /// applied to the embedding side of the eh_proj input.
    pub enorm: &'a TensorDesc,
    /// `blk.{block_idx}.nextn.hnorm.weight` — `[H]`. RMSNorm gain
    /// applied to the hidden side of the eh_proj input.
    pub hnorm: &'a TensorDesc,
    /// `blk.{block_idx}.nextn.shared_head_norm.weight` — `[H]`.
    /// RMSNorm gain applied between the MTP block output and the
    /// (shared) lm_head projection.
    pub shared_head_norm: &'a TensorDesc,
}

/// Per-layer DFlash drafter weights. Same tensor-name conventions as a
/// regular Qwen3-family attention block (NOT gated like Qwen3.6 base —
/// `attn_q.weight` is `[H, n_q · head_dim]`, not `[H, 2 · n_q · head_dim]`).
/// The drafter has its own `post_attention_norm` per layer and a final
/// `output_norm` (see `DFlashHead`).
#[derive(Clone)]
pub struct DFlashLayer<'a> {
    pub attn_norm: &'a TensorDesc,
    pub q: &'a TensorDesc,
    pub k: &'a TensorDesc,
    pub v: &'a TensorDesc,
    pub o: &'a TensorDesc,
    pub q_norm: &'a TensorDesc,
    pub k_norm: &'a TensorDesc,
    pub post_attention_norm: &'a TensorDesc,
    pub ffn_gate: &'a TensorDesc,
    pub ffn_up: &'a TensorDesc,
    pub ffn_down: &'a TensorDesc,
    /// True for sliding-window-attention layers (per the GGUF
    /// `sliding_window_pattern` array). False for full-attention layers.
    pub is_swa: bool,
    /// DFlash 2 two-tap dynamic convolution tensors. `Some` iff the
    /// drafter GGUF carries `blk.N.{attn,ffn}_conv_{base,proj}` (llama.cpp
    /// PR 27342 conventions). `None` for DFlash 1 drafters.
    pub conv: Option<DFlashConvTensors<'a>>,
}

/// DFlash 2 per-layer two-tap dynamic depthwise convolution weights.
/// One conv pair wraps attention (side 0 after `attn_norm`, side 1 after
/// `attn_output` before residual), one wraps the FFN (side 0 after
/// `ffn_norm`, side 1 after `ffn_down` before residual). See
/// inco.ai/blog/dflash2 "A Lightweight Local Convolution".
#[derive(Clone)]
pub struct DFlashConvTensors<'a> {
    /// `blk.N.attn_conv_base`, F32 `[H, kernel, 2]` — static base kernel,
    /// last dim = side (0 = pre-sublayer, 1 = post-sublayer).
    pub attn_base: &'a TensorDesc,
    /// `blk.N.attn_conv_proj.weight`, `[H, 2 · kernel · n_groups]` —
    /// dynamic per-token coefficient projection (computed from the normed
    /// sublayer input, used by BOTH sides).
    pub attn_proj: &'a TensorDesc,
    /// `blk.N.ffn_conv_base`, F32 `[H, kernel, 2]`.
    pub ffn_base: &'a TensorDesc,
    /// `blk.N.ffn_conv_proj.weight`, `[H, 2 · kernel · n_groups]`.
    pub ffn_proj: &'a TensorDesc,
}

/// DFlash 2 candidate path-selector weights (global, not per-layer).
/// Scores adjacent candidate pairs: `S(a,b) = U(b) + ⟨A(a) ⊙ W_h·h, B(b)⟩`.
#[derive(Clone)]
pub struct DFlashSelectorTensors<'a> {
    /// `selector_predecessor.weight`, `[rank, V_target]` — per-token A
    /// embedding table (row per token id).
    pub predecessor: &'a TensorDesc,
    /// `selector_successor.weight`, `[rank, V_target]` — per-token B
    /// embedding table.
    pub successor: &'a TensorDesc,
    /// `selector_hidden.weight`, `[H, rank]` — context gate projection
    /// applied to the drafter's final (post-`output_norm`) hidden.
    pub hidden: &'a TensorDesc,
}

/// Static config for the DFlash drafter, derived from GGUF metadata.
#[derive(Clone, Copy, Debug)]
pub struct DFlashConfig {
    pub n_layer: u32,
    /// Drafter hidden size. Must equal target's `hidden_size` for shared
    /// `tok_embd` / `lm_head` to compose.
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub n_q_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub rope_theta: f32,
    /// Sliding-window size (in tokens) for SWA layers. 0 if no SWA layers.
    pub swa_window: u32,
    /// `block_size` from GGUF — number of noise tokens fed to the drafter
    /// per call (e.g. 16 for the 27B-DFlash). Maximum tokens emitted per
    /// outer step is also `block_size`.
    pub block_size: u32,
    /// Token id used for unfilled noise positions in the drafter input.
    pub mask_token_id: i32,
    /// `K = target_layer_ids.len()`. Number of target layers whose hidden
    /// states get fused via `dflash_fc`.
    pub n_target_features_layers: u32,
    /// DFlash 2 conv kernel size (taps). 0 for DFlash 1 drafters.
    pub conv_kernel_size: u32,
    /// DFlash 2 conv group size (channels sharing one dynamic
    /// correction coefficient). 0 for DFlash 1 drafters.
    pub conv_group_size: u32,
    /// DFlash 2 selector embedding rank (256 for the released drafters).
    /// 0 for DFlash 1 drafters.
    pub selector_rank: u32,
    /// DFlash 2 selector top-k candidates per draft position. 0 for
    /// DFlash 1 drafters — this doubles as the "is DFlash 2" flag.
    pub selector_top_k: u32,
}

/// DFlash drafter head. Loaded from a separate GGUF (e.g.
/// `spiritbuun/Qwen3.6-27B-DFlash-GGUF`) and bound to a pre-existing
/// target [`Model`]. Shares `tok_embd` and `output` (lm_head) with
/// the target — the drafter GGUF doesn't carry its own.
pub struct DFlashHead<'a> {
    pub config: DFlashConfig,
    /// K target layer indices whose hiddens get fused.
    pub target_layer_ids: Vec<u32>,
    /// `dflash_fc.weight` (v1) / `fc.weight` (v2), shape `[K · H_target, H_drafter]`.
    pub fc: &'a TensorDesc,
    /// `dflash_hidden_norm.weight` (v1) / `enc.output_norm.weight` (v2),
    /// shape `[H_drafter]`.
    pub hidden_norm: &'a TensorDesc,
    /// `output_norm.weight`, shape `[H_drafter]`. The drafter's own final
    /// RMSNorm before the (target's) lm_head.
    pub output_norm: &'a TensorDesc,
    pub layers: Vec<DFlashLayer<'a>>,
    /// DFlash 2 path-selector tensors. `Some` iff `selector_hidden.weight`
    /// is present in the drafter GGUF.
    pub selector: Option<DFlashSelectorTensors<'a>>,
}

/// Bound model: a static description (Arch) plus tensor references into the
/// mmap'd GGUF. The `GgufFile` it borrows from must outlive this view.
pub struct Model<'a> {
    pub arch: Arch,
    pub token_embd: &'a TensorDesc,
    pub output_norm: &'a TensorDesc,
    /// `lm_head` weight. Falls back to `token_embd.weight` when the GGUF
    /// has no separate `output.weight` (tied embeddings — the small Qwen3.5
    /// variants ship this way; the 27B does not).
    pub lm_head: &'a TensorDesc,
    pub blocks: Vec<Block<'a>>,
    /// True when `lm_head` is the same tensor as `token_embd`.
    pub tied_embeddings: bool,
    /// MTP head, present when the GGUF was produced by the patched
    /// `mtp-converter` lcpp branch. `None` for older GGUFs (rejected
    /// MTP tensors at convert time). Bound by tensor presence, not
    /// metadata — the upstream converter doesn't always write
    /// `qwen35.nextn_predict_layers`. See `docs/H4-MTP.md` §2.
    pub mtp: Option<MtpHead<'a>>,
}

impl<'a> Model<'a> {
    pub fn from_gguf(g: &'a GgufFile) -> Result<Self, LoadError> {
        let arch_str = g.architecture();
        let Some(arch_name) = arch_str.as_deref() else {
            return Err(LoadError::UnsupportedArch(arch_str));
        };
        if arch_name == "deepseek4" {
            return Err(LoadError::DeepSeek4RequiresNativeRuntime);
        }
        if arch_name != "qwen35" && arch_name != "qwen35moe" {
            return Err(LoadError::UnsupportedArch(Some(arch_name.to_string())));
        }
        if let Some(key) = prism_hadamard_metadata_key(g) {
            return Err(LoadError::PrismBasisUnsupported {
                key: key.to_owned(),
            });
        }

        let arch = build_arch_from_metadata(g, arch_name)?;

        // Top-level tensors.
        let token_embd = need(g, "token_embd.weight")?;
        check_shape(
            token_embd,
            &[arch.hidden_size as u64, arch.vocab_size as u64],
        )?;
        let output_norm = need(g, "output_norm.weight")?;
        check_shape(output_norm, &[arch.hidden_size as u64])?;
        let (lm_head, tied_embeddings) = match g.find("output.weight") {
            Some(t) => {
                check_shape(t, &[arch.hidden_size as u64, arch.vocab_size as u64])?;
                (t, false)
            }
            None => (token_embd, true),
        };

        // Per-block.
        let mut blocks: Vec<Block<'a>> = Vec::with_capacity(arch.n_layer as usize);
        for i in 0..arch.n_layer {
            let attn_norm = need(g, &format!("blk.{i}.attn_norm.weight"))?;
            check_shape(attn_norm, &[arch.hidden_size as u64])?;
            let post_attention_norm = need(g, &format!("blk.{i}.post_attention_norm.weight"))?;
            check_shape(post_attention_norm, &[arch.hidden_size as u64])?;
            let (ffn_gate, ffn_up, ffn_down, ffn_moe) = if arch.kind == ArchKind::Dense {
                let ffn_gate = need(g, &format!("blk.{i}.ffn_gate.weight"))?;
                check_shape(
                    ffn_gate,
                    &[arch.hidden_size as u64, arch.intermediate_size as u64],
                )?;
                let ffn_up = need(g, &format!("blk.{i}.ffn_up.weight"))?;
                check_shape(
                    ffn_up,
                    &[arch.hidden_size as u64, arch.intermediate_size as u64],
                )?;
                let ffn_down = need(g, &format!("blk.{i}.ffn_down.weight"))?;
                check_shape(
                    ffn_down,
                    &[arch.intermediate_size as u64, arch.hidden_size as u64],
                )?;
                (ffn_gate, ffn_up, ffn_down, None)
            } else {
                let h = arch.hidden_size as u64;
                let f_exp = arch.expert_feed_forward_length as u64;
                let f_shared = arch.expert_shared_feed_forward_length as u64;
                let n_exp = arch.expert_count as u64;

                let gate_inp = need(g, &format!("blk.{i}.ffn_gate_inp.weight"))?;
                check_shape(gate_inp, &[h, n_exp])?;
                let gate_exps = need(g, &format!("blk.{i}.ffn_gate_exps.weight"))?;
                check_shape(gate_exps, &[h, f_exp, n_exp])?;
                let up_exps = need(g, &format!("blk.{i}.ffn_up_exps.weight"))?;
                check_shape(up_exps, &[h, f_exp, n_exp])?;
                let down_exps = need(g, &format!("blk.{i}.ffn_down_exps.weight"))?;
                check_shape(down_exps, &[f_exp, h, n_exp])?;
                let gate_inp_shexp = need(g, &format!("blk.{i}.ffn_gate_inp_shexp.weight"))?;
                check_shape_one_of(gate_inp_shexp, &[&[h], &[h, 1]])?;

                let ffn_gate = need(g, &format!("blk.{i}.ffn_gate_shexp.weight"))?;
                check_shape(ffn_gate, &[h, f_shared])?;
                let ffn_up = need(g, &format!("blk.{i}.ffn_up_shexp.weight"))?;
                check_shape(ffn_up, &[h, f_shared])?;
                let ffn_down = need(g, &format!("blk.{i}.ffn_down_shexp.weight"))?;
                check_shape(ffn_down, &[f_shared, h])?;

                (
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    Some(MoeFfn {
                        gate_inp,
                        gate_exps,
                        up_exps,
                        down_exps,
                        gate_inp_shexp,
                    }),
                )
            };

            let block = match arch.layer_kind(i) {
                LayerKind::GatedDeltaNet => {
                    // GDN tensors. Combined dim for in_proj_qkv = (n_k + n_k + n_v) * head_dim
                    // = 2*16*128 + 48*128 = 4096 + 6144 = 10240 for 27B,
                    // but qwen3.5-0.8B uses 2*16*128 + 16*128 = 4096+2048 = 6144 — matches!
                    let qkv_heads = checked_add_dim(
                        checked_double_dim(arch.gdn_n_k_heads as u64, "2 * gdn_n_k_heads")?,
                        arch.gdn_n_v_heads as u64,
                        "2 * gdn_n_k_heads + gdn_n_v_heads",
                    )?;
                    let qkv_out =
                        checked_mul_dim(qkv_heads, arch.gdn_head_dim as u64, "gdn qkv_out")?;
                    let z_out = checked_mul_dim(
                        arch.gdn_n_v_heads as u64,
                        arch.gdn_head_dim as u64,
                        "gdn z_out",
                    )?;
                    let in_proj_qkv = need(g, &format!("blk.{i}.attn_qkv.weight"))?;
                    check_shape(in_proj_qkv, &[arch.hidden_size as u64, qkv_out])?;
                    let in_proj_z = need(g, &format!("blk.{i}.attn_gate.weight"))?;
                    check_shape(in_proj_z, &[arch.hidden_size as u64, z_out])?;
                    // GGUF `ssm_beta.weight` feeds β = sigmoid(...).
                    // GGUF `ssm_alpha.weight` feeds the softplus → gate path.
                    // The names are confusing — don't read them as the
                    // Greek letters they share with the math; trust the
                    // qwen35.cpp reference implementation.
                    let beta_proj = need(g, &format!("blk.{i}.ssm_beta.weight"))?;
                    check_shape(
                        beta_proj,
                        &[arch.hidden_size as u64, arch.gdn_n_v_heads as u64],
                    )?;
                    let alpha_proj = need(g, &format!("blk.{i}.ssm_alpha.weight"))?;
                    check_shape(
                        alpha_proj,
                        &[arch.hidden_size as u64, arch.gdn_n_v_heads as u64],
                    )?;
                    let a_log = need(g, &format!("blk.{i}.ssm_a"))?;
                    check_shape(a_log, &[arch.gdn_n_v_heads as u64])?;
                    let dt_bias = need(g, &format!("blk.{i}.ssm_dt.bias"))?;
                    check_shape(dt_bias, &[arch.gdn_n_v_heads as u64])?;
                    let conv1d = need(g, &format!("blk.{i}.ssm_conv1d.weight"))?;
                    let conv_dim =
                        checked_mul_dim(qkv_heads, arch.gdn_head_dim as u64, "gdn conv_dim")?;
                    check_shape(conv1d, &[arch.gdn_conv_kernel as u64, conv_dim])?;
                    let norm = need(g, &format!("blk.{i}.ssm_norm.weight"))?;
                    check_shape(norm, &[arch.gdn_head_dim as u64])?;
                    let out_proj = need(g, &format!("blk.{i}.ssm_out.weight"))?;
                    check_shape(out_proj, &[z_out, arch.hidden_size as u64])?;
                    Block::Gdn(GdnBlock {
                        attn_norm,
                        post_attention_norm,
                        ffn_gate,
                        ffn_up,
                        ffn_down,
                        in_proj_qkv,
                        in_proj_z,
                        beta_proj,
                        alpha_proj,
                        a_log,
                        dt_bias,
                        conv1d,
                        norm,
                        out_proj,
                        ffn_moe: ffn_moe.clone(),
                    })
                }
                LayerKind::GatedAttention => {
                    let q_dim = checked_mul_dim(
                        arch.n_q_heads as u64,
                        arch.attn_head_dim as u64,
                        "attn q_dim",
                    )?;
                    let kv_dim = checked_mul_dim(
                        arch.n_kv_heads as u64,
                        arch.attn_head_dim as u64,
                        "attn kv_dim",
                    )?;
                    // Gated attention: q_proj outputs 2× the heads' worth — first
                    // half is Q, second half is the sigmoid output gate. So the
                    // weight matrix is [hidden, 2*q_dim].
                    let q = need(g, &format!("blk.{i}.attn_q.weight"))?;
                    check_shape(
                        q,
                        &[
                            arch.hidden_size as u64,
                            checked_double_dim(q_dim, "2 * attn q_dim")?,
                        ],
                    )?;
                    let k = need(g, &format!("blk.{i}.attn_k.weight"))?;
                    check_shape(k, &[arch.hidden_size as u64, kv_dim])?;
                    let v = need(g, &format!("blk.{i}.attn_v.weight"))?;
                    check_shape(v, &[arch.hidden_size as u64, kv_dim])?;
                    let o = need(g, &format!("blk.{i}.attn_output.weight"))?;
                    check_shape(o, &[q_dim, arch.hidden_size as u64])?;
                    let q_norm = need(g, &format!("blk.{i}.attn_q_norm.weight"))?;
                    check_shape(q_norm, &[arch.attn_head_dim as u64])?;
                    let k_norm = need(g, &format!("blk.{i}.attn_k_norm.weight"))?;
                    check_shape(k_norm, &[arch.attn_head_dim as u64])?;
                    Block::Attn(AttnBlock {
                        attn_norm,
                        post_attention_norm,
                        ffn_gate,
                        ffn_up,
                        ffn_down,
                        q,
                        k,
                        v,
                        o,
                        q_norm,
                        k_norm,
                        ffn_moe,
                    })
                }
            };
            blocks.push(block);
        }

        // MTP head — bind iff the eh_proj tensor exists at blk.{n_layer}.
        // Qwen3.5/3.6 ships at most ONE MTP block (`mtp_num_hidden_layers=1`)
        // located at block index `arch.n_layer` (one past the last base
        // layer). All MTP-aware GGUFs from the patched mtp-converter put
        // it there; older GGUFs (pre-converter-patch) don't have it.
        let mtp = bind_mtp_head(g, &arch)?;

        Ok(Self {
            arch,
            token_embd,
            output_norm,
            lm_head,
            blocks,
            tied_embeddings,
            mtp,
        })
    }
}

/// Probe for the MTP head at `blk.{arch.n_layer}.*` and bind it if the
/// eh_proj tensor is present. Tensor presence is the ground truth — the
/// `qwen35.nextn_predict_layers` metadata key may be absent on some
/// GGUFs even when the tensors are there (and vice versa for older
/// metadata-only stubs).
///
/// Returns `Ok(None)` if the GGUF has no MTP head; `Err` only on
/// partial / shape-mismatched MTP tensor sets (the eh_proj being
/// present but other required tensors missing is a converter bug
/// worth surfacing).
fn bind_mtp_head<'a>(g: &'a GgufFile, arch: &Arch) -> Result<Option<MtpHead<'a>>, LoadError> {
    let i = arch.n_layer; // MTP block index
    let eh_proj_name = format!("blk.{i}.nextn.eh_proj.weight");
    let Some(eh_proj) = g.find(&eh_proj_name) else {
        // No MTP head present — older GGUF or non-MTP-aware converter.
        return Ok(None);
    };
    // eh_proj: [2*H, H]
    check_shape(
        eh_proj,
        &[
            checked_double_dim(arch.hidden_size as u64, "mtp 2 * hidden_size")?,
            arch.hidden_size as u64,
        ],
    )?;

    let enorm = need(g, &format!("blk.{i}.nextn.enorm.weight"))?;
    check_shape(enorm, &[arch.hidden_size as u64])?;
    let hnorm = need(g, &format!("blk.{i}.nextn.hnorm.weight"))?;
    check_shape(hnorm, &[arch.hidden_size as u64])?;
    let shared_head_norm = need(g, &format!("blk.{i}.nextn.shared_head_norm.weight"))?;
    check_shape(shared_head_norm, &[arch.hidden_size as u64])?;

    // The MTP block's own attention + FFN tensors live at the same
    // block index, with the SAME tensor names as a regular full-attn
    // block. The converter at lcpp@mtp-converter remaps HF
    // `mtp.layers.0.*` → `model.layers.{n_base}.*`. So we can reuse
    // the AttnBlock binder logic verbatim.
    let attn_norm = need(g, &format!("blk.{i}.attn_norm.weight"))?;
    check_shape(attn_norm, &[arch.hidden_size as u64])?;
    let post_attention_norm = need(g, &format!("blk.{i}.post_attention_norm.weight"))?;
    check_shape(post_attention_norm, &[arch.hidden_size as u64])?;
    let (ffn_gate, ffn_up, ffn_down, ffn_moe) = if arch.kind == ArchKind::Dense {
        let ffn_gate = need(g, &format!("blk.{i}.ffn_gate.weight"))?;
        check_shape(
            ffn_gate,
            &[arch.hidden_size as u64, arch.intermediate_size as u64],
        )?;
        let ffn_up = need(g, &format!("blk.{i}.ffn_up.weight"))?;
        check_shape(
            ffn_up,
            &[arch.hidden_size as u64, arch.intermediate_size as u64],
        )?;
        let ffn_down = need(g, &format!("blk.{i}.ffn_down.weight"))?;
        check_shape(
            ffn_down,
            &[arch.intermediate_size as u64, arch.hidden_size as u64],
        )?;
        (ffn_gate, ffn_up, ffn_down, None)
    } else {
        let h = arch.hidden_size as u64;
        let f_exp = arch.expert_feed_forward_length as u64;
        let f_shared = arch.expert_shared_feed_forward_length as u64;
        let n_exp = arch.expert_count as u64;

        let gate_inp = need(g, &format!("blk.{i}.ffn_gate_inp.weight"))?;
        check_shape(gate_inp, &[h, n_exp])?;
        let gate_exps = need(g, &format!("blk.{i}.ffn_gate_exps.weight"))?;
        check_shape(gate_exps, &[h, f_exp, n_exp])?;
        let up_exps = need(g, &format!("blk.{i}.ffn_up_exps.weight"))?;
        check_shape(up_exps, &[h, f_exp, n_exp])?;
        let down_exps = need(g, &format!("blk.{i}.ffn_down_exps.weight"))?;
        check_shape(down_exps, &[f_exp, h, n_exp])?;
        let gate_inp_shexp = need(g, &format!("blk.{i}.ffn_gate_inp_shexp.weight"))?;
        check_shape_one_of(gate_inp_shexp, &[&[h], &[h, 1]])?;

        let ffn_gate = need(g, &format!("blk.{i}.ffn_gate_shexp.weight"))?;
        check_shape(ffn_gate, &[h, f_shared])?;
        let ffn_up = need(g, &format!("blk.{i}.ffn_up_shexp.weight"))?;
        check_shape(ffn_up, &[h, f_shared])?;
        let ffn_down = need(g, &format!("blk.{i}.ffn_down_shexp.weight"))?;
        check_shape(ffn_down, &[f_shared, h])?;

        (
            ffn_gate,
            ffn_up,
            ffn_down,
            Some(MoeFfn {
                gate_inp,
                gate_exps,
                up_exps,
                down_exps,
                gate_inp_shexp,
            }),
        )
    };
    let q_dim = checked_mul_dim(
        arch.n_q_heads as u64,
        arch.attn_head_dim as u64,
        "mtp q_dim",
    )?;
    let kv_dim = checked_mul_dim(
        arch.n_kv_heads as u64,
        arch.attn_head_dim as u64,
        "mtp kv_dim",
    )?;
    let q = need(g, &format!("blk.{i}.attn_q.weight"))?;
    // Gated attention: q_proj outputs 2× q_dim (Q + sigmoid gate).
    check_shape(
        q,
        &[
            arch.hidden_size as u64,
            checked_double_dim(q_dim, "mtp 2 * q_dim")?,
        ],
    )?;
    let k = need(g, &format!("blk.{i}.attn_k.weight"))?;
    check_shape(k, &[arch.hidden_size as u64, kv_dim])?;
    let v = need(g, &format!("blk.{i}.attn_v.weight"))?;
    check_shape(v, &[arch.hidden_size as u64, kv_dim])?;
    let o = need(g, &format!("blk.{i}.attn_output.weight"))?;
    check_shape(o, &[q_dim, arch.hidden_size as u64])?;
    let q_norm = need(g, &format!("blk.{i}.attn_q_norm.weight"))?;
    check_shape(q_norm, &[arch.attn_head_dim as u64])?;
    let k_norm = need(g, &format!("blk.{i}.attn_k_norm.weight"))?;
    check_shape(k_norm, &[arch.attn_head_dim as u64])?;

    Ok(Some(MtpHead {
        block_idx: i,
        attn: AttnBlock {
            attn_norm,
            post_attention_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            q,
            k,
            v,
            o,
            q_norm,
            k_norm,
            ffn_moe,
        },
        eh_proj,
        enorm,
        hnorm,
        shared_head_norm,
    }))
}

/// Open a separately-distributed DFlash drafter GGUF and bind it to a
/// pre-existing target [`Model`]. The drafter doesn't carry its own
/// vocab embedding or lm_head; it shares the target's via `target_model.token_embd`
/// and `target_model.lm_head`. Hidden size compatibility is enforced.
///
/// Reference: `~/models/spiritbuun-dflash/dflash-draft-3.6-q8_0.gguf`.
/// Arch string: `dflash-draft`. KV namespace: `dflash-draft.*` (NOT
/// `qwen35.dflash.*` — that's a different fork's convention).
pub fn open_dflash_drafter<'a>(
    drafter_gguf: &'a GgufFile,
    target_model: &Model<'_>,
) -> Result<DFlashHead<'a>, LoadError> {
    /// Metadata-key + tensor-name table for the two drafter GGUF
    /// conventions:
    /// * v1 — arch `dflash-draft` (spiritbuun converter): KV under
    ///   `dflash-draft.*` / `dflash-draft.dflash.*`; tensors
    ///   `dflash_fc.weight`, `dflash_hidden_norm.weight`,
    ///   `blk.N.post_attention_norm.weight`.
    /// * v2 — arch `dflash` (llama.cpp PR 27342 converter, DFlash 2 era,
    ///   e.g. `incoai/Qwen3.8-27B-DFlash2-GGUF`): KV under `dflash.*`;
    ///   mask token in `tokenizer.ggml.mask_token_id`; tensors
    ///   `fc.weight`, `enc.output_norm.weight`, `blk.N.ffn_norm.weight`;
    ///   optional DFlash 2 conv/selector tensors.
    struct DrafterKeys {
        block_count: &'static str,
        embedding_length: &'static str,
        feed_forward_length: &'static str,
        head_count: &'static str,
        head_count_kv: &'static str,
        key_length: &'static str,
        freq_base: &'static str,
        sliding_window: &'static str,
        sliding_window_pattern: &'static str,
        block_size: &'static str,
        mask_token_id: &'static str,
        target_layer_ids: &'static str,
        /// `Some` for v1 (explicit key); `None` for v2 (derived as
        /// `K · H_target`).
        n_target_features: Option<&'static str>,
        fc_tensor: &'static str,
        hidden_norm_tensor: &'static str,
        /// Per-layer pre-FFN norm tensor suffix under `blk.{i}.`.
        post_attn_norm_suffix: &'static str,
    }
    const V1_KEYS: DrafterKeys = DrafterKeys {
        block_count: "dflash-draft.block_count",
        embedding_length: "dflash-draft.embedding_length",
        feed_forward_length: "dflash-draft.feed_forward_length",
        head_count: "dflash-draft.attention.head_count",
        head_count_kv: "dflash-draft.attention.head_count_kv",
        key_length: "dflash-draft.attention.key_length",
        freq_base: "dflash-draft.rope.freq_base",
        sliding_window: "dflash-draft.attention.sliding_window",
        sliding_window_pattern: "dflash-draft.attention.sliding_window_pattern",
        block_size: "dflash-draft.dflash.block_size",
        mask_token_id: "dflash-draft.dflash.mask_token_id",
        target_layer_ids: "dflash-draft.dflash.target_layer_ids",
        n_target_features: Some("dflash-draft.dflash.n_target_features"),
        fc_tensor: "dflash_fc.weight",
        hidden_norm_tensor: "dflash_hidden_norm.weight",
        post_attn_norm_suffix: "post_attention_norm",
    };
    const V2_KEYS: DrafterKeys = DrafterKeys {
        block_count: "dflash.block_count",
        embedding_length: "dflash.embedding_length",
        feed_forward_length: "dflash.feed_forward_length",
        head_count: "dflash.attention.head_count",
        head_count_kv: "dflash.attention.head_count_kv",
        key_length: "dflash.attention.key_length",
        freq_base: "dflash.rope.freq_base",
        sliding_window: "dflash.attention.sliding_window",
        sliding_window_pattern: "dflash.attention.sliding_window_pattern",
        block_size: "dflash.block_size",
        mask_token_id: "tokenizer.ggml.mask_token_id",
        target_layer_ids: "dflash.target_layers",
        n_target_features: None,
        fc_tensor: "fc.weight",
        hidden_norm_tensor: "enc.output_norm.weight",
        post_attn_norm_suffix: "ffn_norm",
    };

    let arch_str = drafter_gguf.architecture();
    let keys: &DrafterKeys = match arch_str.as_deref() {
        Some("dflash-draft") => &V1_KEYS,
        Some("dflash") => &V2_KEYS,
        _ => return Err(LoadError::UnsupportedArch(arch_str)),
    };

    // -------- Read drafter config from KV metadata --------
    let n_layer = drafter_gguf
        .get_u64(keys.block_count)
        .ok_or(LoadError::BadMetadata(keys.block_count))?;
    let n_layer = u64_to_u32(keys.block_count, n_layer)?;
    let hidden_size_raw = drafter_gguf
        .get_u64(keys.embedding_length)
        .ok_or(LoadError::BadMetadata(keys.embedding_length))?;
    let hidden_size = u64_to_u32(keys.embedding_length, hidden_size_raw)?;
    let intermediate_size_raw = drafter_gguf
        .get_u64(keys.feed_forward_length)
        .ok_or(LoadError::BadMetadata(keys.feed_forward_length))?;
    let intermediate_size = u64_to_u32(keys.feed_forward_length, intermediate_size_raw)?;
    let n_q_heads_raw = drafter_gguf
        .get_u64(keys.head_count)
        .ok_or(LoadError::BadMetadata(keys.head_count))?;
    let n_q_heads = u64_to_u32(keys.head_count, n_q_heads_raw)?;
    let n_kv_heads_raw = drafter_gguf
        .get_u64(keys.head_count_kv)
        .ok_or(LoadError::BadMetadata(keys.head_count_kv))?;
    let n_kv_heads = u64_to_u32(keys.head_count_kv, n_kv_heads_raw)?;
    let head_dim_raw = drafter_gguf
        .get_u64(keys.key_length)
        .ok_or(LoadError::BadMetadata(keys.key_length))?;
    let head_dim = u64_to_u32(keys.key_length, head_dim_raw)?;
    let rope_theta = drafter_gguf.get_f32(keys.freq_base).unwrap_or(10_000_000.0);
    let swa_window = drafter_gguf
        .get_u64(keys.sliding_window)
        .map_or(Ok(0), |v| u64_to_u32(keys.sliding_window, v))?;
    let block_size = drafter_gguf
        .get_u64(keys.block_size)
        .ok_or(LoadError::BadMetadata(keys.block_size))?;
    let block_size = u64_to_u32(keys.block_size, block_size)?;
    let mask_token_id = drafter_gguf
        .get_u64(keys.mask_token_id)
        .ok_or(LoadError::BadMetadata(keys.mask_token_id))?;
    let mask_token_id = u64_to_i32(keys.mask_token_id, mask_token_id)?;
    let target_layer_ids: Vec<u32> = drafter_gguf
        .get_u64_array(keys.target_layer_ids)
        .map_err(|_| LoadError::BadMetadata(keys.target_layer_ids))?
        .ok_or(LoadError::BadMetadata(keys.target_layer_ids))?
        .into_iter()
        .map(|v| u64_to_u32(keys.target_layer_ids, v))
        .collect::<Result<Vec<_>, _>>()?;
    let n_target_features = match keys.n_target_features {
        Some(key) => {
            let raw = drafter_gguf
                .get_u64(key)
                .ok_or(LoadError::BadMetadata(key))?;
            u64_to_u32(key, raw)?
        }
        // v2 drops the explicit key; the fc shape check below still
        // cross-validates the derived value.
        None => {
            let derived = checked_mul_dim(
                target_layer_ids.len() as u64,
                target_model.arch.hidden_size as u64,
                "dflash target_layer_ids.len * target_h",
            )?;
            u64_to_u32(keys.target_layer_ids, derived)?
        }
    };
    let swa_pattern: Vec<bool> = drafter_gguf
        .get_bool_array(keys.sliding_window_pattern)
        .map_err(|_| LoadError::BadMetadata(keys.sliding_window_pattern))?
        .ok_or(LoadError::BadMetadata(keys.sliding_window_pattern))?;

    // A/B experiment (adversarial review F1, QWEN_DFLASH2_CAPTURE_SHIFT=1,
    // v2 drafters only): llama.cpp feeds the drafter the INPUT to target
    // layer `lid` (`llama_get_embeddings_layer_inp` = output of `lid-1`),
    // while this engine's capture machinery snapshots the OUTPUT of layer
    // `lid`. Shifting the ids by -1 makes our post-layer capture deliver
    // the reference's features. The v1 (`dflash-draft`) GGUF's ids are
    // trusted as already matching this engine's convention.
    let target_layer_ids = if keys.n_target_features.is_none()
        && std::env::var("QWEN_DFLASH2_CAPTURE_SHIFT").as_deref() == Ok("1")
    {
        target_layer_ids
            .into_iter()
            .map(|lid| {
                lid.checked_sub(1).ok_or(LoadError::BadMetadata(
                    "QWEN_DFLASH2_CAPTURE_SHIFT cannot shift target layer id 0",
                ))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        target_layer_ids
    };

    // -------- DFlash 2 conv/selector metadata (v2 only, presence-gated) --------
    // Presence contract mirrors llama.cpp PR 27342: `selector_hidden.weight`
    // in the GGUF means "this is a DFlash 2 drafter" and all four keys +
    // conv tensors become mandatory.
    let is_dflash2 = drafter_gguf.find("selector_hidden.weight").is_some();
    let (conv_kernel_size, conv_group_size, selector_rank, selector_top_k) = if is_dflash2 {
        let kernel = drafter_gguf
            .get_u64("dflash.conv_kernel_size")
            .ok_or(LoadError::BadMetadata("dflash.conv_kernel_size"))?;
        let kernel = u64_to_u32("dflash.conv_kernel_size", kernel)?;
        let group = drafter_gguf
            .get_u64("dflash.conv_group_size")
            .ok_or(LoadError::BadMetadata("dflash.conv_group_size"))?;
        let group = u64_to_u32("dflash.conv_group_size", group)?;
        let rank = drafter_gguf
            .get_u64("dflash.selector_rank")
            .ok_or(LoadError::BadMetadata("dflash.selector_rank"))?;
        let rank = u64_to_u32("dflash.selector_rank", rank)?;
        let top_k = drafter_gguf
            .get_u64("dflash.selector_top_k")
            .ok_or(LoadError::BadMetadata("dflash.selector_top_k"))?;
        let top_k = u64_to_u32("dflash.selector_top_k", top_k)?;
        ensure_nonzero("dflash.conv_kernel_size", kernel)?;
        ensure_cap("dflash.conv_kernel_size", kernel, 8)?;
        ensure_nonzero("dflash.conv_group_size", group)?;
        ensure_nonzero("dflash.selector_rank", rank)?;
        ensure_cap("dflash.selector_rank", rank, MAX_ARCH_DIM)?;
        ensure_nonzero("dflash.selector_top_k", top_k)?;
        ensure_cap("dflash.selector_top_k", top_k, 64)?;
        if hidden_size % group != 0 {
            return Err(LoadError::BadMetadata(
                "dflash.conv_group_size must divide embedding_length",
            ));
        }
        (kernel, group, rank, top_k)
    } else {
        (0, 0, 0, 0)
    };

    // -------- Compatibility validation --------
    let target_h = target_model.arch.hidden_size;
    ensure_cap("dflash-draft.block_count", n_layer, MAX_ARCH_LAYERS)?;
    ensure_nonzero("dflash-draft.block_count", n_layer)?;
    ensure_cap("dflash-draft.embedding_length", hidden_size, MAX_ARCH_DIM)?;
    ensure_nonzero("dflash-draft.embedding_length", hidden_size)?;
    ensure_cap(
        "dflash-draft.feed_forward_length",
        intermediate_size,
        MAX_ARCH_DIM,
    )?;
    ensure_nonzero("dflash-draft.feed_forward_length", intermediate_size)?;
    ensure_cap("dflash-draft.attention.head_count", n_q_heads, MAX_ARCH_DIM)?;
    ensure_nonzero("dflash-draft.attention.head_count", n_q_heads)?;
    ensure_cap(
        "dflash-draft.attention.head_count_kv",
        n_kv_heads,
        MAX_ARCH_DIM,
    )?;
    ensure_nonzero("dflash-draft.attention.head_count_kv", n_kv_heads)?;
    ensure_divisible(
        "dflash-draft.attention.head_count",
        n_q_heads,
        "dflash-draft.attention.head_count_kv",
        n_kv_heads,
    )?;
    ensure_cap("dflash-draft.attention.key_length", head_dim, MAX_ARCH_DIM)?;
    ensure_nonzero("dflash-draft.attention.key_length", head_dim)?;
    ensure_cap(
        "dflash-draft.dflash.block_size",
        block_size,
        MAX_DFLASH_BLOCK_SIZE,
    )?;
    ensure_nonzero("dflash-draft.dflash.block_size", block_size)?;
    ensure_cap(
        "dflash-draft.dflash.n_target_features",
        n_target_features,
        MAX_ARCH_DIM,
    )?;
    ensure_nonzero("dflash-draft.dflash.n_target_features", n_target_features)?;
    if hidden_size != target_h {
        return Err(LoadError::BadMetadata(
            "dflash-draft.embedding_length must equal target's hidden_size \
             (drafter shares target's tok_embd / lm_head; H mismatch breaks composition)",
        ));
    }
    let target_feature_layers =
        u32::try_from(target_layer_ids.len()).map_err(|_| LoadError::MetadataOverflow {
            key: "dflash-draft.dflash.target_layer_ids",
            got: target_layer_ids.len() as u64,
        })?;
    ensure_cap(
        "dflash-draft.dflash.target_layer_ids",
        target_feature_layers,
        MAX_ARCH_LAYERS,
    )?;
    ensure_nonzero(
        "dflash-draft.dflash.target_layer_ids",
        target_feature_layers,
    )?;
    let expected_n_target_features = checked_mul_dim(
        target_layer_ids.len() as u64,
        target_h as u64,
        "dflash target_layer_ids.len * target_h",
    )?;
    if n_target_features as u64 != expected_n_target_features {
        return Err(LoadError::BadMetadata(
            "dflash-draft.dflash.n_target_features mismatch: \
             expected K · H_target",
        ));
    }
    for &lid in &target_layer_ids {
        if lid >= target_model.arch.n_layer {
            return Err(LoadError::BadMetadata(
                "dflash-draft.dflash.target_layer_ids references a layer past target's n_layer",
            ));
        }
    }
    if swa_pattern.len() != n_layer as usize {
        return Err(LoadError::BadMetadata(
            "dflash-draft.attention.sliding_window_pattern length must equal block_count",
        ));
    }
    // Fail closed: a missing/zero sliding_window with SWA layers present
    // would silently run those layers as FULL attention (swa_window=0 is
    // the kernels' "no SWA" sentinel) — a semantic corruption, not an
    // error (2026-08-19 adversarial review, F9).
    if swa_window == 0 && swa_pattern.iter().any(|&is_swa| is_swa) {
        return Err(LoadError::BadMetadata(
            "drafter declares SWA layers but attention.sliding_window is missing or 0",
        ));
    }

    let q_dim = checked_mul_dim(n_q_heads as u64, head_dim as u64, "dflash q_dim")?;
    if q_dim > MAX_KERNEL_DIM {
        return Err(LoadError::MetadataCap {
            key: "dflash-draft.attention.q_dim",
            got: q_dim,
            cap: MAX_KERNEL_DIM,
        });
    }
    let kv_dim = checked_mul_dim(n_kv_heads as u64, head_dim as u64, "dflash kv_dim")?;
    if kv_dim > MAX_KERNEL_DIM {
        return Err(LoadError::MetadataCap {
            key: "dflash-draft.attention.kv_dim",
            got: kv_dim,
            cap: MAX_KERNEL_DIM,
        });
    }

    // -------- Top-level adornment tensors --------
    let fc = need(drafter_gguf, keys.fc_tensor)?;
    check_shape(fc, &[n_target_features as u64, hidden_size as u64])?;
    let hidden_norm = need(drafter_gguf, keys.hidden_norm_tensor)?;
    check_shape(hidden_norm, &[hidden_size as u64])?;
    let output_norm = need(drafter_gguf, "output_norm.weight")?;
    check_shape(output_norm, &[hidden_size as u64])?;

    // -------- DFlash 2 selector tensors --------
    let selector: Option<DFlashSelectorTensors<'a>> = if is_dflash2 {
        let v_target = target_model.arch.vocab_size as u64;
        let rank = selector_rank as u64;
        let predecessor = need(drafter_gguf, "selector_predecessor.weight")?;
        check_shape(predecessor, &[rank, v_target])?;
        let successor = need(drafter_gguf, "selector_successor.weight")?;
        check_shape(successor, &[rank, v_target])?;
        let sel_hidden = need(drafter_gguf, "selector_hidden.weight")?;
        check_shape(sel_hidden, &[hidden_size as u64, rank])?;
        Some(DFlashSelectorTensors {
            predecessor,
            successor,
            hidden: sel_hidden,
        })
    } else {
        None
    };

    // -------- Per-layer tensors --------
    let h = hidden_size as u64;
    let f = intermediate_size as u64;
    let mut layers: Vec<DFlashLayer<'a>> = Vec::with_capacity(n_layer as usize);
    for i in 0..n_layer {
        let attn_norm = need(drafter_gguf, &format!("blk.{i}.attn_norm.weight"))?;
        check_shape(attn_norm, &[h])?;
        let q = need(drafter_gguf, &format!("blk.{i}.attn_q.weight"))?;
        // NOT gated: shape [H, n_q · head_dim], not [H, 2 · n_q · head_dim].
        check_shape(q, &[h, q_dim])?;
        let k = need(drafter_gguf, &format!("blk.{i}.attn_k.weight"))?;
        check_shape(k, &[h, kv_dim])?;
        let v = need(drafter_gguf, &format!("blk.{i}.attn_v.weight"))?;
        check_shape(v, &[h, kv_dim])?;
        let o = need(drafter_gguf, &format!("blk.{i}.attn_output.weight"))?;
        check_shape(o, &[q_dim, h])?;
        let q_norm = need(drafter_gguf, &format!("blk.{i}.attn_q_norm.weight"))?;
        check_shape(q_norm, &[head_dim as u64])?;
        let k_norm = need(drafter_gguf, &format!("blk.{i}.attn_k_norm.weight"))?;
        check_shape(k_norm, &[head_dim as u64])?;
        let post_attention_norm = need(
            drafter_gguf,
            &format!("blk.{i}.{}.weight", keys.post_attn_norm_suffix),
        )?;
        check_shape(post_attention_norm, &[h])?;
        let ffn_gate = need(drafter_gguf, &format!("blk.{i}.ffn_gate.weight"))?;
        check_shape(ffn_gate, &[h, f])?;
        let ffn_up = need(drafter_gguf, &format!("blk.{i}.ffn_up.weight"))?;
        check_shape(ffn_up, &[h, f])?;
        let ffn_down = need(drafter_gguf, &format!("blk.{i}.ffn_down.weight"))?;
        check_shape(ffn_down, &[f, h])?;
        let conv = if is_dflash2 {
            let kernel = conv_kernel_size as u64;
            let n_groups = (hidden_size / conv_group_size) as u64;
            let proj_out = checked_mul_dim(2 * kernel, n_groups, "dflash2 conv proj out")?;
            let attn_base = need(drafter_gguf, &format!("blk.{i}.attn_conv_base"))?;
            check_shape(attn_base, &[h, kernel, 2])?;
            let attn_proj = need(drafter_gguf, &format!("blk.{i}.attn_conv_proj.weight"))?;
            check_shape(attn_proj, &[h, proj_out])?;
            let ffn_base = need(drafter_gguf, &format!("blk.{i}.ffn_conv_base"))?;
            check_shape(ffn_base, &[h, kernel, 2])?;
            let ffn_proj = need(drafter_gguf, &format!("blk.{i}.ffn_conv_proj.weight"))?;
            check_shape(ffn_proj, &[h, proj_out])?;
            Some(DFlashConvTensors {
                attn_base,
                attn_proj,
                ffn_base,
                ffn_proj,
            })
        } else {
            None
        };
        layers.push(DFlashLayer {
            attn_norm,
            q,
            k,
            v,
            o,
            q_norm,
            k_norm,
            post_attention_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            is_swa: swa_pattern[i as usize],
            conv,
        });
    }

    Ok(DFlashHead {
        config: DFlashConfig {
            n_layer,
            hidden_size,
            intermediate_size,
            n_q_heads,
            n_kv_heads,
            head_dim,
            rope_theta,
            swa_window,
            block_size,
            mask_token_id,
            n_target_features_layers: target_feature_layers,
            conv_kernel_size,
            conv_group_size,
            selector_rank,
            selector_top_k,
        },
        target_layer_ids,
        fc,
        hidden_norm,
        output_norm,
        layers,
        selector,
    })
}

fn need<'a>(g: &'a GgufFile, name: &str) -> Result<&'a TensorDesc, LoadError> {
    g.find(name)
        .ok_or_else(|| LoadError::Missing(name.to_string()))
}

/// GGUF stores matrices with the "in" dimension first: `shape = [in, out]`
/// for `y = x @ W`. We accept either rank-1 `[N]` or rank-2 `[in, out]`
/// matches.
fn check_shape(t: &TensorDesc, expected: &[u64]) -> Result<(), LoadError> {
    if t.shape != expected {
        return Err(LoadError::Shape {
            name: t.name.clone(),
            expected: expected.to_vec(),
            got: t.shape.clone(),
        });
    }
    Ok(())
}

fn check_shape_one_of(t: &TensorDesc, expected_any: &[&[u64]]) -> Result<(), LoadError> {
    if expected_any.iter().any(|shape| t.shape == *shape) {
        Ok(())
    } else {
        Err(LoadError::Shape {
            name: t.name.clone(),
            expected: expected_any.first().copied().unwrap_or(&[]).to_vec(),
            got: t.shape.clone(),
        })
    }
}

/// Build an [`Arch`] from the GGUF's `qwen35.*` metadata keys. Cross-checks
/// against the known constants in [`crate::model`] are the caller's job.
fn build_arch_from_metadata(g: &GgufFile, arch_name: &str) -> Result<Arch, LoadError> {
    let kind = match arch_name {
        "qwen35" => ArchKind::Dense,
        "qwen35moe" => ArchKind::Moe,
        _ => return Err(LoadError::UnsupportedArch(Some(arch_name.to_string()))),
    };
    let p = arch_name;
    // GGUF's `block_count` includes any trailing MTP/NEXTN predict layers
    // (per the patched converter that preserves them). The main forward
    // path iterates only the base layers, so subtract any MTP layers from
    // n_layer here. Older GGUFs (pre-MTP-converter) don't have the
    // nextn_predict_layers key, in which case we default to 0 and the
    // subtraction is a no-op (preserving the prior behavior).
    let block_count_u64 = g
        .get_u64(&format!("{p}.block_count"))
        .ok_or(LoadError::BadMetadata("qwen35.block_count"))?;
    let block_count = u64_to_u32("qwen35.block_count", block_count_u64)?;
    let mtp_n_hidden_layers_u64 = g.get_u64(&format!("{p}.nextn_predict_layers")).unwrap_or(0);
    let mtp_n_hidden_layers = u64_to_u32("qwen35.nextn_predict_layers", mtp_n_hidden_layers_u64)?;
    ensure_cap("qwen35.block_count", block_count, MAX_ARCH_LAYERS)?;
    ensure_cap(
        "qwen35.nextn_predict_layers",
        mtp_n_hidden_layers,
        MAX_ARCH_LAYERS,
    )?;
    // Strict subtraction: a malformed GGUF with `nextn_predict_layers >
    // block_count` previously saturated to a 0-layer model, which then
    // produced a downstream panic in an unrelated module. Reject up front.
    if mtp_n_hidden_layers > block_count {
        return Err(LoadError::MetadataInconsistent {
            key: "qwen35.nextn_predict_layers",
            got: mtp_n_hidden_layers as u64,
            bound_key: "qwen35.block_count",
            bound: block_count as u64,
        });
    }
    let n_layer = block_count - mtp_n_hidden_layers;
    let hidden_size = g
        .get_u64(&format!("{p}.embedding_length"))
        .ok_or(LoadError::BadMetadata("qwen35.embedding_length"))?;
    let hidden_size = u64_to_u32("qwen35.embedding_length", hidden_size)?;
    let intermediate_size = if kind == ArchKind::Dense {
        let value = g
            .get_u64(&format!("{p}.feed_forward_length"))
            .ok_or(LoadError::BadMetadata("qwen35.feed_forward_length"))?;
        u64_to_u32("qwen35.feed_forward_length", value)?
    } else {
        0
    };
    let n_q_heads_raw = g
        .get_u64(&format!("{p}.attention.head_count"))
        .ok_or(LoadError::BadMetadata("qwen35.attention.head_count"))?;
    let n_q_heads = u64_to_u32("qwen35.attention.head_count", n_q_heads_raw)?;
    let n_kv_heads_raw = g
        .get_u64(&format!("{p}.attention.head_count_kv"))
        .ok_or(LoadError::BadMetadata("qwen35.attention.head_count_kv"))?;
    let n_kv_heads = u64_to_u32("qwen35.attention.head_count_kv", n_kv_heads_raw)?;
    let attn_head_dim_raw = g
        .get_u64(&format!("{p}.attention.key_length"))
        .ok_or(LoadError::BadMetadata("qwen35.attention.key_length"))?;
    let attn_head_dim = u64_to_u32("qwen35.attention.key_length", attn_head_dim_raw)?;
    let full_attention_interval = match g.get_u64(&format!("{p}.full_attention_interval")) {
        Some(value) => u64_to_u32("qwen35.full_attention_interval", value)?,
        None => 4,
    };

    // GDN dims.
    // ssm.inner_size = num_v_heads * head_dim
    // ssm.state_size = head_dim
    // ssm.time_step_rank = num_v_heads
    // ssm.group_count = num_k_heads
    // ssm.conv_kernel = conv kernel size
    let ssm_state_size = g
        .get_u64(&format!("{p}.ssm.state_size"))
        .ok_or(LoadError::BadMetadata("qwen35.ssm.state_size"))?;
    let ssm_state_size = u64_to_u32("qwen35.ssm.state_size", ssm_state_size)?;
    let ssm_time_step_rank = g
        .get_u64(&format!("{p}.ssm.time_step_rank"))
        .ok_or(LoadError::BadMetadata("qwen35.ssm.time_step_rank"))?;
    let ssm_time_step_rank = u64_to_u32("qwen35.ssm.time_step_rank", ssm_time_step_rank)?;
    let ssm_group_count = g
        .get_u64(&format!("{p}.ssm.group_count"))
        .ok_or(LoadError::BadMetadata("qwen35.ssm.group_count"))?;
    let ssm_group_count = u64_to_u32("qwen35.ssm.group_count", ssm_group_count)?;
    let ssm_conv_kernel = g
        .get_u64(&format!("{p}.ssm.conv_kernel"))
        .ok_or(LoadError::BadMetadata("qwen35.ssm.conv_kernel"))?;
    let ssm_conv_kernel = u64_to_u32("qwen35.ssm.conv_kernel", ssm_conv_kernel)?;

    let expert_count = if kind == ArchKind::Moe {
        let value = g
            .get_u64(&format!("{p}.expert_count"))
            .ok_or(LoadError::BadMetadata("qwen35moe.expert_count"))?;
        u64_to_u32("qwen35moe.expert_count", value)?
    } else {
        0
    };
    let expert_used_count = if kind == ArchKind::Moe {
        let value = g
            .get_u64(&format!("{p}.expert_used_count"))
            .ok_or(LoadError::BadMetadata("qwen35moe.expert_used_count"))?;
        u64_to_u32("qwen35moe.expert_used_count", value)?
    } else {
        0
    };
    let expert_feed_forward_length = if kind == ArchKind::Moe {
        let value = g
            .get_u64(&format!("{p}.expert_feed_forward_length"))
            .ok_or(LoadError::BadMetadata(
                "qwen35moe.expert_feed_forward_length",
            ))?;
        u64_to_u32("qwen35moe.expert_feed_forward_length", value)?
    } else {
        0
    };
    let expert_shared_feed_forward_length = if kind == ArchKind::Moe {
        let value = g
            .get_u64(&format!("{p}.expert_shared_feed_forward_length"))
            .ok_or(LoadError::BadMetadata(
                "qwen35moe.expert_shared_feed_forward_length",
            ))?;
        u64_to_u32("qwen35moe.expert_shared_feed_forward_length", value)?
    } else {
        0
    };

    // Vocab: derive from token_embd shape since metadata `vocab_size` may be
    // missing on some converters.
    let vocab_size = g
        .find("token_embd.weight")
        .and_then(|t| t.shape.get(1).copied())
        .ok_or(LoadError::BadMetadata("token_embd.weight"))?;
    let vocab_size = u64_to_u32("token_embd.weight.shape[1]", vocab_size)?;

    ensure_cap(
        "qwen35.block_count - nextn_predict_layers",
        n_layer,
        MAX_ARCH_LAYERS,
    )?;
    ensure_nonzero("qwen35.block_count - nextn_predict_layers", n_layer)?;
    ensure_cap("qwen35.embedding_length", hidden_size, MAX_ARCH_DIM)?;
    ensure_nonzero("qwen35.embedding_length", hidden_size)?;
    ensure_cap(
        "qwen35.feed_forward_length",
        intermediate_size,
        MAX_ARCH_DIM,
    )?;
    if kind == ArchKind::Dense {
        ensure_nonzero("qwen35.feed_forward_length", intermediate_size)?;
    }
    ensure_cap("qwen35.attention.head_count", n_q_heads, MAX_ARCH_DIM)?;
    ensure_nonzero("qwen35.attention.head_count", n_q_heads)?;
    ensure_cap("qwen35.attention.head_count_kv", n_kv_heads, MAX_ARCH_DIM)?;
    ensure_nonzero("qwen35.attention.head_count_kv", n_kv_heads)?;
    ensure_divisible(
        "qwen35.attention.head_count",
        n_q_heads,
        "qwen35.attention.head_count_kv",
        n_kv_heads,
    )?;
    ensure_cap("qwen35.attention.key_length", attn_head_dim, MAX_ARCH_DIM)?;
    ensure_nonzero("qwen35.attention.key_length", attn_head_dim)?;
    ensure_cap(
        "qwen35.full_attention_interval",
        full_attention_interval,
        MAX_ARCH_LAYERS,
    )?;
    ensure_nonzero("qwen35.full_attention_interval", full_attention_interval)?;
    ensure_cap("qwen35.ssm.state_size", ssm_state_size, MAX_ARCH_DIM)?;
    ensure_nonzero("qwen35.ssm.state_size", ssm_state_size)?;
    ensure_cap(
        "qwen35.ssm.time_step_rank",
        ssm_time_step_rank,
        MAX_ARCH_DIM,
    )?;
    ensure_nonzero("qwen35.ssm.time_step_rank", ssm_time_step_rank)?;
    ensure_cap("qwen35.ssm.group_count", ssm_group_count, MAX_ARCH_DIM)?;
    ensure_nonzero("qwen35.ssm.group_count", ssm_group_count)?;
    ensure_cap(
        "qwen35.ssm.conv_kernel",
        ssm_conv_kernel,
        MAX_DFLASH_BLOCK_SIZE,
    )?;
    ensure_nonzero("qwen35.ssm.conv_kernel", ssm_conv_kernel)?;
    ensure_cap("qwen35moe.expert_count", expert_count, MAX_ARCH_DIM)?;
    if kind == ArchKind::Moe {
        ensure_nonzero("qwen35moe.expert_count", expert_count)?;
    }
    ensure_cap(
        "qwen35moe.expert_used_count",
        expert_used_count,
        MAX_ARCH_DIM,
    )?;
    if kind == ArchKind::Moe {
        ensure_nonzero("qwen35moe.expert_used_count", expert_used_count)?;
    }
    if kind == ArchKind::Moe && expert_used_count > expert_count {
        return Err(LoadError::MetadataInconsistent {
            key: "qwen35moe.expert_used_count",
            got: expert_used_count as u64,
            bound_key: "qwen35moe.expert_count",
            bound: expert_count as u64,
        });
    }
    ensure_cap(
        "qwen35moe.expert_feed_forward_length",
        expert_feed_forward_length,
        MAX_ARCH_DIM,
    )?;
    if kind == ArchKind::Moe {
        ensure_nonzero(
            "qwen35moe.expert_feed_forward_length",
            expert_feed_forward_length,
        )?;
    }
    ensure_cap(
        "qwen35moe.expert_shared_feed_forward_length",
        expert_shared_feed_forward_length,
        MAX_ARCH_DIM,
    )?;
    if kind == ArchKind::Moe {
        ensure_nonzero(
            "qwen35moe.expert_shared_feed_forward_length",
            expert_shared_feed_forward_length,
        )?;
    }
    ensure_cap("token_embd.weight.shape[1]", vocab_size, MAX_ARCH_DIM)?;
    ensure_nonzero("token_embd.weight.shape[1]", vocab_size)?;

    let q_dim = checked_mul_dim(n_q_heads as u64, attn_head_dim as u64, "qwen35 q_dim")?;
    if q_dim > MAX_KERNEL_DIM {
        return Err(LoadError::MetadataCap {
            key: "qwen35.attention.q_dim",
            got: q_dim,
            cap: MAX_KERNEL_DIM,
        });
    }
    let kv_dim = checked_mul_dim(n_kv_heads as u64, attn_head_dim as u64, "qwen35 kv_dim")?;
    if kv_dim > MAX_KERNEL_DIM {
        return Err(LoadError::MetadataCap {
            key: "qwen35.attention.kv_dim",
            got: kv_dim,
            cap: MAX_KERNEL_DIM,
        });
    }
    let gdn_v_dim = checked_mul_dim(
        ssm_time_step_rank as u64,
        ssm_state_size as u64,
        "qwen35 gdn_v_dim",
    )?;
    if gdn_v_dim > MAX_KERNEL_DIM {
        return Err(LoadError::MetadataCap {
            key: "qwen35.ssm.v_dim",
            got: gdn_v_dim,
            cap: MAX_KERNEL_DIM,
        });
    }
    let gdn_k_dim = checked_mul_dim(
        ssm_group_count as u64,
        ssm_state_size as u64,
        "qwen35 gdn_k_dim",
    )?;
    if gdn_k_dim > MAX_KERNEL_DIM {
        return Err(LoadError::MetadataCap {
            key: "qwen35.ssm.k_dim",
            got: gdn_k_dim,
            cap: MAX_KERNEL_DIM,
        });
    }
    let gdn_conv_heads = checked_add_dim(
        checked_double_dim(ssm_group_count as u64, "qwen35 2 * gdn_n_k_heads")?,
        ssm_time_step_rank as u64,
        "qwen35 gdn conv heads",
    )?;
    let gdn_conv_dim =
        checked_mul_dim(gdn_conv_heads, ssm_state_size as u64, "qwen35 gdn_conv_dim")?;
    if gdn_conv_dim > MAX_KERNEL_DIM {
        return Err(LoadError::MetadataCap {
            key: "qwen35.ssm.conv_dim",
            got: gdn_conv_dim,
            cap: MAX_KERNEL_DIM,
        });
    }
    let gdn_state_elems =
        checked_mul_dim(gdn_v_dim, ssm_state_size as u64, "qwen35 gdn_state_elems")?;
    if gdn_state_elems > MAX_GDN_STATE_ELEMS {
        return Err(LoadError::MetadataCap {
            key: "qwen35.ssm.state_elems",
            got: gdn_state_elems,
            cap: MAX_GDN_STATE_ELEMS,
        });
    }

    // RoPE.
    let rope_theta = g
        .model
        .metadata()
        .get(&format!("{p}.rope.freq_base"))
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .unwrap_or(10_000_000.0);

    Ok(Arch {
        kind,
        n_layer,
        hidden_size,
        intermediate_size,
        vocab_size,
        full_attention_interval,
        n_q_heads,
        n_kv_heads,
        attn_head_dim,
        rope_theta,
        partial_rotary_factor: 0.25,
        gdn_n_v_heads: ssm_time_step_rank,
        gdn_n_k_heads: ssm_group_count,
        gdn_head_dim: ssm_state_size,
        gdn_conv_kernel: ssm_conv_kernel,
        expert_count,
        expert_used_count,
        expert_feed_forward_length,
        expert_shared_feed_forward_length,
        mtp_n_hidden_layers,
    })
}

/// Convenience: how many GDN-typed elements are stored across all layers,
/// in F32 bytes? Useful for budget calculations.
#[allow(dead_code)]
pub fn gdn_state_bytes(arch: &Arch) -> u64 {
    let n_gdn_layers = (0..arch.n_layer)
        .filter(|&i| arch.layer_kind(i) == LayerKind::GatedDeltaNet)
        .count() as u64;
    n_gdn_layers
        .checked_mul(arch.gdn_n_v_heads as u64)
        .and_then(|v| v.checked_mul(arch.gdn_head_dim as u64))
        .and_then(|v| v.checked_mul(arch.gdn_head_dim as u64))
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>() as u64))
        .unwrap_or(u64::MAX)
}

/// Best-effort summary string for diagnostics.
pub fn summary(model: &Model<'_>) -> String {
    let n_gdn = model
        .blocks
        .iter()
        .filter(|b| matches!(b, Block::Gdn(_)))
        .count();
    let n_attn = model
        .blocks
        .iter()
        .filter(|b| matches!(b, Block::Attn(_)))
        .count();
    let token_embd_dtype = model.token_embd.dtype;
    let lm_head_dtype = model.lm_head.dtype;
    format!(
        "Qwen3.5 family {:?}: {} layers ({n_gdn} GDN + {n_attn} full-attn), \
         hidden={} ffn={} moe=(experts:{} topk:{} routed:{} shared:{}) vocab={} | embed={} lm_head={}{}",
        model.arch.kind,
        model.arch.n_layer,
        model.arch.hidden_size,
        model.arch.intermediate_size,
        model.arch.expert_count,
        model.arch.expert_used_count,
        model.arch.expert_feed_forward_length,
        model.arch.expert_shared_feed_forward_length,
        model.arch.vocab_size,
        token_embd_dtype,
        lm_head_dtype,
        if model.tied_embeddings { " (tied)" } else { "" },
    )
}

// Mark the `LayerKind` variant as referenced so it doesn't trip `dead_code`
// in builds where only one error variant is constructed.
#[allow(dead_code)]
const _: fn() = || {
    let _ = LoadError::LayerKind {
        idx: 0,
        expected: LayerKind::GatedAttention,
    };
    let _ = LoadError::ArchMismatch {
        key: "_",
        got: 0,
        expected: 0,
    };
    let _: GgmlType = GgmlType::F32;
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_narrowing_rejects_u32_overflow() {
        let err = u64_to_u32("test", u64::from(u32::MAX) + 1)
            .expect_err("metadata cast must not truncate");
        assert!(matches!(err, LoadError::MetadataOverflow { .. }));
    }

    #[test]
    fn metadata_dimension_math_rejects_overflow() {
        let err = checked_mul_dim(u64::MAX, 2, "test product")
            .expect_err("dimension multiplication must not wrap");
        assert!(matches!(err, LoadError::DimensionOverflow("test product")));
    }

    #[test]
    fn metadata_divisibility_rejects_bad_gqa_ratio() {
        let err = ensure_divisible("q", 13, "kv", 3)
            .expect_err("non-divisible grouped-query ratio must be rejected");
        assert!(matches!(err, LoadError::MetadataNotDivisible { .. }));
    }

    #[test]
    fn prism_metadata_classifier_identifies_rotated_basis_contract() {
        let metadata =
            BTreeMap::from([("prism.hadamard.version".to_owned(), serde_json::json!(1))]);
        let key = prism_hadamard_metadata_key_in(&metadata).expect("Prism metadata key");
        assert_eq!(key, "prism.hadamard.version");
        let error = LoadError::PrismBasisUnsupported {
            key: key.to_owned(),
        };
        assert!(error.to_string().contains("prism.hadamard.version"));
        assert!(
            error
                .to_string()
                .contains("rotated-basis execution contract")
        );
    }

    #[test]
    fn loads_0_8b_f32() {
        let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
        if !std::path::Path::new(path).exists() {
            return;
        }
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load model");
        eprintln!("[loader] {}", summary(&m));
        assert_eq!(m.arch.n_layer, 24);
        assert_eq!(m.arch.hidden_size, 1024);
        assert_eq!(m.arch.intermediate_size, 3584);
        assert_eq!(m.arch.vocab_size, 248_320);
        // 18 GDN, 6 full-attn for 24 layers in 3:1 pattern.
        let gdn = m
            .blocks
            .iter()
            .filter(|b| matches!(b, Block::Gdn(_)))
            .count();
        let attn = m
            .blocks
            .iter()
            .filter(|b| matches!(b, Block::Attn(_)))
            .count();
        assert_eq!(gdn, 18);
        assert_eq!(attn, 6);
        // Pre-MTP-converter GGUF; no MTP head expected.
        assert!(m.mtp.is_none(), "F32 0.8B GGUF should have no MTP head");
    }

    /// H4.0 smoke test: a 0.8B GGUF freshly converted with the
    /// MTP-aware converter (block_count=25, the trailing block being
    /// the NEXTN/MTP head). The main forward path sees only the 24
    /// base layers; the MTP head is bound separately under `m.mtp`.
    #[test]
    fn loads_0_8b_with_mtp() {
        let path = "/Users/tito/models/h4-smoke-test/Qwen3.5-0.8B/qwen3.5-0.8b.Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load model");
        eprintln!("[loader] {}", summary(&m));
        // block_count=25 in metadata, MTP=1, so main forward sees 24.
        assert_eq!(m.arch.n_layer, 24);
        assert_eq!(m.arch.mtp_n_hidden_layers, 1);
        assert_eq!(m.blocks.len(), 24);
        let gdn = m
            .blocks
            .iter()
            .filter(|b| matches!(b, Block::Gdn(_)))
            .count();
        let attn = m
            .blocks
            .iter()
            .filter(|b| matches!(b, Block::Attn(_)))
            .count();
        assert_eq!(gdn, 18);
        assert_eq!(attn, 6);
        // MTP head must bind successfully.
        let mtp = m.mtp.as_ref().expect("MTP head should be bound");
        assert_eq!(mtp.block_idx, 24);
        // Verify shapes (H=1024 for 0.8B).
        assert_eq!(mtp.eh_proj.shape, vec![2 * 1024, 1024]);
        assert_eq!(mtp.enorm.shape, vec![1024]);
        assert_eq!(mtp.hnorm.shape, vec![1024]);
        assert_eq!(mtp.shared_head_norm.shape, vec![1024]);
        // Inner attn block uses the standard tensor layout.
        // 0.8B: n_q_heads=16, n_kv_heads=4, attn_head_dim=128 → q_dim=2048, kv_dim=512.
        assert_eq!(mtp.attn.q.shape, vec![1024, 2 * 2048]); // gated
        assert_eq!(mtp.attn.k.shape, vec![1024, 512]);
        assert_eq!(mtp.attn.v.shape, vec![1024, 512]);
        assert_eq!(mtp.attn.o.shape, vec![2048, 1024]);
    }

    #[test]
    fn loads_27b_q4_k_m() {
        let path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
        if !std::path::Path::new(path).exists() {
            return;
        }
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load model");
        eprintln!("[loader] {}", summary(&m));
        assert_eq!(m.arch.n_layer, 64);
        assert_eq!(m.arch.hidden_size, 5120);
        assert_eq!(m.arch.intermediate_size, 17408);
        let gdn = m
            .blocks
            .iter()
            .filter(|b| matches!(b, Block::Gdn(_)))
            .count();
        let attn = m
            .blocks
            .iter()
            .filter(|b| matches!(b, Block::Attn(_)))
            .count();
        assert_eq!(gdn, 48);
        assert_eq!(attn, 16);
        // Pre-MTP-converter Q4_K_M; no MTP head. The MTP-aware variant lives
        // at brittlewis12/Qwen3.6-27B-MTP-GGUF (see loads_27b_mtp_q4_k_m).
        assert!(m.mtp.is_none(), "non-MTP 27B GGUF should have no MTP head");
    }

    #[test]
    fn loads_35b_a3b_q4_k_m() {
        let path = crate::test_fixtures::A3B_Q4_K_M.path();
        if !std::path::Path::new(path).exists() {
            return;
        }
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load model");
        eprintln!("[loader-moe] {}", summary(&m));
        assert_eq!(m.arch.kind, ArchKind::Moe);
        assert_eq!(m.arch.n_layer, 40);
        assert_eq!(m.arch.hidden_size, 2048);
        assert_eq!(m.arch.full_attention_interval, 4);
        assert_eq!(m.arch.expert_count, 256);
        assert_eq!(m.arch.expert_used_count, 8);
        assert_eq!(m.arch.expert_feed_forward_length, 512);
        assert_eq!(m.arch.expert_shared_feed_forward_length, 512);
        let gdn = m
            .blocks
            .iter()
            .filter(|b| matches!(b, Block::Gdn(_)))
            .count();
        let attn = m
            .blocks
            .iter()
            .filter(|b| matches!(b, Block::Attn(_)))
            .count();
        assert_eq!(gdn, 30);
        assert_eq!(attn, 10);
        for block in &m.blocks {
            match block {
                Block::Gdn(b) => assert!(b.ffn_moe.is_some()),
                Block::Attn(b) => assert!(b.ffn_moe.is_some()),
            }
        }
        assert!(m.mtp.is_none(), "MoE MTP binding is intentionally deferred");
    }

    /// H5.0 smoke test: load the spiritbuun DFlash drafter alongside
    /// a target Qwen3.6-27B model and verify all dims + tensor shapes
    /// match the GGUF metadata exactly. Skipped if either file is
    /// missing locally.
    #[test]
    fn loads_dflash_drafter_3_6_27b() {
        let target_path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
        let drafter_path = crate::test_fixtures::DFLASH_DRAFT_36_Q8_0.path();
        if !std::path::Path::new(target_path).exists()
            || !std::path::Path::new(drafter_path).exists()
        {
            eprintln!("[dflash-load] skipped — fixtures missing");
            return;
        }
        let target_g = GgufFile::open(target_path).expect("open target");
        let target_m = Model::from_gguf(&target_g).expect("load target");
        let drafter_g = GgufFile::open(drafter_path).expect("open drafter");

        let head = open_dflash_drafter(&drafter_g, &target_m).expect("bind drafter");

        // Config from GGUF (cross-checked against direct hexdump in
        // /tmp + the spiritbuun HF README).
        assert_eq!(head.config.n_layer, 5);
        assert_eq!(head.config.hidden_size, 5120);
        assert_eq!(head.config.hidden_size, target_m.arch.hidden_size);
        assert_eq!(head.config.intermediate_size, 17408);
        assert_eq!(head.config.n_q_heads, 32);
        assert_eq!(head.config.n_kv_heads, 8);
        assert_eq!(head.config.head_dim, 128);
        assert_eq!(head.config.swa_window, 2048);
        assert_eq!(head.config.block_size, 16);
        assert_eq!(head.config.mask_token_id, 248070);
        assert_eq!(head.config.n_target_features_layers, 5);
        assert_eq!(head.target_layer_ids, vec![1, 16, 31, 46, 61]);
        assert_eq!(head.layers.len(), 5);

        // Per-layer SWA pattern: [T, T, T, T, F].
        let swa_flags: Vec<bool> = head.layers.iter().map(|l| l.is_swa).collect();
        assert_eq!(swa_flags, vec![true, true, true, true, false]);

        // Tensor shapes (top-level adornments).
        assert_eq!(head.fc.shape, vec![25600, 5120]); // K · H_target = 5 · 5120
        assert_eq!(head.hidden_norm.shape, vec![5120]);
        assert_eq!(head.output_norm.shape, vec![5120]);

        // Layer 0: spot-check Q is NOT gated (shape [H, n_q · head_dim],
        // not [H, 2 · n_q · head_dim]).
        let l0 = &head.layers[0];
        assert_eq!(l0.q.shape, vec![5120, 4096]); // 32 · 128 = 4096
        assert_eq!(l0.k.shape, vec![5120, 1024]); // 8 · 128 = 1024
        assert_eq!(l0.v.shape, vec![5120, 1024]);
        assert_eq!(l0.o.shape, vec![4096, 5120]);
        assert_eq!(l0.ffn_gate.shape, vec![5120, 17408]);
        assert_eq!(l0.ffn_down.shape, vec![17408, 5120]);
    }

    #[test]
    #[ignore = "requires local Qwen3.8 target and Q4 DFlash2 GGUF fixtures"]
    fn loads_dflash2_q4_selector_codebooks() {
        let target_path = "/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf";
        let drafter_path = "/Users/tito/models/incoai-dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf";
        assert!(
            std::path::Path::new(target_path).exists(),
            "missing target fixture"
        );
        assert!(
            std::path::Path::new(drafter_path).exists(),
            "missing drafter fixture"
        );

        let target_g = GgufFile::open(target_path).expect("open target");
        let target_m = Model::from_gguf(&target_g).expect("load target");
        let drafter_g = GgufFile::open(drafter_path).expect("open drafter");
        let head = open_dflash_drafter(&drafter_g, &target_m).expect("bind drafter");
        let selector = head.selector.as_ref().expect("DFlash2 selector");

        assert_eq!(head.config.block_size, 8);
        assert_eq!(head.config.selector_rank, 256);
        assert_eq!(head.config.selector_top_k, 16);
        assert_eq!(selector.predecessor.dtype, GgmlType::Q4_K);
        assert_eq!(selector.successor.dtype, GgmlType::Q4_K);
        assert_eq!(selector.hidden.dtype, GgmlType::Q4_K);
        assert_eq!(selector.predecessor.shape, vec![256, 248320]);
        assert_eq!(selector.successor.shape, vec![256, 248320]);
        assert_eq!(selector.hidden.shape, vec![5120, 256]);
    }

    /// H4.0 27B smoke test: validates the loader bind on the
    /// MTP-aware Q4_K_M GGUF at brittlewis12/Qwen3.6-27B-MTP-GGUF.
    /// Skipped if the file isn't present locally.
    #[test]
    fn loads_27b_mtp_q4_k_m() {
        let path = "/Users/tito/models/Qwen3.6-27B-MTP-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load model");
        eprintln!("[loader] {}", summary(&m));
        // 64 base layers + 1 MTP block at index 64.
        assert_eq!(m.arch.n_layer, 64);
        assert_eq!(m.arch.mtp_n_hidden_layers, 1);
        assert_eq!(m.blocks.len(), 64);
        let mtp = m.mtp.as_ref().expect("MTP head should be bound");
        assert_eq!(mtp.block_idx, 64);
        // 27B: H=5120, F=17408, n_q=24, n_kv=4, head_dim=256
        // → q_dim=24*256=6144, kv_dim=4*256=1024.
        assert_eq!(mtp.eh_proj.shape, vec![2 * 5120, 5120]);
        assert_eq!(mtp.enorm.shape, vec![5120]);
        assert_eq!(mtp.hnorm.shape, vec![5120]);
        assert_eq!(mtp.shared_head_norm.shape, vec![5120]);
        assert_eq!(mtp.attn.q.shape, vec![5120, 2 * 6144]);
        assert_eq!(mtp.attn.k.shape, vec![5120, 1024]);
        assert_eq!(mtp.attn.v.shape, vec![5120, 1024]);
        assert_eq!(mtp.attn.o.shape, vec![6144, 5120]);
    }
}
