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
}

/// DFlash drafter head. Loaded from a separate GGUF (e.g.
/// `spiritbuun/Qwen3.6-27B-DFlash-GGUF`) and bound to a pre-existing
/// target [`Model`]. Shares `tok_embd` and `output` (lm_head) with
/// the target — the drafter GGUF doesn't carry its own.
pub struct DFlashHead<'a> {
    pub config: DFlashConfig,
    /// K target layer indices whose hiddens get fused.
    pub target_layer_ids: Vec<u32>,
    /// `dflash_fc.weight`, shape `[K · H_target, H_drafter]`.
    pub fc: &'a TensorDesc,
    /// `dflash_hidden_norm.weight`, shape `[H_drafter]`.
    pub hidden_norm: &'a TensorDesc,
    /// `output_norm.weight`, shape `[H_drafter]`. The drafter's own final
    /// RMSNorm before the (target's) lm_head.
    pub output_norm: &'a TensorDesc,
    pub layers: Vec<DFlashLayer<'a>>,
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
        if arch_name != "qwen35" && arch_name != "qwen35moe" {
            return Err(LoadError::UnsupportedArch(Some(arch_name.to_string())));
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
                    let qkv_out = (2 * arch.gdn_n_k_heads + arch.gdn_n_v_heads) * arch.gdn_head_dim;
                    let z_out = arch.gdn_n_v_heads * arch.gdn_head_dim;
                    let in_proj_qkv = need(g, &format!("blk.{i}.attn_qkv.weight"))?;
                    check_shape(in_proj_qkv, &[arch.hidden_size as u64, qkv_out as u64])?;
                    let in_proj_z = need(g, &format!("blk.{i}.attn_gate.weight"))?;
                    check_shape(in_proj_z, &[arch.hidden_size as u64, z_out as u64])?;
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
                        (2 * arch.gdn_n_k_heads + arch.gdn_n_v_heads) * arch.gdn_head_dim;
                    check_shape(conv1d, &[arch.gdn_conv_kernel as u64, conv_dim as u64])?;
                    let norm = need(g, &format!("blk.{i}.ssm_norm.weight"))?;
                    check_shape(norm, &[arch.gdn_head_dim as u64])?;
                    let out_proj = need(g, &format!("blk.{i}.ssm_out.weight"))?;
                    check_shape(
                        out_proj,
                        &[
                            (arch.gdn_n_v_heads * arch.gdn_head_dim) as u64,
                            arch.hidden_size as u64,
                        ],
                    )?;
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
                    let q_dim = arch.n_q_heads * arch.attn_head_dim;
                    let kv_dim = arch.n_kv_heads * arch.attn_head_dim;
                    // Gated attention: q_proj outputs 2× the heads' worth — first
                    // half is Q, second half is the sigmoid output gate. So the
                    // weight matrix is [hidden, 2*q_dim].
                    let q = need(g, &format!("blk.{i}.attn_q.weight"))?;
                    check_shape(q, &[arch.hidden_size as u64, 2 * q_dim as u64])?;
                    let k = need(g, &format!("blk.{i}.attn_k.weight"))?;
                    check_shape(k, &[arch.hidden_size as u64, kv_dim as u64])?;
                    let v = need(g, &format!("blk.{i}.attn_v.weight"))?;
                    check_shape(v, &[arch.hidden_size as u64, kv_dim as u64])?;
                    let o = need(g, &format!("blk.{i}.attn_output.weight"))?;
                    check_shape(o, &[q_dim as u64, arch.hidden_size as u64])?;
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
        let mtp = if arch.kind == ArchKind::Dense {
            bind_mtp_head(g, &arch)?
        } else {
            None
        };

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
        &[2 * arch.hidden_size as u64, arch.hidden_size as u64],
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
    let q_dim = arch.n_q_heads * arch.attn_head_dim;
    let kv_dim = arch.n_kv_heads * arch.attn_head_dim;
    let q = need(g, &format!("blk.{i}.attn_q.weight"))?;
    // Gated attention: q_proj outputs 2× q_dim (Q + sigmoid gate).
    check_shape(q, &[arch.hidden_size as u64, 2 * q_dim as u64])?;
    let k = need(g, &format!("blk.{i}.attn_k.weight"))?;
    check_shape(k, &[arch.hidden_size as u64, kv_dim as u64])?;
    let v = need(g, &format!("blk.{i}.attn_v.weight"))?;
    check_shape(v, &[arch.hidden_size as u64, kv_dim as u64])?;
    let o = need(g, &format!("blk.{i}.attn_output.weight"))?;
    check_shape(o, &[q_dim as u64, arch.hidden_size as u64])?;
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
            ffn_moe: None,
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
    let arch_str = drafter_gguf.architecture();
    if arch_str.as_deref() != Some("dflash-draft") {
        return Err(LoadError::UnsupportedArch(arch_str));
    }

    // -------- Read drafter config from KV metadata --------
    let n_layer = drafter_gguf
        .get_u64("dflash-draft.block_count")
        .ok_or(LoadError::BadMetadata("dflash-draft.block_count"))? as u32;
    let hidden_size = drafter_gguf
        .get_u64("dflash-draft.embedding_length")
        .ok_or(LoadError::BadMetadata("dflash-draft.embedding_length"))?
        as u32;
    let intermediate_size = drafter_gguf
        .get_u64("dflash-draft.feed_forward_length")
        .ok_or(LoadError::BadMetadata("dflash-draft.feed_forward_length"))?
        as u32;
    let n_q_heads = drafter_gguf
        .get_u64("dflash-draft.attention.head_count")
        .ok_or(LoadError::BadMetadata("dflash-draft.attention.head_count"))?
        as u32;
    let n_kv_heads = drafter_gguf
        .get_u64("dflash-draft.attention.head_count_kv")
        .ok_or(LoadError::BadMetadata(
            "dflash-draft.attention.head_count_kv",
        ))? as u32;
    let head_dim = drafter_gguf
        .get_u64("dflash-draft.attention.key_length")
        .ok_or(LoadError::BadMetadata("dflash-draft.attention.key_length"))?
        as u32;
    let rope_theta = drafter_gguf
        .get_f32("dflash-draft.rope.freq_base")
        .unwrap_or(10_000_000.0);
    let swa_window = drafter_gguf
        .get_u64("dflash-draft.attention.sliding_window")
        .unwrap_or(0) as u32;
    let block_size = drafter_gguf
        .get_u64("dflash-draft.dflash.block_size")
        .ok_or(LoadError::BadMetadata("dflash-draft.dflash.block_size"))?
        as u32;
    let mask_token_id = drafter_gguf
        .get_u64("dflash-draft.dflash.mask_token_id")
        .ok_or(LoadError::BadMetadata("dflash-draft.dflash.mask_token_id"))?
        as i32;
    let target_layer_ids: Vec<u32> = drafter_gguf
        .get_u64_array("dflash-draft.dflash.target_layer_ids")
        .ok_or(LoadError::BadMetadata(
            "dflash-draft.dflash.target_layer_ids",
        ))?
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let n_target_features = drafter_gguf
        .get_u64("dflash-draft.dflash.n_target_features")
        .ok_or(LoadError::BadMetadata(
            "dflash-draft.dflash.n_target_features",
        ))? as u32;
    let swa_pattern: Vec<bool> = drafter_gguf
        .get_bool_array("dflash-draft.attention.sliding_window_pattern")
        .ok_or(LoadError::BadMetadata(
            "dflash-draft.attention.sliding_window_pattern",
        ))?;

    // -------- Compatibility validation --------
    let target_h = target_model.arch.hidden_size;
    if hidden_size != target_h {
        return Err(LoadError::BadMetadata(
            "dflash-draft.embedding_length must equal target's hidden_size \
             (drafter shares target's tok_embd / lm_head; H mismatch breaks composition)",
        ));
    }
    let expected_n_target_features = (target_layer_ids.len() as u32) * target_h;
    if n_target_features != expected_n_target_features {
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
    if swa_pattern.len() as u32 != n_layer {
        return Err(LoadError::BadMetadata(
            "dflash-draft.attention.sliding_window_pattern length must equal block_count",
        ));
    }

    // -------- Top-level adornment tensors --------
    let fc = need(drafter_gguf, "dflash_fc.weight")?;
    check_shape(fc, &[n_target_features as u64, hidden_size as u64])?;
    let hidden_norm = need(drafter_gguf, "dflash_hidden_norm.weight")?;
    check_shape(hidden_norm, &[hidden_size as u64])?;
    let output_norm = need(drafter_gguf, "output_norm.weight")?;
    check_shape(output_norm, &[hidden_size as u64])?;

    // -------- Per-layer tensors --------
    let q_dim = (n_q_heads * head_dim) as u64;
    let kv_dim = (n_kv_heads * head_dim) as u64;
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
        let post_attention_norm =
            need(drafter_gguf, &format!("blk.{i}.post_attention_norm.weight"))?;
        check_shape(post_attention_norm, &[h])?;
        let ffn_gate = need(drafter_gguf, &format!("blk.{i}.ffn_gate.weight"))?;
        check_shape(ffn_gate, &[h, f])?;
        let ffn_up = need(drafter_gguf, &format!("blk.{i}.ffn_up.weight"))?;
        check_shape(ffn_up, &[h, f])?;
        let ffn_down = need(drafter_gguf, &format!("blk.{i}.ffn_down.weight"))?;
        check_shape(ffn_down, &[f, h])?;
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
            n_target_features_layers: target_layer_ids.len() as u32,
        },
        target_layer_ids,
        fc,
        hidden_norm,
        output_norm,
        layers,
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
    let block_count = g
        .get_u64(&format!("{p}.block_count"))
        .ok_or(LoadError::BadMetadata("qwen35.block_count"))? as u32;
    let mtp_n_hidden_layers = g.get_u64(&format!("{p}.nextn_predict_layers")).unwrap_or(0) as u32;
    let n_layer = block_count.saturating_sub(mtp_n_hidden_layers);
    let hidden_size = g
        .get_u64(&format!("{p}.embedding_length"))
        .ok_or(LoadError::BadMetadata("qwen35.embedding_length"))? as u32;
    let intermediate_size = if kind == ArchKind::Dense {
        g.get_u64(&format!("{p}.feed_forward_length"))
            .ok_or(LoadError::BadMetadata("qwen35.feed_forward_length"))? as u32
    } else {
        0
    };
    let n_q_heads = g
        .get_u64(&format!("{p}.attention.head_count"))
        .ok_or(LoadError::BadMetadata("qwen35.attention.head_count"))? as u32;
    let n_kv_heads =
        g.get_u64(&format!("{p}.attention.head_count_kv"))
            .ok_or(LoadError::BadMetadata("qwen35.attention.head_count_kv"))? as u32;
    let attn_head_dim =
        g.get_u64(&format!("{p}.attention.key_length"))
            .ok_or(LoadError::BadMetadata("qwen35.attention.key_length"))? as u32;
    let full_attention_interval = g
        .get_u64(&format!("{p}.full_attention_interval"))
        .unwrap_or(4) as u32;

    // GDN dims.
    // ssm.inner_size = num_v_heads * head_dim
    // ssm.state_size = head_dim
    // ssm.time_step_rank = num_v_heads
    // ssm.group_count = num_k_heads
    // ssm.conv_kernel = conv kernel size
    let ssm_state_size = g
        .get_u64(&format!("{p}.ssm.state_size"))
        .ok_or(LoadError::BadMetadata("qwen35.ssm.state_size"))? as u32;
    let ssm_time_step_rank =
        g.get_u64(&format!("{p}.ssm.time_step_rank"))
            .ok_or(LoadError::BadMetadata("qwen35.ssm.time_step_rank"))? as u32;
    let ssm_group_count = g
        .get_u64(&format!("{p}.ssm.group_count"))
        .ok_or(LoadError::BadMetadata("qwen35.ssm.group_count"))? as u32;
    let ssm_conv_kernel = g
        .get_u64(&format!("{p}.ssm.conv_kernel"))
        .ok_or(LoadError::BadMetadata("qwen35.ssm.conv_kernel"))? as u32;

    let expert_count = if kind == ArchKind::Moe {
        g.get_u64(&format!("{p}.expert_count"))
            .ok_or(LoadError::BadMetadata("qwen35moe.expert_count"))? as u32
    } else {
        0
    };
    let expert_used_count = if kind == ArchKind::Moe {
        g.get_u64(&format!("{p}.expert_used_count"))
            .ok_or(LoadError::BadMetadata("qwen35moe.expert_used_count"))? as u32
    } else {
        0
    };
    let expert_feed_forward_length = if kind == ArchKind::Moe {
        g.get_u64(&format!("{p}.expert_feed_forward_length"))
            .ok_or(LoadError::BadMetadata(
                "qwen35moe.expert_feed_forward_length",
            ))? as u32
    } else {
        0
    };
    let expert_shared_feed_forward_length = if kind == ArchKind::Moe {
        g.get_u64(&format!("{p}.expert_shared_feed_forward_length"))
            .ok_or(LoadError::BadMetadata(
                "qwen35moe.expert_shared_feed_forward_length",
            ))? as u32
    } else {
        0
    };

    // Vocab: derive from token_embd shape since metadata `vocab_size` may be
    // missing on some converters.
    let vocab_size = g
        .find("token_embd.weight")
        .and_then(|t| t.shape.get(1).copied())
        .ok_or(LoadError::BadMetadata("token_embd.weight"))? as u32;

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
        * arch.gdn_n_v_heads as u64
        * arch.gdn_head_dim as u64
        * arch.gdn_head_dim as u64
        * std::mem::size_of::<f32>() as u64
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
    fn loads_0_8b_f32() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
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
        let path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
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
        let path = "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
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
        let target_path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        let drafter_path = "/Users/tito/models/spiritbuun-dflash/dflash-draft-3.6-q8_0.gguf";
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
