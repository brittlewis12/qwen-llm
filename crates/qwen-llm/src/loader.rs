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
use crate::model::{Arch, LayerKind};
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
}

#[derive(Clone)]
pub enum Block<'a> {
    Gdn(GdnBlock<'a>),
    Attn(AttnBlock<'a>),
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
}

impl<'a> Model<'a> {
    pub fn from_gguf(g: &'a GgufFile) -> Result<Self, LoadError> {
        let arch_str = g.architecture();
        if arch_str.as_deref() != Some("qwen35") {
            return Err(LoadError::UnsupportedArch(arch_str));
        }

        let arch = build_arch_from_metadata(g)?;

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
                    })
                }
            };
            blocks.push(block);
        }

        Ok(Self {
            arch,
            token_embd,
            output_norm,
            lm_head,
            blocks,
            tied_embeddings,
        })
    }
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

/// Build an [`Arch`] from the GGUF's `qwen35.*` metadata keys. Cross-checks
/// against the known constants in [`crate::model`] are the caller's job.
fn build_arch_from_metadata(g: &GgufFile) -> Result<Arch, LoadError> {
    // GGUF's `block_count` includes any trailing MTP/NEXTN predict layers
    // (per the patched converter that preserves them). The main forward
    // path iterates only the base layers, so subtract any MTP layers from
    // n_layer here. Older GGUFs (pre-MTP-converter) don't have the
    // nextn_predict_layers key, in which case we default to 0 and the
    // subtraction is a no-op (preserving the prior behavior).
    let block_count = g
        .get_u64("qwen35.block_count")
        .ok_or(LoadError::BadMetadata("qwen35.block_count"))? as u32;
    let mtp_n_hidden_layers = g.get_u64("qwen35.nextn_predict_layers").unwrap_or(0) as u32;
    let n_layer = block_count.saturating_sub(mtp_n_hidden_layers);
    let hidden_size = g
        .get_u64("qwen35.embedding_length")
        .ok_or(LoadError::BadMetadata("qwen35.embedding_length"))? as u32;
    let intermediate_size =
        g.get_u64("qwen35.feed_forward_length")
            .ok_or(LoadError::BadMetadata("qwen35.feed_forward_length"))? as u32;
    let n_q_heads = g
        .get_u64("qwen35.attention.head_count")
        .ok_or(LoadError::BadMetadata("qwen35.attention.head_count"))? as u32;
    let n_kv_heads =
        g.get_u64("qwen35.attention.head_count_kv")
            .ok_or(LoadError::BadMetadata("qwen35.attention.head_count_kv"))? as u32;
    let attn_head_dim =
        g.get_u64("qwen35.attention.key_length")
            .ok_or(LoadError::BadMetadata("qwen35.attention.key_length"))? as u32;

    // GDN dims.
    // ssm.inner_size = num_v_heads * head_dim
    // ssm.state_size = head_dim
    // ssm.time_step_rank = num_v_heads
    // ssm.group_count = num_k_heads
    // ssm.conv_kernel = conv kernel size
    let ssm_state_size = g
        .get_u64("qwen35.ssm.state_size")
        .ok_or(LoadError::BadMetadata("qwen35.ssm.state_size"))? as u32;
    let ssm_time_step_rank =
        g.get_u64("qwen35.ssm.time_step_rank")
            .ok_or(LoadError::BadMetadata("qwen35.ssm.time_step_rank"))? as u32;
    let ssm_group_count = g
        .get_u64("qwen35.ssm.group_count")
        .ok_or(LoadError::BadMetadata("qwen35.ssm.group_count"))? as u32;
    let ssm_conv_kernel = g
        .get_u64("qwen35.ssm.conv_kernel")
        .ok_or(LoadError::BadMetadata("qwen35.ssm.conv_kernel"))? as u32;

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
        .get("qwen35.rope.freq_base")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .unwrap_or(10_000_000.0);

    Ok(Arch {
        n_layer,
        hidden_size,
        intermediate_size,
        vocab_size,
        n_q_heads,
        n_kv_heads,
        attn_head_dim,
        rope_theta,
        partial_rotary_factor: 0.25,
        gdn_n_v_heads: ssm_time_step_rank,
        gdn_n_k_heads: ssm_group_count,
        gdn_head_dim: ssm_state_size,
        gdn_conv_kernel: ssm_conv_kernel,
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
        "Qwen3.5 family: {} layers ({n_gdn} GDN + {n_attn} full-attn), \
         hidden={} ffn={} vocab={} | embed={} lm_head={}{}",
        model.arch.n_layer,
        model.arch.hidden_size,
        model.arch.intermediate_size,
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
    }

    /// H4 smoke test: a 0.8B GGUF freshly converted with the
    /// MTP-aware converter (block_count=25, the trailing block being
    /// the NEXTN/MTP head). The main forward path must continue to
    /// see only the 24 base layers; the MTP block is ignored for now.
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
        // Confirm the MTP block (blk.24) tensors are present in the GGUF
        // (we just don't load them into Block enum yet).
        assert!(g
            .tensors
            .iter()
            .any(|t| t.name == "blk.24.nextn.eh_proj.weight"));
        assert!(g
            .tensors
            .iter()
            .any(|t| t.name == "blk.24.nextn.enorm.weight"));
        assert!(g
            .tensors
            .iter()
            .any(|t| t.name == "blk.24.nextn.hnorm.weight"));
        assert!(g
            .tensors
            .iter()
            .any(|t| t.name == "blk.24.nextn.shared_head_norm.weight"));
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
    }
}
