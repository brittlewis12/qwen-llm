//! Qwen 3.5 / 3.6 architecture description.
//!
//! Pinned facts (sources: HF config.json for `Qwen/Qwen3.6-27B`,
//! transformers `qwen3_next/modeling_qwen3_next.py` v4.57, llama.cpp
//! `src/models/qwen35.cpp`, vLLM PR #37975, exo PR #1644):
//!
//! * 64 layers, hidden 5120, FFN intermediate 17408, vocab 248,320, no tied
//!   word embeddings.
//! * Layer pattern: `[linear×3, full×1] × 16` → 48 GDN + 16 full-attn.
//! * Full-attn ("Gated Attention"): 24 Q heads / 4 KV heads (GQA 6:1),
//!   `head_dim=256`, partial RoPE (64 of 256 dims), `rope_theta=10_000_000`,
//!   MRoPE sections `[11,11,10]`, qk_norm, `attn_output_gate=true` (q_proj
//!   outputs 2× — second half is sigmoid → multiplied with attn output
//!   before o_proj), no sliding window.
//! * GDN: 48 V heads, 16 K heads (V/K=3), `head_dim=128`, conv1d kernel=4
//!   depthwise with SiLU, scalar gate per head, β = sigmoid(b),
//!   g = -exp(A_log) * softplus(a + dt_bias), L2-norm Q/K inside the kernel
//!   (eps=1e-6), `mamba_ssm_dtype=float32` (state in fp32 — non-negotiable).
//! * FFN: SwiGLU (gate_proj, up_proj, down_proj). No MoE on 27B.
//! * MTP: 1 hidden layer, weights present in checkpoint under `mtp.*`.
//! * GDN projections in Qwen3.5/3.6 are **separated**
//!   (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`), NOT fused
//!   like Qwen3-Next-80B.

/// Static architecture description for a Qwen3.5/3.6 dense variant.
///
/// Smaller siblings (0.8B / 2B / 4B / 9B) have identical kernel ops modulo
/// these counts; 27B is the headline target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arch {
    pub n_layer: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,

    // attention layers
    pub n_q_heads: u32,
    pub n_kv_heads: u32,
    pub attn_head_dim: u32,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32, // 0.25 → rotary dim = head_dim/4

    // GDN layers
    pub gdn_n_v_heads: u32,
    pub gdn_n_k_heads: u32,
    pub gdn_head_dim: u32,
    pub gdn_conv_kernel: u32,

    // MTP
    pub mtp_n_hidden_layers: u32,
}

/// Layer kind in the `[GDN, GDN, GDN, full] × 16` pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    GatedDeltaNet,
    GatedAttention,
}

impl Arch {
    /// Layer pattern is `[GDN×3, full×1]` repeating; full-attn at indices
    /// 3, 7, 11, …
    #[inline]
    pub fn layer_kind(&self, idx: u32) -> LayerKind {
        if idx % 4 == 3 {
            LayerKind::GatedAttention
        } else {
            LayerKind::GatedDeltaNet
        }
    }
}

/// Qwen3.5-27B / Qwen3.6-27B (dense flagship, hybrid GDN + GQA).
pub const QWEN3_27B: Arch = Arch {
    n_layer: 64,
    hidden_size: 5120,
    intermediate_size: 17408,
    vocab_size: 248_320,
    n_q_heads: 24,
    n_kv_heads: 4,
    attn_head_dim: 256,
    rope_theta: 10_000_000.0,
    partial_rotary_factor: 0.25,
    gdn_n_v_heads: 48,
    gdn_n_k_heads: 16,
    gdn_head_dim: 128,
    gdn_conv_kernel: 4,
    mtp_n_hidden_layers: 1,
};

/// Qwen3.5-0.8B — the F32 numerical oracle target.
///
/// Numbers confirmed against the shipping `Qwen3.5-0.8B.F32.gguf` GGUF
/// metadata: block_count=24, embedding_length=1024, feed_forward_length=3584,
/// attention.head_count=8, attention.head_count_kv=2, head_dim=256,
/// ssm.inner_size=2048, ssm.state_size=128, ssm.time_step_rank=16,
/// ssm.group_count=16. (`time_step_rank` = num_v_heads.)
pub const QWEN3_0_8B: Arch = Arch {
    n_layer: 24,
    hidden_size: 1024,
    intermediate_size: 3584,
    vocab_size: 248_320,
    n_q_heads: 8,
    n_kv_heads: 2,
    attn_head_dim: 256,
    rope_theta: 10_000_000.0,
    partial_rotary_factor: 0.25,
    gdn_n_v_heads: 16,
    gdn_n_k_heads: 16,
    gdn_head_dim: 128,
    gdn_conv_kernel: 4,
    mtp_n_hidden_layers: 1,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_pattern_27b() {
        let a = QWEN3_27B;
        let mut full_count = 0;
        let mut gdn_count = 0;
        for i in 0..a.n_layer {
            match a.layer_kind(i) {
                LayerKind::GatedAttention => full_count += 1,
                LayerKind::GatedDeltaNet => gdn_count += 1,
            }
        }
        assert_eq!(full_count, 16);
        assert_eq!(gdn_count, 48);
        assert_eq!(a.layer_kind(0), LayerKind::GatedDeltaNet);
        assert_eq!(a.layer_kind(3), LayerKind::GatedAttention);
        assert_eq!(a.layer_kind(63), LayerKind::GatedAttention);
    }

    #[test]
    fn gqa_ratio_27b() {
        let a = QWEN3_27B;
        assert_eq!(a.n_q_heads / a.n_kv_heads, 6);
        assert_eq!(a.gdn_n_v_heads / a.gdn_n_k_heads, 3);
    }

    #[test]
    fn layer_pattern_0_8b() {
        let a = QWEN3_0_8B;
        let mut full = 0;
        let mut gdn = 0;
        for i in 0..a.n_layer {
            match a.layer_kind(i) {
                LayerKind::GatedAttention => full += 1,
                LayerKind::GatedDeltaNet => gdn += 1,
            }
        }
        assert_eq!(full, 6);
        assert_eq!(gdn, 18);
    }

    #[test]
    fn arch_matches_gguf_0_8b() {
        // Sanity-check: numbers in QWEN3_0_8B match the live GGUF metadata.
        // Skipped if the file isn't present on this box.
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let g = crate::gguf::GgufFile::open(path).expect("open");
        let a = QWEN3_0_8B;
        assert_eq!(g.get_u64("qwen35.block_count"), Some(a.n_layer as u64));
        assert_eq!(
            g.get_u64("qwen35.embedding_length"),
            Some(a.hidden_size as u64)
        );
        assert_eq!(
            g.get_u64("qwen35.feed_forward_length"),
            Some(a.intermediate_size as u64)
        );
        assert_eq!(
            g.get_u64("qwen35.attention.head_count"),
            Some(a.n_q_heads as u64)
        );
        assert_eq!(
            g.get_u64("qwen35.attention.head_count_kv"),
            Some(a.n_kv_heads as u64)
        );
    }
}
