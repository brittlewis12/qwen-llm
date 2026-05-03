//! CPU reference forward pass for Qwen3.5/3.6.
//!
//! Goal: bit-exact (or near-bit-exact, mod fp non-associativity) match
//! against `llama-cli` on `Qwen3.5-0.8B.F32.gguf`. This is the oracle
//! against which every Metal kernel is later validated.
//!
//! Performance is **explicitly not a concern here**. Naive triple-loop
//! matmul, allocate-on-every-step. The point is correctness.
//!
//! Reference: `~/code/llama.cpp/src/models/qwen35.cpp`. The structure of
//! [`Forward::single_token`] mirrors `llm_build_qwen35::llm_build_qwen35`
//! exactly.

use crate::codec::dequant_to_f32;
use crate::gguf::GgufFile;
use crate::loader::{AttnBlock, Block, GdnBlock, Model};
use crate::tensor::TensorDesc;

#[derive(Debug, thiserror::Error)]
pub enum ForwardError {
    #[error("codec: {0}")]
    Codec(#[from] crate::codec::CodecError),
    #[error("token id {0} out of range (vocab {1})")]
    BadToken(i32, u32),
}

/// Mutable per-sequence state: the recurrent buffers that live across
/// decode steps. Conv buffer is the last `kernel-1` qkv-mixed values per
/// channel; SSM state is the GDN per-head outer-product matrix.
pub struct GdnState {
    /// One conv buffer per GDN-typed layer, contiguous in `[conv_dim, kernel-1]`
    /// row-major layout (channel-fastest). Length = n_gdn_layers.
    pub conv: Vec<Vec<f32>>,
    /// One SSM state per GDN-typed layer, contiguous in
    /// `[head_v_dim, head_v_dim, num_v_heads]` layout. Length = n_gdn_layers.
    /// Dtype is f32 (non-negotiable per the architecture spec).
    pub ssm: Vec<Vec<f32>>,
}

impl GdnState {
    /// Allocate zero-initialized state for one fresh sequence.
    pub fn fresh(model: &Model<'_>) -> Self {
        let arch = &model.arch;
        let n_gdn = model
            .blocks
            .iter()
            .filter(|b| matches!(b, Block::Gdn(_)))
            .count();
        let conv_dim = (2 * arch.gdn_n_k_heads + arch.gdn_n_v_heads) * arch.gdn_head_dim;
        let ssm_per_layer =
            arch.gdn_head_dim as usize * arch.gdn_head_dim as usize * arch.gdn_n_v_heads as usize;
        Self {
            conv: (0..n_gdn)
                .map(|_| vec![0.0f32; conv_dim as usize * (arch.gdn_conv_kernel as usize - 1)])
                .collect(),
            ssm: (0..n_gdn).map(|_| vec![0.0f32; ssm_per_layer]).collect(),
        }
    }
}

pub struct Forward<'a> {
    pub gguf: &'a GgufFile,
    pub model: &'a Model<'a>,
}

impl<'a> Forward<'a> {
    pub fn new(gguf: &'a GgufFile, model: &'a Model<'a>) -> Self {
        Self { gguf, model }
    }

    /// Run a single token through the model and return logits over the vocab.
    ///
    /// `token_id`: input token (0..vocab_size).
    /// `position`: 0-indexed position in the sequence (used by RoPE on
    /// full-attention layers; ignored by GDN layers).
    /// `state`: GDN recurrent state, mutated in place. Pass a fresh
    /// [`GdnState`] for the first token of a new sequence.
    /// `kv_cache`: KV cache for full-attention layers (per-layer, per-position).
    pub fn single_token(
        &self,
        token_id: i32,
        position: u32,
        state: &mut GdnState,
        kv_cache: &mut KvCache,
    ) -> Result<Vec<f32>, ForwardError> {
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(ForwardError::BadToken(token_id, arch.vocab_size));
        }

        // (1) Embedding lookup. token_embd has shape [hidden, vocab].
        let mut x = self.embed_token(token_id as u32)?;

        // (2) Per-block.
        let mut gdn_idx: usize = 0;
        for (il, block) in self.model.blocks.iter().enumerate() {
            // Pre-attention RMS norm.
            let attn_norm_w = self.dequant(self.block_attn_norm(block))?;
            let mut cur = rms_norm(&x, &attn_norm_w, RMS_EPS);

            // Attention (or GDN).
            cur = match block {
                Block::Gdn(gb) => {
                    let conv = &mut state.conv[gdn_idx];
                    let ssm = &mut state.ssm[gdn_idx];
                    gdn_idx += 1;
                    self.gdn_step(gb, &cur, conv, ssm)?
                }
                Block::Attn(ab) => self.attn_step(il, ab, &cur, position, kv_cache)?,
            };

            // Residual #1 (around mixer).
            for (xi, ci) in x.iter_mut().zip(cur.iter()) {
                *xi += *ci;
            }

            // Pre-FFN RMS norm.
            let post_norm_w = self.dequant(self.block_post_norm(block))?;
            let post = rms_norm(&x, &post_norm_w, RMS_EPS);

            // SwiGLU FFN.
            let ffn_out = self.ffn(block, &post)?;

            // Residual #2 (around FFN).
            for (xi, fi) in x.iter_mut().zip(ffn_out.iter()) {
                *xi += *fi;
            }
        }

        // (3) Final RMS norm.
        let on_w = self.dequant(self.model.output_norm)?;
        let normed = rms_norm(&x, &on_w, RMS_EPS);

        // (4) LM head: [hidden, vocab] -> [vocab]. mat_vec.
        let lm = self.dequant(self.model.lm_head)?;
        let logits = mat_vec(
            &lm,
            arch.hidden_size as usize,
            arch.vocab_size as usize,
            &normed,
        );
        Ok(logits)
    }

    // -------------- helpers --------------

    fn embed_token(&self, id: u32) -> Result<Vec<f32>, ForwardError> {
        let arch = &self.model.arch;
        let embd = self.dequant(self.model.token_embd)?;
        // token_embd has shape [hidden, vocab]; row `id` is at offset
        // id * hidden (column-major in GGUF terms is row-major as we read).
        // GGUF stores matrices with the "in" axis fastest, so for a
        // [hidden, vocab] tensor the `id`-th token's row is at
        // bytes [id * hidden .. (id+1) * hidden]. Confirmed by inspection
        // of llama.cpp's get_rows behavior on this tensor.
        let h = arch.hidden_size as usize;
        Ok(embd[id as usize * h..(id as usize + 1) * h].to_vec())
    }

    fn dequant(&self, t: &TensorDesc) -> Result<Vec<f32>, ForwardError> {
        let bytes = self.gguf.slice(t);
        Ok(dequant_to_f32(t, bytes)?)
    }

    fn block_attn_norm<'b>(&self, b: &'b Block<'a>) -> &'b TensorDesc {
        match b {
            Block::Gdn(gb) => gb.attn_norm,
            Block::Attn(ab) => ab.attn_norm,
        }
    }
    fn block_post_norm<'b>(&self, b: &'b Block<'a>) -> &'b TensorDesc {
        match b {
            Block::Gdn(gb) => gb.post_attention_norm,
            Block::Attn(ab) => ab.post_attention_norm,
        }
    }

    /// SwiGLU FFN: `down(silu(gate(x)) * up(x))`.
    fn ffn(&self, block: &Block<'a>, x: &[f32]) -> Result<Vec<f32>, ForwardError> {
        let (g, u, d) = match block {
            Block::Gdn(b) => (b.ffn_gate, b.ffn_up, b.ffn_down),
            Block::Attn(b) => (b.ffn_gate, b.ffn_up, b.ffn_down),
        };
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let f = arch.intermediate_size as usize;
        let gate_w = self.dequant(g)?;
        let up_w = self.dequant(u)?;
        let down_w = self.dequant(d)?;

        let gated = mat_vec(&gate_w, h, f, x);
        let upped = mat_vec(&up_w, h, f, x);
        let mut hidden = vec![0.0f32; f];
        for i in 0..f {
            hidden[i] = silu(gated[i]) * upped[i];
        }
        Ok(mat_vec(&down_w, f, h, &hidden))
    }

    /// Full-attention block step. Single-token decode: appends one (K,V)
    /// to the cache, runs attention against the full prefix.
    fn attn_step(
        &self,
        layer_idx: usize,
        ab: &AttnBlock<'a>,
        x: &[f32],
        position: u32,
        kv_cache: &mut KvCache,
    ) -> Result<Vec<f32>, ForwardError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let head_dim = arch.attn_head_dim as usize;
        let n_q = arch.n_q_heads as usize;
        let n_kv = arch.n_kv_heads as usize;
        let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;

        // Q projection: outputs 2× heads — first half is Q, second half is gate.
        let q_w = self.dequant(ab.q)?;
        let q_full = mat_vec(&q_w, h, 2 * n_q * head_dim, x);
        // Split: qcur = first n_q*head_dim per head pair, gate = next n_q*head_dim.
        // Q is laid out as [head0_q (head_dim), head0_gate (head_dim), head1_q, ...].
        let mut qcur = vec![0.0f32; n_q * head_dim];
        let mut gate = vec![0.0f32; n_q * head_dim];
        for hi in 0..n_q {
            let src = &q_full[hi * 2 * head_dim..(hi + 1) * 2 * head_dim];
            qcur[hi * head_dim..(hi + 1) * head_dim].copy_from_slice(&src[..head_dim]);
            gate[hi * head_dim..(hi + 1) * head_dim].copy_from_slice(&src[head_dim..]);
        }

        // Q-norm (per-head RMSNorm on head_dim).
        let qnorm_w = self.dequant(ab.q_norm)?;
        for hi in 0..n_q {
            let s = hi * head_dim;
            let normed = rms_norm(&qcur[s..s + head_dim], &qnorm_w, RMS_EPS);
            qcur[s..s + head_dim].copy_from_slice(&normed);
        }

        // K, V projections.
        let k_w = self.dequant(ab.k)?;
        let v_w = self.dequant(ab.v)?;
        let mut kcur = mat_vec(&k_w, h, n_kv * head_dim, x);
        let vcur = mat_vec(&v_w, h, n_kv * head_dim, x);

        // K-norm (per-head).
        let knorm_w = self.dequant(ab.k_norm)?;
        for hi in 0..n_kv {
            let s = hi * head_dim;
            let normed = rms_norm(&kcur[s..s + head_dim], &knorm_w, RMS_EPS);
            kcur[s..s + head_dim].copy_from_slice(&normed);
        }

        // Partial RoPE on Q and K (first n_rot dims of each head).
        rope_in_place(&mut qcur, n_q, head_dim, n_rot, position, arch.rope_theta);
        rope_in_place(&mut kcur, n_kv, head_dim, n_rot, position, arch.rope_theta);

        // Append to KV cache for this layer.
        kv_cache.append(layer_idx, position, &kcur, &vcur);

        // Attention against full prefix. GQA: each Q head pulls KV from its group.
        // group size = n_q / n_kv.
        let group = n_q / n_kv;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; n_q * head_dim];
        for qh in 0..n_q {
            let kvh = qh / group;
            let q_slice = &qcur[qh * head_dim..(qh + 1) * head_dim];

            // Score against every cached K position for this layer & kv-head.
            let cache = kv_cache.layer(layer_idx);
            let n_pos = cache.len(); // includes the position we just appended
            let mut scores = vec![0.0f32; n_pos];
            for p in 0..n_pos {
                let k_slice = cache.k(p, kvh, head_dim);
                let mut s = 0.0f32;
                for i in 0..head_dim {
                    s += q_slice[i] * k_slice[i];
                }
                scores[p] = s * scale;
            }
            softmax_in_place(&mut scores);

            // Aggregate V.
            let out_slice = &mut attn_out[qh * head_dim..(qh + 1) * head_dim];
            for p in 0..n_pos {
                let v_slice = cache.v(p, kvh, head_dim);
                let w = scores[p];
                for i in 0..head_dim {
                    out_slice[i] += w * v_slice[i];
                }
            }
        }

        // Apply gated-attention sigmoid gate before output projection.
        for i in 0..attn_out.len() {
            attn_out[i] *= sigmoid(gate[i]);
        }

        // Output projection.
        let o_w = self.dequant(ab.o)?;
        Ok(mat_vec(&o_w, n_q * head_dim, h, &attn_out))
    }

    /// One step of GDN recurrence. Updates `conv` and `ssm` in place,
    /// returns the layer's output (hidden_size).
    fn gdn_step(
        &self,
        gb: &GdnBlock<'a>,
        x: &[f32],
        conv: &mut Vec<f32>,
        ssm: &mut Vec<f32>,
    ) -> Result<Vec<f32>, ForwardError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let n_v = arch.gdn_n_v_heads as usize;
        let n_k = arch.gdn_n_k_heads as usize;
        let head_dim = arch.gdn_head_dim as usize;
        let conv_dim = 2 * n_k * head_dim + n_v * head_dim;
        let conv_kernel = arch.gdn_conv_kernel as usize;

        // QKV input projection: [hidden, conv_dim]
        let qkv_w = self.dequant(gb.in_proj_qkv)?;
        let qkv = mat_vec(&qkv_w, h, conv_dim, x);

        // z (output gate) projection: [hidden, n_v * head_dim]
        let z_w = self.dequant(gb.in_proj_z)?;
        let z = mat_vec(&z_w, h, n_v * head_dim, x);

        // β projection then sigmoid: [hidden, n_v]
        let beta_w = self.dequant(gb.beta_proj)?;
        let mut beta = mat_vec(&beta_w, h, n_v, x);
        for v in beta.iter_mut() {
            *v = sigmoid(*v);
        }

        // α projection: [hidden, n_v]
        let alpha_w = self.dequant(gb.alpha_proj)?;
        let mut alpha = mat_vec(&alpha_w, h, n_v, x);
        // Add dt bias.
        let dt = self.dequant(gb.dt_bias)?;
        for (a, &dti) in alpha.iter_mut().zip(dt.iter()) {
            *a += dti;
        }
        // softplus then multiply by ssm_a (already -A_log.exp() at convert)
        let a_log = self.dequant(gb.a_log)?;
        let mut g = vec![0.0f32; n_v];
        for i in 0..n_v {
            g[i] = softplus(alpha[i]) * a_log[i];
        }

        // Conv1d step: shift the conv buffer left by 1, append qkv, then
        // dot with the kernel along the time axis. `ssm_conv1d.weight`
        // has GGUF shape `[kernel, conv_dim]` with `ne[0]=kernel` (fastest)
        // and `ne[1]=conv_dim` (slowest). So channel c at kernel position
        // k is at `conv_w[c * kernel + k]` — channel-blocked, kernel-inner.
        // Confirmed against ggml_ssm_conv (ggml/src/ggml.c) which reads
        // c->ne[0]=d_conv, c->ne[1]=d_inner.
        let conv_w = self.dequant(gb.conv1d)?;

        // conv buffer holds (kernel-1) past time steps, contiguous as
        // [t0_ch0, t0_ch1, ..., t0_chN-1, t1_ch0, ...].
        let kmin1 = conv_kernel - 1;
        let mut conv_input_t = vec![0.0f32; conv_kernel * conv_dim];
        // First (kernel-1) rows are the past.
        for t in 0..kmin1 {
            let src = &conv[t * conv_dim..(t + 1) * conv_dim];
            conv_input_t[t * conv_dim..(t + 1) * conv_dim].copy_from_slice(src);
        }
        // Last row is current qkv.
        conv_input_t[kmin1 * conv_dim..].copy_from_slice(&qkv);

        // Convolve along time axis. Input `conv_input_t` is `[kernel, conv_dim]`
        // row-major over time (each time slice is one full conv_dim row);
        // weight `conv_w` is `[conv_dim, kernel]` (channel-blocked).
        let mut conv_out = vec![0.0f32; conv_dim];
        for c in 0..conv_dim {
            let mut s = 0.0f32;
            for k in 0..conv_kernel {
                s += conv_w[c * conv_kernel + k] * conv_input_t[k * conv_dim + c];
            }
            conv_out[c] = silu(s);
        }
        // Update conv buffer: drop the oldest row, slide.
        for t in 0..kmin1 - 1 {
            let (left, right) = conv.split_at_mut((t + 1) * conv_dim);
            let _ = (left, right);
        }
        // Simpler: rotate.
        for t in 0..kmin1 - 1 {
            let next = (t + 1) * conv_dim;
            let cur = t * conv_dim;
            for i in 0..conv_dim {
                conv[cur + i] = conv[next + i];
            }
        }
        // Last row of conv buffer becomes the current qkv.
        let last = (kmin1 - 1) * conv_dim;
        conv[last..last + conv_dim].copy_from_slice(&qkv);

        // Split conv_out into Q, K, V. Layout is [qkv]: K_head*head_dim Q, then K, then V.
        // Total = 2*n_k*head_dim + n_v*head_dim. Order in qkv buffer: Q, K, V — confirmed
        // by qwen35.cpp lines 296-312 (q_view at offset 0, k_view at offset n_k*head_dim,
        // v_view at offset 2*n_k*head_dim).
        let q_off = 0;
        let k_off = n_k * head_dim;
        let v_off = 2 * n_k * head_dim;
        let mut q_full = conv_out[q_off..q_off + n_k * head_dim].to_vec();
        let mut k_full = conv_out[k_off..k_off + n_k * head_dim].to_vec();
        let v_full = conv_out[v_off..v_off + n_v * head_dim].to_vec();

        // L2-normalize Q and K per head.
        for hi in 0..n_k {
            l2_norm_in_place(&mut q_full[hi * head_dim..(hi + 1) * head_dim], RMS_EPS);
            l2_norm_in_place(&mut k_full[hi * head_dim..(hi + 1) * head_dim], RMS_EPS);
        }

        // Repeat Q/K from n_k heads to n_v heads (3:1 in 27B, 1:1 in 0.8B).
        // ggml_repeat_4d tiles heads as [h0,h1,...,h_{nk-1}, h0,h1,...] —
        // i.e. `src_h = hi % n_k`, NOT `hi / head_ratio`. The latter would
        // give [h0,h0,h0,h1,h1,h1,...] which is wrong for ggml's repeat.
        // For the 0.8B with n_v == n_k, both reduce to identity; for 27B
        // (n_v=48, n_k=16), this matters.
        let q = if n_v == n_k {
            q_full.clone()
        } else {
            let mut out = vec![0.0f32; n_v * head_dim];
            for hi in 0..n_v {
                let src_h = hi % n_k;
                out[hi * head_dim..(hi + 1) * head_dim]
                    .copy_from_slice(&q_full[src_h * head_dim..(src_h + 1) * head_dim]);
            }
            out
        };
        let k = if n_v == n_k {
            k_full.clone()
        } else {
            let mut out = vec![0.0f32; n_v * head_dim];
            for hi in 0..n_v {
                let src_h = hi % n_k;
                out[hi * head_dim..(hi + 1) * head_dim]
                    .copy_from_slice(&k_full[src_h * head_dim..(src_h + 1) * head_dim]);
            }
            out
        };

        // Delta-net recurrence: per V-head, state ∈ R^{head_dim x head_dim}.
        // S_new = exp(g) * S - exp(g) * (k ⊗ k^T) * S * β + β * (v ⊗ k)
        //       ≈ S * exp(g) + β * (v - exp(g) * S^T * k) ⊗ k
        // Following qwen35.cpp + vLLM's GatedDeltaRule reference:
        //   S ← exp(g) * S + β * (v − exp(g) * Sᵀ k) ⊗ k
        //   o = Sᵀ q   (then RMSNormGated, then output projection)
        // State layout in `ssm`: [head, dim_v, dim_k] as
        // ssm[h * head_dim*head_dim + dv * head_dim + dk].
        let mut o = vec![0.0f32; n_v * head_dim];
        for hi in 0..n_v {
            let s_off = hi * head_dim * head_dim;
            let q_h = &q[hi * head_dim..(hi + 1) * head_dim];
            let k_h = &k[hi * head_dim..(hi + 1) * head_dim];
            let v_h = &v_full[hi * head_dim..(hi + 1) * head_dim];
            let g_h = g[hi].exp();
            let b_h = beta[hi];

            // Compute Sᵀ k → vector of length head_dim (row dim_v):
            //   tmp_v[dv] = sum_dk S[dv, dk] * k[dk]
            let mut sk = vec![0.0f32; head_dim];
            for dv in 0..head_dim {
                let mut s = 0.0f32;
                for dk in 0..head_dim {
                    s += ssm[s_off + dv * head_dim + dk] * k_h[dk];
                }
                sk[dv] = s;
            }
            // Update: S ← g_h * S + b_h * (v - g_h * sk) ⊗ k
            for dv in 0..head_dim {
                let coeff = b_h * (v_h[dv] - g_h * sk[dv]);
                for dk in 0..head_dim {
                    let idx = s_off + dv * head_dim + dk;
                    ssm[idx] = g_h * ssm[idx] + coeff * k_h[dk];
                }
            }
            // Output: o[dv] = (sum_dk S[dv, dk] * q[dk]) / sqrt(head_dim)
            // The `1/sqrt(S_v)` scale matches ggml-cpu/ops.cpp gated_delta_net
            // (line 10551: `attn_data[j] = sum * scale` with `scale = 1/sqrtf(S_v)`).
            let o_h = &mut o[hi * head_dim..(hi + 1) * head_dim];
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            for dv in 0..head_dim {
                let mut s = 0.0f32;
                for dk in 0..head_dim {
                    s += ssm[s_off + dv * head_dim + dk] * q_h[dk];
                }
                o_h[dv] = s * scale;
            }
        }

        // RMSNormGated: norm(o) * silu(z), per-head.
        let norm_w = self.dequant(gb.norm)?;
        let mut gated_out = vec![0.0f32; n_v * head_dim];
        for hi in 0..n_v {
            let s = hi * head_dim;
            let normed = rms_norm(&o[s..s + head_dim], &norm_w, RMS_EPS);
            for i in 0..head_dim {
                gated_out[s + i] = normed[i] * silu(z[s + i]);
            }
        }

        // Output projection: [n_v * head_dim, hidden].
        let out_w = self.dequant(gb.out_proj)?;
        Ok(mat_vec(&out_w, n_v * head_dim, h, &gated_out))
    }
}

/// KV cache for full-attention layers, append-only. One entry per
/// (layer, position). Stored as flat f32; not the long-term layout but
/// simple to validate.
pub struct KvCache {
    layers: Vec<LayerKv>,
    head_dim: usize,
    n_kv_heads: usize,
}

struct LayerKv {
    k: Vec<f32>, // [pos, kv_head, head_dim]
    v: Vec<f32>,
    n_pos: usize,
}

impl LayerKv {
    fn len(&self) -> usize {
        self.n_pos
    }
    fn k(&self, pos: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let stride = head_dim;
        let base = pos * (self.k.len() / self.n_pos.max(1));
        let _ = (stride, base);
        // Simpler: head-major-per-pos layout: k[pos * (n_kv*hd) + kv_head*hd .. + hd]
        // But we don't have n_kv here. Recompute via division:
        let per_pos = self.k.len() / self.n_pos.max(1);
        let n_kv = per_pos / head_dim;
        let _ = n_kv;
        &self.k[pos * per_pos + kv_head * head_dim..pos * per_pos + (kv_head + 1) * head_dim]
    }
    fn v(&self, pos: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let per_pos = self.v.len() / self.n_pos.max(1);
        &self.v[pos * per_pos + kv_head * head_dim..pos * per_pos + (kv_head + 1) * head_dim]
    }
}

impl KvCache {
    pub fn new(model: &Model<'_>) -> Self {
        let arch = &model.arch;
        let layers = (0..arch.n_layer)
            .map(|_| LayerKv {
                k: Vec::new(),
                v: Vec::new(),
                n_pos: 0,
            })
            .collect();
        Self {
            layers,
            head_dim: arch.attn_head_dim as usize,
            n_kv_heads: arch.n_kv_heads as usize,
        }
    }

    fn layer(&self, idx: usize) -> &LayerKv {
        &self.layers[idx]
    }

    fn append(&mut self, layer: usize, _pos: u32, k: &[f32], v: &[f32]) {
        let lk = &mut self.layers[layer];
        debug_assert_eq!(k.len(), self.n_kv_heads * self.head_dim);
        debug_assert_eq!(v.len(), self.n_kv_heads * self.head_dim);
        lk.k.extend_from_slice(k);
        lk.v.extend_from_slice(v);
        lk.n_pos += 1;
    }
}

// ----- math primitives -----

const RMS_EPS: f32 = 1e-6;

/// Public re-export of [`rms_norm`] for kernel-validation tests.
pub fn rms_norm_pub(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    rms_norm(x, weight, eps)
}

/// Public re-export of [`mat_vec`] for kernel-validation tests.
pub fn mat_vec_pub(w: &[f32], n_in: usize, n_out: usize, x: &[f32]) -> Vec<f32> {
    mat_vec(w, n_in, n_out, x)
}

fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * scale * wi)
        .collect()
}

/// L2 norm with ggml semantics: `y = x / max(||x||, eps)`. Note the
/// max-with-eps form (NOT `1 / sqrt(sum + eps)` — that's RMSNorm). See
/// `ggml_compute_forward_l2_norm` in `ggml/src/ggml-cpu/ops.cpp:4080`.
fn l2_norm_in_place(x: &mut [f32], eps: f32) {
    let sq: f32 = x.iter().map(|&v| v * v).sum();
    let scale = 1.0 / sq.sqrt().max(eps);
    for v in x.iter_mut() {
        *v *= scale;
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn softmax_in_place(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// `y = W @ x` where `W` has GGUF shape `[n_in, n_out]` meaning `ne[0]=n_in`
/// (fastest axis) and `ne[1]=n_out`. The `o`-th output row of W is stored
/// contiguously at `data[o * n_in .. (o+1) * n_in]`. So:
///
/// ```text
///   y[o] = sum_i x[i] * W[i, o]
///        = sum_i x[i] * data[o * n_in + i]
/// ```
///
/// This matches `ggml_mul_mat(W, x)` semantics where ne[0] is contracted.
fn mat_vec(w: &[f32], n_in: usize, n_out: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(w.len(), n_in * n_out);
    debug_assert_eq!(x.len(), n_in);
    let mut y = vec![0.0f32; n_out];
    for o in 0..n_out {
        let row = &w[o * n_in..(o + 1) * n_in];
        let mut s = 0.0f32;
        for i in 0..n_in {
            s += x[i] * row[i];
        }
        y[o] = s;
    }
    y
}

/// Apply partial-RoPE in place to a `[n_heads * head_dim]` flat buffer,
/// using **NEOX ordering** (pair `(buf[i], buf[i + n_rot/2])`), which is
/// what `GGML_ROPE_TYPE_IMROPE` uses for text-only positions in qwen35.
/// MRoPE sections collapse to `t` (time) for pure text, so this reduces
/// to standard NEOX RoPE with `position` as the time coordinate.
///
/// Reference: ggml/include/ggml.h docstring (line 1836):
/// "IMROPE n_dims = 16 --> [ttyxttyxttyxttyx00] (interleaved M-RoPE,
/// still NEOX ordering)".
///
/// First `n_rot` dims of each head are rotated; the rest pass through.
fn rope_in_place(
    buf: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    n_rot: usize,
    position: u32,
    theta_base: f32,
) {
    let pos = position as f32;
    let half = n_rot / 2;
    for hi in 0..n_heads {
        let h_off = hi * head_dim;
        for i in 0..half {
            // Frequency for the i-th pair (NEOX: pair (i, i+half)).
            // theta_i = position / theta_base^(2i / n_rot)
            let exponent = (2 * i) as f32 / n_rot as f32;
            let freq = pos / theta_base.powf(exponent);
            let (s, c) = freq.sin_cos();
            let a = buf[h_off + i];
            let b = buf[h_off + i + half];
            buf[h_off + i] = a * c - b * s;
            buf[h_off + i + half] = a * s + b * c;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitives_smoke() {
        let mut a = vec![1.0, 2.0, 3.0];
        l2_norm_in_place(&mut a, 1e-6);
        let n: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-3);

        let s = silu(1.0);
        assert!((s - 0.7310586).abs() < 1e-5);

        let p = softplus(0.0);
        assert!((p - (2.0_f32.ln())).abs() < 1e-5);

        let mut sm = vec![1.0, 2.0, 3.0];
        softmax_in_place(&mut sm);
        let total: f32 = sm.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn forward_runs_one_token_0_8b() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let f = Forward::new(&g, &m);
        let mut state = GdnState::fresh(&m);
        let mut kv = KvCache::new(&m);

        let logits = f.single_token(0, 0, &mut state, &mut kv).expect("forward");
        assert_eq!(logits.len(), m.arch.vocab_size as usize);

        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        eprintln!(
            "[forward] vocab={} argmax={} top_logit={:.4}",
            logits.len(),
            argmax,
            logits[argmax]
        );
        assert!(logits[argmax].is_finite());
    }

    /// Sanity test: bypass all transformer blocks. Embed → output_norm →
    /// lm_head. Argmax should be a reasonable token id for the input
    /// even without a real model — just confirms our embedding lookup,
    /// RMSNorm, and mat_vec orientations are coherent.
    #[test]
    fn forward_bypass_blocks_smoke() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let f = Forward::new(&g, &m);

        // Embed token 9419 ("Hello").
        let x = f.embed_token(9419).expect("embed");
        assert_eq!(x.len(), m.arch.hidden_size as usize);
        let x_max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let x_min = x.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!(
            "[bypass] embed[9419]: hidden={} range=[{:.4},{:.4}]",
            x.len(),
            x_min,
            x_max
        );

        // Final RMS norm.
        let on = f.dequant(m.output_norm).expect("dequant");
        let normed = rms_norm(&x, &on, RMS_EPS);
        let n_max = normed.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let n_min = normed.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("[bypass] post-norm range=[{:.4},{:.4}]", n_min, n_max);

        // LM head.
        let lm = f.dequant(m.lm_head).expect("dequant lm");
        let logits = mat_vec(
            &lm,
            m.arch.hidden_size as usize,
            m.arch.vocab_size as usize,
            &normed,
        );
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        eprintln!("[bypass] argmax token={} logit={:.4}", argmax.0, argmax.1);

        // The token most likely to be predicted from the bare embedding
        // of "Hello" is "Hello" itself (since the embedding is essentially
        // the model saying "this token is here"). If our orientations are
        // correct, the argmax should be 9419 (the input token).
        // If it's some random token, we have an embedding/lm_head/mat_vec orientation bug.
        assert!(argmax.1.is_finite());
    }

    /// Oracle test: run "Hello" through our forward and compare against
    /// llm/llama_core's snapshot dump on the same prompt + same model.
    ///
    /// To regenerate the oracle:
    /// ```sh
    /// ~/code/llm/target/release/llm --model /Users/tito/models/Qwen3.5-0.8B.F32.gguf \
    ///   --snapshot /tmp/qwen-oracle --snapshot-id hello_t0 --raw "Hello"
    /// ```
    /// Produces /tmp/qwen-oracle/hello_t0.{f32,json}.
    #[test]
    fn oracle_match_hello_0_8b_f32() {
        let model_path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        let oracle_path = "/tmp/qwen-oracle/hello_t0.f32";
        if !std::path::Path::new(model_path).exists() || !std::path::Path::new(oracle_path).exists()
        {
            eprintln!("[oracle] skipped — fixtures missing");
            return;
        }

        // Load oracle logits (binary f32 vector).
        let oracle_bytes = std::fs::read(oracle_path).expect("read oracle");
        let n = oracle_bytes.len() / 4;
        let oracle: Vec<f32> = (0..n)
            .map(|i| f32::from_le_bytes(oracle_bytes[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();

        // Load + forward.
        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        assert_eq!(oracle.len(), m.arch.vocab_size as usize);

        // Tokenize "Hello" (raw, no chat template, no specials).
        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let ids = tok.encode("Hello", false).expect("tokenize");
        eprintln!("[oracle] 'Hello' -> {ids:?} ({} tokens)", ids.len());

        let f = Forward::new(&g, &m);
        let mut state = GdnState::fresh(&m);
        let mut kv = KvCache::new(&m);

        // Run all tokens, return the logits *after* the last one — same
        // semantics as llm's `--snapshot`.
        let mut last_logits = vec![];
        for (i, &tid) in ids.iter().enumerate() {
            last_logits = f
                .single_token(tid, i as u32, &mut state, &mut kv)
                .expect("forward");
        }
        assert_eq!(last_logits.len(), oracle.len());

        // Compute mismatch metrics.
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let mut argmax_ours = 0usize;
        let mut argmax_oracle = 0usize;
        let mut max_ours = f32::NEG_INFINITY;
        let mut max_oracle = f32::NEG_INFINITY;
        for i in 0..n {
            let d = (last_logits[i] - oracle[i]).abs();
            max_abs = max_abs.max(d);
            sum_abs += d as f64;
            if last_logits[i] > max_ours {
                max_ours = last_logits[i];
                argmax_ours = i;
            }
            if oracle[i] > max_oracle {
                max_oracle = oracle[i];
                argmax_oracle = i;
            }
        }
        let mean_abs = sum_abs / n as f64;
        eprintln!(
            "[oracle] argmax: ours={argmax_ours} (logit {:.4}) | oracle={argmax_oracle} (logit {:.4}) | max|Δ|={max_abs:.4} mean|Δ|={mean_abs:.6}",
            max_ours, max_oracle
        );

        // Cosine similarity for a global signal.
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..n {
            dot += last_logits[i] as f64 * oracle[i] as f64;
            na += (last_logits[i] as f64).powi(2);
            nb += (oracle[i] as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!("[oracle] cosine={cos:.6}");

        // Hard bounds: cosine ≥ 0.9999 and argmax must match. Element-wise
        // |Δ| up to ~1e-2 is fp32 reordering noise across 24 layers; we're
        // currently at 2.4e-3.
        assert!(cos > 0.9999, "cosine={cos} below threshold");
        assert_eq!(
            argmax_ours, argmax_oracle,
            "argmax disagreement: ours={argmax_ours} oracle={argmax_oracle}"
        );
        assert!(max_abs < 0.05, "max|Δ|={max_abs} above noise floor");
    }

    /// 27B Q4_K_M oracle test. Exercises:
    /// * the codec seam (real dequant on Q4_K, Q6_K weights)
    /// * the n_v=48 / n_k=16 GDN head-repeat path (3:1 ratio)
    /// * untied embeddings (token_embd is Q4_K, output is Q6_K)
    /// * 64 layers (16 full-attn + 48 GDN, vs 0.8B's 24)
    ///
    /// Slow: ~minutes per token in debug, single-digit seconds in release.
    /// Marked `#[ignore]` so default `cargo test` doesn't run it; invoke
    /// explicitly with `cargo test --release -p qwen-llm 27b -- --ignored`.
    #[test]
    #[ignore]
    fn oracle_match_27b_q4_k_m() {
        let model_path = "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf";
        let oracle_path = "/tmp/qwen-oracle/hello_27b_q4km.f32";
        if !std::path::Path::new(model_path).exists() || !std::path::Path::new(oracle_path).exists()
        {
            eprintln!("[oracle-27b] skipped — fixtures missing");
            return;
        }

        let oracle_bytes = std::fs::read(oracle_path).expect("read oracle");
        let n = oracle_bytes.len() / 4;
        let oracle: Vec<f32> = (0..n)
            .map(|i| f32::from_le_bytes(oracle_bytes[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();

        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        assert_eq!(oracle.len(), m.arch.vocab_size as usize);
        assert!(!m.tied_embeddings);

        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let ids = tok.encode("Hello", false).expect("tokenize");
        eprintln!("[oracle-27b] {ids:?}");

        let f = Forward::new(&g, &m);
        let mut state = GdnState::fresh(&m);
        let mut kv = KvCache::new(&m);
        let logits = f
            .single_token(ids[0], 0, &mut state, &mut kv)
            .expect("forward");

        let mut argmax_ours = 0usize;
        let mut argmax_oracle = 0usize;
        let mut max_ours = f32::NEG_INFINITY;
        let mut max_oracle = f32::NEG_INFINITY;
        let mut max_abs = 0.0f32;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..n {
            let d = (logits[i] - oracle[i]).abs();
            max_abs = max_abs.max(d);
            if logits[i] > max_ours {
                max_ours = logits[i];
                argmax_ours = i;
            }
            if oracle[i] > max_oracle {
                max_oracle = oracle[i];
                argmax_oracle = i;
            }
            dot += logits[i] as f64 * oracle[i] as f64;
            na += (logits[i] as f64).powi(2);
            nb += (oracle[i] as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!(
            "[oracle-27b] argmax: ours={argmax_ours} ({:.4}) | oracle={argmax_oracle} ({:.4}) | max|Δ|={max_abs:.4} cos={cos:.6}",
            max_ours, max_oracle
        );
        assert_eq!(
            argmax_ours, argmax_oracle,
            "argmax disagreement on 27B Q4_K_M"
        );
        // Q4_K introduces real quant noise — relax cosine to 0.999.
        assert!(cos > 0.999, "cosine={cos} below threshold");
    }

    /// Multi-token prompt (9 tokens). Exercises position > 0 in attention
    /// layers and incremental GDN state across the prefill.
    ///
    /// Oracle generated via:
    /// ```sh
    /// ~/code/llm/target/release/llm --model /Users/tito/models/Qwen3.5-0.8B.F32.gguf \
    ///   --snapshot /tmp/qwen-oracle --snapshot-id longprompt_t0 \
    ///   --raw "The quick brown fox jumps over the lazy dog"
    /// ```
    /// Expected argmax: "." (token 13).
    #[test]
    fn oracle_match_longprompt_0_8b_f32() {
        let model_path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        let oracle_path = "/tmp/qwen-oracle/longprompt_t0.f32";
        if !std::path::Path::new(model_path).exists() || !std::path::Path::new(oracle_path).exists()
        {
            eprintln!("[oracle-long] skipped — fixtures missing");
            return;
        }

        let oracle_bytes = std::fs::read(oracle_path).expect("read oracle");
        let n = oracle_bytes.len() / 4;
        let oracle: Vec<f32> = (0..n)
            .map(|i| f32::from_le_bytes(oracle_bytes[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();

        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        assert_eq!(oracle.len(), m.arch.vocab_size as usize);

        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let ids = tok
            .encode("The quick brown fox jumps over the lazy dog", false)
            .expect("tokenize");
        eprintln!("[oracle-long] {} tokens: {ids:?}", ids.len());
        assert_eq!(ids.len(), 9, "tokenization should match oracle's 9 tokens");

        let f = Forward::new(&g, &m);
        let mut state = GdnState::fresh(&m);
        let mut kv = KvCache::new(&m);

        let mut last_logits = vec![];
        for (i, &tid) in ids.iter().enumerate() {
            last_logits = f
                .single_token(tid, i as u32, &mut state, &mut kv)
                .expect("forward");
        }

        let mut max_abs = 0.0f32;
        let mut argmax_ours = 0usize;
        let mut argmax_oracle = 0usize;
        let mut max_ours = f32::NEG_INFINITY;
        let mut max_oracle = f32::NEG_INFINITY;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..n {
            let d = (last_logits[i] - oracle[i]).abs();
            max_abs = max_abs.max(d);
            if last_logits[i] > max_ours {
                max_ours = last_logits[i];
                argmax_ours = i;
            }
            if oracle[i] > max_oracle {
                max_oracle = oracle[i];
                argmax_oracle = i;
            }
            dot += last_logits[i] as f64 * oracle[i] as f64;
            na += (last_logits[i] as f64).powi(2);
            nb += (oracle[i] as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!(
            "[oracle-long] argmax: ours={argmax_ours} (logit {:.4}) | oracle={argmax_oracle} (logit {:.4}) | max|Δ|={max_abs:.4} cos={cos:.6}",
            max_ours, max_oracle
        );
        assert_eq!(argmax_ours, 13, "argmax should be '.' (token 13)");
        assert!(cos > 0.9999, "cosine={cos} below threshold");
    }
}
