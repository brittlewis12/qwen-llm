//! End-to-end Metal forward pass driver.
//!
//! Mirrors `crate::forward::Forward` but with all heavy ops on Metal,
//! orchestrated through one `MTLCommandBuffer` per token. The CPU oracle
//! (`forward.rs`) stays as the validation reference.
//!
//! v1 scope:
//!
//! * Single-token decode (no batched prefill).
//! * F32-only first (Qwen3.5-0.8B.F32). Quantized lift comes after the
//!   F32 path validates against the oracle — every kernel that takes
//!   `mat_vec_f32` will swap to `mat_vec_q4_K` / `mat_vec_q6_K` based on
//!   tensor `dtype`.
//! * Full-attn layers built from existing kernels (q-proj, k-proj,
//!   v-proj, q-norm, k-norm, RoPE, KV append, scoring, softmax,
//!   V-aggregate, gate, output proj). No fused attn block yet — the
//!   composition lands first; a fused version is a v2 perf project.
//! * Persistent `MetalTensor`s for all weights; one `MetalSession` owns
//!   the per-sequence state (GDN conv buffers + SSM states + KV cache).
//! * Activation arena: per-step scratch buffers held in `MetalSession`,
//!   reused across layers.
//!
//! v2 lift after the F32 forward validates:
//!
//! * Quantized weight path — change `MetalModel::load` to use
//!   `from_gguf_tensor` directly (currently dequants F32 via codec).
//! * Indirect Command Buffer (ICB) — encode the per-token sequence once,
//!   replay it. The encode-only API is already designed for this.
//! * Fused full-attn block.

use crate::gguf::GgufFile;
use crate::loader::{AttnBlock, Block, GdnBlock, Model};
use crate::metal::{
    encode_add_inplace_f32, encode_attn_decode_f32, encode_copy_offset_f32, encode_gdn_step_f32,
    encode_get_rows_f32, encode_l2_norm_batched_f32, encode_mat_vec_f32, encode_mul_f32,
    encode_rms_norm_batched_f32, encode_rms_norm_mul_f32, encode_rmsnorm_gated_f32,
    encode_rope_neox_f32, encode_sigmoid_f32, encode_silu_mul_f32, encode_softplus_f32,
    encode_split_q_gate_f32, encode_ssm_conv_silu_f32, KernelEncoder, MetalContext, MetalError,
    MetalTensor,
};
use crate::tensor::{GgmlType, TensorDesc};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputePipelineState,
};

#[derive(Debug, thiserror::Error)]
pub enum MfError {
    #[error("metal: {0}")]
    Metal(#[from] MetalError),
    #[error("codec: {0}")]
    Codec(#[from] crate::codec::CodecError),
    #[error("token {0} out of vocab range {1}")]
    BadToken(i32, u32),
    #[error("v1 driver requires F32 weights; tensor {name} is {dtype:?}")]
    UnsupportedDtype { name: String, dtype: GgmlType },
}

/// All weight tensors, resident as `MetalTensor`s. Loaded once at session
/// start. v1: only F32 weights are supported here; quantized path comes
/// in v2 by switching `MetalModel::load` to use `MetalTensor::from_gguf_tensor`
/// directly (instead of going through the F32 codec) and the kernel
/// dispatchers to pick `_q4_k`/`_q6_k` based on dtype.
pub struct MetalModel {
    /// Reference back to the loader's bound model. Carries `arch`, the
    /// layer schedule (GDN vs Attn), tied-embedding flag.
    pub arch: crate::model::Arch,
    pub tied_embeddings: bool,

    pub token_embd: MetalTensor,
    pub output_norm: MetalTensor,
    pub lm_head: MetalTensor,

    pub blocks: Vec<MetalBlock>,
}

pub enum MetalBlock {
    Gdn(MetalGdnBlock),
    Attn(MetalAttnBlock),
}

pub struct MetalGdnBlock {
    pub attn_norm: MetalTensor,
    pub post_attn_norm: MetalTensor,
    pub ffn_gate: MetalTensor,
    pub ffn_up: MetalTensor,
    pub ffn_down: MetalTensor,
    pub in_proj_qkv: MetalTensor,
    pub in_proj_z: MetalTensor,
    pub beta_proj: MetalTensor,
    pub alpha_proj: MetalTensor,
    pub a_log: MetalTensor,
    pub dt_bias: MetalTensor,
    pub conv1d: MetalTensor,
    pub norm: MetalTensor,
    pub out_proj: MetalTensor,
}

pub struct MetalAttnBlock {
    pub attn_norm: MetalTensor,
    pub post_attn_norm: MetalTensor,
    pub ffn_gate: MetalTensor,
    pub ffn_up: MetalTensor,
    pub ffn_down: MetalTensor,
    pub q: MetalTensor, // outputs 2x q_dim — Q + gate
    pub k: MetalTensor,
    pub v: MetalTensor,
    pub o: MetalTensor,
    pub q_norm: MetalTensor,
    pub k_norm: MetalTensor,
}

impl MetalModel {
    /// Load weights from an `loader::Model` view. v1 currently only
    /// handles F32 weights — for quantized weights we go through the
    /// codec to dequant on load (acceptable for v1 testing on F32 files;
    /// v2 will skip the dequant for native quant kernels).
    pub fn load(ctx: &MetalContext, gguf: &GgufFile, model: &Model<'_>) -> Result<Self, MfError> {
        let load_tensor = |desc: &TensorDesc| -> Result<MetalTensor, MfError> {
            // For F32, copy bytes straight in. For other dtypes, dequant
            // via codec (TODO: switch to native quant kernels in v2).
            if desc.dtype == GgmlType::F32 {
                Ok(MetalTensor::from_gguf_tensor(ctx, desc, gguf.slice(desc))?)
            } else {
                let f32 = crate::codec::dequant_to_f32(desc, gguf.slice(desc))?;
                Ok(MetalTensor::from_bytes(
                    ctx,
                    bytemuck::cast_slice(&f32),
                    desc.shape.clone(),
                    GgmlType::F32, // re-tag: in-memory tensor is now F32
                )?)
            }
        };

        let token_embd = load_tensor(model.token_embd)?;
        let output_norm = load_tensor(model.output_norm)?;
        let lm_head = load_tensor(model.lm_head)?;

        let mut blocks = Vec::with_capacity(model.blocks.len());
        for b in &model.blocks {
            match b {
                Block::Gdn(g) => {
                    blocks.push(MetalBlock::Gdn(MetalGdnBlock {
                        attn_norm: load_tensor(g.attn_norm)?,
                        post_attn_norm: load_tensor(g.post_attention_norm)?,
                        ffn_gate: load_tensor(g.ffn_gate)?,
                        ffn_up: load_tensor(g.ffn_up)?,
                        ffn_down: load_tensor(g.ffn_down)?,
                        in_proj_qkv: load_tensor(g.in_proj_qkv)?,
                        in_proj_z: load_tensor(g.in_proj_z)?,
                        beta_proj: load_tensor(g.beta_proj)?,
                        alpha_proj: load_tensor(g.alpha_proj)?,
                        a_log: load_tensor(g.a_log)?,
                        dt_bias: load_tensor(g.dt_bias)?,
                        conv1d: load_tensor(g.conv1d)?,
                        norm: load_tensor(g.norm)?,
                        out_proj: load_tensor(g.out_proj)?,
                    }));
                }
                Block::Attn(a) => {
                    blocks.push(MetalBlock::Attn(MetalAttnBlock {
                        attn_norm: load_tensor(a.attn_norm)?,
                        post_attn_norm: load_tensor(a.post_attention_norm)?,
                        ffn_gate: load_tensor(a.ffn_gate)?,
                        ffn_up: load_tensor(a.ffn_up)?,
                        ffn_down: load_tensor(a.ffn_down)?,
                        q: load_tensor(a.q)?,
                        k: load_tensor(a.k)?,
                        v: load_tensor(a.v)?,
                        o: load_tensor(a.o)?,
                        q_norm: load_tensor(a.q_norm)?,
                        k_norm: load_tensor(a.k_norm)?,
                    }));
                }
            }
        }

        Ok(Self {
            arch: model.arch,
            tied_embeddings: model.tied_embeddings,
            token_embd,
            output_norm,
            lm_head,
            blocks,
        })
    }
}

/// Per-sequence state: GDN conv buffers + SSM states (one set per GDN
/// layer), KV cache (one set per attn layer), and a small pool of
/// scratch activation tensors that are reused across layers.
pub struct MetalSession {
    /// (kernel-1) * conv_dim per GDN layer, F32, contiguous.
    pub gdn_conv: Vec<MetalTensor>,
    /// n_v_heads * head_dim * head_dim per GDN layer, F32.
    pub gdn_state: Vec<MetalTensor>,

    /// `[capacity_tokens, n_kv_heads, head_dim]` per attn layer, F32.
    pub kv_k: Vec<MetalTensor>,
    pub kv_v: Vec<MetalTensor>,
    pub kv_n_pos: Vec<usize>,
    pub kv_capacity: usize,

    // Scratch arena — F32 buffers reused across layers within a single
    // forward step. Sized for the largest transient at each role.
    pub x: MetalTensor,            // hidden_size — residual stream
    pub h: MetalTensor,            // hidden_size — post-norm activation
    pub ffn_gate: MetalTensor,     // intermediate_size — FFN gate output
    pub ffn_up: MetalTensor,       // intermediate_size — FFN up output
    pub ffn_inner: MetalTensor,    // intermediate_size — silu(gate)*up
    pub ffn_out: MetalTensor,      // hidden_size — FFN final
    pub gdn_qkv: MetalTensor,      // conv_dim
    pub gdn_qkv_conv: MetalTensor, // conv_dim — post-conv
    pub gdn_z: MetalTensor,        // n_v * head_dim
    pub gdn_b: MetalTensor,        // n_v   (β source, pre-sigmoid)
    pub gdn_beta: MetalTensor,     // n_v   (post-sigmoid)
    pub gdn_a: MetalTensor,        // n_v   (α source, pre-softplus)
    pub gdn_alpha: MetalTensor,    // n_v   (post-softplus * a_log)
    pub gdn_q: MetalTensor,        // n_k * head_dim — Q view of gdn_qkv_conv
    pub gdn_k: MetalTensor,        // n_k * head_dim — K view
    pub gdn_v: MetalTensor,        // n_v * head_dim — V view
    pub gdn_q_norm: MetalTensor,   // n_k * head_dim — l2-normed
    pub gdn_k_norm: MetalTensor,   // n_k * head_dim — l2-normed
    pub gdn_out: MetalTensor,      // n_v * head_dim — recurrence output
    pub gdn_normed: MetalTensor,   // n_v * head_dim — RMSNormGated output
    pub gdn_proj: MetalTensor,     // hidden_size — out_proj output
    pub mixer_out: MetalTensor,    // hidden_size — mixer output (GDN or attn)

    // Attention scratch.
    pub attn_q_full: MetalTensor,   // 2 * q_dim — Q + gate interleaved
    pub attn_q: MetalTensor,        // q_dim — Q only
    pub attn_gate: MetalTensor,     // q_dim — sigmoid'd gate
    pub attn_q_normed: MetalTensor, // q_dim
    pub attn_k_now: MetalTensor,    // kv_dim — current step K
    pub attn_v_now: MetalTensor,    // kv_dim — current step V
    pub attn_k_normed: MetalTensor, // kv_dim
    pub attn_scores: MetalTensor,   // capacity_tokens — scores for current step
    pub attn_o: MetalTensor,        // q_dim — attention output

    pub logits: MetalTensor,  // vocab_size
    pub ids_buf: MetalTensor, // 1-element scratch for the input token id (i32 in an F32 buf)
}

impl MetalSession {
    pub fn fresh(
        ctx: &MetalContext,
        model: &MetalModel,
        kv_capacity: usize,
    ) -> Result<Self, MetalError> {
        let arch = &model.arch;
        let h = arch.hidden_size as u64;
        let f = arch.intermediate_size as u64;
        let head_dim = arch.attn_head_dim as u64;
        let n_q = arch.n_q_heads as u64;
        let n_kv = arch.n_kv_heads as u64;
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;
        let vh = arch.gdn_head_dim as u64;
        let n_v = arch.gdn_n_v_heads as u64;
        let n_k = arch.gdn_n_k_heads as u64;
        let conv_dim = (2 * n_k + n_v) * vh;
        let conv_kernel = arch.gdn_conv_kernel as u64;
        let v_dim = n_v * vh;
        let k_dim = n_k * vh;

        let mut gdn_conv = Vec::new();
        let mut gdn_state = Vec::new();
        for b in &model.blocks {
            if matches!(b, MetalBlock::Gdn(_)) {
                gdn_conv.push(MetalTensor::zeros_f32(
                    ctx,
                    vec![(conv_kernel - 1) * conv_dim],
                )?);
                gdn_state.push(MetalTensor::zeros_f32(ctx, vec![n_v * vh * vh])?);
            }
        }

        let mut kv_k = Vec::new();
        let mut kv_v = Vec::new();
        let mut kv_n_pos = Vec::new();
        for b in &model.blocks {
            if matches!(b, MetalBlock::Attn(_)) {
                kv_k.push(MetalTensor::zeros_f32(
                    ctx,
                    vec![kv_capacity as u64 * kv_dim],
                )?);
                kv_v.push(MetalTensor::zeros_f32(
                    ctx,
                    vec![kv_capacity as u64 * kv_dim],
                )?);
                kv_n_pos.push(0);
            }
        }

        Ok(Self {
            gdn_conv,
            gdn_state,
            kv_k,
            kv_v,
            kv_n_pos,
            kv_capacity,
            x: MetalTensor::zeros_f32(ctx, vec![h])?,
            h: MetalTensor::zeros_f32(ctx, vec![h])?,
            ffn_gate: MetalTensor::zeros_f32(ctx, vec![f])?,
            ffn_up: MetalTensor::zeros_f32(ctx, vec![f])?,
            ffn_inner: MetalTensor::zeros_f32(ctx, vec![f])?,
            ffn_out: MetalTensor::zeros_f32(ctx, vec![h])?,
            gdn_qkv: MetalTensor::zeros_f32(ctx, vec![conv_dim])?,
            gdn_qkv_conv: MetalTensor::zeros_f32(ctx, vec![conv_dim])?,
            gdn_z: MetalTensor::zeros_f32(ctx, vec![v_dim])?,
            gdn_b: MetalTensor::zeros_f32(ctx, vec![n_v])?,
            gdn_beta: MetalTensor::zeros_f32(ctx, vec![n_v])?,
            gdn_a: MetalTensor::zeros_f32(ctx, vec![n_v])?,
            gdn_alpha: MetalTensor::zeros_f32(ctx, vec![n_v])?,
            gdn_q: MetalTensor::zeros_f32(ctx, vec![k_dim])?,
            gdn_k: MetalTensor::zeros_f32(ctx, vec![k_dim])?,
            gdn_v: MetalTensor::zeros_f32(ctx, vec![v_dim])?,
            gdn_q_norm: MetalTensor::zeros_f32(ctx, vec![k_dim])?,
            gdn_k_norm: MetalTensor::zeros_f32(ctx, vec![k_dim])?,
            gdn_out: MetalTensor::zeros_f32(ctx, vec![v_dim])?,
            gdn_normed: MetalTensor::zeros_f32(ctx, vec![v_dim])?,
            gdn_proj: MetalTensor::zeros_f32(ctx, vec![h])?,
            mixer_out: MetalTensor::zeros_f32(ctx, vec![h])?,
            attn_q_full: MetalTensor::zeros_f32(ctx, vec![2 * q_dim])?,
            attn_q: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_gate: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_q_normed: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_k_now: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_v_now: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_k_normed: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_scores: MetalTensor::zeros_f32(ctx, vec![kv_capacity as u64])?,
            attn_o: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            logits: MetalTensor::zeros_f32(ctx, vec![arch.vocab_size as u64])?,
            ids_buf: MetalTensor::zeros_f32(ctx, vec![1])?,
        })
    }
}

/// End-to-end Metal forward driver.
pub struct MetalForward<'a> {
    pub ctx: &'a MetalContext,
    pub model: &'a MetalModel,
}

impl<'a> MetalForward<'a> {
    pub fn new(ctx: &'a MetalContext, model: &'a MetalModel) -> Self {
        Self { ctx, model }
    }

    /// Run a single token through the model. Encodes all kernels into
    /// one command buffer, commits, waits, reads back logits.
    ///
    /// `position` is the 0-indexed sequence position (used by RoPE for
    /// the full-attn layers; ignored by GDN layers).
    pub fn single_token(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<Vec<f32>, MfError> {
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }

        // Stage the token id into the ids_buf (i32 view of the F32 buffer).
        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);

        // (1) Embedding lookup → x.
        encode_get_rows_f32(
            self.ctx,
            &enc,
            &self.model.token_embd,
            &session.ids_buf,
            &session.x,
            1,
            arch.hidden_size as usize,
        )?;

        // (2) Per-block.
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (il, block) in self.model.blocks.iter().enumerate() {
            self.encode_block(
                &enc,
                il,
                block,
                &mut gdn_idx,
                &mut attn_idx,
                position,
                session,
            )?;
        }

        // (3) Final RMSNorm over residual stream.
        encode_rms_norm_mul_f32(
            self.ctx,
            &enc,
            &session.x,
            &self.model.output_norm,
            &session.h,
            RMS_EPS,
        )?;

        // (4) LM head → logits.
        encode_mat_vec_f32(
            self.ctx,
            &enc,
            &self.model.lm_head,
            &session.h,
            &session.logits,
            arch.hidden_size as usize,
            arch.vocab_size as usize,
        )?;

        enc.end();
        cmd_buf.commit();
        unsafe { cmd_buf.waitUntilCompleted() };

        let mut out = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        Ok(out)
    }

    fn encode_block(
        &self,
        enc: &KernelEncoder,
        _il: usize,
        block: &MetalBlock,
        gdn_idx: &mut usize,
        attn_idx: &mut usize,
        position: u32,
        s: &mut MetalSession,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;

        // Pre-mixer norm.
        let attn_norm = match block {
            MetalBlock::Gdn(g) => &g.attn_norm,
            MetalBlock::Attn(a) => &a.attn_norm,
        };
        encode_rms_norm_mul_f32(self.ctx, enc, &s.x, attn_norm, &s.h, RMS_EPS)?;

        // Mixer (GDN or Attn) → mixer_out.
        match block {
            MetalBlock::Gdn(g) => {
                let i = *gdn_idx;
                *gdn_idx += 1;
                self.encode_gdn(enc, g, i, s)?;
            }
            MetalBlock::Attn(a) => {
                let i = *attn_idx;
                *attn_idx += 1;
                self.encode_attn(enc, a, i, position, s)?;
            }
        }

        // Residual #1: x += mixer_out.
        encode_add_inplace_f32(self.ctx, enc, &s.x, &s.mixer_out)?;

        // Pre-FFN norm.
        let post_norm = match block {
            MetalBlock::Gdn(g) => &g.post_attn_norm,
            MetalBlock::Attn(a) => &a.post_attn_norm,
        };
        encode_rms_norm_mul_f32(self.ctx, enc, &s.x, post_norm, &s.h, RMS_EPS)?;

        // SwiGLU FFN.
        let (g_w, u_w, d_w) = match block {
            MetalBlock::Gdn(g) => (&g.ffn_gate, &g.ffn_up, &g.ffn_down),
            MetalBlock::Attn(a) => (&a.ffn_gate, &a.ffn_up, &a.ffn_down),
        };
        encode_mat_vec_f32(
            self.ctx,
            enc,
            g_w,
            &s.h,
            &s.ffn_gate,
            h,
            arch.intermediate_size as usize,
        )?;
        encode_mat_vec_f32(
            self.ctx,
            enc,
            u_w,
            &s.h,
            &s.ffn_up,
            h,
            arch.intermediate_size as usize,
        )?;
        encode_silu_mul_f32(self.ctx, enc, &s.ffn_gate, &s.ffn_up, &s.ffn_inner)?;
        encode_mat_vec_f32(
            self.ctx,
            enc,
            d_w,
            &s.ffn_inner,
            &s.ffn_out,
            arch.intermediate_size as usize,
            h,
        )?;

        // Residual #2: x += ffn_out.
        encode_add_inplace_f32(self.ctx, enc, &s.x, &s.ffn_out)?;
        Ok(())
    }

    fn encode_gdn(
        &self,
        enc: &KernelEncoder,
        gb: &MetalGdnBlock,
        gdn_i: usize,
        s: &mut MetalSession,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let n_v = arch.gdn_n_v_heads as usize;
        let n_k = arch.gdn_n_k_heads as usize;
        let head_dim = arch.gdn_head_dim as usize;
        let conv_dim = (2 * n_k + n_v) * head_dim;
        let v_dim = n_v * head_dim;

        // QKV input projection.
        encode_mat_vec_f32(
            self.ctx,
            enc,
            &gb.in_proj_qkv,
            &s.h,
            &s.gdn_qkv,
            h,
            conv_dim,
        )?;
        // z projection.
        encode_mat_vec_f32(self.ctx, enc, &gb.in_proj_z, &s.h, &s.gdn_z, h, v_dim)?;
        // β source projection (then sigmoid).
        encode_mat_vec_f32(self.ctx, enc, &gb.beta_proj, &s.h, &s.gdn_b, h, n_v)?;
        encode_sigmoid_f32(self.ctx, enc, &s.gdn_b, &s.gdn_beta)?;
        // α source projection.
        encode_mat_vec_f32(self.ctx, enc, &gb.alpha_proj, &s.h, &s.gdn_a, h, n_v)?;
        // α + dt; softplus; * a_log → g.
        // We need: gdn_a = softplus(gdn_a + dt_bias) * a_log.
        // Three small kernels: add (gdn_a + dt_bias), softplus, mul (* a_log).
        encode_add_inplace_f32(self.ctx, enc, &s.gdn_a, &gb.dt_bias)?;
        encode_softplus_f32(self.ctx, enc, &s.gdn_a, &s.gdn_alpha)?;
        encode_mul_f32(self.ctx, enc, &s.gdn_alpha, &gb.a_log, &s.gdn_alpha)?;
        // Now `gdn_alpha` is the per-head g (scalar log-decay).

        // Conv1d step + SiLU. Mutates the conv buffer in place.
        encode_ssm_conv_silu_f32(
            self.ctx,
            enc,
            &s.gdn_qkv,
            &s.gdn_conv[gdn_i],
            &gb.conv1d,
            &s.gdn_qkv_conv,
            conv_dim,
        )?;

        // Split conv output into Q, K, V slices via aliasing offsets.
        // We have separate scratch buffers (gdn_q, gdn_k, gdn_v) so we
        // do small explicit copies for now. The right v2 move is to make
        // the gdn_step kernel read the offsets directly.
        // For v1 correctness: copy via add with 0 (cheap on Metal,
        // exercises the same path as the CPU oracle for shape).
        encode_copy_offset_f32(self.ctx, enc, &s.gdn_qkv_conv, 0, &s.gdn_q, n_k * head_dim)?;
        encode_copy_offset_f32(
            self.ctx,
            enc,
            &s.gdn_qkv_conv,
            n_k * head_dim,
            &s.gdn_k,
            n_k * head_dim,
        )?;
        encode_copy_offset_f32(
            self.ctx,
            enc,
            &s.gdn_qkv_conv,
            2 * n_k * head_dim,
            &s.gdn_v,
            v_dim,
        )?;

        // Per-head L2-norm of Q and K.
        encode_l2_norm_batched_f32(
            self.ctx,
            enc,
            &s.gdn_q,
            &s.gdn_q_norm,
            n_k,
            head_dim,
            RMS_EPS,
        )?;
        encode_l2_norm_batched_f32(
            self.ctx,
            enc,
            &s.gdn_k,
            &s.gdn_k_norm,
            n_k,
            head_dim,
            RMS_EPS,
        )?;

        // Recurrence step (kernel does the head-repeat internally).
        encode_gdn_step_f32(
            self.ctx,
            enc,
            &s.gdn_q_norm,
            &s.gdn_k_norm,
            &s.gdn_v,
            &s.gdn_alpha,
            &s.gdn_beta,
            &s.gdn_state[gdn_i],
            &s.gdn_out,
            n_v,
            n_k,
            head_dim,
        )?;

        // RMSNormGated: y = norm(o) * silu(z), per head.
        encode_rmsnorm_gated_f32(
            self.ctx,
            enc,
            &s.gdn_out,
            &gb.norm,
            &s.gdn_z,
            &s.gdn_normed,
            n_v,
            head_dim,
            RMS_EPS,
        )?;

        // Output projection: [v_dim, hidden] → mixer_out.
        encode_mat_vec_f32(
            self.ctx,
            enc,
            &gb.out_proj,
            &s.gdn_normed,
            &s.mixer_out,
            v_dim,
            h,
        )?;
        Ok(())
    }

    fn encode_attn(
        &self,
        enc: &KernelEncoder,
        ab: &MetalAttnBlock,
        attn_i: usize,
        position: u32,
        s: &mut MetalSession,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let head_dim = arch.attn_head_dim as usize;
        let n_q = arch.n_q_heads as usize;
        let n_kv = arch.n_kv_heads as usize;
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;
        let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;

        // (1) Q projection: outputs 2 * q_dim (Q + gate interleaved per head).
        encode_mat_vec_f32(self.ctx, enc, &ab.q, &s.h, &s.attn_q_full, h, 2 * q_dim)?;

        // (2) Split into Q and gate.
        encode_split_q_gate_f32(
            self.ctx,
            enc,
            &s.attn_q_full,
            &s.attn_q,
            &s.attn_gate,
            n_q,
            head_dim,
        )?;

        // (3) Q-norm (per-head RMSNorm, shared per-channel weight).
        encode_rms_norm_batched_f32(
            self.ctx,
            enc,
            &s.attn_q,
            &ab.q_norm,
            &s.attn_q_normed,
            n_q,
            head_dim,
            RMS_EPS,
        )?;

        // (4) K, V projections.
        encode_mat_vec_f32(self.ctx, enc, &ab.k, &s.h, &s.attn_k_now, h, kv_dim)?;
        encode_mat_vec_f32(self.ctx, enc, &ab.v, &s.h, &s.attn_v_now, h, kv_dim)?;

        // (5) K-norm (per-head). Reuses Q-norm weight tensor type but
        // points at K's weight.
        encode_rms_norm_batched_f32(
            self.ctx,
            enc,
            &s.attn_k_now,
            &ab.k_norm,
            &s.attn_k_normed,
            n_kv,
            head_dim,
            RMS_EPS,
        )?;

        // (6) Partial RoPE on Q (in `attn_q_normed`) and K (in `attn_k_normed`).
        encode_rope_neox_f32(
            self.ctx,
            enc,
            &s.attn_q_normed,
            n_q,
            head_dim,
            n_rot,
            position,
            arch.rope_theta,
        )?;
        encode_rope_neox_f32(
            self.ctx,
            enc,
            &s.attn_k_normed,
            n_kv,
            head_dim,
            n_rot,
            position,
            arch.rope_theta,
        )?;

        // (7) KV cache append. v1 enforces strict-monotonic-from-zero;
        // we copy K and V at slot `position` directly via copy_offset
        // running in reverse direction. Since we don't yet have a
        // "scatter" kernel, we synchronously fill the KV cache slot via
        // CPU-visible memory under StorageModeShared. This requires a
        // wait-on-encoder boundary, which is the kind of CPU/GPU
        // synchronization codex warned against. **Acceptable for v1
        // single-token decode** because the readback is one row of
        // `kv_dim` floats (4 KB), and the next dispatch will see the
        // updated bytes. We just need to make sure the encoder is split
        // so the previous K/V projection has completed before we read.
        //
        // For v2 we'll add a `kv_append_f32` kernel that writes into the
        // cache slot using a small dispatch (no CPU sync needed).
        //
        // For now: emit a `copy_offset_f32` from the encoded K/V into
        // the right cache slot. The cache is `[capacity, n_kv_heads, head_dim]`
        // row-major, so slot `position` starts at `position * kv_stride`
        // bytes (here, in elements). We use copy_offset's offset arg
        // *inverted*: copy_offset reads from src+off into dst[0..n].
        // We need the opposite: copy from src[0..n] into dst+off. So
        // we add a small "scatter_offset" kernel below.
        encode_scatter_offset_f32(
            self.ctx,
            enc,
            &s.attn_k_normed,
            &s.kv_k[attn_i],
            (position as usize) * kv_dim,
            kv_dim,
        )?;
        encode_scatter_offset_f32(
            self.ctx,
            enc,
            &s.attn_v_now,
            &s.kv_v[attn_i],
            (position as usize) * kv_dim,
            kv_dim,
        )?;
        s.kv_n_pos[attn_i] = position as usize + 1;

        // (8) Fused attention decode: scoring + softmax + V-aggregate.
        encode_attn_decode_f32(
            self.ctx,
            enc,
            &s.attn_q_normed,
            &s.kv_k[attn_i],
            &s.kv_v[attn_i],
            &s.attn_o,
            n_q,
            n_kv,
            head_dim,
            s.kv_n_pos[attn_i],
        )?;

        // (9) Apply gated-attention sigmoid gate: attn_o *= sigmoid(gate).
        // We need: y = attn_o * sigmoid(gate). Decompose into
        //   sigmoid(gate) → tmp; attn_o * tmp → attn_o (in-place)
        // We don't have a buffer for tmp. Reuse attn_q (no longer needed
        // after attn_decode).
        encode_sigmoid_f32(self.ctx, enc, &s.attn_gate, &s.attn_q)?;
        encode_mul_f32(self.ctx, enc, &s.attn_o, &s.attn_q, &s.attn_o)?;

        // (10) Output projection: q_dim → hidden.
        encode_mat_vec_f32(self.ctx, enc, &ab.o, &s.attn_o, &s.mixer_out, q_dim, h)?;
        Ok(())
    }
}

/// Helper: scatter `n` floats from `src[0..n]` into `dst[off..off+n]`.
/// Inverse of `copy_offset` (which gathers). Used to write into the KV
/// cache slot for the current position.
fn encode_scatter_offset_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    dst: &MetalTensor,
    dst_off: usize,
    n: usize,
) -> Result<(), MetalError> {
    if src.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset",
            detail: format!("src.n={} != n={n}", src.n_elements()),
        });
    }
    if (dst_off + n) as u64 > dst.n_elements() {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset",
            detail: format!("dst_off+n={} > dst.n={}", dst_off + n, dst.n_elements()),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        dst_off: u32,
    }
    let pso = ctx.pipeline("kernel_scatter_offset_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            dst_off: dst_off as u32,
        },
    );
    enc.set_tensor(1, src);
    enc.set_tensor(2, dst);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024) as usize;
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        objc2_metal::MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

const RMS_EPS: f32 = 1e-6;

/// Public single-block GDN dispatcher for end-to-end validation. Encodes
/// one GDN block (norm → mixer → residual → post_norm → FFN → residual)
/// and reads back the resulting `x` (residual stream).
///
/// Mirrors what `single_token` does for one block but skips the global
/// embedding/lm_head, so we can validate one block at a time against the
/// CPU oracle. The GDN block is the most complex piece in the
/// architecture; if this is bit-tight, the rest of the driver is glue.
impl<'a> MetalForward<'a> {
    pub fn run_one_gdn_block_for_test(
        &self,
        block_idx: usize,
        gdn_idx_in_session: usize,
        s: &mut MetalSession,
    ) -> Result<Vec<f32>, MfError> {
        let block = &self.model.blocks[block_idx];
        let gb = match block {
            MetalBlock::Gdn(g) => g,
            MetalBlock::Attn(_) => {
                return Err(MfError::Metal(MetalError::BadShape {
                    kernel: "run_one_gdn_block",
                    detail: format!("block {block_idx} is not a GDN block"),
                }));
            }
        };

        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);

        // Pre-mixer norm (s.x → s.h).
        encode_rms_norm_mul_f32(self.ctx, &enc, &s.x, &gb.attn_norm, &s.h, RMS_EPS)?;

        // GDN mixer (s.h → s.mixer_out).
        self.encode_gdn(&enc, gb, gdn_idx_in_session, s)?;

        // Residual #1: s.x += s.mixer_out.
        encode_add_inplace_f32(self.ctx, &enc, &s.x, &s.mixer_out)?;

        // Pre-FFN norm (s.x → s.h).
        encode_rms_norm_mul_f32(self.ctx, &enc, &s.x, &gb.post_attn_norm, &s.h, RMS_EPS)?;

        // SwiGLU FFN.
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let f = arch.intermediate_size as usize;
        encode_mat_vec_f32(self.ctx, &enc, &gb.ffn_gate, &s.h, &s.ffn_gate, h, f)?;
        encode_mat_vec_f32(self.ctx, &enc, &gb.ffn_up, &s.h, &s.ffn_up, h, f)?;
        encode_silu_mul_f32(self.ctx, &enc, &s.ffn_gate, &s.ffn_up, &s.ffn_inner)?;
        encode_mat_vec_f32(self.ctx, &enc, &gb.ffn_down, &s.ffn_inner, &s.ffn_out, f, h)?;

        // Residual #2: s.x += s.ffn_out.
        encode_add_inplace_f32(self.ctx, &enc, &s.x, &s.ffn_out)?;

        enc.end();
        cmd_buf.commit();
        unsafe { cmd_buf.waitUntilCompleted() };

        let mut out = vec![0.0f32; h];
        unsafe {
            let src = s.x.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), h);
        }
        Ok(out)
    }

    /// Test helper: write `data` into `s.x` (the residual stream).
    pub fn set_residual_for_test(&self, s: &mut MetalSession, data: &[f32]) {
        unsafe {
            let dst = s.x.buffer.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
    }

    /// Test helper: run one full-attn block end-to-end (norm → attn →
    /// residual → post_norm → FFN → residual) and read back the residual
    /// stream. Mirrors `run_one_gdn_block_for_test` for the attn path.
    pub fn run_one_attn_block_for_test(
        &self,
        block_idx: usize,
        attn_idx_in_session: usize,
        position: u32,
        s: &mut MetalSession,
    ) -> Result<Vec<f32>, MfError> {
        let block = &self.model.blocks[block_idx];
        let ab = match block {
            MetalBlock::Attn(a) => a,
            MetalBlock::Gdn(_) => {
                return Err(MfError::Metal(MetalError::BadShape {
                    kernel: "run_one_attn_block",
                    detail: format!("block {block_idx} is not an attn block"),
                }));
            }
        };

        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);

        encode_rms_norm_mul_f32(self.ctx, &enc, &s.x, &ab.attn_norm, &s.h, RMS_EPS)?;
        self.encode_attn(&enc, ab, attn_idx_in_session, position, s)?;
        encode_add_inplace_f32(self.ctx, &enc, &s.x, &s.mixer_out)?;

        encode_rms_norm_mul_f32(self.ctx, &enc, &s.x, &ab.post_attn_norm, &s.h, RMS_EPS)?;

        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let f = arch.intermediate_size as usize;
        encode_mat_vec_f32(self.ctx, &enc, &ab.ffn_gate, &s.h, &s.ffn_gate, h, f)?;
        encode_mat_vec_f32(self.ctx, &enc, &ab.ffn_up, &s.h, &s.ffn_up, h, f)?;
        encode_silu_mul_f32(self.ctx, &enc, &s.ffn_gate, &s.ffn_up, &s.ffn_inner)?;
        encode_mat_vec_f32(self.ctx, &enc, &ab.ffn_down, &s.ffn_inner, &s.ffn_out, f, h)?;

        encode_add_inplace_f32(self.ctx, &enc, &s.x, &s.ffn_out)?;

        enc.end();
        cmd_buf.commit();
        unsafe { cmd_buf.waitUntilCompleted() };

        let mut out = vec![0.0f32; h];
        unsafe {
            let src = s.x.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), h);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::Forward;
    use crate::gguf::GgufFile;
    use crate::loader::Model;

    /// Validate a single GDN block end-to-end on Metal vs the CPU oracle.
    /// Uses block 0 of Qwen3.5-0.8B-F32 (the first GDN block, n_v=n_k=16).
    /// Compares the residual stream (s.x) after the block against what
    /// the CPU forward produces after running just block 0.
    #[test]
    fn metal_gdn_block_matches_cpu() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[metal-gdn] skipped — model missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        // Build inputs for block 0:
        //   * residual stream = the embedding row for token "Hello" (id 9419)
        // The CPU oracle and Metal driver both run from this same starting state.
        let token_id = 9419usize;
        let h = m.arch.hidden_size as usize;
        let embed =
            crate::codec::dequant_to_f32(m.token_embd, g.slice(m.token_embd)).expect("embed");
        let initial_x: Vec<f32> = embed[token_id * h..(token_id + 1) * h].to_vec();

        // CPU oracle: run Forward through one block manually.
        // We reuse Forward::single_token but only after the embedding
        // step; easier path is to just replicate the block in CPU code,
        // matching what forward.rs does. But that's a duplication risk.
        // Instead: call the public `Forward::single_token` and intercept
        // by limiting blocks. Forward doesn't expose that, so we carve
        // out a CPU-block helper here that mirrors forward.rs:single_token's
        // inner block loop for block 0 only.
        //
        // For the validation we rely on Forward::single_token computing
        // the same x state (post-block-0) — but it doesn't expose that.
        // Simplest: inline the block 0 computation here using the same
        // primitives. Given block 0 of 0.8B is a GDN block, this reads
        // exactly like the GDN inner of forward.rs.
        let cpu_x_after_block0 = run_cpu_block0_for_test(&g, &m, &initial_x);

        // Metal: build a fresh session, plant initial_x in the residual
        // stream, run block 0.
        let mut s = MetalSession::fresh(&ctx, &mm, 4096).expect("session");
        let mf = MetalForward::new(&ctx, &mm);
        mf.set_residual_for_test(&mut s, &initial_x);
        let metal_x = mf
            .run_one_gdn_block_for_test(0, 0, &mut s)
            .expect("metal block 0");

        let max_abs = metal_x
            .iter()
            .zip(cpu_x_after_block0.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot: f64 = metal_x
            .iter()
            .zip(cpu_x_after_block0.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let na: f64 = metal_x.iter().map(|v| (*v as f64).powi(2)).sum();
        let nb: f64 = cpu_x_after_block0.iter().map(|v| (*v as f64).powi(2)).sum();
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!("[metal-gdn-block0] hidden={h} max|Δ|={max_abs:.2e} cos={cos:.6}");
        assert!(max_abs < 1e-3, "block-0 drift {max_abs}");
        assert!(cos > 0.9999, "block-0 cos {cos}");
    }

    /// **End-to-end Metal forward** validated against `llm`/`llama_core`'s
    /// snapshot dump (which uses llama.cpp under the hood and is what
    /// the CPU oracle is also validated against). One token, one
    /// command buffer, all 24 blocks of Qwen3.5-0.8B-F32 chained.
    #[test]
    fn metal_single_token_matches_cpu_oracle() {
        let model_path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        let oracle_path = "/tmp/qwen-oracle/hello_t0.f32";
        if !std::path::Path::new(model_path).exists() || !std::path::Path::new(oracle_path).exists()
        {
            eprintln!("[metal-e2e] skipped — fixtures missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        let oracle_bytes = std::fs::read(oracle_path).expect("read oracle");
        let n = oracle_bytes.len() / 4;
        let oracle: Vec<f32> = (0..n)
            .map(|i| f32::from_le_bytes(oracle_bytes[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();

        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        assert_eq!(oracle.len(), m.arch.vocab_size as usize);

        // Tokenize "Hello" → 9419.
        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let ids = tok.encode("Hello", false).expect("tokenize");
        eprintln!("[metal-e2e] 'Hello' -> {ids:?}");
        assert_eq!(ids.len(), 1);

        let mut s = MetalSession::fresh(&ctx, &mm, 4096).expect("session");
        let mf = MetalForward::new(&ctx, &mm);
        let t = std::time::Instant::now();
        let logits = mf.single_token(ids[0], 0, &mut s).expect("forward");
        let ms = t.elapsed().as_secs_f64() * 1e3;

        let mut max_abs = 0.0f32;
        let mut argmax_ours = 0usize;
        let mut argmax_oracle = 0usize;
        let mut max_ours = f32::NEG_INFINITY;
        let mut max_oracle = f32::NEG_INFINITY;
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
            "[metal-e2e] {ms:.1}ms — argmax: ours={argmax_ours} ({:.4}) | oracle={argmax_oracle} ({:.4}) | max|Δ|={max_abs:.4} cos={cos:.6}",
            max_ours, max_oracle
        );
        assert_eq!(argmax_ours, argmax_oracle, "argmax disagreement");
        assert!(cos > 0.9999, "cos={cos} below threshold");
        assert!(max_abs < 0.05, "max|Δ|={max_abs} above noise floor");
    }

    /// **End-to-end Metal forward, multi-token**. Exercises position > 0
    /// in the attn block (RoPE, KV cache reads at multiple positions).
    /// Oracle: llm/llama_core's snapshot dump for "The quick brown fox
    /// jumps over the lazy dog" (9 tokens), Qwen3.5-0.8B-F32.
    #[test]
    fn metal_multi_token_matches_cpu_oracle() {
        let model_path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        let oracle_path = "/tmp/qwen-oracle/longprompt_t0.f32";
        if !std::path::Path::new(model_path).exists() || !std::path::Path::new(oracle_path).exists()
        {
            eprintln!("[metal-e2e-multi] skipped — fixtures missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };

        let oracle_bytes = std::fs::read(oracle_path).expect("read oracle");
        let n = oracle_bytes.len() / 4;
        let oracle: Vec<f32> = (0..n)
            .map(|i| f32::from_le_bytes(oracle_bytes[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();

        let g = GgufFile::open(model_path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
        let ids = tok
            .encode("The quick brown fox jumps over the lazy dog", false)
            .expect("tokenize");
        eprintln!("[metal-e2e-multi] {} tokens: {ids:?}", ids.len());
        assert_eq!(ids.len(), 9);

        let mut s = MetalSession::fresh(&ctx, &mm, ids.len() + 4).expect("session");
        let mf = MetalForward::new(&ctx, &mm);
        let t = std::time::Instant::now();
        let mut last = vec![];
        for (i, &tid) in ids.iter().enumerate() {
            last = mf.single_token(tid, i as u32, &mut s).expect("forward");
        }
        let total_ms = t.elapsed().as_secs_f64() * 1e3;

        let mut max_abs = 0.0f32;
        let mut argmax_ours = 0usize;
        let mut argmax_oracle = 0usize;
        let mut max_ours = f32::NEG_INFINITY;
        let mut max_oracle = f32::NEG_INFINITY;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..n {
            let d = (last[i] - oracle[i]).abs();
            max_abs = max_abs.max(d);
            if last[i] > max_ours {
                max_ours = last[i];
                argmax_ours = i;
            }
            if oracle[i] > max_oracle {
                max_oracle = oracle[i];
                argmax_oracle = i;
            }
            dot += last[i] as f64 * oracle[i] as f64;
            na += (last[i] as f64).powi(2);
            nb += (oracle[i] as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!(
            "[metal-e2e-multi] {total_ms:.1}ms ({:.1}ms/token) — argmax: ours={argmax_ours} ({:.4}) | oracle={argmax_oracle} ({:.4}) | max|Δ|={max_abs:.4} cos={cos:.6}",
            total_ms / ids.len() as f64,
            max_ours,
            max_oracle
        );
        assert_eq!(argmax_ours, argmax_oracle, "argmax disagreement");
        assert!(cos > 0.9999, "cos={cos} below threshold");
    }

    /// Validate a single full-attention block end-to-end on Metal vs the
    /// CPU oracle. Uses block 3 of Qwen3.5-0.8B-F32 (the first attn block,
    /// n_q=8, n_kv=2, head_dim=256, 4:1 GQA).
    #[test]
    fn metal_attn_block_matches_cpu() {
        let path = "/Users/tito/models/Qwen3.5-0.8B.F32.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[metal-attn] skipped — model missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

        // Inputs: a synthetic but fixed residual stream (no need for a
        // real model state for block-level validation; we just need
        // identical inputs to CPU and GPU paths).
        let h = m.arch.hidden_size as usize;
        let initial_x: Vec<f32> = (0..h).map(|i| ((i % 31) as f32 - 15.0) * 0.02).collect();
        let position: u32 = 0;
        let attn_block_idx = 3usize; // first attn block in 0.8B
                                     // attn_idx_in_session is the 0-indexed count among ATTN blocks
                                     // before this one. block 3 is the first attn block, so 0.
        let attn_idx_in_session = 0usize;

        // CPU reference: replicate exactly what forward.rs:attn_step does.
        let cpu_x = run_cpu_attn_block_for_test(&g, &m, &initial_x, attn_block_idx, position);

        // Metal.
        let mut s = MetalSession::fresh(&ctx, &mm, 4096).expect("session");
        let mf = MetalForward::new(&ctx, &mm);
        mf.set_residual_for_test(&mut s, &initial_x);
        let metal_x = mf
            .run_one_attn_block_for_test(attn_block_idx, attn_idx_in_session, position, &mut s)
            .expect("metal attn block");

        let max_abs = metal_x
            .iter()
            .zip(cpu_x.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot: f64 = metal_x
            .iter()
            .zip(cpu_x.iter())
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let na: f64 = metal_x.iter().map(|v| (*v as f64).powi(2)).sum();
        let nb: f64 = cpu_x.iter().map(|v| (*v as f64).powi(2)).sum();
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!("[metal-attn-block3] hidden={h} max|Δ|={max_abs:.2e} cos={cos:.6}");
        assert!(max_abs < 1e-3, "attn block-3 drift {max_abs}");
        assert!(cos > 0.9999, "attn block-3 cos {cos}");
    }

    /// CPU reference: full-attn block (norm → attn → residual → post_norm
    /// → FFN → residual) replicated inline. Mirrors forward.rs's
    /// single_token block flow for an attn block.
    fn run_cpu_attn_block_for_test(
        gguf: &GgufFile,
        model: &Model<'_>,
        initial_x: &[f32],
        block_idx: usize,
        position: u32,
    ) -> Vec<f32> {
        let block = &model.blocks[block_idx];
        let ab = match block {
            crate::loader::Block::Attn(a) => a,
            _ => panic!("not an attn block"),
        };
        let arch = &model.arch;
        let h = arch.hidden_size as usize;
        let head_dim = arch.attn_head_dim as usize;
        let n_q = arch.n_q_heads as usize;
        let n_kv = arch.n_kv_heads as usize;
        let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;
        let group = n_q / n_kv;
        let theta = arch.rope_theta;

        let mut x = initial_x.to_vec();

        // Pre-mixer norm.
        let attn_norm_w =
            crate::codec::dequant_to_f32(ab.attn_norm, gguf.slice(ab.attn_norm)).unwrap();
        let cur = crate::forward::rms_norm_pub(&x, &attn_norm_w, super::RMS_EPS);

        // Q projection (2 * q_dim) → split.
        let q_w = crate::codec::dequant_to_f32(ab.q, gguf.slice(ab.q)).unwrap();
        let q_full = crate::forward::mat_vec_pub(&q_w, h, 2 * q_dim, &cur);
        let mut qcur = vec![0.0f32; q_dim];
        let mut gate = vec![0.0f32; q_dim];
        for hi in 0..n_q {
            let src = &q_full[hi * 2 * head_dim..(hi + 1) * 2 * head_dim];
            qcur[hi * head_dim..(hi + 1) * head_dim].copy_from_slice(&src[..head_dim]);
            gate[hi * head_dim..(hi + 1) * head_dim].copy_from_slice(&src[head_dim..]);
        }

        // Q-norm.
        let qnorm_w = crate::codec::dequant_to_f32(ab.q_norm, gguf.slice(ab.q_norm)).unwrap();
        for hi in 0..n_q {
            let s = hi * head_dim;
            let n = crate::forward::rms_norm_pub(&qcur[s..s + head_dim], &qnorm_w, super::RMS_EPS);
            qcur[s..s + head_dim].copy_from_slice(&n);
        }

        // K, V.
        let k_w = crate::codec::dequant_to_f32(ab.k, gguf.slice(ab.k)).unwrap();
        let v_w = crate::codec::dequant_to_f32(ab.v, gguf.slice(ab.v)).unwrap();
        let mut kcur = crate::forward::mat_vec_pub(&k_w, h, kv_dim, &cur);
        let vcur = crate::forward::mat_vec_pub(&v_w, h, kv_dim, &cur);

        // K-norm.
        let knorm_w = crate::codec::dequant_to_f32(ab.k_norm, gguf.slice(ab.k_norm)).unwrap();
        for hi in 0..n_kv {
            let s = hi * head_dim;
            let n = crate::forward::rms_norm_pub(&kcur[s..s + head_dim], &knorm_w, super::RMS_EPS);
            kcur[s..s + head_dim].copy_from_slice(&n);
        }

        // RoPE on Q and K (NEOX/IMROPE pairing for text positions).
        rope_in_place_local(&mut qcur, n_q, head_dim, n_rot, position, theta);
        rope_in_place_local(&mut kcur, n_kv, head_dim, n_rot, position, theta);

        // Single-token cache: KV is just the current step.
        // Attention with one position (position itself).
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; q_dim];
        for qh in 0..n_q {
            let kvh = qh / group;
            let q_slice = &qcur[qh * head_dim..(qh + 1) * head_dim];
            let k_slice = &kcur[kvh * head_dim..(kvh + 1) * head_dim];
            let v_slice = &vcur[kvh * head_dim..(kvh + 1) * head_dim];

            let mut s = 0.0f32;
            for i in 0..head_dim {
                s += q_slice[i] * k_slice[i];
            }
            let _score = s * scale;
            // softmax over a single value = 1.0 → out = v
            for i in 0..head_dim {
                attn_out[qh * head_dim + i] = v_slice[i];
            }
        }

        // Apply gated-attention sigmoid gate.
        for i in 0..attn_out.len() {
            let sg = 1.0 / (1.0 + (-gate[i]).exp());
            attn_out[i] *= sg;
        }

        // Output projection.
        let o_w = crate::codec::dequant_to_f32(ab.o, gguf.slice(ab.o)).unwrap();
        let mixer_out = crate::forward::mat_vec_pub(&o_w, q_dim, h, &attn_out);

        // Residual #1.
        for (xi, mo) in x.iter_mut().zip(mixer_out.iter()) {
            *xi += *mo;
        }

        // Pre-FFN norm.
        let post_norm_w = crate::codec::dequant_to_f32(
            ab.post_attention_norm,
            gguf.slice(ab.post_attention_norm),
        )
        .unwrap();
        let cur = crate::forward::rms_norm_pub(&x, &post_norm_w, super::RMS_EPS);

        // FFN.
        let f = arch.intermediate_size as usize;
        let g_w = crate::codec::dequant_to_f32(ab.ffn_gate, gguf.slice(ab.ffn_gate)).unwrap();
        let u_w = crate::codec::dequant_to_f32(ab.ffn_up, gguf.slice(ab.ffn_up)).unwrap();
        let d_w = crate::codec::dequant_to_f32(ab.ffn_down, gguf.slice(ab.ffn_down)).unwrap();
        let gated = crate::forward::mat_vec_pub(&g_w, h, f, &cur);
        let upped = crate::forward::mat_vec_pub(&u_w, h, f, &cur);
        let mut hidden = vec![0.0f32; f];
        for i in 0..f {
            let g = gated[i];
            let silu_g = g / (1.0 + (-g).exp());
            hidden[i] = silu_g * upped[i];
        }
        let ffn_out = crate::forward::mat_vec_pub(&d_w, f, h, &hidden);

        // Residual #2.
        for (xi, fo) in x.iter_mut().zip(ffn_out.iter()) {
            *xi += *fo;
        }
        x
    }

    fn rope_in_place_local(
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

    /// CPU reference: replicate exactly what forward.rs:single_token does
    /// for block 0 of Qwen3.5-0.8B (a GDN block), starting from
    /// `initial_x` (the token embedding).
    fn run_cpu_block0_for_test(gguf: &GgufFile, model: &Model<'_>, initial_x: &[f32]) -> Vec<f32> {
        let f = Forward::new(gguf, model);
        // Use Forward's public single_token but at position 0 with token
        // id derived from initial_x: too indirect. Easier path: take the
        // raw GDN-block computation and replicate it inline.
        //
        // forward.rs's `single_token` already does exactly this. The
        // simplest validation is to run it in full and compare logits —
        // but that requires the full attn block, which Metal doesn't
        // have yet. So instead we replicate just block 0 here.
        //
        // Block 0 in 0.8B is a GDN block. The CPU computation is in
        // forward.rs lines 290-510-ish. We recreate the same flow with
        // the public helpers in `forward`.
        let mut x = initial_x.to_vec();
        let block = &model.blocks[0];
        let gb = match block {
            crate::loader::Block::Gdn(g) => g,
            _ => panic!("block 0 is not GDN"),
        };

        // Pre-mixer norm.
        let attn_norm_w =
            crate::codec::dequant_to_f32(gb.attn_norm, gguf.slice(gb.attn_norm)).unwrap();
        let cur = crate::forward::rms_norm_pub(&x, &attn_norm_w, super::RMS_EPS);

        // GDN inner. The cleanest way to replicate is to hand-call
        // Forward's gdn_step. It takes a GdnState; we'll build a fresh
        // one and ignore the state output.
        let mut state = crate::forward::GdnState::fresh(model);
        let mixer_out = call_gdn_step_directly(&f, gb, &cur, &mut state);

        // Residual #1.
        for (xi, mi) in x.iter_mut().zip(mixer_out.iter()) {
            *xi += *mi;
        }

        // Pre-FFN norm.
        let post_norm_w = crate::codec::dequant_to_f32(
            gb.post_attention_norm,
            gguf.slice(gb.post_attention_norm),
        )
        .unwrap();
        let cur = crate::forward::rms_norm_pub(&x, &post_norm_w, super::RMS_EPS);

        // FFN.
        let arch = &model.arch;
        let h = arch.hidden_size as usize;
        let fdim = arch.intermediate_size as usize;
        let g_w = crate::codec::dequant_to_f32(gb.ffn_gate, gguf.slice(gb.ffn_gate)).unwrap();
        let u_w = crate::codec::dequant_to_f32(gb.ffn_up, gguf.slice(gb.ffn_up)).unwrap();
        let d_w = crate::codec::dequant_to_f32(gb.ffn_down, gguf.slice(gb.ffn_down)).unwrap();

        let gated = crate::forward::mat_vec_pub(&g_w, h, fdim, &cur);
        let upped = crate::forward::mat_vec_pub(&u_w, h, fdim, &cur);
        let mut hidden = vec![0.0f32; fdim];
        for i in 0..fdim {
            let g = gated[i];
            let silu_g = g / (1.0 + (-g).exp());
            hidden[i] = silu_g * upped[i];
        }
        let ffn_out = crate::forward::mat_vec_pub(&d_w, fdim, h, &hidden);

        // Residual #2.
        for (xi, fo) in x.iter_mut().zip(ffn_out.iter()) {
            *xi += *fo;
        }
        x
    }

    /// Use a private CPU-only path to run forward::Forward's gdn_step on
    /// block 0. We can't call it directly because it's private to
    /// Forward; the test re-implements the same math inline.
    fn call_gdn_step_directly(
        f: &Forward,
        gb: &crate::loader::GdnBlock,
        x: &[f32],
        state: &mut crate::forward::GdnState,
    ) -> Vec<f32> {
        // Public re-exports added below in forward.rs would let us avoid
        // this. For now, since Forward's gdn_step is private, we run the
        // *full* Forward::single_token and extract the post-block-0
        // residual. That requires a hook in Forward we don't have.
        //
        // Pragmatic shortcut: re-implement gdn_step here using the public
        // mat_vec_pub and matching the exact CPU forward path. This is
        // ~50 lines of duplication but isolates the test from Forward's
        // internals.
        let arch = &f.model.arch;
        let h = arch.hidden_size as usize;
        let n_v = arch.gdn_n_v_heads as usize;
        let n_k = arch.gdn_n_k_heads as usize;
        let head_dim = arch.gdn_head_dim as usize;
        let conv_dim = 2 * n_k * head_dim + n_v * head_dim;
        let conv_kernel = arch.gdn_conv_kernel as usize;
        let v_dim = n_v * head_dim;

        let qkv_w =
            crate::codec::dequant_to_f32(gb.in_proj_qkv, f.gguf.slice(gb.in_proj_qkv)).unwrap();
        let qkv = crate::forward::mat_vec_pub(&qkv_w, h, conv_dim, x);
        let z_w = crate::codec::dequant_to_f32(gb.in_proj_z, f.gguf.slice(gb.in_proj_z)).unwrap();
        let z = crate::forward::mat_vec_pub(&z_w, h, v_dim, x);

        let beta_w =
            crate::codec::dequant_to_f32(gb.beta_proj, f.gguf.slice(gb.beta_proj)).unwrap();
        let mut beta = crate::forward::mat_vec_pub(&beta_w, h, n_v, x);
        for v in beta.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
        let alpha_w =
            crate::codec::dequant_to_f32(gb.alpha_proj, f.gguf.slice(gb.alpha_proj)).unwrap();
        let mut alpha = crate::forward::mat_vec_pub(&alpha_w, h, n_v, x);
        let dt = crate::codec::dequant_to_f32(gb.dt_bias, f.gguf.slice(gb.dt_bias)).unwrap();
        for (a, &dti) in alpha.iter_mut().zip(dt.iter()) {
            *a += dti;
        }
        let a_log = crate::codec::dequant_to_f32(gb.a_log, f.gguf.slice(gb.a_log)).unwrap();
        let mut g = vec![0.0f32; n_v];
        for i in 0..n_v {
            let sp = if alpha[i] > 20.0 {
                alpha[i]
            } else if alpha[i] < -20.0 {
                alpha[i].exp()
            } else {
                (1.0 + alpha[i].exp()).ln()
            };
            g[i] = sp * a_log[i];
        }

        let conv_w = crate::codec::dequant_to_f32(gb.conv1d, f.gguf.slice(gb.conv1d)).unwrap();
        let kmin1 = conv_kernel - 1;
        let mut conv_input = vec![0.0f32; conv_kernel * conv_dim];
        for t in 0..kmin1 {
            conv_input[t * conv_dim..(t + 1) * conv_dim]
                .copy_from_slice(&state.conv[0][t * conv_dim..(t + 1) * conv_dim]);
        }
        conv_input[kmin1 * conv_dim..].copy_from_slice(&qkv);

        let mut conv_out = vec![0.0f32; conv_dim];
        for c in 0..conv_dim {
            let mut sm = 0.0f32;
            for k in 0..conv_kernel {
                sm += conv_w[c * conv_kernel + k] * conv_input[k * conv_dim + c];
            }
            conv_out[c] = sm / (1.0 + (-sm).exp());
        }
        // (slide buffer; not asked for in test, just for completeness it
        // would happen here, but state.conv is local to this fn here)
        for t in 0..kmin1 - 1 {
            for i in 0..conv_dim {
                state.conv[0][t * conv_dim + i] = state.conv[0][(t + 1) * conv_dim + i];
            }
        }
        let last = (kmin1 - 1) * conv_dim;
        state.conv[0][last..last + conv_dim].copy_from_slice(&qkv);

        // Split conv_out → q,k,v.
        let q_full = conv_out[0..n_k * head_dim].to_vec();
        let k_full = conv_out[n_k * head_dim..2 * n_k * head_dim].to_vec();
        let v_full = conv_out[2 * n_k * head_dim..].to_vec();

        // Per-head L2 norm of Q and K.
        let mut q_full = q_full.clone();
        let mut k_full = k_full.clone();
        for hi in 0..n_k {
            let off = hi * head_dim;
            let sq: f32 = q_full[off..off + head_dim].iter().map(|v| v * v).sum();
            let scale = 1.0 / sq.sqrt().max(super::RMS_EPS);
            for i in 0..head_dim {
                q_full[off + i] *= scale;
            }
            let sq: f32 = k_full[off..off + head_dim].iter().map(|v| v * v).sum();
            let scale = 1.0 / sq.sqrt().max(super::RMS_EPS);
            for i in 0..head_dim {
                k_full[off + i] *= scale;
            }
        }

        // Recurrence.
        let mut o = vec![0.0f32; v_dim];
        for hi in 0..n_v {
            let hk = hi % n_k;
            let s_off = hi * head_dim * head_dim;
            let q_h = &q_full[hk * head_dim..(hk + 1) * head_dim];
            let k_h = &k_full[hk * head_dim..(hk + 1) * head_dim];
            let v_h = &v_full[hi * head_dim..(hi + 1) * head_dim];
            let g_h = g[hi].exp();
            let b_h = beta[hi];
            for j in 0..head_dim * head_dim {
                state.ssm[0][s_off + j] *= g_h;
            }
            let mut sk = vec![0.0f32; head_dim];
            for dv in 0..head_dim {
                let mut sm = 0.0f32;
                for dk in 0..head_dim {
                    sm += state.ssm[0][s_off + dv * head_dim + dk] * k_h[dk];
                }
                sk[dv] = sm;
            }
            for dv in 0..head_dim {
                let coeff = b_h * (v_h[dv] - sk[dv]);
                for dk in 0..head_dim {
                    state.ssm[0][s_off + dv * head_dim + dk] += coeff * k_h[dk];
                }
            }
            let scale = 1.0 / (head_dim as f32).sqrt();
            for dv in 0..head_dim {
                let mut sm = 0.0f32;
                for dk in 0..head_dim {
                    sm += state.ssm[0][s_off + dv * head_dim + dk] * q_h[dk];
                }
                o[hi * head_dim + dv] = sm * scale;
            }
        }

        // RMSNormGated: norm(o) * silu(z), per-head.
        let norm_w = crate::codec::dequant_to_f32(gb.norm, f.gguf.slice(gb.norm)).unwrap();
        let mut gated = vec![0.0f32; v_dim];
        for hi in 0..n_v {
            let off = hi * head_dim;
            let normed =
                crate::forward::rms_norm_pub(&o[off..off + head_dim], &norm_w, super::RMS_EPS);
            for i in 0..head_dim {
                let zi = z[off + i];
                let silu_z = zi / (1.0 + (-zi).exp());
                gated[off + i] = normed[i] * silu_z;
            }
        }

        // Output projection.
        let out_w = crate::codec::dequant_to_f32(gb.out_proj, f.gguf.slice(gb.out_proj)).unwrap();
        crate::forward::mat_vec_pub(&out_w, v_dim, h, &gated)
    }
}
