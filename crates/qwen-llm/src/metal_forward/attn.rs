//! Attention layer execution and KV-cache handling.

use super::*;

pub(super) fn prefill_attn_fused_qkv_g8_enabled() -> bool {
    matches!(
        std::env::var("QWEN_PREFILL_ATTN_FUSED_QKV_G8").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

pub(crate) fn kv_cache_dtype_for_arch(arch: &crate::model::Arch) -> GgmlType {
    let enabled = kv_q8_flag();
    let group = (arch.n_q_heads / arch.n_kv_heads.max(1)) as usize;
    if enabled
        && arch.attn_head_dim == 256
        && ((arch.kind == ArchKind::Dense && group == 6)
            || (arch.kind == ArchKind::Moe && group == 8))
    {
        GgmlType::Q8_0
    } else {
        GgmlType::F16
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum SnapshotKvStorageKind {
    None = 0,
    F16 = 1,
    Q8_0 = 2,
}

impl<'a> MetalForward<'a> {
    pub fn single_token_profiled_concurrent_attn_dense(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(Vec<f32>, TokenProfile), MfError> {
        let arch = &self.model.arch;
        if arch.kind != ArchKind::Dense {
            return Err(MfError::UnsupportedMoe);
        }
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        let t_total = std::time::Instant::now();

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");

        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_get_rows_f32(
                self.ctx,
                &enc,
                &self.model.token_embd,
                &session.ids_buf,
                &session.x,
                1,
                h,
            )?;
            enc.end();
        }

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for block in &self.model.blocks {
            match block {
                MetalBlock::Gdn(_) => {
                    let enc = KernelEncoder::begin(&cmd_buf);
                    self.encode_block(
                        &enc,
                        0,
                        block,
                        &mut gdn_idx,
                        &mut attn_idx,
                        position,
                        session,
                    )?;
                    enc.end();
                }
                MetalBlock::Attn(a) => {
                    let i = attn_idx;
                    attn_idx += 1;
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &a.attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin_concurrent(&cmd_buf);
                        self.encode_attn_front_projections(&enc, a, session)?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        self.encode_attn_after_projections(&enc, a, i, position, session)?;
                        self.encode_post_mixer_ffn(&enc, block, session)?;
                        enc.end();
                    }
                }
            }
        }

        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_rms_norm_mul_f32(
                self.ctx,
                &enc,
                &session.x,
                &self.model.output_norm,
                &session.h,
                RMS_EPS,
            )?;
            encode_mat_vec_dispatch(
                self.ctx,
                &enc,
                &self.model.lm_head,
                &session.h,
                &session.logits,
                arch.hidden_size as usize,
                arch.vocab_size as usize,
            )?;
            enc.end();
        }

        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;
        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let gpu_kernel_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;

        let mut out = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((
            out,
            TokenProfile {
                cpu_encode_ms,
                cpu_to_gpu_complete_ms,
                gpu_kernel_ms,
                total_ms,
                moe_cpu_route_ms: 0.0,
                moe_cmd_count: 1,
            },
        ))
    }

    pub(super) fn encode_attn_front_projections(
        &self,
        enc: &KernelEncoder,
        ab: &MetalAttnBlock,
        s: &mut MetalSession,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let head_dim = arch.attn_head_dim as usize;
        let n_q = arch.n_q_heads as usize;
        let n_kv = arch.n_kv_heads as usize;
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;

        encode_mat_vec_dispatch(self.ctx, enc, &ab.q, &s.h, &s.attn_q_full, h, 2 * q_dim)?;
        encode_mat_vec_dispatch(self.ctx, enc, &ab.k, &s.h, &s.attn_k_now, h, kv_dim)?;
        encode_mat_vec_dispatch(self.ctx, enc, &ab.v, &s.h, &s.attn_v_now, h, kv_dim)?;
        Ok(())
    }

    /// Complete one attention mixer from already-populated Q/K/V projection
    /// buffers. Exposed for diagnostics that batch only the immutable-weight
    /// front projections while retaining sequence-private KV state.
    #[doc(hidden)]
    pub fn encode_attn_after_projections(
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

        let fused_qk_norm_rope = decode_qk_norm_rope_fused_enabled()
            && decode_rope_pair_enabled()
            && decode_attn_sigmoid_mul_enabled();
        if fused_qk_norm_rope {
            encode_qk_rms_norm_rope_f32_packed_consecutive(
                self.ctx,
                enc,
                &s.attn_q_full,
                &ab.q_norm,
                &s.attn_q_normed,
                &s.attn_k_now,
                &ab.k_norm,
                &s.attn_k_normed,
                1,
                n_q,
                n_kv,
                head_dim,
                n_rot,
                position,
                RMS_EPS,
                arch.rope_theta,
            )?;
        } else {
            // v0.432: the default path reads the interleaved q_proj output
            // directly. The compact-gate rollback still keeps the split.
            if decode_attn_sigmoid_mul_enabled() {
                encode_rms_norm_batched_src_strided_f32(
                    self.ctx,
                    enc,
                    &s.attn_q_full,
                    &ab.q_norm,
                    &s.attn_q_normed,
                    n_q,
                    head_dim,
                    2 * head_dim,
                    0,
                    RMS_EPS,
                )?;
            } else {
                encode_split_q_gate_f32(
                    self.ctx,
                    enc,
                    &s.attn_q_full,
                    &s.attn_q,
                    &s.attn_gate,
                    n_q,
                    head_dim,
                )?;
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
            }
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
            if decode_rope_pair_enabled() {
                encode_rope_neox_pair_f32(
                    self.ctx,
                    enc,
                    &s.attn_q_normed,
                    &s.attn_k_normed,
                    n_q,
                    n_kv,
                    head_dim,
                    n_rot,
                    position,
                    arch.rope_theta,
                )?;
            } else {
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
            }
        }
        let kv_dst_off = usize::try_from(checked_u64_mul(
            position as u64,
            kv_dim as u64,
            "kv dst offset overflow",
        )?)
        .map_err(|_| MetalError::BadShape {
            kernel: "attn_step",
            detail: "kv dst offset does not fit usize".into(),
        })?;
        match s.kv_k[attn_i].dtype {
            GgmlType::F16 => encode_scatter_offset_f32_to_f16_kv(
                self.ctx,
                enc,
                &s.attn_k_normed,
                &s.attn_v_now,
                &s.kv_k[attn_i],
                &s.kv_v[attn_i],
                kv_dst_off,
                kv_dim,
            )?,
            GgmlType::Q8_0 => encode_scatter_offset_f32_to_q8_0_kv(
                self.ctx,
                enc,
                &s.attn_k_normed,
                &s.attn_v_now,
                &s.kv_k[attn_i],
                &s.kv_v[attn_i],
                kv_dst_off,
                kv_dim,
            )?,
            other => {
                return Err(MfError::UnsupportedDtype {
                    name: "attention KV cache".into(),
                    dtype: other,
                });
            }
        }
        s.kv_n_pos[attn_i] = position as usize + 1;

        const V4_HEAD_DIM: usize = 256;
        let group = n_q / n_kv;
        let use_v4 = head_dim == V4_HEAD_DIM && matches!(group, 4 | 6 | 8 | 16);
        if use_v4 {
            let nwg = attn_v4_choose_nwg(s.kv_n_pos[attn_i], group);
            let tile_c = attn_v4_choose_tile_c(s.kv_n_pos[attn_i], group);
            encode_attn_decode_v4_f32(
                self.ctx,
                enc,
                &s.attn_q_normed,
                &s.kv_k[attn_i],
                &s.kv_v[attn_i],
                &s.attn_v4_o_partial,
                &s.attn_v4_ml_partial,
                &s.attn_o,
                n_q,
                n_kv,
                head_dim,
                s.kv_n_pos[attn_i],
                nwg,
                tile_c,
            )?;
        } else {
            encode_attn_decode_f16kv_f32(
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
        }

        if decode_attn_sigmoid_mul_enabled() {
            // Strided gate read from the interleaved q_proj output (see the
            // v0.432 comment at the q-norm above).
            encode_sigmoid_mul_gate_strided_f32(
                self.ctx,
                enc,
                &s.attn_q_full,
                &s.attn_o,
                &s.attn_o,
                n_q,
                head_dim,
                2 * head_dim,
                head_dim,
            )?;
        } else {
            encode_sigmoid_f32(self.ctx, enc, &s.attn_gate, &s.attn_q)?;
            encode_mul_f32(self.ctx, enc, &s.attn_o, &s.attn_q, &s.attn_o)?;
        }
        encode_mat_vec_dispatch(self.ctx, enc, &ab.o, &s.attn_o, &s.mixer_out, q_dim, h)?;
        Ok(())
    }

    pub fn encode_attn(
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
        encode_mat_vec_dispatch(self.ctx, enc, &ab.q, &s.h, &s.attn_q_full, h, 2 * q_dim)?;

        let fused_qk_norm_rope = decode_qk_norm_rope_fused_enabled()
            && decode_rope_pair_enabled()
            && decode_attn_sigmoid_mul_enabled();
        if fused_qk_norm_rope {
            // K/V must exist before the joint normalization/rotation dispatch.
            encode_mat_vec_dispatch(self.ctx, enc, &ab.k, &s.h, &s.attn_k_now, h, kv_dim)?;
            encode_mat_vec_dispatch(self.ctx, enc, &ab.v, &s.h, &s.attn_v_now, h, kv_dim)?;
            encode_qk_rms_norm_rope_f32_packed_consecutive(
                self.ctx,
                enc,
                &s.attn_q_full,
                &ab.q_norm,
                &s.attn_q_normed,
                &s.attn_k_now,
                &ab.k_norm,
                &s.attn_k_normed,
                1,
                n_q,
                n_kv,
                head_dim,
                n_rot,
                position,
                RMS_EPS,
                arch.rope_theta,
            )?;
        } else {
            // v0.432: the default path reads Q directly from the interleaved
            // projection. The compact-gate rollback still keeps the split.
            if decode_attn_sigmoid_mul_enabled() {
                encode_rms_norm_batched_src_strided_f32(
                    self.ctx,
                    enc,
                    &s.attn_q_full,
                    &ab.q_norm,
                    &s.attn_q_normed,
                    n_q,
                    head_dim,
                    2 * head_dim,
                    0,
                    RMS_EPS,
                )?;
            } else {
                encode_split_q_gate_f32(
                    self.ctx,
                    enc,
                    &s.attn_q_full,
                    &s.attn_q,
                    &s.attn_gate,
                    n_q,
                    head_dim,
                )?;
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
            }

            // K/V projections retain their prior ordering on the rollback path.
            encode_mat_vec_dispatch(self.ctx, enc, &ab.k, &s.h, &s.attn_k_now, h, kv_dim)?;
            encode_mat_vec_dispatch(self.ctx, enc, &ab.v, &s.h, &s.attn_v_now, h, kv_dim)?;
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
            if decode_rope_pair_enabled() {
                encode_rope_neox_pair_f32(
                    self.ctx,
                    enc,
                    &s.attn_q_normed,
                    &s.attn_k_normed,
                    n_q,
                    n_kv,
                    head_dim,
                    n_rot,
                    position,
                    arch.rope_theta,
                )?;
            } else {
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
            }
        }

        // KV cache append. v1 enforces strict-monotonic-from-zero;
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
        // KV cache append: F32 source → F16 destination (cache is F16 to
        // halve attention bandwidth at long context). Fused K+V scatter
        // (Bulk-API principle): one dispatch writes both, saving 16 dispatches/token.
        let kv_dst_off = usize::try_from(checked_u64_mul(
            position as u64,
            kv_dim as u64,
            "kv dst offset overflow",
        )?)
        .map_err(|_| MetalError::BadShape {
            kernel: "attn_step_q8",
            detail: "kv dst offset does not fit usize".into(),
        })?;
        match s.kv_k[attn_i].dtype {
            GgmlType::F16 => encode_scatter_offset_f32_to_f16_kv(
                self.ctx,
                enc,
                &s.attn_k_normed,
                &s.attn_v_now,
                &s.kv_k[attn_i],
                &s.kv_v[attn_i],
                kv_dst_off,
                kv_dim,
            )?,
            GgmlType::Q8_0 => encode_scatter_offset_f32_to_q8_0_kv(
                self.ctx,
                enc,
                &s.attn_k_normed,
                &s.attn_v_now,
                &s.kv_k[attn_i],
                &s.kv_v[attn_i],
                kv_dst_off,
                kv_dim,
            )?,
            other => {
                return Err(MfError::UnsupportedDtype {
                    name: "attention KV cache".into(),
                    dtype: other,
                });
            }
        }
        s.kv_n_pos[attn_i] = position as usize + 1;

        // (8) Fused attention decode: scoring + softmax + V-aggregate.
        //
        // Selection: v4 (GQA-dedup + online softmax + split-K) when the
        // shape matches its hardcoded constants (head_dim=256, GROUP in
        // {4,6,8,16}). Falls back to f16kv naive kernel for other shapes.
        //
        // v4 gives 2-8× speedup over naive on the 27B shape AND removes
        // the n_pos ≤ ~7000 correctness cliff (naive's threadgroup-mem
        // scores buffer caps out around there).
        const V4_HEAD_DIM: usize = 256;
        let group = n_q / n_kv;
        let use_v4 = head_dim == V4_HEAD_DIM && matches!(group, 4 | 6 | 8 | 16);
        if use_v4 {
            let nwg = attn_v4_choose_nwg(s.kv_n_pos[attn_i], group);
            let tile_c = attn_v4_choose_tile_c(s.kv_n_pos[attn_i], group);
            encode_attn_decode_v4_f32(
                self.ctx,
                enc,
                &s.attn_q_normed,
                &s.kv_k[attn_i],
                &s.kv_v[attn_i],
                &s.attn_v4_o_partial,
                &s.attn_v4_ml_partial,
                &s.attn_o,
                n_q,
                n_kv,
                head_dim,
                s.kv_n_pos[attn_i],
                nwg,
                tile_c,
            )?;
        } else {
            encode_attn_decode_f16kv_f32(
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
        }

        // (9) Apply gated-attention sigmoid gate: attn_o *= sigmoid(gate).
        if decode_attn_sigmoid_mul_enabled() {
            encode_sigmoid_mul_gate_strided_f32(
                self.ctx,
                enc,
                &s.attn_q_full,
                &s.attn_o,
                &s.attn_o,
                n_q,
                head_dim,
                2 * head_dim,
                head_dim,
            )?;
        } else {
            encode_sigmoid_f32(self.ctx, enc, &s.attn_gate, &s.attn_q)?;
            encode_mul_f32(self.ctx, enc, &s.attn_o, &s.attn_q, &s.attn_o)?;
        }

        // (10) Output projection: q_dim → hidden.
        encode_mat_vec_dispatch(self.ctx, enc, &ab.o, &s.attn_o, &s.mixer_out, q_dim, h)?;
        Ok(())
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
        encode_mat_vec_dispatch(self.ctx, &enc, &ab.ffn_gate, &s.h, &s.ffn_gate, h, f)?;
        encode_mat_vec_dispatch(self.ctx, &enc, &ab.ffn_up, &s.h, &s.ffn_up, h, f)?;
        encode_silu_mul_f32(self.ctx, &enc, &s.ffn_gate, &s.ffn_up, &s.ffn_inner)?;
        encode_mat_vec_dispatch(self.ctx, &enc, &ab.ffn_down, &s.ffn_inner, &s.ffn_out, f, h)?;

        encode_add_inplace_f32(self.ctx, &enc, &s.x, &s.ffn_out)?;

        enc.end();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        let mut out = vec![0.0f32; h];
        unsafe {
            let src = s.x.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), h);
        }
        Ok(out)
    }
}
