//! Gated-DeltaNet layer execution.

use super::*;

pub(super) fn gdn_beta_projection_fused(gb: &MetalGdnBlock) -> bool {
    decode_gdn_fused_beta_proj_enabled()
        && gb.beta_proj.dtype == GgmlType::F32
        && !decode_gdn_noop_beta_enabled()
}

pub(super) fn decode_gdn_noop_qkv_enabled() -> bool {
    decode_gdn_noop_front_enabled() || decode_gdn_noop_qkv_flag()
}

pub(super) fn decode_gdn_noop_z_enabled() -> bool {
    decode_gdn_noop_front_enabled() || decode_gdn_noop_z_flag()
}

pub(super) fn decode_gdn_noop_beta_enabled() -> bool {
    decode_gdn_noop_front_enabled() || decode_gdn_noop_beta_flag()
}

pub(super) fn decode_gdn_noop_alpha_enabled() -> bool {
    decode_gdn_noop_front_enabled() || decode_gdn_noop_alpha_flag()
}

/// Install bench-only per-GDN-layer decay capture destinations.
pub fn gdn_alpha_census_capture_install(slots: Vec<(usize, MetalTensor)>) {
    GDN_ALPHA_CENSUS_CAPTURE.with(|capture| *capture.borrow_mut() = Some(slots));
}

/// Uninstall bench-only GDN decay capture.
pub fn gdn_alpha_census_capture_uninstall() {
    GDN_ALPHA_CENSUS_CAPTURE.with(|capture| *capture.borrow_mut() = None);
}

pub(super) fn gdn_alpha_census_capture_slot(gdn_index: usize) -> Option<MetalTensor> {
    GDN_ALPHA_CENSUS_CAPTURE.with(|capture| {
        capture
            .borrow()
            .as_ref()?
            .iter()
            .find(|(index, _)| *index == gdn_index)
            .map(|(_, destination)| destination.clone())
    })
}

impl<'a> MetalForward<'a> {
    pub fn single_token_profiled_concurrent_gdn_dense(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(Vec<f32>, TokenProfile), MfError> {
        self.single_token_profiled_concurrent_gdn_dense_with_tail(token_id, position, session, true)
    }

    pub(super) fn single_token_profiled_concurrent_gdn_dense_with_tail(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        emit_logits: bool,
    ) -> Result<(Vec<f32>, TokenProfile), MfError> {
        session.ensure_usable()?;
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
        let cmd_buf = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| MfError::CommandBuffer {
                status: "unavailable".into(),
                error: "Metal did not provide a command buffer".into(),
            })?;

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
                MetalBlock::Gdn(g) => {
                    let i = gdn_idx;
                    gdn_idx += 1;
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin_concurrent(&cmd_buf);
                        self.encode_gdn_front_projections(&enc, g, session)?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        self.encode_gdn_after_projections(&enc, g, i, session)?;
                        self.encode_post_mixer_ffn(&enc, block, session)?;
                        enc.end();
                    }
                }
                MetalBlock::Attn(_) => {
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
            }
        }

        if emit_logits {
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
        let status = cmd_buf.status();
        let error = cmd_buf.error();
        if status != MTLCommandBufferStatus::Completed || error.is_some() {
            session.poison("concurrent dense single-token command failed");
            return Err(MfError::CommandBuffer {
                status: format!("{status:?}"),
                error: format!("{error:?}"),
            });
        }
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let gpu_kernel_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;

        let out = if emit_logits {
            let mut out = vec![0.0f32; arch.vocab_size as usize];
            unsafe {
                let src = session.logits.buffer.contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
            }
            out
        } else {
            Vec::new()
        };
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

    pub fn single_token_argmax_profiled_concurrent_gdn_dense(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(i32, TokenProfile), MfError> {
        self.single_token_argmax_profiled_concurrent_gdn_dense_with_reduction(
            token_id,
            position,
            session,
            ArgmaxReduction::SpeculativeLowest,
        )
    }

    pub(super) fn single_token_argmax_profiled_concurrent_gdn_dense_with_reduction(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        reduction: ArgmaxReduction,
    ) -> Result<(i32, TokenProfile), MfError> {
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

        let ids_buf = session.ids_buf.clone();
        let argmax_tok = session.argmax_tok.clone();

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");

        {
            let enc = KernelEncoder::begin(&cmd_buf);
            encode_get_rows_f32(
                self.ctx,
                &enc,
                &self.model.token_embd,
                &ids_buf,
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
                MetalBlock::Gdn(g) => {
                    let i = gdn_idx;
                    gdn_idx += 1;
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin_concurrent(&cmd_buf);
                        self.encode_gdn_front_projections(&enc, g, session)?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        self.encode_gdn_after_projections(&enc, g, i, session)?;
                        self.encode_post_mixer_ffn(&enc, block, session)?;
                        enc.end();
                    }
                }
                MetalBlock::Attn(_) => {
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
                h,
                arch.vocab_size as usize,
            )?;
            encode_argmax_reduction(
                self.ctx,
                &enc,
                &session.logits,
                &argmax_tok,
                1,
                arch.vocab_size as usize,
                reduction,
            )?;
            enc.end();
        }

        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;
        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        crate::metal::wait_completed(&cmd_buf)?;
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let gpu_kernel_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;

        let argmax = unsafe {
            let src = argmax_tok.buffer.contents().as_ptr() as *const i32;
            *src
        };
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((
            argmax,
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

    pub fn single_token_profiled_concurrent_gdn_attn_dense(
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
                MetalBlock::Gdn(g) => {
                    let i = gdn_idx;
                    gdn_idx += 1;
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin_concurrent(&cmd_buf);
                        self.encode_gdn_front_projections(&enc, g, session)?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin(&cmd_buf);
                        self.encode_gdn_after_projections(&enc, g, i, session)?;
                        self.encode_post_mixer_ffn(&enc, block, session)?;
                        enc.end();
                    }
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
        crate::metal::wait_completed(&cmd_buf)?;
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

    pub(super) fn encode_gdn_front_projections(
        &self,
        enc: &KernelEncoder,
        gb: &MetalGdnBlock,
        s: &mut MetalSession,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let n_v = arch.gdn_n_v_heads as usize;
        let n_k = arch.gdn_n_k_heads as usize;
        let head_dim = arch.gdn_head_dim as usize;
        let conv_dim = (2 * n_k + n_v) * head_dim;
        let v_dim = n_v * head_dim;

        if decode_gdn_noop_qkv_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_qkv, 0.0)?;
        } else {
            encode_mat_vec_dispatch(
                self.ctx,
                enc,
                &gb.in_proj_qkv,
                &s.h,
                &s.gdn_qkv,
                h,
                conv_dim,
            )?;
        }
        if decode_gdn_noop_z_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_z, 0.0)?;
        } else {
            encode_mat_vec_dispatch(self.ctx, enc, &gb.in_proj_z, &s.h, &s.gdn_z, h, v_dim)?;
        }
        if decode_gdn_noop_beta_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_b, 0.0)?;
        } else if gdn_beta_projection_fused(gb) {
            encode_mat_vec_f32_sigmoid(self.ctx, enc, &gb.beta_proj, &s.h, &s.gdn_beta, h, n_v)?;
        } else {
            encode_mat_vec_dispatch(self.ctx, enc, &gb.beta_proj, &s.h, &s.gdn_b, h, n_v)?;
        }
        if decode_gdn_noop_alpha_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_a, 0.0)?;
        } else {
            encode_mat_vec_dispatch(self.ctx, enc, &gb.alpha_proj, &s.h, &s.gdn_a, h, n_v)?;
        }
        Ok(())
    }

    pub(super) fn encode_gdn_after_projections(
        &self,
        enc: &KernelEncoder,
        gb: &MetalGdnBlock,
        gdn_i: usize,
        s: &mut MetalSession,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let v_dim = arch.gdn_n_v_heads as usize * arch.gdn_head_dim as usize;
        let h = arch.hidden_size as usize;
        let gdn_qkv = s.gdn_qkv.clone();
        let gdn_z = s.gdn_z.clone();
        let gdn_alpha = s.gdn_alpha.clone();
        let gdn_beta = s.gdn_beta.clone();
        let gdn_normed = s.gdn_normed.clone();

        if !gdn_beta_projection_fused(gb) {
            encode_sigmoid_f32(self.ctx, enc, &s.gdn_b, &s.gdn_beta)?;
        }
        encode_gdn_decay_chain_f32(
            self.ctx,
            enc,
            &s.gdn_a,
            &gb.dt_bias,
            &gb.a_log,
            &s.gdn_alpha,
        )?;
        self.encode_gdn_tail(
            enc,
            gb,
            gdn_i,
            s,
            &gdn_qkv,
            &gdn_z,
            &gdn_alpha,
            &gdn_beta,
            &gdn_normed,
        )?;
        if decode_gdn_noop_out_enabled() {
            encode_fill_f32(self.ctx, enc, &s.mixer_out, 0.0)?;
        } else {
            encode_mat_vec_dispatch(
                self.ctx,
                enc,
                &gb.out_proj,
                &gdn_normed,
                &s.mixer_out,
                v_dim,
                h,
            )?;
        }
        Ok(())
    }

    /// Complete the stateful GDN body after a batch executor has produced
    /// QKV and Z rows. Small alpha/beta projections remain sequence-private;
    /// the caller may batch the final output projection from `normed_output`.
    #[doc(hidden)]
    pub fn encode_gdn_after_batched_front(
        &self,
        enc: &KernelEncoder,
        gb: &MetalGdnBlock,
        gdn_i: usize,
        session: &mut MetalSession,
        qkv: &MetalTensor,
        z: &MetalTensor,
        normed_output: &MetalTensor,
    ) -> Result<(), MfError> {
        let h = self.model.arch.hidden_size as usize;
        let n_v = self.model.arch.gdn_n_v_heads as usize;
        if decode_gdn_noop_beta_enabled() {
            encode_fill_f32(self.ctx, enc, &session.gdn_b, 0.0)?;
        } else if gdn_beta_projection_fused(gb) {
            encode_mat_vec_f32_sigmoid(
                self.ctx,
                enc,
                &gb.beta_proj,
                &session.h,
                &session.gdn_beta,
                h,
                n_v,
            )?;
        } else {
            encode_mat_vec_dispatch(
                self.ctx,
                enc,
                &gb.beta_proj,
                &session.h,
                &session.gdn_b,
                h,
                n_v,
            )?;
        }
        if decode_gdn_noop_alpha_enabled() {
            encode_fill_f32(self.ctx, enc, &session.gdn_a, 0.0)?;
        } else {
            encode_mat_vec_dispatch(
                self.ctx,
                enc,
                &gb.alpha_proj,
                &session.h,
                &session.gdn_a,
                h,
                n_v,
            )?;
        }
        if !gdn_beta_projection_fused(gb) {
            encode_sigmoid_f32(self.ctx, enc, &session.gdn_b, &session.gdn_beta)?;
        }
        encode_gdn_decay_chain_f32(
            self.ctx,
            enc,
            &session.gdn_a,
            &gb.dt_bias,
            &gb.a_log,
            &session.gdn_alpha,
        )?;
        let alpha = session.gdn_alpha.clone();
        let beta = session.gdn_beta.clone();
        self.encode_gdn_tail(
            enc,
            gb,
            gdn_i,
            session,
            qkv,
            z,
            &alpha,
            &beta,
            normed_output,
        )?;
        Ok(())
    }

    pub fn encode_gdn(
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

        if decode_gdn_noop_qkv_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_qkv, 0.0)?;
        } else {
            // QKV input projection.
            encode_mat_vec_dispatch(
                self.ctx,
                enc,
                &gb.in_proj_qkv,
                &s.h,
                &s.gdn_qkv,
                h,
                conv_dim,
            )?;
        }
        if decode_gdn_noop_z_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_z, 0.0)?;
        } else {
            // z projection.
            encode_mat_vec_dispatch(self.ctx, enc, &gb.in_proj_z, &s.h, &s.gdn_z, h, v_dim)?;
        }
        let fuse_beta_proj = gdn_beta_projection_fused(gb);
        if decode_gdn_noop_beta_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_b, 0.0)?;
        } else if fuse_beta_proj {
            encode_mat_vec_f32_sigmoid(self.ctx, enc, &gb.beta_proj, &s.h, &s.gdn_beta, h, n_v)?;
        } else {
            // beta source projection.
            encode_mat_vec_dispatch(self.ctx, enc, &gb.beta_proj, &s.h, &s.gdn_b, h, n_v)?;
        }
        if decode_gdn_noop_alpha_enabled() {
            encode_fill_f32(self.ctx, enc, &s.gdn_a, 0.0)?;
        } else {
            // α source projection.
            encode_mat_vec_dispatch(self.ctx, enc, &gb.alpha_proj, &s.h, &s.gdn_a, h, n_v)?;
        }
        if !fuse_beta_proj {
            encode_sigmoid_f32(self.ctx, enc, &s.gdn_b, &s.gdn_beta)?;
        }
        // Decay-chain fusion: gdn_alpha stores exp(softplus(gdn_a + dt_bias) * a_log).
        // Replaces add_inplace + softplus + mul + per-row exp with one
        // per-head fused kernel.
        encode_gdn_decay_chain_f32(
            self.ctx,
            enc,
            &s.gdn_a,
            &gb.dt_bias,
            &gb.a_log,
            &s.gdn_alpha,
        )?;
        // Now `gdn_alpha` is the per-head decay exp(g), reused by every state row.
        if let Some(alpha_destination) = gdn_alpha_census_capture_slot(gdn_i) {
            encode_scatter_offset_f32(self.ctx, enc, &s.gdn_alpha, &alpha_destination, 0, n_v)?;
        }

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

        // Split conv output into Q, K, V via zero-copy views (no dispatch).
        // Per Jeff & Sanjay (avoid copies / use indices instead of pointers):
        // the previous code did 3 copy_offset dispatches per layer × 32
        // GDN layers = 96 dispatches/token just to alias subranges.
        // view_subrange returns a sub-tensor pointing at the same MTLBuffer
        // with shifted offset, consumed by the next kernel directly.
        let q_view = s
            .gdn_qkv_conv
            .view_subrange(0, vec![(n_k * head_dim) as u64]);
        let k_view = s
            .gdn_qkv_conv
            .view_subrange((n_k * head_dim) as u64, vec![(n_k * head_dim) as u64]);
        let v_view = s
            .gdn_qkv_conv
            .view_subrange((2 * n_k * head_dim) as u64, vec![v_dim as u64]);

        if decode_gdn_pair_l2_enabled() {
            encode_l2_norm_pair_batched_f32(
                self.ctx,
                enc,
                &q_view,
                &s.gdn_q_norm,
                &k_view,
                &s.gdn_k_norm,
                n_k,
                head_dim,
                RMS_EPS,
            )?;
        } else {
            encode_l2_norm_batched_f32(
                self.ctx,
                enc,
                &q_view,
                &s.gdn_q_norm,
                n_k,
                head_dim,
                RMS_EPS,
            )?;
            encode_l2_norm_batched_f32(
                self.ctx,
                enc,
                &k_view,
                &s.gdn_k_norm,
                n_k,
                head_dim,
                RMS_EPS,
            )?;
        }

        // Recurrence step (kernel does the head-repeat internally).
        encode_gdn_step_decay_f32(
            self.ctx,
            enc,
            &s.gdn_q_norm,
            &s.gdn_k_norm,
            &v_view,
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
            RMS_EPS * head_dim as f32,
        )?;

        // Output projection: [v_dim, hidden] → mixer_out.
        if decode_gdn_noop_out_enabled() {
            encode_fill_f32(self.ctx, enc, &s.mixer_out, 0.0)?;
        } else {
            encode_mat_vec_dispatch(
                self.ctx,
                enc,
                &gb.out_proj,
                &s.gdn_normed,
                &s.mixer_out,
                v_dim,
                h,
            )?;
        }
        Ok(())
    }

    /// GDN per-token recurrence body, factored out for v0.73a.1 layer-major
    /// batching. Takes pre-computed inputs as zero-copy F32 views (one
    /// row of N-shaped pack buffers) and writes the per-head normed
    /// output to `gdn_normed_out` (also a row view).
    ///
    /// Performs in order: ssm_conv1d+silu (mutates `s.gdn_conv[gdn_i]`)
    /// → l2_norm Q/K → gdn_step (mutates `s.gdn_state[gdn_i]`) →
    /// rmsnorm_gated. Bit-exact with the corresponding inner part of
    /// `encode_gdn` when given the same inputs (validated by
    /// `gdn_tail_matches_inline`).
    ///
    /// The `_qkv_in` / `z_in` arguments alias rows of the layer-major
    /// pack buffers (`gdn_qkv_pack`, `gdn_z_pack`); `alpha_in` /
    /// `beta_in` come from either the per-token session scratch
    /// (`s.gdn_alpha`, `s.gdn_beta`) or packed row views populated by the
    /// Q8 verifier alpha/beta sidecar. The recurrent tail remains sequential
    /// in both cases.
    ///
    /// Caller's responsibility: per-token sequencing of `s.gdn_conv[gdn_i]`
    /// and `s.gdn_state[gdn_i]` (the recurrence is inherently
    /// per-token-sequential), and checkpoint blits between calls.
    pub fn encode_gdn_tail(
        &self,
        enc: &KernelEncoder,
        gb: &MetalGdnBlock,
        gdn_i: usize,
        s: &mut MetalSession,
        qkv_in: &MetalTensor,         // [conv_dim] F32 — one row of gdn_qkv_pack
        z_in: &MetalTensor,           // [v_dim] F32   — one row of gdn_z_pack
        alpha_in: &MetalTensor,       // [n_v] F32     — pre-computed decay exp(g)
        beta_in: &MetalTensor,        // [n_v] F32     — pre-computed sigmoid(beta)
        gdn_normed_out: &MetalTensor, // [v_dim] F32 — one row of gdn_normed_pack
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let n_v = arch.gdn_n_v_heads as usize;
        let n_k = arch.gdn_n_k_heads as usize;
        let head_dim = arch.gdn_head_dim as usize;
        let conv_dim = (2 * n_k + n_v) * head_dim;

        encode_ssm_conv_silu_f32(
            self.ctx,
            enc,
            qkv_in,
            &s.gdn_conv[gdn_i],
            &gb.conv1d,
            &s.gdn_qkv_conv,
            conv_dim,
        )?;
        let q_view = s
            .gdn_qkv_conv
            .view_subrange(0, vec![(n_k * head_dim) as u64]);
        let k_view = s
            .gdn_qkv_conv
            .view_subrange((n_k * head_dim) as u64, vec![(n_k * head_dim) as u64]);
        let v_view = s
            .gdn_qkv_conv
            .view_subrange((2 * n_k * head_dim) as u64, vec![(n_v * head_dim) as u64]);
        if decode_gdn_pair_l2_enabled() {
            encode_l2_norm_pair_batched_f32(
                self.ctx,
                enc,
                &q_view,
                &s.gdn_q_norm,
                &k_view,
                &s.gdn_k_norm,
                n_k,
                head_dim,
                RMS_EPS,
            )?;
        } else {
            encode_l2_norm_batched_f32(
                self.ctx,
                enc,
                &q_view,
                &s.gdn_q_norm,
                n_k,
                head_dim,
                RMS_EPS,
            )?;
            encode_l2_norm_batched_f32(
                self.ctx,
                enc,
                &k_view,
                &s.gdn_k_norm,
                n_k,
                head_dim,
                RMS_EPS,
            )?;
        }
        encode_gdn_step_decay_f32(
            self.ctx,
            enc,
            &s.gdn_q_norm,
            &s.gdn_k_norm,
            &v_view,
            alpha_in,
            beta_in,
            &s.gdn_state[gdn_i],
            &s.gdn_out,
            n_v,
            n_k,
            head_dim,
        )?;
        encode_rmsnorm_gated_f32(
            self.ctx,
            enc,
            &s.gdn_out,
            &gb.norm,
            z_in,
            gdn_normed_out,
            n_v,
            head_dim,
            RMS_EPS * head_dim as f32,
        )?;
        Ok(())
    }
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
        encode_mat_vec_dispatch(self.ctx, &enc, &gb.ffn_gate, &s.h, &s.ffn_gate, h, f)?;
        encode_mat_vec_dispatch(self.ctx, &enc, &gb.ffn_up, &s.h, &s.ffn_up, h, f)?;
        encode_silu_mul_f32(self.ctx, &enc, &s.ffn_gate, &s.ffn_up, &s.ffn_inner)?;
        encode_mat_vec_dispatch(self.ctx, &enc, &gb.ffn_down, &s.ffn_inner, &s.ffn_out, f, h)?;

        // Residual #2: s.x += s.ffn_out.
        encode_add_inplace_f32(self.ctx, &enc, &s.x, &s.ffn_out)?;

        enc.end();
        cmd_buf.commit();
        crate::metal::wait_completed(&cmd_buf)?;
        let mut out = vec![0.0f32; h];
        unsafe {
            let src = s.x.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), h);
        }
        Ok(out)
    }
}
