//! MoE layer execution: routing, expert projections, and fused paths.

use super::*;

pub(super) fn moe_iq3_expert_native_enabled(desc: &TensorDesc) -> bool {
    let truthy = |v: &str| matches!(v, "1" | "true" | "TRUE" | "yes" | "YES");
    let falsey = |v: &str| matches!(v, "0" | "false" | "FALSE" | "no" | "NO");
    if let Ok(v) = std::env::var("QWEN_MOE_IQ3_EXPERT_NATIVE") {
        if truthy(&v) {
            return true;
        }
        if falsey(&v) {
            return false;
        }
    }
    if let Ok(v) = std::env::var("QWEN_PREFILL_MOE_GROUPED_IQ3_GATEUP") {
        if truthy(&v) {
            return true;
        }
        if falsey(&v) {
            return false;
        }
    }
    desc.shape.len() >= 3 && desc.shape[0..3] == [2048, 512, 256]
}

/// Whether decode's routed gate/up for these expert dtypes is one fused
/// dispatch, so it can share the concurrent gate/up wave with the shared
/// expert (measured +8.8% decode on A3B Q4_K). Multi-pass dtypes (separate
/// gate and up products then SiLU*mul) stay on the serial encoder, which
/// orders them.
pub(super) fn routed_gate_up_single_dispatch(gate: GgmlType, up: GgmlType) -> bool {
    gate == up
        && match gate {
            GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::Q8_0 | GgmlType::IQ4_XS => true,
            GgmlType::IQ3_XXS | GgmlType::IQ3_S => {
                decode_moe_iq3_fast_swiglu_enabled() || decode_moe_iq3_fused_swiglu_enabled()
            }
            _ => false,
        }
}

pub(super) fn phase_moe_ffn_split_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN_PHASE_MOE_FFN_SPLIT").as_deref(),
            Ok("1")
                | Ok("2")
                | Ok("deep")
                | Ok("DEEP")
                | Ok("true")
                | Ok("TRUE")
                | Ok("yes")
                | Ok("YES")
        )
    })
}

pub(super) fn phase_moe_ffn_deep_split_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN_PHASE_MOE_FFN_SPLIT").as_deref(),
            Ok("2") | Ok("deep") | Ok("DEEP")
        )
    })
}

pub(super) fn phase_moe_route_split_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN_PHASE_MOE_ROUTE_SPLIT").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    })
}

pub(super) fn phase_moe_route_deep_split_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN_PHASE_MOE_ROUTE_SPLIT").as_deref(),
            Ok("2") | Ok("deep") | Ok("DEEP")
        )
    })
}

pub(super) fn moe_routed_gate_up_decode_supported(gate: GgmlType, up: GgmlType) -> bool {
    matches!(
        (gate, up),
        (GgmlType::Q4_K, GgmlType::Q4_K)
            | (GgmlType::Q5_K, GgmlType::Q5_K)
            | (GgmlType::Q6_K, GgmlType::Q6_K)
            | (GgmlType::Q8_0, GgmlType::Q8_0)
            | (GgmlType::IQ3_XXS, GgmlType::IQ3_XXS)
            | (GgmlType::IQ3_S, GgmlType::IQ3_S)
            | (GgmlType::IQ4_XS, GgmlType::IQ4_XS)
            | (GgmlType::BF16, GgmlType::BF16)
            | (GgmlType::F32, GgmlType::F32)
    )
}

pub(super) fn push_moe_weight_requests<'a>(
    requests: &mut Vec<ModelWeightStorageRequest<'a>>,
    moe: &MoeFfn<'a>,
    router_f16: bool,
) -> Result<(), MfError> {
    if router_f16 {
        push_model_weight_request(requests, moe.gate_inp, ModelWeightStorageKind::ConvertedF16)?;
    } else {
        push_f32_weight_request(requests, moe.gate_inp)?;
    }
    push_native_weight_request(requests, moe.gate_exps)?;
    push_native_weight_request(requests, moe.up_exps)?;
    push_native_weight_request(requests, moe.down_exps)?;
    push_f32_weight_request(requests, moe.gate_inp_shexp)
}

impl<'a> MetalForward<'a> {
    #[allow(dead_code)]
    pub(super) fn route_moe_block(
        &self,
        h_tensor: &MetalTensor,
        moe: &MetalMoeFfn,
    ) -> MoeRouteDecision {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let n_expert = arch.expert_count as usize;
        let n_expert_used = arch.expert_used_count.min(arch.expert_count) as usize;
        let h_cpu = unsafe {
            std::slice::from_raw_parts(
                (h_tensor.buffer.contents().as_ptr() as *const f32)
                    .add((h_tensor.offset / 4) as usize),
                h,
            )
        };
        let mut probs = crate::forward::mat_vec_pub(&moe.gate_inp_cpu, h, n_expert, h_cpu);
        let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in &mut probs {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum.max(1e-20);
        for v in &mut probs {
            *v *= inv;
        }
        let mut ranked: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(n_expert_used);

        let shared_gate_scalar = {
            let s: f32 = h_cpu[..h]
                .iter()
                .zip(&moe.gate_inp_shexp_cpu[..h])
                .map(|(a, b)| a * b)
                .sum();
            1.0 / (1.0 + (-s).exp())
        };
        MoeRouteDecision {
            ranked,
            shared_gate_scalar,
        }
    }

    #[allow(dead_code)]
    pub(super) fn read_moe_route_result(
        &self,
        session: &MetalSession,
        topk: usize,
    ) -> MoeRouteDecision {
        let mut ranked = Vec::with_capacity(topk);
        unsafe {
            let idx_ptr = (session.moe_topk_idx.buffer.contents().as_ptr() as *const i32)
                .add((session.moe_topk_idx.offset / 4) as usize);
            let w_ptr = (session.moe_topk_weight.buffer.contents().as_ptr() as *const f32)
                .add((session.moe_topk_weight.offset / 4) as usize);
            for i in 0..topk {
                ranked.push((*idx_ptr.add(i) as usize, *w_ptr.add(i)));
            }
            let shared_gate_scalar = *((session.moe_shared_gate.buffer.contents().as_ptr()
                as *const f32)
                .add((session.moe_shared_gate.offset / 4) as usize));
            MoeRouteDecision {
                ranked,
                shared_gate_scalar,
            }
        }
    }

    pub(super) fn write_moe_route_result(
        &self,
        session: &mut MetalSession,
        route: &MoeRouteDecision,
    ) {
        unsafe {
            let idx_ptr = (session.moe_topk_idx.buffer.contents().as_ptr() as *mut i32)
                .add((session.moe_topk_idx.offset / 4) as usize);
            let w_ptr = (session.moe_topk_weight.buffer.contents().as_ptr() as *mut f32)
                .add((session.moe_topk_weight.offset / 4) as usize);
            for (i, &(expert, weight)) in route.ranked.iter().enumerate() {
                *idx_ptr.add(i) = expert as i32;
                *w_ptr.add(i) = weight;
            }
            let shared_ptr = (session.moe_shared_gate.buffer.contents().as_ptr() as *mut f32)
                .add((session.moe_shared_gate.offset / 4) as usize);
            *shared_ptr = route.shared_gate_scalar;
        }
    }

    pub(crate) fn encode_moe_route_prepare(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        if decode_moe_noop_route_enabled() {
            return Ok(());
        }
        self.encode_moe_router_logits(enc, session, moe)?;
        self.encode_moe_topk_and_shared_from_logits(enc, session, moe)?;
        Ok(())
    }

    pub(super) fn encode_moe_router_logits(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let n_expert = arch.expert_count as usize;

        let router_probs = session
            .moe_router_probs
            .view_subrange(0, vec![n_expert as u64]);

        encode_mat_vec_dispatch(
            self.ctx,
            enc,
            &moe.gate_inp,
            &session.h,
            &router_probs,
            h,
            n_expert,
        )?;
        Ok(())
    }

    pub(super) fn encode_moe_topk_from_logits(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let n_expert = arch.expert_count as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;

        let router_probs = session
            .moe_router_probs
            .view_subrange(0, vec![n_expert as u64]);
        let topk_idx = session.moe_topk_idx.view_subrange(0, vec![topk as u64]);
        let topk_w = session.moe_topk_weight.view_subrange(0, vec![topk as u64]);

        encode_topk_logits_softmax_f32(
            self.ctx,
            enc,
            &router_probs,
            &topk_idx,
            &topk_w,
            n_expert,
            topk,
        )?;
        Ok(())
    }

    pub(super) fn encode_moe_topk_parallel_from_logits(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let n_expert = arch.expert_count as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;

        let router_probs = session
            .moe_router_probs
            .view_subrange(0, vec![n_expert as u64]);
        let topk_idx = session.moe_topk_idx.view_subrange(0, vec![topk as u64]);
        let topk_w = session.moe_topk_weight.view_subrange(0, vec![topk as u64]);

        encode_topk_logits_softmax_parallel_f32(
            self.ctx,
            enc,
            &router_probs,
            &topk_idx,
            &topk_w,
            n_expert,
            topk,
        )?;
        Ok(())
    }

    pub(super) fn encode_moe_topk_and_shared_from_logits(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let n_expert = arch.expert_count as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;

        if n_expert > 256 || topk > 16 {
            self.encode_moe_topk_from_logits(enc, session)?;
            self.encode_moe_shared_gate(enc, session, moe)?;
            return Ok(());
        }

        let router_probs = session
            .moe_router_probs
            .view_subrange(0, vec![n_expert as u64]);
        let topk_idx = session.moe_topk_idx.view_subrange(0, vec![topk as u64]);
        let topk_w = session.moe_topk_weight.view_subrange(0, vec![topk as u64]);

        encode_topk_logits_softmax_dot_sigmoid_f32(
            self.ctx,
            enc,
            &router_probs,
            &moe.gate_inp_shexp,
            &session.h,
            &topk_idx,
            &topk_w,
            &session.moe_shared_gate,
            n_expert,
            topk,
            h,
        )?;
        Ok(())
    }

    pub(super) fn encode_moe_shared_gate(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        let h = self.model.arch.hidden_size as usize;

        encode_dot_sigmoid_f32(
            self.ctx,
            enc,
            &moe.gate_inp_shexp,
            &session.h,
            &session.moe_shared_gate,
            h,
        )?;
        Ok(())
    }

    pub(crate) fn encode_moe_routed_ffn_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let n_expert = arch.expert_count as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;

        let gate_up_supported =
            moe_routed_gate_up_decode_supported(moe.gate_exps.dtype, moe.up_exps.dtype);
        if !gate_up_supported {
            return Err(MfError::UnsupportedDtype {
                name: "MoE routed gate/up expert banks".into(),
                dtype: moe.gate_exps.dtype,
            });
        }
        if !matches!(
            moe.down_exps.dtype,
            GgmlType::Q4_K
                | GgmlType::Q5_K
                | GgmlType::Q6_K
                | GgmlType::Q8_0
                | GgmlType::IQ4_NL
                | GgmlType::IQ4_XS
                | GgmlType::BF16
                | GgmlType::F32
        ) {
            return Err(MfError::UnsupportedDtype {
                name: "MoE routed down expert bank".into(),
                dtype: moe.down_exps.dtype,
            });
        }

        let moe_inner = session
            .moe_inner
            .view_subrange(0, vec![(topk * f_exp) as u64]);
        let moe_expert_out = session
            .moe_expert_out
            .view_subrange(0, vec![(topk * h) as u64]);
        let topk_idx = session.moe_topk_idx.view_subrange(0, vec![topk as u64]);
        let topk_w = session.moe_topk_weight.view_subrange(0, vec![topk as u64]);

        self.encode_moe_routed_gate_up_gpu(enc, session, moe)?;
        if decode_moe_noop_routed_down_enabled() {
            encode_fill_f32(self.ctx, enc, &session.mixer_out, 0.0)?;
            return Ok(());
        }
        match moe.down_exps.dtype {
            GgmlType::Q4_K => {
                encode_moe_down_q4_K_f32(
                    self.ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &moe_expert_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                encode_moe_weighted_sum_f32(
                    self.ctx,
                    enc,
                    &moe_expert_out,
                    &topk_w,
                    &session.mixer_out,
                    h,
                    topk,
                )?;
            }
            GgmlType::Q5_K => {
                if decode_moe_q5_down_fused_enabled() {
                    encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                        self.ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &topk_w,
                        &session.mixer_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                        1,
                    )?;
                } else {
                    encode_moe_down_q5_K_f32(
                        self.ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &moe_expert_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )?;
                    encode_moe_weighted_sum_f32(
                        self.ctx,
                        enc,
                        &moe_expert_out,
                        &topk_w,
                        &session.mixer_out,
                        h,
                        topk,
                    )?;
                }
            }
            GgmlType::Q6_K => encode_moe_down_weighted_sum_q6_K_f32(
                self.ctx,
                enc,
                &moe.down_exps,
                &moe_inner,
                &topk_idx,
                &topk_w,
                &session.mixer_out,
                f_exp,
                h,
                n_expert,
                topk,
            )?,
            GgmlType::Q8_0 => encode_moe_down_weighted_sum_q8_0_f32(
                self.ctx,
                enc,
                &moe.down_exps,
                &moe_inner,
                &topk_idx,
                &topk_w,
                &session.mixer_out,
                f_exp,
                h,
                n_expert,
                topk,
            )?,
            GgmlType::IQ4_XS => {
                if decode_moe_iq4_down_fast_enabled() {
                    encode_moe_down_iq4_xs_f32_fast(
                        self.ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &moe_expert_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )?;
                } else {
                    encode_moe_down_iq4_xs_f32(
                        self.ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &moe_expert_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )?;
                }
                encode_moe_weighted_sum_f32(
                    self.ctx,
                    enc,
                    &moe_expert_out,
                    &topk_w,
                    &session.mixer_out,
                    h,
                    topk,
                )?;
            }
            GgmlType::IQ4_NL => {
                encode_moe_down_iq4_nl_f32(
                    self.ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &moe_expert_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                encode_moe_weighted_sum_f32(
                    self.ctx,
                    enc,
                    &moe_expert_out,
                    &topk_w,
                    &session.mixer_out,
                    h,
                    topk,
                )?;
            }
            GgmlType::F32 => {
                encode_moe_down_f32_f32(
                    self.ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &moe_expert_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                encode_moe_weighted_sum_f32(
                    self.ctx,
                    enc,
                    &moe_expert_out,
                    &topk_w,
                    &session.mixer_out,
                    h,
                    topk,
                )?;
            }
            GgmlType::BF16 => {
                encode_moe_down_bf16_f32(
                    self.ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &moe_expert_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                encode_moe_weighted_sum_f32(
                    self.ctx,
                    enc,
                    &moe_expert_out,
                    &topk_w,
                    &session.mixer_out,
                    h,
                    topk,
                )?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    pub(crate) fn encode_moe_shared_ffn_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        ffn_gate: &MetalTensor,
        ffn_up: &MetalTensor,
        ffn_down: &MetalTensor,
    ) -> Result<(), MfError> {
        self.encode_moe_shared_ffn_core_gpu(enc, session, ffn_gate, ffn_up, ffn_down)?;
        self.encode_moe_shared_ffn_accumulate_gpu(enc, session)
    }

    pub(super) fn encode_moe_shared_ffn_core_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        ffn_gate: &MetalTensor,
        ffn_up: &MetalTensor,
        ffn_down: &MetalTensor,
    ) -> Result<(), MfError> {
        let fused_inner = self.encode_moe_shared_ffn_gate_up_gpu(enc, session, ffn_gate, ffn_up)?;
        if !fused_inner {
            self.encode_moe_shared_ffn_silu_gpu(enc, session)?;
        }
        self.encode_moe_shared_ffn_down_gpu(enc, session, ffn_down)
    }

    pub(super) fn encode_moe_shared_ffn_gate_up_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        ffn_gate: &MetalTensor,
        ffn_up: &MetalTensor,
    ) -> Result<bool, MfError> {
        let h = self.model.arch.hidden_size as usize;
        let f_shared = self.model.arch.expert_shared_feed_forward_length as usize;
        if decode_shared_swiglu_q8_enabled()
            && ffn_gate.dtype == GgmlType::Q8_0
            && ffn_up.dtype == GgmlType::Q8_0
        {
            let shared_inner_tmp = session.ffn_inner.view_subrange(0, vec![f_shared as u64]);
            encode_shared_swiglu_q8_0_f32(
                self.ctx,
                enc,
                ffn_gate,
                ffn_up,
                &session.h,
                &shared_inner_tmp,
                h,
                f_shared,
            )?;
            return Ok(true);
        }
        let shared_gate_tmp = session.ffn_gate.view_subrange(0, vec![f_shared as u64]);
        let shared_up_tmp = session.ffn_up.view_subrange(0, vec![f_shared as u64]);
        encode_mat_vec_dispatch(
            self.ctx,
            enc,
            ffn_gate,
            &session.h,
            &shared_gate_tmp,
            h,
            f_shared,
        )?;
        encode_mat_vec_dispatch(
            self.ctx,
            enc,
            ffn_up,
            &session.h,
            &shared_up_tmp,
            h,
            f_shared,
        )?;
        Ok(false)
    }

    pub(super) fn encode_moe_shared_ffn_silu_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let f_shared = self.model.arch.expert_shared_feed_forward_length as usize;
        let shared_gate_tmp = session.ffn_gate.view_subrange(0, vec![f_shared as u64]);
        let shared_up_tmp = session.ffn_up.view_subrange(0, vec![f_shared as u64]);
        let shared_inner_tmp = session.ffn_inner.view_subrange(0, vec![f_shared as u64]);
        encode_silu_mul_f32(
            self.ctx,
            enc,
            &shared_gate_tmp,
            &shared_up_tmp,
            &shared_inner_tmp,
        )?;
        Ok(())
    }

    pub(super) fn encode_moe_shared_ffn_down_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        ffn_down: &MetalTensor,
    ) -> Result<(), MfError> {
        let h = self.model.arch.hidden_size as usize;
        let f_shared = self.model.arch.expert_shared_feed_forward_length as usize;
        let shared_inner_tmp = session.ffn_inner.view_subrange(0, vec![f_shared as u64]);
        let shared_out_tmp = session.ffn_out.view_subrange(0, vec![h as u64]);
        encode_mat_vec_dispatch(
            self.ctx,
            enc,
            ffn_down,
            &shared_inner_tmp,
            &shared_out_tmp,
            f_shared,
            h,
        )?;
        Ok(())
    }

    pub(super) fn encode_moe_shared_ffn_accumulate_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let h = self.model.arch.hidden_size as usize;
        let shared_out_tmp = session.ffn_out.view_subrange(0, vec![h as u64]);
        encode_axpy_scalar_f32(
            self.ctx,
            enc,
            &shared_out_tmp,
            &session.moe_shared_gate,
            &session.mixer_out,
        )?;
        Ok(())
    }

    pub(super) fn encode_moe_final_residual_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let h = self.model.arch.hidden_size as usize;
        let shared_out_tmp = session.ffn_out.view_subrange(0, vec![h as u64]);
        if decode_moe_fused_finalizer_enabled() {
            encode_moe_shared_accum_resid_f32(
                self.ctx,
                enc,
                &shared_out_tmp,
                &session.moe_shared_gate,
                &session.mixer_out,
                &session.x,
            )?;
        } else {
            self.encode_moe_shared_ffn_accumulate_gpu(enc, session)?;
            encode_add_inplace_f32(self.ctx, enc, &session.x, &session.mixer_out)?;
        }
        Ok(())
    }

    pub(crate) fn encode_moe_ffn_apply_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        ffn_gate: &MetalTensor,
        ffn_up: &MetalTensor,
        ffn_down: &MetalTensor,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        self.encode_moe_routed_ffn_gpu(enc, session, moe)?;
        self.encode_moe_shared_ffn_core_gpu(enc, session, ffn_gate, ffn_up, ffn_down)?;
        self.encode_moe_final_residual_gpu(enc, session)?;
        Ok(())
    }

    pub(super) fn encode_moe_routed_gate_up_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let n_expert = arch.expert_count as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let moe_inner = session
            .moe_inner
            .view_subrange(0, vec![(topk * f_exp) as u64]);
        let topk_idx = session.moe_topk_idx.view_subrange(0, vec![topk as u64]);

        if decode_moe_noop_routed_gateup_enabled() {
            encode_fill_f32(self.ctx, enc, &moe_inner, 0.0)?;
            return Ok(());
        }

        match moe.gate_exps.dtype {
            GgmlType::Q4_K => encode_moe_swiglu_q4_K_f32(
                self.ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &session.h,
                &topk_idx,
                &moe_inner,
                h,
                f_exp,
                n_expert,
                topk,
            )?,
            GgmlType::Q5_K => {
                let gate_pack = session
                    .moe_expert_out
                    .view_subrange(0, vec![(topk * f_exp) as u64]);
                let up_pack = session
                    .moe_expert_out
                    .view_subrange((topk * f_exp) as u64, vec![(topk * f_exp) as u64]);
                encode_moe_mat_vec_q5_K_f32(
                    self.ctx,
                    enc,
                    &moe.gate_exps,
                    &session.h,
                    &topk_idx,
                    &gate_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_moe_mat_vec_q5_K_f32(
                    self.ctx,
                    enc,
                    &moe.up_exps,
                    &session.h,
                    &topk_idx,
                    &up_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_silu_mul_f32(self.ctx, enc, &gate_pack, &up_pack, &moe_inner)?;
            }
            GgmlType::Q6_K => encode_moe_swiglu_q6_K_f32(
                self.ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &session.h,
                &topk_idx,
                &moe_inner,
                h,
                f_exp,
                n_expert,
                topk,
            )?,
            GgmlType::Q8_0 => encode_moe_swiglu_q8_0_f32(
                self.ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &session.h,
                &topk_idx,
                &moe_inner,
                h,
                f_exp,
                n_expert,
                topk,
            )?,
            GgmlType::IQ3_XXS => {
                if decode_moe_iq3_fast_swiglu_enabled() {
                    encode_moe_swiglu_iq3_xxs_f32_fast(
                        self.ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &session.h,
                        &topk_idx,
                        &moe_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                    )?;
                } else if decode_moe_iq3_fused_swiglu_enabled() {
                    encode_moe_swiglu_iq3_xxs_f32(
                        self.ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &session.h,
                        &topk_idx,
                        &moe_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                    )?;
                } else {
                    let gate_pack = session
                        .moe_expert_out
                        .view_subrange(0, vec![(topk * f_exp) as u64]);
                    let up_pack = session
                        .moe_expert_out
                        .view_subrange((topk * f_exp) as u64, vec![(topk * f_exp) as u64]);
                    encode_moe_mat_vec_iq3_xxs_f32(
                        self.ctx,
                        enc,
                        &moe.gate_exps,
                        &session.h,
                        &topk_idx,
                        &gate_pack,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                    )?;
                    encode_moe_mat_vec_iq3_xxs_f32(
                        self.ctx,
                        enc,
                        &moe.up_exps,
                        &session.h,
                        &topk_idx,
                        &up_pack,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                    )?;
                    encode_silu_mul_f32(self.ctx, enc, &gate_pack, &up_pack, &moe_inner)?;
                }
            }
            GgmlType::IQ3_S => {
                if decode_moe_iq3_fast_swiglu_enabled() {
                    encode_moe_swiglu_iq3_s_f32_fast(
                        self.ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &session.h,
                        &topk_idx,
                        &moe_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                    )?;
                } else if decode_moe_iq3_fused_swiglu_enabled() {
                    encode_moe_swiglu_iq3_s_f32(
                        self.ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &session.h,
                        &topk_idx,
                        &moe_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                    )?;
                } else {
                    let gate_pack = session
                        .moe_expert_out
                        .view_subrange(0, vec![(topk * f_exp) as u64]);
                    let up_pack = session
                        .moe_expert_out
                        .view_subrange((topk * f_exp) as u64, vec![(topk * f_exp) as u64]);
                    encode_moe_mat_vec_iq3_s_f32(
                        self.ctx,
                        enc,
                        &moe.gate_exps,
                        &session.h,
                        &topk_idx,
                        &gate_pack,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                    )?;
                    encode_moe_mat_vec_iq3_s_f32(
                        self.ctx,
                        enc,
                        &moe.up_exps,
                        &session.h,
                        &topk_idx,
                        &up_pack,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                    )?;
                    encode_silu_mul_f32(self.ctx, enc, &gate_pack, &up_pack, &moe_inner)?;
                }
            }
            GgmlType::IQ4_XS => encode_moe_swiglu_iq4_xs_f32(
                self.ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &session.h,
                &topk_idx,
                &moe_inner,
                h,
                f_exp,
                n_expert,
                topk,
            )?,
            GgmlType::F32 => {
                let gate_pack = session
                    .moe_expert_out
                    .view_subrange(0, vec![(topk * f_exp) as u64]);
                let up_pack = session
                    .moe_expert_out
                    .view_subrange((topk * f_exp) as u64, vec![(topk * f_exp) as u64]);
                encode_moe_mat_vec_f32(
                    self.ctx,
                    enc,
                    &moe.gate_exps,
                    &session.h,
                    &topk_idx,
                    &gate_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_moe_mat_vec_f32(
                    self.ctx,
                    enc,
                    &moe.up_exps,
                    &session.h,
                    &topk_idx,
                    &up_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_silu_mul_f32(self.ctx, enc, &gate_pack, &up_pack, &moe_inner)?;
            }
            GgmlType::BF16 => {
                let gate_pack = session
                    .moe_expert_out
                    .view_subrange(0, vec![(topk * f_exp) as u64]);
                let up_pack = session
                    .moe_expert_out
                    .view_subrange((topk * f_exp) as u64, vec![(topk * f_exp) as u64]);
                encode_moe_mat_vec_bf16_f32(
                    self.ctx,
                    enc,
                    &moe.gate_exps,
                    &session.h,
                    &topk_idx,
                    &gate_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_moe_mat_vec_bf16_f32(
                    self.ctx,
                    enc,
                    &moe.up_exps,
                    &session.h,
                    &topk_idx,
                    &up_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_silu_mul_f32(self.ctx, enc, &gate_pack, &up_pack, &moe_inner)?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    pub(super) fn encode_moe_routed_down_only_gpu(
        &self,
        enc: &KernelEncoder,
        session: &mut MetalSession,
        moe: &MetalMoeFfn,
    ) -> Result<bool, MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let n_expert = arch.expert_count as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let moe_inner = session
            .moe_inner
            .view_subrange(0, vec![(topk * f_exp) as u64]);
        let moe_expert_out = session
            .moe_expert_out
            .view_subrange(0, vec![(topk * h) as u64]);
        let topk_idx = session.moe_topk_idx.view_subrange(0, vec![topk as u64]);
        let topk_w = session.moe_topk_weight.view_subrange(0, vec![topk as u64]);

        if decode_moe_noop_routed_down_enabled() {
            encode_fill_f32(self.ctx, enc, &session.mixer_out, 0.0)?;
            return Ok(false);
        }

        let pending = match moe.down_exps.dtype {
            GgmlType::Q4_K => {
                encode_moe_down_q4_K_f32(
                    self.ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &moe_expert_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                true
            }
            GgmlType::Q5_K => {
                if decode_moe_q5_down_fused_enabled() {
                    if f_exp == 512 && decode_moe_q5_down_k512_r2_enabled() {
                        encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
                            self.ctx,
                            enc,
                            &moe.down_exps,
                            &moe_inner,
                            &topk_idx,
                            &topk_w,
                            &session.mixer_out,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                            1,
                        )?;
                    } else {
                        encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                            self.ctx,
                            enc,
                            &moe.down_exps,
                            &moe_inner,
                            &topk_idx,
                            &topk_w,
                            &session.mixer_out,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                            1,
                        )?;
                    }
                    false
                } else {
                    encode_moe_down_q5_K_f32(
                        self.ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &moe_expert_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )?;
                    true
                }
            }
            GgmlType::Q6_K => {
                encode_moe_down_weighted_sum_q6_K_f32(
                    self.ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &topk_w,
                    &session.mixer_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                false
            }
            GgmlType::Q8_0 => {
                encode_moe_down_weighted_sum_q8_0_f32(
                    self.ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &topk_w,
                    &session.mixer_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                false
            }
            GgmlType::IQ4_XS => {
                if decode_moe_iq4_down_fast_enabled() {
                    encode_moe_down_iq4_xs_f32_fast(
                        self.ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &moe_expert_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )?;
                } else {
                    encode_moe_down_iq4_xs_f32(
                        self.ctx,
                        enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &moe_expert_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )?;
                }
                true
            }
            GgmlType::IQ4_NL => {
                encode_moe_down_iq4_nl_f32(
                    self.ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &moe_expert_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                true
            }
            GgmlType::F32 => {
                encode_moe_down_f32_f32(
                    self.ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &moe_expert_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                true
            }
            GgmlType::BF16 => {
                encode_moe_down_bf16_f32(
                    self.ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &moe_expert_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                true
            }
            _ => unreachable!(),
        };
        Ok(pending)
    }

    pub(super) fn encode_moe_ffn_gate_up_wave_gpu(
        &self,
        cmd_buf: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        session: &mut MetalSession,
        ffn_gate: &MetalTensor,
        ffn_up: &MetalTensor,
        moe: &MetalMoeFfn,
        stage_recorder: Option<&mut DecodeStageRecorder>,
        stage_meta: Option<DecodeStageMeta>,
    ) -> Result<bool, MfError> {
        // The wave is one concurrent encoder: the routed gate/up must be a
        // single dispatch (reads `h` and the routing indices, writes only
        // `moe_inner`) to run beside the shared-expert projections.
        debug_assert!(routed_gate_up_single_dispatch(
            moe.gate_exps.dtype,
            moe.up_exps.dtype
        ));
        let enc = begin_decode_stage(cmd_buf, stage_recorder, stage_meta, true)?;
        self.encode_moe_routed_gate_up_gpu(&enc, session, moe)?;
        let shared_inner_fused =
            self.encode_moe_shared_ffn_gate_up_gpu(&enc, session, ffn_gate, ffn_up)?;
        enc.end();
        Ok(shared_inner_fused)
    }

    pub(super) fn encode_moe_ffn_down_wave_gpu(
        &self,
        cmd_buf: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        session: &mut MetalSession,
        ffn_down: &MetalTensor,
        moe: &MetalMoeFfn,
        mut stage_recorder: Option<&mut DecodeStageRecorder>,
        stage_meta: Option<DecodeStageMeta>,
    ) -> Result<bool, MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let n_expert = arch.expert_count as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;
        let moe_inner = session
            .moe_inner
            .view_subrange(0, vec![(topk * f_exp) as u64]);
        let moe_expert_out = session
            .moe_expert_out
            .view_subrange(0, vec![(topk * h) as u64]);
        let topk_idx = session.moe_topk_idx.view_subrange(0, vec![topk as u64]);
        let topk_w = session.moe_topk_weight.view_subrange(0, vec![topk as u64]);

        if decode_moe_noop_routed_down_enabled() {
            let enc = begin_decode_stage(cmd_buf, stage_recorder.as_deref_mut(), stage_meta, true)?;
            encode_fill_f32(self.ctx, &enc, &session.mixer_out, 0.0)?;
            self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
            enc.end();
            return Ok(false);
        }

        let pending = match moe.down_exps.dtype {
            GgmlType::Q4_K => {
                let enc =
                    begin_decode_stage(cmd_buf, stage_recorder.as_deref_mut(), stage_meta, true)?;
                encode_moe_down_q4_K_f32(
                    self.ctx,
                    &enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &moe_expert_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
                enc.end();
                true
            }
            GgmlType::Q5_K => {
                let enc =
                    begin_decode_stage(cmd_buf, stage_recorder.as_deref_mut(), stage_meta, true)?;
                let pending = if decode_moe_q5_down_fused_enabled() {
                    if f_exp == 512 && decode_moe_q5_down_k512_r2_enabled() {
                        encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
                            self.ctx,
                            &enc,
                            &moe.down_exps,
                            &moe_inner,
                            &topk_idx,
                            &topk_w,
                            &session.mixer_out,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                            1,
                        )?;
                    } else {
                        encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                            self.ctx,
                            &enc,
                            &moe.down_exps,
                            &moe_inner,
                            &topk_idx,
                            &topk_w,
                            &session.mixer_out,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                            1,
                        )?;
                    }
                    false
                } else {
                    encode_moe_down_q5_K_f32(
                        self.ctx,
                        &enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &moe_expert_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )?;
                    true
                };
                self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
                enc.end();
                pending
            }
            GgmlType::Q6_K => {
                let enc =
                    begin_decode_stage(cmd_buf, stage_recorder.as_deref_mut(), stage_meta, true)?;
                encode_moe_down_weighted_sum_q6_K_f32(
                    self.ctx,
                    &enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &topk_w,
                    &session.mixer_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
                enc.end();
                false
            }
            GgmlType::Q8_0 => {
                let enc =
                    begin_decode_stage(cmd_buf, stage_recorder.as_deref_mut(), stage_meta, true)?;
                encode_moe_down_weighted_sum_q8_0_f32(
                    self.ctx,
                    &enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &topk_w,
                    &session.mixer_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
                enc.end();
                false
            }
            GgmlType::IQ4_XS => {
                let enc =
                    begin_decode_stage(cmd_buf, stage_recorder.as_deref_mut(), stage_meta, true)?;
                if decode_moe_iq4_down_fast_enabled() {
                    encode_moe_down_iq4_xs_f32_fast(
                        self.ctx,
                        &enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &moe_expert_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )?;
                } else {
                    encode_moe_down_iq4_xs_f32(
                        self.ctx,
                        &enc,
                        &moe.down_exps,
                        &moe_inner,
                        &topk_idx,
                        &moe_expert_out,
                        f_exp,
                        h,
                        n_expert,
                        topk,
                    )?;
                }
                self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
                enc.end();
                true
            }
            GgmlType::IQ4_NL => {
                let enc =
                    begin_decode_stage(cmd_buf, stage_recorder.as_deref_mut(), stage_meta, true)?;
                encode_moe_down_iq4_nl_f32(
                    self.ctx,
                    &enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &moe_expert_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
                enc.end();
                true
            }
            GgmlType::F32 => {
                let enc =
                    begin_decode_stage(cmd_buf, stage_recorder.as_deref_mut(), stage_meta, true)?;
                encode_moe_down_f32_f32(
                    self.ctx,
                    &enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &moe_expert_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
                enc.end();
                true
            }
            GgmlType::BF16 => {
                let enc = begin_decode_stage(cmd_buf, stage_recorder, stage_meta, true)?;
                encode_moe_down_bf16_f32(
                    self.ctx,
                    &enc,
                    &moe.down_exps,
                    &moe_inner,
                    &topk_idx,
                    &moe_expert_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
                enc.end();
                true
            }
            _ => unreachable!(),
        };
        Ok(pending)
    }

    pub(super) fn encode_moe_ffn_final_wave_gpu(
        &self,
        cmd_buf: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        session: &mut MetalSession,
        routed_weighted_sum_is_pending: bool,
        stage_recorder: Option<&mut DecodeStageRecorder>,
        stage_meta: Option<DecodeStageMeta>,
    ) -> Result<(), MfError> {
        let h = self.model.arch.hidden_size as usize;
        let topk = self
            .model
            .arch
            .expert_used_count
            .min(self.model.arch.expert_count) as usize;
        let moe_expert_out = session
            .moe_expert_out
            .view_subrange(0, vec![(topk * h) as u64]);
        let topk_w = session.moe_topk_weight.view_subrange(0, vec![topk as u64]);

        let enc = begin_decode_stage(cmd_buf, stage_recorder, stage_meta, false)?;
        if routed_weighted_sum_is_pending {
            if decode_moe_grouped_finalizer_enabled() {
                let shared_out = session.ffn_out.view_subrange(0, vec![h as u64]);
                encode_moe_grouped_finalizer_f32(
                    self.ctx,
                    &enc,
                    &moe_expert_out,
                    &topk_w,
                    &session.moe_shared_gate,
                    &shared_out,
                    &session.x,
                    h,
                    topk,
                    1,
                )?;
            } else {
                encode_moe_weighted_sum_f32(
                    self.ctx,
                    &enc,
                    &moe_expert_out,
                    &topk_w,
                    &session.mixer_out,
                    h,
                    topk,
                )?;
                self.encode_moe_final_residual_gpu(&enc, session)?;
            }
        } else {
            self.encode_moe_final_residual_gpu(&enc, session)?;
        }
        enc.end();
        Ok(())
    }

    pub(crate) fn encode_moe_ffn_apply_gpu_concurrent_shared(
        &self,
        cmd_buf: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        session: &mut MetalSession,
        ffn_gate: &MetalTensor,
        ffn_up: &MetalTensor,
        ffn_down: &MetalTensor,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        if !routed_gate_up_single_dispatch(moe.gate_exps.dtype, moe.up_exps.dtype) {
            let enc = KernelEncoder::begin(cmd_buf);
            self.encode_moe_ffn_apply_gpu(&enc, session, ffn_gate, ffn_up, ffn_down, moe)?;
            enc.end();
            return Ok(());
        }

        let shared_inner_fused = self
            .encode_moe_ffn_gate_up_wave_gpu(cmd_buf, session, ffn_gate, ffn_up, moe, None, None)?;
        if !shared_inner_fused {
            let enc = KernelEncoder::begin(cmd_buf);
            self.encode_moe_shared_ffn_silu_gpu(&enc, session)?;
            enc.end();
        }
        let routed_weighted_sum_is_pending =
            self.encode_moe_ffn_down_wave_gpu(cmd_buf, session, ffn_down, moe, None, None)?;
        self.encode_moe_ffn_final_wave_gpu(
            cmd_buf,
            session,
            routed_weighted_sum_is_pending,
            None,
            None,
        )?;
        Ok(())
    }

    pub(super) fn encode_stage_profiled_moe_ffn_waves(
        &self,
        cmd_buf: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        recorder: &mut DecodeStageRecorder,
        block_kind: &'static str,
        block_idx: usize,
        local_idx: usize,
        session: &mut MetalSession,
        ffn_gate: &MetalTensor,
        ffn_up: &MetalTensor,
        ffn_down: &MetalTensor,
        moe: &MetalMoeFfn,
    ) -> Result<(), MfError> {
        if !routed_gate_up_single_dispatch(moe.gate_exps.dtype, moe.up_exps.dtype) {
            let enc = begin_decode_stage(
                cmd_buf,
                Some(&mut *recorder),
                Some(DecodeStageMeta {
                    family: "moe_fallback_ffn",
                    block_kind,
                    block_index: Some(block_idx),
                    local_index: Some(local_idx),
                }),
                false,
            )?;
            self.encode_moe_ffn_apply_gpu(&enc, session, ffn_gate, ffn_up, ffn_down, moe)?;
            enc.end();
            return Ok(());
        }

        let shared_inner_fused = self.encode_moe_ffn_gate_up_wave_gpu(
            cmd_buf,
            session,
            ffn_gate,
            ffn_up,
            moe,
            Some(&mut *recorder),
            Some(DecodeStageMeta {
                family: "moe_gate_up_wave",
                block_kind,
                block_index: Some(block_idx),
                local_index: Some(local_idx),
            }),
        )?;
        if !shared_inner_fused {
            let enc = begin_decode_stage(
                cmd_buf,
                Some(&mut *recorder),
                Some(DecodeStageMeta {
                    family: "moe_shared_silu",
                    block_kind,
                    block_index: Some(block_idx),
                    local_index: Some(local_idx),
                }),
                false,
            )?;
            self.encode_moe_shared_ffn_silu_gpu(&enc, session)?;
            enc.end();
        }
        let routed_weighted_sum_is_pending = self.encode_moe_ffn_down_wave_gpu(
            cmd_buf,
            session,
            ffn_down,
            moe,
            Some(&mut *recorder),
            Some(DecodeStageMeta {
                family: "moe_down_wave",
                block_kind,
                block_index: Some(block_idx),
                local_index: Some(local_idx),
            }),
        )?;
        self.encode_moe_ffn_final_wave_gpu(
            cmd_buf,
            session,
            routed_weighted_sum_is_pending,
            Some(&mut *recorder),
            Some(DecodeStageMeta {
                family: "moe_final_wave",
                block_kind,
                block_index: Some(block_idx),
                local_index: Some(local_idx),
            }),
        )
    }

    pub(super) fn encode_moe_block_gpu(
        &self,
        enc: &KernelEncoder,
        block: &MetalBlock,
        mixer_slot: MixerSlot,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        self.encode_moe_mixer_prep(enc, block, mixer_slot, position, session)?;
        let (ffn_gate, ffn_up, ffn_down, moe) = match block {
            MetalBlock::Gdn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
            MetalBlock::Attn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
        };
        let moe = moe.ok_or(MfError::UnsupportedMoe)?;
        self.encode_moe_route_prepare(enc, session, moe)?;
        self.encode_moe_ffn_apply_gpu(enc, session, ffn_gate, ffn_up, ffn_down, moe)
    }

    pub(super) fn moe_block_slot_by_index(
        &self,
        block_idx: usize,
    ) -> Result<(&MetalBlock, MixerSlot), MfError> {
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for (i, block) in self.model.blocks.iter().enumerate() {
            match block {
                MetalBlock::Gdn(_) => {
                    if i == block_idx {
                        return Ok((block, MixerSlot::Gdn(gdn_idx)));
                    }
                    gdn_idx += 1;
                }
                MetalBlock::Attn(_) => {
                    if i == block_idx {
                        return Ok((block, MixerSlot::Attn(attn_idx)));
                    }
                    attn_idx += 1;
                }
            }
        }
        Err(MfError::Metal(MetalError::BadShape {
            kernel: "encode_moe_block_by_index",
            detail: format!(
                "block index {block_idx} >= n_layer {}",
                self.model.blocks.len()
            ),
        }))
    }

    /// Bench hook: encode one MoE block by absolute block index.
    ///
    /// This keeps the public surface free of the internal `MixerSlot` enum while
    /// allowing block-slice microbenchmarks to execute normal attention/MoE work.
    pub fn encode_moe_block_by_index(
        &self,
        enc: &KernelEncoder,
        block_idx: usize,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let (block, slot) = self.moe_block_slot_by_index(block_idx)?;
        self.encode_moe_block_gpu(enc, block, slot, position, session)
    }

    /// Bench hook: run only the mixer + post-mixer norm for an absolute block.
    /// The caller may inspect `session.h` before running route/FFN.
    pub fn encode_moe_mixer_prep_by_index(
        &self,
        enc: &KernelEncoder,
        block_idx: usize,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let (block, slot) = self.moe_block_slot_by_index(block_idx)?;
        self.encode_moe_mixer_prep(enc, block, slot, position, session)
    }

    /// Bench hook: run only the route preparation for an absolute block.
    pub fn encode_moe_route_prepare_by_index(
        &self,
        enc: &KernelEncoder,
        block_idx: usize,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let (block, _) = self.moe_block_slot_by_index(block_idx)?;
        let moe = match block {
            MetalBlock::Gdn(b) => b.ffn_moe.as_ref(),
            MetalBlock::Attn(b) => b.ffn_moe.as_ref(),
        };
        self.encode_moe_route_prepare(enc, session, moe.ok_or(MfError::UnsupportedMoe)?)
    }

    /// Bench hook: run the route + MoE FFN tail after a caller has already
    /// computed mixer residual and post-mixer norm for this block.
    pub fn encode_moe_ffn_after_mixer_by_index(
        &self,
        enc: &KernelEncoder,
        block_idx: usize,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let (block, _) = self.moe_block_slot_by_index(block_idx)?;
        let (ffn_gate, ffn_up, ffn_down, moe) = match block {
            MetalBlock::Gdn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
            MetalBlock::Attn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
        };
        let moe = moe.ok_or(MfError::UnsupportedMoe)?;
        self.encode_moe_route_prepare(enc, session, moe)?;
        self.encode_moe_ffn_apply_gpu(enc, session, ffn_gate, ffn_up, ffn_down, moe)
    }

    /// Bench hook: run the route and MoE FFN tail with the production
    /// concurrent-shared policy after a caller has computed the mixer and norm.
    #[doc(hidden)]
    pub fn encode_moe_ffn_after_mixer_production_by_index(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        block_idx: usize,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let (block, _) = self.moe_block_slot_by_index(block_idx)?;
        let (ffn_gate, ffn_up, ffn_down, moe) = match block {
            MetalBlock::Gdn(block) => (
                &block.ffn_gate,
                &block.ffn_up,
                &block.ffn_down,
                block.ffn_moe.as_ref(),
            ),
            MetalBlock::Attn(block) => (
                &block.ffn_gate,
                &block.ffn_up,
                &block.ffn_down,
                block.ffn_moe.as_ref(),
            ),
        };
        let moe = moe.ok_or(MfError::UnsupportedMoe)?;
        if concurrent_shared_moe_decode_enabled() {
            let encoder = KernelEncoder::begin(command);
            self.encode_moe_route_prepare(&encoder, session, moe)?;
            encoder.end();
            self.encode_moe_ffn_apply_gpu_concurrent_shared(
                command, session, ffn_gate, ffn_up, ffn_down, moe,
            )
        } else {
            let encoder = KernelEncoder::begin(command);
            self.encode_moe_route_prepare(&encoder, session, moe)?;
            self.encode_moe_ffn_apply_gpu(&encoder, session, ffn_gate, ffn_up, ffn_down, moe)?;
            encoder.end();
            Ok(())
        }
    }

    /// Bench hook: encode only the shared-expert gate/up half of the production
    /// concurrent wave after route preparation. The routed inner may be supplied
    /// by an exact cross-lane kernel before the remaining tail is encoded.
    #[doc(hidden)]
    pub fn encode_moe_shared_gate_up_by_index(
        &self,
        encoder: &KernelEncoder,
        block_idx: usize,
        session: &mut MetalSession,
    ) -> Result<bool, MfError> {
        let (block, _) = self.moe_block_slot_by_index(block_idx)?;
        let (ffn_gate, ffn_up) = match block {
            MetalBlock::Gdn(block) => (&block.ffn_gate, &block.ffn_up),
            MetalBlock::Attn(block) => (&block.ffn_gate, &block.ffn_up),
        };
        self.encode_moe_shared_ffn_gate_up_gpu(encoder, session, ffn_gate, ffn_up)
    }

    /// Bench hook: finish production shared/routed down and final waves after a
    /// caller has supplied the exact routed inner and encoded shared gate/up.
    #[doc(hidden)]
    pub fn encode_moe_ffn_after_external_routed_inner_by_index(
        &self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        block_idx: usize,
        session: &mut MetalSession,
        shared_inner_fused: bool,
    ) -> Result<(), MfError> {
        let (block, _) = self.moe_block_slot_by_index(block_idx)?;
        let (ffn_down, moe) = match block {
            MetalBlock::Gdn(block) => (&block.ffn_down, block.ffn_moe.as_ref()),
            MetalBlock::Attn(block) => (&block.ffn_down, block.ffn_moe.as_ref()),
        };
        let moe = moe.ok_or(MfError::UnsupportedMoe)?;
        if !shared_inner_fused {
            let encoder = KernelEncoder::begin(command);
            self.encode_moe_shared_ffn_silu_gpu(&encoder, session)?;
            encoder.end();
        }
        let routed_weighted_sum_is_pending =
            self.encode_moe_ffn_down_wave_gpu(command, session, ffn_down, moe, None, None)?;
        self.encode_moe_ffn_final_wave_gpu(
            command,
            session,
            routed_weighted_sum_is_pending,
            None,
            None,
        )
    }

    pub(super) fn encode_moe_mixer_prep(
        &self,
        enc: &KernelEncoder,
        block: &MetalBlock,
        mixer_slot: MixerSlot,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let (attn_norm, post_norm) = match block {
            MetalBlock::Gdn(b) => (&b.attn_norm, &b.post_attn_norm),
            MetalBlock::Attn(b) => (&b.attn_norm, &b.post_attn_norm),
        };
        encode_rms_norm_mul_f32(self.ctx, enc, &session.x, attn_norm, &session.h, RMS_EPS)?;
        match (block, mixer_slot) {
            (MetalBlock::Gdn(g), MixerSlot::Gdn(idx)) => self.encode_gdn(enc, g, idx, session)?,
            (MetalBlock::Attn(a), MixerSlot::Attn(idx)) => {
                self.encode_attn(enc, a, idx, position, session)?
            }
            _ => {
                return Err(MfError::Metal(MetalError::BadShape {
                    kernel: "single_token_moe",
                    detail: "mixer slot type did not match block kind".into(),
                }));
            }
        }
        if decode_fused_residual_rmsnorm_enabled() {
            encode_residual_rms_norm_mul_f32(
                self.ctx,
                enc,
                &session.x,
                &session.mixer_out,
                post_norm,
                &session.h,
                RMS_EPS,
            )?;
        } else {
            encode_add_inplace_f32(self.ctx, enc, &session.x, &session.mixer_out)?;
            encode_rms_norm_mul_f32(self.ctx, enc, &session.x, post_norm, &session.h, RMS_EPS)?;
        }
        Ok(())
    }

    pub(super) fn encode_single_token_concurrent_gdn_moe_body(
        &self,
        cmd_buf: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(), MfError> {
        let h = self.model.arch.hidden_size as usize;
        {
            let enc = KernelEncoder::begin(cmd_buf);
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
                    let moe = g.ffn_moe.as_ref().ok_or(MfError::UnsupportedMoe)?;
                    {
                        let enc = KernelEncoder::begin(cmd_buf);
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
                        let enc = KernelEncoder::begin_concurrent(cmd_buf);
                        self.encode_gdn_front_projections(&enc, g, session)?;
                        enc.end();
                    }
                    {
                        let enc = KernelEncoder::begin(cmd_buf);
                        self.encode_gdn_after_projections(&enc, g, i, session)?;
                        encode_add_inplace_f32(self.ctx, &enc, &session.x, &session.mixer_out)?;
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.post_attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        self.encode_moe_route_prepare(&enc, session, moe)?;
                        if !concurrent_shared_moe_decode_enabled() {
                            self.encode_moe_ffn_apply_gpu(
                                &enc,
                                session,
                                &g.ffn_gate,
                                &g.ffn_up,
                                &g.ffn_down,
                                moe,
                            )?;
                        }
                        enc.end();
                    }
                    if concurrent_shared_moe_decode_enabled() {
                        self.encode_moe_ffn_apply_gpu_concurrent_shared(
                            cmd_buf,
                            session,
                            &g.ffn_gate,
                            &g.ffn_up,
                            &g.ffn_down,
                            moe,
                        )?;
                    }
                }
                MetalBlock::Attn(a) => {
                    let slot = MixerSlot::Attn(attn_idx);
                    attn_idx += 1;
                    let enc = KernelEncoder::begin(cmd_buf);
                    self.encode_moe_mixer_prep(&enc, block, slot, position, session)?;
                    let moe = a.ffn_moe.as_ref().ok_or(MfError::UnsupportedMoe)?;
                    self.encode_moe_route_prepare(&enc, session, moe)?;
                    if !concurrent_shared_moe_decode_enabled() {
                        self.encode_moe_ffn_apply_gpu(
                            &enc,
                            session,
                            &a.ffn_gate,
                            &a.ffn_up,
                            &a.ffn_down,
                            moe,
                        )?;
                    }
                    enc.end();
                    if concurrent_shared_moe_decode_enabled() {
                        self.encode_moe_ffn_apply_gpu_concurrent_shared(
                            cmd_buf,
                            session,
                            &a.ffn_gate,
                            &a.ffn_up,
                            &a.ffn_down,
                            moe,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn single_token_profiled_concurrent_gdn_moe_tail_inner(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        tail: LmHeadTail<'_>,
        validate_tail: bool,
    ) -> Result<(TokenProfile, LmHeadTailEvidence, std::time::Instant), MfError> {
        session.ensure_usable()?;
        let arch = &self.model.arch;
        if arch.kind != ArchKind::Moe {
            return Err(MfError::UnsupportedMoe);
        }
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let mut evidence = if validate_tail {
            self.validate_lm_head_tail(session, tail)?
        } else {
            if !matches!(tail, LmHeadTail::Resident) {
                return Err(lm_head_tail_error(
                    "unchecked tail execution is resident-only",
                ));
            }
            self.resident_lm_head_tail_evidence(session)
        };
        let t_total = std::time::Instant::now();

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        self.encode_single_token_concurrent_gdn_moe_body(&cmd_buf, position, session)?;
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
            self.encode_lm_head_tail(&enc, session, tail)?;
            enc.end();
        }

        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;
        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        crate::metal::wait_unchecked(&cmd_buf);
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let gpu_kernel_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;
        // Checked on every path: a discarded command leaves partial logits
        // and a half-advanced recurrent state.
        evidence.command_completed = cmd_buf.status() == MTLCommandBufferStatus::Completed;
        evidence.command_error_none = cmd_buf.error().is_none();
        if let Err(failure) = crate::metal::command_buffer_completed(&cmd_buf) {
            session.poison("MoE concurrent-GDN decode command failed");
            return Err(failure.into());
        }

        Ok((
            TokenProfile {
                cpu_encode_ms,
                cpu_to_gpu_complete_ms,
                gpu_kernel_ms,
                total_ms: t_total.elapsed().as_secs_f64() * 1e3,
                moe_cpu_route_ms: 0.0,
                moe_cmd_count: 1,
            },
            evidence,
            t_total,
        ))
    }

    #[doc(hidden)]
    pub fn single_token_profiled_concurrent_gdn_moe_with_tail(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        tail: LmHeadTail<'_>,
    ) -> Result<(TokenProfile, LmHeadTailEvidence), MfError> {
        let (profile, evidence, _) = self.single_token_profiled_concurrent_gdn_moe_tail_inner(
            token_id, position, session, tail, true,
        )?;
        Ok((profile, evidence))
    }

    pub fn single_token_profiled_concurrent_gdn_moe(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(Vec<f32>, TokenProfile), MfError> {
        let (mut profile, evidence, t_total) = self
            .single_token_profiled_concurrent_gdn_moe_tail_inner(
                token_id,
                position,
                session,
                LmHeadTail::Resident,
                false,
            )?;
        debug_assert_eq!(evidence.kind, LmHeadTailKind::Resident);
        let mut out = vec![0.0f32; self.model.arch.vocab_size as usize];
        unsafe {
            let src = (session.logits.buffer.contents().as_ptr() as *const u8)
                .add(session.logits.offset as usize) as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        profile.total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((out, profile))
    }

    pub fn single_token_argmax_profiled_concurrent_gdn_moe(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(i32, TokenProfile), MfError> {
        self.single_token_argmax_profiled_concurrent_gdn_moe_with_reduction(
            token_id,
            position,
            session,
            ArgmaxReduction::SpeculativeLowest,
        )
    }

    pub(super) fn single_token_argmax_profiled_concurrent_gdn_moe_with_reduction(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        reduction: ArgmaxReduction,
    ) -> Result<(i32, TokenProfile), MfError> {
        let arch = &self.model.arch;
        if arch.kind != ArchKind::Moe {
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
                    let moe = g.ffn_moe.as_ref().ok_or(MfError::UnsupportedMoe)?;
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
                        encode_add_inplace_f32(self.ctx, &enc, &session.x, &session.mixer_out)?;
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.post_attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        self.encode_moe_route_prepare(&enc, session, moe)?;
                        if !concurrent_shared_moe_decode_enabled() {
                            self.encode_moe_ffn_apply_gpu(
                                &enc,
                                session,
                                &g.ffn_gate,
                                &g.ffn_up,
                                &g.ffn_down,
                                moe,
                            )?;
                        }
                        enc.end();
                    }
                    if concurrent_shared_moe_decode_enabled() {
                        self.encode_moe_ffn_apply_gpu_concurrent_shared(
                            &cmd_buf,
                            session,
                            &g.ffn_gate,
                            &g.ffn_up,
                            &g.ffn_down,
                            moe,
                        )?;
                    }
                }
                MetalBlock::Attn(a) => {
                    let slot = MixerSlot::Attn(attn_idx);
                    attn_idx += 1;
                    let enc = KernelEncoder::begin(&cmd_buf);
                    self.encode_moe_mixer_prep(&enc, block, slot, position, session)?;
                    let moe = a.ffn_moe.as_ref().ok_or(MfError::UnsupportedMoe)?;
                    self.encode_moe_route_prepare(&enc, session, moe)?;
                    if !concurrent_shared_moe_decode_enabled() {
                        self.encode_moe_ffn_apply_gpu(
                            &enc,
                            session,
                            &a.ffn_gate,
                            &a.ffn_up,
                            &a.ffn_down,
                            moe,
                        )?;
                    }
                    enc.end();
                    if concurrent_shared_moe_decode_enabled() {
                        self.encode_moe_ffn_apply_gpu_concurrent_shared(
                            &cmd_buf,
                            session,
                            &a.ffn_gate,
                            &a.ffn_up,
                            &a.ffn_down,
                            moe,
                        )?;
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

    pub fn single_token_argmax_stage_profiled_concurrent_gdn_moe(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        split_attn_route: bool,
        split_attn_detail: bool,
        split_gdn_after: bool,
    ) -> Result<(i32, DecodeStageProfile), MfError> {
        let arch = &self.model.arch;
        if arch.kind != ArchKind::Moe {
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
        let sample_count = 2 * (2 + self.model.blocks.len() * 16);
        let mut recorder = DecodeStageRecorder::new(self.ctx, sample_count)?;

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");

        {
            let enc = begin_decode_stage(
                &cmd_buf,
                Some(&mut recorder),
                Some(DecodeStageMeta {
                    family: "embed",
                    block_kind: "tail",
                    block_index: None,
                    local_index: None,
                }),
                false,
            )?;
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
        for (block_idx, block) in self.model.blocks.iter().enumerate() {
            match block {
                MetalBlock::Gdn(g) => {
                    let local_idx = gdn_idx;
                    gdn_idx += 1;
                    let moe = g.ffn_moe.as_ref().ok_or(MfError::UnsupportedMoe)?;
                    {
                        let enc = begin_decode_stage(
                            &cmd_buf,
                            Some(&mut recorder),
                            Some(DecodeStageMeta {
                                family: "gdn_pre_norm",
                                block_kind: "gdn",
                                block_index: Some(block_idx),
                                local_index: Some(local_idx),
                            }),
                            false,
                        )?;
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
                        let enc = begin_decode_stage(
                            &cmd_buf,
                            Some(&mut recorder),
                            Some(DecodeStageMeta {
                                family: "gdn_front",
                                block_kind: "gdn",
                                block_index: Some(block_idx),
                                local_index: Some(local_idx),
                            }),
                            true,
                        )?;
                        self.encode_gdn_front_projections(&enc, g, session)?;
                        enc.end();
                    }
                    if split_gdn_after {
                        let v_dim = arch.gdn_n_v_heads as usize * arch.gdn_head_dim as usize;
                        let gdn_qkv = session.gdn_qkv.clone();
                        let gdn_z = session.gdn_z.clone();
                        let gdn_alpha = session.gdn_alpha.clone();
                        let gdn_beta = session.gdn_beta.clone();
                        let gdn_normed = session.gdn_normed.clone();

                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "gdn_beta_alpha",
                                    block_kind: "gdn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
                            if !gdn_beta_projection_fused(g) {
                                encode_sigmoid_f32(
                                    self.ctx,
                                    &enc,
                                    &session.gdn_b,
                                    &session.gdn_beta,
                                )?;
                            }
                            encode_gdn_decay_chain_f32(
                                self.ctx,
                                &enc,
                                &session.gdn_a,
                                &g.dt_bias,
                                &g.a_log,
                                &session.gdn_alpha,
                            )?;
                            enc.end();
                        }
                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "gdn_tail",
                                    block_kind: "gdn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
                            self.encode_gdn_tail(
                                &enc,
                                g,
                                local_idx,
                                session,
                                &gdn_qkv,
                                &gdn_z,
                                &gdn_alpha,
                                &gdn_beta,
                                &gdn_normed,
                            )?;
                            enc.end();
                        }
                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "gdn_out_proj",
                                    block_kind: "gdn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
                            if decode_gdn_noop_out_enabled() {
                                encode_fill_f32(self.ctx, &enc, &session.mixer_out, 0.0)?;
                            } else {
                                encode_mat_vec_dispatch(
                                    self.ctx,
                                    &enc,
                                    &g.out_proj,
                                    &gdn_normed,
                                    &session.mixer_out,
                                    v_dim,
                                    h,
                                )?;
                            }
                            enc.end();
                        }
                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "gdn_resid_post_norm",
                                    block_kind: "gdn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
                            encode_add_inplace_f32(self.ctx, &enc, &session.x, &session.mixer_out)?;
                            encode_rms_norm_mul_f32(
                                self.ctx,
                                &enc,
                                &session.x,
                                &g.post_attn_norm,
                                &session.h,
                                RMS_EPS,
                            )?;
                            enc.end();
                        }
                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: if concurrent_shared_moe_decode_enabled() {
                                        "gdn_route"
                                    } else {
                                        "gdn_route_ffn_serial"
                                    },
                                    block_kind: "gdn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
                            self.encode_moe_route_prepare(&enc, session, moe)?;
                            if !concurrent_shared_moe_decode_enabled() {
                                self.encode_moe_ffn_apply_gpu(
                                    &enc,
                                    session,
                                    &g.ffn_gate,
                                    &g.ffn_up,
                                    &g.ffn_down,
                                    moe,
                                )?;
                            }
                            enc.end();
                        }
                    } else {
                        let enc = begin_decode_stage(
                            &cmd_buf,
                            Some(&mut recorder),
                            Some(DecodeStageMeta {
                                family: if concurrent_shared_moe_decode_enabled() {
                                    "gdn_after_route"
                                } else {
                                    "gdn_after_route_ffn_serial"
                                },
                                block_kind: "gdn",
                                block_index: Some(block_idx),
                                local_index: Some(local_idx),
                            }),
                            false,
                        )?;
                        self.encode_gdn_after_projections(&enc, g, local_idx, session)?;
                        encode_add_inplace_f32(self.ctx, &enc, &session.x, &session.mixer_out)?;
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.post_attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        self.encode_moe_route_prepare(&enc, session, moe)?;
                        if !concurrent_shared_moe_decode_enabled() {
                            self.encode_moe_ffn_apply_gpu(
                                &enc,
                                session,
                                &g.ffn_gate,
                                &g.ffn_up,
                                &g.ffn_down,
                                moe,
                            )?;
                        }
                        enc.end();
                    }
                    if concurrent_shared_moe_decode_enabled() {
                        self.encode_stage_profiled_moe_ffn_waves(
                            &cmd_buf,
                            &mut recorder,
                            "gdn",
                            block_idx,
                            local_idx,
                            session,
                            &g.ffn_gate,
                            &g.ffn_up,
                            &g.ffn_down,
                            moe,
                        )?;
                    }
                }
                MetalBlock::Attn(a) => {
                    let local_idx = attn_idx;
                    attn_idx += 1;
                    let slot = MixerSlot::Attn(local_idx);
                    let moe = a.ffn_moe.as_ref().ok_or(MfError::UnsupportedMoe)?;
                    if split_attn_detail {
                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "attn_pre_norm",
                                    block_kind: "attn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
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
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "attn_front_proj",
                                    block_kind: "attn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
                            self.encode_attn_front_projections(&enc, a, session)?;
                            enc.end();
                        }
                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "attn_body_out",
                                    block_kind: "attn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
                            self.encode_attn_after_projections(
                                &enc, a, local_idx, position, session,
                            )?;
                            enc.end();
                        }
                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "attn_resid_post_norm",
                                    block_kind: "attn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
                            encode_add_inplace_f32(self.ctx, &enc, &session.x, &session.mixer_out)?;
                            encode_rms_norm_mul_f32(
                                self.ctx,
                                &enc,
                                &session.x,
                                &a.post_attn_norm,
                                &session.h,
                                RMS_EPS,
                            )?;
                            enc.end();
                        }
                        let enc = begin_decode_stage(
                            &cmd_buf,
                            Some(&mut recorder),
                            Some(DecodeStageMeta {
                                family: if concurrent_shared_moe_decode_enabled() {
                                    "attn_route"
                                } else {
                                    "attn_route_ffn_serial"
                                },
                                block_kind: "attn",
                                block_index: Some(block_idx),
                                local_index: Some(local_idx),
                            }),
                            false,
                        )?;
                        self.encode_moe_route_prepare(&enc, session, moe)?;
                        if !concurrent_shared_moe_decode_enabled() {
                            self.encode_moe_ffn_apply_gpu(
                                &enc,
                                session,
                                &a.ffn_gate,
                                &a.ffn_up,
                                &a.ffn_down,
                                moe,
                            )?;
                        }
                        enc.end();
                    } else if split_attn_route {
                        {
                            let enc = begin_decode_stage(
                                &cmd_buf,
                                Some(&mut recorder),
                                Some(DecodeStageMeta {
                                    family: "attn_mixer",
                                    block_kind: "attn",
                                    block_index: Some(block_idx),
                                    local_index: Some(local_idx),
                                }),
                                false,
                            )?;
                            self.encode_moe_mixer_prep(&enc, block, slot, position, session)?;
                            enc.end();
                        }
                        let enc = begin_decode_stage(
                            &cmd_buf,
                            Some(&mut recorder),
                            Some(DecodeStageMeta {
                                family: if concurrent_shared_moe_decode_enabled() {
                                    "attn_route"
                                } else {
                                    "attn_route_ffn_serial"
                                },
                                block_kind: "attn",
                                block_index: Some(block_idx),
                                local_index: Some(local_idx),
                            }),
                            false,
                        )?;
                        self.encode_moe_route_prepare(&enc, session, moe)?;
                        if !concurrent_shared_moe_decode_enabled() {
                            self.encode_moe_ffn_apply_gpu(
                                &enc,
                                session,
                                &a.ffn_gate,
                                &a.ffn_up,
                                &a.ffn_down,
                                moe,
                            )?;
                        }
                        enc.end();
                    } else {
                        let enc = begin_decode_stage(
                            &cmd_buf,
                            Some(&mut recorder),
                            Some(DecodeStageMeta {
                                family: if concurrent_shared_moe_decode_enabled() {
                                    "attn_mixer_route"
                                } else {
                                    "attn_mixer_route_ffn_serial"
                                },
                                block_kind: "attn",
                                block_index: Some(block_idx),
                                local_index: Some(local_idx),
                            }),
                            false,
                        )?;
                        self.encode_moe_mixer_prep(&enc, block, slot, position, session)?;
                        self.encode_moe_route_prepare(&enc, session, moe)?;
                        if !concurrent_shared_moe_decode_enabled() {
                            self.encode_moe_ffn_apply_gpu(
                                &enc,
                                session,
                                &a.ffn_gate,
                                &a.ffn_up,
                                &a.ffn_down,
                                moe,
                            )?;
                        }
                        enc.end();
                    }
                    if concurrent_shared_moe_decode_enabled() {
                        self.encode_stage_profiled_moe_ffn_waves(
                            &cmd_buf,
                            &mut recorder,
                            "attn",
                            block_idx,
                            local_idx,
                            session,
                            &a.ffn_gate,
                            &a.ffn_up,
                            &a.ffn_down,
                            moe,
                        )?;
                    }
                }
            }
        }

        {
            let enc = begin_decode_stage(
                &cmd_buf,
                Some(&mut recorder),
                Some(DecodeStageMeta {
                    family: "tail_lm_head_argmax",
                    block_kind: "tail",
                    block_index: None,
                    local_index: None,
                }),
                false,
            )?;
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
            encode_argmax_f32(
                self.ctx,
                &enc,
                &session.logits,
                &argmax_tok,
                1,
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

        let argmax = unsafe {
            let src = argmax_tok.buffer.contents().as_ptr() as *const i32;
            *src
        };
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        let token = TokenProfile {
            cpu_encode_ms,
            cpu_to_gpu_complete_ms,
            gpu_kernel_ms,
            total_ms,
            moe_cpu_route_ms: 0.0,
            moe_cmd_count: 1,
        };
        let profile = recorder.resolve(self.ctx, token)?;
        Ok((argmax, profile))
    }

    pub(super) fn single_token_moe(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<Vec<f32>, MfError> {
        let (logits, _) = if concurrent_gdn_moe_decode_enabled() {
            self.single_token_profiled_concurrent_gdn_moe(token_id, position, session)?
        } else {
            self.single_token_profiled_moe(token_id, position, session)?
        };
        Ok(logits)
    }

    pub(super) fn encode_single_token_argmax_moe_with_reduction(
        &self,
        enc: &KernelEncoder,
        position: u32,
        session: &mut MetalSession,
        ids_buf: &MetalTensor,
        argmax_tok: &MetalTensor,
        reduction: ArgmaxReduction,
    ) -> Result<(), MfError> {
        let arch = &self.model.arch;
        let h = arch.hidden_size as usize;

        encode_get_rows_f32(
            self.ctx,
            enc,
            &self.model.token_embd,
            ids_buf,
            &session.x,
            1,
            h,
        )?;

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for block in &self.model.blocks {
            let slot = match block {
                MetalBlock::Gdn(_) => {
                    let s = MixerSlot::Gdn(gdn_idx);
                    gdn_idx += 1;
                    s
                }
                MetalBlock::Attn(_) => {
                    let s = MixerSlot::Attn(attn_idx);
                    attn_idx += 1;
                    s
                }
            };
            self.encode_moe_block_gpu(enc, block, slot, position, session)?;
        }

        encode_rms_norm_mul_f32(
            self.ctx,
            enc,
            &session.x,
            &self.model.output_norm,
            &session.h,
            RMS_EPS,
        )?;
        encode_mat_vec_dispatch(
            self.ctx,
            enc,
            &self.model.lm_head,
            &session.h,
            &session.logits,
            h,
            arch.vocab_size as usize,
        )?;
        encode_argmax_reduction(
            self.ctx,
            enc,
            &session.logits,
            argmax_tok,
            1,
            arch.vocab_size as usize,
            reduction,
        )?;
        Ok(())
    }

    pub(super) fn single_token_profiled_moe(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<(Vec<f32>, TokenProfile), MfError> {
        let arch = &self.model.arch;
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

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for block in &self.model.blocks {
            let slot = match block {
                MetalBlock::Gdn(_) => {
                    let s = MixerSlot::Gdn(gdn_idx);
                    gdn_idx += 1;
                    s
                }
                MetalBlock::Attn(_) => {
                    let s = MixerSlot::Attn(attn_idx);
                    attn_idx += 1;
                    s
                }
            };
            self.encode_moe_block_gpu(&enc, block, slot, position, session)?;
        }

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
        enc.end();
        let cpu_encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;

        let t_gpu = std::time::Instant::now();
        cmd_buf.commit();
        crate::metal::wait_completed(&cmd_buf)?;
        let cpu_to_gpu_complete_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let gpu_kernel_ms = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e3;

        let mut logits = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, logits.as_mut_ptr(), logits.len());
        }
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((
            logits,
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

    pub(super) fn single_token_argmax_profiled_moe(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
        reduction: ArgmaxReduction,
    ) -> Result<(i32, TokenProfile), MfError> {
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let t_total = std::time::Instant::now();

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let ids_buf = session.ids_buf.clone();
        let argmax_tok = session.argmax_tok.clone();

        let t_encode = std::time::Instant::now();
        let cmd_buf = self.ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd_buf);
        self.encode_single_token_argmax_moe_with_reduction(
            &enc,
            position,
            session,
            &ids_buf,
            &argmax_tok,
            reduction,
        )?;
        enc.end();
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

    /// Decode one MoE token and return per-layer post-norm hidden vectors plus
    /// routed expert ids/weights after each real route kernel. This is bench-only
    /// instrumentation for replaying realistic route patterns in isolated MoE
    /// microbenches.
    pub fn capture_moe_gateup_replay_for_token(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<Vec<MoeRouteReplayRow>, MfError> {
        let arch = &self.model.arch;
        if arch.kind != ArchKind::Moe {
            return Err(MfError::UnsupportedMoe);
        }
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;

        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
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
            cmd.commit();
            crate::metal::wait_completed(&cmd)?;
        }

        let mut routes = Vec::new();
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        for block in &self.model.blocks {
            let slot = match block {
                MetalBlock::Gdn(_) => {
                    let s = MixerSlot::Gdn(gdn_idx);
                    gdn_idx += 1;
                    s
                }
                MetalBlock::Attn(_) => {
                    let s = MixerSlot::Attn(attn_idx);
                    attn_idx += 1;
                    s
                }
            };
            let (ffn_gate, ffn_up, ffn_down, moe) = match block {
                MetalBlock::Gdn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
                MetalBlock::Attn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
            };
            let moe = moe.ok_or(MfError::UnsupportedMoe)?;

            {
                let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                self.encode_moe_mixer_prep(&enc, block, slot, position, session)?;
                enc.end();
                cmd.commit();
                crate::metal::wait_completed(&cmd)?;
            }
            {
                let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                self.encode_moe_route_prepare(&enc, session, moe)?;
                enc.end();
                cmd.commit();
                crate::metal::wait_completed(&cmd)?;
            }

            let route = self.read_moe_route_result(session, topk);
            let hidden = unsafe {
                let src = (session.h.buffer.contents().as_ptr() as *const f32)
                    .add((session.h.offset / 4) as usize);
                std::slice::from_raw_parts(src, h).to_vec()
            };
            let topk_idx = route
                .ranked
                .iter()
                .map(|&(expert, _)| expert as i32)
                .collect();
            let topk_weight = route.ranked.iter().map(|&(_, weight)| weight).collect();
            routes.push(MoeRouteReplayRow {
                topk_idx,
                topk_weight,
                hidden,
            });

            {
                let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                if concurrent_shared_moe_decode_enabled() {
                    self.encode_moe_ffn_apply_gpu_concurrent_shared(
                        &cmd, session, ffn_gate, ffn_up, ffn_down, moe,
                    )?;
                } else {
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_ffn_apply_gpu(&enc, session, ffn_gate, ffn_up, ffn_down, moe)?;
                    enc.end();
                }
                cmd.commit();
                crate::metal::wait_completed(&cmd)?;
            }
        }
        Ok(routes)
    }

    pub(super) fn single_token_phase_profiled_moe(
        &self,
        token_id: i32,
        position: u32,
        session: &mut MetalSession,
    ) -> Result<PhaseProfileOutput, MfError> {
        let arch = &self.model.arch;
        if token_id < 0 || (token_id as u32) >= arch.vocab_size {
            return Err(MfError::BadToken(token_id, arch.vocab_size));
        }
        let h = arch.hidden_size as usize;
        let t_total = std::time::Instant::now();
        unsafe {
            let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = token_id;
        }

        let mut phases: Vec<(String, f64)> = Vec::new();
        {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
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
            cmd.commit();
            crate::metal::wait_completed(&cmd)?;
            phases.push((
                "embedding".into(),
                (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
            ));
        }

        let mut gdn_pre_norm_total_ms = 0.0f64;
        let mut gdn_front_proj_total_ms = 0.0f64;
        let mut gdn_qkv_proj_total_ms = 0.0f64;
        let mut gdn_z_proj_total_ms = 0.0f64;
        let mut gdn_beta_proj_total_ms = 0.0f64;
        let mut gdn_alpha_proj_total_ms = 0.0f64;
        let mut gdn_alpha_beta_total_ms = 0.0f64;
        let mut gdn_tail_total_ms = 0.0f64;
        let mut gdn_tail_conv_total_ms = 0.0f64;
        let mut gdn_tail_l2_total_ms = 0.0f64;
        let mut gdn_tail_step_total_ms = 0.0f64;
        let mut gdn_tail_norm_total_ms = 0.0f64;
        let mut gdn_out_proj_total_ms = 0.0f64;
        let mut gdn_resid_post_total_ms = 0.0f64;
        let mut attn_mixer_total_ms = 0.0f64;
        let mut route_total_ms = 0.0f64;
        let mut route_logits_total_ms = 0.0f64;
        let mut route_topk_total_ms = 0.0f64;
        let mut route_shared_total_ms = 0.0f64;
        let mut ffn_apply_total_ms = 0.0f64;
        let mut ffn_gate_up_wave_total_ms = 0.0f64;
        let mut ffn_shared_silu_total_ms = 0.0f64;
        let mut ffn_down_wave_total_ms = 0.0f64;
        let mut ffn_finalizer_total_ms = 0.0f64;
        let mut ffn_routed_gate_up_total_ms = 0.0f64;
        let mut ffn_routed_down_total_ms = 0.0f64;
        let mut ffn_shared_gate_up_total_ms = 0.0f64;
        let mut ffn_shared_down_total_ms = 0.0f64;
        let mut ffn_fallback_routed_total_ms = 0.0f64;
        let mut ffn_fallback_shared_total_ms = 0.0f64;
        let mut gdn_count = 0usize;
        let mut attn_count = 0usize;
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        let split_gdn_proj = phase_gdn_proj_split_enabled();
        let split_gdn_tail = phase_gdn_tail_split_enabled();
        let route_replay = phase_moe_route_replay_enabled();
        let split_route_deep = !route_replay && phase_moe_route_deep_split_enabled();
        let split_route = !route_replay && (phase_moe_route_split_enabled() || split_route_deep);
        let cpu_route = phase_moe_cpu_route_enabled();
        let split_ffn_apply = phase_moe_ffn_split_enabled();
        let deep_split_ffn_apply = phase_moe_ffn_deep_split_enabled();
        for block in &self.model.blocks {
            let slot = match block {
                MetalBlock::Gdn(_) => {
                    let s = MixerSlot::Gdn(gdn_idx);
                    gdn_idx += 1;
                    gdn_count += 1;
                    s
                }
                MetalBlock::Attn(_) => {
                    let s = MixerSlot::Attn(attn_idx);
                    attn_idx += 1;
                    attn_count += 1;
                    s
                }
            };
            let (ffn_gate, ffn_up, ffn_down, moe) = match block {
                MetalBlock::Gdn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
                MetalBlock::Attn(b) => (&b.ffn_gate, &b.ffn_up, &b.ffn_down, b.ffn_moe.as_ref()),
            };
            let moe = moe.ok_or(MfError::UnsupportedMoe)?;

            match (block, slot) {
                (MetalBlock::Gdn(g), MixerSlot::Gdn(gdn_i)) => {
                    let v_dim = self.model.arch.gdn_n_v_heads as usize
                        * self.model.arch.gdn_head_dim as usize;

                    {
                        let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        enc.end();
                        cmd.commit();
                        crate::metal::wait_completed(&cmd)?;
                        gdn_pre_norm_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }
                    if split_gdn_proj {
                        let n_v = self.model.arch.gdn_n_v_heads as usize;
                        let n_k = self.model.arch.gdn_n_k_heads as usize;
                        let head_dim = self.model.arch.gdn_head_dim as usize;
                        let conv_dim = (2 * n_k + n_v) * head_dim;
                        {
                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            if decode_gdn_noop_qkv_enabled() {
                                encode_fill_f32(self.ctx, &enc, &session.gdn_qkv, 0.0)?;
                            } else {
                                encode_mat_vec_dispatch(
                                    self.ctx,
                                    &enc,
                                    &g.in_proj_qkv,
                                    &session.h,
                                    &session.gdn_qkv,
                                    h,
                                    conv_dim,
                                )?;
                            }
                            enc.end();
                            cmd.commit();
                            crate::metal::wait_completed(&cmd)?;
                            gdn_qkv_proj_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        }
                        {
                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            if decode_gdn_noop_z_enabled() {
                                encode_fill_f32(self.ctx, &enc, &session.gdn_z, 0.0)?;
                            } else {
                                encode_mat_vec_dispatch(
                                    self.ctx,
                                    &enc,
                                    &g.in_proj_z,
                                    &session.h,
                                    &session.gdn_z,
                                    h,
                                    v_dim,
                                )?;
                            }
                            enc.end();
                            cmd.commit();
                            crate::metal::wait_completed(&cmd)?;
                            gdn_z_proj_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        }
                        {
                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            if decode_gdn_noop_beta_enabled() {
                                encode_fill_f32(self.ctx, &enc, &session.gdn_b, 0.0)?;
                            } else if gdn_beta_projection_fused(g) {
                                encode_mat_vec_f32_sigmoid(
                                    self.ctx,
                                    &enc,
                                    &g.beta_proj,
                                    &session.h,
                                    &session.gdn_beta,
                                    h,
                                    n_v,
                                )?;
                            } else {
                                encode_mat_vec_dispatch(
                                    self.ctx,
                                    &enc,
                                    &g.beta_proj,
                                    &session.h,
                                    &session.gdn_b,
                                    h,
                                    n_v,
                                )?;
                            }
                            enc.end();
                            cmd.commit();
                            crate::metal::wait_completed(&cmd)?;
                            gdn_beta_proj_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        }
                        {
                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            if decode_gdn_noop_alpha_enabled() {
                                encode_fill_f32(self.ctx, &enc, &session.gdn_a, 0.0)?;
                            } else {
                                encode_mat_vec_dispatch(
                                    self.ctx,
                                    &enc,
                                    &g.alpha_proj,
                                    &session.h,
                                    &session.gdn_a,
                                    h,
                                    n_v,
                                )?;
                            }
                            enc.end();
                            cmd.commit();
                            crate::metal::wait_completed(&cmd)?;
                            gdn_alpha_proj_total_ms +=
                                (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        }
                    } else {
                        let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        self.encode_gdn_front_projections(&enc, g, session)?;
                        enc.end();
                        cmd.commit();
                        crate::metal::wait_completed(&cmd)?;
                        gdn_front_proj_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }
                    {
                        let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        if !gdn_beta_projection_fused(g) {
                            encode_sigmoid_f32(self.ctx, &enc, &session.gdn_b, &session.gdn_beta)?;
                        }
                        encode_gdn_decay_chain_f32(
                            self.ctx,
                            &enc,
                            &session.gdn_a,
                            &g.dt_bias,
                            &g.a_log,
                            &session.gdn_alpha,
                        )?;
                        enc.end();
                        cmd.commit();
                        crate::metal::wait_completed(&cmd)?;
                        gdn_alpha_beta_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }
                    {
                        let gdn_qkv = session.gdn_qkv.clone();
                        let gdn_z = session.gdn_z.clone();
                        let gdn_alpha = session.gdn_alpha.clone();
                        let gdn_beta = session.gdn_beta.clone();
                        let gdn_normed = session.gdn_normed.clone();
                        if split_gdn_tail {
                            let n_v = self.model.arch.gdn_n_v_heads as usize;
                            let n_k = self.model.arch.gdn_n_k_heads as usize;
                            let head_dim = self.model.arch.gdn_head_dim as usize;
                            let conv_dim = (2 * n_k + n_v) * head_dim;
                            let q_view = session
                                .gdn_qkv_conv
                                .view_subrange(0, vec![(n_k * head_dim) as u64]);
                            let k_view = session.gdn_qkv_conv.view_subrange(
                                (n_k * head_dim) as u64,
                                vec![(n_k * head_dim) as u64],
                            );
                            let v_view = session.gdn_qkv_conv.view_subrange(
                                (2 * n_k * head_dim) as u64,
                                vec![(n_v * head_dim) as u64],
                            );
                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            encode_ssm_conv_silu_f32(
                                self.ctx,
                                &enc,
                                &gdn_qkv,
                                &session.gdn_conv[gdn_i],
                                &g.conv1d,
                                &session.gdn_qkv_conv,
                                conv_dim,
                            )?;
                            enc.end();
                            cmd.commit();
                            crate::metal::wait_completed(&cmd)?;
                            gdn_tail_conv_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            encode_l2_norm_batched_f32(
                                self.ctx,
                                &enc,
                                &q_view,
                                &session.gdn_q_norm,
                                n_k,
                                head_dim,
                                RMS_EPS,
                            )?;
                            encode_l2_norm_batched_f32(
                                self.ctx,
                                &enc,
                                &k_view,
                                &session.gdn_k_norm,
                                n_k,
                                head_dim,
                                RMS_EPS,
                            )?;
                            enc.end();
                            cmd.commit();
                            crate::metal::wait_completed(&cmd)?;
                            gdn_tail_l2_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            encode_gdn_step_decay_f32(
                                self.ctx,
                                &enc,
                                &session.gdn_q_norm,
                                &session.gdn_k_norm,
                                &v_view,
                                &gdn_alpha,
                                &gdn_beta,
                                &session.gdn_state[gdn_i],
                                &session.gdn_out,
                                n_v,
                                n_k,
                                head_dim,
                            )?;
                            enc.end();
                            cmd.commit();
                            crate::metal::wait_completed(&cmd)?;
                            gdn_tail_step_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            encode_rmsnorm_gated_f32(
                                self.ctx,
                                &enc,
                                &session.gdn_out,
                                &g.norm,
                                &gdn_z,
                                &gdn_normed,
                                n_v,
                                head_dim,
                                RMS_EPS * head_dim as f32,
                            )?;
                            enc.end();
                            cmd.commit();
                            crate::metal::wait_completed(&cmd)?;
                            gdn_tail_norm_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        } else {
                            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                            let enc = KernelEncoder::begin(&cmd);
                            self.encode_gdn_tail(
                                &enc,
                                g,
                                gdn_i,
                                session,
                                &gdn_qkv,
                                &gdn_z,
                                &gdn_alpha,
                                &gdn_beta,
                                &gdn_normed,
                            )?;
                            enc.end();
                            cmd.commit();
                            crate::metal::wait_completed(&cmd)?;
                            gdn_tail_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                        }
                    }
                    {
                        let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        encode_mat_vec_dispatch(
                            self.ctx,
                            &enc,
                            &g.out_proj,
                            &session.gdn_normed,
                            &session.mixer_out,
                            v_dim,
                            h,
                        )?;
                        enc.end();
                        cmd.commit();
                        crate::metal::wait_completed(&cmd)?;
                        gdn_out_proj_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }
                    {
                        let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        encode_add_inplace_f32(self.ctx, &enc, &session.x, &session.mixer_out)?;
                        encode_rms_norm_mul_f32(
                            self.ctx,
                            &enc,
                            &session.x,
                            &g.post_attn_norm,
                            &session.h,
                            RMS_EPS,
                        )?;
                        enc.end();
                        cmd.commit();
                        crate::metal::wait_completed(&cmd)?;
                        gdn_resid_post_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }
                }
                (MetalBlock::Attn(_), MixerSlot::Attn(_)) => {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_mixer_prep(&enc, block, slot, position, session)?;
                    enc.end();
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    attn_mixer_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                }
                _ => {
                    return Err(MfError::Metal(MetalError::BadShape {
                        kernel: "single_token_phase_profiled_moe",
                        detail: "mixer slot type did not match block kind".into(),
                    }));
                }
            }

            {
                if cpu_route {
                    let route = self.route_moe_block(&session.h, moe);
                    self.write_moe_route_result(session, &route);
                } else if route_replay {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_route_prepare(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                } else if split_route_deep {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_router_logits(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    route_logits_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_topk_parallel_from_logits(&enc, session)?;
                    enc.end();
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    route_topk_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_shared_gate(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    route_shared_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                } else if split_route {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_router_logits(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    route_logits_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_topk_and_shared_from_logits(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    route_topk_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                } else {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_route_prepare(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    route_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                }
            }

            if split_ffn_apply {
                if deep_split_ffn_apply
                    && moe_routed_gate_up_decode_supported(moe.gate_exps.dtype, moe.up_exps.dtype)
                    && matches!(
                        moe.down_exps.dtype,
                        GgmlType::Q5_K
                            | GgmlType::Q6_K
                            | GgmlType::Q8_0
                            | GgmlType::IQ4_NL
                            | GgmlType::IQ4_XS
                            | GgmlType::BF16
                    )
                {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_routed_gate_up_gpu(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    ffn_routed_gate_up_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    let routed_weighted_sum_is_pending =
                        self.encode_moe_routed_down_only_gpu(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    ffn_routed_down_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    let shared_inner_fused =
                        self.encode_moe_shared_ffn_gate_up_gpu(&enc, session, ffn_gate, ffn_up)?;
                    enc.end();
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    ffn_shared_gate_up_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    if !shared_inner_fused {
                        let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        self.encode_moe_shared_ffn_silu_gpu(&enc, session)?;
                        enc.end();
                        cmd.commit();
                        crate::metal::wait_completed(&cmd)?;
                        ffn_shared_silu_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_shared_ffn_down_gpu(&enc, session, ffn_down)?;
                    enc.end();
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    ffn_shared_down_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    self.encode_moe_ffn_final_wave_gpu(
                        &cmd,
                        session,
                        routed_weighted_sum_is_pending,
                        None,
                        None,
                    )?;
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    ffn_finalizer_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                } else if concurrent_shared_moe_decode_enabled()
                    && routed_gate_up_single_dispatch(moe.gate_exps.dtype, moe.up_exps.dtype)
                {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let shared_inner_fused = self.encode_moe_ffn_gate_up_wave_gpu(
                        &cmd, session, ffn_gate, ffn_up, moe, None, None,
                    )?;
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    ffn_gate_up_wave_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    if !shared_inner_fused {
                        let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                        let enc = KernelEncoder::begin(&cmd);
                        self.encode_moe_shared_ffn_silu_gpu(&enc, session)?;
                        enc.end();
                        cmd.commit();
                        crate::metal::wait_completed(&cmd)?;
                        ffn_shared_silu_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    }

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let routed_weighted_sum_is_pending = self
                        .encode_moe_ffn_down_wave_gpu(&cmd, session, ffn_down, moe, None, None)?;
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    ffn_down_wave_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    self.encode_moe_ffn_final_wave_gpu(
                        &cmd,
                        session,
                        routed_weighted_sum_is_pending,
                        None,
                        None,
                    )?;
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    ffn_finalizer_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                } else {
                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_routed_ffn_gpu(&enc, session, moe)?;
                    enc.end();
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    ffn_fallback_routed_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_shared_ffn_core_gpu(&enc, session, ffn_gate, ffn_up, ffn_down)?;
                    enc.end();
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    ffn_fallback_shared_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

                    let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_final_residual_gpu(&enc, session)?;
                    enc.end();
                    cmd.commit();
                    crate::metal::wait_completed(&cmd)?;
                    ffn_finalizer_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                }
            } else {
                let cmd = self.ctx.queue.commandBuffer().expect("cmd");
                if concurrent_shared_moe_decode_enabled() {
                    self.encode_moe_ffn_apply_gpu_concurrent_shared(
                        &cmd, session, ffn_gate, ffn_up, ffn_down, moe,
                    )?;
                } else {
                    let enc = KernelEncoder::begin(&cmd);
                    self.encode_moe_ffn_apply_gpu(&enc, session, ffn_gate, ffn_up, ffn_down, moe)?;
                    enc.end();
                }
                cmd.commit();
                crate::metal::wait_completed(&cmd)?;
                ffn_apply_total_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            }
        }
        phases.push((
            format!("gdn pre_norm (x{gdn_count})"),
            gdn_pre_norm_total_ms,
        ));
        if split_gdn_proj {
            phases.push((
                format!("gdn qkv proj (x{gdn_count})"),
                gdn_qkv_proj_total_ms,
            ));
            phases.push((format!("gdn z proj (x{gdn_count})"), gdn_z_proj_total_ms));
            phases.push((
                format!("gdn beta proj (x{gdn_count})"),
                gdn_beta_proj_total_ms,
            ));
            phases.push((
                format!("gdn alpha proj (x{gdn_count})"),
                gdn_alpha_proj_total_ms,
            ));
        } else {
            phases.push((
                format!("gdn front proj (x{gdn_count})"),
                gdn_front_proj_total_ms,
            ));
        }
        phases.push((
            format!("gdn alpha/beta (x{gdn_count})"),
            gdn_alpha_beta_total_ms,
        ));
        if split_gdn_tail {
            phases.push((
                format!("gdn tail conv (x{gdn_count})"),
                gdn_tail_conv_total_ms,
            ));
            phases.push((format!("gdn tail l2 (x{gdn_count})"), gdn_tail_l2_total_ms));
            phases.push((
                format!("gdn tail step (x{gdn_count})"),
                gdn_tail_step_total_ms,
            ));
            phases.push((
                format!("gdn tail norm (x{gdn_count})"),
                gdn_tail_norm_total_ms,
            ));
        } else {
            phases.push((format!("gdn tail (x{gdn_count})"), gdn_tail_total_ms));
        }
        phases.push((
            format!("gdn out_proj (x{gdn_count})"),
            gdn_out_proj_total_ms,
        ));
        phases.push((
            format!("gdn resid/post (x{gdn_count})"),
            gdn_resid_post_total_ms,
        ));
        phases.push((format!("attn mixer (x{attn_count})"), attn_mixer_total_ms));
        if route_replay {
            phases.push(("moe route replay (excluded)".into(), route_total_ms));
        } else if split_route {
            phases.push(("moe route logits".into(), route_logits_total_ms));
            if split_route_deep {
                phases.push(("moe route topk".into(), route_topk_total_ms));
                phases.push(("moe route shared gate".into(), route_shared_total_ms));
            } else {
                phases.push(("moe route topk/shared".into(), route_topk_total_ms));
            }
        } else {
            phases.push(("moe route".into(), route_total_ms));
        }
        if split_ffn_apply {
            if ffn_routed_gate_up_total_ms > 0.0 {
                phases.push(("moe ffn routed gate/up".into(), ffn_routed_gate_up_total_ms));
            }
            if ffn_routed_down_total_ms > 0.0 {
                phases.push(("moe ffn routed down".into(), ffn_routed_down_total_ms));
            }
            if ffn_shared_gate_up_total_ms > 0.0 {
                phases.push(("moe ffn shared gate/up".into(), ffn_shared_gate_up_total_ms));
            }
            if ffn_gate_up_wave_total_ms > 0.0 {
                phases.push(("moe ffn gate/up wave".into(), ffn_gate_up_wave_total_ms));
            }
            if ffn_shared_silu_total_ms > 0.0 {
                phases.push(("moe ffn shared silu".into(), ffn_shared_silu_total_ms));
            }
            if ffn_shared_down_total_ms > 0.0 {
                phases.push(("moe ffn shared down".into(), ffn_shared_down_total_ms));
            }
            if ffn_down_wave_total_ms > 0.0 {
                phases.push(("moe ffn down wave".into(), ffn_down_wave_total_ms));
            }
            if ffn_fallback_routed_total_ms > 0.0 {
                phases.push((
                    "moe ffn fallback routed".into(),
                    ffn_fallback_routed_total_ms,
                ));
            }
            if ffn_fallback_shared_total_ms > 0.0 {
                phases.push((
                    "moe ffn fallback shared".into(),
                    ffn_fallback_shared_total_ms,
                ));
            }
            phases.push(("moe ffn finalizer".into(), ffn_finalizer_total_ms));
        } else {
            phases.push(("moe ffn apply".into(), ffn_apply_total_ms));
        }

        {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_rms_norm_mul_f32(
                self.ctx,
                &enc,
                &session.x,
                &self.model.output_norm,
                &session.h,
                RMS_EPS,
            )?;
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd)?;
            phases.push((
                "final norm".into(),
                (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
            ));
        }
        {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_mat_vec_dispatch(
                self.ctx,
                &enc,
                &self.model.lm_head,
                &session.h,
                &session.logits,
                h,
                arch.vocab_size as usize,
            )?;
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd)?;
            phases.push((
                "lm head".into(),
                (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
            ));
        }

        if phase_lm_argmax_enabled() {
            let cmd = self.ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_argmax_f32(
                self.ctx,
                &enc,
                &session.logits,
                &session.argmax_tok,
                1,
                arch.vocab_size as usize,
            )?;
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd)?;
            phases.push((
                "lm argmax".into(),
                (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
            ));
        }

        let mut out = vec![0.0f32; arch.vocab_size as usize];
        unsafe {
            let src = session.logits.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
        Ok((out, total_ms, phases))
    }
}
