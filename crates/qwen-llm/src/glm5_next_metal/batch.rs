//! A synchronous decode round over independent M5 Fast/Half sessions.
//!
//! Dense Q6/Q8 projections reuse dequantized weight fragments through the
//! F32-weight few-row MMA. Routed experts keep F32 operands and use the native
//! token grid. KDA recurrence, latent caches and sparse selection are always
//! per sequence. The first member lends its admitted packed activation
//! scratch for this call; no state or scratch escapes across requests.

use super::*;
use crate::metal::encode_copy_offset_f32;
use objc2_metal::MTLResource;

/// Largest qualified independent decode batch.
pub const GLM5_NEXT_MAX_BATCH_SIZE: usize = 4;

/// Admitted output storage for independent decode batches. It is bound to one
/// loaded weight instance, and can be reused as the active width changes.
///
/// M5 Fast/Half only for B > 1: few-row MMA keeps F32 weights/activations
/// and changes the reduction order. Exact and Fast/F32
/// members are refused before mutation and must use ordinary `forward`.
/// B = 1 always calls ordinary `forward`, including its mode contracts.
pub struct Glm5NextBatch {
    logits: MetalTensor,
    weights_instance: u64,
    max_members: usize,
}

fn projection_geometry(dtype: GgmlType, shape: &[u64]) -> bool {
    matches!(dtype, GgmlType::Q6_K | GgmlType::Q8_0)
        && shape.len() == 2
        && shape[0] > 0
        && shape[1] > 0
        && shape[0].is_multiple_of(if dtype == GgmlType::Q8_0 { 64 } else { 256 })
}
fn projection_supported(w: &MetalTensor) -> bool {
    projection_geometry(w.dtype, &w.shape)
}
fn batch_mode(lineage: Option<PackedLineage>, precision: Option<FastPrecision>) -> bool {
    lineage == Some(PackedLineage::Fast) && precision == Some(FastPrecision::Half)
}

impl Glm5NextBatch {
    /// Future CPU readback/result allowance to include in concurrent admissions.
    pub fn cpu_reserve_bytes(&self) -> u64 {
        self.logits.n_bytes() * 2
    }
    /// Metadata-only eligibility; does not allocate or compile a pipeline.
    pub fn supported(ctx: &MetalContext, weights: &Glm5NextWeights) -> bool {
        if !ctx.device_facts().gpu_families.apple10
            || !weights.packed_prefill_admitted
            || !projection_supported(&weights.output)
        {
            return false;
        }
        weights.blocks.iter().all(|b| {
            if b.attention_hc.mix.dtype != GgmlType::Q8_0 || b.ffn_hc.mix.dtype != GgmlType::Q8_0 {
                return false;
            }
            let mixer = match &b.mixer {
                MixerTensors::Kda(k) => [
                    &k.query, &k.key, &k.value, &k.decay_a, &k.decay_b, &k.beta, &k.gate_a,
                    &k.gate_b, &k.output,
                ]
                .into_iter()
                .all(projection_supported),
                MixerTensors::Mla(m) => [
                    &m.query_a,
                    &m.query_b,
                    &m.latent,
                    &m.indexer.key,
                    &m.indexer.pool_gate,
                    &m.output,
                ]
                .into_iter()
                .all(projection_supported),
            };
            mixer
                && match &b.ffn {
                    FfnTensors::Dense(d) => [&d.gate, &d.up, &d.down]
                        .into_iter()
                        .all(projection_supported),
                    FfnTensors::Moe(m) => {
                        m.router.dtype == GgmlType::F32
                            && [&m.shared.gate, &m.shared.up, &m.shared.down]
                                .into_iter()
                                .all(projection_supported)
                            && crate::metal::all_slots_gate_up_native_supported(
                                m.gate_experts.dtype,
                                m.up_experts.dtype,
                            )
                            && crate::metal::all_slots_down_native_supported(m.down_experts.dtype)
                    }
                }
        })
    }

    /// Admits and allocates only B-wide logits; activations borrow the first
    /// member's existing packed scratch. CPU allowance includes readback and
    /// the returned per-member vectors. Refusal leaves every session untouched.
    pub fn new(ctx: &MetalContext, weights: &Glm5NextWeights, max_members: usize) -> Result<Self> {
        if !(1..=GLM5_NEXT_MAX_BATCH_SIZE).contains(&max_members) || !Self::supported(ctx, weights)
        {
            return invalid("independent batching needs supported M5 weights and 1..=4 members");
        }
        let bytes = (weights.config.vocab_size as u64)
            .checked_mul(max_members as u64)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Glm5NextMetalError::Invalid("batch output size overflow".into()))?;
        let priced = device_price(ctx)(bytes)
            .ok_or_else(|| Glm5NextMetalError::Invalid("batch output price overflow".into()))?;
        let _allocation = ctx.begin_allocation_transaction();
        let admission = evaluate_metal_memory_admission_with_cpu_bytes(
            priced,
            bytes * 2,
            memory::DYNAMIC_RESERVE_BYTES,
            ctx.memory_signals(),
            true,
        );
        if let Some(denied) = admission.refusal() {
            return Err(Glm5NextMetalError::MemoryAdmission {
                denied,
                budget_bytes: admission_budget(&admission),
                advice: CapacityAdvice::NotEvaluated,
            });
        }
        Ok(Self {
            logits: MetalTensor::zeros_f32(
                ctx,
                vec![weights.config.vocab_size as u64, max_members as u64],
            )?,
            weights_instance: weights.instance,
            max_members,
        })
    }

    /// One token per member, in the supplied order. Members may have different
    /// positions. Invalid tokens, capacity, weights, modes and workspace are
    /// checked for the whole group before mutation. After encoding begins, an
    /// error or unwind leaves all members poisoned; there is no partial retry.
    /// Callers observe per-request cancellation outside this synchronous call.
    pub fn step(
        &mut self,
        ctx: &MetalContext,
        sessions: &mut [&mut Glm5NextSession<'_>],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        let count = sessions.len();
        if count == 0 || count != tokens.len() || count > self.max_members {
            return invalid("invalid independent batch width");
        }
        for (s, t) in sessions.iter().zip(tokens) {
            if s.weights.instance != self.weights_instance {
                return invalid("independent batch weights differ");
            }
            if s.poisoned {
                return Err(Glm5NextMetalError::Poisoned);
            }
            if *t >= s.weights.config.vocab_size || s.position >= s.capacity {
                return invalid("invalid independent batch member token/capacity");
            }
        }
        if count == 1 {
            return Ok(vec![sessions[0].forward(ctx, tokens[0])?]);
        }
        if sessions
            .iter()
            .any(|s| !batch_mode(s.packed_lineage(), s.fast_precision()))
        {
            return invalid("independent batch needs Fast/Half members");
        }
        if sessions[0].packed.as_ref().is_none_or(|p| p.rows < count) {
            return invalid("independent batch activation workspace too small");
        }
        if self.logits.buffer.device().registryID() != ctx.device.registryID() {
            return invalid("independent batch device differs");
        }
        for s in sessions.iter_mut() {
            s.poisoned = true;
        }
        for s in sessions.iter() {
            let c = &s.weights.config;
            let visible = s.position + 1;
            if visible >= c.sparse_frontier() as usize {
                let sp = s.sparse.as_ref().ok_or_else(|| {
                    Glm5NextMetalError::Invalid("independent sparse member has no scratch".into())
                })?;
                write_i32(
                    &sp.visible_pools,
                    &[(visible / c.indexer_pool as usize) as i32],
                )?;
                write_i32(&sp.visible_rows, &[visible as i32])?;
                write_i32(&sp.select_status, &vec![-1; c.block_count(MixerKind::Mla)])?;
            }
        }
        let refs: Vec<_> = sessions.iter().map(|s| &**s).collect();
        refs[0].encode_independent_batch(ctx, &refs, tokens, &self.logits)?;
        let p = refs[0].packed.as_ref().expect("preflighted");
        for (block, r) in p.routes.iter().enumerate() {
            if let Some(r) = r {
                let statuses = read_i32(&rows_view(&r.status, count))?;
                if let Some((row, &status)) = statuses
                    .iter()
                    .enumerate()
                    .find(|(_, s)| **s != ROUTE_STATUS_READY)
                {
                    return Err(Glm5NextMetalError::KernelValidation {
                        stage: "independent route",
                        block,
                        row: Some(row),
                        status,
                    });
                }
            }
        }
        for (row, s) in refs.iter().enumerate() {
            if s.position + 1 >= s.weights.config.sparse_frontier() as usize {
                let statuses = read_i32(&s.sparse.as_ref().expect("preflighted").select_status)?;
                if let Some((index, &status)) = statuses
                    .iter()
                    .enumerate()
                    .find(|(_, s)| **s != crate::metal::SELECT_STATUS_OK)
                {
                    return Err(Glm5NextMetalError::KernelValidation {
                        stage: "independent sparse selection",
                        block: mla_block(&s.weights.config, index),
                        row: Some(row),
                        status,
                    });
                }
            }
        }
        let vocab = refs[0].weights.config.vocab_size as usize;
        let result = (0..count)
            .map(|r| read_f32(&row(&self.logits, r, vocab)))
            .collect::<Result<Vec<_>>>()?;
        if result.iter().flatten().any(|x| !x.is_finite()) {
            return invalid("non-finite independent batch logits");
        }
        for s in sessions.iter_mut() {
            s.position += 1;
            s.poisoned = false;
        }
        Ok(result)
    }
}

fn matmat(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    arith: Arith,
    stage: Stage,
    w: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    rows: usize,
) -> Result<()> {
    if matches!(w.dtype, GgmlType::Q6_K | GgmlType::Q8_0)
        && k.is_multiple_of(if w.dtype == GgmlType::Q8_0 { 64 } else { 256 })
    {
        crate::metal::encode_mat_mat_fewrow_gen_f32_weights(ctx, enc, w, x, y, k, m, rows)?;
        return Ok(());
    }
    super::matmat(ctx, enc, arith, stage, w, x, y, k, m, rows)
}
fn row(t: &MetalTensor, r: usize, n: usize) -> MetalTensor {
    t.view_subrange((r * n) as u64, vec![n as u64])
}
fn copy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    dst: &MetalTensor,
    n: usize,
) -> Result<()> {
    encode_copy_offset_f32(ctx, enc, src, 0, dst, n)?;
    Ok(())
}
impl Glm5NextSession<'_> {
    fn encode_independent_batch(
        &self,
        ctx: &MetalContext,
        seqs: &[&Self],
        tokens: &[u32],
        logits: &MetalTensor,
    ) -> Result<()> {
        let w = self.weights;
        let c = &w.config;
        let p = self.packed.as_ref().unwrap();
        let rows = tokens.len();
        let h = c.hidden_size as usize;
        assert!(rows <= p.rows);
        write_i32(
            &rows_view(&p.token, rows),
            &tokens.iter().map(|t| *t as i32).collect::<Vec<_>>(),
        )?;
        let v = |t: &MetalTensor| rows_view(t, rows);
        let command = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| Glm5NextMetalError::Invalid("no command buffer".into()))?;
        let mut enc = KernelEncoder::begin(&command);
        encode_get_rows_f32(
            ctx,
            &enc,
            &w.embedding,
            &v(&p.token),
            &v(&p.embedding),
            rows,
            h,
        )?;
        encode_mhc4_repeat_rows(ctx, &enc, h, rows, &v(&p.embedding), &v(&p.residual[0]))?;
        let mut mla_index = 0;
        for (index, block) in w.blocks.iter().enumerate() {
            let (a, b) = (v(&p.residual[0]), v(&p.residual[1]));
            self.encode_hc_pre_rows(
                ctx,
                &enc,
                rows,
                &a,
                &block.attention_hc,
                &block.attention_norm,
            )?;
            match &block.mixer {
                MixerTensors::Kda(kda) => self.encode_batch_kda(ctx, &enc, seqs, index, kda)?,
                MixerTensors::Mla(mla) => {
                    self.encode_batch_mla(ctx, &mut enc, seqs, index, mla_index, mla)?;
                    mla_index += 1;
                }
            }
            encode_mhc4_post_rows(
                ctx,
                &enc,
                h,
                rows,
                &v(&p.block_out),
                &a,
                &v(&p.post),
                &v(&p.comb),
                &b,
            )?;
            self.encode_hc_pre_rows(ctx, &enc, rows, &b, &block.ffn_hc, &block.ffn_norm)?;
            match &block.ffn {
                FfnTensors::Dense(dense) => {
                    self.stage(&mut enc, Glm5NextStage::DenseFfn, Some(index))?;
                    let f = c.dense_ffn_size as usize;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::DenseFfn,
                        &dense.gate,
                        &v(&p.normed),
                        &v(&p.dense_gate),
                        h,
                        f,
                        rows,
                    )?;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::DenseFfn,
                        &dense.up,
                        &v(&p.normed),
                        &v(&p.dense_up),
                        h,
                        f,
                        rows,
                    )?;
                    let (g, u) = (flat(&p.dense_gate, f * rows), flat(&p.dense_up, f * rows));
                    encode_clamped_swiglu(ctx, &enc, &g, &u, &g, c.swiglu_clamp)?;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::DenseFfn,
                        &dense.down,
                        &v(&p.dense_gate),
                        &v(&p.block_out),
                        f,
                        h,
                        rows,
                    )?;
                }
                FfnTensors::Moe(moe) => {
                    let route = p.routes[index].as_ref().ok_or_else(|| {
                        Glm5NextMetalError::Invalid(format!(
                            "block {index} has no packed route record"
                        ))
                    })?;
                    let (e, f, k) = (
                        c.expert_count as usize,
                        c.expert_ffn_size as usize,
                        c.expert_used_count as usize,
                    );
                    self.stage(&mut enc, Glm5NextStage::Router, Some(index))?;
                    if router_e8p32_selected(
                        stage_lineage(p.lineage, Stage::Router),
                        h,
                        e,
                        moe.router.dtype,
                        rows,
                        || router_e8p32_supported(ctx),
                    ) {
                        crate::metal::encode_mat_mat_f32_router_e8p32_strict(
                            ctx,
                            &enc,
                            &moe.router,
                            &v(&p.normed),
                            &v(&p.router),
                            h,
                            e,
                            rows,
                        )?;
                        #[cfg(test)]
                        router_prefill::record_substitution();
                    } else {
                        matmat(
                            ctx,
                            &enc,
                            p.arith(),
                            Stage::Router,
                            &moe.router,
                            &v(&p.normed),
                            &v(&p.router),
                            h,
                            e,
                            rows,
                        )?;
                    }
                    let spec = LearnedRoute {
                        experts: e,
                        top_k: k,
                        score: RouteScore::Sigmoid,
                        routed_scale: c.expert_weights_scale,
                    };
                    let (route_ids, route_weights) = (v(&route.ids), v(&route.weights));
                    encode_route_learned_rows(
                        ctx,
                        &enc,
                        &spec,
                        rows,
                        &v(&p.router),
                        &moe.selection_bias,
                        &route_ids,
                        &route_weights,
                        &v(&route.status),
                    )?;
                    self.stage(&mut enc, Glm5NextStage::RoutedExperts, Some(index))?;
                    let grouped = GroupedExperts {
                        gate_bank: &moe.gate_experts,
                        up_bank: &moe.up_experts,
                        down_bank: &moe.down_experts,
                        input: &v(&p.normed),
                        ids: &route_ids,
                        weights: &route_weights,
                        status: &v(&route.status),
                        counts: &p.counts,
                        slots: &flat(&p.slots, e * rows),
                        inner: &p.inner.view_subrange(0, vec![f as u64, (k * rows) as u64]),
                        slot_out: &p
                            .slot_out
                            .view_subrange(0, vec![h as u64, (k * rows) as u64]),
                        output: &v(&p.routed),
                    };
                    crate::metal::encode_all_slots_gate_up_swiglu_native_fused_tokens(
                        ctx,
                        &enc,
                        grouped.gate_bank,
                        grouped.up_bank,
                        grouped.input,
                        grouped.ids,
                        grouped.status,
                        grouped.inner,
                        (h, f, e, k),
                        rows,
                        c.swiglu_clamp,
                    )?;
                    crate::metal::encode_all_slots_down_native_tokens(
                        ctx,
                        &enc,
                        grouped.down_bank,
                        grouped.inner,
                        grouped.ids,
                        grouped.status,
                        grouped.slot_out,
                        (f, h, e, k),
                        rows,
                    )?;
                    crate::metal::encode_moe_weighted_sum_packed_f32(
                        ctx,
                        &enc,
                        grouped.slot_out,
                        grouped.weights,
                        grouped.output,
                        h,
                        k,
                        rows,
                    )?;
                    self.stage(&mut enc, Glm5NextStage::SharedExpert, Some(index))?;
                    let sf = c.shared_expert_ffn_size as usize;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::SharedExpert,
                        &moe.shared.gate,
                        &v(&p.normed),
                        &v(&p.shared_gate),
                        h,
                        sf,
                        rows,
                    )?;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::SharedExpert,
                        &moe.shared.up,
                        &v(&p.normed),
                        &v(&p.shared_up),
                        h,
                        sf,
                        rows,
                    )?;
                    let (g, u) = (
                        flat(&p.shared_gate, sf * rows),
                        flat(&p.shared_up, sf * rows),
                    );
                    encode_clamped_swiglu(ctx, &enc, &g, &u, &g, c.swiglu_clamp)?;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::SharedExpert,
                        &moe.shared.down,
                        &v(&p.shared_gate),
                        &v(&p.shared),
                        sf,
                        h,
                        rows,
                    )?;
                    encode_add_f32(
                        ctx,
                        &enc,
                        &flat(&p.routed, h * rows),
                        &flat(&p.shared, h * rows),
                        &flat(&p.block_out, h * rows),
                    )?;
                }
            }
            self.stage(&mut enc, Glm5NextStage::FfnPost, Some(index))?;
            encode_mhc4_post_rows(
                ctx,
                &enc,
                h,
                rows,
                &v(&p.block_out),
                &b,
                &v(&p.post),
                &v(&p.comb),
                &a,
            )?;
        }
        for (r, seq) in seqs.iter().enumerate() {
            let s = &seq.s;
            encode_mhc4_collapse(
                ctx,
                &enc,
                h,
                &p.residual[0].view_subrange((r * h * 4) as u64, vec![h as u64, 4]),
                &s.quarter,
                &s.final_hidden,
            )?;
            encode_rms_norm_mul_f32(
                ctx,
                &enc,
                &s.final_hidden,
                &w.output_norm,
                &s.final_normed,
                c.rms_epsilon,
            )?;
            copy(ctx, &enc, &s.final_normed, &row(&p.normed, r, h), h)?;
        }
        matmat(
            ctx,
            &enc,
            p.arith(),
            Stage::MlaProjection,
            &w.output,
            &v(&p.normed),
            &logits.view_subrange(0, vec![c.vocab_size as u64, rows as u64]),
            h,
            c.vocab_size as usize,
            rows,
        )?;
        enc.end();
        command.commit();
        wait_completed(&command)?;
        Ok(())
    }
    fn encode_batch_kda(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        seqs: &[&Self],
        index: usize,
        kda: &crate::glm5_next::KdaTensors<MetalTensor>,
    ) -> Result<()> {
        let c = &self.weights.config;
        let p = self.packed.as_ref().unwrap();
        let rows = seqs.len();
        let v = |t: &MetalTensor| rows_view(t, rows);
        let (h, width, rank) = (
            c.hidden_size as usize,
            c.kda_width() as usize,
            c.kda_head_dim as usize,
        );
        for (w, y, m) in [
            (&kda.query, &p.q, width),
            (&kda.key, &p.k, width),
            (&kda.value, &p.v, width),
            (&kda.decay_a, &p.rank_a, rank),
            (&kda.beta, &p.raw_beta, c.head_count as usize),
            (&kda.gate_a, &p.rank_b, rank),
        ] {
            matmat(
                ctx,
                enc,
                p.arith(),
                Stage::KdaProjection,
                w,
                &v(&p.normed),
                &v(y),
                h,
                m,
                rows,
            )?;
        }
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::KdaExpand,
            &kda.decay_b,
            &v(&p.rank_a),
            &v(&p.raw_gate),
            rank,
            width,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::KdaExpand,
            &kda.gate_b,
            &v(&p.rank_b),
            &v(&p.output_gate),
            rank,
            width,
            rows,
        )?;
        for (r, seq) in seqs.iter().enumerate() {
            let LayerState::Kda { conv, state } = &seq.layers[index] else {
                return invalid("not KDA");
            };
            encode_kda_decode(
                ctx,
                enc,
                c.head_count as usize,
                &KdaDecode {
                    q: &row(&p.q, r, width),
                    k: &row(&p.k, r, width),
                    v: &row(&p.v, r, width),
                    raw_gate: &row(&p.raw_gate, r, width),
                    raw_beta: &row(&p.raw_beta, r, c.head_count as usize),
                    output_gate: &row(&p.output_gate, r, width),
                    q_conv: &kda.query_conv,
                    k_conv: &kda.key_conv,
                    v_conv: &kda.value_conv,
                    neg_exp_a_log: &kda.neg_exp_a_log,
                    dt_bias: &kda.decay_bias,
                    output_norm: &kda.output_norm,
                    conv_state: conv,
                    state,
                    out: &row(&p.kda_out, r, width),
                },
                c.kda_gate_lower_bound,
                c.rms_epsilon,
            )?;
        }
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::KdaProjection,
            &kda.output,
            &v(&p.kda_out),
            &v(&p.block_out),
            width,
            h,
            rows,
        )
    }
    fn encode_batch_mla(
        &self,
        ctx: &MetalContext,
        enc: &mut KernelEncoder,
        seqs: &[&Self],
        block: usize,
        mla_index: usize,
        mla: &crate::glm5_next::MlaTensors<MetalTensor>,
    ) -> Result<()> {
        let c = &self.weights.config;
        let p = self.packed.as_ref().unwrap();
        let rows = seqs.len();
        let v = |t: &MetalTensor| rows_view(t, rows);
        let h = c.hidden_size as usize;
        let (q_rank, kv) = (c.q_lora_rank as usize, c.kv_lora_rank as usize);
        let heads = c.head_count as usize;
        let head_dim = c.mla_key_head_dim as usize;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::MlaProjection,
            &mla.query_a,
            &v(&p.normed),
            &v(&p.query_a),
            h,
            q_rank,
            rows,
        )?;
        encode_rms_norm_mul_rows_f32(
            ctx,
            enc,
            &v(&p.query_a),
            &mla.query_a_norm,
            &v(&p.query_r),
            rows,
            q_rank,
            c.rms_epsilon,
        )?;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::MlaProjection,
            &mla.query_b,
            &v(&p.query_r),
            &v(&p.query),
            q_rank,
            heads * head_dim,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::MlaProjection,
            &mla.latent,
            &v(&p.normed),
            &v(&p.latent_raw),
            h,
            kv,
            rows,
        )?;
        encode_rms_norm_mul_rows_f32(
            ctx,
            enc,
            &v(&p.latent_raw),
            &mla.latent_norm,
            &v(&p.latent),
            rows,
            kv,
            c.rms_epsilon,
        )?;
        for (w, y) in [
            (&mla.indexer.key, &p.index_key),
            (&mla.indexer.pool_gate, &p.index_gate),
        ] {
            matmat(
                ctx,
                enc,
                p.arith(),
                Stage::MlaProjection,
                w,
                &v(&p.normed),
                &v(y),
                h,
                c.indexer_head_dim as usize,
                rows,
            )?;
        }
        for (r, seq) in seqs.iter().enumerate() {
            let s = &seq.s;
            let LayerState::Mla {
                latent,
                pending,
                pooled,
            } = &seq.layers[block]
            else {
                return invalid("not MLA");
            };
            for (src, dst, n) in [
                (&p.query, &s.query, heads * head_dim),
                (&p.query_r, &s.query_r, q_rank),
                (&p.normed, &s.normed, h),
                (&p.latent, &s.latent, kv),
                (&p.index_key, &s.index_key, c.indexer_head_dim as usize),
                (&p.index_gate, &s.index_gate, c.indexer_head_dim as usize),
            ] {
                copy(ctx, enc, &row(src, r, n), dst, n)?;
            }
            encode_scatter_offset_f32_to_f16(ctx, enc, &s.latent, latent, seq.position * kv, kv)?;
            grouped_matvec(
                ctx,
                enc,
                &mla.key_absorb,
                &s.query,
                &s.query_latent,
                head_dim,
                kv,
                heads,
            )?;
            // Indexer cache maintenance from token zero. The pool this token
            // completes is published before attention: a query at visible length
            // L scores the first L / 4 pools, including its own.
            let index_dim = c.indexer_head_dim as usize;
            seq.stage(enc, Glm5NextStage::MlaIndexer, Some(block))?;
            matvec(
                ctx,
                enc,
                &mla.indexer.key,
                &s.normed,
                &s.index_key,
                h,
                index_dim,
            )?;
            matvec(
                ctx,
                enc,
                &mla.indexer.pool_gate,
                &s.normed,
                &s.index_gate,
                h,
                index_dim,
            )?;
            encode_indexer_append(
                ctx,
                enc,
                &s.index_key,
                &s.index_gate,
                &mla.indexer.key_norm,
                &mla.indexer.key_norm_bias,
                &mla.indexer.pool_position,
                pending,
                pooled,
                seq.position,
                c.layer_norm_epsilon,
            )?;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let visible = seq.position + 1;
            if visible >= c.sparse_frontier() as usize {
                seq.encode_sparse_attention(
                    ctx, enc, mla, latent, pooled, mla_index, visible, scale, block,
                )?;
            } else {
                seq.stage(enc, Glm5NextStage::DenseAttention, Some(block))?;
                encode_latent_attention(
                    ctx,
                    enc,
                    &s.query_latent,
                    latent,
                    &s.no_sink,
                    &s.output_latent,
                    seq.position,
                    1,
                    scale,
                )?;
            }
            seq.stage(enc, Glm5NextStage::MlaOutput, Some(block))?;
            grouped_matvec(
                ctx,
                enc,
                &mla.value_expand,
                &s.output_latent,
                &s.heads_out,
                kv,
                head_dim,
                heads,
            )?;

            copy(
                ctx,
                enc,
                &s.heads_out,
                &row(&p.heads_out, r, heads * head_dim),
                heads * head_dim,
            )?;
        }
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::MlaProjection,
            &mla.output,
            &v(&p.heads_out),
            &v(&p.block_out),
            heads * head_dim,
            h,
            rows,
        )
    }
}

#[cfg(test)]
#[path = "tests/batch_decode.rs"]
mod tests;
