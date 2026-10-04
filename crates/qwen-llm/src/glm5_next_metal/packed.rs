//! Packed (multi-row) prefill for the GLM-5.3 session: one command buffer per
//! chunk of up to `rows` prompt tokens, every block encoded over all rows.
//!
//! Same graph as single-token decode, with batched encoders: mHC rows,
//! row-wise norms, mat-mat projections, packed KDA, causal packed latent
//! attention, row-wise indexer appends, and expert-major routed experts.
//! Latent absorption runs per row through the decode grouped GEMV (exact
//! decode lineage). In [`PackedLineage::Fast`] (default) projections and
//! grouped experts stage activations in half, as llama.cpp's batched prefill
//! does, so packed logits match serial decode closely but not bitwise.
//! [`PackedLineage::Exact`] runs every projection and expert per row through
//! the decode kernels and reproduces serial decode; it exists as the
//! equivalence reference, not as a fast path.

use super::*;
use crate::metal::{
    GroupedExperts, encode_grouped_routed_experts, encode_indexer_append_rows, encode_kda_prefill,
    encode_mat_vec_q8_0_batch_f32, encode_mhc4_collapse_rows, encode_mhc4_controls_rows,
    encode_mhc4_post_rows, encode_mhc4_repeat_rows, encode_rms_norm_mul_rows_f32,
    encode_route_learned_rows,
};

/// Arithmetic lineage of packed prefill.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PackedLineage {
    /// Batched mat-mat projections and expert-major grouped experts.
    #[default]
    Fast,
    /// Decode kernels per row: identical to serial decode, slower.
    Exact,
}

/// Activations for up to `rows` packed tokens, shared by every block.
pub(super) struct PackedScratch {
    rows: usize,
    pub(super) lineage: PackedLineage,
    token: MetalTensor,
    embedding: MetalTensor,
    residual: [MetalTensor; 2],
    normalized: MetalTensor,
    mixes: MetalTensor,
    pre: MetalTensor,
    post: MetalTensor,
    comb: MetalTensor,
    collapsed: MetalTensor,
    normed: MetalTensor,
    block_out: MetalTensor,
    q: MetalTensor,
    k: MetalTensor,
    v: MetalTensor,
    rank_a: MetalTensor,
    raw_gate: MetalTensor,
    raw_beta: MetalTensor,
    rank_b: MetalTensor,
    output_gate: MetalTensor,
    kda_out: MetalTensor,
    query_a: MetalTensor,
    query_r: MetalTensor,
    query: MetalTensor,
    latent_raw: MetalTensor,
    latent: MetalTensor,
    query_latent: MetalTensor,
    output_latent: MetalTensor,
    heads_out: MetalTensor,
    index_key: MetalTensor,
    index_gate: MetalTensor,
    dense_gate: MetalTensor,
    dense_up: MetalTensor,
    router: MetalTensor,
    counts: MetalTensor,
    slots: MetalTensor,
    inner: MetalTensor,
    slot_out: MetalTensor,
    routed: MetalTensor,
    shared_gate: MetalTensor,
    shared_up: MetalTensor,
    shared: MetalTensor,
    /// Per MoE block: ids and weights `[top_k, rows]`, status `[rows]`.
    routes: Vec<Option<RouteRecord>>,
}

impl PackedScratch {
    pub(super) fn new(ctx: &MetalContext, c: &Glm5NextConfig, rows: usize) -> Result<Self> {
        let r = rows as u64;
        let h = c.hidden_size as u64;
        let w = c.kda_width() as u64;
        let heads = c.head_count as u64;
        let d = c.kda_head_dim as u64;
        let kv = c.kv_lora_rank as u64;
        let k = c.expert_used_count as u64;
        let e = c.expert_count as u64;
        let f = |shape: &[u64]| zeros(ctx, shape);
        let routes = c
            .blocks
            .iter()
            .map(|block| match block.ffn {
                crate::glm5_next::FfnKind::Dense => Ok(None),
                crate::glm5_next::FfnKind::Moe => Ok(Some(RouteRecord {
                    ids: zeros_typed(ctx, GgmlType::I32, &[k, r], 4)?,
                    weights: zeros(ctx, &[k, r])?,
                    status: zeros_typed(ctx, GgmlType::I32, &[r], 4)?,
                })),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            rows,
            lineage: PackedLineage::Fast,
            token: zeros_typed(ctx, GgmlType::I32, &[r], 4)?,
            embedding: f(&[h, r])?,
            residual: [f(&[h, 4, r])?, f(&[h, 4, r])?],
            normalized: f(&[c.hc_width() as u64, r])?,
            mixes: f(&[c.hc_mix_count() as u64, r])?,
            pre: f(&[4, r])?,
            post: f(&[4, r])?,
            comb: f(&[4, 4, r])?,
            collapsed: f(&[h, r])?,
            normed: f(&[h, r])?,
            block_out: f(&[h, r])?,
            q: f(&[w, r])?,
            k: f(&[w, r])?,
            v: f(&[w, r])?,
            rank_a: f(&[d, r])?,
            raw_gate: f(&[w, r])?,
            raw_beta: f(&[heads, r])?,
            rank_b: f(&[d, r])?,
            output_gate: f(&[w, r])?,
            kda_out: f(&[w, r])?,
            query_a: f(&[c.q_lora_rank as u64, r])?,
            query_r: f(&[c.q_lora_rank as u64, r])?,
            query: f(&[c.mla_width() as u64, r])?,
            latent_raw: f(&[kv, r])?,
            latent: f(&[kv, r])?,
            query_latent: f(&[kv, heads, r])?,
            output_latent: f(&[kv, heads, r])?,
            heads_out: f(&[c.mla_width() as u64, r])?,
            index_key: f(&[c.indexer_head_dim as u64, r])?,
            index_gate: f(&[c.indexer_head_dim as u64, r])?,
            dense_gate: f(&[c.dense_ffn_size as u64, r])?,
            dense_up: f(&[c.dense_ffn_size as u64, r])?,
            router: f(&[e, r])?,
            counts: zeros_typed(ctx, GgmlType::I32, &[e], 4)?,
            slots: zeros_typed(ctx, GgmlType::I32, &[e * r], 4)?,
            inner: f(&[c.expert_ffn_size as u64, k * r])?,
            slot_out: f(&[h, k * r])?,
            routed: f(&[h, r])?,
            shared_gate: f(&[c.shared_expert_ffn_size as u64, r])?,
            shared_up: f(&[c.shared_expert_ffn_size as u64, r])?,
            shared: f(&[h, r])?,
            routes,
        })
    }
}

/// Prefix view with the leading dimensions of `t` and `rows` in the last.
fn rows_view(t: &MetalTensor, rows: usize) -> MetalTensor {
    let mut shape = t.shape.clone();
    let last = shape.len() - 1;
    shape[last] = rows as u64;
    t.view_subrange(0, shape)
}

/// One-dimensional prefix view of `n` elements.
fn flat(t: &MetalTensor, n: usize) -> MetalTensor {
    t.view_subrange(0, vec![n as u64])
}

#[allow(clippy::too_many_arguments)]
fn matmat(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    lineage: PackedLineage,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    rows: usize,
) -> Result<()> {
    if lineage == PackedLineage::Exact {
        for row in 0..rows {
            let xr = x.view_subrange((row * n_in) as u64, vec![n_in as u64]);
            let yr = y.view_subrange((row * n_out) as u64, vec![n_out as u64]);
            crate::metal_forward::encode_mat_vec_dispatch(ctx, enc, weight, &xr, &yr, n_in, n_out)?;
        }
        return Ok(());
    }
    crate::metal_forward::encode_mat_mat_dispatch_with_policy(
        ctx, enc, weight, x, y, n_in, n_out, rows, true,
    )?;
    Ok(())
}

impl Glm5NextSession<'_> {
    /// Packed prefill of `tokens` in chunks of the session's prefill rows;
    /// returns the last token's logits. Falls back to serial decode when the
    /// session has no packed scratch.
    pub fn prefill_packed(&mut self, ctx: &MetalContext, tokens: &[u32]) -> Result<Vec<f32>> {
        let Some(rows) = self.packed.as_ref().map(|p| p.rows) else {
            return self.prefill(ctx, tokens);
        };
        if tokens.is_empty() {
            return invalid("prefill requires at least one token");
        }
        let chunks: Vec<&[u32]> = tokens.chunks(rows).collect();
        let last = chunks.len() - 1;
        let mut logits = None;
        for (index, chunk) in chunks.into_iter().enumerate() {
            logits = self.step_packed(ctx, chunk, index == last)?;
        }
        logits.ok_or_else(|| Glm5NextMetalError::Invalid("missing prefill logits".into()))
    }

    fn step_packed(
        &mut self,
        ctx: &MetalContext,
        tokens: &[u32],
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        if self.poisoned {
            return invalid("session is poisoned by an earlier failed token");
        }
        let c = &self.weights.config;
        if let Some(&bad) = tokens.iter().find(|&&t| t >= c.vocab_size) {
            return invalid(format!("token {bad} outside vocabulary"));
        }
        if self.position + tokens.len() > self.capacity {
            return invalid(format!(
                "positions {}..{} exceed capacity {}",
                self.position,
                self.position + tokens.len(),
                self.capacity
            ));
        }
        self.poisoned = true;
        self.encode_chunk(ctx, tokens, want_logits)?;
        let packed = self.packed.as_ref().expect("packed scratch");
        for (layer, route) in packed.routes.iter().enumerate() {
            if let Some(route) = route {
                let status = read_i32(&rows_view(&route.status, tokens.len()))?;
                if let Some(bad) = status.iter().find(|&&s| s != ROUTE_STATUS_READY) {
                    return invalid(format!(
                        "block {layer} packed route failed with status {bad}"
                    ));
                }
            }
        }
        let logits = want_logits.then(|| read_f32(&self.s.logits)).transpose()?;
        self.position += tokens.len();
        self.poisoned = false;
        Ok(logits)
    }

    fn encode_chunk(&self, ctx: &MetalContext, tokens: &[u32], want_logits: bool) -> Result<()> {
        let w = self.weights;
        let c = &w.config;
        let p = self.packed.as_ref().expect("packed scratch");
        let rows = tokens.len();
        let h = c.hidden_size as usize;
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        // SAFETY: shared-storage I32 [max rows]; no command is in flight.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ids.as_ptr(),
                p.token
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(p.token.offset as usize)
                    .cast::<i32>(),
                rows,
            );
        }
        let v = |t: &MetalTensor| rows_view(t, rows);
        let command = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| Glm5NextMetalError::Invalid("no command buffer".into()))?;
        let enc = KernelEncoder::begin(&command);
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
        for (index, block) in w.blocks.iter().enumerate() {
            let (a, b) = (v(&p.residual[0]), v(&p.residual[1]));
            self.encode_hc_pre_rows(ctx, &enc, rows, &a, &block.attention_hc)?;
            encode_rms_norm_mul_rows_f32(
                ctx,
                &enc,
                &v(&p.collapsed),
                &block.attention_norm,
                &v(&p.normed),
                rows,
                h,
                c.rms_epsilon,
            )?;
            match (&block.mixer, &self.layers[index]) {
                (MixerTensors::Kda(kda), LayerState::Kda { conv, state }) => {
                    self.encode_kda_rows(ctx, &enc, rows, kda, conv, state)?
                }
                (
                    MixerTensors::Mla(mla),
                    LayerState::Mla {
                        latent,
                        pending,
                        pooled,
                    },
                ) => self.encode_mla_rows(ctx, &enc, rows, mla, latent, pending, pooled)?,
                _ => return invalid(format!("block {index} state does not match its mixer")),
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
            self.encode_hc_pre_rows(ctx, &enc, rows, &b, &block.ffn_hc)?;
            encode_rms_norm_mul_rows_f32(
                ctx,
                &enc,
                &v(&p.collapsed),
                &block.ffn_norm,
                &v(&p.normed),
                rows,
                h,
                c.rms_epsilon,
            )?;
            match &block.ffn {
                FfnTensors::Dense(dense) => {
                    let f = c.dense_ffn_size as usize;
                    matmat(
                        ctx,
                        &enc,
                        p.lineage,
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
                        p.lineage,
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
                        p.lineage,
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
                    matmat(
                        ctx,
                        &enc,
                        p.lineage,
                        &moe.router,
                        &v(&p.normed),
                        &v(&p.router),
                        h,
                        e,
                        rows,
                    )?;
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
                    if p.lineage == PackedLineage::Exact {
                        let s = &self.s;
                        for row in 0..rows {
                            let x = p.normed.view_subrange((row * h) as u64, vec![h as u64]);
                            let ids_r = route.ids.view_subrange((row * k) as u64, vec![k as u64]);
                            let w_r = route
                                .weights
                                .view_subrange((row * k) as u64, vec![k as u64]);
                            let st = route.status.view_subrange(row as u64, vec![1]);
                            let out = p.routed.view_subrange((row * h) as u64, vec![h as u64]);
                            crate::metal::encode_all_slots_gate_up_swiglu(
                                ctx,
                                &enc,
                                &moe.gate_experts,
                                &moe.up_experts,
                                &x,
                                &ids_r,
                                &st,
                                &s.expert_inner,
                                h,
                                f,
                                e,
                                k,
                                c.swiglu_clamp,
                            )?;
                            crate::metal::encode_all_slots_down(
                                ctx,
                                &enc,
                                &moe.down_experts,
                                &s.expert_inner,
                                &ids_r,
                                &st,
                                &s.expert_out,
                                f,
                                h,
                                e,
                                k,
                            )?;
                            encode_moe_weighted_sum_f32(
                                ctx,
                                &enc,
                                &s.expert_out,
                                &w_r,
                                &out,
                                h,
                                k,
                            )?;
                        }
                    } else {
                        encode_grouped_routed_experts(
                            ctx,
                            &enc,
                            &GroupedExperts {
                                gate_bank: &moe.gate_experts,
                                up_bank: &moe.up_experts,
                                down_bank: &moe.down_experts,
                                input: &v(&p.normed),
                                ids: &route_ids,
                                weights: &route_weights,
                                counts: &p.counts,
                                slots: &flat(&p.slots, e * rows),
                                inner: &p.inner.view_subrange(0, vec![f as u64, (k * rows) as u64]),
                                slot_out: &p
                                    .slot_out
                                    .view_subrange(0, vec![h as u64, (k * rows) as u64]),
                                output: &v(&p.routed),
                            },
                            h,
                            f,
                            e,
                            k,
                            rows,
                            c.swiglu_clamp,
                        )?;
                    }
                    let sf = c.shared_expert_ffn_size as usize;
                    matmat(
                        ctx,
                        &enc,
                        p.lineage,
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
                        p.lineage,
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
                        p.lineage,
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
        if want_logits {
            // Head on the last row only.
            let s = &self.s;
            let last = p.residual[0].view_subrange(((rows - 1) * h * 4) as u64, vec![h as u64, 4]);
            encode_mhc4_collapse(ctx, &enc, h, &last, &s.quarter, &s.final_hidden)?;
            encode_rms_norm_mul_f32(
                ctx,
                &enc,
                &s.final_hidden,
                &w.output_norm,
                &s.final_normed,
                c.rms_epsilon,
            )?;
            super::matvec(
                ctx,
                &enc,
                &w.output,
                &s.final_normed,
                &s.logits,
                h,
                c.vocab_size as usize,
            )?;
        }
        enc.end();
        command.commit();
        wait_completed(&command)?;
        Ok(())
    }

    fn encode_hc_pre_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        rows: usize,
        residual: &MetalTensor,
        hc: &crate::glm5_next::HyperConnectionTensors<MetalTensor>,
    ) -> Result<()> {
        let c = &self.weights.config;
        let p = self.packed.as_ref().expect("packed scratch");
        let v = |t: &MetalTensor| rows_view(t, rows);
        let (width, mixes) = (c.hc_width() as usize, c.hc_mix_count() as usize);
        encode_rms_norm_mul_rows_f32(
            ctx,
            enc,
            residual,
            &self.s.ones,
            &v(&p.normalized),
            rows,
            width,
            c.rms_epsilon,
        )?;
        // Exact decode lineage for the Sinkhorn inputs.
        encode_mat_vec_q8_0_batch_f32(
            ctx,
            enc,
            &hc.mix,
            &v(&p.normalized),
            &v(&p.mixes),
            width,
            mixes,
            rows,
        )?;
        encode_mhc4_controls_rows(
            ctx,
            enc,
            rows,
            c.hc_epsilon,
            &v(&p.mixes),
            &hc.scale,
            &hc.base,
            &v(&p.pre),
            &v(&p.post),
            &v(&p.comb),
        )?;
        encode_mhc4_collapse_rows(
            ctx,
            enc,
            c.hidden_size as usize,
            rows,
            residual,
            &v(&p.pre),
            &v(&p.collapsed),
        )?;
        Ok(())
    }

    fn encode_kda_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        rows: usize,
        kda: &crate::glm5_next::KdaTensors<MetalTensor>,
        conv: &MetalTensor,
        state: &MetalTensor,
    ) -> Result<()> {
        let c = &self.weights.config;
        let p = self.packed.as_ref().expect("packed scratch");
        let v = |t: &MetalTensor| rows_view(t, rows);
        let (h, width, rank) = (
            c.hidden_size as usize,
            c.kda_width() as usize,
            c.kda_head_dim as usize,
        );
        let x = v(&p.normed);
        matmat(
            ctx,
            enc,
            p.lineage,
            &kda.query,
            &x,
            &v(&p.q),
            h,
            width,
            rows,
        )?;
        matmat(ctx, enc, p.lineage, &kda.key, &x, &v(&p.k), h, width, rows)?;
        matmat(
            ctx,
            enc,
            p.lineage,
            &kda.value,
            &x,
            &v(&p.v),
            h,
            width,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.lineage,
            &kda.decay_a,
            &x,
            &v(&p.rank_a),
            h,
            rank,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.lineage,
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
            p.lineage,
            &kda.beta,
            &x,
            &v(&p.raw_beta),
            h,
            c.head_count as usize,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.lineage,
            &kda.gate_a,
            &x,
            &v(&p.rank_b),
            h,
            rank,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.lineage,
            &kda.gate_b,
            &v(&p.rank_b),
            &v(&p.output_gate),
            rank,
            width,
            rows,
        )?;
        encode_kda_prefill(
            ctx,
            enc,
            c.head_count as usize,
            rows,
            &KdaDecode {
                q: &v(&p.q),
                k: &v(&p.k),
                v: &v(&p.v),
                raw_gate: &v(&p.raw_gate),
                raw_beta: &v(&p.raw_beta),
                output_gate: &v(&p.output_gate),
                q_conv: &kda.query_conv,
                k_conv: &kda.key_conv,
                v_conv: &kda.value_conv,
                neg_exp_a_log: &kda.neg_exp_a_log,
                dt_bias: &kda.decay_bias,
                output_norm: &kda.output_norm,
                conv_state: conv,
                state,
                out: &v(&p.kda_out),
            },
            c.kda_gate_lower_bound,
            c.rms_epsilon,
        )?;
        matmat(
            ctx,
            enc,
            p.lineage,
            &kda.output,
            &v(&p.kda_out),
            &v(&p.block_out),
            width,
            h,
            rows,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_mla_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        rows: usize,
        mla: &crate::glm5_next::MlaTensors<MetalTensor>,
        latent: &MetalTensor,
        pending: &MetalTensor,
        pooled: &MetalTensor,
    ) -> Result<()> {
        let c = &self.weights.config;
        let p = self.packed.as_ref().expect("packed scratch");
        let v = |t: &MetalTensor| rows_view(t, rows);
        let h = c.hidden_size as usize;
        let (q_rank, kv) = (c.q_lora_rank as usize, c.kv_lora_rank as usize);
        let heads = c.head_count as usize;
        let head_dim = c.mla_key_head_dim as usize;
        let x = v(&p.normed);
        matmat(
            ctx,
            enc,
            p.lineage,
            &mla.query_a,
            &x,
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
            p.lineage,
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
            p.lineage,
            &mla.latent,
            &x,
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
        encode_scatter_offset_f32_to_f16(
            ctx,
            enc,
            &flat(&p.latent, kv * rows),
            latent,
            self.position * kv,
            kv * rows,
        )?;
        for row in 0..rows {
            let q_row = p.query.view_subrange(
                (row * heads * head_dim) as u64,
                vec![(heads * head_dim) as u64],
            );
            let ql_row = p
                .query_latent
                .view_subrange((row * heads * kv) as u64, vec![(heads * kv) as u64]);
            encode_mat_vec_q8_0_grouped_f32(
                ctx,
                enc,
                &mla.key_absorb,
                &q_row,
                &ql_row,
                head_dim,
                kv,
                heads,
            )?;
        }
        encode_latent_attention(
            ctx,
            enc,
            &v(&p.query_latent),
            latent,
            &self.s.no_sink,
            &v(&p.output_latent),
            self.position,
            rows,
            1.0 / (head_dim as f32).sqrt(),
        )?;
        for row in 0..rows {
            let ol_row = p
                .output_latent
                .view_subrange((row * heads * kv) as u64, vec![(heads * kv) as u64]);
            let ho_row = p.heads_out.view_subrange(
                (row * heads * head_dim) as u64,
                vec![(heads * head_dim) as u64],
            );
            encode_mat_vec_q8_0_grouped_f32(
                ctx,
                enc,
                &mla.value_expand,
                &ol_row,
                &ho_row,
                kv,
                head_dim,
                heads,
            )?;
        }
        matmat(
            ctx,
            enc,
            p.lineage,
            &mla.output,
            &v(&p.heads_out),
            &v(&p.block_out),
            heads * head_dim,
            h,
            rows,
        )?;
        let index_dim = c.indexer_head_dim as usize;
        matmat(
            ctx,
            enc,
            p.lineage,
            &mla.indexer.key,
            &x,
            &v(&p.index_key),
            h,
            index_dim,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.lineage,
            &mla.indexer.pool_gate,
            &x,
            &v(&p.index_gate),
            h,
            index_dim,
            rows,
        )?;
        encode_indexer_append_rows(
            ctx,
            enc,
            &v(&p.index_key),
            &v(&p.index_gate),
            &mla.indexer.key_norm,
            &mla.indexer.key_norm_bias,
            &mla.indexer.pool_position,
            pending,
            pooled,
            self.position,
            rows,
            c.layer_norm_epsilon,
        )?;
        Ok(())
    }
}
