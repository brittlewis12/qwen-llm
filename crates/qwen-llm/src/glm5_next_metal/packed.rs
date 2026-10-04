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
    /// Present when the session reaches the sparse frontier.
    sparse: Option<PackedSparseScratch>,
}

/// Packed sparse-selection scratch: chunk-wide indexer queries, weights and
/// per-row visibility; microbatch scores, pools and expanded rows (reused
/// microbatch after microbatch and block after block on the serial
/// encoder); one sticky status per (MLA block, chunk row).
pub(super) struct PackedSparseScratch {
    index_query: MetalTensor,
    index_query_f16: MetalTensor,
    index_weights: MetalTensor,
    visible_pools: MetalTensor,
    visible_rows: MetalTensor,
    scores: MetalTensor,
    pool_ids: MetalTensor,
    pool_counts: MetalTensor,
    row_ids: MetalTensor,
    row_counts: MetalTensor,
    select_status: MetalTensor,
    rows: usize,
    microbatch: usize,
}

impl PackedSparseScratch {
    fn new(ctx: &MetalContext, c: &Glm5NextConfig, capacity: u64, rows: u64) -> Result<Self> {
        let specs = memory::packed_sparse_specs(c, capacity, rows);
        let mut b = SpecBuffers::allocate(ctx, "packed_sparse", &specs)?;
        let s = Self {
            index_query: b.take("index_query")?,
            index_query_f16: b.take("index_query_f16")?,
            index_weights: b.take("index_weights")?,
            visible_pools: b.take("visible_pools")?,
            visible_rows: b.take("visible_rows")?,
            scores: b.take("scores")?,
            pool_ids: b.take("pool_ids")?,
            pool_counts: b.take("pool_counts")?,
            row_ids: b.take("row_ids")?,
            row_counts: b.take("row_counts")?,
            select_status: b.take("select_status")?,
            rows: rows as usize,
            microbatch: rows.min(memory::PACKED_SPARSE_QUERIES) as usize,
        };
        b.finish()?;
        Ok(s)
    }
}

impl PackedScratch {
    pub(super) fn new(
        ctx: &MetalContext,
        c: &Glm5NextConfig,
        rows: usize,
        capacity: usize,
    ) -> Result<Self> {
        let r = rows as u64;
        let specs = memory::packed_scratch_specs(c, r);
        let mut b = SpecBuffers::allocate(ctx, "packed_scratch", &specs)?;
        let scratch = Self {
            rows,
            lineage: PackedLineage::Fast,
            token: b.take("token")?,
            embedding: b.take("embedding")?,
            residual: [b.take("residual_a")?, b.take("residual_b")?],
            normalized: b.take("normalized")?,
            mixes: b.take("mixes")?,
            pre: b.take("pre")?,
            post: b.take("post")?,
            comb: b.take("comb")?,
            collapsed: b.take("collapsed")?,
            normed: b.take("normed")?,
            block_out: b.take("block_out")?,
            q: b.take("q")?,
            k: b.take("k")?,
            v: b.take("v")?,
            rank_a: b.take("rank_a")?,
            raw_gate: b.take("raw_gate")?,
            raw_beta: b.take("raw_beta")?,
            rank_b: b.take("rank_b")?,
            output_gate: b.take("output_gate")?,
            kda_out: b.take("kda_out")?,
            query_a: b.take("query_a")?,
            query_r: b.take("query_r")?,
            query: b.take("query")?,
            latent_raw: b.take("latent_raw")?,
            latent: b.take("latent")?,
            query_latent: b.take("query_latent")?,
            output_latent: b.take("output_latent")?,
            heads_out: b.take("heads_out")?,
            index_key: b.take("index_key")?,
            index_gate: b.take("index_gate")?,
            dense_gate: b.take("dense_gate")?,
            dense_up: b.take("dense_up")?,
            router: b.take("router")?,
            counts: b.take("counts")?,
            slots: b.take("slots")?,
            inner: b.take("inner")?,
            slot_out: b.take("slot_out")?,
            routed: b.take("routed")?,
            shared_gate: b.take("shared_gate")?,
            shared_up: b.take("shared_up")?,
            shared: b.take("shared")?,
            routes: RouteRecord::for_blocks(ctx, c, Some(r))?,
            sparse: (capacity >= c.sparse_frontier() as usize)
                .then(|| PackedSparseScratch::new(ctx, c, capacity as u64, r))
                .transpose()?,
        };
        b.finish()?;
        Ok(scratch)
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

/// Per-head latent absorption or expansion over `rows` token-major rows:
/// grouped Q8_0 mat-mat (F32 accumulation) over whole 128-row blocks in fast
/// lineage, and the decode grouped GEMV per remaining row (every row in exact
/// lineage).
#[allow(clippy::too_many_arguments)]
fn absorb_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    lineage: PackedLineage,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    groups: usize,
    rows: usize,
) -> Result<()> {
    let (in_width, out_width) = (n_in * groups, n_out * groups);
    let blocked = match lineage {
        PackedLineage::Fast => rows / 128 * 128,
        PackedLineage::Exact => 0,
    };
    if blocked > 0 {
        crate::metal::encode_mat_mat_q8_0_grouped_f32(
            ctx,
            enc,
            weight,
            &input.view_subrange(0, vec![in_width as u64, blocked as u64]),
            &output.view_subrange(0, vec![out_width as u64, blocked as u64]),
            n_in,
            n_out,
            groups,
            blocked,
        )?;
    }
    for row in blocked..rows {
        let x = input.view_subrange((row * in_width) as u64, vec![in_width as u64]);
        let y = output.view_subrange((row * out_width) as u64, vec![out_width as u64]);
        encode_mat_vec_q8_0_grouped_f32(ctx, enc, weight, &x, &y, n_in, n_out, groups)?;
    }
    Ok(())
}

impl Glm5NextSession<'_> {
    /// Packed prefill of `tokens` in chunks of the session's prefill rows;
    /// returns the last token's logits. Falls back to serial decode when the
    /// session has no packed scratch. Rows below the sparse frontier attend
    /// densely; rows at or past it run sparse selection in microbatches.
    pub fn prefill_packed(&mut self, ctx: &MetalContext, tokens: &[u32]) -> Result<Vec<f32>> {
        let Some(rows) = self.packed.as_ref().map(|p| p.rows) else {
            return self.prefill(ctx, tokens);
        };
        self.validate_request(tokens)?;
        let chunks: Vec<&[u32]> = tokens.chunks(rows).collect();
        let last = chunks.len() - 1;
        let mut logits = None;
        for (index, chunk) in chunks.into_iter().enumerate() {
            logits = self.step_packed(ctx, chunk, index == last)?;
        }
        logits.ok_or_else(|| Glm5NextMetalError::Invalid("missing prefill logits".into()))
    }

    /// Packed sparse selector statuses `[MLA block][chunk row]` (tests).
    #[cfg(test)]
    pub(super) fn packed_sparse_statuses(&self) -> Result<(Vec<i32>, usize)> {
        let sp = self
            .packed
            .as_ref()
            .and_then(|p| p.sparse.as_ref())
            .ok_or_else(|| Glm5NextMetalError::Invalid("no packed sparse scratch".into()))?;
        Ok((read_i32(&sp.select_status)?, sp.rows))
    }

    /// Positions left in the dense range (visible length below the sparse
    /// frontier) from the current position.
    fn dense_rows_remaining(&self) -> usize {
        (self.weights.config.sparse_frontier() as usize - 1).saturating_sub(self.position)
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
        // Rows from `dense` on attend over the indexer's selection.
        let dense = self.dense_rows_remaining().min(tokens.len());
        let pool = c.indexer_pool as usize;
        let mla = c.block_count(MixerKind::Mla);
        if dense < tokens.len() {
            let packed = self.packed.as_ref().expect("packed scratch");
            let Some(sp) = packed.sparse.as_ref() else {
                return invalid("packed rows past the sparse frontier need packed sparse scratch");
            };
            let visible: Vec<i32> = (0..sp.rows)
                .map(|r| (self.position + r + 1).min(self.capacity) as i32)
                .collect();
            #[allow(unused_mut)]
            let mut pools: Vec<i32> = visible.iter().map(|&l| l / pool as i32).collect();
            #[cfg(test)]
            if let Some(row) = self.corrupt_sparse_row.take() {
                pools[row] = 0;
            }
            write_i32(&sp.visible_rows, &visible)?;
            write_i32(&sp.visible_pools, &pools)?;
            // Unwritten slots fail the check below.
            write_i32(&sp.select_status, &vec![-1; mla * sp.rows])?;
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
        if dense < tokens.len() {
            let sp = packed.sparse.as_ref().expect("checked before encoding");
            let status = read_i32(&sp.select_status)?;
            for block in 0..mla {
                for row in dense..tokens.len() {
                    let code = status[block * sp.rows + row];
                    if code != crate::metal::SELECT_STATUS_OK {
                        return invalid(format!(
                            "MLA block {block} row {row} sparse selection failed with status {code}"
                        ));
                    }
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
        let mut mla_index = 0;
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
                ) => {
                    self.encode_mla_rows(ctx, &enc, rows, mla, latent, pending, pooled, mla_index)?;
                    mla_index += 1;
                }
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
        mla_index: usize,
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
        absorb_rows(
            ctx,
            enc,
            p.lineage,
            &mla.key_absorb,
            &p.query,
            &p.query_latent,
            head_dim,
            kv,
            heads,
            rows,
        )?;
        // Indexer cache maintenance first: every pool the chunk completes is
        // published before attention, and each row scores only its own
        // visible prefix (the first L / 4 pools).
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
        // Rows before the frontier attend densely; the rest attend over the
        // indexer's selection.
        let scale = 1.0 / (head_dim as f32).sqrt();
        let dense = self.dense_rows_remaining().min(rows);
        if dense > 0 {
            encode_latent_attention(
                ctx,
                enc,
                &rows_view(&p.query_latent, dense),
                latent,
                &self.s.no_sink,
                &rows_view(&p.output_latent, dense),
                self.position,
                dense,
                scale,
            )?;
        }
        if dense < rows {
            self.encode_sparse_rows(ctx, enc, mla, latent, pooled, mla_index, dense, rows, scale)?;
        }
        absorb_rows(
            ctx,
            enc,
            p.lineage,
            &mla.value_expand,
            &p.output_latent,
            &p.heads_out,
            kv,
            head_dim,
            heads,
            rows,
        )?;
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
        Ok(())
    }

    /// Sparse attention for chunk rows `first..rows` (all at or past the
    /// frontier): chunk-wide indexer queries (F16-rounded) and head weights
    /// scaled by 1 / sqrt(heads * dim), then per microbatch of up to
    /// [`memory::PACKED_SPARSE_QUERIES`] rows: scores over each row's visible
    /// pools, exact top-512 into the (block, row) status slots, expansion
    /// and online attention over exactly the selected latent rows. The same
    /// kernels as decode, so per-row results match serial decode under the
    /// exact lineage.
    #[allow(clippy::too_many_arguments)]
    fn encode_sparse_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        mla: &crate::glm5_next::MlaTensors<MetalTensor>,
        latent: &MetalTensor,
        pooled: &MetalTensor,
        mla_index: usize,
        first: usize,
        rows: usize,
        scale: f32,
    ) -> Result<()> {
        let c = &self.weights.config;
        let p = self.packed.as_ref().expect("packed scratch");
        let sp = p.sparse.as_ref().ok_or_else(|| {
            Glm5NextMetalError::Invalid("packed sparse rows without packed sparse scratch".into())
        })?;
        let (h, q_rank) = (c.hidden_size as usize, c.q_lora_rank as usize);
        let (ih, id) = (c.indexer_head_count as usize, c.indexer_head_dim as usize);
        let (heads, kv) = (c.head_count as usize, c.kv_lora_rank as usize);
        let query_width = ih * id;
        let count = rows - first;
        // Projections over the sparse rows (row-wise in the exact lineage).
        let sub = |t: &MetalTensor, width: usize| {
            t.view_subrange((first * width) as u64, vec![width as u64, count as u64])
        };
        matmat(
            ctx,
            enc,
            p.lineage,
            &mla.indexer.query,
            &sub(&p.query_r, q_rank),
            &sub(&sp.index_query, query_width),
            q_rank,
            query_width,
            count,
        )?;
        encode_scatter_offset_f32_to_f16(
            ctx,
            enc,
            &sub(&sp.index_query, query_width).view_subrange(0, vec![(query_width * count) as u64]),
            &sp.index_query_f16,
            first * query_width,
            query_width * count,
        )?;
        let weights = sub(&sp.index_weights, ih);
        matmat(
            ctx,
            enc,
            p.lineage,
            &mla.indexer.head_weights,
            &sub(&p.normed, h),
            &weights,
            h,
            ih,
            count,
        )?;
        crate::metal::encode_scale_f32_in_place(
            ctx,
            enc,
            &weights,
            1.0 / (query_width as f32).sqrt(),
        )?;
        let pool_capacity = pooled.shape[1] as usize;
        let top_pools = c.selected_pool_count() as usize;
        let row_slots = c.selection_width() as usize;
        let pool = c.indexer_pool as usize;
        let flat = |t: &MetalTensor| t.view_subrange(0, vec![(kv * heads) as u64, rows as u64]);
        let mut start = first;
        while start < rows {
            let n = sp.microbatch.min(rows - start);
            let i32_rows = |t: &MetalTensor| t.view_subrange(start as u64, vec![n as u64]);
            let cols =
                |t: &MetalTensor, width: usize| t.view_subrange(0, vec![width as u64, n as u64]);
            // Visibility grows with the row: the last row bounds the batch.
            let max_pools = (self.position + start + n) / pool;
            let queries_f16 = sp.index_query_f16.view_subrange(
                (start * query_width) as u64,
                vec![id as u64, ih as u64, n as u64],
            );
            let head_weights = sp
                .index_weights
                .view_subrange((start * ih) as u64, vec![ih as u64, n as u64]);
            let (visible_pools, visible_rows) =
                (i32_rows(&sp.visible_pools), i32_rows(&sp.visible_rows));
            let scores = cols(&sp.scores, pool_capacity);
            let (pool_ids, pool_counts) = (
                cols(&sp.pool_ids, top_pools),
                sp.pool_counts.view_subrange(0, vec![n as u64]),
            );
            let (row_ids, row_counts) = (
                cols(&sp.row_ids, row_slots),
                sp.row_counts.view_subrange(0, vec![n as u64]),
            );
            let status = sp
                .select_status
                .view_subrange((mla_index * sp.rows + start) as u64, vec![n as u64]);
            crate::metal::encode_lightning_scores_f16_matrix(
                ctx,
                enc,
                &crate::metal::LightningScores {
                    queries: &queries_f16,
                    head_weights: &head_weights,
                    keys: pooled,
                    visible_counts: &visible_pools,
                    scores: &scores,
                },
                ih,
                id,
                pool_capacity,
                max_pools,
                n,
            )?;
            crate::metal::encode_select_top_k_ids(
                ctx,
                enc,
                &crate::metal::TopKSelection {
                    scores: &scores,
                    visible_counts: &visible_pools,
                    ids: &pool_ids,
                    counts: &pool_counts,
                    status: &status,
                },
                pool_capacity,
                max_pools,
                top_pools,
                n,
            )?;
            crate::metal::encode_indexer_expand_selection(
                ctx,
                enc,
                &crate::metal::IndexerSelection {
                    pool_ids: &pool_ids,
                    pool_counts: &pool_counts,
                    visible_rows: &visible_rows,
                    row_ids: &row_ids,
                    row_counts: &row_counts,
                },
                top_pools,
                row_slots,
                n,
            )?;
            crate::metal::encode_online_selected_attention_f16(
                ctx,
                enc,
                &crate::metal::SelectedAttention {
                    queries: &flat(&p.query_latent),
                    raw_cache: latent,
                    raw_cache_before_chunk: latent,
                    compressed_cache: latent,
                    selected_ids: &row_ids,
                    selected_counts: &row_counts,
                    visible_counts: &visible_rows,
                    sinks: &self.s.no_sink,
                    output: &flat(&p.output_latent),
                },
                crate::metal::SelectedAttentionShape {
                    head_count: heads,
                    query_count: n,
                    query_token_offset: start,
                    token_count: rows,
                    chunk_start_position: self.position,
                    window: 0,
                    raw_cache_is_chunk: false,
                    selected_slots: row_slots,
                    compressed_capacity: latent.shape[1] as usize,
                    scale,
                    direct: true,
                },
            )?;
            start += n;
        }
        Ok(())
    }
}
