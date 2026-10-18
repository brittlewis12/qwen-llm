//! Encode-only F32 attention. Workspace metadata is provisional: a failed encode
//! or command completion requires the caller to poison/discard the workspace.
use super::*;
use objc2_metal::MTLSize;

#[cfg(test)]
mod tests;

type Weights<'a> = op::AttentionWeights<Linear<'a>>;

struct Constants {
    query_norm: MetalTensor,
    kv_norm: MetalTensor,
    sink: MetalTensor,
    frequencies: MetalTensor,
    compressor_norm: Option<MetalTensor>,
    key_norm: Option<MetalTensor>,
}

struct OwnerCache {
    compressed: MetalTensor,
    keys: MetalTensor,
    pending_value: MetalTensor,
    pending_gate: MetalTensor,
}

pub(super) struct Workspace {
    config: Config,
    capacity: usize,
    next_layer: usize,
    next_position: usize,
    constants: Vec<Constants>,
    raw: MetalTensor,
    owners: Vec<OwnerCache>,
    query_rank: MetalTensor,
    query: MetalTensor,
    latent: MetalTensor,
    gate: MetalTensor,
    key: MetalTensor,
    index_query: MetalTensor,
    index_weights: MetalTensor,
    index_scores: MetalTensor,
    block_scores: MetalTensor,
    candidates: MetalTensor,
    sort_scores: MetalTensor,
    sort_ids: MetalTensor,
    selected: MetalTensor,
    attention_scores: MetalTensor,
    attended: MetalTensor,
    low: MetalTensor,
    output: MetalTensor,
    status: MetalTensor,
}

fn checked_product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .filter(|&n| n <= u32::MAX as usize / 4)
            .ok_or_else(|| "GPU attention extent overflow".into())
    })
}

fn validate(c: &Config, capacity: usize) -> Result<()> {
    op::require(
        capacity > 0 && capacity <= u32::MAX as usize / 8,
        "GPU attention capacity",
    )?;
    op::require(
        [
            c.hidden,
            c.heads,
            c.head_dim,
            c.query_rank,
            c.output_rank,
            c.groups,
            c.index_heads,
            c.index_dim,
            c.index_topk,
            c.window,
            c.candidate_block,
            c.candidate_topk,
        ]
        .iter()
        .all(|&x| x > 0 && x <= u32::MAX as usize / 4)
            && c.heads % c.groups == 0
            && c.rope_dim % 2 == 0
            && c.rope_dim <= c.head_dim
            && c.rope_dim <= c.index_dim
            && c.norm_eps.is_finite()
            && c.norm_eps > 0.0,
        "GPU attention geometry",
    )?;
    for d in [
        [40, c.window, c.head_dim],
        [capacity, c.head_dim, 1],
        [capacity, c.index_dim, 1],
        [c.heads, c.head_dim, 1],
        [c.index_heads, c.index_dim, 1],
        [c.groups, c.output_rank, 1],
        [
            c.heads,
            c.window
                .checked_add(c.index_topk)
                .ok_or("attention overflow")?,
            1,
        ],
    ] {
        checked_product(&d)?;
    }
    Ok(())
}

fn shapes(c: &Config, capacity: usize) -> Result<Vec<usize>> {
    validate(c, capacity)?;
    let mut sizes = vec![
        40 * c.window * c.head_dim,
        c.query_rank,
        c.heads * c.head_dim,
        c.head_dim,
        c.head_dim,
        c.index_dim,
        c.index_heads * c.index_dim,
        c.index_heads,
        capacity,
        capacity.div_ceil(c.candidate_block),
        capacity.div_ceil(c.candidate_block),
        capacity.next_power_of_two(),
        capacity.next_power_of_two(),
        c.index_topk,
        c.heads * (c.window + c.index_topk),
        c.heads * c.head_dim,
        c.groups * c.output_rank,
        c.hidden,
        1,
    ];
    for ratio in [2, 2, 2, 1] {
        sizes.extend([
            capacity.div_ceil(ratio) * c.head_dim,
            capacity.div_ceil(ratio) * c.index_dim,
            c.head_dim,
            c.head_dim,
        ]);
    }
    for _ in 0..40 {
        sizes.extend([
            c.query_rank,
            c.head_dim,
            c.heads,
            (c.rope_dim / 2).max(1),
            c.head_dim,
            c.index_dim,
        ]);
    }
    Ok(sizes)
}

pub(super) fn bytes(c: &Config, capacity: usize) -> Result<u64> {
    shapes(c, capacity)?.into_iter().try_fold(0u64, |sum, n| {
        sum.checked_add(n as u64 * 4 + 17408)
            .ok_or_else(|| "GPU attention memory overflow".into())
    })
}

fn tensor(ctx: &MetalContext, n: usize, dtype: GgmlType) -> Result<MetalTensor> {
    MetalTensor::zeros_dtype_unstaged(ctx, vec![n as u64], dtype).map_err(|e| e.to_string())
}
fn pipeline(ctx: &MetalContext, enc: &KernelEncoder, suffix: &str) -> Result<()> {
    enc.set_pipeline(
        &ctx.pipeline(&format!("kernel_deepseek_v41_attention_{suffix}"))
            .map_err(|e| e.to_string())?,
    );
    Ok(())
}
fn dispatch(enc: &KernelEncoder, n: usize) {
    if n > 0 {
        enc.dispatch(
            MTLSize {
                width: n,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
    }
}
fn linear(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    w: &Linear<'_>,
    x: &MetalTensor,
    y: &MetalTensor,
) -> Result<()> {
    metal::encode_mat_vec_f32(ctx, enc, &w.tensor, x, y, w.input, w.output)
        .map_err(|e| e.to_string())
}
fn norm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    w: &MetalTensor,
    n: usize,
    eps: f32,
) -> Result<()> {
    valid_view(x, n)?;
    metal::encode_rms_norm_mul_f32(ctx, enc, x, w, x, eps).map_err(|e| e.to_string())
}
fn rope(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    freq: &MetalTensor,
    dim: usize,
    heads: usize,
    count: usize,
    position: usize,
    inverse: bool,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    pipeline(ctx, enc, "rope")?;
    enc.set_bytes(
        0,
        &[dim as u32, count as u32, position as u32, inverse as u32],
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, freq);
    dispatch(enc, heads * count);
    Ok(())
}
fn copy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    y: &MetalTensor,
    n: usize,
) -> Result<()> {
    pipeline(ctx, enc, "copy")?;
    enc.set_bytes(0, &(n as u32));
    enc.set_tensor(1, x);
    enc.set_tensor(2, y);
    dispatch(enc, n);
    Ok(())
}

fn projection(w: &Linear<'_>, input: usize, output: usize) -> Result<()> {
    valid_view(&w.tensor, checked_product(&[input, output])?)?;
    op::require(
        w.input == input && w.output == output && w.tensor.dtype == GgmlType::F32,
        "GPU attention projection geometry/dtype",
    )
}

fn valid_view(t: &MetalTensor, count: usize) -> Result<()> {
    let elements = t.shape.iter().try_fold(1u64, |n, &d| n.checked_mul(d));
    let end = (count as u64)
        .checked_mul(4)
        .and_then(|n| n.checked_add(t.offset));
    op::require(
        t.dtype == GgmlType::F32
            && elements == Some(count as u64)
            && t.offset % 4 == 0
            && end.is_some_and(|end| end <= t.buffer.length() as u64),
        "GPU attention buffer bounds/dtype",
    )
}

impl Workspace {
    pub(super) fn capture(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        snapshot: &capture::Snapshot,
        input: &MetalTensor,
        id: usize,
        position: usize,
    ) -> Result<()> {
        let count = ((position + 1) / if id < 20 { 2 } else { 1 }).min(self.config.index_topk);
        snapshot.encode(
            ctx,
            enc,
            input,
            &self.output,
            &self.selected,
            &self.candidates,
            count,
        )
    }

    pub(super) fn new(
        ctx: &MetalContext,
        c: &Config,
        capacity: usize,
        weights: &[&Weights<'_>],
    ) -> Result<Self> {
        let size = bytes(c, capacity)?;
        op::require(weights.len() == 40, "GPU attention layer count")?;
        for n in shapes(c, capacity)? {
            let priced = ctx
                .price_shared_buffer_upper(n as u64 * 4)
                .map_err(|e| e.to_string())?;
            op::require(
                priced.priced_upper_bytes <= n as u64 * 4 + 16384,
                "GPU attention allocation page slack",
            )?;
        }
        let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
            size,
            0,
            0,
            ctx.memory_signals(),
            true,
        );
        if let Some(reason) = admission.refusal() {
            return Err(format!("GPU attention admission: {reason:?}"));
        }
        let constants = weights
            .iter()
            .enumerate()
            .map(|(id, w)| {
                validate_weights(c, id, w)?;
                let up = |x: &[f32]| upload(ctx, x, vec![x.len().max(1) as u64]);
                Ok(Constants {
                    query_norm: up(&w.query_norm)?,
                    kv_norm: up(&w.kv_norm)?,
                    sink: up(&w.sink)?,
                    frequencies: if w.frequencies.is_empty() {
                        up(&[0.0])?
                    } else {
                        up(&w.frequencies)?
                    },
                    compressor_norm: w.compressor.as_ref().map(|w| up(&w.norm)).transpose()?,
                    key_norm: w
                        .index
                        .as_ref()
                        .filter(|w| w.key.is_some())
                        .map(|w| up(&w.key_norm))
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let f = |n| tensor(ctx, n, GgmlType::F32);
        let i = |n| tensor(ctx, n, GgmlType::I32);
        let owners = [2, 2, 2, 1]
            .into_iter()
            .map(|ratio| {
                Ok(OwnerCache {
                    compressed: f(capacity.div_ceil(ratio) * c.head_dim)?,
                    keys: f(capacity.div_ceil(ratio) * c.index_dim)?,
                    pending_value: f(c.head_dim)?,
                    pending_gate: f(c.head_dim)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config: c.clone(),
            capacity,
            next_layer: 0,
            next_position: 0,
            constants,
            raw: f(40 * c.window * c.head_dim)?,
            owners,
            query_rank: f(c.query_rank)?,
            query: f(c.heads * c.head_dim)?,
            latent: f(c.head_dim)?,
            gate: f(c.head_dim)?,
            key: f(c.index_dim)?,
            index_query: f(c.index_heads * c.index_dim)?,
            index_weights: f(c.index_heads)?,
            index_scores: f(capacity)?,
            block_scores: f(capacity.div_ceil(c.candidate_block))?,
            candidates: i(capacity.div_ceil(c.candidate_block))?,
            sort_scores: f(capacity.next_power_of_two())?,
            sort_ids: i(capacity.next_power_of_two())?,
            selected: i(c.index_topk)?,
            attention_scores: f(c.heads * (c.window + c.index_topk))?,
            attended: f(c.heads * c.head_dim)?,
            low: f(c.groups * c.output_rank)?,
            output: f(c.hidden)?,
            status: i(1)?,
        })
    }

    fn sort(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        scores: &MetalTensor,
        n: usize,
        mask: bool,
    ) -> Result<()> {
        // The shared radix selector rejects nonfinite scores. Here -Inf masked
        // rows still participate in top-k, and forced-last blocks use +Inf.
        let padded = n.max(1).next_power_of_two();
        pipeline(ctx, enc, "sort_init")?;
        enc.set_bytes(
            0,
            &[
                n as u32,
                padded as u32,
                mask as u32,
                self.config.candidate_block as u32,
            ],
        );
        enc.set_tensor(1, scores);
        enc.set_tensor(2, &self.candidates);
        enc.set_tensor(3, &self.sort_scores);
        enc.set_tensor(4, &self.sort_ids);
        enc.set_tensor(5, &self.status);
        dispatch(enc, padded);
        let mut width = 2;
        while width <= padded {
            let mut stride = width / 2;
            while stride > 0 {
                pipeline(ctx, enc, "sort_step")?;
                enc.set_bytes(0, &[padded as u32, stride as u32, width as u32, 0]);
                enc.set_tensor(1, &self.sort_scores);
                enc.set_tensor(2, &self.sort_ids);
                dispatch(enc, padded);
                stride /= 2;
            }
            width *= 2;
        }
        Ok(())
    }

    /// Requires a serial encoder and strict token-major forty-layer order.
    /// Weights and their constants must be the immutable weights passed to new.
    /// The returned tensor aliases scratch and must be consumed before reuse.
    /// No submission, wait, host activation access, or HC occurs here.
    pub(super) fn encode(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        id: usize,
        position: usize,
        w: &Weights<'_>,
        input: &MetalTensor,
    ) -> Result<&MetalTensor> {
        op::require(
            !enc.is_concurrent(),
            "GPU attention requires serial encoder",
        )?;
        let c = &self.config;
        op::require(
            id == self.next_layer && position == self.next_position && position < self.capacity,
            "GPU attention order/capacity",
        )?;
        validate_weights(c, id, w)?;
        valid_view(input, c.hidden)?;
        let constants = &self.constants[id];
        linear(ctx, enc, &w.query_a, input, &self.query_rank)?;
        norm(
            ctx,
            enc,
            &self.query_rank,
            &constants.query_norm,
            c.query_rank,
            w.eps,
        )?;
        linear(ctx, enc, &w.query_b, &self.query_rank, &self.query)?;
        rope(
            ctx,
            enc,
            &self.query,
            &constants.frequencies,
            c.head_dim,
            c.heads,
            c.rope_dim / 2,
            position,
            false,
        )?;
        linear(ctx, enc, &w.kv, input, &self.latent)?;
        norm(
            ctx,
            enc,
            &self.latent,
            &constants.kv_norm,
            c.head_dim,
            w.eps,
        )?;
        rope(
            ctx,
            enc,
            &self.latent,
            &constants.frequencies,
            c.head_dim,
            1,
            c.rope_dim / 2,
            position,
            false,
        )?;
        let raw = self.raw.view_subrange(
            (id * c.window * c.head_dim) as u64,
            vec![(c.window * c.head_dim) as u64],
        );
        let row = raw.view_subrange(
            (position % c.window * c.head_dim) as u64,
            vec![c.head_dim as u64],
        );
        copy(ctx, enc, &self.latent, &row, c.head_dim)?;
        let owner = [2, 8, 14, 20].into_iter().rposition(|owner| owner <= id);
        let mut selected_count = 0;
        if let Some(owner) = owner {
            let cache = &self.owners[owner];
            let ratio = if owner == 3 { 1 } else { 2 };
            let length = (position + 1) / ratio;
            if id == [2, 8, 14, 20][owner] {
                let compressor = w.compressor.as_ref().ok_or("missing compressor")?;
                linear(ctx, enc, &compressor.kv, input, &self.latent)?;
                if ratio == 2 {
                    linear(
                        ctx,
                        enc,
                        compressor.gate.as_ref().ok_or("missing gate")?,
                        input,
                        &self.gate,
                    )?;
                    if position % 2 == 0 {
                        copy(ctx, enc, &self.latent, &cache.pending_value, c.head_dim)?;
                        copy(ctx, enc, &self.gate, &cache.pending_gate, c.head_dim)?;
                    } else {
                        pipeline(ctx, enc, "pool")?;
                        enc.set_bytes(0, &(c.head_dim as u32));
                        enc.set_tensor(1, &cache.pending_value);
                        enc.set_tensor(2, &cache.pending_gate);
                        enc.set_tensor(3, &self.latent);
                        enc.set_tensor(4, &self.gate);
                        dispatch(enc, c.head_dim);
                    }
                }
                if (position + 1) % ratio == 0 {
                    norm(
                        ctx,
                        enc,
                        &self.latent,
                        constants.compressor_norm.as_ref().ok_or("missing norm")?,
                        c.head_dim,
                        w.eps,
                    )?;
                    let index = w.index.as_ref().ok_or("missing index")?;
                    linear(
                        ctx,
                        enc,
                        index.key.as_ref().ok_or("missing key")?,
                        &self.latent,
                        &self.key,
                    )?;
                    norm(
                        ctx,
                        enc,
                        &self.key,
                        constants.key_norm.as_ref().ok_or("missing key norm")?,
                        c.index_dim,
                        w.eps,
                    )?;
                    let group_position = position + 1 - ratio;
                    rope(
                        ctx,
                        enc,
                        &self.key,
                        &constants.frequencies,
                        c.index_dim,
                        1,
                        c.rope_dim / 2,
                        group_position,
                        false,
                    )?;
                    rope(
                        ctx,
                        enc,
                        &self.latent,
                        &constants.frequencies,
                        c.head_dim,
                        1,
                        c.rope_dim / 2,
                        group_position,
                        false,
                    )?;
                    copy(
                        ctx,
                        enc,
                        &self.key,
                        &cache.keys.view_subrange(
                            ((length - 1) * c.index_dim) as u64,
                            vec![c.index_dim as u64],
                        ),
                        c.index_dim,
                    )?;
                    copy(
                        ctx,
                        enc,
                        &self.latent,
                        &cache.compressed.view_subrange(
                            ((length - 1) * c.head_dim) as u64,
                            vec![c.head_dim as u64],
                        ),
                        c.head_dim,
                    )?;
                }
            }
            selected_count = length.min(c.index_topk);
            if [2, 8, 14, 20, 24, 28, 32, 36].contains(&id) && length > 0 {
                let index = w.index.as_ref().ok_or("missing index")?;
                linear(ctx, enc, &index.query, &self.query_rank, &self.index_query)?;
                linear(ctx, enc, &index.weights, input, &self.index_weights)?;
                rope(
                    ctx,
                    enc,
                    &self.index_query,
                    &constants.frequencies,
                    c.index_dim,
                    c.index_heads,
                    c.rope_dim / 2,
                    position,
                    false,
                )?;
                pipeline(ctx, enc, "index")?;
                enc.set_bytes(
                    0,
                    &[length as u32, c.index_dim as u32, c.index_heads as u32, 0],
                );
                enc.set_tensor(1, &self.index_query);
                enc.set_tensor(2, &self.index_weights);
                enc.set_tensor(3, &cache.keys);
                enc.set_tensor(4, &self.index_scores);
                dispatch(enc, length);
                if id == 20 {
                    let blocks = length.div_ceil(c.candidate_block);
                    pipeline(ctx, enc, "blocks")?;
                    enc.set_bytes(0, &[length as u32, c.candidate_block as u32]);
                    enc.set_tensor(1, &self.index_scores);
                    enc.set_tensor(2, &self.block_scores);
                    dispatch(enc, blocks);
                    self.sort(ctx, enc, &self.block_scores, blocks, false)?;
                    pipeline(ctx, enc, "keep")?;
                    enc.set_bytes(0, &[blocks as u32, blocks.min(c.candidate_topk) as u32]);
                    enc.set_tensor(1, &self.sort_ids);
                    enc.set_tensor(2, &self.candidates);
                    dispatch(enc, blocks);
                }
                self.sort(ctx, enc, &self.index_scores, length, id > 20)?;
                pipeline(ctx, enc, "chronological")?;
                enc.set_bytes(0, &(selected_count as u32));
                enc.set_tensor(1, &self.sort_ids);
                enc.set_tensor(2, &self.selected);
                dispatch(enc, selected_count);
            }
        }
        let raw_count = (position + 1).min(c.window);
        pipeline(ctx, enc, "sink")?;
        enc.set_bytes(
            0,
            &[
                c.head_dim as u32,
                raw_count as u32,
                ((position + 1 - raw_count) % c.window) as u32,
                c.window as u32,
            ],
        );
        enc.set_bytes(
            1,
            &[selected_count as u32, (c.window + c.index_topk) as u32],
        );
        enc.set_tensor(2, &self.query);
        enc.set_tensor(3, &raw);
        enc.set_tensor(4, &self.owners[owner.unwrap_or(0)].compressed);
        enc.set_tensor(5, &self.selected);
        enc.set_tensor(6, &constants.sink);
        enc.set_tensor(7, &self.attention_scores);
        enc.set_tensor(8, &self.attended);
        let length = owner.map_or(0, |o| (position + 1) / if o == 3 { 1 } else { 2 });
        enc.set_bytes(9, &(length as u32));
        enc.set_tensor(10, &self.status);
        dispatch(enc, c.heads);
        rope(
            ctx,
            enc,
            &self.attended,
            &constants.frequencies,
            c.head_dim,
            c.heads,
            c.rope_dim / 2,
            position,
            true,
        )?;
        let group_width = c.heads / c.groups * c.head_dim;
        for (group, projection) in w.output_a.iter().enumerate() {
            linear(
                ctx,
                enc,
                projection,
                &self
                    .attended
                    .view_subrange((group * group_width) as u64, vec![group_width as u64]),
                &self
                    .low
                    .view_subrange((group * c.output_rank) as u64, vec![c.output_rank as u64]),
            )?;
        }
        linear(ctx, enc, &w.output_b, &self.low, &self.output)?;
        pipeline(ctx, enc, "finite")?;
        enc.set_bytes(0, &(c.hidden as u32));
        enc.set_tensor(1, &self.output);
        enc.set_tensor(2, &self.status);
        dispatch(enc, c.hidden);
        self.next_layer = (id + 1) % 40;
        if self.next_layer == 0 {
            self.next_position += 1;
        }
        Ok(&self.output)
    }

    /// Sticky device error bits: 1 NaN selection score, 2 invalid selected ID,
    /// 4 nonfinite attention. Inspect only after the caller's command completes.
    pub(super) fn status(&self) -> &MetalTensor {
        &self.status
    }
}

fn validate_weights(c: &Config, id: usize, w: &Weights<'_>) -> Result<()> {
    let owns = [2, 8, 14, 20].contains(&id);
    let produces = [2, 8, 14, 20, 24, 28, 32, 36].contains(&id);
    op::require(
        w.compressor.is_some() == owns && w.index.is_some() == produces,
        "GPU attention ownership weights",
    )?;
    op::require(
        w.query_norm
            .iter()
            .chain(&w.kv_norm)
            .chain(&w.sink)
            .chain(&w.frequencies)
            .all(|x| x.is_finite()),
        "GPU attention finite constants",
    )?;
    projection(&w.query_a, c.hidden, c.query_rank)?;
    projection(&w.query_b, c.query_rank, c.heads * c.head_dim)?;
    projection(&w.kv, c.hidden, c.head_dim)?;
    projection(&w.output_b, c.groups * c.output_rank, c.hidden)?;
    op::require(
        w.head_dim == c.head_dim
            && w.window == c.window
            && w.query_norm.len() == c.query_rank
            && w.kv_norm.len() == c.head_dim
            && w.sink.len() == c.heads
            && w.frequencies.len() == c.rope_dim / 2
            && w.output_a.len() == c.groups
            && w.eps == c.norm_eps
            && w.candidate_block_size == c.candidate_block
            && w.candidate_topk_blocks == c.candidate_topk,
        "GPU attention weights",
    )?;
    for p in &w.output_a {
        projection(p, c.heads / c.groups * c.head_dim, c.output_rank)?;
    }
    if [2, 8, 14, 20, 24, 28, 32, 36].contains(&id) {
        let index = w.index.as_ref().ok_or("missing GPU index weights")?;
        op::require(
            index.heads == c.index_heads
                && index.dim == c.index_dim
                && index.topk == c.index_topk
                && index.key.is_some() == owns
                && (owns || index.key_norm.is_empty())
                && index.key_norm.iter().all(|x| x.is_finite()),
            "GPU index geometry",
        )?;
        projection(&index.query, c.query_rank, c.index_heads * c.index_dim)?;
        projection(&index.weights, c.hidden, c.index_heads)?;
        if [2, 8, 14, 20].contains(&id) {
            let compressor = w.compressor.as_ref().ok_or("missing GPU compressor")?;
            op::require(
                compressor.ratio == if id == 20 { 1 } else { 2 }
                    && compressor.norm.len() == c.head_dim
                    && compressor.gate.is_some() == (id != 20)
                    && compressor.norm.iter().all(|x| x.is_finite())
                    && index.key_norm.len() == c.index_dim,
                "GPU compressor geometry",
            )?;
            projection(&compressor.kv, c.hidden, c.head_dim)?;
            if id != 20 {
                projection(
                    compressor.gate.as_ref().ok_or("missing GPU gate")?,
                    c.hidden,
                    c.head_dim,
                )?;
            }
            projection(
                index.key.as_ref().ok_or("missing GPU key")?,
                c.head_dim,
                c.index_dim,
            )?;
        }
    }
    Ok(())
}
