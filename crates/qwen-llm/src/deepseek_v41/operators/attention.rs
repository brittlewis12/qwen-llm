use super::{Linear, Projection, Result, dot, require, rms_norm, rotary};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct CompressorWeights<P = Linear> {
    pub ratio: usize,
    pub kv: P,
    pub gate: Option<P>,
    pub norm: Vec<f32>,
}
#[derive(Clone, Debug, Default)]
pub struct CompressorState {
    next: usize,
    values: Vec<Vec<f32>>,
    scores: Vec<Vec<f32>>,
}
impl CompressorState {
    pub fn push<P: Projection>(
        &mut self,
        w: &CompressorWeights<P>,
        x: &[f32],
        position: usize,
        eps: f32,
    ) -> Result<Option<Vec<f32>>> {
        require(
            position == self.next && matches!(w.ratio, 1 | 2),
            "compressor position/ratio",
        )?;
        let value = w.kv.forward(x)?;
        let score = if w.ratio == 2 {
            w.gate
                .as_ref()
                .ok_or("missing compressor gate")?
                .forward(x)?
        } else {
            vec![0.0; value.len()]
        };
        require(
            score.len() == value.len() && w.norm.len() == value.len(),
            "compressor width",
        )?;
        self.values.push(value);
        self.scores.push(score);
        self.next += 1;
        if self.values.len() < w.ratio {
            return Ok(None);
        }
        let pooled = (0..w.norm.len())
            .map(|j| {
                let max = self
                    .scores
                    .iter()
                    .map(|s| s[j])
                    .fold(f32::NEG_INFINITY, f32::max);
                let denom: f32 = self.scores.iter().map(|s| (s[j] - max).exp()).sum();
                self.values
                    .iter()
                    .zip(&self.scores)
                    .map(|(v, s)| v[j] * ((s[j] - max).exp() / denom))
                    .sum()
            })
            .collect::<Vec<_>>();
        let result = rms_norm(&pooled, &w.norm, eps)?;
        self.values.clear();
        self.scores.clear();
        Ok(Some(result))
    }
}

#[derive(Clone, Debug)]
pub struct IndexWeights<P = Linear> {
    pub query: P,
    pub weights: P,
    pub key: Option<P>,
    pub key_norm: Vec<f32>,
    pub heads: usize,
    pub dim: usize,
    pub topk: usize,
}
#[derive(Clone, Debug)]
pub struct AttentionWeights<P = Linear> {
    pub query_a: P,
    pub query_norm: Vec<f32>,
    pub query_b: P,
    pub kv: P,
    pub kv_norm: Vec<f32>,
    pub sink: Vec<f32>,
    /// One [rank, heads_per_group * head_dim] matrix per output group.
    pub output_a: Vec<P>,
    pub output_b: P,
    pub compressor: Option<CompressorWeights<P>>,
    pub index: Option<IndexWeights<P>>,
    pub frequencies: Vec<f32>,
    pub head_dim: usize,
    pub window: usize,
    pub eps: f32,
    pub candidate_block_size: usize,
    pub candidate_topk_blocks: usize,
}
#[derive(Clone, Debug, Default)]
pub(super) struct LayerState {
    next: usize,
    window: Vec<Vec<f32>>,
    compressor: CompressorState,
    compressed: Vec<Vec<f32>>,
    keys: Vec<Vec<f32>>,
}
#[derive(Clone, Debug, Default)]
pub struct Session {
    pub(super) layers: BTreeMap<usize, LayerState>,
    pub(super) published: Option<(usize, usize)>,
    selection: Option<(usize, usize, Vec<usize>)>,
    candidates: Option<(usize, Vec<bool>)>,
}

/// Logical chronological rows, independent of the reference's physical SWA ring.
#[derive(Clone, Debug)]
pub struct CacheSnapshot {
    pub next_position: usize,
    pub window: Vec<Vec<f32>>,
    pub compressed: Vec<Vec<f32>>,
    pub keys: Vec<Vec<f32>>,
    pub pending_values: Vec<Vec<f32>>,
    pub pending_scores: Vec<Vec<f32>>,
    pub published_owner: Option<(usize, usize)>,
    pub selection: Option<(usize, usize, Vec<usize>)>,
    pub candidates: Option<(usize, Vec<bool>)>,
}

struct LayerCheckpoint {
    next: usize,
    window: Vec<Vec<f32>>,
    compressor: CompressorState,
    compressed_len: usize,
    keys_len: usize,
}

fn owner(layer: usize) -> Option<usize> {
    [2, 8, 14, 20].into_iter().filter(|&x| x <= layer).last()
}
fn producer(layer: usize) -> Option<usize> {
    [2, 8, 14, 20, 24, 28, 32, 36]
        .into_iter()
        .filter(|&x| x <= layer)
        .last()
}

/// Position-sorted top-k; ties choose the lower position. PyTorch does not
/// promise a stable tie order, so oracle selection comparisons avoid ties.
pub fn topk(scores: &[f32], k: usize) -> Vec<usize> {
    let mut indices: Vec<_> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    indices.truncate(k.min(indices.len()));
    indices.sort_unstable();
    indices
}
pub fn candidate_blocks(scores: &[f32], block_size: usize, k: usize) -> Result<Vec<bool>> {
    require(block_size > 0 && k > 0, "candidate geometry")?;
    let mut blocks: Vec<_> = scores
        .chunks(block_size)
        .map(|c| c.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        .collect();
    if let Some(last) = blocks.last_mut() {
        *last = f32::INFINITY;
    }
    let keep = topk(&blocks, k);
    Ok((0..scores.len())
        .map(|i| keep.contains(&(i / block_size)))
        .collect())
}

/// Shared K=V with a zero-valued sink in the softmax denominator.
pub fn sparse_attention(
    query: &[f32],
    rows: &[Vec<f32>],
    indices: &[usize],
    sink: f32,
) -> Result<Vec<f32>> {
    require(
        !query.is_empty()
            && indices
                .iter()
                .all(|&i| i < rows.len() && rows[i].len() == query.len()),
        "sparse attention geometry",
    )?;
    let scale = (query.len() as f32).sqrt().recip();
    let mut scores: Vec<_> = indices
        .iter()
        .map(|&i| dot(query, &rows[i]) * scale)
        .collect();
    let max = scores.iter().copied().fold(sink, f32::max);
    // Reuse exponentials without changing each term's multiply-then-divide order.
    scores.iter_mut().for_each(|s| *s = (*s - max).exp());
    let denom = (sink - max).exp() + scores.iter().sum::<f32>();
    Ok((0..query.len())
        .map(|j| {
            indices
                .iter()
                .zip(&scores)
                .map(|(&i, s)| rows[i][j] * s / denom)
                .sum()
        })
        .collect())
}

impl Session {
    pub fn cache_snapshot(&self, layer: usize) -> Option<CacheSnapshot> {
        self.layers.get(&layer).map(|s| CacheSnapshot {
            next_position: s.next,
            window: s.window.clone(),
            compressed: s.compressed.clone(),
            keys: s.keys.clone(),
            pending_values: s.compressor.values.clone(),
            pending_scores: s.compressor.scores.clone(),
            published_owner: self.published,
            selection: self.selection.clone(),
            candidates: self.candidates.clone(),
        })
    }
    /// Transactional on failure; callers execute layers in ascending order for
    /// each token. Reindex/reuse reject caches published for another token/owner.
    pub fn attention<P: Projection>(
        &mut self,
        layer: usize,
        position: usize,
        x: &[f32],
        w: &AttentionWeights<P>,
    ) -> Result<Vec<f32>> {
        // Only the affected layer mutates. Completed caches are append-only,
        // so rollback needs lengths, not a clone of the accumulated prompt.
        let checkpoint = self.layers.get(&layer).map(|s| LayerCheckpoint {
            next: s.next,
            window: s.window.clone(),
            compressor: s.compressor.clone(),
            compressed_len: s.compressed.len(),
            keys_len: s.keys.len(),
        });
        let published = self.published;
        let selection = (producer(layer) == Some(layer)).then(|| self.selection.clone());
        let candidates = (layer == 20).then(|| self.candidates.clone());
        let result = self.attention_inner(layer, position, x, w);
        if result.is_err() {
            if let Some(c) = checkpoint {
                let s = self
                    .layers
                    .get_mut(&layer)
                    .expect("existing layer checkpoint");
                s.next = c.next;
                s.window = c.window;
                s.compressor = c.compressor;
                s.compressed.truncate(c.compressed_len);
                s.keys.truncate(c.keys_len);
            } else {
                self.layers.remove(&layer);
            }
            self.published = published;
            if let Some(s) = selection {
                self.selection = s;
            }
            if let Some(c) = candidates {
                self.candidates = c;
            }
        }
        result
    }
    fn attention_inner<P: Projection>(
        &mut self,
        layer: usize,
        position: usize,
        x: &[f32],
        w: &AttentionWeights<P>,
    ) -> Result<Vec<f32>> {
        require(
            layer < 40
                && w.window > 0
                && w.head_dim > 0
                && !w.sink.is_empty()
                && !w.output_a.is_empty(),
            "attention geometry",
        )?;
        let qr = rms_norm(&w.query_a.forward(x)?, &w.query_norm, w.eps)?;
        let mut query = w.query_b.forward(&qr)?;
        require(query.len() == w.head_dim * w.sink.len(), "query heads")?;
        for q in query.chunks_exact_mut(w.head_dim) {
            rotary(q, &w.frequencies, position, false)?;
        }
        let mut raw = rms_norm(&w.kv.forward(x)?, &w.kv_norm, w.eps)?;
        require(raw.len() == w.head_dim, "KV width")?;
        rotary(&mut raw, &w.frequencies, position, false)?;
        let state = self.layers.entry(layer).or_default();
        require(state.next == position, "layer position discontinuity")?;
        state.window.push(raw);
        if state.window.len() > w.window {
            state.window.remove(0);
        }
        state.next += 1;
        let mut rows = state.window.clone();
        let mut selected: Vec<_> = (0..rows.len()).collect();
        if let Some(owner) = owner(layer) {
            let ratio = if layer < 20 { 2 } else { 1 };
            let length = (position + 1) / ratio;
            if layer == owner {
                let cw = w.compressor.as_ref().ok_or("missing owner compressor")?;
                require(cw.ratio == ratio, "owner compression ratio")?;
                if let Some(mut latent) = state.compressor.push(cw, x, position, w.eps)? {
                    let iw = w.index.as_ref().ok_or("missing owner indexer")?;
                    let mut key = rms_norm(
                        &iw.key
                            .as_ref()
                            .ok_or("missing index key projection")?
                            .forward(&latent)?,
                        &iw.key_norm,
                        w.eps,
                    )?;
                    require(
                        key.len() == iw.dim && latent.len() == w.head_dim,
                        "compressed/key width",
                    )?;
                    let group_position = position + 1 - ratio;
                    rotary(&mut key, &w.frequencies, group_position, false)?;
                    rotary(&mut latent, &w.frequencies, group_position, false)?;
                    state.keys.push(key);
                    state.compressed.push(latent);
                }
                // Publish even when the pair is incomplete: the previous layer's
                // owner may have a same-sized but entirely different key cache.
                self.published = Some((owner, position));
            }
            require(
                self.published == Some((owner, position)),
                "stale KV/index owner",
            )?;
            let source = self.layers.get(&owner).ok_or("missing owner state")?;
            require(
                source.keys.len() == length && source.compressed.len() == length,
                "owner cache length",
            )?;
            if producer(layer) == Some(layer) {
                let iw = w.index.as_ref().ok_or("missing selection indexer")?;
                require(iw.heads > 0 && iw.dim > 0, "index geometry")?;
                let mut iq = iw.query.forward(&qr)?;
                let weights = iw.weights.forward(x)?;
                require(
                    iq.len() == iw.heads * iw.dim
                        && weights.len() == iw.heads
                        && source.keys.iter().all(|k| k.len() == iw.dim),
                    "index projection shape",
                )?;
                for q in iq.chunks_exact_mut(iw.dim) {
                    rotary(q, &w.frequencies, position, false)?;
                }
                let scale = (iw.dim as f32).sqrt().recip() * (iw.heads as f32).sqrt().recip();
                let mut scores: Vec<_> = source
                    .keys
                    .iter()
                    .map(|k| {
                        iq.chunks_exact(iw.dim)
                            .zip(&weights)
                            .map(|(q, v)| dot(q, k).max(0.0) * (v * scale))
                            .sum()
                    })
                    .collect();
                if layer == 20 {
                    self.candidates = Some((
                        position,
                        candidate_blocks(&scores, w.candidate_block_size, w.candidate_topk_blocks)?,
                    ));
                }
                if layer > 20 {
                    let (stamp, mask) = self.candidates.as_ref().ok_or("missing candidates")?;
                    require(
                        *stamp == position && mask.len() == scores.len(),
                        "stale candidates",
                    )?;
                    for (s, keep) in scores.iter_mut().zip(mask) {
                        if !keep {
                            *s = f32::NEG_INFINITY;
                        }
                    }
                }
                // Match reference topk even when masked candidates are fewer than k.
                self.selection = Some((layer, position, topk(&scores, iw.topk)));
            }
            let (selection_owner, stamp, indices) =
                self.selection.as_ref().ok_or("missing selection")?;
            require(
                Some(*selection_owner) == producer(layer) && *stamp == position,
                "stale selection",
            )?;
            for &i in indices {
                let row = source.compressed.get(i).ok_or("selection outside cache")?;
                selected.push(rows.len());
                rows.push(row.clone());
            }
        }
        let mut output = Vec::new();
        for (q, sink) in query.chunks_exact(w.head_dim).zip(&w.sink) {
            let mut o = sparse_attention(q, &rows, &selected, *sink)?;
            rotary(&mut o, &w.frequencies, position, true)?;
            output.extend(o);
        }
        let low = P::forward_grouped(&w.output_a, &output)?;
        w.output_b.forward(&low)
    }

    /// Token-major continuation, including arbitrary odd/even chunk boundaries.
    /// Each row traverses all supplied layers before the next row begins.
    pub fn decode_rows<P: Projection>(
        &mut self,
        start: usize,
        rows: &[Vec<f32>],
        layers: &[(usize, AttentionWeights<P>)],
    ) -> Result<Vec<Vec<f32>>> {
        let mut next = self.clone();
        let mut result = Vec::new();
        for (offset, row) in rows.iter().enumerate() {
            let mut x = row.clone();
            for (layer, w) in layers {
                x = next.attention_inner(*layer, start + offset, &x, w)?;
            }
            result.push(x);
        }
        *self = next;
        Ok(result)
    }
}
