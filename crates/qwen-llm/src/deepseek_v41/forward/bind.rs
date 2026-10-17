use super::*;
use attention::{AttentionWeights, CompressorWeights, IndexWeights};

pub(super) struct Binder<'a> {
    pub(super) ctx: &'a MetalContext,
    pub(super) weights: &'a std::collections::BTreeMap<String, MetalTensor>,
}
impl<'a> Binder<'a> {
    fn tensor(&self, name: &str, shape: &[usize]) -> Result<&'a MetalTensor> {
        let t = self
            .weights
            .get(name)
            .ok_or_else(|| format!("missing dense tensor {name}"))?;
        let expected: Vec<_> = shape.iter().rev().map(|&n| n as u64).collect();
        op::require(
            t.shape == expected,
            &format!("shape mismatch {name}: {:?} != {expected:?}", t.shape),
        )?;
        f32_slice(t)?;
        Ok(t)
    }
    fn vector(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        Ok(f32_slice(self.tensor(name, &[n])?)?.to_vec())
    }
    pub(super) fn linear(&self, name: &str, input: usize, output: usize) -> Result<Linear<'a>> {
        Ok(Linear {
            ctx: self.ctx,
            tensor: self.tensor(name, &[output, input])?.clone(),
            input,
            output,
        })
    }
    pub(super) fn hc(&self, layer: usize, kind: &str, c: &Config) -> Result<Hc<'a>> {
        let p = format!("layers.{layer}.hc_{kind}");
        Ok(Hc {
            projection: self.linear(&format!("{p}_fn"), 4 * c.hidden, 24)?,
            scale: self.vector(&format!("{p}_scale"), 3)?.try_into().unwrap(),
            base: self.vector(&format!("{p}_base"), 24)?,
            norm: self.vector(&format!("layers.{layer}.{kind}_norm.weight"), c.hidden)?,
        })
    }
    pub(super) fn attention(&self, layer: usize, c: &Config) -> Result<AttentionWeights<'a>> {
        let p = format!("layers.{layer}.attn");
        let lin =
            |name: &str, input, output| self.linear(&format!("{p}.{name}.weight"), input, output);
        let vec = |name: &str, n| self.vector(&format!("{p}.{name}"), n);
        let owner = [2, 8, 14, 20].contains(&layer);
        let index = [2, 8, 14, 20, 24, 28, 32, 36].contains(&layer);
        let group_input = c.heads * c.head_dim / c.groups;
        let output_a = lin("wo_a", group_input, c.groups * c.output_rank)?;
        let output_a = (0..c.groups)
            .map(|g| Linear {
                ctx: self.ctx,
                tensor: output_a.tensor.view_subrange(
                    (g * c.output_rank * group_input) as u64,
                    vec![group_input as u64, c.output_rank as u64],
                ),
                input: group_input,
                output: c.output_rank,
            })
            .collect();
        Ok(AttentionWeights {
            query_a: lin("wq_a", c.hidden, c.query_rank)?,
            query_norm: vec("q_norm.weight", c.query_rank)?,
            query_b: lin("wq_b", c.query_rank, c.heads * c.head_dim)?,
            kv: lin("wkv", c.hidden, c.head_dim)?,
            kv_norm: vec("kv_norm.weight", c.head_dim)?,
            sink: vec("attn_sink", c.heads)?,
            output_a,
            output_b: lin("wo_b", c.groups * c.output_rank, c.hidden)?,
            compressor: if owner {
                Some(CompressorWeights {
                    ratio: if layer < 20 { 2 } else { 1 },
                    kv: lin("compressor.wkv", c.hidden, c.head_dim)?,
                    gate: if layer < 20 {
                        Some(lin("compressor.wgate", c.hidden, c.head_dim)?)
                    } else {
                        None
                    },
                    norm: vec("compressor.norm.weight", c.head_dim)?,
                })
            } else {
                None
            },
            index: if index {
                Some(IndexWeights {
                    query: lin("indexer.wq_b", c.query_rank, c.index_heads * c.index_dim)?,
                    weights: lin("indexer.weights_proj", c.hidden, c.index_heads)?,
                    key: if owner {
                        Some(lin("indexer.wk", c.head_dim, c.index_dim)?)
                    } else {
                        None
                    },
                    key_norm: if owner {
                        vec("indexer.k_norm.weight", c.index_dim)?
                    } else {
                        Vec::new()
                    },
                    heads: c.index_heads,
                    dim: c.index_dim,
                    topk: c.index_topk,
                })
            } else {
                None
            },
            frequencies: op::rotary_frequencies(
                c.rope_dim,
                if layer < 2 {
                    c.rope_theta
                } else {
                    c.compress_theta
                },
                if layer < 2 { 0 } else { c.original_context },
                c.rope_factor,
                c.beta_fast,
                c.beta_slow,
            )?,
            head_dim: c.head_dim,
            window: c.window,
            eps: c.norm_eps,
            candidate_block_size: c.candidate_block,
            candidate_topk_blocks: c.candidate_topk,
        })
    }
}

impl<'a> Forward<'a> {
    /// The map MUST be generated by the official tokenizer's normalization
    /// pipeline (see build_compressed_token_map), not an identity/raw-ID map.
    /// raw_pad is normally 2; it is translated through the same supplied map.
    pub fn from_resident(
        ctx: &'a MetalContext,
        resident: &'a ResidentWeights,
        token_map: Vec<u32>,
        raw_pad: u32,
    ) -> Result<Self> {
        Self::from_resident_with_config(ctx, resident, token_map, raw_pad, Config::default())
    }

    pub fn from_resident_with_config(
        ctx: &'a MetalContext,
        resident: &'a ResidentWeights,
        token_map: Vec<u32>,
        raw_pad: u32,
        config: Config,
    ) -> Result<Self> {
        let c = &config;
        for n in [
            c.hidden,
            c.intermediate,
            c.vocab,
            c.heads,
            c.head_dim,
            c.query_rank,
            c.output_rank,
            c.groups,
            c.experts,
            c.topk,
            c.index_heads,
            c.index_dim,
            c.index_topk,
            c.window,
            c.candidate_block,
            c.candidate_topk,
            c.hc_iterations,
        ] {
            op::require(n > 0 && n <= u32::MAX as usize, "invalid forward dimension")?;
        }
        op::require(
            c.heads % c.groups == 0
                && c.topk <= c.experts
                && c.topk <= 16
                && c.hidden % 32 == 0
                && c.intermediate % 32 == 0
                && c.rope_dim <= c.head_dim
                && c.rope_dim <= c.index_dim,
            "invalid attention/expert geometry",
        )?;
        for width in [
            c.heads.checked_mul(c.head_dim),
            c.groups.checked_mul(c.output_rank),
            c.index_heads.checked_mul(c.index_dim),
            c.hidden.checked_mul(5),
        ] {
            op::require(
                width.is_some_and(|n| n <= u32::MAX as usize),
                "projection width overflow",
            )?;
        }
        op::require(c.experts <= i32::MAX as usize, "expert ID overflow")?;
        op::require(
            c.norm_eps.is_finite()
                && c.norm_eps > 0.0
                && c.hc_eps.is_finite()
                && c.hc_eps > 0.0
                && c.clamp.is_finite()
                && c.clamp > 0.0
                && c.gate_temperature.is_finite()
                && c.gate_temperature > 0.0
                && c.route_scale.is_finite(),
            "invalid forward scalar",
        )?;
        op::require(
            token_map.len() == c.vocab
                && token_map
                    .iter()
                    .all(|&id| id < engram::COMPRESSED_VOCAB_SIZE),
            "compressed token map shape/range",
        )?;
        let pad = *token_map
            .get(raw_pad as usize)
            .ok_or("pad outside raw vocabulary")?;
        if c.vocab == 129280 {
            let mut seen = vec![false; engram::COMPRESSED_VOCAB_SIZE as usize];
            for &id in &token_map {
                seen[id as usize] = true;
            }
            op::require(
                seen.iter().all(|&v| v),
                "compressed map must cover exactly 99092 IDs",
            )?;
        }
        let b = Binder {
            ctx,
            weights: &resident.dense,
        };
        let mut layers = Vec::with_capacity(40);
        for id in 0..40 {
            let p = format!("layers.{id}.ffn");
            let mut banks = Vec::new();
            for (part, input, output) in [
                ("w1", c.hidden, c.intermediate),
                ("w3", c.hidden, c.intermediate),
                ("w2", c.intermediate, c.hidden),
            ] {
                let t = resident
                    .bank(id, part)
                    .ok_or_else(|| format!("missing expert bank {id}/{part}"))?;
                let expected = vec![input as u64, output as u64, c.experts as u64];
                let bytes = (input as u64 / 32)
                    .checked_mul(17)
                    .and_then(|n| n.checked_mul(output as u64))
                    .and_then(|n| n.checked_mul(c.experts as u64))
                    .and_then(|n| n.checked_add(t.offset))
                    .ok_or("expert extent overflow")?;
                op::require(
                    t.dtype == GgmlType::MXFP4
                        && t.shape == expected
                        && t.offset % 16 == 0
                        && bytes <= t.buffer.length() as u64,
                    &format!("invalid expert bank {id}/{part}"),
                )?;
                banks.push(t);
            }
            let engram = if [1, 14].contains(&id) {
                let p = format!("layers.{id}.engram");
                Some(Engram {
                    projection: b.linear(&format!("{p}.wkv.weight"), 6144, 5 * c.hidden)?,
                    q: f32_slice(b.tensor(&format!("{p}.q_weight"), &[4, c.hidden])?)?.to_vec(),
                    k: f32_slice(b.tensor(&format!("{p}.k_weight"), &[4, c.hidden])?)?.to_vec(),
                })
            } else {
                None
            };
            layers.push(Layer {
                attention: b.attention(id, c)?,
                attn_hc: b.hc(id, "attn", c)?,
                ffn_hc: b.hc(id, "ffn", c)?,
                router: b.linear(&format!("{p}.gate.weight"), c.hidden, c.experts)?,
                bias: b.vector(&format!("{p}.gate.bias"), c.experts)?,
                shared: [
                    b.linear(
                        &format!("{p}.shared_experts.w1.weight"),
                        c.hidden,
                        c.intermediate,
                    )?,
                    b.linear(
                        &format!("{p}.shared_experts.w3.weight"),
                        c.hidden,
                        c.intermediate,
                    )?,
                    b.linear(
                        &format!("{p}.shared_experts.w2.weight"),
                        c.intermediate,
                        c.hidden,
                    )?,
                ],
                banks: [banks[0], banks[1], banks[2]],
                engram,
            });
        }
        let embedding = b.tensor("embed.weight", &[c.vocab, c.hidden])?;
        let norm = b.vector("norm.weight", c.hidden)?;
        let head = b.linear("head.weight", c.hidden, c.vocab)?;
        Ok(Self {
            ctx,
            engram_rows: EngramRows::Disk(&resident.engram),
            config,
            token_map,
            pad,
            layers,
            embedding,
            norm,
            head,
            identity: Arc::new(()),
        })
    }
}
