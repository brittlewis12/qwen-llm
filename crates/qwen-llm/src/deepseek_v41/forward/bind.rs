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
        let rows = t
            .shape
            .iter()
            .skip(1)
            .try_fold(1u64, |n, &d| n.checked_mul(d))
            .ok_or("tensor extent overflow")?;
        let end = row_bytes(t)?
            .checked_mul(rows)
            .and_then(|n| n.checked_add(t.offset))
            .ok_or("tensor extent overflow")?;
        op::require(
            end <= t.buffer.length() as u64,
            "tensor outside resident buffer",
        )?;
        Ok(t)
    }
    fn vector(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        decode_rows(self.tensor(name, &[n])?, 0, 1)
    }
    pub(super) fn linear(&self, name: &str, input: usize, output: usize) -> Result<Linear<'a>> {
        let tensor = self.tensor(name, &[output, input])?;
        op::require(
            dense_linear_supported(tensor.dtype),
            "unsupported linear storage",
        )?;
        Ok(Linear {
            ctx: self.ctx,
            tensor: tensor.clone(),
            native_control: native_control(self.ctx, tensor.dtype)?,
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
    #[cfg(test)]
    pub(super) fn attention(&self, layer: usize, c: &Config) -> Result<AttentionWeights<'a>> {
        self.attention_with_schedule(layer, c, &super::super::gguf::Schedule::default())
    }
    pub(super) fn attention_with_schedule(
        &self,
        layer: usize,
        c: &Config,
        schedule: &super::super::gguf::Schedule,
    ) -> Result<AttentionWeights<'a>> {
        let p = format!("layers.{layer}.attn");
        let lin =
            |name: &str, input, output| self.linear(&format!("{p}.{name}.weight"), input, output);
        let vec = |name: &str, n| self.vector(&format!("{p}.{name}"), n);
        let owner = schedule.owns_kv(layer);
        let index = schedule.produces_index(layer);
        let ratio = schedule.ratio(layer);
        let group_input = c.heads * c.head_dim / c.groups;
        let output_a = lin("wo_a", group_input, c.groups * c.output_rank)?;
        let stride = row_bytes(&output_a.tensor)?;
        let output_a = (0..c.groups)
            .map(|g| Linear {
                ctx: self.ctx,
                native_control: output_a.native_control.clone(),
                tensor: output_a.tensor.view_bytes(
                    g as u64 * c.output_rank as u64 * stride,
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
                    ratio,
                    kv: lin("compressor.wkv", c.hidden, c.head_dim)?,
                    gate: if ratio > 1 {
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
                if ratio == 0 {
                    c.rope_theta
                } else {
                    c.compress_theta
                },
                if ratio == 0 { 0 } else { c.original_context },
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
        Self::from_resident_with_config(ctx, resident, token_map, raw_pad, resident.config.clone())
    }

    pub fn from_resident_with_config(
        ctx: &'a MetalContext,
        resident: &'a ResidentWeights,
        token_map: Vec<u32>,
        raw_pad: u32,
        config: Config,
    ) -> Result<Self> {
        let c = &config;
        resident.schedule.validate().map_err(|e| e.to_string())?;
        resident
            .engram_hash
            .validate()
            .map_err(|e| format!("{e:?}"))?;
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
                && c.rope_dim <= c.head_dim
                && c.rope_dim <= c.index_dim,
            "invalid attention/expert geometry",
        )?;
        super::super::execution::validate(c).map_err(|e| e.to_string())?;
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
                    .all(|&id| id < resident.engram_hash.compressed_vocab_size),
            "compressed token map shape/range",
        )?;
        let pad = *token_map
            .get(raw_pad as usize)
            .ok_or("pad outside raw vocabulary")?;
        let b = Binder {
            ctx,
            weights: &resident.dense,
        };
        let mut layers = Vec::with_capacity(resident.schedule.layers);
        for id in 0..resident.schedule.layers {
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
                let bytes = row_bytes(t)?
                    .checked_mul(output as u64)
                    .and_then(|n| n.checked_mul(c.experts as u64))
                    .and_then(|n| n.checked_add(t.offset))
                    .ok_or("expert extent overflow")?;
                op::require(
                    metal::all_slots_down_native_supported(t.dtype)
                        && t.shape == expected
                        && t.offset % 16 == 0
                        && bytes <= t.buffer.length() as u64,
                    &format!("invalid expert bank {id}/{part}"),
                )?;
                banks.push(t);
            }
            let engram = if resident.engram_hash.layers.iter().any(|p| p.layer == id) {
                let p = format!("layers.{id}.engram");
                Some(Engram {
                    projection: b.linear(&format!("{p}.wkv.weight"), 6144, 5 * c.hidden)?,
                    gate: if let Some(t) = b.weights.get(&format!("{p}.gate.weight")) {
                        op::require(
                            t.shape == vec![c.hidden as u64, 4],
                            "folded Engram gate shape",
                        )?;
                        EngramGate::Folded(decode_rows(t, 0, 4)?)
                    } else {
                        EngramGate::Separate {
                            q: decode_rows(
                                b.tensor(&format!("{p}.q_weight"), &[4, c.hidden])?,
                                0,
                                4,
                            )?,
                            k: decode_rows(
                                b.tensor(&format!("{p}.k_weight"), &[4, c.hidden])?,
                                0,
                                4,
                            )?,
                        }
                    },
                })
            } else {
                None
            };
            layers.push(Layer {
                attention: b.attention_with_schedule(id, c, &resident.schedule)?,
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
            engram_hash: resident.engram_hash.clone(),
            schedule: resident.schedule.clone(),
            config,
            token_map,
            pad,
            layers,
            embedding,
            norm,
            head,
            prefill_rows: super::prefill_rows_for_resident(resident.bytes.resident_bytes()),
            identity: Arc::new(()),
        })
    }
}
