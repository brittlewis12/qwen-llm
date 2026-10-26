use super::*;
use attention::{AttentionWeights, CompressorWeights, IndexWeights};
use binding::{LayerLinears, LinearSpec};

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
    fn linear_spec(&self, spec: &LinearSpec) -> Result<Linear<'a>> {
        self.linear(&spec.name, spec.input, spec.output)
    }
    #[cfg(test)]
    pub(super) fn hc(&self, layer: usize, kind: &str, c: &Config) -> Result<Hc<'a>> {
        self.hc_with_linear(layer, kind, c, &LinearSpec::hc(layer, kind, c))
    }
    fn hc_with_linear(
        &self,
        layer: usize,
        kind: &str,
        c: &Config,
        projection: &LinearSpec,
    ) -> Result<Hc<'a>> {
        let p = format!("layers.{layer}.hc_{kind}");
        Ok(Hc {
            projection: self.linear_spec(projection)?,
            scale: self.vector(&format!("{p}_scale"), 3)?.try_into().unwrap(),
            base: self.vector(&format!("{p}_base"), 24)?,
            norm: self.vector(&format!("layers.{layer}.{kind}_norm.weight"), c.hidden)?,
        })
    }
    #[cfg(test)]
    pub(super) fn attention(&self, layer: usize, c: &Config) -> Result<AttentionWeights<'a>> {
        self.attention_with_schedule(layer, c, &super::super::gguf::Schedule::default())
    }
    #[cfg(test)]
    pub(super) fn attention_with_schedule(
        &self,
        layer: usize,
        c: &Config,
        schedule: &super::super::gguf::Schedule,
    ) -> Result<AttentionWeights<'a>> {
        self.attention_with_linears(
            layer,
            c,
            schedule,
            &LayerLinears::new(c, schedule, layer, false),
        )
    }
    fn attention_with_linears(
        &self,
        layer: usize,
        c: &Config,
        schedule: &super::super::gguf::Schedule,
        linears: &LayerLinears,
    ) -> Result<AttentionWeights<'a>> {
        let p = format!("layers.{layer}.attn");
        let lin = |spec: &LinearSpec| self.linear_spec(spec);
        let optional = |spec: &Option<LinearSpec>| spec.as_ref().map(lin).transpose();
        let vec = |name: &str, n| self.vector(&format!("{p}.{name}"), n);
        let ratio = schedule.ratio(layer);
        let group_input = c.heads * c.head_dim / c.groups;
        let output_a = lin(&linears.output_a)?;
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
            query_a: lin(&linears.query_a)?,
            query_norm: vec("q_norm.weight", c.query_rank)?,
            query_b: lin(&linears.query_b)?,
            kv: lin(&linears.kv)?,
            kv_norm: vec("kv_norm.weight", c.head_dim)?,
            sink: vec("attn_sink", c.heads)?,
            output_a,
            output_b: lin(&linears.output_b)?,
            compressor: if let Some(kv) = &linears.compressor_kv {
                Some(CompressorWeights {
                    ratio,
                    kv: lin(kv)?,
                    gate: optional(&linears.compressor_gate)?,
                    norm: vec("compressor.norm.weight", c.head_dim)?,
                })
            } else {
                None
            },
            index: if let Some(query) = &linears.index_query {
                Some(IndexWeights {
                    query: lin(query)?,
                    weights: lin(linears
                        .index_weights
                        .as_ref()
                        .expect("index query has weights"))?,
                    key: optional(&linears.index_key)?,
                    key_norm: if linears.index_key.is_some() {
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
            let linears = LayerLinears::new(
                c,
                &resident.schedule,
                id,
                resident.engram_hash.layers.iter().any(|h| h.layer == id),
            );
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
            let engram = if let Some(projection) = &linears.engram {
                let p = format!("layers.{id}.engram");
                Some(Engram {
                    projection: b.linear_spec(projection)?,
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
                attention: b.attention_with_linears(id, c, &resident.schedule, &linears)?,
                attn_hc: b.hc_with_linear(id, "attn", c, &linears.attn_hc)?,
                ffn_hc: b.hc_with_linear(id, "ffn", c, &linears.ffn_hc)?,
                router: b.linear_spec(&linears.router)?,
                bias: b.vector(&format!("layers.{id}.ffn.gate.bias"), c.experts)?,
                shared: [
                    b.linear_spec(&linears.shared[0])?,
                    b.linear_spec(&linears.shared[1])?,
                    b.linear_spec(&linears.shared[2])?,
                ],
                banks: [banks[0], banks[1], banks[2]],
                engram,
            });
        }
        let embedding = b.tensor("embed.weight", &[c.vocab, c.hidden])?;
        let norm = b.vector("norm.weight", c.hidden)?;
        let head = b.linear_spec(&LinearSpec::head(c))?;
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
