//! Projection descriptors shared by resident binding and metadata-only planning.
use super::*;
use crate::deepseek_v41::gguf::Schedule;

pub(in crate::deepseek_v41) struct LinearSpec {
    pub name: String,
    pub input: usize,
    pub output: usize,
}

impl LinearSpec {
    fn new(name: String, input: usize, output: usize) -> Self {
        Self {
            name,
            input,
            output,
        }
    }

    pub(super) fn head(c: &Config) -> Self {
        Self::new("head.weight".into(), c.hidden, c.vocab)
    }

    pub(super) fn hc(layer: usize, kind: &str, c: &Config) -> Self {
        Self::new(format!("layers.{layer}.hc_{kind}_fn"), 4 * c.hidden, 24)
    }
}

pub(super) struct LayerLinears {
    pub query_a: LinearSpec,
    pub query_b: LinearSpec,
    pub kv: LinearSpec,
    pub output_a: LinearSpec,
    pub output_b: LinearSpec,
    pub compressor_kv: Option<LinearSpec>,
    pub compressor_gate: Option<LinearSpec>,
    pub index_query: Option<LinearSpec>,
    pub index_weights: Option<LinearSpec>,
    pub index_key: Option<LinearSpec>,
    pub attn_hc: LinearSpec,
    pub ffn_hc: LinearSpec,
    pub router: LinearSpec,
    pub shared: [LinearSpec; 3],
    pub engram: Option<LinearSpec>,
}

impl LayerLinears {
    pub fn new(c: &Config, schedule: &Schedule, layer: usize, engram: bool) -> Self {
        let p = format!("layers.{layer}");
        let attn =
            |name, input, output| LinearSpec::new(format!("{p}.attn.{name}.weight"), input, output);
        let owner = schedule.owns_kv(layer);
        let index = schedule.produces_index(layer);
        Self {
            query_a: attn("wq_a", c.hidden, c.query_rank),
            query_b: attn("wq_b", c.query_rank, c.heads * c.head_dim),
            kv: attn("wkv", c.hidden, c.head_dim),
            // All grouped output views share this one binding/control allocation.
            output_a: attn(
                "wo_a",
                c.heads * c.head_dim / c.groups,
                c.groups * c.output_rank,
            ),
            output_b: attn("wo_b", c.groups * c.output_rank, c.hidden),
            compressor_kv: owner.then(|| attn("compressor.wkv", c.hidden, c.head_dim)),
            compressor_gate: (owner && schedule.ratio(layer) > 1)
                .then(|| attn("compressor.wgate", c.hidden, c.head_dim)),
            index_query: index
                .then(|| attn("indexer.wq_b", c.query_rank, c.index_heads * c.index_dim)),
            index_weights: index.then(|| attn("indexer.weights_proj", c.hidden, c.index_heads)),
            index_key: (index && owner).then(|| attn("indexer.wk", c.head_dim, c.index_dim)),
            attn_hc: LinearSpec::hc(layer, "attn", c),
            ffn_hc: LinearSpec::hc(layer, "ffn", c),
            router: LinearSpec::new(format!("{p}.ffn.gate.weight"), c.hidden, c.experts),
            shared: [
                ("w1", c.hidden, c.intermediate),
                ("w3", c.hidden, c.intermediate),
                ("w2", c.intermediate, c.hidden),
            ]
            .map(|(part, input, output)| {
                LinearSpec::new(
                    format!("{p}.ffn.shared_experts.{part}.weight"),
                    input,
                    output,
                )
            }),
            engram: engram
                .then(|| LinearSpec::new(format!("{p}.engram.wkv.weight"), 6144, 5 * c.hidden)),
        }
    }

    pub fn into_specs(self) -> impl Iterator<Item = LinearSpec> {
        [
            Some(self.query_a),
            Some(self.query_b),
            Some(self.kv),
            Some(self.output_a),
            Some(self.output_b),
            self.compressor_kv,
            self.compressor_gate,
            self.index_query,
            self.index_weights,
            self.index_key,
            Some(self.attn_hc),
            Some(self.ffn_hc),
            Some(self.router),
            self.engram,
        ]
        .into_iter()
        .flatten()
        .chain(self.shared)
    }
}

pub(in crate::deepseek_v41) fn linear_inventory(
    c: &Config,
    schedule: &Schedule,
    hash: &engram::HashConfig,
) -> impl Iterator<Item = LinearSpec> {
    let mut specs = vec![LinearSpec::head(c)];
    for layer in 0..schedule.layers {
        specs.extend(
            LayerLinears::new(
                c,
                schedule,
                layer,
                hash.layers.iter().any(|h| h.layer == layer),
            )
            .into_specs(),
        );
    }
    specs.into_iter()
}

pub(super) const NATIVE_CONTROL_INITIAL: [i32; 2] = [0, metal::ROUTE_STATUS_READY];

pub(super) fn native_control_bytes(dtype: GgmlType) -> Result<u64> {
    op::require(
        dense_linear_supported(dtype),
        &format!("unsupported NativeF32 linear storage: {dtype:?}"),
    )?;
    Ok(
        if matches!(
            dtype,
            GgmlType::F32
                | GgmlType::F16
                | GgmlType::BF16
                | GgmlType::IQ2_XXS
                | GgmlType::IQ1_S
                | GgmlType::IQ1_M
        ) {
            0
        } else {
            std::mem::size_of_val(&NATIVE_CONTROL_INITIAL) as u64
        },
    )
}
