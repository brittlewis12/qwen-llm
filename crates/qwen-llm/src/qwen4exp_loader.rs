//! Strict GGUF tensor-name and shape binding for Qwen3.8-Flash-Next.
//!
//! Backend-specific dtype qualification remains the responsibility of the
//! executable weight loader. This layer rejects non-weight storage and
//! malformed quantized row geometry.

use crate::gguf::GgufFile;
use crate::qwen4exp::{MixerKind, Qwen4ExpConfig, Qwen4ExpError};
use crate::tensor::{GgmlType, TensorDesc};
use std::collections::HashSet;

#[derive(Clone)]
pub struct GatedResidualWeights<'a> {
    pub norm: &'a TensorDesc,
    pub down: &'a TensorDesc,
    pub up: &'a TensorDesc,
    pub inject: &'a TensorDesc,
}

#[derive(Clone)]
pub struct FinalGatedResidualWeights<'a> {
    pub norm: &'a TensorDesc,
    pub down: &'a TensorDesc,
    pub up: &'a TensorDesc,
}

#[derive(Clone)]
pub struct GatedDeltaNetWeights<'a> {
    pub qkv: &'a TensorDesc,
    pub gate: &'a TensorDesc,
    pub beta: &'a TensorDesc,
    pub alpha: &'a TensorDesc,
    pub a: &'a TensorDesc,
    pub dt_bias: &'a TensorDesc,
    pub conv: &'a TensorDesc,
    pub norm: &'a TensorDesc,
    pub output: &'a TensorDesc,
}

#[derive(Clone)]
pub struct QwenSparseAttentionWeights<'a> {
    pub query_gate: &'a TensorDesc,
    pub key: &'a TensorDesc,
    pub value: &'a TensorDesc,
    pub output: &'a TensorDesc,
    pub query_norm: &'a TensorDesc,
    pub key_norm: &'a TensorDesc,
    pub index_query: &'a TensorDesc,
    pub index_key: &'a TensorDesc,
    pub index_query_norm: &'a TensorDesc,
    pub index_key_norm: &'a TensorDesc,
}

#[derive(Clone)]
pub enum MixerWeights<'a> {
    GatedDeltaNet(GatedDeltaNetWeights<'a>),
    QwenSparseAttention(QwenSparseAttentionWeights<'a>),
}

#[derive(Clone)]
pub struct MoeWeights<'a> {
    pub router: &'a TensorDesc,
    pub routed_gate: &'a TensorDesc,
    pub routed_up: &'a TensorDesc,
    pub routed_down: &'a TensorDesc,
    pub shared_router: &'a TensorDesc,
    pub shared_gate: &'a TensorDesc,
    pub shared_up: &'a TensorDesc,
    pub shared_down: &'a TensorDesc,
}

#[derive(Clone)]
pub struct PleLayerWeights<'a> {
    pub key: &'a TensorDesc,
    pub value: &'a TensorDesc,
    pub key_norm: &'a TensorDesc,
    pub query_norm: &'a TensorDesc,
    pub conv_norm: &'a TensorDesc,
    pub conv: &'a TensorDesc,
}

#[derive(Clone)]
pub struct Qwen4ExpBlock<'a> {
    pub attention_residual: GatedResidualWeights<'a>,
    pub mixer: MixerWeights<'a>,
    pub ffn_residual: GatedResidualWeights<'a>,
    pub moe: MoeWeights<'a>,
    pub ple: Option<PleLayerWeights<'a>>,
}

pub struct Qwen4ExpModel<'a> {
    pub config: Qwen4ExpConfig,
    pub token_embedding: &'a TensorDesc,
    pub output: &'a TensorDesc,
    pub tied_embeddings: bool,
    pub final_residual: FinalGatedResidualWeights<'a>,
    pub ple_embedding: Option<&'a TensorDesc>,
    pub blocks: Vec<Qwen4ExpBlock<'a>>,
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpLoadError {
    #[error(transparent)]
    Config(#[from] Qwen4ExpError),
    #[error("missing required tensor: {0}")]
    Missing(String),
    #[error("shape mismatch for {name}: expected {expected:?}, got {got:?}")]
    Shape {
        name: String,
        expected: Vec<u64>,
        got: Vec<u64>,
    },
    #[error("tensor {name} uses unsupported or unknown GGML dtype {dtype:?}")]
    UnknownDtype { name: String, dtype: GgmlType },
    #[error("metadata-derived tensor dimension overflow: {0}")]
    DimensionOverflow(&'static str),
    #[error(
        "tensor {name} row width {row_width} is not aligned to {block_size}-element {dtype:?} blocks"
    )]
    RowAlignment {
        name: String,
        row_width: u64,
        block_size: u64,
        dtype: GgmlType,
    },
    #[error("unexpected qwen4exp tensor set: {0:?}")]
    UnexpectedTensors(Vec<String>),
}

impl<'a> Qwen4ExpModel<'a> {
    pub fn from_gguf(gguf: &'a GgufFile) -> Result<Self, Qwen4ExpLoadError> {
        let config = Qwen4ExpConfig::from_gguf(gguf)?;
        let dims = Dimensions::new(&config)?;
        let mut used = HashSet::with_capacity(gguf.tensors.len());

        let token_embedding = bind(
            gguf,
            &mut used,
            "token_embd.weight".into(),
            &[dims.hidden, dims.vocab],
        )?;
        let (output, tied_embeddings) = if gguf.find("output.weight").is_some() {
            (
                bind(
                    gguf,
                    &mut used,
                    "output.weight".into(),
                    &[dims.hidden, dims.vocab],
                )?,
                false,
            )
        } else {
            (token_embedding, true)
        };
        let final_residual = FinalGatedResidualWeights {
            norm: bind(
                gguf,
                &mut used,
                "output_hc_norm.weight".into(),
                &[dims.hyper_hidden],
            )?,
            down: bind(
                gguf,
                &mut used,
                "output_hc_down.weight".into(),
                &[dims.hyper_hidden, dims.hyper_rank],
            )?,
            up: bind(
                gguf,
                &mut used,
                "output_hc_up.weight".into(),
                &[dims.hyper_rank, dims.hyper_hidden],
            )?,
        };

        let ple_embedding = if let Some(ple) = &config.ple {
            let logical_rows = ple
                .head_offsets
                .last()
                .zip(ple.head_vocab_sizes.last())
                .and_then(|(&offset, &size)| offset.checked_add(size))
                .ok_or(Qwen4ExpLoadError::DimensionOverflow("PLE table rows"))?;
            Some(bind_ple_embedding(
                gguf,
                &mut used,
                ple.embedding_head_dim as u64,
                logical_rows,
            )?)
        } else {
            None
        };

        let mut blocks = Vec::with_capacity(config.layer_count as usize);
        for layer in 0..config.layer_count {
            let attention_residual = bind_residual(gguf, &mut used, layer, "attn", &dims)?;
            let mixer = match config
                .mixer_kind(layer)
                .expect("validated layer index must have a mixer kind")
            {
                MixerKind::GatedDeltaNet => {
                    MixerWeights::GatedDeltaNet(bind_gdn(gguf, &mut used, layer, &dims)?)
                }
                MixerKind::QwenSparseAttention => {
                    MixerWeights::QwenSparseAttention(bind_qsa(gguf, &mut used, layer, &dims)?)
                }
            };
            let ffn_residual = bind_residual(gguf, &mut used, layer, "ffn", &dims)?;
            let moe = bind_moe(gguf, &mut used, layer, &dims)?;
            let ple = if config
                .ple
                .as_ref()
                .is_some_and(|ple| ple.layers.contains(&layer))
            {
                Some(bind_ple(gguf, &mut used, layer, &dims)?)
            } else {
                None
            };
            blocks.push(Qwen4ExpBlock {
                attention_residual,
                mixer,
                ffn_residual,
                moe,
                ple,
            });
        }

        let mut unexpected: Vec<_> = gguf
            .tensors
            .iter()
            .filter(|tensor| !used.contains(tensor.name.as_str()))
            .map(|tensor| tensor.name.clone())
            .collect();
        unexpected.sort();
        if !unexpected.is_empty() {
            return Err(Qwen4ExpLoadError::UnexpectedTensors(unexpected));
        }

        Ok(Self {
            config,
            token_embedding,
            output,
            tied_embeddings,
            final_residual,
            ple_embedding,
            blocks,
        })
    }
}

struct Dimensions {
    hidden: u64,
    vocab: u64,
    hyper_count: u64,
    hyper_hidden: u64,
    hyper_rank: u64,
    attention_q: u64,
    attention_kv: u64,
    attention_head: u64,
    gdn_qkv: u64,
    gdn_value: u64,
    gdn_heads: u64,
    gdn_conv: u64,
    index_query: u64,
    index_key: u64,
    index_head: u64,
    experts: u64,
    expert_ffn: u64,
    shared_ffn: u64,
    ple_conv: u64,
}

impl Dimensions {
    fn new(config: &Qwen4ExpConfig) -> Result<Self, Qwen4ExpLoadError> {
        let hidden = config.hidden_size as u64;
        let hyper_count = config.hyper_connection.count as u64;
        let hyper_hidden = checked_mul(hidden, hyper_count, "hyper-connection width")?;
        let attention_q = checked_mul(
            config.attention.query_heads as u64,
            config.attention.key_head_dim as u64,
            "attention query width",
        )?;
        let attention_kv = checked_mul(
            config.attention.kv_heads as u64,
            config.attention.key_head_dim as u64,
            "attention KV width",
        )?;
        let gdn_key = checked_mul(
            config.gated_delta_net.key_heads as u64,
            config.gated_delta_net.key_head_dim as u64,
            "GDN key width",
        )?;
        let gdn_value = checked_mul(
            config.gated_delta_net.value_heads as u64,
            config.gated_delta_net.value_head_dim as u64,
            "GDN value width",
        )?;
        let gdn_qkv = checked_add(
            checked_mul(gdn_key, 2, "GDN query and key width")?,
            gdn_value,
            "GDN QKV width",
        )?;

        Ok(Self {
            hidden,
            vocab: config.vocab_size as u64,
            hyper_count,
            hyper_hidden,
            hyper_rank: config.hyper_connection.low_rank as u64,
            attention_q,
            attention_kv,
            attention_head: config.attention.key_head_dim as u64,
            gdn_qkv,
            gdn_value,
            gdn_heads: config.gated_delta_net.value_heads as u64,
            gdn_conv: config.gated_delta_net.conv_kernel as u64,
            index_query: checked_mul(
                config.qsa.query_heads as u64,
                config.qsa.head_dim as u64,
                "QSA index query width",
            )?,
            index_key: checked_mul(
                config.qsa.key_heads as u64,
                config.qsa.head_dim as u64,
                "QSA index key width",
            )?,
            index_head: config.qsa.head_dim as u64,
            experts: config.moe.expert_count as u64,
            expert_ffn: config.moe.expert_intermediate_size as u64,
            shared_ffn: config.moe.shared_expert_intermediate_size as u64,
            ple_conv: config.ple.as_ref().map_or(0, |ple| ple.conv_kernel as u64),
        })
    }
}

fn bind_residual<'a>(
    gguf: &'a GgufFile,
    used: &mut HashSet<String>,
    layer: u32,
    role: &str,
    dims: &Dimensions,
) -> Result<GatedResidualWeights<'a>, Qwen4ExpLoadError> {
    let prefix = format!("blk.{layer}.hc_{role}");
    Ok(GatedResidualWeights {
        norm: bind(
            gguf,
            used,
            format!("{prefix}_norm.weight"),
            &[dims.hyper_hidden],
        )?,
        down: bind(
            gguf,
            used,
            format!("{prefix}_down.weight"),
            &[dims.hyper_hidden, dims.hyper_rank],
        )?,
        up: bind(
            gguf,
            used,
            format!("{prefix}_up.weight"),
            &[dims.hyper_rank, dims.hyper_hidden],
        )?,
        inject: bind(
            gguf,
            used,
            format!("{prefix}_inject.weight"),
            &[dims.hyper_hidden, dims.hyper_count],
        )?,
    })
}

fn bind_gdn<'a>(
    gguf: &'a GgufFile,
    used: &mut HashSet<String>,
    layer: u32,
    dims: &Dimensions,
) -> Result<GatedDeltaNetWeights<'a>, Qwen4ExpLoadError> {
    let prefix = format!("blk.{layer}");
    Ok(GatedDeltaNetWeights {
        qkv: bind(
            gguf,
            used,
            format!("{prefix}.attn_qkv.weight"),
            &[dims.hidden, dims.gdn_qkv],
        )?,
        gate: bind(
            gguf,
            used,
            format!("{prefix}.attn_gate.weight"),
            &[dims.hidden, dims.gdn_value],
        )?,
        beta: bind(
            gguf,
            used,
            format!("{prefix}.ssm_beta.weight"),
            &[dims.hidden, dims.gdn_heads],
        )?,
        alpha: bind(
            gguf,
            used,
            format!("{prefix}.ssm_alpha.weight"),
            &[dims.hidden, dims.gdn_heads],
        )?,
        a: bind(gguf, used, format!("{prefix}.ssm_a"), &[dims.gdn_heads])?,
        dt_bias: bind(
            gguf,
            used,
            format!("{prefix}.ssm_dt.bias"),
            &[dims.gdn_heads],
        )?,
        conv: bind(
            gguf,
            used,
            format!("{prefix}.ssm_conv1d.weight"),
            &[dims.gdn_conv, dims.gdn_qkv],
        )?,
        norm: bind(
            gguf,
            used,
            format!("{prefix}.ssm_norm.weight"),
            &[dims.gdn_value / dims.gdn_heads],
        )?,
        output: bind(
            gguf,
            used,
            format!("{prefix}.ssm_out.weight"),
            &[dims.gdn_value, dims.hidden],
        )?,
    })
}

fn bind_qsa<'a>(
    gguf: &'a GgufFile,
    used: &mut HashSet<String>,
    layer: u32,
    dims: &Dimensions,
) -> Result<QwenSparseAttentionWeights<'a>, Qwen4ExpLoadError> {
    let prefix = format!("blk.{layer}");
    Ok(QwenSparseAttentionWeights {
        query_gate: bind(
            gguf,
            used,
            format!("{prefix}.attn_q.weight"),
            &[
                dims.hidden,
                checked_mul(dims.attention_q, 2, "gated query width")?,
            ],
        )?,
        key: bind(
            gguf,
            used,
            format!("{prefix}.attn_k.weight"),
            &[dims.hidden, dims.attention_kv],
        )?,
        value: bind(
            gguf,
            used,
            format!("{prefix}.attn_v.weight"),
            &[dims.hidden, dims.attention_kv],
        )?,
        output: bind(
            gguf,
            used,
            format!("{prefix}.attn_output.weight"),
            &[dims.attention_q, dims.hidden],
        )?,
        query_norm: bind(
            gguf,
            used,
            format!("{prefix}.attn_q_norm.weight"),
            &[dims.attention_head],
        )?,
        key_norm: bind(
            gguf,
            used,
            format!("{prefix}.attn_k_norm.weight"),
            &[dims.attention_head],
        )?,
        index_query: bind(
            gguf,
            used,
            format!("{prefix}.indexer.q_proj.weight"),
            &[dims.hidden, dims.index_query],
        )?,
        index_key: bind(
            gguf,
            used,
            format!("{prefix}.indexer.k_proj.weight"),
            &[dims.hidden, dims.index_key],
        )?,
        index_query_norm: bind(
            gguf,
            used,
            format!("{prefix}.indexer.q_norm.weight"),
            &[dims.index_head],
        )?,
        index_key_norm: bind(
            gguf,
            used,
            format!("{prefix}.indexer.k_norm.weight"),
            &[dims.index_head],
        )?,
    })
}

fn bind_moe<'a>(
    gguf: &'a GgufFile,
    used: &mut HashSet<String>,
    layer: u32,
    dims: &Dimensions,
) -> Result<MoeWeights<'a>, Qwen4ExpLoadError> {
    let prefix = format!("blk.{layer}");
    Ok(MoeWeights {
        router: bind(
            gguf,
            used,
            format!("{prefix}.ffn_gate_inp.weight"),
            &[dims.hidden, dims.experts],
        )?,
        routed_gate: bind(
            gguf,
            used,
            format!("{prefix}.ffn_gate_exps.weight"),
            &[dims.hidden, dims.expert_ffn, dims.experts],
        )?,
        routed_up: bind(
            gguf,
            used,
            format!("{prefix}.ffn_up_exps.weight"),
            &[dims.hidden, dims.expert_ffn, dims.experts],
        )?,
        routed_down: bind(
            gguf,
            used,
            format!("{prefix}.ffn_down_exps.weight"),
            &[dims.expert_ffn, dims.hidden, dims.experts],
        )?,
        shared_router: bind(
            gguf,
            used,
            format!("{prefix}.ffn_gate_inp_shexp.weight"),
            &[dims.hidden],
        )?,
        shared_gate: bind(
            gguf,
            used,
            format!("{prefix}.ffn_gate_shexp.weight"),
            &[dims.hidden, dims.shared_ffn],
        )?,
        shared_up: bind(
            gguf,
            used,
            format!("{prefix}.ffn_up_shexp.weight"),
            &[dims.hidden, dims.shared_ffn],
        )?,
        shared_down: bind(
            gguf,
            used,
            format!("{prefix}.ffn_down_shexp.weight"),
            &[dims.shared_ffn, dims.hidden],
        )?,
    })
}

fn bind_ple<'a>(
    gguf: &'a GgufFile,
    used: &mut HashSet<String>,
    layer: u32,
    dims: &Dimensions,
) -> Result<PleLayerWeights<'a>, Qwen4ExpLoadError> {
    let prefix = format!("blk.{layer}.ple");
    Ok(PleLayerWeights {
        key: bind(
            gguf,
            used,
            format!("{prefix}_key.weight"),
            &[dims.hidden, dims.hyper_hidden],
        )?,
        value: bind(
            gguf,
            used,
            format!("{prefix}_value.weight"),
            &[dims.hidden, dims.hidden],
        )?,
        key_norm: bind(
            gguf,
            used,
            format!("{prefix}_norm_key.weight"),
            &[dims.hyper_hidden],
        )?,
        query_norm: bind(
            gguf,
            used,
            format!("{prefix}_norm_query.weight"),
            &[dims.hyper_hidden],
        )?,
        conv_norm: bind(
            gguf,
            used,
            format!("{prefix}_norm_conv.weight"),
            &[dims.hyper_hidden],
        )?,
        conv: bind(
            gguf,
            used,
            format!("{prefix}_conv1d.weight"),
            &[dims.ple_conv, dims.hyper_hidden],
        )?,
    })
}

fn bind_ple_embedding<'a>(
    gguf: &'a GgufFile,
    used: &mut HashSet<String>,
    row_width: u64,
    logical_rows: u64,
) -> Result<&'a TensorDesc, Qwen4ExpLoadError> {
    let name = "per_layer_token_embd.weight".to_string();
    let tensor = gguf
        .find(&name)
        .ok_or_else(|| Qwen4ExpLoadError::Missing(name.clone()))?;
    if !ple_embedding_shape_is_valid(&tensor.shape, row_width, logical_rows) {
        return Err(Qwen4ExpLoadError::Shape {
            name,
            expected: vec![row_width, logical_rows],
            got: tensor.shape.clone(),
        });
    }
    validate_weight_storage(tensor)?;
    used.insert(tensor.name.clone());
    Ok(tensor)
}

fn ple_embedding_shape_is_valid(shape: &[u64], row_width: u64, logical_rows: u64) -> bool {
    shape.len() == 2 && shape[0] == row_width && shape[1] >= logical_rows
}

fn bind<'a>(
    gguf: &'a GgufFile,
    used: &mut HashSet<String>,
    name: String,
    expected: &[u64],
) -> Result<&'a TensorDesc, Qwen4ExpLoadError> {
    let tensor = gguf
        .find(&name)
        .ok_or_else(|| Qwen4ExpLoadError::Missing(name.clone()))?;
    if tensor.shape != expected {
        return Err(Qwen4ExpLoadError::Shape {
            name,
            expected: expected.to_vec(),
            got: tensor.shape.clone(),
        });
    }
    validate_weight_storage(tensor)?;
    used.insert(tensor.name.clone());
    Ok(tensor)
}

fn validate_weight_storage(tensor: &TensorDesc) -> Result<(), Qwen4ExpLoadError> {
    if matches!(
        tensor.dtype,
        GgmlType::Unknown
            | GgmlType::I8
            | GgmlType::I16
            | GgmlType::I32
            | GgmlType::I64
            | GgmlType::F64
    ) {
        return Err(Qwen4ExpLoadError::UnknownDtype {
            name: tensor.name.clone(),
            dtype: tensor.dtype,
        });
    }
    let (block_size, _) =
        tensor
            .dtype
            .storage_layout()
            .ok_or_else(|| Qwen4ExpLoadError::UnknownDtype {
                name: tensor.name.clone(),
                dtype: tensor.dtype,
            })?;
    if block_size == 0 || !tensor.shape[0].is_multiple_of(block_size) {
        return Err(Qwen4ExpLoadError::RowAlignment {
            name: tensor.name.clone(),
            row_width: tensor.shape[0],
            block_size,
            dtype: tensor.dtype,
        });
    }
    Ok(())
}

fn checked_mul(a: u64, b: u64, label: &'static str) -> Result<u64, Qwen4ExpLoadError> {
    a.checked_mul(b)
        .ok_or(Qwen4ExpLoadError::DimensionOverflow(label))
}

fn checked_add(a: u64, b: u64, label: &'static str) -> Result<u64, Qwen4ExpLoadError> {
    a.checked_add(b)
        .ok_or(Qwen4ExpLoadError::DimensionOverflow(label))
}

#[cfg(test)]
mod tests {
    use super::{ple_embedding_shape_is_valid, validate_weight_storage};
    use crate::qwen4exp::Qwen4ExpConfig;
    use crate::tensor::{GgmlType, TensorDesc};

    #[test]
    fn release_schema_has_exact_tensor_count() {
        let config = Qwen4ExpConfig::flash_next_reference();
        let gdn_layers = config
            .compress_ratios
            .iter()
            .filter(|&&ratio| ratio == 0)
            .count();
        let qsa_layers = config.qsa_layer_count();
        let ple_layers = config.ple.as_ref().map_or(0, |ple| ple.layers.len());
        let expected = 6 + gdn_layers * 25 + qsa_layers * 26 + ple_layers * 6;
        assert_eq!(expected, 1_224);
    }

    #[test]
    fn ple_table_accepts_exact_or_padded_storage_only() {
        assert!(ple_embedding_shape_is_valid(
            &[160, 320_001_446],
            160,
            320_001_446
        ));
        assert!(ple_embedding_shape_is_valid(
            &[160, 320_001_536],
            160,
            320_001_446
        ));
        assert!(!ple_embedding_shape_is_valid(
            &[160, 320_001_445],
            160,
            320_001_446
        ));
        assert!(!ple_embedding_shape_is_valid(
            &[128, 320_001_536],
            160,
            320_001_446
        ));
    }

    #[test]
    fn quantized_storage_requires_row_alignment() {
        let tensor = |width| TensorDesc {
            name: "test.weight".into(),
            shape: vec![width, 64],
            dtype: GgmlType::Q8_0,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: 0,
        };
        assert!(validate_weight_storage(&tensor(32)).is_ok());
        assert!(validate_weight_storage(&tensor(31)).is_err());
    }
}
