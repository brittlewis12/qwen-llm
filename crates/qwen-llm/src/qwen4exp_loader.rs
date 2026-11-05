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
pub struct GatedResidualWeights {
    pub norm: TensorDesc,
    pub down: TensorDesc,
    pub up: TensorDesc,
    pub inject: TensorDesc,
}

#[derive(Clone)]
pub struct FinalGatedResidualWeights {
    pub norm: TensorDesc,
    pub down: TensorDesc,
    pub up: TensorDesc,
}

#[derive(Clone)]
pub struct GatedDeltaNetWeights {
    pub qkv: TensorDesc,
    pub gate: TensorDesc,
    pub beta: TensorDesc,
    pub alpha: TensorDesc,
    pub a: TensorDesc,
    pub dt_bias: TensorDesc,
    pub conv: TensorDesc,
    pub norm: TensorDesc,
    pub output: TensorDesc,
}

#[derive(Clone)]
pub struct QwenSparseAttentionWeights {
    pub query_gate: TensorDesc,
    pub key: TensorDesc,
    pub value: TensorDesc,
    pub output: TensorDesc,
    pub query_norm: TensorDesc,
    pub key_norm: TensorDesc,
    pub index_query: TensorDesc,
    pub index_key: TensorDesc,
    pub index_query_norm: TensorDesc,
    pub index_key_norm: TensorDesc,
}

#[derive(Clone)]
pub enum MixerWeights {
    GatedDeltaNet(GatedDeltaNetWeights),
    QwenSparseAttention(QwenSparseAttentionWeights),
}

#[derive(Clone)]
pub struct MoeWeights {
    pub router: TensorDesc,
    pub routed_gate: TensorDesc,
    pub routed_up: TensorDesc,
    pub routed_down: TensorDesc,
    pub shared_router: TensorDesc,
    pub shared_gate: TensorDesc,
    pub shared_up: TensorDesc,
    pub shared_down: TensorDesc,
}

#[derive(Clone)]
pub struct PleLayerWeights {
    pub key: TensorDesc,
    pub value: TensorDesc,
    pub key_norm: TensorDesc,
    pub query_norm: TensorDesc,
    pub conv_norm: TensorDesc,
    pub conv: TensorDesc,
}

#[derive(Clone)]
pub struct Qwen4ExpBlock {
    pub attention_residual: GatedResidualWeights,
    pub mixer: MixerWeights,
    pub ffn_residual: GatedResidualWeights,
    pub moe: MoeWeights,
    pub ple: Option<PleLayerWeights>,
}

pub struct Qwen4ExpModel {
    pub config: Qwen4ExpConfig,
    pub token_embedding: TensorDesc,
    pub output: TensorDesc,
    pub tied_embeddings: bool,
    pub final_residual: FinalGatedResidualWeights,
    pub ple_embedding: Option<TensorDesc>,
    pub blocks: Vec<Qwen4ExpBlock>,
    /// Canonical split views; merged weights retain their original bytes and offsets.
    pub tensors: Vec<TensorDesc>,
    /// Recognized speculation weights, excluded from execution and residency.
    pub ignored_mtp: Vec<String>,
    /// Owned bank destinations in `tensors`; these require a byte permutation
    /// rather than a contiguous GGUF view. Do not slice their descriptors directly.
    pub repacks: Vec<ExpertBankRepack>,
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
    #[error("ambiguous tensor layout: {0}")]
    AmbiguousLayout(String),
    #[error("MTP inventory does not match declared nextn_predict_layers: {0}")]
    MtpInventory(String),
    #[error("unexpected qwen4exp tensor set: {0:?}")]
    UnexpectedTensors(Vec<String>),
}

impl Qwen4ExpModel {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, Qwen4ExpLoadError> {
        let config = Qwen4ExpConfig::from_gguf(gguf)?;
        let arch = gguf.get_str("general.architecture").unwrap_or("qwen4exp");
        let mtp_layers = gguf
            .get_u64(&format!("{arch}.nextn_predict_layers"))
            .or_else(|| gguf.get_u64("qwen4exp.nextn_predict_layers"))
            .unwrap_or(0);
        Self::from_inventory(config, &gguf.tensors, mtp_layers)
    }

    /// Bind synthetic or mapped inventories without allocating device resources.
    pub fn from_inventory(
        config: Qwen4ExpConfig,
        source: &[TensorDesc],
        mtp_layers: u64,
    ) -> Result<Self, Qwen4ExpLoadError> {
        config.validate()?;
        let (tensors, ignored_mtp, repacks) = canonical_inventory(&config, source, mtp_layers)?;
        let gguf = tensors.as_slice();
        let dims = Dimensions::new(&config)?;
        let mut used = HashSet::with_capacity(gguf.len());

        let token_embedding = bind(
            gguf,
            &mut used,
            "token_embd.weight".into(),
            &[dims.hidden, dims.vocab],
        )?;
        let (output, tied_embeddings) = if find(gguf, "output.weight").is_some() {
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
            (token_embedding.clone(), true)
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
            let logical_rows = ple.logical_row_count()?;
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
            tensors,
            ignored_mtp,
            repacks,
        })
    }
}

#[derive(Clone)]
pub struct ExpertBankRepack {
    pub source: TensorDesc,
    pub targets: [TensorDesc; 2],
    expert_bytes: usize,
}

impl ExpertBankRepack {
    /// Copy one bank, preserving every packed quantization byte. All bounds are
    /// checked before writing, so malformed spans leave the destination untouched.
    pub fn copy_bank(
        &self,
        bank: usize,
        source: &[u8],
        output: &mut [std::mem::MaybeUninit<u8>],
    ) -> Result<(), Qwen4ExpLoadError> {
        if bank >= 2
            || source.len() != self.source.n_bytes as usize
            || output.len() != self.targets[bank].n_bytes as usize
        {
            return Err(Qwen4ExpLoadError::DimensionOverflow(
                "expert repack byte span",
            ));
        }
        for (expert, dst) in output.chunks_exact_mut(self.expert_bytes).enumerate() {
            let start = (expert * 2 + bank) * self.expert_bytes;
            // SAFETY: the validated source and destination spans are disjoint
            // and cover exactly one complete expert bank, including quant bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source.as_ptr().add(start),
                    dst.as_mut_ptr().cast::<u8>(),
                    self.expert_bytes,
                );
            }
        }
        Ok(())
    }
}

impl Qwen4ExpModel {
    /// Metal-resident descriptors. Repack sources and targets are excluded:
    /// the source is read once from the GGUF mapping during repacking and the
    /// targets are owned copies created from it.
    pub(crate) fn residency_sources(&self) -> Vec<&TensorDesc> {
        self.tensors
            .iter()
            .filter(|desc| {
                desc.name != "per_layer_token_embd.weight"
                    && !self
                        .repacks
                        .iter()
                        .any(|r| r.targets.iter().any(|t| t.name == desc.name))
            })
            .collect()
    }

    /// Host bytes read once for expert-bank repacking (not Metal-resident).
    pub(crate) fn repacked_source_bytes(&self) -> u64 {
        self.repacks.iter().map(|r| r.source.n_bytes).sum()
    }
}

fn find<'a>(tensors: &'a [TensorDesc], name: &str) -> Option<&'a TensorDesc> {
    tensors.iter().find(|tensor| tensor.name == name)
}

/// Names and GGUF dimensions for the split trunk layout. PLE rows may be padded
/// on disk and output.weight may be omitted for tied embeddings.
pub fn expected_tensor_inventory(
    config: &Qwen4ExpConfig,
) -> Result<Vec<(String, Vec<u64>)>, Qwen4ExpLoadError> {
    config.validate()?;
    let d = Dimensions::new(config)?;
    let mut out = Vec::new();
    let mut add = |name: String, shape: &[u64]| out.push((name, shape.to_vec()));
    add("token_embd.weight".into(), &[d.hidden, d.vocab]);
    add("output.weight".into(), &[d.hidden, d.vocab]);
    add("output_hc_norm.weight".into(), &[d.hyper_hidden]);
    add(
        "output_hc_down.weight".into(),
        &[d.hyper_hidden, d.hyper_rank],
    );
    add(
        "output_hc_up.weight".into(),
        &[d.hyper_rank, d.hyper_hidden],
    );
    if let Some(ple) = &config.ple {
        add(
            "per_layer_token_embd.weight".into(),
            &[u64::from(ple.embedding_head_dim), ple.logical_row_count()?],
        );
    }
    for layer in 0..config.layer_count {
        let p = format!("blk.{layer}");
        for role in ["attn", "ffn"] {
            add(format!("{p}.hc_{role}_norm.weight"), &[d.hyper_hidden]);
            add(
                format!("{p}.hc_{role}_down.weight"),
                &[d.hyper_hidden, d.hyper_rank],
            );
            add(
                format!("{p}.hc_{role}_up.weight"),
                &[d.hyper_rank, d.hyper_hidden],
            );
            add(
                format!("{p}.hc_{role}_inject.weight"),
                &[d.hyper_hidden, d.hyper_count],
            );
        }
        let mixer = match config.mixer_kind(layer).unwrap() {
            MixerKind::GatedDeltaNet => vec![
                ("attn_qkv.weight", vec![d.hidden, d.gdn_qkv]),
                ("attn_gate.weight", vec![d.hidden, d.gdn_value]),
                ("ssm_beta.weight", vec![d.hidden, d.gdn_heads]),
                ("ssm_alpha.weight", vec![d.hidden, d.gdn_heads]),
                ("ssm_a", vec![d.gdn_heads]),
                ("ssm_dt.bias", vec![d.gdn_heads]),
                ("ssm_conv1d.weight", vec![d.gdn_conv, d.gdn_qkv]),
                ("ssm_norm.weight", vec![d.gdn_value / d.gdn_heads]),
                ("ssm_out.weight", vec![d.gdn_value, d.hidden]),
            ],
            MixerKind::QwenSparseAttention => vec![
                ("attn_q.weight", vec![d.hidden, d.attention_q * 2]),
                ("attn_k.weight", vec![d.hidden, d.attention_kv]),
                ("attn_v.weight", vec![d.hidden, d.attention_kv]),
                ("attn_output.weight", vec![d.attention_q, d.hidden]),
                ("attn_q_norm.weight", vec![d.attention_head]),
                ("attn_k_norm.weight", vec![d.attention_head]),
                ("indexer.q_proj.weight", vec![d.hidden, d.index_query]),
                ("indexer.k_proj.weight", vec![d.hidden, d.index_key]),
                ("indexer.q_norm.weight", vec![d.index_head]),
                ("indexer.k_norm.weight", vec![d.index_head]),
            ],
        };
        for (name, shape) in mixer {
            add(format!("{p}.{name}"), &shape);
        }
        for (name, shape) in [
            ("ffn_gate_inp.weight", vec![d.hidden, d.experts]),
            (
                "ffn_gate_exps.weight",
                vec![d.hidden, d.expert_ffn, d.experts],
            ),
            (
                "ffn_up_exps.weight",
                vec![d.hidden, d.expert_ffn, d.experts],
            ),
            (
                "ffn_down_exps.weight",
                vec![d.expert_ffn, d.hidden, d.experts],
            ),
            ("ffn_gate_inp_shexp.weight", vec![d.hidden]),
            ("ffn_gate_shexp.weight", vec![d.hidden, d.shared_ffn]),
            ("ffn_up_shexp.weight", vec![d.hidden, d.shared_ffn]),
            ("ffn_down_shexp.weight", vec![d.shared_ffn, d.hidden]),
        ] {
            add(format!("{p}.{name}"), &shape);
        }
        if config
            .ple
            .as_ref()
            .is_some_and(|ple| ple.layers.contains(&layer))
        {
            for (name, shape) in [
                ("ple_key.weight", vec![d.hidden, d.hyper_hidden]),
                ("ple_value.weight", vec![d.hidden, d.hidden]),
                ("ple_norm_key.weight", vec![d.hyper_hidden]),
                ("ple_norm_query.weight", vec![d.hyper_hidden]),
                ("ple_norm_conv.weight", vec![d.hyper_hidden]),
                ("ple_conv1d.weight", vec![d.ple_conv, d.hyper_hidden]),
            ] {
                add(format!("{p}.{name}"), &shape);
            }
        }
    }
    Ok(out)
}

// Exact roles, never prefix matches: an unknown nextn suffix must fail closed.
fn known_nextn_role(role: &str) -> bool {
    matches!(
        role,
        "nextn.eh_proj.weight"
            | "nextn.embed_tokens.weight"
            | "nextn.enorm.weight"
            | "nextn.hnorm.weight"
            | "nextn.shared_head_norm.weight"
            | "nextn.shared_head_head.weight"
            | "nextn.hc_head_norm.weight"
            | "nextn.hc_head_down.weight"
            | "nextn.hc_head_up.weight"
            | "nextn_fc_embedding.weight"
            | "nextn_fc_hidden.weight"
            | "nextn_hc_head.weight"
    )
}

fn known_root_nextn_name(name: &str) -> bool {
    matches!(
        name,
        "nextn.pre_projection.weight"
            | "nextn.post_projection.weight"
            | "nextn_fc_embedding.weight"
            | "nextn_fc_hidden.weight"
            | "nextn_hc_head.weight"
    )
}

fn known_root_mtp_name(name: &str) -> bool {
    matches!(
        name,
        "mtp.fc.weight"
            | "mtp.pre_fc_norm_embedding.weight"
            | "mtp.pre_fc_norm_hidden.weight"
            | "mtp.norm.weight"
            | "mtp.fc_embedding.weight"
            | "mtp.fc_hidden.weight"
            | "mtp.hyper_connection_mixer.weight"
    )
}

// Layouts follow llama.cpp conversion/qwen4exp.py and src/models/qwen4exp.cpp
// (a91eab67d838), plus the shared QKV and expert gate/up loaders.
fn canonical_inventory(
    config: &Qwen4ExpConfig,
    source: &[TensorDesc],
    mtp_layers: u64,
) -> Result<(Vec<TensorDesc>, Vec<String>, Vec<ExpertBankRepack>), Qwen4ExpLoadError> {
    let mut seen = HashSet::new();
    let mut tensors = Vec::new();
    let mut ignored = Vec::new();
    let mut repacks = Vec::new();
    if mtp_layers > u64::from(u32::MAX - config.layer_count) {
        return Err(Qwen4ExpLoadError::DimensionOverflow("MTP layer range"));
    }
    let trunk_layers = u64::from(config.layer_count);
    let mtp_end = trunk_layers + mtp_layers;
    let mut mtp_signature_blocks = HashSet::new();
    let mut saw_root_mtp = false;
    let mut ordinary_mtp_blocks = HashSet::new();
    let mut appended_config = config.clone();
    if mtp_layers > 1 {
        return Err(Qwen4ExpLoadError::MtpInventory(
            "only one MTP block is qualified".into(),
        ));
    }
    appended_config.layer_count += mtp_layers as u32;
    appended_config.compress_ratios.extend(std::iter::repeat_n(
        config
            .compress_ratios
            .iter()
            .copied()
            .find(|r| *r > 0)
            .unwrap_or(4),
        mtp_layers as usize,
    ));
    let appended_names = expected_tensor_inventory(&appended_config)?;
    for desc in source {
        if !seen.insert(desc.name.as_str()) {
            return Err(Qwen4ExpLoadError::AmbiguousLayout(format!(
                "duplicate tensor {}",
                desc.name
            )));
        }
        let block = desc
            .name
            .strip_prefix("blk.")
            .and_then(|s| s.split_once('.'));
        let block_index = block.and_then(|(n, _)| n.parse::<u64>().ok());
        let declared_mtp_block = block_index.is_some_and(|n| n >= trunk_layers && n < mtp_end);
        let nextn_named = desc.name.starts_with("nextn_")
            || desc.name.starts_with("nextn.")
            || block
                .is_some_and(|(_, role)| role.starts_with("nextn_") || role.starts_with("nextn."));
        if nextn_named {
            let known = match block {
                Some((_, role)) => known_nextn_role(role),
                None => known_root_nextn_name(&desc.name),
            };
            if !known {
                return Err(Qwen4ExpLoadError::MtpInventory(format!(
                    "tensor {:?} has an unknown MTP role",
                    desc.name
                )));
            }
            if mtp_layers == 0 {
                return Err(Qwen4ExpLoadError::MtpInventory(format!(
                    "tensor {:?} is MTP but nextn_predict_layers is 0",
                    desc.name
                )));
            }
            match block_index {
                Some(index) if index < trunk_layers || index >= mtp_end => {
                    return Err(Qwen4ExpLoadError::MtpInventory(format!(
                        "MTP tensor {:?} declares block {index} outside the {trunk_layers}-block trunk and {mtp_layers} declared MTP block(s)",
                        desc.name
                    )));
                }
                Some(index) => {
                    if desc.name.ends_with(".nextn.eh_proj.weight") {
                        mtp_signature_blocks.insert(index);
                    }
                }
                None => saw_root_mtp = true,
            }
            ignored.push(desc.name.clone());
        } else if desc.name.starts_with("mtp.") {
            if !known_root_mtp_name(&desc.name) {
                return Err(Qwen4ExpLoadError::MtpInventory(format!(
                    "tensor {:?} has an unknown root MTP role",
                    desc.name
                )));
            }
            if mtp_layers == 0 {
                return Err(Qwen4ExpLoadError::MtpInventory(format!(
                    "tensor {:?} is MTP but nextn_predict_layers is 0",
                    desc.name
                )));
            }
            saw_root_mtp = true;
            ignored.push(desc.name.clone());
        } else if declared_mtp_block {
            // A default converter MTP block is a full QSA block. Only known
            // trunk roles may accompany its nextn signature.
            let (_, role) = block.unwrap();
            if !appended_names.iter().any(|(name, _)| name == &desc.name)
                && !matches!(
                    role,
                    "hc_attn_down_inject.weight"
                        | "hc_ffn_down_inject.weight"
                        | "attn_qkv.weight"
                        | "ffn_gate_up_exps.weight"
                )
            {
                return Err(Qwen4ExpLoadError::MtpInventory(format!(
                    "tensor {:?} has an unknown appended MTP role",
                    desc.name
                )));
            }
            ordinary_mtp_blocks.insert(block_index.unwrap());
            ignored.push(desc.name.clone());
        } else {
            tensors.push(desc.clone());
        }
    }
    if mtp_layers > 0 {
        if !saw_root_mtp && mtp_signature_blocks.is_empty() {
            return Err(Qwen4ExpLoadError::MtpInventory(format!(
                "nextn_predict_layers={mtp_layers} but the inventory has no MTP signature tensor"
            )));
        }
        for index in trunk_layers..mtp_end {
            if (!saw_root_mtp || ordinary_mtp_blocks.contains(&index))
                && !mtp_signature_blocks.contains(&index)
            {
                return Err(Qwen4ExpLoadError::MtpInventory(format!(
                    "MTP block {index} is declared but has no nextn signature tensor"
                )));
            }
        }
    }
    let schema = expected_tensor_inventory(config)?;
    for desc in &mut tensors {
        if desc.shape
            == [
                u64::from(config.hidden_size),
                u64::from(config.hyper_connection.count),
            ]
            && schema.iter().any(|(name, shape)| {
                name == &desc.name
                    && shape
                        == &[u64::from(config.hidden_size)
                            * u64::from(config.hyper_connection.count)]
            })
        {
            desc.shape =
                vec![u64::from(config.hidden_size) * u64::from(config.hyper_connection.count)];
        }
    }
    for layer in 0..config.layer_count {
        let p = format!("blk.{layer}");
        let mut groups = vec![
            (
                "hc_attn_down_inject",
                vec!["hc_attn_down", "hc_attn_inject"],
            ),
            ("hc_ffn_down_inject", vec!["hc_ffn_down", "hc_ffn_inject"]),
        ];
        match config.mixer_kind(layer).unwrap() {
            MixerKind::GatedDeltaNet => groups.extend([
                ("attn_qkv_gate", vec!["attn_qkv", "attn_gate"]),
                ("ssm_beta_alpha", vec!["ssm_beta", "ssm_alpha"]),
                ("ssm_beta_alpha_cat", vec!["ssm_beta", "ssm_alpha"]),
            ]),
            MixerKind::QwenSparseAttention => {
                groups.push(("attn_qkv", vec!["attn_q", "attn_k", "attn_v"]))
            }
        }
        for (merged, parts) in groups {
            let name = format!("{p}.{merged}.weight");
            let Some(index) = tensors.iter().position(|t| t.name == name) else {
                continue;
            };
            let names: Vec<_> = parts.iter().map(|s| format!("{p}.{s}.weight")).collect();
            if names.iter().any(|n| tensors.iter().any(|t| &t.name == n)) {
                return Err(Qwen4ExpLoadError::AmbiguousLayout(name));
            }
            let shapes: Vec<_> = names
                .iter()
                .map(|name| &schema.iter().find(|(n, _)| n == name).unwrap().1)
                .collect();
            let rows = shapes
                .iter()
                .try_fold(0u64, |sum, shape| checked_add(sum, shape[1], "merged rows"))?;
            let desc = tensors.remove(index);
            let expected = vec![shapes[0][0], rows];
            if desc.shape != expected {
                return Err(Qwen4ExpLoadError::Shape {
                    name,
                    expected,
                    got: desc.shape,
                });
            }
            validate_weight_storage(&desc)?;
            let (block, bytes) = desc.dtype.storage_layout().unwrap();
            let row_bytes = checked_mul(desc.shape[0] / block, bytes, "merged row bytes")?;
            if checked_mul(row_bytes, rows, "merged bytes")? != desc.n_bytes {
                return Err(Qwen4ExpLoadError::DimensionOverflow(
                    "merged byte length disagrees with shape",
                ));
            }
            let mut offset = desc.data_offset;
            for (name, shape) in names.into_iter().zip(shapes) {
                let n_bytes = checked_mul(row_bytes, shape[1], "split view bytes")?;
                tensors.push(TensorDesc {
                    name,
                    shape: shape.clone(),
                    data_offset: offset,
                    n_bytes,
                    ..desc.clone()
                });
                offset = checked_add(offset, n_bytes, "split view offset")?;
            }
        }
    }
    for layer in 0..config.layer_count {
        let name = format!("blk.{layer}.ffn_gate_up_exps.weight");
        let Some(index) = tensors.iter().position(|t| t.name == name) else {
            continue;
        };
        let source = tensors.remove(index);
        let names = [
            format!("blk.{layer}.ffn_gate_exps.weight"),
            format!("blk.{layer}.ffn_up_exps.weight"),
        ];
        if names.iter().any(|n| tensors.iter().any(|t| &t.name == n)) {
            return Err(Qwen4ExpLoadError::AmbiguousLayout(name));
        }
        let shape = schema
            .iter()
            .find(|(n, _)| n == &names[0])
            .unwrap()
            .1
            .clone();
        let expected = vec![
            shape[0],
            checked_mul(shape[1], 2, "merged expert rows")?,
            shape[2],
        ];
        if source.shape != expected {
            return Err(Qwen4ExpLoadError::Shape {
                name,
                expected,
                got: source.shape,
            });
        }
        validate_weight_storage(&source)?;
        let (block, bytes) = source.dtype.storage_layout().unwrap();
        let expert_bytes = checked_mul(
            checked_mul(shape[0] / block, bytes, "expert row bytes")?,
            shape[1],
            "expert bytes",
        )?;
        if checked_mul(
            checked_mul(expert_bytes, 2, "expert pair bytes")?,
            shape[2],
            "expert bank bytes",
        )? != source.n_bytes
        {
            return Err(Qwen4ExpLoadError::DimensionOverflow(
                "merged expert byte length",
            ));
        }
        let n_bytes = source.n_bytes / 2;
        let targets = names.map(|name| TensorDesc {
            name,
            shape: shape.clone(),
            data_offset: 0,
            n_bytes,
            ..source.clone()
        });
        tensors.extend(targets.clone());
        repacks.push(ExpertBankRepack {
            source,
            targets,
            expert_bytes: usize::try_from(expert_bytes)
                .map_err(|_| Qwen4ExpLoadError::DimensionOverflow("expert host bytes"))?,
        });
    }
    Ok((tensors, ignored, repacks))
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

fn bind_residual(
    gguf: &[TensorDesc],
    used: &mut HashSet<String>,
    layer: u32,
    role: &str,
    dims: &Dimensions,
) -> Result<GatedResidualWeights, Qwen4ExpLoadError> {
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

fn bind_gdn(
    gguf: &[TensorDesc],
    used: &mut HashSet<String>,
    layer: u32,
    dims: &Dimensions,
) -> Result<GatedDeltaNetWeights, Qwen4ExpLoadError> {
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

fn bind_qsa(
    gguf: &[TensorDesc],
    used: &mut HashSet<String>,
    layer: u32,
    dims: &Dimensions,
) -> Result<QwenSparseAttentionWeights, Qwen4ExpLoadError> {
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

fn bind_moe(
    gguf: &[TensorDesc],
    used: &mut HashSet<String>,
    layer: u32,
    dims: &Dimensions,
) -> Result<MoeWeights, Qwen4ExpLoadError> {
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

fn bind_ple(
    gguf: &[TensorDesc],
    used: &mut HashSet<String>,
    layer: u32,
    dims: &Dimensions,
) -> Result<PleLayerWeights, Qwen4ExpLoadError> {
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

fn bind_ple_embedding(
    gguf: &[TensorDesc],
    used: &mut HashSet<String>,
    row_width: u64,
    logical_rows: u64,
) -> Result<TensorDesc, Qwen4ExpLoadError> {
    let name = "per_layer_token_embd.weight".to_string();
    let tensor = find(gguf, &name).ok_or_else(|| Qwen4ExpLoadError::Missing(name.clone()))?;
    if !ple_embedding_shape_is_valid(&tensor.shape, row_width, logical_rows) {
        return Err(Qwen4ExpLoadError::Shape {
            name,
            expected: vec![row_width, logical_rows],
            got: tensor.shape.clone(),
        });
    }
    validate_weight_storage(tensor)?;
    used.insert(tensor.name.clone());
    Ok(tensor.clone())
}

fn ple_embedding_shape_is_valid(shape: &[u64], row_width: u64, logical_rows: u64) -> bool {
    shape.len() == 2 && shape[0] == row_width && shape[1] >= logical_rows
}

fn bind(
    gguf: &[TensorDesc],
    used: &mut HashSet<String>,
    name: String,
    expected: &[u64],
) -> Result<TensorDesc, Qwen4ExpLoadError> {
    let tensor = find(gguf, &name).ok_or_else(|| Qwen4ExpLoadError::Missing(name.clone()))?;
    if tensor.shape != expected {
        return Err(Qwen4ExpLoadError::Shape {
            name,
            expected: expected.to_vec(),
            got: tensor.shape.clone(),
        });
    }
    validate_weight_storage(tensor)?;
    used.insert(tensor.name.clone());
    Ok(tensor.clone())
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
    use super::*;
    use crate::qwen4exp::Qwen4ExpConfig;
    use crate::tensor::{GgmlType, TensorDesc};

    fn fixture(config: &Qwen4ExpConfig) -> Vec<TensorDesc> {
        let mut offset = 0;
        expected_tensor_inventory(config)
            .unwrap()
            .into_iter()
            .map(|(name, shape)| {
                let n_bytes = shape.iter().product::<u64>() * 4;
                let desc = TensorDesc {
                    name,
                    shape,
                    dtype: GgmlType::F32,
                    shard_idx: 0,
                    data_offset: offset,
                    n_bytes,
                };
                offset += n_bytes;
                desc
            })
            .collect()
    }

    fn alternate(layers: u32) -> Qwen4ExpConfig {
        let mut c = Qwen4ExpConfig::flash_next_reference();
        c.layer_count = layers;
        c.hidden_size = 4096;
        c.moe.expert_count = 64;
        c.moe.experts_per_token = 4;
        c.moe.expert_intermediate_size = 256;
        c.moe.shared_expert_intermediate_size = 512;
        c.hyper_connection.count = 3;
        c.hyper_connection.low_rank = 128;
        c.attention.kv_heads = 4;
        c.ple.as_mut().unwrap().embedding_head_dim = 256;
        c.compress_ratios = (0..layers)
            .map(|i| if i % 4 == 3 { 2 } else { 0 })
            .collect();
        c.qsa.token_budget = 512;
        c
    }

    #[test]
    fn artifact_contract_admits_reference_and_alternate_sizes_with_derived_inventories() {
        for config in [
            Qwen4ExpConfig::flash_next_reference(),
            alternate(8),
            alternate(52),
        ] {
            config.validate_artifact().unwrap();
            let capacity =
                crate::qwen4exp_runtime::Qwen4ExpSessionCapacity::for_forward_limit(&config, 16)
                    .unwrap();
            crate::qwen4exp_text_session::Qwen4ExpTextSessionMetalGeometry::from_config(
                &config,
                capacity.qsa_physical_capacity(),
            )
            .unwrap();
            let source = fixture(&config);
            let count = 6
                + (config.layer_count as usize - config.qsa_layer_count()) * 25
                + config.qsa_layer_count() * 26
                + 6;
            assert_eq!(source.len(), count);
            let model = Qwen4ExpModel::from_inventory(config.clone(), &source, 0).unwrap();
            assert_eq!(model.blocks.len(), config.layer_count as usize);
            assert_eq!(
                model.ple_embedding.unwrap().shape[0],
                u64::from(config.ple.unwrap().embedding_head_dim)
            );
        }
    }

    #[test]
    fn artifact_contract_refuses_malformed_and_unsupported_geometry() {
        let reference = alternate(8);
        let changes: Vec<Box<dyn Fn(&mut Qwen4ExpConfig)>> = vec![
            Box::new(|c| c.attention.kv_heads = 5),
            Box::new(|c| c.gated_delta_net.value_head_dim = 64),
            Box::new(|c| c.compress_ratios[0] = 4),
            Box::new(|c| c.moe.experts_per_token = 65),
            Box::new(|c| c.hyper_connection.count = 1),
            Box::new(|c| c.hyper_connection.low_rank = 0),
            Box::new(|c| c.qsa.query_heads = 8),
            Box::new(|c| c.qsa.head_dim = 256),
            Box::new(|c| c.ple = None),
            Box::new(|c| c.ple.as_mut().unwrap().layers = vec![2]),
            Box::new(|c| c.ple.as_mut().unwrap().embedding_head_dim = 160),
            Box::new(|c| c.moe.experts_per_token = 17),
            Box::new(|c| c.moe.expert_count = 0),
            Box::new(|c| c.moe.expert_count = 513),
            Box::new(|c| c.qsa.token_budget = 4_096),
            Box::new(|c| c.compress_ratios[3] = 4),
            Box::new(|c| c.gated_delta_net.conv_kernel = 3),
        ];
        for change in changes {
            let mut config = reference.clone();
            change(&mut config);
            assert!(config.validate_artifact().is_err(), "{config:?}");
        }
        let mut dense = reference;
        dense.moe.expert_count = 0;
        assert!(matches!(
            dense.validate_artifact(),
            Err(Qwen4ExpError::UnsupportedShape {
                feature: "dense FFN",
                ..
            })
        ));
    }

    fn merge(source: &mut Vec<TensorDesc>, merged: &str, names: &[&str]) {
        let parts: Vec<_> = names
            .iter()
            .map(|name| source.remove(source.iter().position(|t| t.name == *name).unwrap()))
            .collect();
        let mut desc = parts[0].clone();
        desc.name = merged.into();
        desc.shape[1] = parts.iter().map(|p| p.shape[1]).sum();
        desc.n_bytes = parts.iter().map(|p| p.n_bytes).sum();
        source.push(desc);
    }

    #[test]
    fn tied_output_and_unsqueezed_hc_norms_follow_config_inventory() {
        let c = alternate(8);
        let mut source = fixture(&c);
        source.retain(|t| t.name != "output.weight");
        for tensor in &mut source {
            if tensor.name == "output_hc_norm.weight"
                || tensor.name.ends_with(".hc_attn_norm.weight")
                || tensor.name.ends_with(".hc_ffn_norm.weight")
                || tensor.name.contains(".ple_norm_")
            {
                tensor.shape = vec![
                    u64::from(c.hidden_size),
                    u64::from(c.hyper_connection.count),
                ];
            }
        }
        let model = Qwen4ExpModel::from_inventory(c.clone(), &source, 0).unwrap();
        assert!(model.tied_embeddings);
        assert_eq!(model.output.name, "token_embd.weight");
        assert_eq!(
            model.tensors.len() + 1,
            expected_tensor_inventory(&c).unwrap().len()
        );
        assert_eq!(
            model.final_residual.norm.shape,
            vec![u64::from(c.hidden_size) * u64::from(c.hyper_connection.count)]
        );
    }

    #[test]
    fn fused_expert_banks_copy_packed_bytes_without_changing_values() {
        let mut c = alternate(4);
        c.hidden_size = 512;
        c.ple.as_mut().unwrap().embedding_head_dim = 32;
        c.moe.expert_count = 3;
        c.moe.experts_per_token = 2;
        c.moe.expert_intermediate_size = 32;
        let mut source = fixture(&c);
        let gate = source
            .iter()
            .position(|t| t.name == "blk.0.ffn_gate_exps.weight")
            .unwrap();
        let mut merged = source.remove(gate);
        source.retain(|t| t.name != "blk.0.ffn_up_exps.weight");
        merged.name = "blk.0.ffn_gate_up_exps.weight".into();
        merged.shape[1] *= 2;
        merged.dtype = GgmlType::IQ3_XXS;
        let (block, bytes) = merged.dtype.storage_layout().unwrap();
        merged.n_bytes = merged.shape.iter().product::<u64>() / block * bytes;
        source.push(merged.clone());
        let model = Qwen4ExpModel::from_inventory(c.clone(), &source, 0).unwrap();
        assert_eq!(model.repacks.len(), 1);
        let repack = &model.repacks[0];
        let bytes: Vec<_> = (0..merged.n_bytes)
            .map(|i| (i.wrapping_mul(13) % 251) as u8)
            .collect();
        for bank in 0..2 {
            let mut output =
                vec![std::mem::MaybeUninit::new(0); repack.targets[bank].n_bytes as usize];
            repack.copy_bank(bank, &bytes, &mut output).unwrap();
            let actual: Vec<_> = output
                .into_iter()
                .map(|b| unsafe { b.assume_init() })
                .collect();
            let expected: Vec<_> = bytes
                .chunks_exact(repack.expert_bytes)
                .enumerate()
                .filter(|(i, _)| i % 2 == bank)
                .flat_map(|(_, b)| b.iter().copied())
                .collect();
            assert_eq!(actual, expected);
            assert_eq!(repack.targets[bank].dtype, merged.dtype);
        }
        assert_eq!(model.residency_sources().len(), model.tensors.len() - 3);
        let resident: Vec<_> = model
            .residency_sources()
            .iter()
            .map(|desc| desc.name.as_str())
            .collect();
        assert!(!resident.contains(&"blk.0.ffn_gate_up_exps.weight"));
        assert!(!resident.contains(&"blk.0.ffn_gate_exps.weight"));
        assert!(!resident.contains(&"blk.0.ffn_up_exps.weight"));
        assert_eq!(model.repacked_source_bytes(), merged.n_bytes);
        assert_eq!(
            repack.targets[0].n_bytes + repack.targets[1].n_bytes,
            merged.n_bytes
        );
        let mut sentinel = vec![std::mem::MaybeUninit::new(7); repack.targets[0].n_bytes as usize];
        assert!(repack.copy_bank(0, &bytes[1..], &mut sentinel).is_err());
        assert!(
            sentinel
                .into_iter()
                .all(|b| unsafe { b.assume_init() } == 7)
        );
        let mut malformed = source.clone();
        malformed.last_mut().unwrap().shape[2] += 1;
        assert!(matches!(
            Qwen4ExpModel::from_inventory(c, &malformed, 0),
            Err(Qwen4ExpLoadError::Shape { .. })
        ));
    }

    #[test]
    fn merged_llama_cpp_projections_preserve_split_shapes_and_byte_ranges() {
        for beta_name in ["ssm_beta_alpha", "ssm_beta_alpha_cat"] {
            let config = alternate(8);
            let split = fixture(&config);
            let mut source = split.clone();
            for layer in 0..config.layer_count {
                for (merged, parts) in [
                    (
                        "hc_attn_down_inject",
                        vec!["hc_attn_down", "hc_attn_inject"],
                    ),
                    ("hc_ffn_down_inject", vec!["hc_ffn_down", "hc_ffn_inject"]),
                ] {
                    let names: Vec<_> = parts
                        .iter()
                        .map(|p| format!("blk.{layer}.{p}.weight"))
                        .collect();
                    merge(
                        &mut source,
                        &format!("blk.{layer}.{merged}.weight"),
                        &names.iter().map(String::as_str).collect::<Vec<_>>(),
                    );
                }
            }
            merge(
                &mut source,
                "blk.0.attn_qkv_gate.weight",
                &["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"],
            );
            merge(
                &mut source,
                &format!("blk.0.{beta_name}.weight"),
                &["blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight"],
            );
            merge(
                &mut source,
                "blk.3.attn_qkv.weight",
                &[
                    "blk.3.attn_q.weight",
                    "blk.3.attn_k.weight",
                    "blk.3.attn_v.weight",
                ],
            );
            let model = Qwen4ExpModel::from_inventory(config, &source, 0).unwrap();
            assert_eq!(model.tensors.len(), split.len());
            for tensor in &model.tensors {
                let original = split.iter().find(|t| t.name == tensor.name).unwrap();
                assert_eq!(tensor.shape, original.shape);
                assert_eq!(tensor.n_bytes, original.n_bytes);
            }
            let qkv = source
                .iter()
                .find(|t| t.name == "blk.0.attn_qkv_gate.weight")
                .unwrap();
            let q = model
                .tensors
                .iter()
                .find(|t| t.name == "blk.0.attn_qkv.weight")
                .unwrap();
            let gate = model
                .tensors
                .iter()
                .find(|t| t.name == "blk.0.attn_gate.weight")
                .unwrap();
            assert_eq!(q.data_offset, qkv.data_offset);
            assert_eq!(gate.data_offset, q.data_offset + q.n_bytes);
            assert_eq!(
                gate.data_offset + gate.n_bytes,
                qkv.data_offset + qkv.n_bytes
            );
        }
    }

    #[test]
    fn llama_cpp_v090_default_metadata_merged_views_and_appended_mtp() {
        use std::collections::HashMap;
        // c061df198 (#29761): block_count includes a full-QSA MTP block;
        // 2280f0fd1 (#31028): row concatenation, no grouped BA permutation.
        let mut header = crate::qwen4exp::tests::metadata_fixture_with("qwen4exp", true, |c| {
            c.layer_count = 4;
            c.hidden_size = 256;
            c.vocab_size = 32;
            c.hyper_connection.count = 2;
            c.hyper_connection.low_rank = 32;
            c.attention.query_heads = 4;
            c.attention.kv_heads = 1;
            c.gated_delta_net.key_heads = 1;
            c.gated_delta_net.value_heads = 2;
            c.gated_delta_net.inner_size = 256;
            c.moe.expert_count = 2;
            c.moe.experts_per_token = 1;
            c.moe.expert_intermediate_size = 32;
            c.moe.shared_expert_intermediate_size = 32;
            c.qsa.token_budget = 8;
            c.compress_ratios = vec![0, 0, 0, 4];
            let ple = c.ple.as_mut().unwrap();
            ple.token_vocab_size = 32;
            ple.heads_per_ngram = 4;
            ple.embedding_head_dim = 32;
            ple.eos_token_id = 1;
            ple.image_token_id = Some(2);
            ple.head_vocab_sizes = vec![8; 8];
            ple.head_offsets = (0..8).map(|i| i * 8).collect();
        });
        let c = Qwen4ExpConfig::from_gguf(&header).unwrap();
        c.validate_artifact().unwrap();
        assert_eq!(header.get_u64("qwen4exp.block_count"), Some(5));
        assert_eq!(header.get_u64("qwen4exp.nextn_predict_layers"), Some(1));
        assert_eq!(
            header
                .get_i64_array("qwen4exp.attention.compress_ratios")
                .unwrap()
                .unwrap(),
            vec![0, 0, 0, 4, 4]
        );
        let mut split = quantized_fixture(&c);
        // One merged HC tensor has one dtype, including the inject rows.
        for d in &mut split {
            if d.name.ends_with("inject.weight") {
                d.dtype = GgmlType::Q8_0;
                d.n_bytes = d.shape.iter().product::<u64>() / 32 * 34;
            }
        }
        // Q4_K_M promotes the merged GDN QKV/z projection to Q5_K.
        // Exercise its 256-element quant blocks at a nonzero shard offset.
        for d in &mut split {
            if d.name.ends_with("attn_qkv.weight") || d.name.ends_with("attn_gate.weight") {
                d.dtype = GgmlType::Q5_K;
                let (block, bytes) = d.dtype.storage_layout().unwrap();
                d.n_bytes = d.shape.iter().product::<u64>() / block * bytes;
            }
        }
        let mut extended = c.clone();
        extended.layer_count += 1;
        extended.compress_ratios.push(4);
        let mut physical = split.clone();
        physical.extend(
            quantized_fixture(&extended)
                .into_iter()
                .filter(|d| d.name.starts_with("blk.4.")),
        );
        for (role, shape) in [
            ("eh_proj", vec![512, 256]),
            ("enorm", vec![256]),
            ("hnorm", vec![256, 2]),
            ("hc_head_norm", vec![256, 2]),
            ("hc_head_down", vec![512, 32]),
            ("hc_head_up", vec![32, 512]),
        ] {
            physical.push(TensorDesc {
                name: format!("blk.4.nextn.{role}.weight"),
                n_bytes: shape.iter().product::<u64>() * 4,
                shape,
                dtype: GgmlType::F32,
                shard_idx: 0,
                data_offset: 0,
            });
        }
        // Preserve converter emission order: cache the first half and emit
        // the concatenation when the second half arrives. QSA Q/K/V stay split.
        let mut groups = Vec::new();
        for layer in 0..5 {
            for (merged, parts) in [
                ("hc_attn_down_inject", ["hc_attn_down", "hc_attn_inject"]),
                ("hc_ffn_down_inject", ["hc_ffn_down", "hc_ffn_inject"]),
            ] {
                groups.push((
                    format!("blk.{layer}.{merged}.weight"),
                    parts.map(|p| format!("blk.{layer}.{p}.weight")),
                ));
            }
            if layer < 3 {
                for (merged, parts) in [
                    ("attn_qkv_gate", ["attn_qkv", "attn_gate"]),
                    ("ssm_beta_alpha", ["ssm_beta", "ssm_alpha"]),
                ] {
                    groups.push((
                        format!("blk.{layer}.{merged}.weight"),
                        parts.map(|p| format!("blk.{layer}.{p}.weight")),
                    ));
                }
            }
        }
        let bytes_for = |d: &TensorDesc| -> Vec<u8> {
            let seed = d
                .name
                .bytes()
                .fold(0u8, |acc, b| acc.wrapping_mul(31).wrapping_add(b));
            (0..d.n_bytes)
                .map(|i| seed.wrapping_add((i % 251) as u8))
                .collect()
        };
        let mut payload = HashMap::new();
        let mut emitted = Vec::new();
        for d in &physical {
            if let Some((name, parts)) = groups.iter().find(|(_, p)| p.contains(&d.name)) {
                if d.name != parts[1] {
                    continue;
                }
                let first = physical.iter().find(|d| d.name == parts[0]).unwrap();
                let mut merged = first.clone();
                merged.name = name.clone();
                merged.shape[1] += d.shape[1];
                // MTP HC halves also share one dtype after quantization.
                merged.n_bytes = merged.shape.iter().product::<u64>()
                    / merged.dtype.storage_layout().unwrap().0
                    * merged.dtype.storage_layout().unwrap().1;
                let mut data = bytes_for(first);
                let mut second = d.clone();
                second.dtype = first.dtype;
                second.n_bytes = second.shape.iter().product::<u64>()
                    / first.dtype.storage_layout().unwrap().0
                    * first.dtype.storage_layout().unwrap().1;
                data.extend(bytes_for(&second));
                assert_eq!(data.len() as u64, merged.n_bytes);
                payload.insert(name.clone(), data);
                emitted.push(merged);
            } else {
                payload.insert(d.name.clone(), bytes_for(d));
                emitted.push(d.clone());
            }
        }
        let mut offsets = [4096u64, 8192];
        for (i, d) in emitted.iter_mut().enumerate() {
            d.shard_idx = i % 2;
            d.data_offset = offsets[d.shard_idx].next_multiple_of(32);
            offsets[d.shard_idx] = d.data_offset + d.n_bytes;
        }
        assert!(
            emitted
                .iter()
                .any(|d| d.name == "blk.4.hc_attn_down_inject.weight")
        );
        assert!(
            !emitted
                .iter()
                .any(|d| d.name == "blk.4.attn_qkv_gate.weight")
        );
        header.tensors = emitted.clone();
        let model = Qwen4ExpModel::from_gguf(&header).unwrap();
        assert_eq!(model.config, c);
        assert_eq!(model.blocks.len(), 4);
        assert_eq!(
            model.ignored_mtp.len(),
            physical
                .iter()
                .filter(|d| d.name.starts_with("blk.4."))
                .count()
                - 2
        );
        assert!(!model.tensors.iter().any(|d| d.name.starts_with("blk.4.")));
        crate::qwen4exp_residency::validate_native_dtypes(&model.tensors).unwrap();
        for view in &model.tensors {
            let original = split.iter().find(|d| d.name == view.name).unwrap();
            let source = emitted
                .iter()
                .find(|d| {
                    d.shard_idx == view.shard_idx
                        && d.data_offset <= view.data_offset
                        && view.data_offset + view.n_bytes <= d.data_offset + d.n_bytes
                })
                .unwrap();
            let offset = (view.data_offset - source.data_offset) as usize;
            assert_eq!(
                &payload[&source.name][offset..offset + view.n_bytes as usize],
                bytes_for(original),
                "{}",
                view.name
            );
            assert!(view.data_offset.is_multiple_of(32));
        }
        // Pre-quantized checkpoints skip the merges; both layouts admit.
        header.tensors = physical;
        let old = Qwen4ExpModel::from_gguf(&header).unwrap();
        assert_eq!(old.blocks.len(), 4);
        assert_eq!(old.tensors.len(), split.len());
    }

    #[test]
    fn inventory_refuses_missing_unknown_ambiguous_and_malformed_merged_tensors() {
        let c = alternate(8);
        let mut source = fixture(&c);
        source.pop();
        assert!(matches!(
            Qwen4ExpModel::from_inventory(c.clone(), &source, 0),
            Err(Qwen4ExpLoadError::Missing(_))
        ));
        let mut source = fixture(&c);
        let mut extra = source[0].clone();
        extra.name = "blk.99.ffn_gate_inp.weight".into();
        source.push(extra);
        assert!(matches!(
            Qwen4ExpModel::from_inventory(c.clone(), &source, 0),
            Err(Qwen4ExpLoadError::UnexpectedTensors(_))
        ));
        let mut source = fixture(&c);
        let duplicate = source[0].clone();
        source.push(duplicate);
        assert!(matches!(
            Qwen4ExpModel::from_inventory(c.clone(), &source, 0),
            Err(Qwen4ExpLoadError::AmbiguousLayout(_))
        ));
        let mut source = fixture(&c);
        merge(
            &mut source,
            "blk.0.attn_qkv_gate.weight",
            &["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"],
        );
        let index = source
            .iter()
            .position(|t| t.name == "blk.0.attn_qkv_gate.weight")
            .unwrap();
        source[index].shape[1] += 1;
        assert!(matches!(
            Qwen4ExpModel::from_inventory(c.clone(), &source, 0),
            Err(Qwen4ExpLoadError::Shape { .. })
        ));
        source[index].shape[1] -= 1;
        source.push(
            fixture(&c)
                .into_iter()
                .find(|t| t.name == "blk.0.attn_gate.weight")
                .unwrap(),
        );
        assert!(matches!(
            Qwen4ExpModel::from_inventory(c, &source, 0),
            Err(Qwen4ExpLoadError::AmbiguousLayout(_))
        ));
    }

    #[test]
    fn mtp_names_and_declared_appended_blocks_are_excluded_when_consistent() {
        let c = alternate(8);
        let mut source = fixture(&c);
        for name in [
            "nextn_fc_embedding.weight",
            "nextn.pre_projection.weight",
            "nextn.post_projection.weight",
            "blk.8.nextn.eh_proj.weight",
            "blk.8.nextn.hc_head_down.weight",
            "mtp.fc_hidden.weight",
            "blk.8.attn_q.weight",
        ] {
            let mut desc = source[0].clone();
            desc.name = name.into();
            desc.dtype = GgmlType::I32;
            source.push(desc);
        }
        let model = Qwen4ExpModel::from_inventory(c.clone(), &source, 1).unwrap();
        assert_eq!(model.ignored_mtp.len(), 7);
        assert_eq!(model.tensors.len(), fixture(&c).len());
    }

    #[test]
    fn mtp_inventory_must_match_declared_nextn_layers() {
        let c = alternate(8);
        let trunk = fixture(&c);
        let append = |source: &mut Vec<TensorDesc>, name: &str| {
            let mut desc = source[0].clone();
            desc.name = name.into();
            desc.dtype = GgmlType::I32;
            source.push(desc);
        };
        let mut signature = trunk.clone();
        append(&mut signature, "blk.8.nextn.eh_proj.weight");

        // Declared MTP with no signature tensor: previously this silently
        // consumed a trunk layer; now it must refuse.
        assert!(matches!(
            Qwen4ExpModel::from_inventory(c.clone(), &trunk, 1),
            Err(Qwen4ExpLoadError::MtpInventory(_))
        ));

        // Ordinary tensors in the appended range are not MTP evidence.
        let mut ordinary_only = trunk.clone();
        append(&mut ordinary_only, "blk.8.attn_q.weight");
        assert!(matches!(
            Qwen4ExpModel::from_inventory(c.clone(), &ordinary_only, 1),
            Err(Qwen4ExpLoadError::MtpInventory(_))
        ));

        // MTP tensors present but metadata declares none.
        assert!(matches!(
            Qwen4ExpModel::from_inventory(c.clone(), &signature, 0),
            Err(Qwen4ExpLoadError::MtpInventory(_))
        ));

        // Two declared blocks but only the first has a signature.
        assert!(matches!(
            Qwen4ExpModel::from_inventory(c.clone(), &signature, 2),
            Err(Qwen4ExpLoadError::MtpInventory(_))
        ));

        // Signatures outside the declared range are inconsistent.
        for name in ["blk.9.nextn.eh_proj.weight", "blk.7.nextn.eh_proj.weight"] {
            let mut misplaced = trunk.clone();
            append(&mut misplaced, name);
            assert!(matches!(
                Qwen4ExpModel::from_inventory(c.clone(), &misplaced, 1),
                Err(Qwen4ExpLoadError::MtpInventory(_))
            ));
        }

        // Unknown MTP roles refuse in both layouts.
        for name in [
            "blk.8.nextn.bogus.weight",
            "mtp.bogus.weight",
            "blk.8.nextn.eh_proj.bogus.weight",
            "mtp.fc.bogus.weight",
            "mtp.layers.999.bogus.weight",
            "blk.8.bogus.weight",
        ] {
            let mut unknown = trunk.clone();
            append(&mut unknown, name);
            assert!(matches!(
                Qwen4ExpModel::from_inventory(c.clone(), &unknown, 1),
                Err(Qwen4ExpLoadError::MtpInventory(_))
            ));
        }

        // A plain out-of-range block is still an unexpected tensor, not MTP.
        let mut unknown_block = trunk.clone();
        append(&mut unknown_block, "blk.8.attn_q.weight");
        assert!(matches!(
            Qwen4ExpModel::from_inventory(c.clone(), &unknown_block, 0),
            Err(Qwen4ExpLoadError::UnexpectedTensors(_))
        ));

        // Root-level MTP evidence is consistent with a declaration.
        let mut root_mtp = trunk.clone();
        append(&mut root_mtp, "nextn.pre_projection.weight");
        assert!(Qwen4ExpModel::from_inventory(c.clone(), &root_mtp, 1).is_ok());
        let mut root_mtp = trunk.clone();
        append(&mut root_mtp, "mtp.fc.weight");
        assert!(Qwen4ExpModel::from_inventory(c, &root_mtp, 1).is_ok());
    }

    fn quantized_fixture(config: &Qwen4ExpConfig) -> Vec<TensorDesc> {
        let mut offset = 0_u64;
        expected_tensor_inventory(config)
            .unwrap()
            .into_iter()
            .map(|(name, shape)| {
                let dtype = if name.ends_with("down_exps.weight") {
                    GgmlType::IQ4_NL
                } else if name.ends_with("exps.weight") {
                    GgmlType::IQ3_XXS
                } else if name == "output.weight" {
                    GgmlType::Q6_K
                } else if name == "per_layer_token_embd.weight" {
                    GgmlType::IQ4_NL
                } else if name.contains("indexer.q_proj") || name.contains("indexer.k_proj") {
                    GgmlType::BF16
                } else if name.ends_with("norm.weight")
                    || name.ends_with("norm_key.weight")
                    || name.ends_with("norm_query.weight")
                    || name.ends_with("norm_conv.weight")
                    || name.ends_with("inject.weight")
                    || name.ends_with("gate_inp.weight")
                    || name.ends_with("gate_inp_shexp.weight")
                    || name.ends_with("ssm_beta.weight")
                    || name.ends_with("ssm_alpha.weight")
                    || name.ends_with("ssm_a")
                    || name.ends_with("ssm_dt.bias")
                    || name.ends_with("ssm_conv1d.weight")
                    || name.ends_with("ple_conv1d.weight")
                {
                    GgmlType::F32
                } else {
                    GgmlType::Q8_0
                };
                let (block, bytes) = dtype.storage_layout().unwrap();
                assert!(
                    shape[0].is_multiple_of(block),
                    "{name} row width {} is not aligned to {block}",
                    shape[0]
                );
                offset = offset.next_multiple_of(32);
                let n_bytes = shape.iter().product::<u64>() / block * bytes;
                let desc = TensorDesc {
                    name,
                    shape,
                    dtype,
                    shard_idx: 0,
                    data_offset: offset,
                    n_bytes,
                };
                offset += n_bytes;
                desc
            })
            .collect()
    }

    #[test]
    fn quantized_merged_inventory_preserves_alignment_and_tied_output() {
        let config = alternate(8);
        let mut split = quantized_fixture(&config);
        // The converter quantizes each merged HC projection as one tensor;
        // both canonical halves therefore share the same Q8_0 row encoding.
        for d in &mut split {
            if d.name.ends_with("inject.weight") {
                d.dtype = GgmlType::Q8_0;
                d.n_bytes = d.shape.iter().product::<u64>() / 32 * 34;
            }
        }
        assert!(split.iter().all(|desc| desc.data_offset.is_multiple_of(32)));
        let mut source = split.clone();
        for layer in 0..config.layer_count {
            for (merged, parts) in [
                (
                    "hc_attn_down_inject",
                    vec!["hc_attn_down", "hc_attn_inject"],
                ),
                ("hc_ffn_down_inject", vec!["hc_ffn_down", "hc_ffn_inject"]),
            ] {
                let names: Vec<_> = parts
                    .iter()
                    .map(|part| format!("blk.{layer}.{part}.weight"))
                    .collect();
                merge(
                    &mut source,
                    &format!("blk.{layer}.{merged}.weight"),
                    &names.iter().map(String::as_str).collect::<Vec<_>>(),
                );
            }
        }
        merge(
            &mut source,
            "blk.0.attn_qkv_gate.weight",
            &["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"],
        );
        merge(
            &mut source,
            "blk.0.ssm_beta_alpha.weight",
            &["blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight"],
        );
        merge(
            &mut source,
            "blk.3.attn_qkv.weight",
            &[
                "blk.3.attn_q.weight",
                "blk.3.attn_k.weight",
                "blk.3.attn_v.weight",
            ],
        );
        // Lay out physical merged tensors anew: their bytes must not overlap
        // the former split offsets. Exercise both shards and nonzero offsets.
        let mut shard_ends = [4096u64, 8192];
        for (i, d) in source.iter_mut().enumerate() {
            d.shard_idx = i % 2;
            d.data_offset = shard_ends[d.shard_idx].next_multiple_of(32);
            shard_ends[d.shard_idx] = d.data_offset + d.n_bytes;
        }
        let model = Qwen4ExpModel::from_inventory(config.clone(), &source, 0).unwrap();
        let requests = model.residency_sources();
        let lengths = shard_ends.map(|n| (n + 4096) as usize);
        let retained =
            crate::metal::plan_retained_storage(&lengths, &requests, 4096, 8 << 30, 32).unwrap();
        assert_eq!(retained.entries.len(), requests.len());
        for (entry, desc) in retained.entries.iter().zip(&requests) {
            assert_eq!(entry.shard_idx, desc.shard_idx);
            assert_eq!(entry.data_offset, desc.data_offset);
            assert_eq!(entry.n_bytes, desc.n_bytes);
            assert!(matches!(
                entry.disposition,
                crate::metal::RetainedStorageDisposition::View { .. }
            ));
        }
        for canonical in &model.tensors {
            let original = split.iter().find(|d| d.name == canonical.name).unwrap();
            assert_eq!(canonical.name, original.name);
            assert_eq!(canonical.shape, original.shape);
            assert_eq!(canonical.dtype, original.dtype);
            assert_eq!(canonical.n_bytes, original.n_bytes);
            assert!(canonical.data_offset.is_multiple_of(32));
        }

        crate::qwen4exp_residency::validate_native_dtypes(&model.tensors).unwrap();
        let mut tied = split;
        tied.retain(|desc| desc.name != "output.weight");
        let model = Qwen4ExpModel::from_inventory(config, &tied, 0).unwrap();
        assert!(model.tied_embeddings);
        assert_eq!(model.output.name, "token_embd.weight");
        assert_eq!(model.output.dtype, GgmlType::Q8_0);
        assert!(
            !model
                .residency_sources()
                .iter()
                .any(|desc| desc.name == "output.weight")
        );
    }

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
