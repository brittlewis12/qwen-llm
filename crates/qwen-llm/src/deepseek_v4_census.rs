//! Deterministic storage census for the frozen DeepSeek V4 Flash-0731 asset.

use crate::deepseek_v4::{
    AttentionKind, AttentionLane, CompressorWeights, DeepSeekV4Model, HyperConnectionWeights,
    RouterWeights,
};
use crate::gguf::GgufFile;
use crate::tensor::{GgmlType, TensorDesc, ggml_type_layout};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};

pub const FLASH_0731_PROFILE: &str = "deepseek-v4-flash-0731";
const FLASH_0731_ASSET_ID: &str = "deepseek-v4-flash-0731-ud-iq3_xxs";
const FLASH_0731_CENSUS_SHA256: &str =
    "f4397fae14a6df04786324006ce41ea0489d4b246f68e742207446098684e4fc";
const FLASH_0731_SHARDS: [(&str, u64, &str); 4] = [
    (
        "DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf",
        5_257_664,
        "9758eb3d78e1afe8852543931703f4f1cd6fbb07f492d4ed853f5d2f6e43be5a",
    ),
    (
        "DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00002-of-00004.gguf",
        49_485_728_288,
        "afcfd59721d4da86bc3301e16ca624af202d8af3fa3f9fbbfbb04b3b47666cfd",
    ),
    (
        "DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00003-of-00004.gguf",
        49_437_886_752,
        "64eaf514a763597ba7bb50866583d8db5eabbbbce3cb2f616d749af3890155ca",
    ),
    (
        "DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00004-of-00004.gguf",
        4_071_015_712,
        "5df52988c56348a22d15da809e9ac4f0cc59cc1c412347f1481dda4685ce89b2",
    ),
];

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u16)]
pub enum DeepSeekV4TensorRole {
    TokenEmbedding = 0,
    OutputNorm = 1,
    Output = 2,
    OutputHcFunction = 3,
    OutputHcScale = 4,
    OutputHcBase = 5,
    AttentionNorm = 6,
    AttentionSinks = 7,
    AttentionQa = 8,
    AttentionQaNorm = 9,
    AttentionQb = 10,
    AttentionKv = 11,
    AttentionKvNorm = 12,
    AttentionOutputA = 13,
    AttentionOutputB = 14,
    AttentionCompressorKv = 15,
    AttentionCompressorGate = 16,
    AttentionCompressorApe = 17,
    AttentionCompressorNorm = 18,
    IndexerQ = 19,
    IndexerProjection = 20,
    IndexerCompressorKv = 21,
    IndexerCompressorGate = 22,
    IndexerCompressorApe = 23,
    IndexerCompressorNorm = 24,
    AttentionHcFunction = 25,
    AttentionHcScale = 26,
    AttentionHcBase = 27,
    FfnNorm = 28,
    RoutedGateInput = 29,
    RoutedGate = 30,
    RoutedUp = 31,
    RoutedDown = 32,
    SharedGate = 33,
    SharedUp = 34,
    SharedDown = 35,
    HashRouter = 36,
    RouterCorrectionBias = 37,
    FfnHcFunction = 38,
    FfnHcScale = 39,
    FfnHcBase = 40,
}

impl DeepSeekV4TensorRole {
    const ALL: [Self; 41] = [
        Self::TokenEmbedding,
        Self::OutputNorm,
        Self::Output,
        Self::OutputHcFunction,
        Self::OutputHcScale,
        Self::OutputHcBase,
        Self::AttentionNorm,
        Self::AttentionSinks,
        Self::AttentionQa,
        Self::AttentionQaNorm,
        Self::AttentionQb,
        Self::AttentionKv,
        Self::AttentionKvNorm,
        Self::AttentionOutputA,
        Self::AttentionOutputB,
        Self::AttentionCompressorKv,
        Self::AttentionCompressorGate,
        Self::AttentionCompressorApe,
        Self::AttentionCompressorNorm,
        Self::IndexerQ,
        Self::IndexerProjection,
        Self::IndexerCompressorKv,
        Self::IndexerCompressorGate,
        Self::IndexerCompressorApe,
        Self::IndexerCompressorNorm,
        Self::AttentionHcFunction,
        Self::AttentionHcScale,
        Self::AttentionHcBase,
        Self::FfnNorm,
        Self::RoutedGateInput,
        Self::RoutedGate,
        Self::RoutedUp,
        Self::RoutedDown,
        Self::SharedGate,
        Self::SharedUp,
        Self::SharedDown,
        Self::HashRouter,
        Self::RouterCorrectionBias,
        Self::FfnHcFunction,
        Self::FfnHcScale,
        Self::FfnHcBase,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TokenEmbedding => "token_embedding",
            Self::OutputNorm => "output_norm",
            Self::Output => "output",
            Self::OutputHcFunction => "output_hc_function",
            Self::OutputHcScale => "output_hc_scale",
            Self::OutputHcBase => "output_hc_base",
            Self::AttentionNorm => "attention_norm",
            Self::AttentionSinks => "attention_sinks",
            Self::AttentionQa => "attention_q_a",
            Self::AttentionQaNorm => "attention_q_a_norm",
            Self::AttentionQb => "attention_q_b",
            Self::AttentionKv => "attention_kv",
            Self::AttentionKvNorm => "attention_kv_norm",
            Self::AttentionOutputA => "attention_output_a",
            Self::AttentionOutputB => "attention_output_b",
            Self::AttentionCompressorKv => "attention_compressor_kv",
            Self::AttentionCompressorGate => "attention_compressor_gate",
            Self::AttentionCompressorApe => "attention_compressor_ape",
            Self::AttentionCompressorNorm => "attention_compressor_norm",
            Self::IndexerQ => "indexer_q",
            Self::IndexerProjection => "indexer_projection",
            Self::IndexerCompressorKv => "indexer_compressor_kv",
            Self::IndexerCompressorGate => "indexer_compressor_gate",
            Self::IndexerCompressorApe => "indexer_compressor_ape",
            Self::IndexerCompressorNorm => "indexer_compressor_norm",
            Self::AttentionHcFunction => "attention_hc_function",
            Self::AttentionHcScale => "attention_hc_scale",
            Self::AttentionHcBase => "attention_hc_base",
            Self::FfnNorm => "ffn_norm",
            Self::RoutedGateInput => "routed_gate_input",
            Self::RoutedGate => "routed_gate",
            Self::RoutedUp => "routed_up",
            Self::RoutedDown => "routed_down",
            Self::SharedGate => "shared_gate",
            Self::SharedUp => "shared_up",
            Self::SharedDown => "shared_down",
            Self::HashRouter => "hash_router",
            Self::RouterCorrectionBias => "router_correction_bias",
            Self::FfnHcFunction => "ffn_hc_function",
            Self::FfnHcScale => "ffn_hc_scale",
            Self::FfnHcBase => "ffn_hc_base",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StorageCensus {
    pub dtype: String,
    pub dtype_tag: i32,
    pub tensor_count: u64,
    pub element_count: u64,
    pub storage_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorStorageCensus {
    pub dtype: String,
    pub dtype_tag: i32,
    pub element_count: u64,
    pub storage_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShardCensus {
    pub index: u32,
    pub basename: String,
    pub file_bytes: u64,
    pub tensor_count: u64,
    pub tensor_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CensusTotals {
    pub shard_count: u32,
    pub file_bytes: u64,
    pub tensor_count: u64,
    pub element_count: u64,
    pub tensor_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RoleCensus {
    pub role: String,
    pub role_id: u16,
    pub storage: Vec<StorageCensus>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LayerCensus {
    pub layer: u32,
    pub attention: String,
    pub router: String,
    pub routed_gate: TensorStorageCensus,
    pub routed_up: TensorStorageCensus,
    pub routed_down: TensorStorageCensus,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeepSeekV4CensusV1 {
    pub schema_version: u32,
    pub profile: String,
    pub shards: Vec<ShardCensus>,
    pub totals: CensusTotals,
    pub dtypes: Vec<StorageCensus>,
    pub roles: Vec<RoleCensus>,
    pub layers: Vec<LayerCensus>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedShardIdentity {
    pub index: u32,
    pub basename: String,
    pub file_bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedDeepSeekV4AssetV1 {
    pub manifest_schema_version: u32,
    pub asset_id: String,
    pub shards: Vec<PinnedShardIdentity>,
    pub census_sha256: String,
    pub census: DeepSeekV4CensusV1,
}

#[derive(Debug, thiserror::Error)]
pub enum DeepSeekV4CensusError {
    #[error("census arithmetic overflow while accumulating {0}")]
    Overflow(&'static str),
    #[error("invalid census: {0}")]
    Invalid(String),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Default)]
struct MutableStorage {
    name: String,
    tensor_count: u64,
    element_count: u64,
    storage_bytes: u64,
}

impl MutableStorage {
    fn add(&mut self, desc: &TensorDesc) -> Result<(), DeepSeekV4CensusError> {
        self.name = desc.dtype.wire_name().into();
        self.tensor_count = checked_add(self.tensor_count, 1, "tensor count")?;
        self.element_count = checked_add(
            self.element_count,
            desc.checked_n_elements()
                .ok_or(DeepSeekV4CensusError::Overflow("tensor element count"))?,
            "element count",
        )?;
        self.storage_bytes = checked_add(self.storage_bytes, desc.n_bytes, "storage bytes")?;
        Ok(())
    }

    fn freeze(self, dtype_tag: i32) -> StorageCensus {
        StorageCensus {
            dtype: self.name,
            dtype_tag,
            tensor_count: self.tensor_count,
            element_count: self.element_count,
            storage_bytes: self.storage_bytes,
        }
    }
}

impl DeepSeekV4CensusV1 {
    /// Bind the strict Flash-0731 schema and census its storage descriptors.
    ///
    /// Strict binding validates the 9 MiB I32 hash-router payload. It does not
    /// scan or hash the quantized weight banks.
    pub fn from_gguf_flash_0731(gguf: &GgufFile) -> Result<Self, DeepSeekV4CensusError> {
        let model = DeepSeekV4Model::from_gguf_flash_0731(gguf)
            .map_err(|error| DeepSeekV4CensusError::Invalid(error.to_string()))?;
        Self::from_model(gguf, &model)
    }

    fn from_model(
        gguf: &GgufFile,
        model: &DeepSeekV4Model<'_>,
    ) -> Result<Self, DeepSeekV4CensusError> {
        let bindings = collect_bindings(model);
        if bindings.len() != model.source_tensor_count || bindings.len() != gguf.tensors.len() {
            return Err(DeepSeekV4CensusError::Invalid(format!(
                "typed binding count {} does not match model {} and GGUF {}",
                bindings.len(),
                model.source_tensor_count,
                gguf.tensors.len()
            )));
        }
        let source_by_name = gguf
            .tensors
            .iter()
            .map(|desc| (desc.name.as_str(), desc))
            .collect::<HashMap<_, _>>();
        if source_by_name.len() != gguf.tensors.len()
            || bindings.iter().any(|(_, desc)| {
                source_by_name
                    .get(desc.name.as_str())
                    .is_none_or(|source| !std::ptr::eq(*desc, *source))
            })
        {
            return Err(DeepSeekV4CensusError::Invalid(
                "typed census bindings do not belong to the supplied GGUF".into(),
            ));
        }
        let bound_names = bindings
            .iter()
            .map(|(_, desc)| desc.name.as_str())
            .collect::<HashSet<_>>();
        if bound_names.len() != bindings.len()
            || gguf
                .tensors
                .iter()
                .any(|desc| !bound_names.contains(desc.name.as_str()))
        {
            return Err(DeepSeekV4CensusError::Invalid(
                "typed census bindings are duplicate or incomplete".into(),
            ));
        }

        let mut shard_rows = gguf
            .shards
            .iter()
            .enumerate()
            .map(|(index, shard)| {
                Ok(ShardCensus {
                    index: u32::try_from(index).map_err(|_| {
                        DeepSeekV4CensusError::Invalid("shard index exceeds u32".into())
                    })?,
                    basename: shard
                        .path
                        .file_name()
                        .ok_or_else(|| {
                            DeepSeekV4CensusError::Invalid(format!(
                                "shard path has no basename: {}",
                                shard.path.display()
                            ))
                        })?
                        .to_string_lossy()
                        .into_owned(),
                    file_bytes: u64::try_from(shard.mmap_len()).map_err(|_| {
                        DeepSeekV4CensusError::Invalid("shard size exceeds u64".into())
                    })?,
                    tensor_count: 0,
                    tensor_bytes: 0,
                })
            })
            .collect::<Result<Vec<_>, DeepSeekV4CensusError>>()?;
        let mut dtype_rows = BTreeMap::<i32, MutableStorage>::new();
        let mut total_elements = 0_u64;
        let mut total_tensor_bytes = 0_u64;
        for desc in &gguf.tensors {
            let shard = shard_rows.get_mut(desc.shard_idx).ok_or_else(|| {
                DeepSeekV4CensusError::Invalid(format!(
                    "tensor {} references missing shard {}",
                    desc.name, desc.shard_idx
                ))
            })?;
            shard.tensor_count = checked_add(shard.tensor_count, 1, "shard tensor count")?;
            shard.tensor_bytes =
                checked_add(shard.tensor_bytes, desc.n_bytes, "shard tensor bytes")?;
            dtype_rows.entry(desc.dtype as i32).or_default().add(desc)?;
            total_elements = checked_add(
                total_elements,
                desc.checked_n_elements()
                    .ok_or(DeepSeekV4CensusError::Overflow("total tensor elements"))?,
                "total tensor elements",
            )?;
            total_tensor_bytes =
                checked_add(total_tensor_bytes, desc.n_bytes, "total tensor bytes")?;
        }

        let mut role_rows = BTreeMap::<(u16, i32), MutableStorage>::new();
        for (role, desc) in bindings {
            role_rows
                .entry((role as u16, desc.dtype as i32))
                .or_default()
                .add(desc)?;
        }
        let mut roles = Vec::with_capacity(DeepSeekV4TensorRole::ALL.len());
        for role in DeepSeekV4TensorRole::ALL {
            let role_id = role as u16;
            let keys = role_rows
                .range((role_id, i32::MIN)..=(role_id, i32::MAX))
                .map(|(key, _)| *key)
                .collect::<Vec<_>>();
            let storage = keys
                .into_iter()
                .map(|key| {
                    role_rows
                        .remove(&key)
                        .expect("role key came from the same map")
                        .freeze(key.1)
                })
                .collect::<Vec<_>>();
            if storage.is_empty() {
                return Err(DeepSeekV4CensusError::Invalid(format!(
                    "role {} has no tensors",
                    role.as_str()
                )));
            }
            roles.push(RoleCensus {
                role: role.as_str().into(),
                role_id: role as u16,
                storage,
            });
        }
        if !role_rows.is_empty() {
            return Err(DeepSeekV4CensusError::Invalid(
                "unrecognized role IDs remain after aggregation".into(),
            ));
        }

        let layers = model
            .blocks
            .iter()
            .enumerate()
            .map(|(layer, block)| {
                let attention_kind = model
                    .config
                    .attention_kinds
                    .get(layer)
                    .copied()
                    .ok_or_else(|| {
                        DeepSeekV4CensusError::Invalid(format!(
                            "layer {layer} has no attention schedule entry"
                        ))
                    })?;
                Ok(LayerCensus {
                    layer: u32::try_from(layer).map_err(|_| {
                        DeepSeekV4CensusError::Invalid("layer index exceeds u32".into())
                    })?,
                    attention: attention_name(attention_kind).into(),
                    router: match block.moe.router {
                        RouterWeights::TokenHash { .. } => "hash",
                        RouterWeights::Learned { .. } => "learned",
                    }
                    .into(),
                    routed_gate: tensor_storage(block.moe.gate_experts)?,
                    routed_up: tensor_storage(block.moe.up_experts)?,
                    routed_down: tensor_storage(block.moe.down_experts)?,
                })
            })
            .collect::<Result<Vec<_>, DeepSeekV4CensusError>>()?;

        let file_bytes = shard_rows.iter().try_fold(0_u64, |total, shard| {
            checked_add(total, shard.file_bytes, "total file bytes")
        })?;
        let census = Self {
            schema_version: 1,
            profile: FLASH_0731_PROFILE.into(),
            totals: CensusTotals {
                shard_count: u32::try_from(shard_rows.len()).map_err(|_| {
                    DeepSeekV4CensusError::Invalid("shard count exceeds u32".into())
                })?,
                file_bytes,
                tensor_count: u64::try_from(gguf.tensors.len()).map_err(|_| {
                    DeepSeekV4CensusError::Invalid("tensor count exceeds u64".into())
                })?,
                element_count: total_elements,
                tensor_bytes: total_tensor_bytes,
            },
            shards: shard_rows,
            dtypes: dtype_rows
                .into_iter()
                .map(|(dtype_tag, row)| row.freeze(dtype_tag))
                .collect(),
            roles,
            layers,
        };
        census.validate()?;
        Ok(census)
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>, DeepSeekV4CensusError> {
        self.validate()?;
        Ok(serde_json::to_vec(self)?)
    }

    pub fn sha256(&self) -> Result<String, DeepSeekV4CensusError> {
        Ok(sha256_hex(&self.canonical_json()?))
    }

    pub fn validate(&self) -> Result<(), DeepSeekV4CensusError> {
        if self.schema_version != 1 || self.profile != FLASH_0731_PROFILE {
            return Err(DeepSeekV4CensusError::Invalid(
                "unsupported schema version or profile".into(),
            ));
        }
        if self.shards.len() != self.totals.shard_count as usize
            || self.totals.shard_count != 4
            || self.totals.tensor_count != 1_328
            || self.layers.len() != 43
            || self.roles.len() != DeepSeekV4TensorRole::ALL.len()
        {
            return Err(DeepSeekV4CensusError::Invalid(
                "shard, layer, or role cardinality changed".into(),
            ));
        }
        for (index, shard) in self.shards.iter().enumerate() {
            if shard.index as usize != index
                || shard.basename.is_empty()
                || shard.file_bytes == 0
                || shard.tensor_bytes > shard.file_bytes
                || (shard.tensor_count == 0) != (shard.tensor_bytes == 0)
            {
                return Err(DeepSeekV4CensusError::Invalid(
                    "shard rows are malformed or out of canonical order".into(),
                ));
            }
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.layer as usize != index
                || layer.attention != attention_name(flash_0731_attention_kind(index))
                || layer.router != if index < 3 { "hash" } else { "learned" }
            {
                return Err(DeepSeekV4CensusError::Invalid(format!(
                    "layer {index} schedule or router changed"
                )));
            }
            validate_tensor_storage(&layer.routed_gate)?;
            validate_tensor_storage(&layer.routed_up)?;
            validate_tensor_storage(&layer.routed_down)?;
            if layer.routed_gate != layer.routed_up {
                return Err(DeepSeekV4CensusError::Invalid(format!(
                    "layer {index} routed gate/up storage differs"
                )));
            }
        }
        for (expected, role) in DeepSeekV4TensorRole::ALL.iter().zip(&self.roles) {
            if role.role_id != *expected as u16
                || role.role != expected.as_str()
                || role.storage.is_empty()
                || !strictly_sorted(role.storage.iter().map(|row| row.dtype_tag))
            {
                return Err(DeepSeekV4CensusError::Invalid(format!(
                    "role {} is malformed or out of order",
                    expected.as_str()
                )));
            }
            for row in &role.storage {
                validate_storage(row)?;
            }
            let role_count = checked_sum(
                role.storage.iter().map(|row| row.tensor_count),
                "role tensor count",
            )?;
            if role_count != expected_role_tensor_count(*expected) {
                return Err(DeepSeekV4CensusError::Invalid(format!(
                    "role {} has {role_count} tensors, expected {}",
                    expected.as_str(),
                    expected_role_tensor_count(*expected)
                )));
            }
        }
        if !strictly_sorted(self.dtypes.iter().map(|row| row.dtype_tag)) {
            return Err(DeepSeekV4CensusError::Invalid(
                "dtype rows are not in canonical tag order".into(),
            ));
        }
        for row in &self.dtypes {
            validate_storage(row)?;
        }

        let shard_file_bytes = checked_sum(
            self.shards.iter().map(|row| row.file_bytes),
            "shard file bytes",
        )?;
        let shard_tensor_count = checked_sum(
            self.shards.iter().map(|row| row.tensor_count),
            "shard tensor count",
        )?;
        let shard_tensor_bytes = checked_sum(
            self.shards.iter().map(|row| row.tensor_bytes),
            "shard tensor bytes",
        )?;
        let dtype_tensor_count = checked_sum(
            self.dtypes.iter().map(|row| row.tensor_count),
            "dtype tensor count",
        )?;
        let dtype_element_count = checked_sum(
            self.dtypes.iter().map(|row| row.element_count),
            "dtype element count",
        )?;
        let dtype_tensor_bytes = checked_sum(
            self.dtypes.iter().map(|row| row.storage_bytes),
            "dtype tensor bytes",
        )?;
        let role_tensor_count = checked_sum(
            self.roles
                .iter()
                .flat_map(|role| role.storage.iter())
                .map(|row| row.tensor_count),
            "role tensor count",
        )?;
        let role_element_count = checked_sum(
            self.roles
                .iter()
                .flat_map(|role| role.storage.iter())
                .map(|row| row.element_count),
            "role element count",
        )?;
        let role_tensor_bytes = checked_sum(
            self.roles
                .iter()
                .flat_map(|role| role.storage.iter())
                .map(|row| row.storage_bytes),
            "role tensor bytes",
        )?;
        if shard_file_bytes != self.totals.file_bytes
            || shard_tensor_count != self.totals.tensor_count
            || shard_tensor_bytes != self.totals.tensor_bytes
            || dtype_tensor_count != self.totals.tensor_count
            || dtype_element_count != self.totals.element_count
            || dtype_tensor_bytes != self.totals.tensor_bytes
            || role_tensor_count != self.totals.tensor_count
            || role_element_count != self.totals.element_count
            || role_tensor_bytes != self.totals.tensor_bytes
        {
            return Err(DeepSeekV4CensusError::Invalid(
                "census subtotals do not match totals".into(),
            ));
        }
        if aggregate_storage(self.roles.iter().flat_map(|role| role.storage.iter()))? != self.dtypes
        {
            return Err(DeepSeekV4CensusError::Invalid(
                "global dtype rows do not match role dtype rows".into(),
            ));
        }
        for (role, rows) in [
            (
                DeepSeekV4TensorRole::RoutedGate,
                self.layers
                    .iter()
                    .map(|layer| &layer.routed_gate)
                    .collect::<Vec<_>>(),
            ),
            (
                DeepSeekV4TensorRole::RoutedUp,
                self.layers
                    .iter()
                    .map(|layer| &layer.routed_up)
                    .collect::<Vec<_>>(),
            ),
            (
                DeepSeekV4TensorRole::RoutedDown,
                self.layers
                    .iter()
                    .map(|layer| &layer.routed_down)
                    .collect::<Vec<_>>(),
            ),
        ] {
            if aggregate_tensor_storage(rows.into_iter())? != self.roles[role as usize].storage {
                return Err(DeepSeekV4CensusError::Invalid(format!(
                    "layer storage does not match role {}",
                    role.as_str()
                )));
            }
        }
        Ok(())
    }
}

impl PinnedDeepSeekV4AssetV1 {
    pub fn parse(json: &str) -> Result<Self, DeepSeekV4CensusError> {
        let asset: Self = serde_json::from_str(json)?;
        asset.validate()?;
        Ok(asset)
    }

    pub fn validate(&self) -> Result<(), DeepSeekV4CensusError> {
        if self.manifest_schema_version != 1 || self.asset_id != FLASH_0731_ASSET_ID {
            return Err(DeepSeekV4CensusError::Invalid(
                "unsupported asset manifest or asset ID".into(),
            ));
        }
        self.census.validate()?;
        if self.census_sha256 != FLASH_0731_CENSUS_SHA256 {
            return Err(DeepSeekV4CensusError::Invalid(
                "unsupported pinned census digest".into(),
            ));
        }
        if self.shards.len() != self.census.shards.len()
            || self.census.sha256()? != self.census_sha256
        {
            return Err(DeepSeekV4CensusError::Invalid(
                "pinned shard count or census digest changed".into(),
            ));
        }
        for (index, ((identity, census), expected)) in self
            .shards
            .iter()
            .zip(&self.census.shards)
            .zip(FLASH_0731_SHARDS)
            .enumerate()
        {
            let (basename, file_bytes, sha256) = expected;
            if identity.index != census.index
                || identity.index as usize != index
                || identity.basename != census.basename
                || identity.basename != basename
                || identity.file_bytes != census.file_bytes
                || identity.file_bytes != file_bytes
                || identity.sha256 != sha256
            {
                return Err(DeepSeekV4CensusError::Invalid(format!(
                    "pinned shard {} identity is malformed",
                    identity.index
                )));
            }
        }
        Ok(())
    }

    /// Rebuild the strict census and compare it with the frozen manifest.
    ///
    /// This validates hash-router values but does not hash the quantized shard
    /// payloads; shard SHA-256 verification belongs to asset provisioning.
    pub fn validate_observed(&self, gguf: &GgufFile) -> Result<(), DeepSeekV4CensusError> {
        self.validate()?;
        let observed = DeepSeekV4CensusV1::from_gguf_flash_0731(gguf)?;
        if observed != self.census {
            return Err(DeepSeekV4CensusError::Invalid(
                "observed model census differs from pinned asset".into(),
            ));
        }
        Ok(())
    }
}

fn collect_bindings<'a>(
    model: &DeepSeekV4Model<'a>,
) -> Vec<(DeepSeekV4TensorRole, &'a TensorDesc)> {
    let mut bindings = Vec::with_capacity(model.source_tensor_count);
    bindings.push((DeepSeekV4TensorRole::TokenEmbedding, model.token_embedding));
    bindings.push((DeepSeekV4TensorRole::OutputNorm, model.output_norm));
    bindings.push((DeepSeekV4TensorRole::Output, model.output));
    push_hyper_connection(
        &mut bindings,
        &model.output_hyper_connection,
        DeepSeekV4TensorRole::OutputHcFunction,
        DeepSeekV4TensorRole::OutputHcScale,
        DeepSeekV4TensorRole::OutputHcBase,
    );
    for block in &model.blocks {
        bindings.extend([
            (DeepSeekV4TensorRole::AttentionNorm, block.attention.norm),
            (DeepSeekV4TensorRole::AttentionSinks, block.attention.sinks),
            (DeepSeekV4TensorRole::AttentionQa, block.attention.q_a),
            (
                DeepSeekV4TensorRole::AttentionQaNorm,
                block.attention.q_a_norm,
            ),
            (DeepSeekV4TensorRole::AttentionQb, block.attention.q_b),
            (DeepSeekV4TensorRole::AttentionKv, block.attention.kv),
            (
                DeepSeekV4TensorRole::AttentionKvNorm,
                block.attention.kv_norm,
            ),
            (
                DeepSeekV4TensorRole::AttentionOutputA,
                block.attention.output_a,
            ),
            (
                DeepSeekV4TensorRole::AttentionOutputB,
                block.attention.output_b,
            ),
        ]);
        match &block.attention.lane {
            AttentionLane::SlidingWindow => {}
            AttentionLane::CompressedSparse {
                compressor,
                indexer,
            } => {
                push_compressor(
                    &mut bindings,
                    compressor,
                    DeepSeekV4TensorRole::AttentionCompressorKv,
                    DeepSeekV4TensorRole::AttentionCompressorGate,
                    DeepSeekV4TensorRole::AttentionCompressorApe,
                    DeepSeekV4TensorRole::AttentionCompressorNorm,
                );
                bindings.extend([
                    (DeepSeekV4TensorRole::IndexerQ, indexer.q),
                    (DeepSeekV4TensorRole::IndexerProjection, indexer.projection),
                ]);
                push_compressor(
                    &mut bindings,
                    &indexer.compressor,
                    DeepSeekV4TensorRole::IndexerCompressorKv,
                    DeepSeekV4TensorRole::IndexerCompressorGate,
                    DeepSeekV4TensorRole::IndexerCompressorApe,
                    DeepSeekV4TensorRole::IndexerCompressorNorm,
                );
            }
            AttentionLane::HeavilyCompressed { compressor } => push_compressor(
                &mut bindings,
                compressor,
                DeepSeekV4TensorRole::AttentionCompressorKv,
                DeepSeekV4TensorRole::AttentionCompressorGate,
                DeepSeekV4TensorRole::AttentionCompressorApe,
                DeepSeekV4TensorRole::AttentionCompressorNorm,
            ),
        }
        push_hyper_connection(
            &mut bindings,
            &block.attention_hyper_connection,
            DeepSeekV4TensorRole::AttentionHcFunction,
            DeepSeekV4TensorRole::AttentionHcScale,
            DeepSeekV4TensorRole::AttentionHcBase,
        );
        bindings.extend([
            (DeepSeekV4TensorRole::FfnNorm, block.ffn_norm),
            (DeepSeekV4TensorRole::RoutedGateInput, block.moe.gate_input),
            (DeepSeekV4TensorRole::RoutedGate, block.moe.gate_experts),
            (DeepSeekV4TensorRole::RoutedUp, block.moe.up_experts),
            (DeepSeekV4TensorRole::RoutedDown, block.moe.down_experts),
            (DeepSeekV4TensorRole::SharedGate, block.moe.gate_shared),
            (DeepSeekV4TensorRole::SharedUp, block.moe.up_shared),
            (DeepSeekV4TensorRole::SharedDown, block.moe.down_shared),
        ]);
        match block.moe.router {
            RouterWeights::TokenHash { token_to_expert } => {
                bindings.push((DeepSeekV4TensorRole::HashRouter, token_to_expert));
            }
            RouterWeights::Learned { correction_bias } => {
                bindings.push((DeepSeekV4TensorRole::RouterCorrectionBias, correction_bias))
            }
        }
        push_hyper_connection(
            &mut bindings,
            &block.ffn_hyper_connection,
            DeepSeekV4TensorRole::FfnHcFunction,
            DeepSeekV4TensorRole::FfnHcScale,
            DeepSeekV4TensorRole::FfnHcBase,
        );
    }
    bindings
}

fn push_compressor<'a>(
    bindings: &mut Vec<(DeepSeekV4TensorRole, &'a TensorDesc)>,
    compressor: &CompressorWeights<'a>,
    kv: DeepSeekV4TensorRole,
    gate: DeepSeekV4TensorRole,
    ape: DeepSeekV4TensorRole,
    norm: DeepSeekV4TensorRole,
) {
    bindings.extend([
        (kv, compressor.kv),
        (gate, compressor.gate),
        (ape, compressor.ape),
        (norm, compressor.norm),
    ]);
}

fn push_hyper_connection<'a>(
    bindings: &mut Vec<(DeepSeekV4TensorRole, &'a TensorDesc)>,
    hyper_connection: &HyperConnectionWeights<'a>,
    function: DeepSeekV4TensorRole,
    scale: DeepSeekV4TensorRole,
    base: DeepSeekV4TensorRole,
) {
    bindings.extend([
        (function, hyper_connection.function),
        (scale, hyper_connection.scale),
        (base, hyper_connection.base),
    ]);
}

fn tensor_storage(desc: &TensorDesc) -> Result<TensorStorageCensus, DeepSeekV4CensusError> {
    Ok(TensorStorageCensus {
        dtype: desc.dtype.wire_name().into(),
        dtype_tag: desc.dtype as i32,
        element_count: desc
            .checked_n_elements()
            .ok_or(DeepSeekV4CensusError::Overflow(
                "grouped MoE tensor elements",
            ))?,
        storage_bytes: desc.n_bytes,
    })
}

fn expected_role_tensor_count(role: DeepSeekV4TensorRole) -> u64 {
    match role {
        DeepSeekV4TensorRole::TokenEmbedding
        | DeepSeekV4TensorRole::OutputNorm
        | DeepSeekV4TensorRole::Output
        | DeepSeekV4TensorRole::OutputHcFunction
        | DeepSeekV4TensorRole::OutputHcScale
        | DeepSeekV4TensorRole::OutputHcBase => 1,
        DeepSeekV4TensorRole::AttentionCompressorKv
        | DeepSeekV4TensorRole::AttentionCompressorGate
        | DeepSeekV4TensorRole::AttentionCompressorApe
        | DeepSeekV4TensorRole::AttentionCompressorNorm => 41,
        DeepSeekV4TensorRole::IndexerQ
        | DeepSeekV4TensorRole::IndexerProjection
        | DeepSeekV4TensorRole::IndexerCompressorKv
        | DeepSeekV4TensorRole::IndexerCompressorGate
        | DeepSeekV4TensorRole::IndexerCompressorApe
        | DeepSeekV4TensorRole::IndexerCompressorNorm => 21,
        DeepSeekV4TensorRole::HashRouter => 3,
        DeepSeekV4TensorRole::RouterCorrectionBias => 40,
        _ => 43,
    }
}

fn validate_storage(row: &StorageCensus) -> Result<(), DeepSeekV4CensusError> {
    if row.tensor_count == 0 || row.element_count == 0 || row.storage_bytes == 0 {
        return Err(DeepSeekV4CensusError::Invalid(format!(
            "{} storage row contains a zero count",
            row.dtype
        )));
    }
    validate_dtype_storage(
        &row.dtype,
        row.dtype_tag,
        row.element_count,
        row.storage_bytes,
    )
}

fn validate_tensor_storage(row: &TensorStorageCensus) -> Result<(), DeepSeekV4CensusError> {
    if row.element_count == 0 || row.storage_bytes == 0 {
        return Err(DeepSeekV4CensusError::Invalid(format!(
            "{} tensor storage contains a zero count",
            row.dtype
        )));
    }
    validate_dtype_storage(
        &row.dtype,
        row.dtype_tag,
        row.element_count,
        row.storage_bytes,
    )
}

fn validate_dtype_storage(
    wire_name: &str,
    dtype_tag: i32,
    element_count: u64,
    storage_bytes: u64,
) -> Result<(), DeepSeekV4CensusError> {
    let raw = u32::try_from(dtype_tag)
        .map_err(|_| DeepSeekV4CensusError::Invalid(format!("negative dtype tag {dtype_tag}")))?;
    let dtype = GgmlType::from_raw(raw);
    if dtype == GgmlType::Unknown || dtype.wire_name() != wire_name {
        return Err(DeepSeekV4CensusError::Invalid(format!(
            "dtype tag {dtype_tag} does not match wire name {wire_name}"
        )));
    }
    let (block_size, type_size) = ggml_type_layout(dtype).ok_or_else(|| {
        DeepSeekV4CensusError::Invalid(format!("dtype {wire_name} has no storage layout"))
    })?;
    if block_size == 0 || element_count % block_size != 0 {
        return Err(DeepSeekV4CensusError::Invalid(format!(
            "{wire_name} element count {element_count} is not block aligned"
        )));
    }
    let expected_bytes = element_count
        .checked_div(block_size)
        .and_then(|blocks| blocks.checked_mul(type_size))
        .ok_or(DeepSeekV4CensusError::Overflow("dtype storage bytes"))?;
    if expected_bytes != storage_bytes {
        return Err(DeepSeekV4CensusError::Invalid(format!(
            "{wire_name} storage has {storage_bytes} bytes, expected {expected_bytes}"
        )));
    }
    Ok(())
}

fn aggregate_storage<'a>(
    rows: impl IntoIterator<Item = &'a StorageCensus>,
) -> Result<Vec<StorageCensus>, DeepSeekV4CensusError> {
    let mut aggregated = BTreeMap::<i32, StorageCensus>::new();
    for row in rows {
        let entry = aggregated
            .entry(row.dtype_tag)
            .or_insert_with(|| StorageCensus {
                dtype: row.dtype.clone(),
                dtype_tag: row.dtype_tag,
                tensor_count: 0,
                element_count: 0,
                storage_bytes: 0,
            });
        if entry.dtype != row.dtype {
            return Err(DeepSeekV4CensusError::Invalid(format!(
                "dtype tag {} has conflicting wire names",
                row.dtype_tag
            )));
        }
        entry.tensor_count = checked_add(
            entry.tensor_count,
            row.tensor_count,
            "aggregate tensor count",
        )?;
        entry.element_count = checked_add(
            entry.element_count,
            row.element_count,
            "aggregate element count",
        )?;
        entry.storage_bytes = checked_add(
            entry.storage_bytes,
            row.storage_bytes,
            "aggregate storage bytes",
        )?;
    }
    Ok(aggregated.into_values().collect())
}

fn aggregate_tensor_storage<'a>(
    rows: impl IntoIterator<Item = &'a TensorStorageCensus>,
) -> Result<Vec<StorageCensus>, DeepSeekV4CensusError> {
    let rows = rows
        .into_iter()
        .map(|row| StorageCensus {
            dtype: row.dtype.clone(),
            dtype_tag: row.dtype_tag,
            tensor_count: 1,
            element_count: row.element_count,
            storage_bytes: row.storage_bytes,
        })
        .collect::<Vec<_>>();
    aggregate_storage(rows.iter())
}

fn attention_name(kind: AttentionKind) -> &'static str {
    match kind {
        AttentionKind::SlidingWindow => "local",
        AttentionKind::CompressedSparse => "csa",
        AttentionKind::HeavilyCompressed => "hca",
    }
}

fn flash_0731_attention_kind(layer: usize) -> AttentionKind {
    if layer < 2 {
        AttentionKind::SlidingWindow
    } else if layer.is_multiple_of(2) {
        AttentionKind::CompressedSparse
    } else {
        AttentionKind::HeavilyCompressed
    }
}

fn checked_add(left: u64, right: u64, field: &'static str) -> Result<u64, DeepSeekV4CensusError> {
    left.checked_add(right)
        .ok_or(DeepSeekV4CensusError::Overflow(field))
}

fn checked_sum(
    values: impl IntoIterator<Item = u64>,
    field: &'static str,
) -> Result<u64, DeepSeekV4CensusError> {
    values
        .into_iter()
        .try_fold(0_u64, |total, value| checked_add(total, value, field))
}

fn strictly_sorted(values: impl IntoIterator<Item = i32>) -> bool {
    let mut previous = None;
    for value in values {
        if previous.is_some_and(|previous| previous >= value) {
            return false;
        }
        previous = Some(value);
    }
    true
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_wire_order_is_closed_and_stable() {
        for (index, role) in DeepSeekV4TensorRole::ALL.iter().enumerate() {
            assert_eq!(*role as usize, index);
            assert!(!role.as_str().is_empty());
        }
        assert_eq!(DeepSeekV4TensorRole::ALL.len(), 41);
    }

    #[test]
    fn checked_accumulation_rejects_overflow() {
        assert!(checked_add(u64::MAX, 1, "test").is_err());
        assert!(checked_sum([u64::MAX, 1], "test").is_err());
    }

    #[test]
    fn profile_schedule_is_canonical() {
        let kinds = (0..43).map(flash_0731_attention_kind).collect::<Vec<_>>();
        assert_eq!(
            kinds
                .iter()
                .filter(|&&kind| kind == AttentionKind::SlidingWindow)
                .count(),
            2
        );
        assert_eq!(
            kinds
                .iter()
                .filter(|&&kind| kind == AttentionKind::CompressedSparse)
                .count(),
            21
        );
        assert_eq!(
            kinds
                .iter()
                .filter(|&&kind| kind == AttentionKind::HeavilyCompressed)
                .count(),
            20
        );
    }
}
