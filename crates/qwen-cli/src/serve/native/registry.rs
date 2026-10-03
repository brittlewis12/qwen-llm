//! Explicit CPU-bound assets; matrices are read only for an admitted execution.

use crate::bounded_file::read_regular_file_bounded;
use crate::linear_transport::deployment::{CpuDeployment, ExecutionMode};
use crate::linear_transport::{SCAN_BUFFER_BYTES, VerifiedTransport, parse_unique_json};
use anyhow::{Context, Result, ensure};
use qwen_llm::{gguf::GgufFile, runtime::LoadedModel};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Mutex,
};

const CONFIG_BYTES: usize = 64 * 1024;
const MAX_ASSETS: usize = 64;
pub(crate) const MAX_STAGED_BYTES: u64 = 512 * 1024 * 1024;
pub(crate) const STAGING_OVERHEAD_BYTES: u64 = SCAN_BUFFER_BYTES as u64 + 4 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema_version: u32,
    assets: Vec<AssetConfig>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AssetConfig {
    alias: String,
    path: PathBuf,
    #[serde(default)]
    allow_unvalidated_transfer: bool,
}

pub(crate) struct Registry {
    assets: BTreeMap<String, Asset>,
    deployment: CpuDeployment,
}
struct Asset {
    source_layers: Vec<u32>,
    matrix_bytes: u64,
    metadata: Value,
    transport: Mutex<VerifiedTransport>,
}
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct MatrixKey {
    pub(crate) alias: String,
    pub(crate) layer: u32,
}
#[derive(Default)]
pub(crate) struct Staged {
    pub(crate) matrices: BTreeMap<MatrixKey, Vec<u8>>,
}

impl Registry {
    /// Load this same retained GGUF after binding; the pathname is only a label.
    pub(crate) fn open(
        path: &Path,
        gguf: &GgufFile,
        checkpoint: &mut impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        checkpoint()?;
        let config: Config = serde_json::from_value(parse_unique_json(
            &read_regular_file_bounded(path, CONFIG_BYTES)?,
        )?)?;
        ensure!(
            config.schema_version == 1,
            "unsupported lens config version"
        );
        ensure!(
            config.assets.len() <= MAX_ASSETS,
            "too many configured lenses"
        );
        let mut aliases = BTreeSet::new();
        for asset in &config.assets {
            ensure!(
                !asset.alias.is_empty()
                    && asset.alias.len() <= 64
                    && asset
                        .alias
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
                    && asset.alias != "plain",
                "invalid or reserved lens alias"
            );
            ensure!(aliases.insert(&asset.alias), "duplicate lens alias");
            ensure!(!asset.path.as_os_str().is_empty(), "empty lens asset path");
        }
        let parent = path
            .parent()
            .context("lens config has no parent directory")?;
        let arch = qwen_llm::loader::Model::from_gguf(gguf)?.arch;
        let architecture = gguf
            .get_str("general.architecture")
            .context("missing architecture")?;
        let mut opened = Vec::new();
        for config in config.assets {
            let transport = VerifiedTransport::open_validated_checked(
                &parent.join(&config.path),
                |m| {
                    ensure!(
                        m.model.architecture == architecture
                            && m.model.n_layers == arch.n_layer
                            && m.model.hidden_size == arch.hidden_size
                            && m.model.vocab_size == arch.vocab_size,
                        "deployment geometry mismatch"
                    );
                    Ok(())
                },
                checkpoint,
            )
            .with_context(|| format!("verify configured lens {}", config.alias))?;
            opened.push((config, transport));
        }
        let exact = opened
            .iter()
            .any(|(_, t)| t.manifest().model.exact_binding.is_some());
        let deployment = CpuDeployment::from_gguf(gguf, ExecutionMode::Scalar, exact, checkpoint)?;
        Self::bind(opened, deployment, checkpoint)
    }

    fn bind(
        opened: Vec<(AssetConfig, VerifiedTransport)>,
        deployment: CpuDeployment,
        checkpoint: &mut impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        let mut assets = BTreeMap::new();
        for (config, transport) in opened {
            checkpoint()?;
            let binding = deployment
                .bind(&transport, config.allow_unvalidated_transfer)
                .with_context(|| format!("bind configured lens {}", config.alias))?;
            let m = transport.manifest();
            let metadata = json!({"alias":config.alias,"kind":"fitted_linear_transport",
                "identity":transport.manifest_digest()?,"identity_kind":"blake3_compact_json_sorted_objects_array_order_preserved",
                "payload_blake3":transport.payload_blake3(),"available":true,"unavailable_reason":null,
                "method":m.transport.method,"source_layers":m.transport.source_layers,"target_layer":m.transport.target_layer,
                "readout_modes":["full_vocabulary"],"direction_rows":[],"direction_covectors":[],
                "transfer":binding["status"],"binding":binding,
                "score_semantics":{"kind":"logit","softmax_applied":false,"normalization":"deployed_output_rmsnorm_and_lm_head",
                    "candidate_universe":"full_model_vocabulary","generation_distribution":false}});
            assets.insert(
                config.alias,
                Asset {
                    source_layers: m.transport.source_layers.clone(),
                    matrix_bytes: transport.matrix_bytes(),
                    metadata,
                    transport: Mutex::new(transport),
                },
            );
        }
        checkpoint()?;
        Ok(Self { assets, deployment })
    }

    pub(crate) fn validate_loaded(&self, loaded: &LoadedModel) -> Result<()> {
        CpuDeployment::validate_loaded_identity(Some(self.deployment.identity), loaded)
    }
    pub(crate) fn metadata(&self) -> impl Iterator<Item = &Value> {
        self.assets.values().map(|a| &a.metadata)
    }
    pub(crate) fn asset(&self, alias: &str) -> Result<&Value> {
        Ok(&self
            .assets
            .get(alias)
            .context("unknown fitted lens alias")?
            .metadata)
    }
    fn matrix(&self, key: &MatrixKey) -> Result<&Asset> {
        let asset = self
            .assets
            .get(&key.alias)
            .context("unknown fitted lens alias")?;
        ensure!(
            asset.source_layers.contains(&key.layer),
            "lens source layer absent"
        );
        Ok(asset)
    }
    pub(crate) fn matrix_bytes(&self, keys: &BTreeSet<MatrixKey>) -> Result<u64> {
        ensure!(
            keys.len() <= super::readouts::MAX_HEAD_EVALUATIONS,
            "too many unique staged matrices"
        );
        let mut total = 0u64;
        for key in keys {
            total = total
                .checked_add(self.matrix(key)?.matrix_bytes)
                .context("staged matrix byte count overflow")?;
        }
        ensure!(
            total <= MAX_STAGED_BYTES,
            "selected matrices exceed staging byte limit"
        );
        Ok(total)
    }
    pub(crate) fn stage(
        &self,
        keys: &BTreeSet<MatrixKey>,
        expected_bytes: u64,
        mut checkpoint: impl FnMut() -> Result<()>,
    ) -> Result<Staged> {
        ensure!(
            self.matrix_bytes(keys)? == expected_bytes,
            "staging authorization mismatch"
        );
        let mut staged = Staged::default();
        for key in keys {
            checkpoint()?;
            let bytes = self
                .matrix(key)?
                .transport
                .lock()
                .map_err(|_| anyhow::anyhow!("lens payload reader poisoned"))?
                .read_matrix_checked(key.layer, &mut checkpoint)?;
            staged.matrices.insert(key.clone(), bytes);
        }
        checkpoint()?;
        Ok(staged)
    }
}

#[cfg(test)]
pub(crate) mod tests;
