//! Shared retained-file access for published imports and data-only producers.
use super::*;
use crate::linear_transport::VerifiedTransport;

enum Storage {
    Legacy {
        manifest: FullLensManifest,
        file: std::fs::File,
        matrix_hashes: Vec<blake3::Hash>,
    },
    Data(VerifiedTransport),
}

pub(crate) struct FullAccess {
    storage: Storage,
    pub(super) transport: FullTransport,
    pub(super) payload: FullPayload,
    pub(super) runtime_binding: Option<serde_json::Value>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FullExecutionMode {
    Scalar,
    Packed,
    Projection,
}

/// Retains the verified payload and the CPU binding to the GGUF subsequently loaded.
pub(crate) struct BoundFullAccess {
    access: FullAccess,
    deployment_identity: Option<(u64, u64)>,
}

impl std::ops::Deref for BoundFullAccess {
    type Target = FullAccess;
    fn deref(&self) -> &Self::Target {
        &self.access
    }
}
impl std::ops::DerefMut for BoundFullAccess {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.access
    }
}
impl BoundFullAccess {
    pub(super) fn validate_loaded(&self, loaded: &LoadedModel) -> Result<()> {
        match &self.access.storage {
            Storage::Legacy { manifest, .. } => validate_deployed_model(manifest, loaded),
            Storage::Data(_) => {
                let identity = loaded.workspace_lens_identity();
                ensure!(
                    self.deployment_identity
                        == Some((identity.model_locator_id, identity.tokenizer_metadata_id)),
                    "loaded deployment differs from CPU-bound retained GGUF"
                );
                loaded.validate_passive_workspace_lens_output()?;
                Ok(())
            }
        }
    }
}

pub(crate) struct CpuFullDeployment {
    architecture: String,
    arch: qwen_llm::model::Arch,
    identity: (u64, u64),
    content: Option<String>,
    content_bytes_hashed: u64,
}
impl CpuFullDeployment {
    pub(crate) fn from_gguf(
        gguf: &GgufFile,
        mode: FullExecutionMode,
        exact: bool,
        _cache: Option<&Path>,
    ) -> Result<Self> {
        let bound = qwen_llm::loader::Model::from_gguf(gguf)
            .context("bind CPU Qwen geometry and tensor inventory")?;
        let arch = bound.arch;
        ensure!(
            mode != FullExecutionMode::Packed || arch.kind == qwen_llm::model::ArchKind::Dense,
            "packed capture is not exposed by the workspace-lens facade for ordinary MoE; use scalar read-full"
        );
        qwen_llm::workspace_lens::validate_opened_output_head(
            gguf,
            mode != FullExecutionMode::Scalar,
        )?;
        let identity = qwen_llm::runtime::opened_gguf_lightweight_identity_parts(gguf)?;
        let (content, content_bytes_hashed) = if exact {
            let verified =
                qwen_llm::checkpoint_identity::verified_checkpoint_content_identity(gguf)?;
            (Some(hex(&verified.content_id)), verified.bytes_hashed)
        } else {
            (None, 0)
        };
        Ok(Self {
            architecture: gguf
                .get_str("general.architecture")
                .context("missing runtime architecture")?
                .into(),
            arch,
            identity,
            content,
            content_bytes_hashed,
        })
    }
}

impl FullAccess {
    pub(crate) fn is_data_directory(directory: &Path) -> Result<bool> {
        let probe: serde_json::Value = read_json_file(&directory.join(FULL_MANIFEST_NAME))?;
        Ok(probe.get("schema").and_then(serde_json::Value::as_str)
            == Some("llm.lens.linear_transport"))
    }
    pub(crate) fn open(directory: &Path, trace: bool) -> Result<Self> {
        validate_artifact_directory(directory, "full transport")?;
        if Self::is_data_directory(directory)? {
            let data = VerifiedTransport::open(directory)?;
            let m = data.manifest();
            let transport = FullTransport {
                method: m.transport.method.clone(),
                target_layer: m.transport.target_layer,
                source_layers: m.transport.source_layers.clone(),
                capture_site: "post_block_residual".into(),
                orientation: m.transport.orientation.clone(),
                hidden_size: m.model.hidden_size,
                bias: m.transport.bias.clone(),
                storage_dtype: "f16_le".into(),
            };
            let payload = FullPayload {
                path: m.payload.path.clone(),
                dtype: m.payload.dtype.clone(),
                shape: m.payload.shape.map(|n| n as usize),
                byte_length: m.payload.byte_length,
                blake3: data.payload_blake3().to_owned(),
            };
            return Ok(Self {
                storage: Storage::Data(data),
                transport,
                payload,
                runtime_binding: None,
            });
        }
        let manifest: FullLensManifest = read_json_file(&directory.join(FULL_MANIFEST_NAME))?;
        if trace {
            validate_trace_full_manifest(&manifest)?;
        } else {
            validate_manifest(&manifest)?;
        }
        let (mut file, length) = open_regular_file(&directory.join(&manifest.payload.path))?;
        ensure!(
            length as u64 == manifest.payload.byte_length,
            "full transport length mismatch"
        );
        let mut matrix = vec![0; usize::try_from(transport_matrix_bytes(&manifest)?)?];
        let mut whole = Blake3Hasher::new();
        let mut matrix_hashes = Vec::new();
        for &layer in &manifest.transport.source_layers {
            file.read_exact(&mut matrix)?;
            ensure_finite_f16(&matrix, layer as usize, 0)?;
            whole.update(&matrix);
            matrix_hashes.push(blake3::hash(&matrix));
        }
        ensure!(
            whole.finalize().to_hex().as_str() == manifest.payload.blake3,
            "full transport BLAKE3 mismatch"
        );
        Ok(Self {
            transport: manifest.transport.clone(),
            payload: manifest.payload.clone(),
            storage: Storage::Legacy {
                manifest,
                file,
                matrix_hashes,
            },
            runtime_binding: None,
        })
    }
    pub(super) fn is_data(&self) -> bool {
        matches!(self.storage, Storage::Data(_))
    }
    pub(crate) fn contains_layer(&self, layer: u32) -> bool {
        self.transport.source_layers.contains(&layer)
    }
    pub(super) fn matrix_bytes(&self) -> Result<u64> {
        u64::from(self.transport.hidden_size)
            .checked_pow(2)
            .and_then(|n| n.checked_mul(2))
            .context("matrix byte count overflow")
    }
    pub(super) fn read_matrix(&mut self, layer: u32) -> Result<Vec<u8>> {
        let bytes = self.matrix_bytes()?;
        match &mut self.storage {
            Storage::Data(data) => data.read_matrix(layer),
            Storage::Legacy {
                manifest,
                file,
                matrix_hashes,
            } => {
                let slot = manifest
                    .transport
                    .source_layers
                    .iter()
                    .position(|&l| l == layer)
                    .context("source layer absent")?;
                ensure!(
                    file.metadata()?.len() == manifest.payload.byte_length,
                    "payload length changed"
                );
                file.seek(SeekFrom::Start(slot as u64 * bytes))?;
                let mut matrix = vec![0; usize::try_from(bytes)?];
                file.read_exact(&mut matrix)?;
                ensure!(
                    blake3::hash(&matrix) == matrix_hashes[slot],
                    "transport matrix changed after verification"
                );
                Ok(matrix)
            }
        }
    }
    pub(crate) fn requires_exact_binding(&self) -> bool {
        matches!(&self.storage, Storage::Data(d) if d.manifest().model.exact_binding.is_some())
    }
    pub(crate) fn bind_cpu(
        mut self,
        deployment: Option<&CpuFullDeployment>,
        allow: bool,
    ) -> Result<BoundFullAccess> {
        let mut deployment_identity = None;
        if let Storage::Data(data) = &self.storage {
            let d = deployment.context("generic transport requires CPU deployment preflight")?;
            let tokenizer = format!("{:016x}", d.identity.1);
            let status = data.bind(
                &d.architecture,
                d.arch.n_layer,
                d.arch.hidden_size,
                d.arch.vocab_size,
                d.content.as_deref().unwrap_or(""),
                &tokenizer,
                true,
                allow,
            )?;
            self.runtime_binding = Some(
                serde_json::json!({"status":status, "architecture":d.architecture,
                "n_layers":d.arch.n_layer,"hidden_size":d.arch.hidden_size,"vocab_size":d.arch.vocab_size,
                "gguf_content_blake3":d.content,"tokenizer_metadata_id":tokenizer,
                "qualification":"producer_claims_only", "binding_phase":"cpu_before_metal",
                "content_identity_provenance": if d.content.is_some() { "retained_bytes_hashed" } else { "not_verified" },
                "content_bytes_hashed": d.content_bytes_hashed}),
            );
            deployment_identity = Some(d.identity);
        }
        Ok(BoundFullAccess {
            access: self,
            deployment_identity,
        })
    }
    pub(crate) fn bind_opened(
        self,
        gguf: &GgufFile,
        mode: FullExecutionMode,
        cache: Option<&Path>,
        allow: bool,
    ) -> Result<BoundFullAccess> {
        let model = qwen_llm::loader::Model::from_gguf(gguf)?;
        match &self.storage {
            Storage::Data(data) => {
                let expected = &data.manifest().model;
                ensure!(
                    Some(expected.architecture.as_str()) == gguf.get_str("general.architecture")
                        && expected.n_layers == model.arch.n_layer
                        && expected.hidden_size == model.arch.hidden_size
                        && expected.vocab_size == model.arch.vocab_size,
                    "deployment geometry mismatch"
                );
            }
            Storage::Legacy { manifest, .. } => {
                validate_deployed_geometry(manifest, gguf, model.arch)?
            }
        }
        let descriptor =
            CpuFullDeployment::from_gguf(gguf, mode, self.requires_exact_binding(), cache)?;
        self.bind_cpu(Some(&descriptor), allow)
    }
    pub(crate) fn acknowledge_transfer(&self, allow: bool) -> Result<()> {
        let bound =
            matches!(&self.storage, Storage::Data(d) if d.manifest().model.exact_binding.is_some());
        ensure!(
            allow || bound,
            "unbound full transport requires allow_unvalidated_transfer=true"
        );
        Ok(())
    }
    pub(super) fn producer_contract(&self) -> Option<serde_json::Value> {
        match &self.storage {
            Storage::Data(d) => Some(d.original_manifest().clone()),
            _ => None,
        }
    }
    pub(super) fn manifest_digest(&self) -> Result<String> {
        match &self.storage {
            Storage::Data(d) => {
                let mut canonical = d.original_manifest().clone();
                canonical.sort_all_objects();
                digest_json(&canonical)
            }
            Storage::Legacy { manifest, .. } => digest_json(manifest),
        }
    }
    pub(super) fn validation_status(&self) -> String {
        match &self.storage {
            Storage::Legacy { manifest, .. } => manifest.transfer.validation_status.clone(),
            Storage::Data(_) => self
                .runtime_binding
                .as_ref()
                .and_then(|v| v["status"].as_str())
                .unwrap_or("not_bound")
                .into(),
        }
    }
    pub(super) fn authenticate_trace_model(&self, model: &mut TraceFullModel) {
        if let Some(content) = self
            .runtime_binding
            .as_ref()
            .and_then(|v| v["gguf_content_blake3"].as_str())
        {
            model.content_blake3 = Some(content.into());
            model.content_authenticated = true;
        }
    }
    pub(super) fn bound_content_blake3(&self) -> Option<&str> {
        self.runtime_binding
            .as_ref()
            .and_then(|v| v["gguf_content_blake3"].as_str())
    }
    pub(super) fn readout_artifact(&self, directory: &Path) -> Result<serde_json::Value> {
        match &self.storage {
            Storage::Legacy { manifest: m, .. } => Ok(serde_json::to_value(FullReadoutArtifact {
                manifest: directory.join(FULL_MANIFEST_NAME),
                manifest_canonical_json_blake3: self.manifest_digest()?,
                payload_blake3: m.payload.blake3.clone(),
                method: m.transport.method.clone(),
                target_layer: m.transport.target_layer,
                orientation: m.transport.orientation.clone(),
                source_repository: m.source.repository.clone(),
                source_revision: m.source.revision.clone(),
                fitted_checkpoint: m.model.fitted_checkpoint.clone(),
                fitted_checkpoint_revision: m.model.fitted_checkpoint_revision.clone(),
                fit_n_prompts: m.fit.n_prompts,
                fit_max_sequence_length: m.fit.max_sequence_length,
                fit_skip_first: m.fit.skip_first,
            })?),
            Storage::Data(_) => Ok(serde_json::json!({"manifest":directory.join("lens.json"),
                "manifest_canonical_json_blake3":self.manifest_digest()?, "payload_blake3":self.payload.blake3,
                "method":self.transport.method, "target_layer":self.transport.target_layer,
                "orientation":self.transport.orientation, "producer_contract":self.producer_contract(),
                "runtime_binding":self.runtime_binding})),
        }
    }
    pub(super) fn trace_summary(&self) -> TraceFullLens {
        match &self.storage {
            Storage::Legacy { manifest, .. } => trace_full_lens_summary(manifest),
            Storage::Data(_) => TraceFullLens {
                kind: "linear_transport",
                method: self.transport.method.clone(),
                target_layer: self.transport.target_layer,
                source_site: self.transport.capture_site.clone(),
                source_repository: String::new(),
                source_revision: String::new(),
                source_filename: String::new(),
                payload_blake3: self.payload.blake3.clone(),
                scoring: "deployed_output_rmsnorm_and_lm_head_full_vocabulary_logits_no_softmax",
                producer_contract: self.producer_contract(),
                runtime_binding: self.runtime_binding.clone(),
            },
        }
    }
}
