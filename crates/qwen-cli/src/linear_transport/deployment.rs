//! CPU deployment binding independent of the CLI's published-import facade.

use super::VerifiedTransport;
use anyhow::{Context, Result, ensure};
use qwen_llm::{gguf::GgufFile, runtime::LoadedModel};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExecutionMode {
    Scalar,
    Packed,
    Projection,
}

pub(crate) struct CpuDeployment {
    architecture: String,
    arch: qwen_llm::model::Arch,
    pub(crate) identity: (u64, u64),
    content: Option<String>,
    content_bytes_hashed: u64,
}

impl CpuDeployment {
    pub(crate) fn from_gguf(gguf: &GgufFile, mode: ExecutionMode, exact: bool) -> Result<Self> {
        let bound = qwen_llm::loader::Model::from_gguf(gguf)
            .context("bind CPU Qwen geometry and tensor inventory")?;
        let arch = bound.arch;
        ensure!(
            mode != ExecutionMode::Packed || arch.kind == qwen_llm::model::ArchKind::Dense,
            "packed capture is not exposed by the workspace-lens facade for ordinary MoE; use scalar read-full"
        );
        qwen_llm::workspace_lens::validate_opened_output_head(gguf, mode != ExecutionMode::Scalar)?;
        let identity = qwen_llm::runtime::opened_gguf_lightweight_identity_parts(gguf)?;
        let (content, content_bytes_hashed) = if exact {
            let verified =
                qwen_llm::checkpoint_identity::verified_checkpoint_content_identity(gguf)?;
            let content = verified
                .content_id
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            (Some(content), verified.bytes_hashed)
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

    pub(crate) fn bind(&self, data: &VerifiedTransport, allow: bool) -> Result<Value> {
        let tokenizer = format!("{:016x}", self.identity.1);
        let status = data.bind(
            &self.architecture,
            self.arch.n_layer,
            self.arch.hidden_size,
            self.arch.vocab_size,
            self.content.as_deref().unwrap_or(""),
            &tokenizer,
            true,
            allow,
        )?;
        Ok(json!({
            "status":status, "architecture":self.architecture,
            "n_layers":self.arch.n_layer,"hidden_size":self.arch.hidden_size,"vocab_size":self.arch.vocab_size,
            "gguf_content_blake3":self.content,"tokenizer_metadata_id":tokenizer,
            "qualification":"producer_claims_only", "binding_phase":"cpu_before_metal",
            "content_identity_provenance":if self.content.is_some() { "retained_bytes_hashed" } else { "not_verified" },
            "content_bytes_hashed":self.content_bytes_hashed,
        }))
    }

    pub(crate) fn validate_loaded_identity(
        expected: Option<(u64, u64)>,
        loaded: &LoadedModel,
    ) -> Result<()> {
        let identity = loaded.workspace_lens_identity();
        ensure!(
            expected == Some((identity.model_locator_id, identity.tokenizer_metadata_id)),
            "loaded deployment differs from CPU-bound retained GGUF"
        );
        loaded.validate_passive_workspace_lens_output()?;
        Ok(())
    }
}
