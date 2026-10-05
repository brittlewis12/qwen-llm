//! CPU artifact preparation shared by GLM-5.3 product lanes (run, info,
//! bench, serve): strict binding, execution coverage, tokenizer and stop set,
//! each with a stable refusal code. Not request or device admission, content
//! authentication, or numerical qualification.

use super::{ExecutionMode, Glm5NextConfig, Glm5NextModel, generation_stops};
use crate::gguf::GgufFile;
use crate::tokenizer::NativeTokenizer;

#[derive(Debug, thiserror::Error)]
#[error("GLM-5.3 artifact {code}: {message}")]
pub struct Glm5NextAdmissionError {
    code: &'static str,
    message: String,
}

impl Glm5NextAdmissionError {
    /// Stable refusal code: `glm5_next_configuration`,
    /// `glm5_next_tensor_inventory`, `glm5_next_execution_coverage`,
    /// `glm5_next_tokenizer` or `glm5_next_generation_stops`.
    pub fn code(&self) -> &'static str {
        self.code
    }
}

fn failure(code: &'static str, error: impl std::fmt::Display) -> Glm5NextAdmissionError {
    Glm5NextAdmissionError {
        code,
        message: error.to_string(),
    }
}

/// Layout-only stage: configuration, every executed tensor bound once and in
/// range, and serial-decode coverage. Separate from tokenizer construction so
/// lanes can time setup honestly.
pub struct Glm5NextArtifactLayout<'a> {
    source: &'a GgufFile,
    model: Glm5NextModel<'a>,
    packed_prefill: bool,
}

impl<'a> Glm5NextArtifactLayout<'a> {
    pub fn inspect(source: &'a GgufFile) -> Result<Self, Glm5NextAdmissionError> {
        let config =
            Glm5NextConfig::from_gguf(source).map_err(|e| failure("glm5_next_configuration", e))?;
        let model = Glm5NextModel::bind(config, &source.tensors)
            .map_err(|e| failure("glm5_next_tensor_inventory", e))?;
        for tensor in &source.tensors {
            source
                .try_slice(tensor)
                .map_err(|e| failure("glm5_next_tensor_inventory", e))?;
        }
        model
            .validate_execution(ExecutionMode::SerialDecode)
            .map_err(|e| failure("glm5_next_execution_coverage", e))?;
        let packed_prefill = model
            .validate_execution(ExecutionMode::PackedPrefill)
            .is_ok();
        Ok(Self {
            source,
            model,
            packed_prefill,
        })
    }

    pub fn config(&self) -> &Glm5NextConfig {
        &self.model.config
    }

    pub fn model(&self) -> &Glm5NextModel<'a> {
        &self.model
    }

    /// Whether every executed weight cell has a packed-prefill path.
    pub fn packed_prefill(&self) -> bool {
        self.packed_prefill
    }

    pub fn prepare_tokenizer(self) -> Result<Glm5NextPreparedArtifact<'a>, Glm5NextAdmissionError> {
        let tokenizer = NativeTokenizer::from_gguf(self.source)
            .map_err(|e| failure("glm5_next_tokenizer", e))?;
        Ok(Glm5NextPreparedArtifact {
            layout: self,
            tokenizer,
        })
    }
}

/// A GLM-5.3 artifact whose layout, coverage and tokenizer are admitted.
pub struct Glm5NextPreparedArtifact<'a> {
    layout: Glm5NextArtifactLayout<'a>,
    tokenizer: NativeTokenizer,
}

impl<'a> Glm5NextPreparedArtifact<'a> {
    pub fn inspect(source: &'a GgufFile) -> Result<Self, Glm5NextAdmissionError> {
        Glm5NextArtifactLayout::inspect(source)?.prepare_tokenizer()
    }

    pub fn config(&self) -> &Glm5NextConfig {
        self.layout.config()
    }

    pub fn model(&self) -> &Glm5NextModel<'a> {
        self.layout.model()
    }

    /// Whether every executed weight cell has a packed-prefill path.
    pub fn packed_prefill(&self) -> bool {
        self.layout.packed_prefill()
    }

    pub fn tokenizer(&self) -> &NativeTokenizer {
        &self.tokenizer
    }

    /// Generation policy is separate: forward-only lanes do not stop.
    pub fn generation_stops(&self) -> Result<Vec<i32>, Glm5NextAdmissionError> {
        generation_stops(self.layout.source, self.config().vocab_size)
            .map_err(|e| failure("glm5_next_generation_stops", e))
    }

    /// Text-chat eligibility of this artifact (template, markers, stops).
    /// Raw input does not depend on it.
    pub fn chat_profile(
        &self,
    ) -> Result<crate::glm5_next_chat::VerifiedChatProfile, crate::glm5_next_chat::ChatError> {
        let stops = self
            .generation_stops()
            .map_err(|e| crate::glm5_next_chat::ChatError::unverified(e.to_string()))?;
        crate::glm5_next_chat::verify_profile(self.layout.source, &self.tokenizer, &stops)
    }
}
