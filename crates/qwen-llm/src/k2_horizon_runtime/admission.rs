//! CPU artifact preparation shared by product lanes. This is not request/device
//! admission, content authentication, or numerical qualification.
use super::{K2HorizonConfig, K2RuntimePlan, residency};
use crate::{
    gguf::GgufFile, k2_horizon::K2HorizonModel, metal::host_page_size_bytes, tensor::GgmlType,
    tokenizer::NativeTokenizer,
};

#[derive(Debug, thiserror::Error)]
#[error("K2 artifact {code}: {message}")]
pub struct K2AdmissionError {
    code: &'static str,
    message: String,
}

impl K2AdmissionError {
    pub fn code(&self) -> &'static str {
        self.code
    }
}

fn failure(code: &'static str, error: impl std::fmt::Display) -> K2AdmissionError {
    K2AdmissionError {
        code,
        message: error.to_string(),
    }
}

pub fn validate_generation_stops(stops: &[i32]) -> Result<(), K2AdmissionError> {
    if stops != [1] {
        return Err(failure(
            "k2_generation_stops",
            "raw generation requires EOS 1 only; extra stop metadata is unsupported",
        ));
    }
    Ok(())
}

pub struct K2PreparedArtifact<'a> {
    layout: K2ArtifactLayout<'a>,
    tokenizer: NativeTokenizer,
}

/// Layout-only stage, separate from tokenizer construction for honest setup timing.
pub struct K2ArtifactLayout<'a> {
    source: &'a GgufFile,
    plan: K2RuntimePlan<'a>,
    output_dtype: GgmlType,
}

impl<'a> K2ArtifactLayout<'a> {
    /// Bind geometry, complete tensor/storage ranges, native embedding support,
    /// retained-storage layout without creating a Metal context. Tokenizer
    /// construction is the separate `prepare_tokenizer` stage.
    /// The one-position plan tests artifact layout only; execution must still
    /// plan the actual capacity against the actual device and memory signals.
    pub fn inspect(source: &'a GgufFile) -> Result<Self, K2AdmissionError> {
        K2HorizonConfig::from_gguf(source).map_err(|e| failure("k2_configuration", e))?;
        let bound =
            K2HorizonModel::from_gguf(source).map_err(|e| failure("k2_tensor_inventory", e))?;
        residency::validate_embedding(bound.token_embedding.dtype)
            .map_err(|e| failure("k2_embedding_storage", e))?;
        let output_dtype = bound.output.dtype;
        let page = host_page_size_bytes().map_err(|e| failure("k2_host_layout", e))?;
        let plan = K2RuntimePlan::inspect(source, 1, page, usize::MAX)
            .map_err(|e| failure("k2_retained_storage", e))?;
        Ok(Self {
            source,
            plan,
            output_dtype,
        })
    }

    pub fn prepare_tokenizer(self) -> Result<K2PreparedArtifact<'a>, K2AdmissionError> {
        let tokenizer =
            NativeTokenizer::from_gguf(self.source).map_err(|e| failure("k2_tokenizer", e))?;
        Ok(K2PreparedArtifact {
            layout: self,
            tokenizer,
        })
    }
}

impl<'a> K2PreparedArtifact<'a> {
    pub fn inspect(source: &'a GgufFile) -> Result<Self, K2AdmissionError> {
        K2ArtifactLayout::inspect(source)?.prepare_tokenizer()
    }

    pub fn config(&self) -> &K2HorizonConfig {
        self.layout.plan.config()
    }
    pub fn tokenizer(&self) -> &NativeTokenizer {
        &self.tokenizer
    }
    pub fn output_dtype(&self) -> GgmlType {
        self.layout.output_dtype
    }
    pub fn into_tokenizer(self) -> NativeTokenizer {
        self.tokenizer
    }

    /// Generation policy is deliberately separate: forward-only lens execution
    /// does not sample or stop, so it must not reject unrelated EOS metadata.
    pub fn generation_stops(&self) -> Result<Vec<i32>, K2AdmissionError> {
        let stops = self
            .layout
            .source
            .stop_token_ids()
            .map_err(|e| failure("k2_generation_stops", e))?;
        validate_generation_stops(&stops)?;
        Ok(stops)
    }
}

#[cfg(test)]
mod tests;
