//! Stable ownership boundary for loaded-model execution.
//!
//! The Metal modules expose the low-level execution pieces directly:
//! [`MetalContext`] owns device resources, [`MetalModel`] owns resident
//! weights, and [`MetalSession`] owns mutable per-sequence state. This module
//! gives callers semantic names for those lifetimes without changing the
//! underlying execution path.

use crate::gguf::{GgufError, GgufFile};
use crate::loader::{LoadError, Model};
use crate::metal::{MetalContext, MetalError};
use crate::metal_forward::{MetalForward, MetalModel, MetalSession, MfError};
use crate::model::Arch;
use crate::tokenizer::{TokError, Tokenizer};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("metal runtime: {0}")]
    Metal(#[from] MetalError),
    #[error("gguf load: {0}")]
    Gguf(#[from] GgufError),
    #[error("model bind: {0}")]
    Load(#[from] LoadError),
    #[error("metal model load: {0}")]
    MetalModel(#[from] MfError),
    #[error("tokenizer load: {0}")]
    Tokenizer(#[from] TokError),
    #[error("sequence max_context_tokens must be >= 1")]
    EmptySequenceCapacity,
    #[error(
        "sequence position mismatch: caller position {caller_position}, tracked position {tracked_position}"
    )]
    SequencePositionMismatch {
        caller_position: usize,
        tracked_position: usize,
    },
    #[error(
        "sequence capacity exceeded: position {position} + {n_tokens} tokens > max_context_tokens {max_context_tokens}"
    )]
    SequenceCapacityExceeded {
        position: usize,
        n_tokens: usize,
        max_context_tokens: usize,
    },
}

struct RuntimeInner {
    ctx: MetalContext,
}

/// Backend/device execution environment.
///
/// `Runtime` owns the reusable Metal device resources. It deliberately does
/// not own models or sequence state, so benchmarks and future applications can
/// make cold-load, warm-model, and warm-session boundaries explicit.
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

impl Runtime {
    /// Create a runtime backed by the system default Metal device.
    pub fn metal() -> Result<Self, RuntimeError> {
        Ok(Self {
            inner: Arc::new(RuntimeInner {
                ctx: MetalContext::new()?,
            }),
        })
    }

    /// Alias for callers that prefer constructor-style naming.
    pub fn new_metal() -> Result<Self, RuntimeError> {
        Self::metal()
    }

    /// Access the underlying Metal context for low-level paths.
    pub fn context(&self) -> &MetalContext {
        &self.inner.ctx
    }

    /// Human-readable device description.
    pub fn describe(&self) -> String {
        self.context().describe()
    }

    /// Open a GGUF, bind its model metadata, and load resident Metal weights.
    pub fn load_model(&self, path: impl AsRef<Path>) -> Result<LoadedModel, RuntimeError> {
        let path = path.as_ref();
        let gguf = GgufFile::open(path)?;
        let bound = Model::from_gguf(&gguf)?;
        let metal_model = MetalModel::load(self.context(), &gguf, &bound)?;
        Ok(LoadedModel {
            runtime: self.clone(),
            path: path.to_path_buf(),
            gguf,
            metal_model,
        })
    }
}

/// One model loaded into a [`Runtime`].
///
/// This is the expensive model-load boundary: the GGUF is mmap'd and the Metal
/// weights are resident. Per-request mutable state is created separately as a
/// [`Sequence`].
pub struct LoadedModel {
    runtime: Runtime,
    path: PathBuf,
    gguf: GgufFile,
    metal_model: MetalModel,
}

impl LoadedModel {
    /// Path used to load this model.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Underlying GGUF view, retained for metadata and tokenizer access.
    pub fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    /// Resident Metal weights.
    pub fn metal_model(&self) -> &MetalModel {
        &self.metal_model
    }

    /// Runtime backing this loaded model.
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Metal context backing this loaded model.
    pub fn context(&self) -> &MetalContext {
        self.runtime.context()
    }

    /// Architecture metadata copied into the resident Metal model.
    pub fn arch(&self) -> Arch {
        self.metal_model.arch
    }

    /// Create a tokenizer from the same GGUF metadata as this loaded model.
    ///
    /// Tokenizer construction remains explicit for phase 1 so synthetic
    /// benchmark rows do not pay tokenizer setup unless they need it.
    pub fn tokenizer(&self) -> Result<Tokenizer, RuntimeError> {
        Ok(Tokenizer::from_gguf(&self.gguf)?)
    }

    /// Create fresh per-sequence state with the requested context capacity.
    pub fn create_sequence(&self, config: SequenceConfig) -> Result<Sequence, RuntimeError> {
        if config.max_context_tokens == 0 {
            return Err(RuntimeError::EmptySequenceCapacity);
        }
        let state =
            MetalSession::fresh(self.context(), &self.metal_model, config.max_context_tokens)?;
        Ok(Sequence {
            max_context_tokens: config.max_context_tokens,
            position: 0,
            state,
        })
    }

    /// Borrowed execution driver over this loaded model.
    pub fn forward(&self) -> MetalForward<'_> {
        MetalForward::new(self.context(), &self.metal_model)
    }
}

/// Allocation/configuration for one [`Sequence`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceConfig {
    /// Maximum token positions the sequence state can hold.
    pub max_context_tokens: usize,
}

impl SequenceConfig {
    pub fn new(max_context_tokens: usize) -> Self {
        Self { max_context_tokens }
    }

    pub fn try_new(max_context_tokens: usize) -> Result<Self, RuntimeError> {
        if max_context_tokens == 0 {
            return Err(RuntimeError::EmptySequenceCapacity);
        }
        Ok(Self { max_context_tokens })
    }
}

/// User-facing mutable state for one generated sequence.
///
/// Internally this is the current `MetalSession`: KV cache, GDN state, scratch,
/// ids, logits, and related per-sequence buffers.
pub struct Sequence {
    max_context_tokens: usize,
    position: usize,
    state: MetalSession,
}

impl Sequence {
    pub fn max_context_tokens(&self) -> usize {
        self.max_context_tokens
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn remaining_context_tokens(&self) -> usize {
        self.max_context_tokens.saturating_sub(self.position)
    }

    pub fn check_position(&self, caller_position: usize) -> Result<(), RuntimeError> {
        if caller_position != self.position {
            return Err(RuntimeError::SequencePositionMismatch {
                caller_position,
                tracked_position: self.position,
            });
        }
        Ok(())
    }

    pub fn ensure_can_append(&self, n_tokens: usize) -> Result<(), RuntimeError> {
        let end =
            self.position
                .checked_add(n_tokens)
                .ok_or(RuntimeError::SequenceCapacityExceeded {
                    position: self.position,
                    n_tokens,
                    max_context_tokens: self.max_context_tokens,
                })?;
        if end > self.max_context_tokens {
            return Err(RuntimeError::SequenceCapacityExceeded {
                position: self.position,
                n_tokens,
                max_context_tokens: self.max_context_tokens,
            });
        }
        Ok(())
    }

    pub fn advance_by(&mut self, n_tokens: usize) -> Result<(), RuntimeError> {
        self.ensure_can_append(n_tokens)?;
        self.position += n_tokens;
        Ok(())
    }

    /// Low-level bridge for the existing Metal execution functions.
    pub fn metal_session(&self) -> &MetalSession {
        &self.state
    }

    /// Low-level mutable bridge for the existing Metal execution functions.
    pub fn metal_session_mut(&mut self) -> &mut MetalSession {
        &mut self.state
    }
}
