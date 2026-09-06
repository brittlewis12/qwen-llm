//! Workspace-lens error type.

use super::*;

#[derive(Debug, thiserror::Error)]
pub enum WorkspaceLensError {
    #[error("runtime: {0}")]
    Runtime(#[from] RuntimeError),
    #[error("forward: {0}")]
    Forward(#[from] MfError),
    #[error("metal: {0}")]
    Metal(#[from] MetalError),
    #[error("packed prefill: {0}")]
    PackedPrefill(#[from] DFlashError),
    #[error("workspace-lens operations currently require a dense model, got {0:?}")]
    UnsupportedArchitecture(ArchKind),
    #[error("layer {layer} is out of range for {n_layers} layers")]
    InvalidLayer { layer: u32, n_layers: u32 },
    #[error("linear role {role:?} is not present on layer {layer}")]
    InvalidLinearRole { layer: u32, role: LinearRole },
    #[error("linear {id:?} has shape {shape:?}, expected a two-dimensional row bank")]
    InvalidLinearShape {
        id: WorkspaceLensLinear,
        shape: Vec<u64>,
    },
    #[error(
        "linear {id:?} uses {dtype:?}; the native activation VJP supports Q8_0, BF16, F16, and F32"
    )]
    UnsupportedLinearDtype {
        id: WorkspaceLensLinear,
        dtype: GgmlType,
    },
    #[error("layer {layer} {role:?} has shape {got:?}, expected {expected:?}")]
    InvalidDenseFfnShape {
        layer: u32,
        role: LinearRole,
        got: [usize; 2],
        expected: [usize; 2],
    },
    #[error("layer {layer} post-attention norm must be F32 [{expected}], got {dtype:?} {shape:?}")]
    InvalidDenseFfnNorm {
        layer: u32,
        dtype: GgmlType,
        shape: Vec<u64>,
        expected: usize,
    },
    #[error("layer {layer} is not a GDN layer")]
    NotGdnLayer { layer: u32 },
    #[error("GDN prompt capture requires layer > 0 so its real input residual can be observed")]
    GdnCaptureRequiresPreviousLayer,
    #[error("GDN prompt capture requires at least one token")]
    EmptyGdnPrompt,
    #[error("GDN prompt capture length {got} exceeds the bounded isolated-sequence limit {max}")]
    GdnPromptTooLong { got: usize, max: usize },
    #[error(
        "layer {layer} GDN geometry must use head_dim=128 with nonzero V/K heads and V divisible by K; got n_v={n_v} n_k={n_k} head_dim={head_dim}"
    )]
    UnsupportedGdnGeometry {
        layer: u32,
        n_v: usize,
        n_k: usize,
        head_dim: usize,
    },
    #[error("layer {layer} {role:?} has shape {got:?}, expected {expected:?}")]
    InvalidGdnLinearShape {
        layer: u32,
        role: LinearRole,
        got: [usize; 2],
        expected: [usize; 2],
    },
    #[error(
        "layer {layer} GDN tensor {name} must be F32 with {expected_elements} elements, got {dtype:?} {shape:?}"
    )]
    InvalidGdnTensor {
        layer: u32,
        name: &'static str,
        dtype: GgmlType,
        shape: Vec<u64>,
        expected_elements: usize,
    },
    #[error("GDN capture belongs to a different model identity")]
    GdnCaptureModelMismatch,
    #[error("GDN capture belongs to a different loaded-model owner")]
    GdnCaptureOwnerMismatch,
    #[error("layer {layer} is not a full-attention layer")]
    NotAttentionLayer { layer: u32 },
    #[error("attention prompt capture requires layer > 0")]
    AttnCaptureRequiresPreviousLayer,
    #[error("attention prompt capture requires a fresh sequence at position zero, got {0}")]
    AttnCaptureRequiresFreshSequence(usize),
    #[error("attention prompt capture requires at least one token")]
    EmptyAttnPrompt,
    #[error("attention prompt capture length {got} exceeds the bounded F32-oracle limit {max}")]
    AttnPromptTooLong { got: usize, max: usize },
    #[error("attention capture belongs to a different model identity")]
    AttnCaptureModelMismatch,
    #[error("attention capture belongs to a different loaded-model owner")]
    AttnCaptureOwnerMismatch,
    #[error("layer {layer} {role:?} has shape {got:?}, expected {expected:?}")]
    InvalidAttnLinearShape {
        layer: u32,
        role: LinearRole,
        got: [usize; 2],
        expected: [usize; 2],
    },
    #[error(
        "layer {layer} attention tensor {name} must be F32 with {expected_elements} elements, got {dtype:?} {shape:?}"
    )]
    InvalidAttnTensor {
        layer: u32,
        name: &'static str,
        dtype: GgmlType,
        shape: Vec<u64>,
        expected_elements: usize,
    },
    #[error("workspace prompt capture requires a fresh sequence at position zero, got {0}")]
    WorkspaceCaptureRequiresFreshSequence(usize),
    #[error("workspace prompt capture requires at least one token")]
    EmptyWorkspacePrompt,
    #[error("workspace prompt capture length {got} exceeds the bounded limit {max}")]
    WorkspacePromptTooLong { got: usize, max: usize },
    #[error("workspace capture belongs to a different model identity")]
    WorkspaceCaptureModelMismatch,
    #[error("workspace capture belongs to a different loaded-model owner")]
    WorkspaceCaptureOwnerMismatch,
    #[error("workspace VJP requires at least one source layer")]
    EmptyWorkspaceSourceLayers,
    #[error(
        "workspace source layer {source_layer} must be strictly below target layer {target_layer}"
    )]
    WorkspaceSourceNotBeforeTarget {
        source_layer: u32,
        target_layer: u32,
    },
    #[error("workspace row fit requires at least one output coordinate")]
    EmptyWorkspaceOutputRows,
    #[error("workspace output rows must be strictly increasing and unique")]
    WorkspaceOutputRowsNotStrict,
    #[error("workspace output coordinate {row} is out of range for hidden size {hidden_size}")]
    WorkspaceOutputRowOutOfRange { row: u32, hidden_size: usize },
    #[error("workspace readout fit requires at least one target covector")]
    EmptyWorkspaceTargetCovectors,
    #[error(
        "workspace target covector bank length {got} is not a multiple of hidden size {hidden_size}"
    )]
    WorkspaceTargetCovectorSize { got: usize, hidden_size: usize },
    #[error("workspace target covector at flat index {index} is not finite")]
    NonFiniteWorkspaceTargetCovector { index: usize },
    #[error("workspace VJP trajectory contains a non-finite value at flat index {index}")]
    NonFiniteWorkspaceVjpTrajectory { index: usize },
    #[error("workspace replay diagnostic for layer {layer} is not finite")]
    NonFiniteWorkspaceReplayDiagnostic { layer: u32 },
    #[error("workspace reduction produced a non-finite {stage} at flat index {index}")]
    NonFiniteWorkspaceReduction { stage: &'static str, index: usize },
    #[error(
        "workspace prompt length {n_tokens} leaves no valid positions with skip_first={skip_first}; require length >= skip_first + 2"
    )]
    WorkspaceNoValidPositions { n_tokens: usize, skip_first: usize },
    #[error("workspace replay diagnostic schedule changed across fitted rows")]
    WorkspaceDiagnosticScheduleMismatch,
    #[error("workspace query batch {got} exceeds the bounded limit {max}")]
    WorkspaceQueryBatchTooLarge { got: usize, max: usize },
    #[error("selected-token readout extraction requires at least one token ID")]
    EmptyTokenReadoutSelection,
    #[error("selected-token readout count {got} exceeds model vocabulary size {vocab_size}")]
    TokenReadoutCountExceedsVocabulary { got: usize, vocab_size: u32 },
    #[error(
        "selected-token readout ID {token_id} is out of range for vocabulary size {vocab_size}"
    )]
    TokenReadoutIdOutOfRange { token_id: u32, vocab_size: u32 },
    #[error("selected-token readout ID {token_id} occurs more than once")]
    DuplicateTokenReadoutId { token_id: u32 },
    #[error("LM head has native shape {got:?}, expected {expected:?}")]
    InvalidTokenReadoutLmHeadShape { got: Vec<u64>, expected: [usize; 2] },
    #[error("LM head dtype {dtype:?} is unsupported for selected-token readout extraction")]
    UnsupportedTokenReadoutLmHeadDtype { dtype: GgmlType },
    #[error("output norm must be F32 [{expected}], got {dtype:?} {shape:?}")]
    InvalidTokenReadoutOutputNorm {
        dtype: GgmlType,
        shape: Vec<u64>,
        expected: usize,
    },
    #[error("workspace-lens readout {name} contains a non-finite value at flat index {index}")]
    NonFiniteTokenReadoutData { name: &'static str, index: usize },
    #[error("full-vocabulary lens readout requires at least one prompt token")]
    EmptyFullReadoutPrompt,
    #[error("packed full-vocabulary readout position count {got} exceeds the limit {max}")]
    PackedFullReadoutTooLong { got: usize, max: usize },
    #[error("F16 transport has {got} bytes, expected exactly {expected}")]
    InvalidFullReadoutTransportSize { got: usize, expected: usize },
    #[error("prepared F16 transport belongs to a different loaded model")]
    PreparedF16TransportModelMismatch,
    #[error("packed capture layer {layer} occurs more than once")]
    DuplicatePackedCaptureLayer { layer: u32 },
    #[error("packed capture belongs to a different loaded model")]
    PackedCaptureModelMismatch,
    #[error("source layer {source_layer} is not present in the packed capture")]
    PackedCaptureLayerNotFound { source_layer: u32 },
    #[error(
        "packed transported-vector source position {source_position} is outside capture range {start_position}..{end_position}"
    )]
    PackedTransportedVectorPositionOutOfRange {
        source_position: usize,
        start_position: usize,
        end_position: usize,
    },
    #[error("packed transported-vector source position {source_position} occurs more than once")]
    DuplicatePackedTransportedVectorPosition { source_position: usize },
    #[error("full-vocabulary prompt capture requires a fresh sequence at position zero, got {0}")]
    FullReadoutRequiresFreshSequence(usize),
    #[error("full-vocabulary lens top-k {got} is outside the supported range 1..={max}")]
    InvalidFullReadoutTopK { got: usize, max: usize },
    #[error("full-vocabulary GPU workspace requires a nonzero row capacity")]
    EmptyFullReadoutWorkspace,
    #[error("full-vocabulary GPU workspace has {capacity} rows but this call requires {required}")]
    FullReadoutWorkspaceTooSmall { capacity: usize, required: usize },
    #[error("full-vocabulary GPU workspace has no bound F16 transport")]
    FullReadoutTransportNotBound,
    #[error(
        "full-vocabulary GPU workspace memory admission denied: reason={reason:?} requested={requested_bytes} working_set_headroom={working_set_headroom_bytes:?} process_remaining={process_remaining_bytes:?}"
    )]
    FullReadoutMemoryAdmissionDenied {
        reason: MetalMemoryAdmissionReason,
        requested_bytes: u64,
        working_set_headroom_bytes: Option<u64>,
        process_remaining_bytes: Option<u64>,
    },
    #[error(
        "full-vocabulary lens top-k returned token ID {token_id} outside vocabulary {vocab_size}"
    )]
    InvalidFullReadoutToken { token_id: i32, vocab_size: u32 },
    #[error(
        "workspace-lens allocation for {name} requires {requested_bytes} bytes, exceeding the {max_bytes}-byte budget"
    )]
    WorkspaceLensResultByteBudgetExceeded {
        name: &'static str,
        requested_bytes: usize,
        max_bytes: usize,
    },
    #[error("host allocation for {name} failed for {elements} elements")]
    WorkspaceLensHostAllocationFailed { name: &'static str, elements: usize },
    #[error("{name} length {got} does not match expected length {expected}")]
    ActivationSize {
        name: &'static str,
        got: usize,
        expected: usize,
    },
    #[error("n_query must be nonzero")]
    EmptyQueryBatch,
    #[error("cotangent length {got} does not match n_query={n_query} x n_out={n_out} ({expected})")]
    CotangentSize {
        got: usize,
        expected: usize,
        n_query: usize,
        n_out: usize,
    },
    #[error("workspace-lens tensor size overflow")]
    SizeOverflow,
    #[error("sequence position {0} exceeds the ordinary-Qwen u32 position contract")]
    PositionOverflow(usize),
    #[error("Metal did not provide a command buffer")]
    MissingCommandBuffer,
    #[error("Metal command buffer failed: status={status} error={error}")]
    CommandBuffer { status: String, error: String },
}
