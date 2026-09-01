use serde_json::Value;

/// Origin of the model-facing system message before wire protocols converge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SystemSource {
    Instructions,
    System,
    Developer,
}

impl SystemSource {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Instructions => "instructions",
            Self::System => "system",
            Self::Developer => "developer",
        }
    }
}

/// One validated conversation turn, independent of its wire representation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Turn {
    User(String),
    Assistant {
        reasoning: Option<String>,
        visible: String,
        /// Tool calls emitted in this assistant turn, in wire order.
        calls: Vec<ToolCall>,
    },
    /// Consecutive results coalesced into one model-facing tool-result turn.
    ToolResults(Vec<ToolResult>),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolCall {
    pub(crate) call_id: String,
    pub(crate) name: String,
    /// Raw JSON string exactly as the provider replayed it.
    pub(crate) arguments: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolResult {
    pub(crate) call_id: String,
    pub(crate) name: String,
    pub(crate) output: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolDefinition {
    pub(crate) name: String,
    pub(crate) description: Option<String>,
    pub(crate) parameters: Value,
    pub(crate) strict: Option<bool>,
}

/// Validated model-facing transcript and tool declarations.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ModelRequest {
    pub(crate) system: Option<String>,
    pub(crate) system_source: Option<SystemSource>,
    pub(crate) turns: Vec<Turn>,
    pub(crate) tools: Vec<ToolDefinition>,
}
