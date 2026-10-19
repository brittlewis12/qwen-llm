//! Conservative memory arithmetic for buffers that retain generated tool
//! output. Kept independent of any model family so serve can use the same
//! measured peak bound for each parser.

/// Bytes held at once, at most, per byte of a buffered tool block while it
/// is parsed and its calls are published: the text itself, the parsed
/// values (container slots, map entries and scalar copies, which dominate:
/// `[[1],[1],…]` costs far more than its text) and the serialized arguments
/// a server keeps and embeds in events and the response body. A retained
/// conservative model, measured with counting allocators (16-byte malloc
/// quanta), never a universal bound: the parse and one publication alone
/// (`tests/glm53_tool_block_peak.rs`) reach about 152 for arrays nested to
/// the decoder's 128-level limit; serve's real handler end to end
/// (`serve/http/tests/tool_block_memory_tests.rs` in qwen-cli) measured, on
/// 2026-10-05, about 149 streaming and 205 non-streaming with one-byte
/// pieces when non-streaming output was collected one allocation per piece
/// (theoretical worst about 225); since 2026-10-08 (contiguous collection)
/// about 146 non-streaming, and Qwen XML and DeepSeek DSML blocks of the
/// same shapes measure at most 147.5.
pub const TOOL_BLOCK_PEAK_FACTOR: usize = 256;

/// [`TOOL_BLOCK_PEAK_FACTOR`] applied to a block of `bytes` (saturating;
/// for measurements and tests). Admission uses [`checked_tool_block_peak_bytes`].
pub const fn tool_block_peak_bytes(bytes: usize) -> usize {
    bytes.saturating_mul(TOOL_BLOCK_PEAK_FACTOR)
}

/// [`TOOL_BLOCK_PEAK_FACTOR`] applied to a block of `bytes`, `None` on
/// overflow or past `isize::MAX` (no allocation can be that large).
pub const fn checked_tool_block_peak_bytes(bytes: usize) -> Option<usize> {
    if bytes > isize::MAX as usize {
        return None;
    }
    bytes.checked_mul(TOOL_BLOCK_PEAK_FACTOR)
}
