//! Native byte-level BPE tokenizers for supported model families.
//!
//! Qwen 3.5/3.6 uses the Qwen35 pretokenizer and a 248,320-entry vocabulary.
//! DeepSeek V4 uses the JoyAI/DeepSeek-V3 pretokenizer and a 129,280-entry
//! vocabulary. Both consume token, type, and merge arrays directly from GGUF.
//!
//! ## Implementation choice
//!
//! Tokenization is a non-hot-path operation: it runs once per prompt at
//! ingest, and again only when streaming user output. The throughput
//! ceiling we care about is in the kernels and graph executor.
//!
//! Three real options:
//!
//! | option | byte-perfect w/ llama-cli oracle | drops llama-cpp link | LOC |
//! |---|---|---|---|
//! | **native GGUF family paths (default)** | yes - differentially tested | no | in-tree |
//! | **`llama-cpp-sys-2` oracle backend** | yes — shared codepath | no | ~50 |
//! | **`tokenizers` (huggingface) crate** | not guaranteed (BPE tie-break edges) | yes | ~30 |
//!
//! The llama.cpp-backed backend remains available as a Qwen oracle. DeepSeek
//! V4 uses an in-tree JoyAI path because the currently linked llama.cpp
//! revision predates the `deepseek4` model architecture.

use crate::gguf::GgufFile;
use rustc_hash::FxHashMap as HashMap;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ffi::CString;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::{Once, OnceLock};
use unicode_general_category::{GeneralCategory, get_general_category};

const NATIVE_MAX_INPUT_BYTES: usize = 8 * 1024 * 1024;

/// SHA-256 over the raw concatenation of signed token IDs in little-endian
/// order. This is the production prompt-token identity contract.
pub fn token_ids_sha256_i32le(tokens: &[i32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

/// Backend-agnostic tokenizer interface. Implemented by the default native
/// tokenizer and the llama.cpp oracle backend.
pub trait Tokenize {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<i32>, TokError>;
    fn try_decode_piece(&self, token: i32) -> Result<String, TokError>;
    fn try_decode(&self, tokens: &[i32]) -> Result<String, TokError>;
    fn decode(&self, tokens: &[i32]) -> String;
    fn n_vocab(&self) -> u32;
    fn bos(&self) -> Option<i32>;
    fn eos(&self) -> Option<i32>;
}

#[derive(Debug, thiserror::Error)]
pub enum TokError {
    #[error("path is not valid UTF-8: {0:?}")]
    BadPath(std::path::PathBuf),
    #[error("path contains interior NUL: {0:?}")]
    PathContainsNul(std::path::PathBuf),
    #[error("llama_model_load_from_file returned null")]
    LoadFailed,
    #[error("llama_model_get_vocab returned null")]
    NoVocab,
    #[error("llama_vocab_n_tokens returned invalid count {0}")]
    BadVocabSize(i32),
    #[error("gguf tokenizer load failed: {0}")]
    Gguf(#[from] crate::gguf::GgufError),
    #[error("unsupported GGUF tokenizer model={model:?} pre={pre:?}")]
    UnsupportedNativeTokenizer { model: String, pre: String },
    #[error("gguf tokenizer metadata is invalid: {0}")]
    BadMetadata(String),
    #[error("native tokenizer input is too large: {bytes} bytes exceeds cap {max_bytes}")]
    NativeInputTooLong { bytes: usize, max_bytes: usize },
    #[error("input text is too large for llama.cpp tokenizer: {0} bytes")]
    InputBytesTooLong(usize),
    #[error("tokenize failed: input too long ({0} tokens needed)")]
    InputTooLong(i32),
    #[error("tokenize result overflowed llama.cpp int32 limits")]
    TokenizeOverflow,
    #[error("token id {0} is outside tokenizer vocab")]
    InvalidToken(i32),
    #[error("decode piece failed for token {token}: {detail}")]
    DecodePieceFailed { token: i32, detail: String },
    #[error("detokenize failed: {0}")]
    DetokenizeFailed(String),
}

/// One-time `llama_backend_init` + log silencing. llama.cpp / ggml
/// otherwise emit a multi-screen stderr banner on every model load
/// (including vocab-only loads via `Tokenizer::open`).
///
/// `llama_log_set(None, ...)` does NOT silence — passing `None` makes
/// llama.cpp fall back to its default stderr logger. To actually drop
/// messages we install a no-op callback. Set `QWEN_LLM_LLAMA_LOGS=1`
/// to keep verbose logging for debugging.
static BACKEND_INIT: Once = Once::new();

unsafe extern "C" fn void_log(
    _level: llama_cpp_sys_2::ggml_log_level,
    _text: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) {
}

fn ensure_backend() {
    BACKEND_INIT.call_once(|| unsafe {
        if std::env::var_os("QWEN_LLM_LLAMA_LOGS").is_none() {
            llama_cpp_sys_2::llama_log_set(Some(void_log), std::ptr::null_mut());
        }
        llama_cpp_sys_2::llama_backend_init();
    });
}

/// llama.cpp-backed oracle tokenizer for a Qwen3.5/3.6 GGUF model.
///
/// Holds an owned `llama_model` pointer (so vocab metadata stays valid).
/// The pointer is freed in `Drop`.
pub struct LlamaCppTokenizer {
    model: NonNull<llama_cpp_sys_2::llama_model>,
    vocab: NonNull<llama_cpp_sys_2::llama_vocab>,
    n_vocab: u32,
    bos: Option<i32>,
    eos: Option<i32>,
}

// SAFETY: the tokenizer owns the llama model handle until Drop, and llama.cpp's
// tokenization API is explicitly documented as thread-safe in `llama.h`.
unsafe impl Send for LlamaCppTokenizer {}
unsafe impl Sync for LlamaCppTokenizer {}

fn checked_i32_len(n: usize) -> Result<i32, TokError> {
    i32::try_from(n).map_err(|_| TokError::InputBytesTooLong(n))
}

fn validate_native_input_len(n: usize) -> Result<(), TokError> {
    if n > NATIVE_MAX_INPUT_BYTES {
        return Err(TokError::NativeInputTooLong {
            bytes: n,
            max_bytes: NATIVE_MAX_INPUT_BYTES,
        });
    }
    Ok(())
}

fn checked_i32_count(n: usize) -> Result<i32, TokError> {
    i32::try_from(n)
        .map_err(|_| TokError::DetokenizeFailed(format!("token slice length {n} exceeds i32::MAX")))
}

fn needed_count_or_overflow(n: i32) -> Result<usize, TokError> {
    if n == i32::MIN {
        return Err(TokError::TokenizeOverflow);
    }
    usize::try_from(n.unsigned_abs()).map_err(|_| TokError::TokenizeOverflow)
}

impl LlamaCppTokenizer {
    fn checked_token(&self, token: i32) -> Result<i32, TokError> {
        if token < 0 || token >= self.n_vocab as i32 {
            return Err(TokError::InvalidToken(token));
        }
        Ok(token)
    }

    /// Open a GGUF file just for its tokenizer. Cheap (vocab-only load).
    ///
    /// We re-open the file via llama.cpp's loader rather than reusing the
    /// already-open [`GgufFile`] mmap, because llama.cpp's own loader is
    /// the canonical thing that interprets `tokenizer.ggml.*` keys and
    /// builds the BPE state. Cross-checking against the same file keeps
    /// the seam tight.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TokError> {
        ensure_backend();
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| TokError::BadPath(path.as_ref().to_path_buf()))?;
        let cpath = CString::new(path_str)
            .map_err(|_| TokError::PathContainsNul(path.as_ref().to_path_buf()))?;

        // Vocab-only load: skip GPU, skip compute. We don't want llama.cpp
        // to allocate any tensor buffers, just to parse the tokenizer.
        // SAFETY: standard FFI sequence per llama.cpp public API.
        let mut params = unsafe { llama_cpp_sys_2::llama_model_default_params() };
        params.vocab_only = true;
        let model = NonNull::new(unsafe {
            llama_cpp_sys_2::llama_model_load_from_file(cpath.as_ptr(), params)
        })
        .ok_or(TokError::LoadFailed)?;
        let vocab = NonNull::new(unsafe {
            llama_cpp_sys_2::llama_model_get_vocab(model.as_ptr()) as *mut _
        });
        let Some(vocab) = vocab else {
            unsafe { llama_cpp_sys_2::llama_model_free(model.as_ptr()) };
            return Err(TokError::NoVocab);
        };
        let n_vocab = unsafe { llama_cpp_sys_2::llama_vocab_n_tokens(vocab.as_ptr()) };
        if n_vocab < 0 {
            unsafe { llama_cpp_sys_2::llama_model_free(model.as_ptr()) };
            return Err(TokError::BadVocabSize(n_vocab));
        }
        let bos_raw = unsafe { llama_cpp_sys_2::llama_vocab_bos(vocab.as_ptr()) };
        let eos_raw = unsafe { llama_cpp_sys_2::llama_vocab_eos(vocab.as_ptr()) };
        // -1 means "model has no such special token"; surface that as None.
        Ok(Self {
            model,
            vocab,
            n_vocab: n_vocab as u32,
            bos: if bos_raw < 0 { None } else { Some(bos_raw) },
            eos: if eos_raw < 0 { None } else { Some(eos_raw) },
        })
    }

    /// Convenience: open the same path that backs an already-loaded
    /// [`GgufFile`]. This keeps engine + tokenizer pinned to the same
    /// on-disk file by construction.
    pub fn from_gguf(_g: &GgufFile, path: impl AsRef<Path>) -> Result<Self, TokError> {
        Self::open(path)
    }

    pub fn n_vocab(&self) -> u32 {
        self.n_vocab
    }

    pub fn bos(&self) -> Option<i32> {
        self.bos
    }

    pub fn eos(&self) -> Option<i32> {
        self.eos
    }

    /// Encode a UTF-8 string. `add_special` adds BOS/EOS per model
    /// convention; for Qwen3.5 chat use we typically don't (BOS is
    /// already in the chat template).
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<i32>, TokError> {
        let text_len = checked_i32_len(text.len())?;
        // First call with `n_tokens_max=0` returns negative `n_needed`.
        // SAFETY: read-only call, vocab is non-null and held for the
        // lifetime of `self`.
        let needed = unsafe {
            llama_cpp_sys_2::llama_tokenize(
                self.vocab.as_ptr(),
                text.as_ptr() as *const i8,
                text_len,
                std::ptr::null_mut(),
                0,
                add_special,
                /* parse_special = */ true,
            )
        };
        if needed == i32::MIN {
            return Err(TokError::TokenizeOverflow);
        }
        let n = if needed < 0 {
            needed_count_or_overflow(needed)?
        } else {
            usize::try_from(needed).map_err(|_| TokError::TokenizeOverflow)?
        };
        let n_i32 = i32::try_from(n).map_err(|_| TokError::TokenizeOverflow)?;
        let mut buf: Vec<i32> = vec![0; n];
        let written = unsafe {
            llama_cpp_sys_2::llama_tokenize(
                self.vocab.as_ptr(),
                text.as_ptr() as *const i8,
                text_len,
                buf.as_mut_ptr(),
                n_i32,
                add_special,
                true,
            )
        };
        if written == i32::MIN {
            return Err(TokError::TokenizeOverflow);
        }
        if written < 0 {
            return Err(TokError::InputTooLong(-written));
        }
        buf.truncate(written as usize);
        Ok(buf)
    }

    pub fn try_decode_piece(&self, token: i32) -> Result<String, TokError> {
        let token = self.checked_token(token)?;
        let mut cap = 32usize;
        loop {
            let mut buf = vec![0u8; cap];
            let n = unsafe {
                llama_cpp_sys_2::llama_token_to_piece(
                    self.vocab.as_ptr(),
                    token,
                    buf.as_mut_ptr() as *mut i8,
                    i32::try_from(buf.len()).map_err(|_| TokError::DecodePieceFailed {
                        token,
                        detail: format!("buffer length {} exceeds i32::MAX", buf.len()),
                    })?,
                    /* lstrip = */ 0,
                    /* special = */ true,
                )
            };
            if n == 0 {
                return Ok(String::new());
            }
            if n > 0 {
                let n = usize::try_from(n).map_err(|_| TokError::DecodePieceFailed {
                    token,
                    detail: "returned byte count did not fit usize".into(),
                })?;
                buf.truncate(n);
                return Ok(String::from_utf8_lossy(&buf).into_owned());
            }
            if n == i32::MIN {
                return Err(TokError::DecodePieceFailed {
                    token,
                    detail: "required byte count overflowed i32".into(),
                });
            }
            cap = needed_count_or_overflow(n)?;
        }
    }

    pub fn try_decode(&self, tokens: &[i32]) -> Result<String, TokError> {
        for &token in tokens {
            self.checked_token(token)?;
        }
        let n_tokens = checked_i32_count(tokens.len())?;
        let mut cap = tokens.len().saturating_mul(8).max(32);
        loop {
            let mut buf = vec![0u8; cap];
            let n = unsafe {
                llama_cpp_sys_2::llama_detokenize(
                    self.vocab.as_ptr(),
                    tokens.as_ptr(),
                    n_tokens,
                    buf.as_mut_ptr() as *mut i8,
                    i32::try_from(buf.len()).map_err(|_| {
                        TokError::DetokenizeFailed(format!(
                            "buffer length {} exceeds i32::MAX",
                            buf.len()
                        ))
                    })?,
                    /* remove_special = */ false,
                    /* unparse_special = */ true,
                )
            };
            if n >= 0 {
                let n = usize::try_from(n).map_err(|_| {
                    TokError::DetokenizeFailed("returned byte count did not fit usize".into())
                })?;
                buf.truncate(n);
                return Ok(String::from_utf8_lossy(&buf).into_owned());
            }
            if n == i32::MIN {
                return Err(TokError::DetokenizeFailed(
                    "required output byte count overflowed i32".into(),
                ));
            }
            cap = needed_count_or_overflow(n)?;
        }
    }

    /// Decode one token to its UTF-8 piece. Special tokens are rendered
    /// as their literal `<|...|>` form.
    pub fn decode_piece(&self, token: i32) -> String {
        self.try_decode_piece(token).unwrap_or_default()
    }

    /// Decode a sequence of tokens.
    pub fn decode(&self, tokens: &[i32]) -> String {
        self.try_decode(tokens).unwrap_or_else(|_| {
            let mut out = String::new();
            for &t in tokens {
                out.push_str(&self.decode_piece(t));
            }
            out
        })
    }
}

impl Tokenize for LlamaCppTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<i32>, TokError> {
        LlamaCppTokenizer::encode(self, text, add_special)
    }
    fn try_decode_piece(&self, token: i32) -> Result<String, TokError> {
        LlamaCppTokenizer::try_decode_piece(self, token)
    }
    fn try_decode(&self, tokens: &[i32]) -> Result<String, TokError> {
        LlamaCppTokenizer::try_decode(self, tokens)
    }
    fn decode(&self, tokens: &[i32]) -> String {
        LlamaCppTokenizer::decode(self, tokens)
    }
    fn n_vocab(&self) -> u32 {
        LlamaCppTokenizer::n_vocab(self)
    }
    fn bos(&self) -> Option<i32> {
        LlamaCppTokenizer::bos(self)
    }
    fn eos(&self) -> Option<i32> {
        LlamaCppTokenizer::eos(self)
    }
}

impl Drop for LlamaCppTokenizer {
    fn drop(&mut self) {
        // SAFETY: paired with the load above. After this call the vocab
        // pointer is dangling; no method on `self` is reachable post-drop.
        unsafe { llama_cpp_sys_2::llama_model_free(self.model.as_ptr()) };
        // Note: not calling `llama_backend_free` here — it's a
        // process-global teardown and we may have multiple Tokenizers
        // outstanding. Leaving it leaked-on-exit is the conventional
        // pattern in the llama.cpp ecosystem.
    }
}

pub type Tokenizer = NativeTokenizer;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PretokenizerKind {
    Qwen35,
    JoyAi,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenAttr {
    Undefined,
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenAttr {
    fn from_gguf(n: i64) -> Result<Self, TokError> {
        match n {
            0 => Ok(Self::Undefined),
            1 => Ok(Self::Normal),
            2 => Ok(Self::Unknown),
            3 => Ok(Self::Control),
            4 => Ok(Self::UserDefined),
            5 => Ok(Self::Unused),
            6 => Ok(Self::Byte),
            other => Err(TokError::BadMetadata(format!(
                "unsupported tokenizer.ggml.token_type value {other}"
            ))),
        }
    }

    fn is_partition_special(self) -> bool {
        matches!(self, Self::Control | Self::UserDefined | Self::Unknown)
    }

    fn is_decode_literal(self) -> bool {
        matches!(self, Self::Control | Self::UserDefined | Self::Unknown)
    }
}

#[derive(Clone, Debug)]
struct NativeToken {
    text: String,
    attr: TokenAttr,
}

#[derive(Clone, Debug)]
struct SpecialToken {
    text: String,
    id: i32,
}

/// Pure-Rust byte-level BPE tokenizer for supported GGUF model families.
///
/// This is intentionally not universal. Architecture and pretokenizer
/// combinations are accepted through a closed dispatch so malformed Qwen
/// metadata cannot silently select another model family's rules.
pub struct NativeTokenizer {
    id_to_token: Vec<NativeToken>,
    pair_merges: HashMap<u64, MergeInfo>,
    byte_token_ids: [i32; 256],
    special_matcher: SpecialMatcher,
    decoded_piece_bytes: Vec<OnceLock<Box<[u8]>>>,
    bos: Option<i32>,
    eos: Option<i32>,
    add_bos: bool,
    add_eos: bool,
    pretokenizer: PretokenizerKind,
}

impl NativeTokenizer {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TokError> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    pub fn from_gguf(g: &GgufFile) -> Result<Self, TokError> {
        let model = required_str(g, "tokenizer.ggml.model")?;
        let pre = required_str(g, "tokenizer.ggml.pre")?;
        let architecture = g.architecture();
        let pretokenizer = match (architecture.as_deref(), model, pre) {
            (Some("qwen35" | "qwen35moe"), "gpt2", "qwen35") => PretokenizerKind::Qwen35,
            (Some("deepseek4"), "gpt2", "joyai-llm") => PretokenizerKind::JoyAi,
            _ => {
                return Err(TokError::UnsupportedNativeTokenizer {
                    model: model.to_string(),
                    pre: pre.to_string(),
                });
            }
        };

        let token_texts = required_string_array(g, "tokenizer.ggml.tokens")?;
        let token_types = required_i64_array(g, "tokenizer.ggml.token_type")?;
        if token_texts.len() != token_types.len() {
            return Err(TokError::BadMetadata(format!(
                "tokenizer.ggml.tokens has {} entries but token_type has {}",
                token_texts.len(),
                token_types.len()
            )));
        }

        let mut id_to_token = Vec::with_capacity(token_texts.len());
        let mut token_to_id: HashMap<&str, i32> = HashMap::default();
        token_to_id.reserve(token_texts.len());
        for (id, (text, ty)) in token_texts.into_iter().zip(token_types).enumerate() {
            if token_to_id
                .insert(text, id_to_i32("tokenizer.ggml.tokens", id)?)
                .is_some()
            {
                return Err(TokError::BadMetadata(format!(
                    "duplicate tokenizer token text {text:?}"
                )));
            }
            id_to_token.push(NativeToken {
                text: text.to_owned(),
                attr: TokenAttr::from_gguf(ty)?,
            });
        }

        let mut byte_token_ids = [0i32; 256];
        for byte in 0u8..=255 {
            let byte_text = byte_to_unicode(byte).to_string();
            let token = token_to_id
                .get(byte_text.as_str())
                .copied()
                .ok_or_else(|| {
                    TokError::BadMetadata(format!("missing byte token for byte 0x{byte:02x}"))
                })?;
            byte_token_ids[byte as usize] = token;
        }

        let merges = required_string_array(g, "tokenizer.ggml.merges")?;
        let mut pair_merges = HashMap::default();
        pair_merges.reserve(merges.len());
        let mut merged_text = String::new();
        for (rank, &merge) in merges.iter().enumerate() {
            let (left, right) = split_merge(merge)?;
            let left_id = token_to_id.get(left).copied().ok_or_else(|| {
                TokError::BadMetadata(format!(
                    "merge {merge:?} references missing left token {left:?}"
                ))
            })?;
            let right_id = token_to_id.get(right).copied().ok_or_else(|| {
                TokError::BadMetadata(format!(
                    "merge {merge:?} references missing right token {right:?}"
                ))
            })?;
            merged_text.clear();
            merged_text.reserve(left.len() + right.len());
            merged_text.push_str(left);
            merged_text.push_str(right);
            let merged_id = token_to_id
                .get(merged_text.as_str())
                .copied()
                .ok_or_else(|| {
                    TokError::BadMetadata(format!(
                        "merge {merge:?} has no merged token {merged_text:?} in vocab"
                    ))
                })?;
            let old = pair_merges.insert(
                pair_key(left_id, right_id),
                MergeInfo {
                    rank: rank as u32,
                    merged_id,
                },
            );
            if old.is_some() {
                return Err(TokError::BadMetadata(format!(
                    "duplicate tokenizer.ggml.merges pair {left:?} {right:?}"
                )));
            }
        }

        let default_special = match pretokenizer {
            PretokenizerKind::Qwen35 => Some(11),
            PretokenizerKind::JoyAi => None,
        };
        let bos = optional_token_id(g, "tokenizer.ggml.bos_token_id")?.or(default_special);
        let eos = optional_token_id(g, "tokenizer.ggml.eos_token_id")?.or(default_special);
        validate_optional_token_id("tokenizer.ggml.bos_token_id", bos, id_to_token.len())?;
        validate_optional_token_id("tokenizer.ggml.eos_token_id", eos, id_to_token.len())?;
        let add_bos = optional_bool(g, "tokenizer.ggml.add_bos_token")?.unwrap_or(false);
        let add_eos = optional_bool(g, "tokenizer.ggml.add_eos_token")?.unwrap_or(false);
        validate_special_addition_config(pretokenizer, bos, eos, add_bos, add_eos)?;

        let mut special_tokens = Vec::new();
        for (id, token) in id_to_token.iter().enumerate() {
            if token.attr.is_partition_special()
                || (pretokenizer == PretokenizerKind::Qwen35 && is_qwen_control_text(&token.text))
            {
                special_tokens.push(SpecialToken {
                    text: token.text.clone(),
                    id: id_to_i32("tokenizer.ggml.tokens", id)?,
                });
            }
        }
        special_tokens.sort_by(|a, b| {
            b.text
                .len()
                .cmp(&a.text.len())
                .then_with(|| a.id.cmp(&b.id))
        });
        let special_matcher = SpecialMatcher::new(&special_tokens);

        let decoded_piece_bytes = (0..id_to_token.len()).map(|_| OnceLock::new()).collect();

        Ok(Self {
            id_to_token,
            pair_merges,
            byte_token_ids,
            special_matcher,
            decoded_piece_bytes,
            bos,
            eos,
            add_bos,
            add_eos,
            pretokenizer,
        })
    }

    fn checked_token(&self, token: i32) -> Result<usize, TokError> {
        if token < 0 || token as usize >= self.id_to_token.len() {
            return Err(TokError::InvalidToken(token));
        }
        Ok(token as usize)
    }

    pub fn n_vocab(&self) -> u32 {
        self.id_to_token.len().try_into().unwrap_or(u32::MAX)
    }

    pub fn bos(&self) -> Option<i32> {
        self.bos
    }

    pub fn eos(&self) -> Option<i32> {
        self.eos
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<i32>, TokError> {
        validate_native_input_len(text.len())?;
        let mut out = Vec::new();
        if add_special && self.add_bos {
            out.push(self.bos.expect("validated bos id"));
        }

        for fragment in self.partition_special(text) {
            match fragment {
                Fragment::Token(id) => out.push(id),
                Fragment::Text(s) => self.encode_raw(s, &mut out)?,
            }
        }

        if add_special && self.add_eos {
            out.push(self.eos.expect("validated eos id"));
        }
        Ok(out)
    }

    fn partition_special<'a>(&self, text: &'a str) -> Vec<Fragment<'a>> {
        self.special_matcher.partition(text)
    }

    fn encode_raw(&self, text: &str, out: &mut Vec<i32>) -> Result<(), TokError> {
        let pieces = match self.pretokenizer {
            PretokenizerKind::Qwen35 => qwen35_pretokenize(text),
            PretokenizerKind::JoyAi => joyai_pretokenize(text),
        };
        for piece in pieces {
            self.encode_bpe_piece(piece.as_bytes(), out);
        }
        Ok(())
    }

    fn encode_bpe_piece(&self, piece: &[u8], out: &mut Vec<i32>) {
        if piece.is_empty() {
            return;
        }
        let mut symbols = symbols_for_piece(piece, &self.byte_token_ids);
        if symbols.is_empty() {
            return;
        }

        let mut queue = BinaryHeap::new();
        for i in 1..symbols.len() {
            self.add_bigram(&symbols, i - 1, i, &mut queue);
        }

        while let Some(bigram) = queue.pop() {
            if !valid_bigram(&symbols, &bigram) {
                continue;
            }
            let left = bigram.left;
            let right = bigram.right;
            symbols[left].id = bigram.merged_id;
            symbols[right].alive = false;
            let next = symbols[right].next;
            symbols[left].next = next;
            if let Some(next) = next {
                symbols[next].prev = Some(left);
            }
            if let Some(prev) = symbols[left].prev {
                self.add_bigram(&symbols, prev, left, &mut queue);
            }
            if let Some(next) = symbols[left].next {
                self.add_bigram(&symbols, left, next, &mut queue);
            }
        }

        let mut idx = Some(0usize);
        while let Some(i) = idx {
            let sym = &symbols[i];
            if sym.alive {
                out.push(sym.id);
            }
            idx = sym.next;
        }
    }

    fn add_bigram(
        &self,
        symbols: &[Symbol],
        left: usize,
        right: usize,
        queue: &mut BinaryHeap<Bigram>,
    ) {
        if !symbols[left].alive || !symbols[right].alive {
            return;
        }
        let left_id = symbols[left].id;
        let right_id = symbols[right].id;
        if let Some(&merge) = self.pair_merges.get(&pair_key(left_id, right_id)) {
            queue.push(Bigram {
                left,
                right,
                left_id,
                right_id,
                rank: merge.rank,
                merged_id: merge.merged_id,
            });
        }
    }

    pub fn try_decode_piece(&self, token: i32) -> Result<String, TokError> {
        let token = self.checked_token(token)?;
        let bytes = self.decode_token_bytes(token);
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    /// Return the exact decoded bytes for one token.
    ///
    /// Unlike [`Self::try_decode_piece`], this accessor does not replace
    /// invalid UTF-8. It is the required primitive for byte-level grammar
    /// matching, where lossy text conversion can change admissibility.
    pub fn try_decode_piece_bytes_exact(&self, token: i32) -> Result<&[u8], TokError> {
        let token = self.checked_token(token)?;
        Ok(self.decode_token_bytes(token))
    }

    /// Whether a token is ordinary generated content rather than control,
    /// unknown, user-defined, unused, or padded vocabulary state.
    ///
    /// This is a conservative Qwen grammar-content policy, not a universal
    /// protocol rule: another model may intentionally generate user-defined
    /// tokens. Callers must still reject empty decoded pieces before
    /// constructing a token-level grammar graph.
    pub fn is_ordinary_content_token(&self, token: i32) -> Result<bool, TokError> {
        let token = self.checked_token(token)?;
        let data = &self.id_to_token[token];
        Ok(matches!(data.attr, TokenAttr::Normal | TokenAttr::Byte)
            && !is_qwen_control_text(&data.text))
    }

    pub fn try_decode(&self, tokens: &[i32]) -> Result<String, TokError> {
        let mut ids = Vec::with_capacity(tokens.len());
        let mut total = 0usize;
        for &token in tokens {
            let idx = self.checked_token(token)?;
            ids.push(idx);
            total += self.decode_token_bytes(idx).len();
        }
        let mut bytes = Vec::with_capacity(total);
        for idx in ids {
            bytes.extend_from_slice(self.decode_token_bytes(idx));
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn decode_token_bytes(&self, token: usize) -> &[u8] {
        self.decoded_piece_bytes[token].get_or_init(|| {
            decode_token_bytes_uncached(&self.id_to_token[token]).into_boxed_slice()
        })
    }

    pub fn decode_piece(&self, token: i32) -> String {
        self.try_decode_piece(token).unwrap_or_default()
    }

    pub fn decode(&self, tokens: &[i32]) -> String {
        self.try_decode(tokens).unwrap_or_else(|_| {
            let mut out = String::new();
            for &t in tokens {
                out.push_str(&self.decode_piece(t));
            }
            out
        })
    }
}

impl Tokenize for NativeTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<i32>, TokError> {
        NativeTokenizer::encode(self, text, add_special)
    }
    fn try_decode_piece(&self, token: i32) -> Result<String, TokError> {
        NativeTokenizer::try_decode_piece(self, token)
    }
    fn try_decode(&self, tokens: &[i32]) -> Result<String, TokError> {
        NativeTokenizer::try_decode(self, tokens)
    }
    fn decode(&self, tokens: &[i32]) -> String {
        NativeTokenizer::decode(self, tokens)
    }
    fn n_vocab(&self) -> u32 {
        NativeTokenizer::n_vocab(self)
    }
    fn bos(&self) -> Option<i32> {
        NativeTokenizer::bos(self)
    }
    fn eos(&self) -> Option<i32> {
        NativeTokenizer::eos(self)
    }
}

#[derive(Clone, Copy, Debug)]
enum Fragment<'a> {
    Text(&'a str),
    Token(i32),
}

#[derive(Default)]
struct SpecialTrieNode {
    edges: HashMap<u8, usize>,
    terminal: Option<i32>,
}

struct SpecialMatcher {
    nodes: Vec<SpecialTrieNode>,
}

impl SpecialMatcher {
    fn new(tokens: &[SpecialToken]) -> Self {
        let mut nodes = vec![SpecialTrieNode::default()];
        for token in tokens {
            if token.text.is_empty() {
                continue;
            }
            let mut idx = 0usize;
            for &byte in token.text.as_bytes() {
                let next = if let Some(&child) = nodes[idx].edges.get(&byte) {
                    child
                } else {
                    let child = nodes.len();
                    nodes.push(SpecialTrieNode::default());
                    nodes[idx].edges.insert(byte, child);
                    child
                };
                idx = next;
            }
            nodes[idx].terminal = Some(token.id);
        }
        Self { nodes }
    }

    fn partition<'a>(&self, text: &'a str) -> Vec<Fragment<'a>> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut raw_start = 0usize;
        let mut pos = 0usize;
        while pos < bytes.len() {
            if let Some((end, id)) = self.match_at(bytes, pos) {
                if raw_start < pos {
                    out.push(Fragment::Text(&text[raw_start..pos]));
                }
                out.push(Fragment::Token(id));
                raw_start = end;
                pos = end;
            } else {
                pos += 1;
            }
        }
        if raw_start < text.len() {
            out.push(Fragment::Text(&text[raw_start..]));
        }
        out
    }

    fn match_at(&self, bytes: &[u8], start: usize) -> Option<(usize, i32)> {
        let mut idx = 0usize;
        let mut pos = start;
        let mut best = None;
        while pos < bytes.len() {
            let Some(&next) = self.nodes[idx].edges.get(&bytes[pos]) else {
                break;
            };
            idx = next;
            pos += 1;
            if let Some(id) = self.nodes[idx].terminal {
                best = Some((pos, id));
            }
        }
        best
    }
}

#[derive(Clone, Debug)]
struct Symbol {
    id: i32,
    prev: Option<usize>,
    next: Option<usize>,
    alive: bool,
}

fn symbols_for_piece(piece: &[u8], byte_token_ids: &[i32; 256]) -> Vec<Symbol> {
    let mut symbols = Vec::with_capacity(piece.len());
    for (i, &byte) in piece.iter().enumerate() {
        symbols.push(Symbol {
            id: byte_token_ids[byte as usize],
            prev: i.checked_sub(1),
            next: (i + 1 < piece.len()).then_some(i + 1),
            alive: true,
        });
    }
    symbols
}

#[derive(Clone, Copy, Debug)]
struct MergeInfo {
    rank: u32,
    merged_id: i32,
}

fn pair_key(left: i32, right: i32) -> u64 {
    debug_assert!(left >= 0 && right >= 0);
    ((left as u32 as u64) << 32) | (right as u32 as u64)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Bigram {
    left: usize,
    right: usize,
    left_id: i32,
    right_id: i32,
    rank: u32,
    merged_id: i32,
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn valid_bigram(symbols: &[Symbol], bigram: &Bigram) -> bool {
    let Some(left) = symbols.get(bigram.left) else {
        return false;
    };
    let Some(right) = symbols.get(bigram.right) else {
        return false;
    };
    if !left.alive
        || !right.alive
        || left.next != Some(bigram.right)
        || right.prev != Some(bigram.left)
    {
        return false;
    }
    left.id == bigram.left_id && right.id == bigram.right_id
}

#[derive(Clone, Copy, Debug, Default)]
struct CharFlags {
    is_number: bool,
    is_letter: bool,
    is_accent_mark: bool,
    is_punct_or_symbol: bool,
    is_whitespace: bool,
    any: bool,
}

impl CharFlags {
    fn for_char(ch: char) -> Self {
        let category = get_general_category(ch);
        Self {
            is_number: matches!(
                category,
                GeneralCategory::DecimalNumber
                    | GeneralCategory::LetterNumber
                    | GeneralCategory::OtherNumber
            ),
            is_letter: matches!(
                category,
                GeneralCategory::UppercaseLetter
                    | GeneralCategory::LowercaseLetter
                    | GeneralCategory::TitlecaseLetter
                    | GeneralCategory::ModifierLetter
                    | GeneralCategory::OtherLetter
            ),
            is_accent_mark: matches!(
                category,
                GeneralCategory::NonspacingMark
                    | GeneralCategory::SpacingMark
                    | GeneralCategory::EnclosingMark
            ),
            is_punct_or_symbol: matches!(
                category,
                GeneralCategory::ConnectorPunctuation
                    | GeneralCategory::DashPunctuation
                    | GeneralCategory::OpenPunctuation
                    | GeneralCategory::ClosePunctuation
                    | GeneralCategory::InitialPunctuation
                    | GeneralCategory::FinalPunctuation
                    | GeneralCategory::OtherPunctuation
                    | GeneralCategory::MathSymbol
                    | GeneralCategory::CurrencySymbol
                    | GeneralCategory::ModifierSymbol
                    | GeneralCategory::OtherSymbol
            ),
            is_whitespace: ch.is_whitespace(),
            any: true,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CharInfo {
    ch: char,
    start: usize,
    flags: CharFlags,
}

fn qwen35_pretokenize(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    let chars: Vec<CharInfo> = text
        .char_indices()
        .map(|(start, ch)| CharInfo {
            ch,
            start,
            flags: CharFlags::for_char(ch),
        })
        .collect();
    let mut out = Vec::new();
    let mut prev = 0usize;
    let mut pos = 0usize;
    while pos < chars.len() {
        let ch = chars[pos].ch;
        let flags = chars[pos].flags;

        if ch == '\'' && pos + 1 < chars.len() {
            let next = chars[pos + 1].ch.to_ascii_lowercase();
            if matches!(next, 's' | 't' | 'm' | 'd') {
                push_token(text, &chars, &mut out, &mut prev, pos + 2);
                pos += 2;
                continue;
            }
            if pos + 2 < chars.len() {
                let next2 = chars[pos + 2].ch.to_ascii_lowercase();
                if (next == 'r' && next2 == 'e')
                    || (next == 'v' && next2 == 'e')
                    || (next == 'l' && next2 == 'l')
                {
                    push_token(text, &chars, &mut out, &mut prev, pos + 3);
                    pos += 3;
                    continue;
                }
            }
        }

        if !(ch == '\r' || ch == '\n' || flags.is_number)
            && (flags.is_letter
                || flags.is_accent_mark
                || char_flags(&chars, pos + 1).is_accent_mark
                || char_flags(&chars, pos + 1).is_letter)
        {
            pos += 1;
            while {
                let f = char_flags(&chars, pos);
                f.is_letter || f.is_accent_mark
            } {
                pos += 1;
            }
            push_token(text, &chars, &mut out, &mut prev, pos);
            continue;
        }

        if flags.is_number {
            pos += 1;
            push_token(text, &chars, &mut out, &mut prev, pos);
            continue;
        }

        let mut flags2 = if ch == ' ' {
            char_flags(&chars, pos + 1)
        } else {
            flags
        };
        if !(flags2.is_whitespace || flags2.is_letter || flags2.is_accent_mark || flags2.is_number)
            && flags.any
        {
            if ch == ' ' {
                pos += 1;
            }
            while !(flags2.is_whitespace
                || flags2.is_letter
                || flags2.is_accent_mark
                || flags2.is_number)
                && flags2.any
            {
                pos += 1;
                flags2 = char_flags(&chars, pos);
            }
            while char_at(&chars, pos).is_some_and(|c| c == '\r' || c == '\n') {
                pos += 1;
            }
            push_token(text, &chars, &mut out, &mut prev, pos);
            continue;
        }

        let mut num_whitespaces = 0usize;
        let mut last_end_r_or_n = 0usize;
        while char_flags(&chars, pos + num_whitespaces).is_whitespace {
            let c = chars[pos + num_whitespaces].ch;
            if c == '\r' || c == '\n' {
                last_end_r_or_n = pos + num_whitespaces + 1;
            }
            num_whitespaces += 1;
        }
        if last_end_r_or_n > 0 {
            pos = last_end_r_or_n;
            push_token(text, &chars, &mut out, &mut prev, pos);
            continue;
        }
        if num_whitespaces > 1 && pos + num_whitespaces < chars.len() {
            pos += num_whitespaces - 1;
            push_token(text, &chars, &mut out, &mut prev, pos);
            continue;
        }
        if num_whitespaces > 0 {
            pos += num_whitespaces;
            push_token(text, &chars, &mut out, &mut prev, pos);
            continue;
        }

        pos += 1;
        push_token(text, &chars, &mut out, &mut prev, pos);
    }
    out
}

/// DeepSeek V3/V4 and JoyAI apply three isolated regex splits in sequence:
/// numbers in groups of at most three, fixed CJK/kana runs, then the main
/// letter/punctuation/whitespace expression. This scalar implementation keeps
/// number and CJK regions as hard boundaries while applying the final split.
fn joyai_pretokenize(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    let chars: Vec<CharInfo> = text
        .char_indices()
        .map(|(start, ch)| CharInfo {
            ch,
            start,
            flags: CharFlags::for_char(ch),
        })
        .collect();
    let mut out = Vec::new();
    let mut prev = 0usize;
    let mut pos = 0usize;
    while pos < chars.len() {
        let end = if chars[pos].flags.is_number {
            let mut end = pos + 1;
            while end < chars.len() && end - pos < 3 && chars[end].flags.is_number {
                end += 1;
            }
            end
        } else {
            let cjk_region = is_joyai_cjk(chars[pos].ch);
            let mut region_end = pos + 1;
            while region_end < chars.len()
                && !chars[region_end].flags.is_number
                && is_joyai_cjk(chars[region_end].ch) == cjk_region
            {
                region_end += 1;
            }
            joyai_main_token_end(&chars, pos, region_end)
        };
        push_token(text, &chars, &mut out, &mut prev, end);
        pos = end;
    }
    out
}

fn joyai_main_token_end(chars: &[CharInfo], pos: usize, region_end: usize) -> usize {
    let current = chars[pos];

    if current.ch.is_ascii_punctuation()
        && pos + 1 < region_end
        && chars[pos + 1].ch.is_ascii_alphabetic()
    {
        let mut end = pos + 2;
        while end < region_end && chars[end].ch.is_ascii_alphabetic() {
            end += 1;
        }
        return end;
    }

    if is_letter_or_mark(current.flags) {
        return scan_joyai_letters_and_marks(chars, pos + 1, region_end);
    }
    if current.ch != '\r'
        && current.ch != '\n'
        && !current.flags.is_letter
        && !current.flags.is_punct_or_symbol
        && pos + 1 < region_end
        && is_letter_or_mark(chars[pos + 1].flags)
    {
        return scan_joyai_letters_and_marks(chars, pos + 2, region_end);
    }

    let punct_start =
        if current.ch == ' ' && pos + 1 < region_end && chars[pos + 1].flags.is_punct_or_symbol {
            Some(pos + 1)
        } else if current.flags.is_punct_or_symbol {
            Some(pos)
        } else {
            None
        };
    if let Some(punct_start) = punct_start {
        let mut end = punct_start + 1;
        while end < region_end && chars[end].flags.is_punct_or_symbol {
            end += 1;
        }
        while end < region_end && matches!(chars[end].ch, '\r' | '\n') {
            end += 1;
        }
        return end;
    }

    if current.flags.is_whitespace {
        return joyai_whitespace_end(chars, pos, region_end);
    }

    let mut end = pos + 1;
    while end < region_end && joyai_is_gap_char(chars[end].flags) {
        if end + 1 < region_end && is_letter_or_mark(chars[end + 1].flags) {
            break;
        }
        end += 1;
    }
    end
}

fn scan_joyai_letters_and_marks(chars: &[CharInfo], mut pos: usize, region_end: usize) -> usize {
    while pos < region_end && is_letter_or_mark(chars[pos].flags) {
        pos += 1;
    }
    pos
}

fn joyai_whitespace_end(chars: &[CharInfo], pos: usize, region_end: usize) -> usize {
    let mut end = pos;
    let mut last_newline_end = None;
    while end < region_end && chars[end].flags.is_whitespace {
        end += 1;
        if matches!(chars[end - 1].ch, '\r' | '\n') {
            last_newline_end = Some(end);
        }
    }
    if let Some(last_newline_end) = last_newline_end {
        return last_newline_end;
    }
    if end == region_end {
        return end;
    }
    if end - pos > 1 {
        return end - 1;
    }
    end
}

fn is_letter_or_mark(flags: CharFlags) -> bool {
    flags.is_letter || flags.is_accent_mark
}

fn joyai_is_gap_char(flags: CharFlags) -> bool {
    !flags.is_number
        && !flags.is_letter
        && !flags.is_accent_mark
        && !flags.is_punct_or_symbol
        && !flags.is_whitespace
}

fn is_joyai_cjk(ch: char) -> bool {
    matches!(ch as u32, 0x4e00..=0x9fa5 | 0x3040..=0x30ff)
}

fn push_token<'a>(
    text: &'a str,
    chars: &[CharInfo],
    out: &mut Vec<&'a str>,
    prev: &mut usize,
    end_pos: usize,
) {
    let end = if end_pos == chars.len() {
        text.len()
    } else {
        chars[end_pos].start
    };
    if *prev < end {
        out.push(&text[*prev..end]);
    }
    *prev = end;
}

fn char_flags(chars: &[CharInfo], pos: usize) -> CharFlags {
    chars.get(pos).map(|c| c.flags).unwrap_or_default()
}

fn char_at(chars: &[CharInfo], pos: usize) -> Option<char> {
    chars.get(pos).map(|c| c.ch)
}

fn byte_to_unicode(byte: u8) -> char {
    match byte {
        0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff => byte as char,
        _ => {
            let mut n = 0u32;
            for b in 0u8..=255 {
                if matches!(b, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff) {
                    continue;
                }
                if b == byte {
                    return char::from_u32(256 + n).expect("byte unicode scalar");
                }
                n += 1;
            }
            unreachable!("all u8 values are covered")
        }
    }
}

fn unicode_to_byte(ch: char) -> Option<u8> {
    let cpt = ch as u32;
    if matches!(cpt, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff) {
        return Some(cpt as u8);
    }
    let mut n = 0u32;
    for b in 0u8..=255 {
        if matches!(b, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff) {
            continue;
        }
        if cpt == 256 + n {
            return Some(b);
        }
        n += 1;
    }
    None
}

fn unknown_byte_text(ch: char, token_text: &str) -> Vec<u8> {
    let mut out = String::from("[UNK_BYTE_0x");
    let mut buf = [0u8; 4];
    for byte in ch.encode_utf8(&mut buf).as_bytes() {
        out.push_str(&format!("{byte:02x}"));
    }
    out.push_str(token_text);
    out.push(']');
    out.into_bytes()
}

fn decode_token_bytes_uncached(data: &NativeToken) -> Vec<u8> {
    if data.attr.is_decode_literal() || is_qwen_control_text(&data.text) {
        return data.text.as_bytes().to_vec();
    }
    if matches!(data.attr, TokenAttr::Unused | TokenAttr::Undefined) {
        return Vec::new();
    }
    if data.attr == TokenAttr::Byte
        && let Some(byte) = parse_hex_byte_token(&data.text)
    {
        return vec![byte];
    }

    let mut out = Vec::with_capacity(data.text.len());
    for ch in data.text.chars() {
        if let Some(byte) = unicode_to_byte(ch) {
            out.push(byte);
        } else {
            out.extend(unknown_byte_text(ch, &data.text));
        }
    }
    out
}

fn split_merge(merge: &str) -> Result<(&str, &str), TokError> {
    let bytes = merge.as_bytes();
    let Some(rel) = bytes.iter().skip(1).position(|&b| b == b' ') else {
        return Err(TokError::BadMetadata(format!(
            "malformed tokenizer.ggml.merges entry {merge:?}"
        )));
    };
    let sep = rel + 1;
    let left = &merge[..sep];
    let right = &merge[sep + 1..];
    if left.is_empty() || right.is_empty() {
        return Err(TokError::BadMetadata(format!(
            "malformed tokenizer.ggml.merges entry {merge:?}"
        )));
    }
    Ok((left, right))
}

fn required_str<'a>(g: &'a GgufFile, key: &str) -> Result<&'a str, TokError> {
    g.get_str(key)
        .ok_or_else(|| TokError::BadMetadata(format!("missing string metadata key {key:?}")))
}

fn required_string_array<'a>(g: &'a GgufFile, key: &str) -> Result<Vec<&'a str>, TokError> {
    let value = g
        .model
        .metadata()
        .get(key)
        .ok_or_else(|| TokError::BadMetadata(format!("missing array metadata key {key:?}")))?;
    let arr = value
        .as_array()
        .ok_or_else(|| TokError::BadMetadata(format!("metadata key {key:?} is not an array")))?;
    let mut out = Vec::with_capacity(arr.len());
    for (idx, value) in arr.iter().enumerate() {
        let s = value.as_str().ok_or_else(|| {
            TokError::BadMetadata(format!("metadata key {key:?}[{idx}] is not a string"))
        })?;
        out.push(s);
    }
    Ok(out)
}

fn required_i64_array(g: &GgufFile, key: &str) -> Result<Vec<i64>, TokError> {
    let value = g
        .model
        .metadata()
        .get(key)
        .ok_or_else(|| TokError::BadMetadata(format!("missing array metadata key {key:?}")))?;
    let arr = value
        .as_array()
        .ok_or_else(|| TokError::BadMetadata(format!("metadata key {key:?} is not an array")))?;
    let mut out = Vec::with_capacity(arr.len());
    for (idx, value) in arr.iter().enumerate() {
        let n = value
            .as_i64()
            .or_else(|| value.as_u64().and_then(|n| i64::try_from(n).ok()))
            .ok_or_else(|| {
                TokError::BadMetadata(format!("metadata key {key:?}[{idx}] is not an integer"))
            })?;
        out.push(n);
    }
    Ok(out)
}

fn validate_optional_token_id(
    key: &str,
    value: Option<i32>,
    n_vocab: usize,
) -> Result<(), TokError> {
    let Some(value) = value else { return Ok(()) };
    if value < 0 || value as usize >= n_vocab {
        return Err(TokError::BadMetadata(format!(
            "metadata key {key:?} token id {value} is outside vocab size {n_vocab}"
        )));
    }
    Ok(())
}

fn optional_bool(g: &GgufFile, key: &str) -> Result<Option<bool>, TokError> {
    let Some(value) = g.model.metadata().get(key) else {
        return Ok(None);
    };
    value
        .as_bool()
        .map(Some)
        .ok_or_else(|| TokError::BadMetadata(format!("metadata key {key:?} is not a bool")))
}

fn optional_token_id(g: &GgufFile, key: &str) -> Result<Option<i32>, TokError> {
    let Some(value) = g.model.metadata().get(key) else {
        return Ok(None);
    };
    value_to_i32(value, key).map(Some)
}

fn value_to_i32(value: &Value, key: &str) -> Result<i32, TokError> {
    let n = value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|n| i64::try_from(n).ok()))
        .ok_or_else(|| TokError::BadMetadata(format!("metadata key {key:?} is not an integer")))?;
    i32::try_from(n)
        .map_err(|_| TokError::BadMetadata(format!("metadata key {key:?} value {n} exceeds i32")))
}

fn validate_special_addition_config(
    pretokenizer: PretokenizerKind,
    bos: Option<i32>,
    eos: Option<i32>,
    add_bos: bool,
    add_eos: bool,
) -> Result<(), TokError> {
    if pretokenizer == PretokenizerKind::JoyAi && (bos.is_none() || eos.is_none()) {
        return Err(TokError::BadMetadata(
            "JoyAI tokenizer requires explicit BOS and EOS token ids".into(),
        ));
    }
    if add_bos && bos.is_none() {
        return Err(TokError::BadMetadata(
            "tokenizer enables BOS insertion without a BOS token id".into(),
        ));
    }
    if add_eos && eos.is_none() {
        return Err(TokError::BadMetadata(
            "tokenizer enables EOS insertion without an EOS token id".into(),
        ));
    }
    Ok(())
}

fn id_to_i32(key: &str, id: usize) -> Result<i32, TokError> {
    i32::try_from(id).map_err(|_| TokError::BadMetadata(format!("{key} index {id} exceeds i32")))
}

fn is_qwen_control_text(text: &str) -> bool {
    matches!(
        text,
        "<|endoftext|>"
            | "<|im_start|>"
            | "<|im_end|>"
            | "<|fim_prefix|>"
            | "<|fim_middle|>"
            | "<|fim_suffix|>"
            | "<|fim_pad|>"
    )
}

fn parse_hex_byte_token(text: &str) -> Option<u8> {
    let hex = text.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fancy_regex::Regex;
    use proptest::prelude::*;

    const DS4_0731_IQ3: &str = "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf";

    #[test]
    fn raw_i32le_token_digest_matches_frozen_vectors() {
        assert_eq!(
            token_ids_sha256_i32le(&[]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            token_ids_sha256_i32le(&[1, -2, 248_319]),
            "3f37364bc87f9ff835c64d4bdb3d993fe35e530097697da8cbacb5e6f92119d5"
        );
    }

    struct OraclePair {
        path: &'static str,
        ffi: LlamaCppTokenizer,
        native: NativeTokenizer,
    }

    fn oracle_pair() -> Option<&'static OraclePair> {
        static ORACLE: std::sync::OnceLock<Option<OraclePair>> = std::sync::OnceLock::new();
        ORACLE
            .get_or_init(|| {
                let path = fixture()?;
                Some(OraclePair {
                    path,
                    ffi: LlamaCppTokenizer::open(path).expect("open ffi tokenizer"),
                    native: NativeTokenizer::open(path).expect("open native tokenizer"),
                })
            })
            .as_ref()
    }

    fn fixtures() -> Vec<&'static str> {
        let candidates = [
            "/Users/tito/models/Qwen3.5-0.8B.F32.gguf",
            "/Users/tito/models/Qwen3.5-0.8B-BF16.gguf",
            "/Users/tito/models/Qwen3.5-4B-BF16.gguf",
            "/Users/tito/models/Qwen3.5-27B-Q4_K_M.gguf",
            "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf",
            "/Users/tito/models/Qwen3.6-27B-MTP-Q4_K_M.gguf",
            "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        ];
        candidates
            .iter()
            .copied()
            .filter(|p| Path::new(p).exists())
            .collect()
    }

    fn fixture() -> Option<&'static str> {
        fixtures().into_iter().next()
    }

    fn joyai_reference_tokens(text: &str) -> Vec<String> {
        const PATTERNS: [&str; 3] = [
            r"\p{N}{1,3}",
            "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+",
            r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##,
        ];
        let mut pieces = vec![text.to_string()];
        for pattern in PATTERNS {
            let regex = Regex::new(pattern).expect("compile JoyAI reference regex");
            let mut split = Vec::new();
            for piece in pieces {
                let mut last = 0usize;
                for matched in regex.find_iter(&piece) {
                    let matched = matched.expect("match JoyAI reference regex");
                    if matched.start() > last {
                        split.push(piece[last..matched.start()].to_string());
                    }
                    split.push(matched.as_str().to_string());
                    last = matched.end();
                }
                if last < piece.len() {
                    split.push(piece[last..].to_string());
                }
            }
            pieces = split;
        }
        pieces
    }

    fn adversarial_prompts() -> &'static [&'static str] {
        &[
            "",
            "Hello, world!",
            "   leading and trailing   ",
            "line1\nline2\r\nline3",
            "\n\n\n",
            "\t\tfn main() { println!(\"hi\"); }",
            "I can't believe they're testing Qwen's tokenizer.",
            "quote opener: 'verbose and \"quoted\" text",
            "digits 1 12 123 1234 １２３ ①Ⅻ",
            "数字123和标点，emoji🙂 + variation❤\u{fe0f}",
            "family emoji: 👨‍👩‍👧‍👦 and scientist 👩🏽‍🔬",
            "combining: e\u{301} cafe\u{301} a\u{20dd}",
            "nbsp:\u{00a0}thin:\u{2009}em:\u{2003}zwsp:\u{200b}",
            "cjk + ascii + digits: 上海 2010 Boston 未来",
            "<|im_start|>user\nhi<|im_end|>",
            "x<|im_start|><|im_end|>y",
            "<|endoftext|><|im_start|>assistant\n<think>hi</think>",
            "<|fim_prefix|>code<|fim_middle|>body<|fim_suffix|>",
            "```rust\nfn f(x: usize) -> usize { x + 1 }\n```",
            "json: {\"tools\":[{\"name\":\"search\",\"parameters\":{\"query\":\"hi\"}}]}",
        ]
    }

    fn assert_tokenizers_match(
        ffi: &LlamaCppTokenizer,
        native: &NativeTokenizer,
        path: &str,
        prompt: &str,
        add_special: bool,
    ) {
        assert_eq!(native.n_vocab(), ffi.n_vocab(), "path={path}");
        assert_eq!(native.bos(), ffi.bos(), "path={path}");
        assert_eq!(native.eos(), ffi.eos(), "path={path}");
        let ffi_ids = ffi.encode(prompt, add_special).expect("ffi encode");
        let native_ids = native.encode(prompt, add_special).expect("native encode");
        assert_eq!(
            native_ids, ffi_ids,
            "encode mismatch path={path} prompt={prompt:?} add_special={add_special}"
        );
        assert_eq!(
            native.try_decode(&native_ids).expect("native decode"),
            ffi.try_decode(&ffi_ids).expect("ffi decode"),
            "decode mismatch path={path} prompt={prompt:?} add_special={add_special}"
        );
    }

    #[derive(Clone, Copy)]
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 32) as u32
        }

        fn gen_range(&mut self, upper: u32) -> u32 {
            if upper == 0 {
                0
            } else {
                self.next_u32() % upper
            }
        }
    }

    fn generated_unicode_prompts() -> Vec<String> {
        let chunks = [
            " ",
            "\t",
            "\n",
            "\r\n",
            "a",
            "Z",
            "foo",
            "bar",
            "'s",
            "'re",
            "123",
            "１２３",
            "①",
            "Ⅻ",
            "数字",
            "上海",
            "🙂",
            "👨‍👩‍👧‍👦",
            "👩🏽‍🔬",
            "e\u{301}",
            "❤\u{fe0f}",
            "\u{00a0}",
            "\u{2009}",
            "\u{2003}",
            "\u{200b}",
            "{",
            "}",
            "[",
            "]",
            ":",
            ",",
            "\"",
            "<|im_start|>",
            "<|im_end|>",
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
        ];
        let mut rng = Lcg::new(0x1234_5678_9abc_def0);
        let mut out = Vec::with_capacity(96);
        for _ in 0..96 {
            let n = 1 + rng.gen_range(24) as usize;
            let mut s = String::new();
            for _ in 0..n {
                s.push_str(chunks[rng.gen_range(chunks.len() as u32) as usize]);
            }
            out.push(s);
        }
        out
    }

    fn fuzz_prompt_strategy() -> impl Strategy<Value = String> {
        const CHUNKS: &[&str] = &[
            " ",
            "\t",
            "\n",
            "\r\n",
            "a",
            "Z",
            "foo",
            "bar",
            "'s",
            "'re",
            "123",
            "１２３",
            "①",
            "Ⅻ",
            "数字",
            "上海",
            "🙂",
            "👨‍👩‍👧‍👦",
            "👩🏽‍🔬",
            "e\u{301}",
            "❤\u{fe0f}",
            "\u{00a0}",
            "\u{2009}",
            "\u{2003}",
            "\u{200b}",
            "{",
            "}",
            "[",
            "]",
            ":",
            ",",
            "\"",
            "<|im_start|>",
            "<|im_end|>",
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
            "```rust\n",
            "```\n",
        ];
        prop::collection::vec(0usize..CHUNKS.len(), 0..40).prop_map(|idxs| {
            let mut s = String::new();
            for idx in idxs {
                s.push_str(CHUNKS[idx]);
            }
            s
        })
    }

    fn special_token_texts(native: &NativeTokenizer) -> Vec<String> {
        let mut texts = Vec::new();
        for token in &native.id_to_token {
            if token.attr.is_partition_special() || is_qwen_control_text(&token.text) {
                texts.push(token.text.clone());
            }
        }
        texts.sort();
        texts.dedup();
        texts
    }

    #[test]
    fn opens_and_encodes() {
        let Some(path) = fixture() else { return };
        let tok = Tokenizer::open(path).expect("open tokenizer");
        assert!(tok.n_vocab() >= 248_000, "n_vocab={}", tok.n_vocab());

        // A canonical short prompt that should produce a well-known short
        // token sequence on Qwen3 family.
        let ids = tok.encode("Hello, world!", false).expect("tokenize");
        assert!(!ids.is_empty());
        assert!(
            ids.len() <= 8,
            "sanity: short prompt got {} tokens",
            ids.len()
        );

        let decoded = tok.decode(&ids);
        assert_eq!(
            decoded.trim_end_matches('\0'),
            "Hello, world!",
            "round-trip mismatch"
        );

        eprintln!(
            "[tokenizer] n_vocab={} bos={:?} eos={:?} 'Hello, world!' -> {:?} ({} tokens)",
            tok.n_vocab(),
            tok.bos(),
            tok.eos(),
            ids,
            ids.len()
        );
    }

    #[test]
    fn special_tokens_round_trip() {
        let Some(path) = fixture() else { return };
        let tok = Tokenizer::open(path).expect("open");
        // <|im_start|> is one of the chat template specials.
        let ids = tok.encode("<|im_start|>user\nhi<|im_end|>", false).unwrap();
        let s = tok.decode(&ids);
        assert!(s.contains("<|im_start|>"), "got: {s:?}");
        assert!(s.contains("<|im_end|>"), "got: {s:?}");
    }

    #[test]
    fn rejects_path_with_interior_nul() {
        let err = LlamaCppTokenizer::open(Path::new("/tmp/bad\0path.gguf"))
            .err()
            .expect("nul path");
        assert!(matches!(err, TokError::PathContainsNul(_)));
    }

    #[test]
    fn checked_i32_len_rejects_oversized_inputs() {
        let err = checked_i32_len(i32::MAX as usize + 1).expect_err("oversized input");
        assert!(matches!(err, TokError::InputBytesTooLong(_)));
    }

    #[test]
    fn native_input_cap_rejects_oversized_inputs() {
        let err = validate_native_input_len(NATIVE_MAX_INPUT_BYTES + 1)
            .expect_err("native oversized input");
        assert!(matches!(err, TokError::NativeInputTooLong { .. }));
    }

    #[test]
    fn special_insertion_requires_declared_token_ids() {
        assert!(
            validate_special_addition_config(PretokenizerKind::Qwen35, None, Some(1), true, false)
                .is_err()
        );
        assert!(
            validate_special_addition_config(PretokenizerKind::Qwen35, Some(0), None, false, true)
                .is_err()
        );
        assert!(
            validate_special_addition_config(PretokenizerKind::Qwen35, None, None, false, false)
                .is_ok()
        );
        assert!(
            validate_special_addition_config(PretokenizerKind::JoyAi, None, Some(1), false, false)
                .is_err()
        );
    }

    #[test]
    fn rejects_invalid_decode_token() {
        let Some(path) = fixture() else { return };
        let tok = Tokenizer::open(path).expect("open tokenizer");
        let err = tok
            .try_decode_piece(-1)
            .expect_err("negative token should fail");
        assert!(matches!(err, TokError::InvalidToken(-1)));
        let err = tok
            .try_decode(&[tok.n_vocab() as i32])
            .expect_err("oob token should fail");
        assert!(matches!(err, TokError::InvalidToken(_)));
    }

    #[test]
    fn native_matches_llama_cpp_oracle_on_edge_prompts() {
        let Some(path) = fixture() else { return };
        let ffi = LlamaCppTokenizer::open(path).expect("open ffi tokenizer");
        let native = NativeTokenizer::open(path).expect("open native tokenizer");
        for prompt in adversarial_prompts() {
            for add_special in [false, true] {
                assert_tokenizers_match(&ffi, &native, path, prompt, add_special);
            }
        }
    }

    #[test]
    fn native_matches_llama_cpp_on_generated_unicode_prompts() {
        let Some(path) = fixture() else { return };
        let ffi = LlamaCppTokenizer::open(path).expect("open ffi tokenizer");
        let native = NativeTokenizer::open(path).expect("open native tokenizer");
        for prompt in generated_unicode_prompts() {
            for add_special in [false, true] {
                assert_tokenizers_match(&ffi, &native, path, &prompt, add_special);
            }
        }
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 64,
            max_shrink_iters: 0,
            .. proptest::test_runner::Config::default()
        })]

        #[test]
        fn native_matches_llama_cpp_property_fuzz(
            prompt in fuzz_prompt_strategy(),
            add_special in any::<bool>(),
        ) {
            if let Some(oracle) = oracle_pair() {
                assert_tokenizers_match(
                    &oracle.ffi,
                    &oracle.native,
                    oracle.path,
                    &prompt,
                    add_special,
                );
            }
        }

        #[test]
        fn joyai_pretokenizer_matches_regex_property_fuzz(prompt in fuzz_prompt_strategy()) {
            let actual = joyai_pretokenize(&prompt)
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>();
            prop_assert_eq!(actual, joyai_reference_tokens(&prompt));
        }
    }

    #[test]
    #[ignore = "exhaustive cross-model oracle sweep"]
    fn native_matches_llama_cpp_across_model_matrix() {
        let fixtures = fixtures();
        if fixtures.is_empty() {
            return;
        }
        let prompts = [
            "Hello, world!",
            "<|im_start|>user\nhi<|im_end|>",
            "combining: e\u{301} cafe\u{301}",
            "family emoji: 👨‍👩‍👧‍👦 and scientist 👩🏽‍🔬",
            "nbsp:\u{00a0}thin:\u{2009}em:\u{2003}zwsp:\u{200b}",
            "<|fim_prefix|>code<|fim_middle|>body<|fim_suffix|>",
        ];
        for path in fixtures {
            let ffi = LlamaCppTokenizer::open(path).expect("open ffi tokenizer");
            let native = NativeTokenizer::open(path).expect("open native tokenizer");
            for prompt in prompts {
                for add_special in [false, true] {
                    assert_tokenizers_match(&ffi, &native, path, prompt, add_special);
                }
            }
        }
    }

    #[test]
    fn native_rejects_invalid_decode_token() {
        let Some(path) = fixture() else { return };
        let native = match NativeTokenizer::open(path) {
            Ok(native) => native,
            Err(TokError::UnsupportedNativeTokenizer { .. }) => return,
            Err(err) => panic!("open native tokenizer: {err}"),
        };
        assert!(matches!(
            native.try_decode_piece(-1),
            Err(TokError::InvalidToken(-1))
        ));
        assert!(matches!(
            native.try_decode(&[native.n_vocab() as i32]),
            Err(TokError::InvalidToken(_))
        ));
        assert!(matches!(
            native.try_decode_piece_bytes_exact(-1),
            Err(TokError::InvalidToken(-1))
        ));
        assert!(matches!(
            native.is_ordinary_content_token(native.n_vocab() as i32),
            Err(TokError::InvalidToken(_))
        ));
    }

    #[test]
    fn native_exact_piece_bytes_preserve_utf8_boundaries() {
        let Some(path) = fixture() else { return };
        let native = NativeTokenizer::open(path).expect("open native tokenizer");
        let text = "JSON: {\"emoji\":\"🙂\",\"city\":\"上海\"}";
        let ids = native
            .encode(text, false)
            .expect("encode exact-byte fixture");
        let mut decoded = Vec::new();
        for id in ids {
            assert!(
                native
                    .is_ordinary_content_token(id)
                    .expect("classify content token")
            );
            decoded.extend_from_slice(
                native
                    .try_decode_piece_bytes_exact(id)
                    .expect("decode exact piece bytes"),
            );
        }
        assert_eq!(decoded, text.as_bytes());
    }

    #[test]
    fn native_exact_piece_bytes_do_not_replace_invalid_utf8() {
        let Some(path) = fixture() else { return };
        let native = NativeTokenizer::open(path).expect("open native tokenizer");
        let token = native.byte_token_ids[0xff];
        assert_eq!(
            native
                .try_decode_piece_bytes_exact(token)
                .expect("decode exact byte token"),
            &[0xff]
        );
        assert_eq!(
            native
                .try_decode_piece(token)
                .expect("decode lossy byte token"),
            "\u{fffd}"
        );
    }

    #[test]
    fn native_content_policy_excludes_chat_control_tokens() {
        let Some(path) = fixture() else { return };
        let native = NativeTokenizer::open(path).expect("open native tokenizer");
        let control = native
            .encode("<|im_start|>", false)
            .expect("encode control token");
        assert_eq!(control.len(), 1);
        assert!(
            !native
                .is_ordinary_content_token(control[0])
                .expect("classify control token")
        );

        let content = native.encode("json", false).expect("encode content token");
        assert!(!content.is_empty());
        assert!(content.into_iter().all(|id| {
            native
                .is_ordinary_content_token(id)
                .expect("classify ordinary token")
        }));
    }

    #[test]
    fn native_decode_piece_matches_llama_cpp_for_full_vocab() {
        let Some(path) = fixture() else { return };
        let ffi = LlamaCppTokenizer::open(path).expect("open ffi tokenizer");
        let native = match NativeTokenizer::open(path) {
            Ok(native) => native,
            Err(TokError::UnsupportedNativeTokenizer { .. }) => return,
            Err(err) => panic!("open native tokenizer: {err}"),
        };
        assert_eq!(native.n_vocab(), ffi.n_vocab());
        for id in 0..ffi.n_vocab() as i32 {
            assert_eq!(
                native.try_decode_piece(id).expect("native piece"),
                ffi.try_decode_piece(id).expect("ffi piece"),
                "decode_piece mismatch at token {id}"
            );
        }
    }

    #[test]
    fn native_random_token_sequences_match_llama_cpp() {
        let Some(path) = fixture() else { return };
        let ffi = LlamaCppTokenizer::open(path).expect("open ffi tokenizer");
        let native = NativeTokenizer::open(path).expect("open native tokenizer");
        let mut rng = Lcg::new(0x5eed_fade_dead_beef);
        for len in [0usize, 1, 2, 3, 4, 7, 16, 31, 64] {
            for _ in 0..32 {
                let tokens: Vec<i32> = (0..len)
                    .map(|_| rng.gen_range(native.n_vocab()) as i32)
                    .collect();
                assert_eq!(
                    native.try_decode(&tokens).expect("native decode"),
                    ffi.try_decode(&tokens).expect("ffi decode"),
                    "random detokenize mismatch len={len} tokens={tokens:?}"
                );
            }
        }
    }

    #[test]
    fn native_lazy_decode_cache_is_thread_safe() {
        let Some(pair) = oracle_pair() else { return };
        let native = &pair.native;
        let ids = [0, native.n_vocab() as i32 / 2, native.n_vocab() as i32 - 1];

        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(move || {
                    for id in ids {
                        native.try_decode_piece(id).expect("concurrent decode");
                    }
                });
            }
        });
    }

    #[test]
    fn native_special_token_literals_match_llama_cpp() {
        let Some(path) = fixture() else { return };
        let ffi = LlamaCppTokenizer::open(path).expect("open ffi tokenizer");
        let native = NativeTokenizer::open(path).expect("open native tokenizer");
        let specials = special_token_texts(&native);
        for text in &specials {
            assert_tokenizers_match(&ffi, &native, path, text, false);
            assert_tokenizers_match(&ffi, &native, path, text, true);
        }
        for pair in specials.windows(2).take(32) {
            let joined = format!("{}{}", pair[0], pair[1]);
            assert_tokenizers_match(&ffi, &native, path, &joined, false);
        }
    }

    #[test]
    #[ignore = "exhaustive cross-model special-token oracle sweep"]
    fn native_special_token_literals_match_llama_cpp_across_model_matrix() {
        for path in fixtures() {
            let ffi = LlamaCppTokenizer::open(path).expect("open ffi tokenizer");
            let native = NativeTokenizer::open(path).expect("open native tokenizer");
            let specials = special_token_texts(&native);
            for text in &specials {
                assert_tokenizers_match(&ffi, &native, path, text, false);
                assert_tokenizers_match(&ffi, &native, path, text, true);
            }
        }
    }

    #[test]
    fn byte_unicode_mapping_round_trips_all_bytes() {
        for byte in 0u8..=255 {
            let ch = byte_to_unicode(byte);
            assert_eq!(unicode_to_byte(ch), Some(byte), "byte {byte}");
        }
    }

    #[test]
    fn qwen35_pretokenizer_keeps_combining_marks_with_letters() {
        let parts = qwen35_pretokenize("e\u{301} cafe\u{301}!");
        assert_eq!(parts, vec!["e\u{301}", " cafe\u{301}", "!"]);
    }

    #[test]
    fn joyai_pretokenizer_matches_sequential_regex_reference() {
        let cases = [
            "",
            "Hello, world!",
            "digits 1 12 123 1234 １２３４ ①Ⅻ",
            "中文かなカナ mixed 123 punctuation!!!\r\nnext",
            "e\u{301} cafe\u{301} \u{309b}\u{309c}\u{30a0}\u{30fb}",
            "spaces   before and trailing   ",
            "\t\r\n\n  next",
            "emoji 👨‍👩‍👧‍👦 symbols ❤\u{fe0f}",
            "unicode16: a\u{105c0}b digits \u{11bf0}\u{11bf1}\u{11bf2}\u{11bf3}",
            "boundaries: \u{303f}\u{3040}\u{309f}\u{30a0}\u{30ff}\u{3100}\u{4dff}\u{4e00}\u{9fa5}\u{9fa6}",
            "control:\u{0000}abc\u{200b}def",
        ];
        for case in cases {
            let actual = joyai_pretokenize(case)
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>();
            assert_eq!(actual, joyai_reference_tokens(case), "case={case:?}");
        }
    }

    #[test]
    fn tokenizer_unicode_policy_is_pinned_to_unicode_16() {
        assert_eq!(unicode_general_category::UNICODE_VERSION, (16, 0, 0));
        assert!(CharFlags::for_char('\u{105c0}').is_letter);
        assert!(CharFlags::for_char('\u{11bf0}').is_number);
        assert_eq!(joyai_pretokenize("a\u{105c0}b"), ["a\u{105c0}b"]);
    }

    #[test]
    #[ignore = "requires the local 95.93 GiB DeepSeek V4 Flash-0731 IQ3 fixture"]
    fn deepseek_v4_0731_native_tokenizer_smoke() {
        assert!(Path::new(DS4_0731_IQ3).exists(), "missing DS4 fixture");
        let tokenizer = NativeTokenizer::open(DS4_0731_IQ3).expect("open DS4 tokenizer");
        assert_eq!(tokenizer.pretokenizer, PretokenizerKind::JoyAi);
        assert_eq!(tokenizer.n_vocab(), 129_280);
        assert_eq!(tokenizer.bos(), Some(0));
        assert_eq!(tokenizer.eos(), Some(1));
        let hello = tokenizer
            .encode("Hello, world!", false)
            .expect("encode DS4 greeting");
        assert_eq!(hello, [19_923, 14, 2_058, 3]);
        assert_eq!(
            tokenizer
                .encode("Hello, world!", true)
                .expect("encode DS4 greeting with specials"),
            hello
        );
        assert_eq!(tokenizer.decode(&hello), "Hello, world!");
        assert_eq!(
            tokenizer.encode("<｜User｜>", false).expect("encode role"),
            [128_803]
        );
        assert_eq!(
            tokenizer
                .encode(&"A\n\t@".repeat(32), false)
                .expect("encode first HCA boundary prefix"),
            [35, 201, 200, 34].repeat(32)
        );
        let mut position_129 = [35, 201, 200, 34].repeat(32);
        position_129.push(35);
        assert_eq!(
            tokenizer
                .encode(&("A\n\t@".repeat(32) + "A"), false)
                .expect("encode position-129 prefix"),
            position_129
        );
        let mut position_254 = [35, 201, 200, 34].repeat(63);
        position_254.extend([35, 201]);
        assert_eq!(
            tokenizer
                .encode(&("A\n\t@".repeat(63) + "A\n"), false)
                .expect("encode position-254 prefix"),
            position_254
        );
        let mut position_255 = [35, 201, 200, 34].repeat(63);
        position_255.extend([35, 201, 200]);
        assert_eq!(
            tokenizer
                .encode(&("A\n\t@".repeat(63) + "A\n\t"), false)
                .expect("encode position-255 CLI prefix"),
            position_255
        );
        assert_eq!(
            tokenizer
                .encode(&"A\n\t@".repeat(64), false)
                .expect("encode second HCA boundary prefix"),
            [35, 201, 200, 34].repeat(64)
        );
        let mut position_257 = [35, 201, 200, 34].repeat(64);
        position_257.push(35);
        assert_eq!(
            tokenizer
                .encode(&("A\n\t@".repeat(64) + "A"), false)
                .expect("encode position-257 CLI prefix"),
            position_257
        );
        let mut position_382 = [35, 201, 200, 34].repeat(95);
        position_382.extend([35, 201]);
        assert_eq!(
            tokenizer
                .encode(&("A\n\t@".repeat(95) + "A\n"), false)
                .expect("encode position-382 prefix"),
            position_382
        );
        let mut position_383 = [35, 201, 200, 34].repeat(95);
        position_383.extend([35, 201, 200]);
        assert_eq!(
            tokenizer
                .encode(&("A\n\t@".repeat(95) + "A\n\t"), false)
                .expect("encode position-383 prefix"),
            position_383
        );
        assert_eq!(
            tokenizer
                .encode(&"A\n\t@".repeat(96), false)
                .expect("encode third HCA boundary prefix"),
            [35, 201, 200, 34].repeat(96)
        );
        let mut position_385 = [35, 201, 200, 34].repeat(96);
        position_385.push(35);
        assert_eq!(
            tokenizer
                .encode(&("A\n\t@".repeat(96) + "A"), false)
                .expect("encode position-385 CLI prefix"),
            position_385
        );
        let mut position_510 = [35, 201, 200, 34].repeat(127);
        position_510.extend([35, 201]);
        assert_eq!(
            tokenizer
                .encode(&("A\n\t@".repeat(127) + "A\n"), false)
                .expect("encode position-510 prefix"),
            position_510
        );
        let mut position_511 = [35, 201, 200, 34].repeat(127);
        position_511.extend([35, 201, 200]);
        assert_eq!(
            tokenizer
                .encode(&("A\n\t@".repeat(127) + "A\n\t"), false)
                .expect("encode position-511 prefix"),
            position_511
        );
        assert_eq!(
            tokenizer
                .encode(&"A\n\t@".repeat(128), false)
                .expect("encode fourth HCA boundary prefix"),
            [35, 201, 200, 34].repeat(128)
        );
        let mut position_513 = [35, 201, 200, 34].repeat(128);
        position_513.push(35);
        assert_eq!(
            tokenizer
                .encode(&("A\n\t@".repeat(128) + "A"), false)
                .expect("encode position-513 CLI prefix"),
            position_513
        );
    }

    #[test]
    #[ignore = "requires local DeepSeek V4 fixture and current llama.cpp tokenizer binary"]
    fn deepseek_v4_native_matches_current_llama_cpp_cli() {
        const LLAMA_TOKENIZE: &str = "/Users/tito/code/llama.cpp/build/bin/llama-tokenize";
        assert!(Path::new(DS4_0731_IQ3).exists(), "missing DS4 fixture");
        assert!(
            Path::new(LLAMA_TOKENIZE).exists(),
            "missing llama-tokenize oracle"
        );
        let tokenizer = NativeTokenizer::open(DS4_0731_IQ3).expect("open DS4 tokenizer");
        let prompts = [
            "Hello, world!",
            "digits 1 12 123 1234 １２３４ ①Ⅻ",
            "中文かなカナ mixed 123 punctuation!!!\r\nnext",
            "e\u{301} cafe\u{301} 👨‍👩‍👧‍👦",
            "spaces   before and trailing   ",
            "a\u{105c0}b \u{11bf0}\u{11bf1}\u{11bf2}\u{11bf3}",
            "<｜User｜>hello<｜Assistant｜>",
        ];
        for prompt in prompts {
            let output = std::process::Command::new(LLAMA_TOKENIZE)
                .args([
                    "-m",
                    DS4_0731_IQ3,
                    "--ids",
                    "--no-bos",
                    "--no-escape",
                    "--log-disable",
                    "-p",
                    prompt,
                ])
                .output()
                .expect("run llama-tokenize");
            assert!(
                output.status.success(),
                "llama-tokenize failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let oracle: Vec<i32> =
                serde_json::from_slice(&output.stdout).expect("parse llama token ids");
            let native = tokenizer.encode(prompt, false).expect("native encode");
            assert_eq!(native, oracle, "prompt={prompt:?}");
        }
    }

    #[test]
    fn special_matcher_prefers_longest_same_start() {
        let matcher = SpecialMatcher::new(&[
            SpecialToken {
                text: "<|im|>".into(),
                id: 1,
            },
            SpecialToken {
                text: "<|im_start|>".into(),
                id: 2,
            },
            SpecialToken {
                text: "<|im_end|>".into(),
                id: 3,
            },
        ]);
        let parts = matcher.partition("x<|im_start|>y<|im_end|>z");
        assert!(matches!(parts[0], Fragment::Text("x")));
        assert!(matches!(parts[1], Fragment::Token(2)));
        assert!(matches!(parts[2], Fragment::Text("y")));
        assert!(matches!(parts[3], Fragment::Token(3)));
        assert!(matches!(parts[4], Fragment::Text("z")));
    }
}
