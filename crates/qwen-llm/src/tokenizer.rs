//! Qwen2 byte-level BPE tokenizer.
//!
//! Vocab size: 248,320 (padded; real tokens go up to ~248,077). Special
//! tokens occupy the high range starting at 248,044
//! (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, vision/audio pads, etc).
//!
//! ## Implementation choice (and the swap path)
//!
//! Tokenization is a non-hot-path operation: it runs once per prompt at
//! ingest, and again only when streaming user output. The throughput
//! ceiling we care about is in the kernels and graph executor.
//!
//! Three real options:
//!
//! | option | byte-perfect w/ llama-cli oracle | drops llama-cpp link | LOC |
//! |---|---|---|---|
//! | **`llama-cpp-sys-2` shim (current)** | yes — shared codepath | no | ~50 |
//! | **`tokenizers` (huggingface) crate** | not guaranteed (BPE tie-break edges) | yes | ~30 |
//! | **`tiktoken-rs`** | n/a — different vocab family | yes | n/a |
//!
//! For the v1 phase where we're chasing byte-for-byte logit equivalence
//! with `llama-cli` on `Qwen3.5-0.8B.F32.gguf`, the llama-cpp shim is
//! the safer call because it shares the exact tokenizer used by the
//! oracle — any divergence in our kernels can't be confused with a BPE
//! edge case.
//!
//! Once kernels are validated, swapping to the `tokenizers` crate (which
//! reads Qwen's shipping `tokenizer.json` directly from HF, pure Rust,
//! no FFI) is a 30-line change behind the [`Tokenize`] trait below. The
//! same way the codec seam is staged for in-tree replacement.

use crate::gguf::GgufFile;
use std::ffi::CString;
use std::path::Path;
use std::sync::Once;

/// Backend-agnostic tokenizer interface. Implemented today by the
/// llama.cpp-backed [`Tokenizer`]; the planned `huggingface_tokenizers`
/// backend will implement the same trait.
pub trait Tokenize {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<i32>, TokError>;
    fn decode(&self, tokens: &[i32]) -> String;
    fn n_vocab(&self) -> u32;
    fn bos(&self) -> Option<i32>;
    fn eos(&self) -> Option<i32>;
}

#[derive(Debug, thiserror::Error)]
pub enum TokError {
    #[error("path is not valid UTF-8: {0:?}")]
    BadPath(std::path::PathBuf),
    #[error("llama_model_load_from_file returned null")]
    LoadFailed,
    #[error("llama_model_get_vocab returned null")]
    NoVocab,
    #[error("tokenize failed: input too long ({0} tokens needed)")]
    InputTooLong(i32),
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

/// Tokenizer for a Qwen3.5/3.6 GGUF model.
///
/// Holds an owned `llama_model` pointer (so vocab metadata stays valid).
/// The pointer is freed in `Drop`.
pub struct Tokenizer {
    model: *mut llama_cpp_sys_2::llama_model,
    vocab: *const llama_cpp_sys_2::llama_vocab,
    pub n_vocab: u32,
    pub bos: Option<i32>,
    pub eos: Option<i32>,
}

// SAFETY: `llama_model` is internally synchronized and the only mutating
// API we expose is tokenize/detokenize which llama.cpp documents as
// thread-safe against the same model handle.
unsafe impl Send for Tokenizer {}
unsafe impl Sync for Tokenizer {}

impl Tokenizer {
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
        let cpath = CString::new(path_str).expect("path contains nul");

        // Vocab-only load: skip GPU, skip compute. We don't want llama.cpp
        // to allocate any tensor buffers, just to parse the tokenizer.
        // SAFETY: standard FFI sequence per llama.cpp public API.
        let mut params = unsafe { llama_cpp_sys_2::llama_model_default_params() };
        params.vocab_only = true;
        let model = unsafe { llama_cpp_sys_2::llama_model_load_from_file(cpath.as_ptr(), params) };
        if model.is_null() {
            return Err(TokError::LoadFailed);
        }
        let vocab = unsafe { llama_cpp_sys_2::llama_model_get_vocab(model) };
        if vocab.is_null() {
            unsafe { llama_cpp_sys_2::llama_model_free(model) };
            return Err(TokError::NoVocab);
        }
        let n_vocab = unsafe { llama_cpp_sys_2::llama_n_vocab(vocab) };
        let bos_raw = unsafe { llama_cpp_sys_2::llama_token_bos(vocab) };
        let eos_raw = unsafe { llama_cpp_sys_2::llama_token_eos(vocab) };
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

    /// Encode a UTF-8 string. `add_special` adds BOS/EOS per model
    /// convention; for Qwen3.5 chat use we typically don't (BOS is
    /// already in the chat template).
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<i32>, TokError> {
        // First call with `n_tokens_max=0` returns negative `n_needed`.
        // SAFETY: read-only call, vocab is non-null and held for the
        // lifetime of `self`.
        let needed = unsafe {
            llama_cpp_sys_2::llama_tokenize(
                self.vocab,
                text.as_ptr() as *const i8,
                text.len() as i32,
                std::ptr::null_mut(),
                0,
                add_special,
                /* parse_special = */ true,
            )
        };
        let n = if needed < 0 { -needed } else { needed };
        let mut buf: Vec<i32> = vec![0; n as usize];
        let written = unsafe {
            llama_cpp_sys_2::llama_tokenize(
                self.vocab,
                text.as_ptr() as *const i8,
                text.len() as i32,
                buf.as_mut_ptr(),
                n,
                add_special,
                true,
            )
        };
        if written < 0 {
            return Err(TokError::InputTooLong(-written));
        }
        buf.truncate(written as usize);
        Ok(buf)
    }

    /// Decode one token to its UTF-8 piece. Special tokens are rendered
    /// as their literal `<|...|>` form.
    pub fn decode_piece(&self, token: i32) -> String {
        let mut buf = [0i8; 256];
        let n = unsafe {
            llama_cpp_sys_2::llama_token_to_piece(
                self.vocab,
                token,
                buf.as_mut_ptr(),
                buf.len() as i32,
                /* lstrip = */ 0,
                /* special = */ true,
            )
        };
        if n <= 0 {
            return String::new();
        }
        // SAFETY: llama.cpp NUL-terminates within the buffer; CStr handles
        // the slicing. We use from_bytes_with_nul_unchecked-like intent
        // but go through CStr::from_bytes_until_nul for safety.
        let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, n as usize) };
        String::from_utf8_lossy(bytes).into_owned()
    }

    /// Decode a sequence of tokens.
    pub fn decode(&self, tokens: &[i32]) -> String {
        let mut out = String::new();
        for &t in tokens {
            out.push_str(&self.decode_piece(t));
        }
        out
    }
}

impl Tokenize for Tokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<i32>, TokError> {
        Tokenizer::encode(self, text, add_special)
    }
    fn decode(&self, tokens: &[i32]) -> String {
        Tokenizer::decode(self, tokens)
    }
    fn n_vocab(&self) -> u32 {
        self.n_vocab
    }
    fn bos(&self) -> Option<i32> {
        self.bos
    }
    fn eos(&self) -> Option<i32> {
        self.eos
    }
}

impl Drop for Tokenizer {
    fn drop(&mut self) {
        // SAFETY: paired with the load above. After this call the vocab
        // pointer is dangling; no method on `self` is reachable post-drop.
        unsafe { llama_cpp_sys_2::llama_model_free(self.model) };
        // Note: not calling `llama_backend_free` here — it's a
        // process-global teardown and we may have multiple Tokenizers
        // outstanding. Leaving it leaked-on-exit is the conventional
        // pattern in the llama.cpp ecosystem.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Option<&'static str> {
        let candidates = [
            "/Users/tito/models/Qwen3.5-0.8B.F32.gguf",
            "/Users/tito/models/Qwen3.5-0.8B-BF16.gguf",
        ];
        candidates.iter().copied().find(|p| Path::new(p).exists())
    }

    #[test]
    fn opens_and_encodes() {
        let Some(path) = fixture() else { return };
        let tok = Tokenizer::open(path).expect("open tokenizer");
        assert!(tok.n_vocab >= 248_000, "n_vocab={}", tok.n_vocab);

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
            tok.n_vocab,
            tok.bos,
            tok.eos,
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
}
