//! Incremental UTF-8 assembly across token boundaries.
//!
//! Byte-level BPE routinely splits one multibyte character across two
//! tokens, so decoding each token independently (`decode_piece`, which is
//! `from_utf8_lossy` per token) yields U+FFFD for CJK, emoji, and accented
//! text. That corrupts responses *and* breaks checkpoint reuse: clients
//! echo the corrupted text back, so the re-encoded history no longer
//! matches the tokens the completed-turn snapshot was keyed on.
//!
//! This assembler buffers raw token bytes and releases only complete UTF-8,
//! holding an incomplete tail until its continuation arrives.

#[derive(Debug, Default)]
pub(crate) struct Utf8Assembler {
    pending: Vec<u8>,
}

impl Utf8Assembler {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Append raw token bytes; returns the text now decodable in full.
    pub(crate) fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    out.push_str(text);
                    self.pending.clear();
                    return out;
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    if valid > 0 {
                        out.push_str(
                            std::str::from_utf8(&self.pending[..valid]).expect("validated prefix"),
                        );
                    }
                    match error.error_len() {
                        // Truncated sequence: keep the tail for the next token.
                        None => {
                            self.pending.drain(..valid);
                            return out;
                        }
                        // Genuinely invalid bytes: emit the replacement the
                        // lossy path would have produced and continue.
                        Some(invalid) => {
                            out.push('\u{fffd}');
                            self.pending.drain(..valid + invalid);
                        }
                    }
                }
            }
        }
    }

    /// Flush at end of generation; a dangling partial sequence becomes one
    /// replacement character (it can never complete).
    pub(crate) fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        self.pending.clear();
        "\u{fffd}".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reassembles_multibyte_split_across_tokens() {
        // "数学" split mid-character, as byte-level BPE does.
        let full = "数学: 2+3=5 → ✓";
        let bytes = full.as_bytes();
        for split in 1..bytes.len() {
            let mut assembler = Utf8Assembler::new();
            let mut out = assembler.push(&bytes[..split]);
            out.push_str(&assembler.push(&bytes[split..]));
            out.push_str(&assembler.finish());
            assert_eq!(out, full, "split at {split} lost bytes");
        }
    }

    #[test]
    fn byte_at_a_time_matches_the_original() {
        for full in ["héllo wörld", "🎉 emoji 🚀 test", "普通の日本語テキスト"] {
            let mut assembler = Utf8Assembler::new();
            let mut out = String::new();
            for byte in full.as_bytes() {
                out.push_str(&assembler.push(&[*byte]));
            }
            out.push_str(&assembler.finish());
            assert_eq!(out, full);
        }
    }

    #[test]
    fn lossy_per_token_decode_is_the_bug_this_prevents() {
        let bytes = "→".as_bytes(); // 3 bytes
        let lossy: String = bytes
            .chunks(1)
            .map(|chunk| String::from_utf8_lossy(chunk).into_owned())
            .collect();
        assert!(lossy.contains('\u{fffd}'), "per-token decode corrupts");
        let mut assembler = Utf8Assembler::new();
        let mut assembled = String::new();
        for chunk in bytes.chunks(1) {
            assembled.push_str(&assembler.push(chunk));
        }
        assert_eq!(assembled, "→");
    }

    #[test]
    fn invalid_bytes_degrade_without_stalling() {
        let mut assembler = Utf8Assembler::new();
        let out = assembler.push(&[0xff, b'o', b'k']);
        assert_eq!(out, "\u{fffd}ok");
        let mut assembler = Utf8Assembler::new();
        assembler.push(&[0xe4, 0xbd]); // truncated 3-byte sequence
        assert_eq!(assembler.finish(), "\u{fffd}");
    }
}
