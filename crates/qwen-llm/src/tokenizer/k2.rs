use super::{
    CharFlags, CharInfo, NATIVE_MAX_INPUT_BYTES, TokError, char_at, char_flags, push_token,
};
use std::borrow::Cow;
use unicode_normalization::UnicodeNormalization;

pub(super) fn validate_pair_config(
    add_sep: Option<bool>,
    sep: Option<i32>,
) -> Result<(), TokError> {
    if sep.is_some_and(|id| id != 1) || (add_sep == Some(true) && sep != Some(1)) {
        return Err(TokError::BadMetadata(
            "K2 paired separator must be token 1; native encode supports only single sequences"
                .into(),
        ));
    }
    Ok(())
}

pub(super) fn normalize(text: &str) -> Result<Cow<'_, str>, TokError> {
    if text.is_ascii() {
        return Ok(Cow::Borrowed(text));
    }
    let mut output = String::with_capacity(text.len());
    for ch in text.nfc() {
        let bytes = output.len() + ch.len_utf8();
        if bytes > NATIVE_MAX_INPUT_BYTES {
            return Err(TokError::NativeInputTooLong {
                bytes,
                max_bytes: NATIVE_MAX_INPUT_BYTES,
            });
        }
        output.push(ch);
    }
    Ok(Cow::Owned(output))
}

fn letter(chars: &[CharInfo], pos: usize) -> bool {
    let flags = char_flags(chars, pos);
    flags.is_letter
        || flags.is_accent_mark
        || matches!(char_at(chars, pos), Some('\u{200c}' | '\u{200d}'))
}

/// The pinned HF Split regex, before byte encoding. Unlike Qwen35, numbers
/// group up to three, letter runs include joiners, and punctuation may eat marks.
pub(super) fn pretokenize(text: &str) -> Vec<&str> {
    let chars: Vec<_> = text
        .char_indices()
        .map(|(start, ch)| CharInfo {
            ch,
            start,
            flags: CharFlags::for_char(ch),
        })
        .collect();
    let mut out = Vec::new();
    let mut prev = 0;
    let mut pos = 0;
    while pos < chars.len() {
        let ch = chars[pos].ch;
        let flags = chars[pos].flags;
        if ch == '\'' && pos + 1 < chars.len() {
            let next = match chars[pos + 1].ch {
                '\u{17f}' => 's',
                c => c.to_ascii_lowercase(),
            };
            let end = if matches!(next, 's' | 't' | 'm' | 'd') {
                Some(pos + 2)
            } else if pos + 2 < chars.len()
                && matches!(
                    (next, chars[pos + 2].ch.to_ascii_lowercase()),
                    ('r', 'e') | ('v', 'e') | ('l', 'l')
                )
            {
                Some(pos + 3)
            } else {
                None
            };
            if let Some(end) = end {
                push_token(text, &chars, &mut out, &mut prev, end);
                pos = end;
                continue;
            }
        }
        if ch != '\r'
            && ch != '\n'
            && !flags.is_number
            && (letter(&chars, pos) || letter(&chars, pos + 1))
        {
            pos += 1;
            while letter(&chars, pos) {
                pos += 1;
            }
        } else if flags.is_number {
            let start = pos;
            while pos - start < 3 && char_flags(&chars, pos).is_number {
                pos += 1;
            }
        } else {
            let punct_start = pos + usize::from(ch == ' ');
            let punct = |i| {
                let f = char_flags(&chars, i);
                f.any && !f.is_whitespace && !f.is_letter && !f.is_number
            };
            if punct(punct_start) {
                pos = punct_start + 1;
                while punct(pos) {
                    pos += 1;
                }
                while matches!(char_at(&chars, pos), Some('\r' | '\n')) {
                    pos += 1;
                }
            } else {
                let start = pos;
                let mut end = pos;
                let mut last_newline = None;
                while char_flags(&chars, end).is_whitespace {
                    end += 1;
                    if matches!(char_at(&chars, end - 1), Some('\r' | '\n')) {
                        last_newline = Some(end);
                    }
                }
                pos = if let Some(newline) = last_newline {
                    newline
                } else if end < chars.len() && end - start > 1 {
                    end - 1
                } else if end > start {
                    end
                } else {
                    pos + 1
                };
            }
        }
        push_token(text, &chars, &mut out, &mut prev, pos);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{NativeTokenizer, PretokenizerKind, validate_special_addition_config};
    use proptest::prelude::*;
    use serde_json::Value;

    fn fixtures() -> Value {
        serde_json::from_str(include_str!("../../tests/fixtures/k2_tokenizer_hf.json")).unwrap()
    }

    #[test]
    fn normalization_and_splits_match_pinned_hf() {
        for profile in fixtures()["profiles"].as_array().unwrap() {
            for case in profile["cases"].as_array().unwrap() {
                let text = case["text"].as_str().unwrap();
                let normalized = normalize(text).unwrap();
                assert_eq!(normalized, case["normalized"].as_str().unwrap(), "{text:?}");
                let expected: Vec<_> = case["pieces"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap())
                    .collect();
                assert_eq!(pretokenize(&normalized), expected, "{text:?}");
            }
        }
    }

    #[test]
    fn single_sequence_does_not_append_pair_separator() {
        assert!(validate_pair_config(Some(true), Some(1)).is_ok());
        assert!(validate_pair_config(None, None).is_ok());
        assert!(validate_pair_config(Some(true), None).is_err());
        assert!(validate_pair_config(Some(true), Some(0)).is_err());
        let kind = PretokenizerKind::K2Horizon;
        assert!(validate_special_addition_config(kind, Some(0), Some(1), true, false).is_ok());
        for (bos, eos, add_bos, add_eos) in [
            (None, Some(1), true, false),
            (Some(0), None, true, false),
            (Some(1), Some(0), true, false),
            (Some(0), Some(1), false, false),
            (Some(0), Some(1), true, true),
        ] {
            assert!(validate_special_addition_config(kind, bos, eos, add_bos, add_eos).is_err());
        }
    }

    #[test]
    fn added_tokens_are_matched_before_nfc_without_rematching() {
        use crate::tokenizer::{
            NativeToken, SpecialMatcher, SpecialToken, TokenAttr, byte_to_unicode,
        };
        let mut tokens: Vec<_> = (0..=255u8)
            .map(|byte| NativeToken {
                text: byte_to_unicode(byte).to_string(),
                attr: TokenAttr::Normal,
            })
            .collect();
        tokens.push(NativeToken {
            text: "\u{e9}".into(),
            attr: TokenAttr::UserDefined,
        });
        let tokenizer = NativeTokenizer {
            id_to_token: tokens,
            pair_merges: Default::default(),
            byte_token_ids: std::array::from_fn(|i| i as i32),
            special_matcher: SpecialMatcher::new(&[SpecialToken {
                text: "\u{e9}".into(),
                id: 256,
            }]),
            decoded_piece_bytes: (0..257).map(|_| std::sync::OnceLock::new()).collect(),
            bos: None,
            eos: None,
            add_bos: false,
            add_eos: false,
            pretokenizer: PretokenizerKind::K2Horizon,
        };
        assert_eq!(tokenizer.encode("\u{e9}", false).unwrap(), [256]);
        assert_eq!(tokenizer.encode("e\u{301}", false).unwrap(), [0xc3, 0xa9]);
        assert_eq!(
            tokenizer.encode("e\u{301}\u{e9}", false).unwrap(),
            [0xc3, 0xa9, 256]
        );
    }

    #[test]
    fn normalization_expansion_is_bounded() {
        let text = "\u{344}".repeat(NATIVE_MAX_INPUT_BYTES / 4 + 1);
        assert!(text.len() < NATIVE_MAX_INPUT_BYTES);
        assert!(matches!(
            normalize(&text),
            Err(TokError::NativeInputTooLong { .. })
        ));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]
        #[test]
        fn splitter_matches_regex_for_unicode_input(chars in prop::collection::vec(any::<char>(), 0..256)) {
            static REGEX: std::sync::OnceLock<fancy_regex::Regex> = std::sync::OnceLock::new();
            let regex = REGEX.get_or_init(|| {
                let data = fixtures();
                let pattern = data["profiles"][0]["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"].as_str().unwrap();
                fancy_regex::Regex::new(pattern).unwrap()
            });
            let text: String = chars.into_iter().collect();
            let expected: Vec<_> = regex.find_iter(&text).map(|m| m.unwrap().as_str()).collect();
            prop_assert_eq!(pretokenize(&text), expected);
        }
    }

    fn compare(tokenizer: &NativeTokenizer, profile: &Value) {
        for case in profile["cases"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            for (add, field) in [(false, "ids"), (true, "ids_with_special")] {
                let expected: Vec<i32> = serde_json::from_value(case[field].clone()).unwrap();
                assert_eq!(
                    tokenizer.encode(text, add).unwrap(),
                    expected,
                    "{} {text:?} add={add}",
                    profile["profile"]
                );
            }
            let ids: Vec<i32> = serde_json::from_value(case["ids"].clone()).unwrap();
            assert_eq!(
                tokenizer.try_decode(&ids).unwrap(),
                case["decoded"].as_str().unwrap(),
                "{text:?}"
            );
        }
    }

    #[test]
    #[ignore = "CPU only; generate tokenizer-only containers with generate_k2_tokenizer_fixtures.py"]
    fn both_stage_vocabularies_match_hf_ids() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/profiles/k2-tokenizer");
        for profile in fixtures()["profiles"].as_array().unwrap() {
            let path = root
                .join(profile["revision"].as_str().unwrap())
                .join("vocab-only.gguf");
            let tokenizer = NativeTokenizer::open(path).unwrap();
            compare(&tokenizer, profile);
        }
    }

    #[test]
    #[ignore = "CPU/header-only; requires K2_GGUF, no inference or GPU"]
    fn downloaded_q8_tokenizer_matches_hf_ids() {
        let path = std::env::var("K2_GGUF").expect("set K2_GGUF explicitly");
        let tokenizer = NativeTokenizer::open(path).unwrap();
        let fixtures = fixtures();
        let profile = fixtures["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["profile"] == "posttrained")
            .unwrap();
        compare(&tokenizer, profile);
    }
}
