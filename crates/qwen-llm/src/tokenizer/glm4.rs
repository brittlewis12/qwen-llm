use super::{char_flags, k2};

/// GLM-4/5 `glm4` HF Split regex, before byte encoding: K2's llama3-shaped
/// splitter with the letter class exactly `\p{L}`. Combining marks and
/// joiners are not letters; they fall into `[^\s\p{L}\p{N}]`.
pub(super) fn pretokenize(text: &str) -> Vec<&str> {
    k2::split_llama3_shaped(text, |chars, pos| char_flags(chars, pos).is_letter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::tokenizer::{
        NativeToken, NativeTokenizer, PretokenizerKind, SpecialMatcher, TokenAttr, byte_to_unicode,
        validate_special_addition_config, whole_piece_ids,
    };
    use proptest::prelude::*;
    use serde_json::Value;
    use std::path::PathBuf;

    const BOS: i32 = 154_822;
    const EOS: i32 = 154_820;
    const EOT: i32 = 154_827;
    const EOM: i32 = 154_829;
    const ALTERNATE_GGUF: &str = "/Volumes/wdblack/weights-archive/glm-5.3-flash-ud-iq3_xxs/UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf";
    const LLAMA_TOKENIZE: &str = "/Users/tito/code/llama.cpp/build-glm5/bin/llama-tokenize";

    fn fixtures() -> Value {
        serde_json::from_str(include_str!("../../tests/fixtures/glm4_tokenizer_hf.json")).unwrap()
    }

    fn cases(data: &Value) -> &[Value] {
        data["cases"].as_array().unwrap()
    }

    fn ids(case: &Value, field: &str) -> Vec<i32> {
        serde_json::from_value(case[field].clone()).unwrap()
    }

    #[test]
    fn splitter_trap_cases() {
        for (text, expected) in [
            // Marks are not letters: they split runs and join punctuation.
            ("cafe\u{301} x", &["cafe", "\u{301}", " x"][..]),
            ("e\u{301}\u{301}!", &["e", "\u{301}\u{301}!"]),
            (
                "\u{915}\u{93f}\u{924}\u{93e}\u{92c}",
                &["\u{915}", "\u{93f}\u{924}", "\u{93e}\u{92c}"],
            ),
            // Joiners are not letters either, but may prefix one letter run.
            ("a\u{200d}b\u{200c}c", &["a", "\u{200d}b", "\u{200c}c"]),
            // HF folds U+017F into the case-insensitive contraction.
            ("'\u{17f}top it'S", &["'\u{17f}", "top", " it", "'S"]),
            ("I'M WE'VE", &["I", "'M", " WE", "'VE"]),
            (
                "1234567 \u{b2}\u{bd}",
                &["123", "456", "7", " ", "\u{b2}\u{bd}"],
            ),
            (
                "\u{661}\u{662}\u{663}\u{664}",
                &["\u{661}\u{662}\u{663}", "\u{664}"],
            ),
            ("a  b   ", &["a", " ", " b", "   "]),
            ("x \r\n\n\ty", &["x", " \r\n\n", "\ty"]),
            ("\u{a0}\u{a0}nbsp", &["\u{a0}", "\u{a0}nbsp"]),
            ("{\"a\"::\n\n}", &["{\"", "a", "\"::\n\n", "}"]),
            (
                "\u{4f60}\u{597d}\u{ff0c}\u{4e16}",
                &["\u{4f60}\u{597d}", "\u{ff0c}\u{4e16}"],
            ),
            (
                "\u{1f468}\u{200d}\u{1f469} ok",
                &["\u{1f468}\u{200d}\u{1f469}", " ok"],
            ),
        ] {
            assert_eq!(pretokenize(text), expected, "{text:?}");
        }
        // K2 keeps the same Devanagari word whole: the two kinds really differ.
        assert_eq!(
            k2::pretokenize("\u{915}\u{93f}\u{924}\u{93e}\u{92c}"),
            ["\u{915}\u{93f}\u{924}\u{93e}\u{92c}"]
        );
    }

    #[test]
    fn splits_match_pinned_hf() {
        for case in cases(&fixtures()) {
            let text = case["text"].as_str().unwrap();
            let expected: Vec<_> = case["pieces"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            assert_eq!(pretokenize(text), expected, "{text:?}");
        }
    }

    #[test]
    fn fixture_pins_policy_and_exercises_ignore_merges() {
        let data = fixtures();
        assert!(data["normalizer"].is_null());
        assert_eq!(data["model_options"]["ignore_merges"], true);
        assert_eq!(data["model_options"]["byte_fallback"], false);
        let exercised: Vec<_> = cases(&data)
            .iter()
            .filter(|case| case.get("ids_without_ignore_merges").is_some())
            .map(|case| case["text"].as_str().unwrap())
            .collect();
        for word in ["kohol", "wirkungen", "tiquetas", "ramientas"] {
            assert!(exercised.contains(&word), "{word}");
        }
        // The leading-space variant is not itself a vocabulary token.
        assert!(!exercised.contains(&" kohol"));
        // HF adds nothing for single sequences: no implicit [gMASK] BOS.
        for case in cases(&data) {
            assert_eq!(case["ids"], case["ids_with_special"]);
        }
    }

    #[test]
    fn rejects_implicit_bos_or_eos_insertion() {
        let kind = PretokenizerKind::Glm4;
        assert!(validate_special_addition_config(kind, Some(BOS), Some(EOS), false, false).is_ok());
        assert!(validate_special_addition_config(kind, None, None, false, false).is_ok());
        assert!(validate_special_addition_config(kind, Some(BOS), Some(EOS), true, false).is_err());
        assert!(validate_special_addition_config(kind, Some(BOS), Some(EOS), false, true).is_err());
    }

    fn byte_level(text: &str) -> String {
        text.bytes().map(byte_to_unicode).collect()
    }

    #[test]
    fn whole_piece_lookup_bypasses_bpe_for_normal_tokens_only() {
        let mut tokens: Vec<_> = (0..=255u8)
            .map(|byte| NativeToken {
                text: byte_to_unicode(byte).to_string(),
                attr: TokenAttr::Normal,
            })
            .collect();
        for (text, attr) in [
            (byte_level("abc"), TokenAttr::Normal),
            (byte_level(" \u{e9}"), TokenAttr::Normal),
            (byte_level("xyz"), TokenAttr::Control),
            ("not\u{2603}bytelevel".into(), TokenAttr::Normal),
        ] {
            tokens.push(NativeToken { text, attr });
        }
        let whole = whole_piece_ids(&tokens);
        assert_eq!(whole.len(), 258);
        assert_eq!(whole.get(&b"abc"[..]), Some(&256));
        assert_eq!(whole.get(" \u{e9}".as_bytes()), Some(&257));
        assert_eq!(whole.get(&b"xyz"[..]), None);

        let n = tokens.len();
        let mut tokenizer = NativeTokenizer {
            id_to_token: tokens,
            pair_merges: Default::default(),
            byte_token_ids: std::array::from_fn(|i| i as i32),
            special_matcher: SpecialMatcher::new(&[]),
            decoded_piece_bytes: (0..n).map(|_| std::sync::OnceLock::new()).collect(),
            bos: None,
            eos: None,
            add_bos: false,
            add_eos: false,
            pretokenizer: PretokenizerKind::Glm4,
            whole_piece_ids: Some(whole),
        };
        // Only complete pre-tokens hit the lookup; "abcd" and "xyz" use BPE.
        assert_eq!(
            tokenizer.encode("abc \u{e9} abcd xyz", false).unwrap(),
            [256, 257, 32, 97, 98, 99, 100, 32, 120, 121, 122]
        );
        tokenizer.whole_piece_ids = None;
        assert_eq!(tokenizer.encode("abc", false).unwrap(), [97, 98, 99]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]
        #[test]
        fn splitter_matches_regex_for_unicode_input(chars in prop::collection::vec(any::<char>(), 0..256)) {
            static REGEX: std::sync::OnceLock<fancy_regex::Regex> = std::sync::OnceLock::new();
            let regex = REGEX.get_or_init(|| {
                let data = fixtures();
                let pattern = data["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"].as_str().unwrap();
                fancy_regex::Regex::new(pattern).unwrap()
            });
            let text: String = chars.into_iter().collect();
            let expected: Vec<_> = regex.find_iter(&text).map(|m| m.unwrap().as_str()).collect();
            prop_assert_eq!(pretokenize(&text), expected);
        }
    }

    fn gguf_paths() -> Vec<PathBuf> {
        let mut paths = vec![crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required()];
        if std::path::Path::new(ALTERNATE_GGUF).exists() {
            paths.push(ALTERNATE_GGUF.into());
        }
        paths
    }

    #[test]
    #[ignore = "CPU/header-only; requires GLM53_GGUF (GLM-5.3-Flash shard 1), no inference or GPU"]
    fn gguf_tokenizer_matches_hf_ids() {
        let data = fixtures();
        for path in gguf_paths() {
            let gguf = GgufFile::open(&path).unwrap();
            // GLM declares EOM, which GgufFile::stop_token_ids does not collect;
            // the GLM family owns its stop set.
            assert_eq!(gguf.stop_token_ids().unwrap(), [EOS, EOT]);
            assert_eq!(
                gguf.get_u64("tokenizer.ggml.eom_token_id"),
                Some(EOM as u64)
            );
            let mut tokenizer = NativeTokenizer::from_gguf(&gguf).unwrap();
            assert_eq!(tokenizer.pretokenizer, PretokenizerKind::Glm4);
            assert_eq!(tokenizer.n_vocab(), 154_880);
            assert_eq!((tokenizer.bos(), tokenizer.eos()), (Some(BOS), Some(EOS)));
            assert!(!tokenizer.add_bos && !tokenizer.add_eos);
            for case in cases(&data) {
                let text = case["text"].as_str().unwrap();
                let expected = ids(case, "ids");
                assert_eq!(tokenizer.encode(text, false).unwrap(), expected, "{text:?}");
                assert_eq!(
                    tokenizer.encode(text, true).unwrap(),
                    ids(case, "ids_with_special"),
                    "{text:?}"
                );
                assert_eq!(
                    tokenizer.try_decode(&expected).unwrap(),
                    case["decoded"].as_str().unwrap(),
                    "{text:?}"
                );
                let bytes: Vec<u8> = expected
                    .iter()
                    .flat_map(|&id| tokenizer.try_decode_piece_bytes_exact(id).unwrap())
                    .copied()
                    .collect();
                assert_eq!(bytes, text.as_bytes(), "{text:?}");
            }
            // Without the whole-piece lookup the same vocabulary reproduces
            // HF with ignore_merges=false, proving the lookup is what differs.
            tokenizer.whole_piece_ids = None;
            for case in cases(&data) {
                if case.get("ids_without_ignore_merges").is_some() {
                    let text = case["text"].as_str().unwrap();
                    assert_eq!(
                        tokenizer.encode(text, false).unwrap(),
                        ids(case, "ids_without_ignore_merges"),
                        "{text:?}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "CPU/vocab-only; requires GLM53_GGUF and a glm4 ignore_merges llama-tokenize (GLM_LLAMA_TOKENIZE)"]
    fn gguf_tokenizer_matches_llama_tokenize_cli() {
        let binary = std::env::var("GLM_LLAMA_TOKENIZE").unwrap_or_else(|_| LLAMA_TOKENIZE.into());
        assert!(
            std::path::Path::new(&binary).exists(),
            "missing llama-tokenize oracle {binary}"
        );
        let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
        let tokenizer = NativeTokenizer::open(&path).unwrap();
        let prompt_file =
            std::env::temp_dir().join(format!("glm4-llama-tokenize-{}.txt", std::process::id()));
        let data = fixtures();
        let mut mismatches = Vec::new();
        for case in cases(&data) {
            let text = case["text"].as_str().unwrap();
            std::fs::write(&prompt_file, text).unwrap();
            let output = std::process::Command::new(&binary)
                .arg("-m")
                .arg(&path)
                .arg("-f")
                .arg(&prompt_file)
                .args(["--ids", "--no-escape", "--log-disable"])
                .output()
                .expect("run llama-tokenize");
            assert!(
                output.status.success(),
                "llama-tokenize failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let oracle: Vec<i32> = serde_json::from_slice(&output.stdout).unwrap();
            let native = tokenizer.encode(text, false).unwrap();
            if native != oracle {
                mismatches.push(format!("{text:?}: native={native:?} llama={oracle:?}"));
            }
        }
        let _ = std::fs::remove_file(&prompt_file);
        assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    }
}
