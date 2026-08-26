use qwen_llm::gguf::GgufFile;
use qwen_llm::qwen4exp::Qwen4ExpConfig;

#[test]
#[ignore = "set QWEN4EXP_METADATA_GGUF to a standalone metadata shard"]
fn parses_live_qwen4exp_metadata_when_available() {
    let path = std::env::var_os("QWEN4EXP_METADATA_GGUF")
        .expect("QWEN4EXP_METADATA_GGUF must point to a metadata GGUF");
    let gguf = GgufFile::open(path).expect("open Qwen4Exp metadata GGUF");
    let actual = Qwen4ExpConfig::from_gguf(&gguf).expect("parse Qwen4Exp metadata");
    assert_eq!(actual, Qwen4ExpConfig::flash_next_reference());
}
