use qwen_llm::gguf::GgufFile;
use qwen_llm::qwen4exp::Qwen4ExpConfig;
use qwen_llm::qwen4exp_loader::{MixerWeights, Qwen4ExpModel};
use qwen_llm::tensor::GgmlType;

#[test]
#[ignore = "set QWEN4EXP_METADATA_GGUF to a standalone metadata shard"]
fn parses_live_qwen4exp_metadata_when_available() {
    let path = std::env::var_os("QWEN4EXP_METADATA_GGUF")
        .expect("QWEN4EXP_METADATA_GGUF must point to a metadata GGUF");
    let gguf = GgufFile::open(path).expect("open Qwen4Exp metadata GGUF");
    let actual = Qwen4ExpConfig::from_gguf(&gguf).expect("parse Qwen4Exp metadata");
    assert_eq!(actual, Qwen4ExpConfig::flash_next_reference());

    let model = Qwen4ExpModel::from_gguf(&gguf).expect("bind Qwen4Exp tensors");
    assert_eq!(gguf.tensors.len(), 1_224);
    assert_eq!(model.blocks.len(), 48);
    assert_eq!(model.ple_embedding.unwrap().dtype, GgmlType::IQ4_NL);
    assert!(model.blocks[1].ple.is_some());
    assert_eq!(
        model
            .blocks
            .iter()
            .filter(|block| block.ple.is_some())
            .count(),
        1
    );
    assert_eq!(
        model
            .blocks
            .iter()
            .filter(|block| matches!(block.mixer, MixerWeights::QwenSparseAttention(_)))
            .count(),
        12
    );
}
