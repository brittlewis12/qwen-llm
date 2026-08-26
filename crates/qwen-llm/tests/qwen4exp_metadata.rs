use qwen_llm::gguf::GgufFile;
use qwen_llm::qwen4exp::Qwen4ExpConfig;
use qwen_llm::qwen4exp_loader::{MixerWeights, Qwen4ExpModel};
use qwen_llm::qwen4exp_ple::PleIq4NlTable;
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
    let ple_embedding = model.ple_embedding.unwrap();
    assert_eq!(ple_embedding.dtype, GgmlType::IQ4_NL);
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

    let table = PleIq4NlTable::new(
        ple_embedding,
        gguf.try_slice(ple_embedding).expect("slice PLE embedding"),
        actual
            .ple
            .as_ref()
            .unwrap()
            .logical_row_count()
            .expect("count logical PLE rows"),
    )
    .expect("bind IQ4_NL PLE table");
    let row_ids = actual
        .ple
        .as_ref()
        .unwrap()
        .row_ids(42, &[])
        .expect("hash PLE rows");
    assert_eq!(row_ids.len(), 16);
    assert_eq!(table.row_width(), 160);
    let mut packed = vec![0; table.packed_staging_bytes(row_ids.len()).unwrap()];
    table
        .gather_packed_into(&row_ids, &mut packed)
        .expect("gather released PLE rows");
    assert_eq!(packed.len(), 1_440);
    let mut decoded = vec![0.0; table.f32_staging_elements(row_ids.len()).unwrap()];
    table
        .gather_f32_into(&row_ids, &mut decoded)
        .expect("decode released PLE rows");
    assert_eq!(decoded.len(), 2_560);
    assert!(decoded.iter().all(|value| value.is_finite()));
}
