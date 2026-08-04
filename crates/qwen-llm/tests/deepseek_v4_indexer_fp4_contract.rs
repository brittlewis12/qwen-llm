use qwen_llm::deepseek_v4_oracle::{
    DeepSeekV4IndexerFp4Row, DeepSeekV4OracleError, INDEXER_FP4_BLOCK_COUNT,
    INDEXER_FP4_BLOCK_VALUES, INDEXER_FP4_ROW_BYTES, INDEXER_FP4_SCALE_BYTES,
    INDEXER_FP4_VALUE_BYTES, INDEXER_FP4_VALUES_PER_ROW, indexer_fp4_e2m1_code,
    indexer_fp4_ue8m0_scale, indexer_fp4_ue8m0_scale_code, pack_indexer_fp4_row,
    packed_indexer_scores, top_k_indices, unpack_indexer_fp4_row,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const FIXTURE_JSON: &str = include_str!("fixtures/deepseek_v4_indexer_fp4_contract_v1.json");
const FIXTURE_SHA256: &str = "0e5e2b251a960d417e7977608a363b83e072e2d90bc286cc52820b0ea7dc2b1f";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    codebook_bits: Vec<u32>,
    evidence_kind: String,
    generator_version: u32,
    invalid_rows: Vec<InvalidRowCase>,
    layout: Layout,
    pack_rejections: Vec<PackRejectionCase>,
    rounding_cases: Vec<RoundingCase>,
    rows: Vec<RowCase>,
    scale_cases: Vec<ScaleCase>,
    schema_version: u32,
    score_cases: Vec<ScoreCase>,
    sources: Sources,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Layout {
    amax_floor_bits: u32,
    bf16_input_semantics: bool,
    block_count: usize,
    block_values: usize,
    canonical_scale_codes: [u8; 2],
    fixture_row_envelope: String,
    nibble_order: String,
    rounding: String,
    rounding_case_input: String,
    row_bytes: usize,
    scale_bytes: usize,
    scale_offset: usize,
    scale_order: String,
    scale_case_input: String,
    scalar_row_domain: String,
    upstream_storage: String,
    value_bytes: usize,
    values_per_row: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RoundingCase {
    code: u8,
    input_bits: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RowCase {
    decoded_bits: Vec<u32>,
    input_bits: Vec<u32>,
    name: String,
    packed_bytes: Vec<u8>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScaleCase {
    code: u8,
    decoded_scale_bits: u32,
    maximum_bits: u32,
    name: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InvalidRowCase {
    error_category: String,
    mutations: Vec<ByteMutation>,
    name: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ByteMutation {
    byte_index: usize,
    byte_value: u8,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PackRejectionCase {
    dimension: usize,
    error_category: String,
    input_bits: u32,
    name: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScoreCase {
    evidence_kind: String,
    head_weight_semantics: String,
    key_rows: Vec<Vec<u8>>,
    name: String,
    normalization_bits: u32,
    query_rows: Vec<Vec<u8>>,
    raw_head_weight_bits: Vec<u32>,
    scaled_head_weight_bits: Vec<u32>,
    score_bits: Vec<u32>,
    top2: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Sources {
    dwarfstar: Source,
    official: Source,
    vllm: Source,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    files: BTreeMap<String, String>,
    repository: String,
    revision: String,
    role: String,
}

fn fixture() -> Fixture {
    serde_json::from_str(FIXTURE_JSON).expect("valid strict FP4 fixture")
}

fn packed_row(bytes: &[u8]) -> DeepSeekV4IndexerFp4Row {
    let bytes: [u8; INDEXER_FP4_ROW_BYTES] = bytes.try_into().expect("68-byte packed row");
    DeepSeekV4IndexerFp4Row::from_bytes(bytes).expect("canonical packed row")
}

fn assert_error_category(error: DeepSeekV4OracleError, category: &str, case: &str) {
    match (category, error) {
        ("noncanonical_scale_code", DeepSeekV4OracleError::Invalid { name, detail }) => {
            assert_eq!(name, "indexer UE8M0 scale", "{case}");
            assert_eq!(detail, "canonical code must be in 1..=253", "{case}");
        }
        ("decoded_f32_overflow", DeepSeekV4OracleError::Invalid { name, detail }) => {
            assert_eq!(name, "decoded indexer FP4 row", "{case}");
            assert_eq!(detail, "must contain only finite values", "{case}");
        }
        (category, error) => panic!("{case}: unexpected {category} error {error}"),
    }
}

#[test]
fn fixture_provenance_and_layout_are_pinned() {
    assert_eq!(
        format!("{:x}", Sha256::digest(FIXTURE_JSON)),
        FIXTURE_SHA256
    );
    let fixture = fixture();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.generator_version, 1);
    assert_eq!(
        fixture.evidence_kind,
        "official BF16 QAT source, vLLM packed reference, and DwarfStar QAT cross-check"
    );

    assert_eq!(
        fixture.sources.official.repository,
        "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731"
    );
    assert_eq!(
        fixture.sources.official.revision,
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    );
    assert_eq!(
        fixture.sources.official.role,
        "model placement and BF16-input E2M1/UE8M0 semantics"
    );
    assert_eq!(
        fixture.sources.official.files["inference/kernel.py"],
        "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2"
    );
    assert_eq!(
        fixture.sources.official.files["inference/model.py"],
        "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f"
    );

    assert_eq!(
        fixture.sources.vllm.repository,
        "https://github.com/vllm-project/vllm"
    );
    assert_eq!(
        fixture.sources.vllm.revision,
        "b40d859c7b07ae244bcd8c6eecdcdbd9a3afaa07"
    );
    assert_eq!(
        fixture.sources.vllm.role,
        "independent packed nibble, scale, Q, and paged K reference"
    );
    assert_eq!(
        fixture.sources.vllm.files["vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py"],
        "8671d02dc1c2c495c3b5f16608faca59fd8d6507664fdcd93eda49ab0be1c6cb"
    );
    assert_eq!(
        fixture.sources.vllm.files["vllm/models/deepseek_v4/common/ops/fused_indexer_q.py"],
        "2c0f33f5a6b06af371011fb5683af84fcf7ee502043041a57f74bb4b116d0495"
    );
    assert_eq!(
        fixture.sources.vllm.files["tests/kernels/test_fused_indexer_q_rope_quant.py"],
        "4536a4eb2315102f5a2d9684a613f1599a10992e813d355e5b1b9dfb7f31442a"
    );

    assert_eq!(
        fixture.sources.dwarfstar.repository,
        "https://github.com/antirez/ds4"
    );
    assert_eq!(
        fixture.sources.dwarfstar.revision,
        "54b36ed9ba42da31b24f2d1a5feb075c2475dbb1"
    );
    assert_eq!(
        fixture.sources.dwarfstar.role,
        "independent Hadamard plus FP4 simulation cross-check"
    );
    assert_eq!(
        fixture.sources.dwarfstar.files["ds4.c"],
        "af5df58420632c453657ffdfc2c7cb84e75135bbcc20deaca3fedf970c13930c"
    );
    assert_eq!(
        fixture.sources.dwarfstar.files["metal/dsv4_kv.metal"],
        "b8a77a47f145ec13ee1528a038a86baff19f66f6843c3f639a51128aa0551835"
    );

    let layout = fixture.layout;
    assert_eq!(layout.values_per_row, INDEXER_FP4_VALUES_PER_ROW);
    assert_eq!(layout.block_values, INDEXER_FP4_BLOCK_VALUES);
    assert_eq!(layout.block_count, INDEXER_FP4_BLOCK_COUNT);
    assert_eq!(layout.value_bytes, INDEXER_FP4_VALUE_BYTES);
    assert_eq!(layout.scale_bytes, INDEXER_FP4_SCALE_BYTES);
    assert_eq!(layout.scale_offset, INDEXER_FP4_VALUE_BYTES);
    assert_eq!(layout.row_bytes, INDEXER_FP4_ROW_BYTES);
    assert_eq!(layout.canonical_scale_codes, [1, 253]);
    assert_eq!(layout.amax_floor_bits, 0x01c0_0000);
    assert!(layout.bf16_input_semantics);
    assert_eq!(
        layout.nibble_order,
        "dimension 2*i low; dimension 2*i+1 high"
    );
    assert_eq!(layout.scale_order, "dimensions 0..31,32..63,64..95,96..127");
    assert_eq!(layout.rounding, "E2M1 round-to-nearest-even");
    assert_eq!(
        layout.rounding_case_input,
        "finite scaled E2M1 conversion operand"
    );
    assert_eq!(
        layout.scale_case_input,
        "already-BF16-rounded nonnegative block amax"
    );
    assert_eq!(
        layout.scalar_row_domain,
        "construction requires every decoded value to remain finite F32; physically encodable overflow rows are rejected"
    );
    assert_eq!(
        layout.upstream_storage,
        "Q uses separate value/scale tensors; paged K stores all block values before block scales"
    );
    assert_eq!(
        layout.fixture_row_envelope,
        "64 packed value bytes followed by 4 scale bytes"
    );
}

#[test]
fn fixture_rounding_scales_and_rows_match_the_rust_oracle() {
    let fixture = fixture();
    for case in fixture.rounding_cases {
        assert_eq!(
            indexer_fp4_e2m1_code(f32::from_bits(case.input_bits)).unwrap(),
            case.code
        );
    }
    for case in fixture.scale_cases {
        assert_eq!(
            indexer_fp4_ue8m0_scale_code(f32::from_bits(case.maximum_bits)).unwrap(),
            case.code,
            "{}",
            case.name
        );
        assert_eq!(
            indexer_fp4_ue8m0_scale(case.code).unwrap().to_bits(),
            case.decoded_scale_bits,
            "{}",
            case.name
        );
    }

    assert_eq!(fixture.codebook_bits.len(), 16);
    for (code, &expected) in fixture.codebook_bits.iter().enumerate() {
        let mut bytes = [0u8; INDEXER_FP4_ROW_BYTES];
        bytes[0] = code as u8;
        bytes[INDEXER_FP4_VALUE_BYTES..].fill(127);
        let row = DeepSeekV4IndexerFp4Row::from_bytes(bytes).unwrap();
        assert_eq!(unpack_indexer_fp4_row(&row).unwrap()[0].to_bits(), expected);
    }

    for case in fixture.rows {
        let input = case
            .input_bits
            .iter()
            .map(|&bits| f32::from_bits(bits))
            .collect::<Vec<_>>();
        let packed = pack_indexer_fp4_row(&input).unwrap();
        if case.name == "bf16_amax_and_four_block_order" {
            assert_eq!(case.input_bits[31], 6.0f32.to_bits() + 1);
            assert_eq!(case.input_bits[32], 0x40c1_0000);
            assert_eq!(
                &case.packed_bytes[INDEXER_FP4_VALUE_BYTES..],
                &[127, 128, 129, 130]
            );
        }
        assert_eq!(
            packed.as_bytes().as_slice(),
            case.packed_bytes,
            "{}",
            case.name
        );
        assert_eq!(
            unpack_indexer_fp4_row(&packed)
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            case.decoded_bits,
            "{}",
            case.name
        );
    }
}

#[test]
fn fixture_invalid_rows_and_upper_domain_fail_closed() {
    let fixture = fixture();
    assert_eq!(fixture.invalid_rows.len(), 20);
    for case in fixture.invalid_rows {
        let mut bytes = [0u8; INDEXER_FP4_ROW_BYTES];
        bytes[INDEXER_FP4_VALUE_BYTES..].fill(127);
        for mutation in case.mutations {
            bytes[mutation.byte_index] = mutation.byte_value;
        }
        let error = DeepSeekV4IndexerFp4Row::from_bytes(bytes).unwrap_err();
        assert_error_category(error, &case.error_category, &case.name);
    }

    assert_eq!(fixture.pack_rejections.len(), 1);
    for case in fixture.pack_rejections {
        let mut values = [0.0f32; INDEXER_FP4_VALUES_PER_ROW];
        values[case.dimension] = f32::from_bits(case.input_bits);
        let error = pack_indexer_fp4_row(&values).unwrap_err();
        assert_error_category(error, &case.error_category, &case.name);
    }
}

#[test]
fn fixture_packed_scores_match_the_rust_oracle() {
    let fixture = fixture();
    assert_eq!(fixture.score_cases.len(), 3);
    for case in &fixture.score_cases {
        assert_eq!(
            case.evidence_kind,
            "deterministic scalar transcription over decoded packed operands"
        );
        assert_eq!(
            case.head_weight_semantics,
            "projection weights pre-scaled once by the recorded normalization"
        );
        let normalization = f32::from_bits(case.normalization_bits);
        let raw_weights = case
            .raw_head_weight_bits
            .iter()
            .map(|&bits| f32::from_bits(bits))
            .collect::<Vec<_>>();
        let scaled_weights = case
            .scaled_head_weight_bits
            .iter()
            .map(|&bits| f32::from_bits(bits))
            .collect::<Vec<_>>();
        assert_eq!(
            raw_weights
                .iter()
                .map(|weight| (*weight * normalization).to_bits())
                .collect::<Vec<_>>(),
            case.scaled_head_weight_bits,
            "{}",
            case.name
        );
        let queries = case
            .query_rows
            .iter()
            .map(|bytes| packed_row(bytes))
            .collect::<Vec<_>>();
        let keys = case
            .key_rows
            .iter()
            .map(|bytes| packed_row(bytes))
            .collect::<Vec<_>>();
        let scores = packed_indexer_scores(&queries, &scaled_weights, &keys).unwrap();
        assert_eq!(
            scores
                .iter()
                .map(|score| score.to_bits())
                .collect::<Vec<_>>(),
            case.score_bits,
            "{}",
            case.name
        );
        assert_eq!(
            top_k_indices(&scores, 2).unwrap(),
            case.top2,
            "{}",
            case.name
        );
        match case.name.as_str() {
            "relu_before_negative_weight" => {
                assert!(scores[0] < 0.0);
                assert!(scores[1] > 0.0);
                assert_eq!(scores[2].to_bits(), 0.0f32.to_bits());
            }
            "exact_head_cancellation_and_row_tie" => {
                assert!(scores.iter().all(|score| score.to_bits() == 0));
                assert_eq!(case.top2, [0, 1]);
            }
            _ => {}
        }
    }
}
