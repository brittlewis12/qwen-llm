use qwen_llm::deepseek_v4_census::{PinnedDeepSeekV4AssetV1, RoleCensus, StorageCensus};
use qwen_llm::gguf::GgufFile;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::path::PathBuf;

const FIXTURE_JSON: &str =
    include_str!("fixtures/deepseek_v4_flash_0731_ud_iq3_xxs_census_v1.json");
const DEFAULT_MODEL: &str = "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf";

fn fixture() -> PinnedDeepSeekV4AssetV1 {
    PinnedDeepSeekV4AssetV1::parse(FIXTURE_JSON).expect("valid pinned DS4 census")
}

fn recompute_census_digest(fixture: &mut PinnedDeepSeekV4AssetV1) {
    let digest = Sha256::digest(serde_json::to_vec(&fixture.census).unwrap());
    fixture.census_sha256.clear();
    for byte in digest {
        write!(&mut fixture.census_sha256, "{byte:02x}").unwrap();
    }
}

fn role<'a>(fixture: &'a PinnedDeepSeekV4AssetV1, name: &str) -> &'a RoleCensus {
    fixture
        .census
        .roles
        .iter()
        .find(|role| role.role == name)
        .unwrap_or_else(|| panic!("missing role {name}"))
}

fn storage<'a>(role: &'a RoleCensus, dtype: &str) -> &'a StorageCensus {
    role.storage
        .iter()
        .find(|storage| storage.dtype == dtype)
        .unwrap_or_else(|| panic!("missing {dtype} storage for {}", role.role))
}

#[test]
fn pinned_fixture_is_canonical_and_self_consistent() {
    let fixture = fixture();
    assert_eq!(fixture.manifest_schema_version, 1);
    assert_eq!(fixture.asset_id, "deepseek-v4-flash-0731-ud-iq3_xxs");
    assert_eq!(
        fixture.census_sha256,
        "f4397fae14a6df04786324006ce41ea0489d4b246f68e742207446098684e4fc"
    );
    assert_eq!(fixture.census.totals.shard_count, 4);
    assert_eq!(fixture.census.totals.file_bytes, 102_999_888_416);
    assert_eq!(fixture.census.totals.tensor_count, 1_328);
    assert_eq!(fixture.census.totals.element_count, 284_334_567_511);
    assert_eq!(fixture.census.totals.tensor_bytes, 102_994_542_940);
    assert_eq!(
        fixture
            .shards
            .iter()
            .map(|shard| shard.sha256.as_str())
            .collect::<Vec<_>>(),
        [
            "9758eb3d78e1afe8852543931703f4f1cd6fbb07f492d4ed853f5d2f6e43be5a",
            "afcfd59721d4da86bc3301e16ca624af202d8af3fa3f9fbbfbb04b3b47666cfd",
            "64eaf514a763597ba7bb50866583d8db5eabbbbce3cb2f616d749af3890155ca",
            "5df52988c56348a22d15da809e9ac4f0cc59cc1c412347f1481dda4685ce89b2",
        ]
    );
}

#[test]
fn global_dtype_census_is_pinned() {
    let fixture = fixture();
    let observed = fixture
        .census
        .dtypes
        .iter()
        .map(|row| {
            (
                row.dtype.as_str(),
                row.dtype_tag,
                row.tensor_count,
                row.element_count,
                row.storage_bytes,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        observed,
        [
            ("F32", 0, 662, 41_266_775, 165_067_100),
            ("Q8_0", 8, 321, 4_928_307_200, 5_236_326_400),
            ("Q6_K", 14, 170, 2_292_187_136, 1_880_309_760),
            ("IQ3_XXS", 18, 41, 88_046_829_568, 33_705_426_944,),
            ("IQ3_S", 21, 2, 4_294_967_296, 1_845_493_760),
            ("IQ2_S", 22, 84, 180_388_626_432, 57_780_731_904),
            ("I32", 26, 3, 2_327_040, 9_308_160),
            ("BF16", 30, 43, 45_088_768, 90_177_536),
            ("MXFP4", 39, 2, 4_294_967_296, 2_281_701_376),
        ]
    );
}

#[test]
fn routed_moe_outliers_and_baselines_are_pinned() {
    let fixture = fixture();
    assert_eq!(
        storage(role(&fixture, "routed_gate"), "IQ2_S").tensor_count,
        42
    );
    assert_eq!(
        storage(role(&fixture, "routed_gate"), "IQ3_S").tensor_count,
        1
    );
    assert_eq!(
        storage(role(&fixture, "routed_up"), "IQ2_S").tensor_count,
        42
    );
    assert_eq!(
        storage(role(&fixture, "routed_up"), "IQ3_S").tensor_count,
        1
    );
    assert_eq!(
        storage(role(&fixture, "routed_down"), "IQ3_XXS").tensor_count,
        41
    );
    assert_eq!(
        storage(role(&fixture, "routed_down"), "MXFP4").tensor_count,
        2
    );

    let layer_26 = &fixture.census.layers[26];
    assert_eq!(layer_26.attention, "csa");
    assert_eq!(layer_26.routed_gate.dtype, "IQ3_S");
    assert_eq!(layer_26.routed_gate.storage_bytes, 922_746_880);
    assert_eq!(layer_26.routed_up.dtype, "IQ3_S");
    assert_eq!(layer_26.routed_down.dtype, "MXFP4");
    assert_eq!(layer_26.routed_down.storage_bytes, 1_140_850_688);

    let layer_42 = &fixture.census.layers[42];
    assert_eq!(layer_42.attention, "csa");
    assert_eq!(layer_42.routed_gate.dtype, "IQ2_S");
    assert_eq!(layer_42.routed_up.dtype, "IQ2_S");
    assert_eq!(layer_42.routed_down.dtype, "MXFP4");
}

#[test]
fn fixture_rejects_unknown_fields_and_digest_drift() {
    let mut unknown: serde_json::Value = serde_json::from_str(FIXTURE_JSON).unwrap();
    unknown
        .as_object_mut()
        .unwrap()
        .insert("undeclared".into(), true.into());
    assert!(serde_json::from_value::<PinnedDeepSeekV4AssetV1>(unknown).is_err());

    let mut drifted: PinnedDeepSeekV4AssetV1 = serde_json::from_str(FIXTURE_JSON).unwrap();
    let basename = format!("{}.drift", drifted.shards[0].basename);
    drifted.shards[0].basename.clone_from(&basename);
    drifted.census.shards[0].basename = basename;
    drifted.census.validate().unwrap();
    let error = drifted.validate().unwrap_err().to_string();
    assert!(error.contains("census digest changed"), "{error}");
}

#[test]
fn fixture_rejects_rehashed_dtype_identity_and_cross_table_drift() {
    let mut dtype_drift: PinnedDeepSeekV4AssetV1 = serde_json::from_str(FIXTURE_JSON).unwrap();
    dtype_drift.census.dtypes[0].dtype = "BF16".into();
    recompute_census_digest(&mut dtype_drift);
    let error = dtype_drift.validate().unwrap_err().to_string();
    assert!(error.contains("does not match wire name"), "{error}");

    let mut layer_drift: PinnedDeepSeekV4AssetV1 = serde_json::from_str(FIXTURE_JSON).unwrap();
    layer_drift.census.layers[0].routed_gate.element_count += 256;
    layer_drift.census.layers[0].routed_gate.storage_bytes += 82;
    layer_drift.census.layers[0].routed_up.element_count += 256;
    layer_drift.census.layers[0].routed_up.storage_bytes += 82;
    recompute_census_digest(&mut layer_drift);
    let error = layer_drift.validate().unwrap_err().to_string();
    assert!(
        error.contains("layer storage does not match role routed_gate"),
        "{error}"
    );

    let mut balanced_drift: PinnedDeepSeekV4AssetV1 = serde_json::from_str(FIXTURE_JSON).unwrap();
    balanced_drift.census.roles[1].storage[0].element_count += 1;
    balanced_drift.census.roles[1].storage[0].storage_bytes += 4;
    balanced_drift.census.roles[5].storage[0].element_count -= 1;
    balanced_drift.census.roles[5].storage[0].storage_bytes -= 4;
    balanced_drift.census.validate().unwrap();
    recompute_census_digest(&mut balanced_drift);
    let error = balanced_drift.validate().unwrap_err().to_string();
    assert!(
        error.contains("unsupported pinned census digest"),
        "{error}"
    );

    let mut shard_drift: PinnedDeepSeekV4AssetV1 = serde_json::from_str(FIXTURE_JSON).unwrap();
    for shard in &mut shard_drift.shards {
        shard.sha256 = "0".repeat(64);
    }
    let error = shard_drift.validate().unwrap_err().to_string();
    assert!(error.contains("identity is malformed"), "{error}");
}

#[test]
#[ignore = "mmaps the local DS4 asset and validates its 9 MiB hash-router tables"]
fn live_asset_matches_pinned_schema_and_quant_census() {
    let model_path = std::env::var_os("DSV4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    assert!(
        model_path.exists(),
        "missing DS4 fixture at {}",
        model_path.display()
    );
    let gguf = GgufFile::open(&model_path).expect("open DS4 fixture");
    fixture()
        .validate_observed(&gguf)
        .expect("live census matches pinned fixture");
}
