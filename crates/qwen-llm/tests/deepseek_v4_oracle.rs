use half::f16;
use qwen_llm::deepseek_v4_oracle::{
    CompressorState, HyperConnectionControls, RopeDirection, RopeParameters,
    attention_fp8_nope_bf16_rope_roundtrip_in_place, attention_fp8_nope_roundtrip_in_place,
    bf16_roundtrip_in_place, clamped_swiglu, compressor_pool, grouped_low_rank_output,
    grouped_low_rank_projection, hash_route, hyper_connection_collapse, hyper_connection_head,
    hyper_connection_post, hyper_connection_pre, indexer_qat_roundtrip_in_place, indexer_scores,
    learned_route, rope_tail_in_place, shared_kv_attention, shared_kv_projection, split_sinkhorn,
    sqrt_softplus_scores, top_k_indices,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const FIXTURE_JSON: &str = include_str!("fixtures/deepseek_v4_oracle_v1.json");

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: u32,
    generator_version: u32,
    sources: Sources,
    transcription_provenance: Vec<TranscriptionProvenance>,
    known_reference_differences: KnownReferenceDifferences,
    mhc: MhcFixture,
    rope: RopeFixture,
    shared_kv_projection: SharedKvProjectionFixture,
    attention_output: AttentionOutputFixture,
    compressor_ratio4: Ratio4Fixture,
    compressor_ratio128: Ratio128Fixture,
    indexer: IndexerFixture,
    routing: RoutingFixture,
    cache_roundtrip: CacheRoundtripFixture,
    llama_cpp_cpu: LlamaCppCpuFixture,
    dwarfstar_direct: DwarfstarDirectFixture,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Sources {
    vllm: String,
    sglang: String,
    llama_cpp: String,
    dwarfstar: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TranscriptionProvenance {
    cases: Vec<String>,
    sources: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KnownReferenceDifferences {
    fp4_tiny_scale_and_midpoints: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DwarfstarDirectFixture {
    revision: String,
    tree: String,
    repository: String,
    license: String,
    source_path: String,
    source_sha256: String,
    tracked_worktree_clean: bool,
    harness_path: String,
    harness_sha256: String,
    compiler: String,
    compiler_version: String,
    compile_flags: Vec<String>,
    platform: String,
    configuration: String,
    symbols: Vec<String>,
    vectors: DwarfstarVectors,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DwarfstarVectors {
    sinkhorn: Vec<f32>,
    hc_post: Vec<f32>,
    rope_local: Vec<f32>,
    rope_yarn: Vec<f32>,
    rope_yarn_inverse: Vec<f32>,
    ratio4_pool: Vec<f32>,
    ratio128_pool: Vec<f32>,
    indexer_qat: Vec<f32>,
    router_scores: Vec<f32>,
    router_selected: Vec<usize>,
    router_weights: Vec<f32>,
    swiglu: Vec<f32>,
    compressor_transitions: DwarfstarCompressorTransitions,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DwarfstarCompressorTransitions {
    hash_algorithm: String,
    rms_epsilon: f32,
    rotary_dim: usize,
    rope_theta: f32,
    rope_scale_factor: f32,
    original_context_length: u32,
    beta_fast: f32,
    beta_slow: f32,
    recipes: DwarfstarCompressorRecipes,
    cases: Vec<DwarfstarCompressorCase>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DwarfstarCompressorRecipes {
    input: String,
    kv_weight: String,
    gate_weight: String,
    ape: String,
    norm: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DwarfstarCompressorCase {
    name: String,
    seed: u32,
    input_dim: usize,
    head_dim: usize,
    ratio: usize,
    layer: usize,
    positions: usize,
    projection_type: String,
    state_rows: usize,
    state_width: usize,
    records: Vec<DwarfstarCompressorRecord>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DwarfstarCompressorRecord {
    position: u32,
    emitted: bool,
    kv_hash: String,
    score_hash: String,
    start_position: Option<u32>,
    output: Option<Vec<f32>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LlamaCppCpuFixture {
    revision: String,
    tree: String,
    repository: String,
    license: String,
    tracked_worktree_clean: bool,
    source_files: Vec<SourceFile>,
    harness_path: String,
    harness_sha256: String,
    build_project_path: String,
    build_project_sha256: String,
    build_toolchain: BuildToolchain,
    configuration: String,
    symbols: Vec<String>,
    vectors: LlamaCppVectors,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceFile {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuildToolchain {
    cmake_version: String,
    generator: String,
    c_compiler: String,
    c_compiler_id: String,
    c_compiler_target: String,
    c_compiler_version: String,
    c_flags: String,
    c_flags_release: String,
    cxx_compiler: String,
    cxx_compiler_id: String,
    cxx_compiler_target: String,
    cxx_compiler_version: String,
    cxx_flags: String,
    cxx_flags_release: String,
    exe_linker_flags: String,
    exe_linker_flags_release: String,
    osx_architectures: String,
    osx_deployment_target: String,
    osx_sysroot: String,
    system_name: String,
    system_processor: String,
    system_version: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LlamaCppVectors {
    status: String,
    backend: String,
    thread_count: usize,
    tensor_layout: String,
    hc_comb: LlamaCppHcComb,
    hc_pre: LlamaCppHcPre,
    hc_post: LlamaCppHcPost,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LlamaCppHcComb {
    connection_count: usize,
    token_count: usize,
    mixes_shape: Vec<usize>,
    mixes: Vec<f32>,
    scale_shape: Vec<usize>,
    scale: Vec<f32>,
    base_shape: Vec<usize>,
    base: Vec<f32>,
    epsilon: f32,
    iterations: usize,
    output_shape: Vec<usize>,
    output: Vec<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LlamaCppHcPre {
    hidden_size: usize,
    connection_count: usize,
    token_count: usize,
    residual_shape: Vec<usize>,
    residual: Vec<f32>,
    weights_shape: Vec<usize>,
    weights: Vec<f32>,
    output_shape: Vec<usize>,
    output: Vec<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LlamaCppHcPost {
    hidden_size: usize,
    connection_count: usize,
    token_count: usize,
    block_output_shape: Vec<usize>,
    block_output: Vec<f32>,
    residual_shape: Vec<usize>,
    residual: Vec<f32>,
    weights_shape: Vec<usize>,
    weights: Vec<f32>,
    combination_shape: Vec<usize>,
    combination: Vec<f32>,
    output_shape: Vec<usize>,
    output: Vec<f32>,
}

#[derive(Deserialize)]
struct MhcFixture {
    hidden_size: usize,
    connection_count: usize,
    rms_eps: f32,
    hc_eps: f32,
    sinkhorn_iterations: usize,
    residual: Vec<f32>,
    function: Vec<f32>,
    scale: Vec<f32>,
    base: Vec<f32>,
    expected_mixes: Vec<f32>,
    expected_pre: Vec<f32>,
    expected_post: Vec<f32>,
    expected_combination: Vec<f32>,
    expected_input: Vec<f32>,
    block_output: Vec<f32>,
    expected_post_output: Vec<f32>,
    head_function: Vec<f32>,
    head_scale: f32,
    head_base: Vec<f32>,
    expected_head_output: Vec<f32>,
}

#[derive(Deserialize)]
struct RopeFixture {
    input: Vec<f32>,
    head_count: usize,
    head_dim: usize,
    rotary_dim: usize,
    local_position: u32,
    expected_local: Vec<f32>,
    yarn_position: u32,
    expected_yarn: Vec<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SharedKvProjectionFixture {
    hidden_size: usize,
    q_lora_rank: usize,
    head_count: usize,
    head_dim: usize,
    input: Vec<f32>,
    q_a: Vec<f32>,
    q_a_norm: Vec<f32>,
    q_b: Vec<f32>,
    kv_weight: Vec<f32>,
    kv_norm: Vec<f32>,
    expected_q_lora_raw: Vec<f32>,
    expected_q_lora: Vec<f32>,
    expected_queries: Vec<f32>,
    expected_kv_raw: Vec<f32>,
    expected_kv: Vec<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AttentionOutputFixture {
    head_count: usize,
    head_dim: usize,
    queries: Vec<f32>,
    raw_kv: Vec<f32>,
    compressed_kv: Vec<f32>,
    compressed_allowed: Vec<bool>,
    sinks: Vec<f32>,
    expected_attention: Vec<f32>,
    inverse_position: u32,
    expected_inverse: Vec<f32>,
    group_count: usize,
    rank: usize,
    hidden_size: usize,
    output_a: Vec<f32>,
    output_b: Vec<f32>,
    expected_low_rank: Vec<f32>,
    expected_output: Vec<f32>,
}

#[derive(Clone, Copy, Deserialize)]
struct FixtureRope {
    rotary_dim: usize,
    theta: f32,
    scaling_factor: f32,
    original_context_length: u32,
    beta_fast: f32,
    beta_slow: f32,
}

impl From<FixtureRope> for RopeParameters {
    fn from(value: FixtureRope) -> Self {
        Self::yarn(
            value.rotary_dim,
            value.theta,
            value.scaling_factor,
            value.original_context_length,
            value.beta_fast,
            value.beta_slow,
        )
    }
}

#[derive(Deserialize)]
struct Ratio4Fixture {
    head_dim: usize,
    ape: Vec<f32>,
    norm_weight: Vec<f32>,
    rope: FixtureRope,
    projected_kv: Vec<Vec<f32>>,
    projected_scores: Vec<Vec<f32>>,
    emitted: Vec<EmittedFixture>,
    snapshots: Vec<SnapshotFixture>,
}

#[derive(Deserialize)]
struct EmittedFixture {
    position: u32,
    value: Vec<f32>,
}

#[derive(Deserialize)]
struct SnapshotFixture {
    position: u32,
    kv: Vec<f32>,
    scores: Vec<Option<f32>>,
}

#[derive(Deserialize)]
struct Ratio128Fixture {
    head_dim: usize,
    ape: Vec<f32>,
    norm_weight: Vec<f32>,
    rope: FixtureRope,
    projected_kv: Vec<Vec<f32>>,
    projected_scores: Vec<Vec<f32>>,
    emitted: Vec<EmittedFixture>,
    snapshots: Vec<SnapshotFixture>,
    expected_kv_state: Vec<f32>,
    expected_score_state: Vec<f32>,
}

#[derive(Deserialize)]
struct IndexerFixture {
    qat_input: Vec<f32>,
    expected_qat: Vec<f32>,
    head_count: usize,
    head_dim: usize,
    queries: Vec<f32>,
    head_weights: Vec<f32>,
    compressed_keys: Vec<f32>,
    expected_scores: Vec<f32>,
    top_k: usize,
    expected_top_k: Vec<usize>,
}

#[derive(Deserialize)]
struct RoutingFixture {
    logits: Vec<f32>,
    expected_scores: Vec<f32>,
    bias: Vec<f32>,
    top_k: usize,
    routed_scale: f32,
    expected_experts: Vec<usize>,
    expected_weights: Vec<f32>,
    hash_experts: Vec<usize>,
    expected_hash_weights: Vec<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheRoundtripFixture {
    rotary_dim: usize,
    input: Vec<f32>,
    expected_attention_cache: Vec<f32>,
    bf16_input: Vec<f32>,
    expected_bf16: Vec<f32>,
}

fn fixture() -> Fixture {
    serde_json::from_str(FIXTURE_JSON).expect("valid DS4 oracle fixture")
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index}: expected {expected}, got {actual}"
        );
    }
}

fn assert_snapshot(actual: &[f32], expected: &[Option<f32>], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
        if let Some(expected) = expected {
            assert!(
                (actual - expected).abs() <= tolerance,
                "state value {index}: expected {expected}, got {actual}"
            );
        } else {
            assert_eq!(actual, f32::NEG_INFINITY, "state value {index}");
        }
    }
}

fn compressor_recipe_value(value: u64, modulus: u64, center: i64, denominator: f32) -> f32 {
    ((value % modulus) as i64 - center) as f32 / denominator
}

fn compressor_input_value(seed: u32, position: u32, column: usize) -> f32 {
    const NUMERATORS: [i8; 8] = [-4, -3, -2, -1, 1, 2, 3, 4];
    let index = u64::from(seed) * 31 + u64::from(position) * 5 + column as u64 * 7;
    f32::from(NUMERATORS[index as usize % NUMERATORS.len()]) / 4.0
}

fn compressor_signed_weight(value: u64, base: u64, span: u64, denominator: f32) -> f32 {
    let magnitude = (base + ((value >> 1) % span)) as f32 / denominator;
    if value & 1 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn compressor_kv_weight(seed: u32, row: usize, column: usize) -> f32 {
    let value = u64::from(seed) * 17 + row as u64 * 5 + column as u64 * 13;
    compressor_signed_weight(value, 500, 1001, 1000.0)
}

fn compressor_gate_weight(seed: u32, row: usize, column: usize) -> f32 {
    let value = u64::from(seed) * 23 + row as u64 * 11 + column as u64 * 19;
    compressor_signed_weight(value, 250, 751, 997.0)
}

fn compressor_ape_value(seed: u32, position: usize, row: usize) -> f32 {
    compressor_recipe_value(
        u64::from(seed) * 29 + position as u64 * 13 + row as u64 * 3,
        257,
        128,
        1000.0,
    )
}

fn compressor_norm_value(seed: u32, row: usize) -> f32 {
    0.75 + ((u64::from(seed) * 5 + row as u64 * 7) % 101) as f32 / 1000.0
}

fn f16_roundtrip(value: f32) -> f32 {
    f16::from_f32(value).to_f32()
}

fn f32_state_hash(values: &[f32]) -> String {
    let mut hash = 14_695_981_039_346_656_037u64;
    for value in values {
        let bits = if *value == 0.0 { 0 } else { value.to_bits() };
        for byte in bits.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(1_099_511_628_211);
        }
    }
    format!("{hash:016x}")
}

#[test]
fn fixture_provenance_is_pinned() {
    assert_eq!(
        format!("{:x}", Sha256::digest(FIXTURE_JSON.as_bytes())),
        "e1da4713e3a259d86fbede552fcff0f8f151d39e7a637b219c7d5f9826b9a4a5"
    );
    let fixture = fixture();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.generator_version, 5);
    assert_eq!(
        fixture.sources.vllm,
        "b40d859c7b07ae244bcd8c6eecdcdbd9a3afaa07"
    );
    assert_eq!(
        fixture.sources.sglang,
        "58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8"
    );
    assert_eq!(
        fixture.sources.llama_cpp,
        "876a4321163249c43ca4e986818fab5ab081f282"
    );
    assert_eq!(
        fixture.sources.dwarfstar,
        "54b36ed9ba42da31b24f2d1a5feb075c2475dbb1"
    );
    assert_eq!(fixture.dwarfstar_direct.revision, fixture.sources.dwarfstar);
    assert_eq!(
        fixture.dwarfstar_direct.tree,
        "5807bbe362672ccd02251ba70c043ce311bc4737"
    );
    assert_eq!(
        fixture.dwarfstar_direct.repository,
        "https://github.com/antirez/ds4"
    );
    assert_eq!(fixture.dwarfstar_direct.license, "MIT");
    assert_eq!(fixture.dwarfstar_direct.source_path, "ds4.c");
    assert_eq!(
        fixture.dwarfstar_direct.source_sha256,
        "af5df58420632c453657ffdfc2c7cb84e75135bbcc20deaca3fedf970c13930c"
    );
    assert!(fixture.dwarfstar_direct.tracked_worktree_clean);
    assert_eq!(
        fixture.dwarfstar_direct.harness_path,
        "scripts/reference/dsv4_dwarfstar_oracle.c"
    );
    assert_eq!(
        fixture.dwarfstar_direct.harness_sha256,
        "8651249cb995385d3fe27f7b801141e7eb1aaa001649c991270c1f9cddcd2950"
    );
    assert_eq!(fixture.dwarfstar_direct.compiler, "clang");
    assert_eq!(
        fixture.dwarfstar_direct.compiler_version,
        "Apple clang version 17.0.0 (clang-1700.6.3.2)\nTarget: arm64-apple-darwin24.6.0\nThread model: posix"
    );
    assert_eq!(
        fixture.dwarfstar_direct.compile_flags,
        [
            "-std=c11",
            "-O0",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,-dead_strip",
            "-lm",
            "-lpthread",
        ]
    );
    assert_eq!(fixture.dwarfstar_direct.platform, "darwin");
    assert_eq!(
        fixture.dwarfstar_direct.configuration,
        "C11 O0 DS4_NO_GPU direct source include"
    );
    assert!(
        fixture
            .dwarfstar_direct
            .symbols
            .iter()
            .any(|symbol| symbol == "hc_split_sinkhorn_one")
    );
    assert!(
        fixture
            .dwarfstar_direct
            .symbols
            .iter()
            .any(|symbol| symbol == "compressor_decode_one")
    );
    let llama_cpp = &fixture.llama_cpp_cpu;
    assert_eq!(llama_cpp.revision, fixture.sources.llama_cpp);
    assert_eq!(llama_cpp.tree, "b127fd3e9b45bef820e6e6914f53f71270a9a6f9");
    assert_eq!(
        llama_cpp.repository,
        "https://github.com/ggml-org/llama.cpp"
    );
    assert_eq!(llama_cpp.license, "MIT");
    assert!(llama_cpp.tracked_worktree_clean);
    assert_eq!(
        llama_cpp.harness_path,
        "scripts/reference/dsv4_llama_cpp_cpu_oracle.cpp"
    );
    assert_eq!(
        llama_cpp.harness_sha256,
        "dca503457c43eb730db0fa6b5d69b58928f8d9ce8d7fc7c1248dbe3680d3c8ca"
    );
    assert_eq!(
        llama_cpp.build_project_path,
        "scripts/reference/CMakeLists.txt"
    );
    assert_eq!(
        llama_cpp.build_project_sha256,
        "f5ac07964737c142624d36033392afe5ae72ca867f5c72ce9b242379d3c4925a"
    );
    assert_eq!(llama_cpp.build_toolchain.cmake_version, "4.2.1");
    assert_eq!(llama_cpp.build_toolchain.generator, "Unix Makefiles");
    assert_eq!(llama_cpp.build_toolchain.c_compiler, "/usr/bin/cc");
    assert_eq!(llama_cpp.build_toolchain.c_compiler_id, "AppleClang");
    assert_eq!(llama_cpp.build_toolchain.c_compiler_target, "<default>");
    assert_eq!(
        llama_cpp.build_toolchain.c_compiler_version,
        "17.0.0.17000603"
    );
    assert_eq!(llama_cpp.build_toolchain.c_flags, "<default>");
    assert_eq!(llama_cpp.build_toolchain.c_flags_release, "-O3 -DNDEBUG");
    assert_eq!(llama_cpp.build_toolchain.cxx_compiler, "/usr/bin/c++");
    assert_eq!(llama_cpp.build_toolchain.cxx_compiler_id, "AppleClang");
    assert_eq!(llama_cpp.build_toolchain.cxx_compiler_target, "<default>");
    assert_eq!(
        llama_cpp.build_toolchain.cxx_compiler_version,
        "17.0.0.17000603"
    );
    assert_eq!(llama_cpp.build_toolchain.cxx_flags, "<default>");
    assert_eq!(llama_cpp.build_toolchain.cxx_flags_release, "-O3 -DNDEBUG");
    assert_eq!(llama_cpp.build_toolchain.exe_linker_flags, "<default>");
    assert_eq!(
        llama_cpp.build_toolchain.exe_linker_flags_release,
        "<default>"
    );
    assert_eq!(llama_cpp.build_toolchain.osx_architectures, "<default>");
    assert_eq!(llama_cpp.build_toolchain.osx_deployment_target, "<default>");
    assert_eq!(llama_cpp.build_toolchain.osx_sysroot, "<default>");
    assert_eq!(llama_cpp.build_toolchain.system_name, "Darwin");
    assert_eq!(llama_cpp.build_toolchain.system_processor, "arm64");
    assert_eq!(llama_cpp.build_toolchain.system_version, "24.6.0");
    assert_eq!(
        llama_cpp.configuration,
        "static CPU-only GGML, one thread, reference compute plan"
    );
    let expected_sources = [
        (
            "ggml/include/ggml.h",
            "c65c30fdb4dce95eac71c26bb38ae8423fbc80d79db91d2b2ffaea8c4e46276a",
        ),
        (
            "ggml/src/ggml.c",
            "9e40ad07323c7925f06a105119dfb07c1d4a21d3263a9e9bd0bd21792c42e1e4",
        ),
        (
            "ggml/src/ggml-cpu/ops.cpp",
            "dd7265a7402515002d4679f18d5d1d423456e87f256f7cc11af3bbb4148bf773",
        ),
    ];
    assert_eq!(llama_cpp.source_files.len(), expected_sources.len());
    for (source, (expected_path, expected_sha256)) in
        llama_cpp.source_files.iter().zip(expected_sources)
    {
        assert_eq!(source.path, expected_path);
        assert_eq!(source.sha256, expected_sha256);
    }
    assert_eq!(
        llama_cpp.symbols,
        ["ggml_dsv4_hc_comb", "ggml_dsv4_hc_pre", "ggml_dsv4_hc_post",]
    );
    assert_eq!(llama_cpp.vectors.status, "success");
    assert_eq!(llama_cpp.vectors.backend, "ggml-cpu-reference");
    assert_eq!(llama_cpp.vectors.thread_count, 1);
    assert_eq!(llama_cpp.vectors.tensor_layout, "flat GGML ne[0]-fastest");
    assert!(fixture.transcription_provenance.iter().all(|entry| {
        !entry.cases.is_empty() && entry.sources.iter().all(|source| source.contains(':'))
    }));
    assert!(
        fixture
            .known_reference_differences
            .fp4_tiny_scale_and_midpoints
            .contains("SGLang")
    );
}

#[test]
#[ignore = "requires uv, CMake, a C++ compiler, and pinned llama.cpp/DwarfStar checkouts"]
fn fixture_regeneration_has_no_drift() {
    let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let status = std::process::Command::new("uv")
        .args([
            "run",
            "scripts/reference/generate_dsv4_oracle.py",
            "--check",
        ])
        .current_dir(repository)
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
fn direct_dwarfstar_scalar_helpers_match() {
    let fixture = fixture().dwarfstar_direct.vectors;

    let mixes = (0..24)
        .map(|index| (index as f32 - 11.5) * 0.13)
        .collect::<Vec<_>>();
    let base = (0..24)
        .map(|index| ((index * 7 % 13) as f32 - 6.0) * 0.04)
        .collect::<Vec<_>>();
    let controls = split_sinkhorn(&mixes, &[0.7, -0.4, 1.2], &base, 4, 20, 1e-6).unwrap();
    let mut split = controls.pre.clone();
    split.extend_from_slice(&controls.post);
    split.extend_from_slice(&controls.combination);
    assert_close(&split, &fixture.sinkhorn, 3e-6);

    let residual = (0..12)
        .map(|index| (index as f32 - 5.0) * 0.2)
        .collect::<Vec<_>>();
    let post = hyper_connection_post(
        &[0.4, -0.7, 1.1],
        &residual,
        &HyperConnectionControls {
            pre: controls.pre,
            post: controls.post,
            combination: controls.combination,
        },
        3,
        4,
    )
    .unwrap();
    assert_close(&post, &fixture.hc_post, 3e-6);

    let rope_input = [9.0, 8.0, 7.0, 6.0, 1.0, -2.0, 3.0, -4.0];
    let mut local = rope_input;
    rope_tail_in_place(
        &mut local,
        1,
        8,
        17,
        RopeParameters::local(4, 10_000.0),
        RopeDirection::Forward,
    )
    .unwrap();
    assert_close(&local, &fixture.rope_local, 3e-6);
    let mut yarn = rope_input;
    rope_tail_in_place(
        &mut yarn,
        1,
        8,
        65_536,
        RopeParameters::yarn(4, 160_000.0, 16.0, 65_536, 32.0, 1.0),
        RopeDirection::Forward,
    )
    .unwrap();
    assert_close(&yarn, &fixture.rope_yarn, 3e-5);
    let mut yarn_inverse = rope_input;
    rope_tail_in_place(
        &mut yarn_inverse,
        1,
        8,
        65_536,
        RopeParameters::yarn(4, 160_000.0, 16.0, 65_536, 32.0, 1.0),
        RopeDirection::Inverse,
    )
    .unwrap();
    assert_close(&yarn_inverse, &fixture.rope_yarn_inverse, 3e-5);

    let ratio4_kv = (0..8)
        .flat_map(|row| {
            (0..8).map(move |column| ((row * 11 + column * 5) % 19) as f32 * 0.17 - 1.53)
        })
        .collect::<Vec<_>>();
    let ratio4_scores = (0..8)
        .flat_map(|row| {
            (0..8).map(move |column| ((row * 7 + column * 3) % 13) as f32 * 0.09 - 0.54)
        })
        .collect::<Vec<_>>();
    assert_close(
        &compressor_pool(&ratio4_kv, &ratio4_scores, 4, 4).unwrap(),
        &fixture.ratio4_pool,
        3e-6,
    );

    let mut ratio128_kv = Vec::with_capacity(256);
    let mut ratio128_scores = Vec::with_capacity(256);
    for row in 0..128 {
        ratio128_kv.push((row as f32 * 0.07).sin());
        ratio128_kv.push((row as f32 * 0.11).cos());
        ratio128_scores.push(((row * 3) % 17) as f32 * 0.08 - 0.64);
        ratio128_scores.push(((row * 7) % 19) as f32 * 0.06 - 0.54);
    }
    assert_close(
        &compressor_pool(&ratio128_kv, &ratio128_scores, 2, 128).unwrap(),
        &fixture.ratio128_pool,
        4e-6,
    );

    let mut qat = (0..128)
        .map(|index| (index as f32 * 0.17).sin() + (index as f32 * 0.031).cos() * 0.25)
        .collect::<Vec<_>>();
    indexer_qat_roundtrip_in_place(&mut qat).unwrap();
    assert_close(&qat, &fixture.indexer_qat, 4e-6);

    let logits = [-4.0, -0.5, 0.0, 1.25, 3.0, 0.0, -2.5, 2.0];
    let scores = sqrt_softplus_scores(&logits).unwrap();
    assert_close(&scores, &fixture.router_scores, 3e-7);
    let routed = learned_route(
        &scores,
        &[0.0, 0.25, 0.0, -0.1, -3.0, 0.0, 2.5, 0.2],
        3,
        1.5,
    )
    .unwrap();
    assert_eq!(routed.expert_ids, fixture.router_selected);
    assert_close(&routed.weights, &fixture.router_weights, 3e-7);
    let swiglu =
        clamped_swiglu(&[-20.0, -0.5, 2.0, 20.0], &[-20.0, 0.25, -3.0, 20.0], 10.0).unwrap();
    assert_close(&swiglu, &fixture.swiglu, 3e-6);
}

#[test]
fn direct_dwarfstar_compressor_transitions_match() {
    let transitions = fixture().dwarfstar_direct.vectors.compressor_transitions;
    assert_eq!(
        transitions.hash_algorithm,
        "fnv1a64-f32-le; signed zero and score sentinel canonicalized"
    );
    assert_eq!(
        transitions.recipes.input,
        "[-4,-3,-2,-1,1,2,3,4][(seed*31+position*5+column*7)%8]/4"
    );
    assert_eq!(
        transitions.recipes.kv_weight,
        "value=seed*17+row*5+column*13; (value%2?-1:1)*(500+((value>>1)%1001))/1000"
    );
    assert_eq!(
        transitions.recipes.gate_weight,
        "value=seed*23+row*11+column*19; (value%2?-1:1)*(250+((value>>1)%751))/997"
    );
    assert_eq!(
        transitions.recipes.ape,
        "((seed*29+position*13+row*3)%257-128)/1000"
    );
    assert_eq!(transitions.recipes.norm, "3/4+((seed*5+row*7)%101)/1000");
    assert_eq!(transitions.cases.len(), 3);

    let rope = RopeParameters::yarn(
        transitions.rotary_dim,
        transitions.rope_theta,
        transitions.rope_scale_factor,
        transitions.original_context_length,
        transitions.beta_fast,
        transitions.beta_slow,
    );
    for case in transitions.cases {
        let expected = match case.name.as_str() {
            "ratio4_attention" => (1, 512, 4, 2, 9),
            "ratio4_indexer" => (2, 128, 4, 2, 9),
            "ratio128_attention" => (3, 512, 128, 3, 257),
            name => panic!("unexpected DwarfStar compressor case {name}"),
        };
        assert_eq!(
            (
                case.seed,
                case.head_dim,
                case.ratio,
                case.layer,
                case.positions
            ),
            expected
        );
        assert_eq!(case.input_dim, 17);
        assert_eq!(case.projection_type, "f16");
        let coefficient = if case.ratio == 4 { 2 } else { 1 };
        assert_eq!(case.state_rows, coefficient * case.ratio);
        assert_eq!(case.state_width, coefficient * case.head_dim);
        assert_eq!(case.records.len(), case.positions);

        let mut state = CompressorState::new(case.ratio, case.head_dim).unwrap();
        let ape = (0..case.ratio)
            .flat_map(|position| {
                (0..case.state_width)
                    .map(move |row| f16_roundtrip(compressor_ape_value(case.seed, position, row)))
            })
            .collect::<Vec<_>>();
        let norm = (0..case.head_dim)
            .map(|row| compressor_norm_value(case.seed, row))
            .collect::<Vec<_>>();

        for record in case.records {
            assert_eq!(record.position as usize, state.next_position() as usize);
            let input = (0..case.input_dim)
                .map(|column| compressor_input_value(case.seed, record.position, column))
                .collect::<Vec<_>>();
            let projected_kv = (0..case.state_width)
                .map(|row| {
                    input
                        .iter()
                        .enumerate()
                        .map(|(column, &input)| {
                            f16_roundtrip(compressor_kv_weight(case.seed, row, column)) * input
                        })
                        .sum::<f32>()
                })
                .collect::<Vec<_>>();
            let projected_scores = (0..case.state_width)
                .map(|row| {
                    input
                        .iter()
                        .enumerate()
                        .map(|(column, &input)| {
                            f16_roundtrip(compressor_gate_weight(case.seed, row, column)) * input
                        })
                        .sum::<f32>()
                })
                .collect::<Vec<_>>();
            let emitted = state
                .push_projected(
                    record.position,
                    &projected_kv,
                    &projected_scores,
                    &ape,
                    &norm,
                    transitions.rms_epsilon,
                    rope,
                )
                .unwrap();
            assert_eq!(f32_state_hash(state.kv_state()), record.kv_hash);
            assert_eq!(f32_state_hash(state.score_state()), record.score_hash);
            assert_eq!(emitted.is_some(), record.emitted);

            match (emitted, record.start_position, record.output) {
                (Some(mut emitted), Some(start_position), Some(expected_output)) => {
                    assert_eq!(emitted.start_position, start_position);
                    if case.head_dim == 512 {
                        attention_fp8_nope_roundtrip_in_place(
                            &mut emitted.value,
                            transitions.rotary_dim,
                        )
                        .unwrap();
                    } else {
                        indexer_qat_roundtrip_in_place(&mut emitted.value).unwrap();
                    }
                    assert_close(&emitted.value, &expected_output, 6e-5);
                }
                (None, None, None) => {}
                _ => panic!(
                    "incoherent DwarfStar compressor emission at position {}",
                    record.position
                ),
            }
        }
    }
}

#[test]
fn direct_llama_cpp_cpu_mhc_primitives_match() {
    let vectors = fixture().llama_cpp_cpu.vectors;
    let combination = vectors.hc_comb;
    let parameter_count = combination.connection_count * (combination.connection_count + 2);
    let matrix_size = combination.connection_count * combination.connection_count;
    assert_eq!(
        combination.mixes_shape,
        vec![parameter_count, combination.token_count]
    );
    assert_eq!(combination.scale_shape, vec![3]);
    assert_eq!(combination.base_shape, vec![parameter_count]);
    assert_eq!(
        combination.output_shape,
        vec![
            combination.connection_count,
            combination.connection_count,
            combination.token_count,
        ]
    );
    for token in 0..combination.token_count {
        let mixes = &combination.mixes[token * parameter_count..(token + 1) * parameter_count];
        let expected = &combination.output[token * matrix_size..(token + 1) * matrix_size];
        let actual = split_sinkhorn(
            mixes,
            &combination.scale,
            &combination.base,
            combination.connection_count,
            combination.iterations,
            combination.epsilon,
        )
        .unwrap();
        assert_close(&actual.combination, expected, 4e-6);
    }

    let pre = vectors.hc_pre;
    let residual_size = pre.hidden_size * pre.connection_count;
    assert_eq!(
        pre.residual_shape,
        vec![pre.hidden_size, pre.connection_count, pre.token_count]
    );
    assert_eq!(
        pre.weights_shape,
        vec![pre.connection_count, pre.token_count]
    );
    assert_eq!(pre.output_shape, vec![pre.hidden_size, pre.token_count]);
    for token in 0..pre.token_count {
        let residual = &pre.residual[token * residual_size..(token + 1) * residual_size];
        let weights =
            &pre.weights[token * pre.connection_count..(token + 1) * pre.connection_count];
        let expected = &pre.output[token * pre.hidden_size..(token + 1) * pre.hidden_size];
        let actual =
            hyper_connection_collapse(residual, weights, pre.hidden_size, pre.connection_count)
                .unwrap();
        assert_close(&actual, expected, 3e-7);
    }

    let post = vectors.hc_post;
    let residual_size = post.hidden_size * post.connection_count;
    let matrix_size = post.connection_count * post.connection_count;
    assert_eq!(
        post.block_output_shape,
        vec![post.hidden_size, post.token_count]
    );
    assert_eq!(
        post.residual_shape,
        vec![post.hidden_size, post.connection_count, post.token_count]
    );
    assert_eq!(
        post.weights_shape,
        vec![post.connection_count, post.token_count]
    );
    assert_eq!(
        post.combination_shape,
        vec![
            post.connection_count,
            post.connection_count,
            post.token_count
        ]
    );
    assert_eq!(
        post.output_shape,
        vec![post.hidden_size, post.connection_count, post.token_count]
    );
    assert_eq!(post.residual, pre.residual);
    assert_eq!(post.combination, combination.output);
    for token in 0..post.token_count {
        let block_output =
            &post.block_output[token * post.hidden_size..(token + 1) * post.hidden_size];
        let residual = &post.residual[token * residual_size..(token + 1) * residual_size];
        let controls = HyperConnectionControls {
            pre: Vec::new(),
            post: post.weights[token * post.connection_count..(token + 1) * post.connection_count]
                .to_vec(),
            combination: post.combination[token * matrix_size..(token + 1) * matrix_size].to_vec(),
        };
        let expected = &post.output[token * residual_size..(token + 1) * residual_size];
        let actual = hyper_connection_post(
            block_output,
            residual,
            &controls,
            post.hidden_size,
            post.connection_count,
        )
        .unwrap();
        assert_close(&actual, expected, 4e-6);
    }
}

#[test]
fn mhc_matches_reference_fixture() {
    let fixture = fixture().mhc;
    let pre = hyper_connection_pre(
        &fixture.residual,
        fixture.hidden_size,
        fixture.connection_count,
        &fixture.function,
        &fixture.scale,
        &fixture.base,
        fixture.rms_eps,
        fixture.sinkhorn_iterations,
        fixture.hc_eps,
    )
    .unwrap();
    assert_close(&pre.mixes, &fixture.expected_mixes, 3e-6);
    assert_close(&pre.controls.pre, &fixture.expected_pre, 3e-6);
    assert_close(&pre.controls.post, &fixture.expected_post, 3e-6);
    assert_close(
        &pre.controls.combination,
        &fixture.expected_combination,
        3e-6,
    );
    assert_close(&pre.input, &fixture.expected_input, 3e-6);

    let post = hyper_connection_post(
        &fixture.block_output,
        &fixture.residual,
        &pre.controls,
        fixture.hidden_size,
        fixture.connection_count,
    )
    .unwrap();
    assert_close(&post, &fixture.expected_post_output, 3e-6);

    let head = hyper_connection_head(
        &fixture.residual,
        fixture.hidden_size,
        fixture.connection_count,
        &fixture.head_function,
        fixture.head_scale,
        &fixture.head_base,
        fixture.rms_eps,
        fixture.hc_eps,
    )
    .unwrap();
    assert_close(&head, &fixture.expected_head_output, 3e-6);
}

#[test]
fn rope_matches_reference_fixture() {
    let fixture = fixture().rope;
    let mut local = fixture.input.clone();
    rope_tail_in_place(
        &mut local,
        fixture.head_count,
        fixture.head_dim,
        fixture.local_position,
        RopeParameters::local(fixture.rotary_dim, 10_000.0),
        RopeDirection::Forward,
    )
    .unwrap();
    assert_close(&local, &fixture.expected_local, 3e-6);

    let mut yarn = fixture.input;
    rope_tail_in_place(
        &mut yarn,
        fixture.head_count,
        fixture.head_dim,
        fixture.yarn_position,
        RopeParameters::yarn(fixture.rotary_dim, 160_000.0, 16.0, 65_536, 32.0, 1.0),
        RopeDirection::Forward,
    )
    .unwrap();
    assert_close(&yarn, &fixture.expected_yarn, 3e-5);
}

#[test]
fn shared_kv_attention_and_output_match_reference_fixture() {
    let fixture = fixture();
    let projection = fixture.shared_kv_projection;
    assert_eq!(projection.input.len(), projection.hidden_size);
    let projected = shared_kv_projection(
        &projection.input,
        &projection.q_a,
        &projection.q_a_norm,
        &projection.q_b,
        &projection.kv_weight,
        &projection.kv_norm,
        projection.q_lora_rank,
        projection.head_count,
        projection.head_dim,
        1e-6,
    )
    .unwrap();
    assert_close(&projected.q_lora_raw, &projection.expected_q_lora_raw, 3e-6);
    assert_close(&projected.q_lora, &projection.expected_q_lora, 3e-6);
    assert_close(&projected.queries, &projection.expected_queries, 3e-6);
    assert_close(&projected.kv_raw, &projection.expected_kv_raw, 3e-6);
    assert_close(&projected.kv, &projection.expected_kv, 3e-6);

    let attention = fixture.attention_output;
    let output = shared_kv_attention(
        &attention.queries,
        attention.head_count,
        attention.head_dim,
        &attention.raw_kv,
        &attention.compressed_kv,
        Some(&attention.compressed_allowed),
        &attention.sinks,
    )
    .unwrap();
    assert_close(&output, &attention.expected_attention, 3e-6);
    let mut inverse = output;
    rope_tail_in_place(
        &mut inverse,
        attention.head_count,
        attention.head_dim,
        attention.inverse_position,
        RopeParameters::yarn(4, 160_000.0, 16.0, 65_536, 32.0, 1.0),
        RopeDirection::Inverse,
    )
    .unwrap();
    assert_close(&inverse, &attention.expected_inverse, 3e-5);
    let low_rank = grouped_low_rank_projection(
        &inverse,
        attention.head_count,
        attention.head_dim,
        attention.group_count,
        attention.rank,
        &attention.output_a,
    )
    .unwrap();
    assert_close(&low_rank, &attention.expected_low_rank, 4e-6);
    let projected_output = grouped_low_rank_output(
        &inverse,
        attention.head_count,
        attention.head_dim,
        attention.group_count,
        attention.rank,
        &attention.output_a,
        &attention.output_b,
        attention.hidden_size,
    )
    .unwrap();
    assert_close(&projected_output, &attention.expected_output, 4e-6);
}

#[test]
fn cache_roundtrips_match_reference_fixture() {
    let fixture = fixture().cache_roundtrip;
    let mut attention = fixture.input;
    attention_fp8_nope_bf16_rope_roundtrip_in_place(&mut attention, fixture.rotary_dim).unwrap();
    assert_close(&attention, &fixture.expected_attention_cache, 1e-7);

    let mut bf16 = fixture.bf16_input;
    bf16_roundtrip_in_place(&mut bf16).unwrap();
    assert_eq!(bf16, fixture.expected_bf16);
}

#[test]
fn ratio4_state_matches_after_every_token() {
    let fixture = fixture().compressor_ratio4;
    let mut state = CompressorState::new(4, fixture.head_dim).unwrap();
    let mut restored: Option<CompressorState> = None;
    let mut emitted_index = 0;
    for position in 0..fixture.projected_kv.len() {
        let emitted = state
            .push_projected(
                position as u32,
                &fixture.projected_kv[position],
                &fixture.projected_scores[position],
                &fixture.ape,
                &fixture.norm_weight,
                1e-6,
                fixture.rope.into(),
            )
            .unwrap();
        if let Some(restored) = &mut restored {
            let restored_emitted = restored
                .push_projected(
                    position as u32,
                    &fixture.projected_kv[position],
                    &fixture.projected_scores[position],
                    &fixture.ape,
                    &fixture.norm_weight,
                    1e-6,
                    fixture.rope.into(),
                )
                .unwrap();
            assert_eq!(restored_emitted, emitted);
            assert_eq!(restored, &state);
        }
        let snapshot = &fixture.snapshots[position];
        assert_eq!(snapshot.position, position as u32);
        assert_close(state.kv_state(), &snapshot.kv, 1e-7);
        assert_snapshot(state.score_state(), &snapshot.scores, 1e-7);
        if let Some(emitted) = emitted {
            let expected = &fixture.emitted[emitted_index];
            assert_eq!(expected.position, position as u32);
            assert_close(&emitted.value, &expected.value, 4e-6);
            emitted_index += 1;
        }
        if position == 3 {
            restored = Some(
                CompressorState::from_snapshot(
                    4,
                    fixture.head_dim,
                    state.next_position(),
                    state.kv_state().to_vec(),
                    state.score_state().to_vec(),
                )
                .unwrap(),
            );
        }
    }
    assert_eq!(emitted_index, fixture.emitted.len());
}

#[test]
fn ratio128_boundary_matches_reference_fixture() {
    let fixture = fixture().compressor_ratio128;
    let mut state = CompressorState::new(128, fixture.head_dim).unwrap();
    let mut restored: Option<CompressorState> = None;
    let mut emitted_index = 0;
    for position in 0..fixture.projected_kv.len() {
        let emitted = state
            .push_projected(
                position as u32,
                &fixture.projected_kv[position],
                &fixture.projected_scores[position],
                &fixture.ape,
                &fixture.norm_weight,
                1e-6,
                fixture.rope.into(),
            )
            .unwrap();
        if let Some(restored) = &mut restored {
            let restored_emitted = restored
                .push_projected(
                    position as u32,
                    &fixture.projected_kv[position],
                    &fixture.projected_scores[position],
                    &fixture.ape,
                    &fixture.norm_weight,
                    1e-6,
                    fixture.rope.into(),
                )
                .unwrap();
            assert_eq!(restored_emitted, emitted);
            assert_eq!(restored, &state);
        }
        if let Some(snapshot) = fixture
            .snapshots
            .iter()
            .find(|snapshot| snapshot.position == position as u32)
        {
            assert_close(state.kv_state(), &snapshot.kv, 1e-7);
            assert_snapshot(state.score_state(), &snapshot.scores, 1e-7);
        }
        if let Some(emitted) = emitted {
            let expected = &fixture.emitted[emitted_index];
            assert_eq!(expected.position, position as u32);
            assert_eq!(
                emitted.start_position,
                position as u32 + 1 - state.ratio() as u32
            );
            assert_close(&emitted.value, &expected.value, 6e-6);
            emitted_index += 1;
        }
        if position == 128 {
            restored = Some(
                CompressorState::from_snapshot(
                    128,
                    fixture.head_dim,
                    state.next_position(),
                    state.kv_state().to_vec(),
                    state.score_state().to_vec(),
                )
                .unwrap(),
            );
        }
    }
    assert_eq!(emitted_index, fixture.emitted.len());
    assert_close(state.kv_state(), &fixture.expected_kv_state, 2e-7);
    assert_close(state.score_state(), &fixture.expected_score_state, 2e-7);
}

#[test]
fn indexer_and_routing_match_reference_fixture() {
    let fixture = fixture();
    let indexer = fixture.indexer;
    let mut qat = indexer.qat_input;
    indexer_qat_roundtrip_in_place(&mut qat).unwrap();
    assert_close(&qat, &indexer.expected_qat, 3e-6);
    let scores = indexer_scores(
        &indexer.queries,
        &indexer.head_weights,
        &indexer.compressed_keys,
        indexer.head_count,
        indexer.head_dim,
    )
    .unwrap();
    assert_close(&scores, &indexer.expected_scores, 2e-6);
    assert_eq!(
        top_k_indices(&scores, indexer.top_k).unwrap(),
        indexer.expected_top_k
    );

    let routing = fixture.routing;
    let router_scores = sqrt_softplus_scores(&routing.logits).unwrap();
    assert_close(&router_scores, &routing.expected_scores, 2e-7);
    let learned = learned_route(
        &router_scores,
        &routing.bias,
        routing.top_k,
        routing.routed_scale,
    )
    .unwrap();
    assert_eq!(learned.expert_ids, routing.expected_experts);
    assert_close(&learned.weights, &routing.expected_weights, 3e-7);
    let hashed = hash_route(&router_scores, &routing.hash_experts, routing.routed_scale).unwrap();
    assert_eq!(hashed.expert_ids, routing.hash_experts);
    assert_close(&hashed.weights, &routing.expected_hash_weights, 3e-7);
}
