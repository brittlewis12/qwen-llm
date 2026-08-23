//! Feature-gated, single-block DFlash K0-S evidence producer.

use anyhow::{Context, Result, anyhow, ensure};
use clap::Parser;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily};
use qwen_llm::{
    gguf::GgufFile,
    loader::{Model, open_dflash_drafter},
    metal::{
        DispatchCensusRow, KernelTraceCounters, KernelTraceGuard, MetalContext, MetalTensor,
        diagnostics_observer_active_counts, dispatch_census_begin, dispatch_census_take,
        kernel_trace_begin, kernel_trace_snapshot,
    },
    metal_dflash::{
        DFLASH_K0S_BLOCK_SIZE, DFLASH_K0S_HIDDEN, DFLASH_K0S_LATTICE_ROWS, DFLASH_K0S_RANK,
        DFLASH_K0S_TOP_K, DFLASH_K0S_VOCAB, DFlashDecoder, DFlashK0sCapture, DFlashK0sChain,
        DFlashK0sChainEvent, DFlashK0sCodebookSide, DFlashK0sDispatchCensusRow,
        DFlashK0sNonFiniteClass, DFlashK0sParitySummary, DFlashK0sRawRow, DFlashK0sSlotIssue,
        DFlashK0sTopKIssue, MetalDFlashHead, MetalDFlashLayerMajorScratch, MetalDFlashSession,
        dflash_k0s_embedded_metallib_bytes, dflash_k0s_embedded_metallib_identity,
        dflash_k0s_event_envelope_sha256, dflash_k0s_scalar_contract_fixture,
        dflash_k0s_traverse_slots, prefill_tokens_with_multi_hidden,
    },
    metal_forward::{MetalForward, MetalModel, MetalSession, SessionSnapshot},
    tokenizer::{NativeTokenizer, Tokenizer, token_ids_sha256_i32le},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ffi::OsString,
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    os::unix::{
        ffi::OsStrExt,
        fs::{MetadataExt, OpenOptionsExt},
    },
    path::{Component, Path, PathBuf},
    process::Command,
};

const SCHEMA: &str = "qwen.dflash_k0s_lattice";
const SCHEMA_VERSION: u32 = 1;
const MANIFEST_SCHEMA: &str = "qwen.dflash_k0s_external_manifest";
const AUTHORITY: &str = "development_k0s_conditional_on_authenticated_z_only_no_projection_parity_no_rng_acceptance_k0l_e1b_verifier_product_authority";
const CAPTURE_DEFINITION: &str = "qwen.dflash_k0s.capture.v1";
const SCALAR_DOMAIN: &str = "qwen.dflash_k0s.scalar_contract_fixture.v2";
const SCALAR_DIGEST: &str = "8c22bf3b4ee51efaf315137a866feef8fc8019d5b21c887411b532bd79fa60e9";
const MAX_TRACE: u64 = 64 << 20;
const MAX_SIDECAR: u64 = 64 << 20;
const MAX_COMBINED: u64 = 128 << 20;
const RECORDS: usize = 16;
const READ_CHUNK: usize = 1 << 20;
const COMMAND_MANIFEST_PLACEHOLDER: &str = "${MANIFEST_SHA256}";
const MAX_FIXED_CHAINS: usize = 64;
const MAX_DISPATCH_ROWS: usize = 256;
const INVENTORY_SCHEMA: &str = "qwen.dflash_k0s_inventory";
const INVENTORY_SPEC_SCHEMA: &str = "qwen.dflash_k0s_inventory_spec";
const INVENTORY_AUTHORITY: &str =
    "development_k0s_inventory_only_no_model_forward_or_semantic_authority";
const MAX_GGUF_TENSORS: u64 = 8192;
const MAX_GGUF_METADATA: u64 = 4096;
const MAX_GGUF_STRINGS_BYTES: u64 = 16 << 20;
const MAX_GGUF_ARRAY_ITEMS: u64 = 500_000;
const MAX_GGUF_OBJECTS: u64 = 600_000;
const MAX_GGUF_HEADER_BYTES: u64 = 64 << 20;
const K0S_SELECTOR_WEIGHT_DTYPE: &str = "Q4_K";
const K0S_SELECTOR_WEIGHT_DTYPE_ID: i32 = 12;
const K0S_SELECTOR_KERNEL: &str = "kernel_mat_mat_q4_K_f32";
const REQUIRED_SOURCE_ROLES: [&str; 11] = [
    "metal_dflash_rs",
    "bench_rs",
    "dflash_k0s_rs",
    "qwen_llm_cargo_toml",
    "qwen_cli_cargo_toml",
    "metal_rs",
    "metal_forward_rs",
    "dflash2_metal",
    "mat_mat_mma8_metal",
    "mat_mat_q4_k_metal",
    "build_rs",
];

#[derive(Parser, Debug)]
pub struct DflashK0sArgs {
    /// Prospective acquisition attempt identifier.
    #[arg(long)]
    attempt_id: String,
    /// Single-file target GGUF fixed by the external manifest.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Single-file Q4 DFlash 2 drafter GGUF fixed by the external manifest.
    #[arg(long)]
    drafter: PathBuf,
    /// Fixed prompt; the authenticated command manifest binds its exact bytes.
    #[arg(short = 'p', long)]
    prompt: String,
    /// Fixed carry token; this is never sampled from prompt logits.
    #[arg(long)]
    carry_token: i32,
    /// Preregistered literal carry for the continuation draft.
    #[arg(long)]
    continuation_carry_token: i32,
    /// Static prospective K0-S manifest.
    #[arg(long)]
    manifest: PathBuf,
    /// Independently supplied SHA-256 of the exact static manifest bytes.
    #[arg(long)]
    manifest_sha256: String,
    /// Authenticated command manifest containing the exact argv array.
    #[arg(long)]
    command_manifest: PathBuf,
    /// Authenticated compiled scalar fixture JSON.
    #[arg(long)]
    fixture: PathBuf,
    /// Request temperature; no separate proposal temperature exists.
    #[arg(long)]
    temperature: f32,
    /// NAME:CARRY:S0,S1,S2,S3,S4,S5,S6. Repeat for fixed traversals.
    #[arg(long, required = true)]
    fixed_chain: Vec<String>,
    /// Exclusive-create 16-record JSONL output.
    #[arg(long)]
    trace_output: PathBuf,
    /// Exclusive-create raw-row sidecar output.
    #[arg(long)]
    sidecar_output: PathBuf,
}

#[derive(Parser, Debug)]
pub struct DflashK0sInventoryArgs {
    #[arg(short = 'm', long)]
    model: PathBuf,
    #[arg(long)]
    drafter: PathBuf,
    #[arg(short = 'p', long)]
    prompt: String,
    #[arg(long)]
    carry_token: i32,
    #[arg(long)]
    inventory_spec: PathBuf,
    #[arg(long)]
    inventory_spec_sha256: String,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct FileClaim {
    path: String,
    bytes: u64,
    sha256: String,
    max_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RoleClaim {
    role: String,
    #[serde(flatten)]
    file: FileClaim,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct IgnoredPolicy {
    top_k: u64,
    top_p_f32_bits: String,
    min_p_f32_bits: String,
    grammar: Option<Value>,
    penalties: Option<Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct IgnoredPolicies {
    variant_a: IgnoredPolicy,
    variant_b: IgnoredPolicy,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RequestClaim {
    temperature_f32_bits: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ExpectedRequest {
    request: RequestClaim,
    ignored_target_policy: IgnoredPolicies,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Binding {
    production_call_id: String,
    drafter_checkpoint_sha256: String,
    proposal_construction_id: String,
    noise_input_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StaticChain {
    name: String,
    initial_carry: i32,
    slots: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CaptureContext {
    definition_version: String,
    carry_token: i32,
    noise_start_position: u32,
    target_context_len: u64,
    context_hidden_watermark: u64,
    kv_context_watermark: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct BuildClaim {
    commit: String,
    source_sha256: String,
    dirty: bool,
    compiler: String,
    compiler_version: String,
    target: String,
    profile: String,
    features: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct HostClaim {
    os: String,
    arch: String,
    device_name: String,
    device_registry_id: u64,
    device_family: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CheckoutClaim {
    path: String,
    commit: String,
    tree: String,
    dirty: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct DispatchClaim {
    family: String,
    tag: Option<String>,
    encoder_ordinal: u64,
    encoder_concurrent: bool,
    kernel: String,
    grid: [u64; 3],
    threads: [u64; 3],
    grid_threadgroups: u64,
    threadgroup_threads: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SelectorDispatchPredicate {
    tag: String,
    kernel: String,
    weight_dtype: String,
    input_dtype: String,
    output_dtype: String,
    weight_dtype_id: i32,
    input_dtype_id: i32,
    output_dtype_id: i32,
    n: u64,
    h: u64,
    r: u64,
    grid: [u64; 3],
    threads: [u64; 3],
    metal_source_sha256: String,
    metallib_sha256: String,
    build_source_sha256: String,
    allowed_environment: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PreparationBinding {
    inventory: FileClaim,
    inventory_spec: FileClaim,
    preparation_spec: FileClaim,
    seal_path: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TensorClaim {
    role: String,
    asset_role: String,
    name: String,
    dtype: String,
    shape: Vec<u64>,
    offset: u64,
    bytes: u64,
    sha256: String,
    orientation: String,
    row_domain: Option<RowDomain>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RowDomain {
    first: u64,
    count: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SemanticReference {
    commit: String,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SemanticReferences {
    mlx_model_mlx_py: SemanticReference,
    vllm_qwen3_dflash2_py: SemanticReference,
    vllm_speculator_py: SemanticReference,
    llama_cpp_dflash_cpp: SemanticReference,
    llama_cpp_speculative_cpp: SemanticReference,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ScalarVector {
    name: String,
    a_f32_bits: Vec<String>,
    z_f32_bits: Vec<String>,
    successor_f32_bits: Vec<String>,
    unary_f32_bits: String,
    score_f32_bits: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ScalarContract {
    artifact: FileClaim,
    compiler: String,
    compiler_version: String,
    target: String,
    profile: String,
    fixture_domain: String,
    fixture_sha256: String,
    vectors: Vec<ScalarVector>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    schema_version: u32,
    run_id: String,
    attempt_id: String,
    trace_max_bytes: u64,
    sidecar_max_bytes: u64,
    reducer: FileClaim,
    executable: FileClaim,
    fixture: FileClaim,
    command: FileClaim,
    sources: Vec<RoleClaim>,
    assets: Vec<RoleClaim>,
    semantic_references: SemanticReferences,
    tensors: Vec<TensorClaim>,
    expected_request: ExpectedRequest,
    expected_prompt: InventoryPrompt,
    expected_binding: Binding,
    expected_continuation_carry_token: i32,
    expected_rng_domains: Vec<String>,
    expected_fixed_chains: Vec<StaticChain>,
    expected_capture_context: CaptureContext,
    expected_build: BuildClaim,
    expected_host: HostClaim,
    embedded_metallib_sha256: String,
    selector_dispatch_predicate: SelectorDispatchPredicate,
    preparation_binding: PreparationBinding,
    scalar_contract: ScalarContract,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CommandManifest {
    argv: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScalarFixtureFile {
    schema: String,
    schema_version: u32,
    fixture_domain: String,
    fixture_sha256: String,
    vectors: Vec<ScalarVector>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TensorRequirement {
    role: String,
    asset_role: String,
    name: String,
    dtype: String,
    shape: Vec<u64>,
    orientation: String,
    row_domain: Option<RowDomain>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct InventoryTokenizer {
    vocab_size: u64,
    token_embd_name: String,
    token_embd_shape: Vec<u64>,
    token_embd_dtype: String,
    token_count: u64,
    tokenizer_tokens_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct InventoryPrompt {
    utf8_hex: String,
    token_ids: Vec<i32>,
    token_ids_sha256_i32le: String,
    tokenizer_identity_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ParserCaps {
    header_bytes: u64,
    metadata: u64,
    tensors: u64,
    strings_bytes: u64,
    array_items: u64,
    objects: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct HostPredicate {
    os: String,
    arch: String,
    device_name: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InventorySpec {
    schema: String,
    schema_version: u32,
    run_id: String,
    inventory_max_bytes: u64,
    checkout: CheckoutClaim,
    build: BuildClaim,
    sources: Vec<RoleClaim>,
    executable: FileClaim,
    reducer: FileClaim,
    scalar_fixture: FileClaim,
    command_template: FileClaim,
    embedded_metallib: FileClaim,
    assets: Vec<RoleClaim>,
    tensor_requirements: Vec<TensorRequirement>,
    tokenizer: InventoryTokenizer,
    prompt: InventoryPrompt,
    carry_token: i32,
    expected_mask_token: i32,
    parser_caps: ParserCaps,
    host_predicate: HostPredicate,
    command: Vec<String>,
    environment: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct InventoryGguf {
    role: String,
    version: u32,
    tensor_count: u64,
    metadata_count: u64,
}

#[derive(Serialize)]
struct InventoryMaskNoise {
    metadata_key: String,
    mask_token: i32,
    noise_tokens: Vec<i32>,
    noise_sha256_i32le: String,
}

#[derive(Serialize)]
struct InventoryArtifact {
    schema: &'static str,
    schema_version: u32,
    authority: &'static str,
    inventory_spec_sha256: String,
    run_id: String,
    checkout: CheckoutClaim,
    build: BuildClaim,
    sources: Vec<RoleClaim>,
    executable: FileClaim,
    reducer: FileClaim,
    scalar_fixture: FileClaim,
    command_template: FileClaim,
    embedded_metallib: FileClaim,
    device: HostClaim,
    assets: Vec<RoleClaim>,
    gguf: Vec<InventoryGguf>,
    tensors: Vec<TensorClaim>,
    tokenizer: InventoryTokenizer,
    prompt: InventoryPrompt,
    mask_noise: InventoryMaskNoise,
    parser_caps: ParserCaps,
    command: Vec<String>,
    environment: BTreeMap<String, String>,
}

#[derive(Debug)]
struct InventoryGgufScan {
    version: u32,
    tensor_count: u64,
    metadata_count: u64,
    tokenizer_count: Option<u64>,
    tokenizer_sha256: Option<String>,
    masks: Vec<(String, u64)>,
}

struct OpenedInput {
    path: PathBuf,
    file: File,
    bytes: u64,
    device: u64,
    inode: u64,
    mtime: i64,
    mtime_nsec: i64,
    ctime: i64,
    ctime_nsec: i64,
    sha256: String,
}

fn hash_file(file: &File) -> Result<String> {
    let mut reader = file.try_clone()?;
    reader.rewind()?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; READ_CHUNK];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex(&digest.finalize()))
}

struct GgufCursor {
    file: File,
    size: u64,
    strings: u64,
    arrays: u64,
    objects: u64,
}

impl GgufCursor {
    fn take(&mut self, count: usize) -> Result<Vec<u8>> {
        let offset = self.file.stream_position()?;
        let count_u64 = u64::try_from(count)?;
        ensure!(
            offset <= self.size && count_u64 <= self.size - offset,
            "truncated GGUF"
        );
        let mut bytes = vec![0; count];
        self.file.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string_bytes(&mut self) -> Result<Vec<u8>> {
        let count = self.u64()?;
        ensure!(count <= 1 << 20, "GGUF string exceeds per-string cap");
        self.strings = self
            .strings
            .checked_add(count)
            .context("GGUF string budget overflow")?;
        ensure!(
            self.strings <= MAX_GGUF_STRINGS_BYTES,
            "GGUF cumulative string budget exceeded"
        );
        self.take(usize::try_from(count)?)
    }

    fn string(&mut self) -> Result<String> {
        String::from_utf8(self.string_bytes()?).context("GGUF string is not UTF-8")
    }

    fn skip_value(
        &mut self,
        dtype: u32,
        depth: usize,
        token_digest: Option<&mut Sha256>,
    ) -> Result<Option<u64>> {
        ensure!(depth <= 4, "GGUF metadata nesting exceeds cap");
        self.objects = self
            .objects
            .checked_add(1)
            .context("GGUF object overflow")?;
        ensure!(self.objects <= MAX_GGUF_OBJECTS, "GGUF object cap exceeded");
        let scalar = match dtype {
            0 | 1 | 7 => Some(1),
            2 | 3 => Some(2),
            4..=6 => Some(4),
            10..=12 => Some(8),
            _ => None,
        };
        if let Some(count) = scalar {
            let raw = self.take(count)?;
            let unsigned = match dtype {
                0 => Some(raw[0] as u64),
                2 => Some(u16::from_le_bytes(raw.try_into().unwrap()) as u64),
                4 => Some(u32::from_le_bytes(raw.try_into().unwrap()) as u64),
                10 => Some(u64::from_le_bytes(raw.try_into().unwrap())),
                _ => None,
            };
            return Ok(unsigned);
        }
        match dtype {
            8 => {
                let _ = self.string_bytes()?;
                Ok(None)
            }
            9 => {
                let child = self.u32()?;
                let count = self.u64()?;
                self.arrays = self
                    .arrays
                    .checked_add(count)
                    .context("GGUF array overflow")?;
                ensure!(
                    self.arrays <= MAX_GGUF_ARRAY_ITEMS,
                    "GGUF array cap exceeded"
                );
                if let Some(digest) = token_digest {
                    ensure!(child == 8, "tokenizer token metadata is not a string array");
                    for _ in 0..count {
                        self.objects = self
                            .objects
                            .checked_add(1)
                            .context("GGUF object overflow")?;
                        ensure!(self.objects <= MAX_GGUF_OBJECTS, "GGUF object cap exceeded");
                        let value = self.string_bytes()?;
                        digest.update((value.len() as u64).to_le_bytes());
                        digest.update(&value);
                    }
                    Ok(Some(count))
                } else {
                    for _ in 0..count {
                        self.skip_value(child, depth + 1, None)?;
                    }
                    Ok(None)
                }
            }
            _ => Err(anyhow!("unsupported GGUF metadata type {dtype}")),
        }
    }
}

fn scan_inventory_gguf(input: &OpenedInput) -> Result<InventoryGgufScan> {
    let mut cursor = GgufCursor {
        file: input.file.try_clone()?,
        size: input.bytes,
        strings: 0,
        arrays: 0,
        objects: 0,
    };
    cursor.file.rewind()?;
    ensure!(cursor.take(4)? == b"GGUF", "bad GGUF magic");
    let version = cursor.u32()?;
    ensure!(version == 3, "inventory requires GGUF v3");
    let tensor_count = cursor.u64()?;
    let metadata_count = cursor.u64()?;
    ensure!(
        tensor_count <= MAX_GGUF_TENSORS && metadata_count <= MAX_GGUF_METADATA,
        "GGUF count cap exceeded"
    );
    let mut tokenizer_count = None;
    let mut tokenizer_sha256 = None;
    let mut masks = Vec::new();
    let mut metadata_keys = HashSet::new();
    for _ in 0..metadata_count {
        let key = cursor.string()?;
        ensure!(
            metadata_keys.insert(key.clone()),
            "duplicate GGUF metadata key"
        );
        let dtype = cursor.u32()?;
        if key == "tokenizer.ggml.tokens" {
            let mut digest = Sha256::new();
            tokenizer_count = cursor.skip_value(dtype, 0, Some(&mut digest))?;
            tokenizer_sha256 = Some(hex(&digest.finalize()));
        } else {
            let scalar = cursor.skip_value(dtype, 0, None)?;
            if matches!(
                key.as_str(),
                "dflash-draft.dflash.mask_token_id" | "tokenizer.ggml.mask_token_id"
            ) {
                masks.push((
                    key.clone(),
                    scalar.context("mask-token metadata must be u64")?,
                ));
            }
            if key == "tokenizer.ggml.token_count" {
                tokenizer_count = Some(scalar.context("token count metadata must be u64")?);
            }
        }
    }
    for _ in 0..tensor_count {
        let _ = cursor.string()?;
        let dimensions = cursor.u32()?;
        ensure!((1..=4).contains(&dimensions), "GGUF tensor rank invalid");
        cursor
            .file
            .seek(SeekFrom::Current(i64::from(dimensions) * 8 + 12))?;
        ensure!(
            cursor.file.stream_position()? <= input.bytes,
            "truncated GGUF tensor table"
        );
    }
    ensure!(
        cursor.file.stream_position()? <= MAX_GGUF_HEADER_BYTES,
        "GGUF header cap exceeded"
    );
    Ok(InventoryGgufScan {
        version,
        tensor_count,
        metadata_count,
        tokenizer_count,
        tokenizer_sha256,
        masks,
    })
}

impl OpenedInput {
    fn open(path: &Path, label: &str, maximum: Option<u64>) -> Result<Self> {
        let canonical = canonical_input(path, label)?;
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&canonical)
            .with_context(|| format!("open {label} {}", canonical.display()))?;
        let metadata = file.metadata()?;
        ensure!(metadata.is_file(), "{label} is not a regular file");
        ensure!(metadata.nlink() == 1, "{label} must not be hard-linked");
        if let Some(maximum) = maximum {
            ensure!(metadata.len() <= maximum, "{label} exceeds {maximum} bytes");
        }
        let sha256 = hash_file(&file)?;
        Ok(Self {
            path: canonical,
            file,
            bytes: metadata.len(),
            device: metadata.dev(),
            inode: metadata.ino(),
            mtime: metadata.mtime(),
            mtime_nsec: metadata.mtime_nsec(),
            ctime: metadata.ctime(),
            ctime_nsec: metadata.ctime_nsec(),
            sha256,
        })
    }

    fn verify(&self, claim: &FileClaim, label: &str) -> Result<()> {
        ensure!(
            claim.max_bytes > 0 && claim.bytes <= claim.max_bytes,
            "{label} cap invalid"
        );
        ensure!(
            claim.path == self.path.to_string_lossy(),
            "{label} canonical path mismatch"
        );
        ensure!(
            claim.bytes == self.bytes && claim.sha256 == self.sha256,
            "{label} identity mismatch"
        );
        Ok(())
    }

    fn final_custody_check(&self, label: &str) -> Result<()> {
        let metadata = self.file.metadata()?;
        ensure!(
            metadata.is_file()
                && metadata.nlink() == 1
                && metadata.dev() == self.device
                && metadata.ino() == self.inode
                && metadata.len() == self.bytes
                && metadata.mtime() == self.mtime
                && metadata.mtime_nsec() == self.mtime_nsec
                && metadata.ctime() == self.ctime
                && metadata.ctime_nsec() == self.ctime_nsec,
            "final custody metadata changed for {label}"
        );
        ensure!(
            hash_file(&self.file)? == self.sha256,
            "final custody hash changed for {label}"
        );
        Ok(())
    }
}

struct ReservedOutputs {
    sidecar_path: PathBuf,
    trace_path: PathBuf,
    sidecar: File,
    trace: File,
}

fn reserve_outputs(sidecar: &Path, trace: &Path) -> Result<ReservedOutputs> {
    let sidecar_path = canonical_output(sidecar, "sidecar output")?;
    let trace_path = canonical_output(trace, "trace output")?;
    ensure!(sidecar_path != trace_path, "trace and sidecar paths alias");
    let sidecar = create_new_nofollow(&sidecar_path, "sidecar")?;
    let trace = create_new_nofollow(&trace_path, "trace")?;
    let sm = sidecar.metadata()?;
    let tm = trace.metadata()?;
    ensure!(
        sm.is_file() && tm.is_file() && sm.nlink() == 1 && tm.nlink() == 1,
        "reserved artifacts are not unique regular files"
    );
    ensure!(
        (sm.dev(), sm.ino()) != (tm.dev(), tm.ino()),
        "reserved artifact inode alias"
    );
    Ok(ReservedOutputs {
        sidecar_path,
        trace_path,
        sidecar,
        trace,
    })
}

fn create_new_nofollow(path: &Path, label: &str) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("exclusive-create {label} {}", path.display()))
}

fn lexical_absolute(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                ensure!(normalized.pop(), "path escapes filesystem root");
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    Ok(normalized)
}

fn canonical_input(path: &Path, label: &str) -> Result<PathBuf> {
    let lexical = lexical_absolute(path)?;
    let canonical = path
        .canonicalize()
        .with_context(|| format!("canonicalize {label} {}", path.display()))?;
    ensure!(
        lexical == canonical,
        "{label} contains a symlink or noncanonical alias"
    );
    let metadata = std::fs::symlink_metadata(&lexical)?;
    ensure!(
        !metadata.file_type().is_symlink() && metadata.is_file(),
        "{label} must be a non-symlink regular file"
    );
    Ok(canonical)
}

fn canonical_output(path: &Path, label: &str) -> Result<PathBuf> {
    let name = path.file_name().context("output filename is absent")?;
    ensure!(
        !name.as_bytes().is_empty() && name != "." && name != "..",
        "{label} filename invalid"
    );
    let lexical = lexical_absolute(path)?;
    let parent = lexical.parent().context("output parent is absent")?;
    let canonical_parent = parent
        .canonicalize()
        .with_context(|| format!("canonicalize {label} parent"))?;
    ensure!(
        parent == canonical_parent,
        "{label} parent contains a symlink or alias"
    );
    ensure!(
        canonical_parent.is_dir(),
        "{label} parent is not a directory"
    );
    Ok(canonical_parent.join(name))
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("String write");
    }
    out
}

fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn valid_git_oid(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn behavior_environment<I>(variables: I) -> Result<BTreeMap<String, String>>
where
    I: IntoIterator<Item = (OsString, OsString)>,
{
    let mut observed = BTreeMap::new();
    for (raw_name, raw_value) in variables {
        let name = raw_name
            .to_str()
            .context("process environment contains a non-Unicode name")?;
        let value = raw_value
            .to_str()
            .with_context(|| format!("process environment {name} has a non-Unicode value"))?;
        let behavior_affecting = name.starts_with("QWEN_")
            || name.starts_with("MTL_")
            || name.starts_with("METAL_")
            || name.starts_with("GGML_METAL_")
            || name.starts_with("DYLD_");
        if !behavior_affecting {
            continue;
        }
        ensure!(
            name == "QWEN_METAL_LEASE_WAIT" && value == "1",
            "forbidden behavior-affecting environment variable {name}"
        );
        ensure!(
            observed.insert(name.to_owned(), value.to_owned()).is_none(),
            "duplicate behavior-affecting environment variable {name}"
        );
    }
    ensure!(
        observed == BTreeMap::from([("QWEN_METAL_LEASE_WAIT".into(), "1".into())]),
        "literal QWEN_METAL_LEASE_WAIT=1 is required"
    );
    Ok(observed)
}

fn observed_behavior_environment() -> Result<BTreeMap<String, String>> {
    behavior_environment(std::env::vars_os())
}

fn compiled_target() -> String {
    match (std::env::consts::ARCH, std::env::consts::OS) {
        ("aarch64", "macos") => "aarch64-apple-darwin".to_owned(),
        (arch, os) => format!("{arch}-{os}"),
    }
}

fn metal_family_capabilities(device: &ProtocolObject<dyn MTLDevice>) -> String {
    let families = [
        ("apple1", MTLGPUFamily::Apple1),
        ("apple2", MTLGPUFamily::Apple2),
        ("apple3", MTLGPUFamily::Apple3),
        ("apple4", MTLGPUFamily::Apple4),
        ("apple5", MTLGPUFamily::Apple5),
        ("apple6", MTLGPUFamily::Apple6),
        ("apple7", MTLGPUFamily::Apple7),
        ("apple8", MTLGPUFamily::Apple8),
        ("apple9", MTLGPUFamily::Apple9),
        ("apple10", MTLGPUFamily::Apple10),
        ("mac2", MTLGPUFamily::Mac2),
        ("common1", MTLGPUFamily::Common1),
        ("common2", MTLGPUFamily::Common2),
        ("common3", MTLGPUFamily::Common3),
        ("metal3", MTLGPUFamily::Metal3),
        ("metal4", MTLGPUFamily::Metal4),
    ];
    let supported = families
        .into_iter()
        .filter_map(|(name, family)| device.supportsFamily(family).then_some(name))
        .collect::<Vec<_>>();
    format!("mtl-gpu-family-v1:{}", supported.join(","))
}

fn bits(value: u32) -> String {
    format!("0x{value:08x}")
}

fn static_chain(value: &str) -> Result<StaticChain> {
    let mut pieces = value.split(':');
    let name = pieces.next().unwrap_or_default().to_owned();
    let carry = pieces
        .next()
        .context("fixed chain carry is absent")?
        .parse::<i32>()?;
    let slots = pieces
        .next()
        .context("fixed chain slots are absent")?
        .split(',')
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(pieces.next().is_none(), "fixed chain has trailing fields");
    ensure!(
        !name.is_empty() && name.len() <= 128 && name.is_ascii(),
        "fixed chain name invalid"
    );
    ensure!(
        slots.len() == 7 && slots.iter().all(|slot| *slot < 16),
        "fixed chain requires seven slots in 0..16"
    );
    Ok(StaticChain {
        name,
        initial_carry: carry,
        slots,
    })
}

fn validate_chain_set(chains: &[StaticChain], production_carry: i32) -> Result<()> {
    ensure!(!chains.is_empty(), "at least one fixed chain is required");
    ensure!(
        chains.len() <= MAX_FIXED_CHAINS,
        "fixed chain count exceeds {MAX_FIXED_CHAINS}"
    );
    let names = chains
        .iter()
        .map(|chain| chain.name.as_str())
        .collect::<HashSet<_>>();
    ensure!(
        names.len() == chains.len(),
        "fixed chain names must be unique"
    );
    ensure!(
        chains
            .iter()
            .all(|chain| chain.initial_carry == production_carry),
        "every fixed chain must use the one production carry"
    );
    Ok(())
}

fn keys(value: &Value, expected: &[&str], label: &str) -> Result<()> {
    let object = value
        .as_object()
        .with_context(|| format!("{label} must be an object"))?;
    ensure!(
        object
            .keys()
            .map(String::as_str)
            .eq(expected.iter().copied()),
        "{label} keys/order mismatch"
    );
    Ok(())
}

fn validate_manifest_order(value: &Value) -> Result<()> {
    keys(
        value,
        &[
            "schema",
            "schema_version",
            "run_id",
            "attempt_id",
            "trace_max_bytes",
            "sidecar_max_bytes",
            "reducer",
            "executable",
            "fixture",
            "command",
            "sources",
            "assets",
            "semantic_references",
            "tensors",
            "expected_request",
            "expected_prompt",
            "expected_binding",
            "expected_continuation_carry_token",
            "expected_rng_domains",
            "expected_fixed_chains",
            "expected_capture_context",
            "expected_build",
            "expected_host",
            "embedded_metallib_sha256",
            "selector_dispatch_predicate",
            "preparation_binding",
            "scalar_contract",
        ],
        "manifest",
    )?;
    for name in ["reducer", "executable", "fixture", "command"] {
        keys(
            &value[name],
            &["path", "bytes", "sha256", "max_bytes"],
            name,
        )?;
    }
    for name in ["sources", "assets"] {
        for item in value[name]
            .as_array()
            .context("manifest role list invalid")?
        {
            keys(
                item,
                &["role", "path", "bytes", "sha256", "max_bytes"],
                name,
            )?;
        }
    }
    keys(
        &value["semantic_references"],
        &[
            "mlx_model_mlx_py",
            "vllm_qwen3_dflash2_py",
            "vllm_speculator_py",
            "llama_cpp_dflash_cpp",
            "llama_cpp_speculative_cpp",
        ],
        "semantic references",
    )?;
    for reference in value["semantic_references"]
        .as_object()
        .context("semantic references invalid")?
        .values()
    {
        keys(reference, &["commit", "sha256"], "semantic reference")?;
    }
    for tensor in value["tensors"]
        .as_array()
        .context("tensor claims invalid")?
    {
        keys(
            tensor,
            &[
                "role",
                "asset_role",
                "name",
                "dtype",
                "shape",
                "offset",
                "bytes",
                "sha256",
                "orientation",
                "row_domain",
            ],
            "tensor claim",
        )?;
        if !tensor["row_domain"].is_null() {
            keys(&tensor["row_domain"], &["first", "count"], "row domain")?;
        }
    }
    keys(
        &value["expected_request"],
        &["request", "ignored_target_policy"],
        "expected_request",
    )?;
    keys(
        &value["expected_request"]["request"],
        &["temperature_f32_bits"],
        "request",
    )?;
    keys(
        &value["expected_request"]["ignored_target_policy"],
        &["variant_a", "variant_b"],
        "ignored_target_policy",
    )?;
    for name in ["variant_a", "variant_b"] {
        keys(
            &value["expected_request"]["ignored_target_policy"][name],
            &[
                "top_k",
                "top_p_f32_bits",
                "min_p_f32_bits",
                "grammar",
                "penalties",
            ],
            name,
        )?;
    }
    keys(
        &value["expected_binding"],
        &[
            "production_call_id",
            "drafter_checkpoint_sha256",
            "proposal_construction_id",
            "noise_input_sha256",
        ],
        "binding",
    )?;
    keys(
        &value["expected_prompt"],
        &[
            "utf8_hex",
            "token_ids",
            "token_ids_sha256_i32le",
            "tokenizer_identity_sha256",
        ],
        "expected prompt",
    )?;
    for chain in value["expected_fixed_chains"]
        .as_array()
        .context("fixed chains invalid")?
    {
        keys(chain, &["name", "initial_carry", "slots"], "fixed chain")?;
    }
    keys(
        &value["expected_capture_context"],
        &[
            "definition_version",
            "carry_token",
            "noise_start_position",
            "target_context_len",
            "context_hidden_watermark",
            "kv_context_watermark",
        ],
        "capture context",
    )?;
    keys(
        &value["expected_build"],
        &[
            "commit",
            "source_sha256",
            "dirty",
            "compiler",
            "compiler_version",
            "target",
            "profile",
            "features",
        ],
        "build",
    )?;
    keys(
        &value["expected_host"],
        &[
            "os",
            "arch",
            "device_name",
            "device_registry_id",
            "device_family",
        ],
        "host",
    )?;
    keys(
        &value["selector_dispatch_predicate"],
        &[
            "tag",
            "kernel",
            "weight_dtype",
            "input_dtype",
            "output_dtype",
            "weight_dtype_id",
            "input_dtype_id",
            "output_dtype_id",
            "n",
            "h",
            "r",
            "grid",
            "threads",
            "metal_source_sha256",
            "metallib_sha256",
            "build_source_sha256",
            "allowed_environment",
        ],
        "selector dispatch predicate",
    )?;
    keys(
        &value["preparation_binding"],
        &[
            "inventory",
            "inventory_spec",
            "preparation_spec",
            "seal_path",
        ],
        "preparation binding",
    )?;
    for name in ["inventory", "inventory_spec", "preparation_spec"] {
        keys(
            &value["preparation_binding"][name],
            &["path", "bytes", "sha256", "max_bytes"],
            "preparation binding claim",
        )?;
    }
    keys(
        &value["scalar_contract"],
        &[
            "artifact",
            "compiler",
            "compiler_version",
            "target",
            "profile",
            "fixture_domain",
            "fixture_sha256",
            "vectors",
        ],
        "scalar contract",
    )?;
    keys(
        &value["scalar_contract"]["artifact"],
        &["path", "bytes", "sha256", "max_bytes"],
        "scalar artifact",
    )?;
    for vector in value["scalar_contract"]["vectors"]
        .as_array()
        .context("scalar vectors invalid")?
    {
        keys(
            vector,
            &[
                "name",
                "a_f32_bits",
                "z_f32_bits",
                "successor_f32_bits",
                "unary_f32_bits",
                "score_f32_bits",
            ],
            "scalar vector",
        )?;
    }
    Ok(())
}

fn validate_inventory_spec_order(value: &Value) -> Result<()> {
    keys(
        value,
        &[
            "schema",
            "schema_version",
            "run_id",
            "inventory_max_bytes",
            "checkout",
            "build",
            "sources",
            "executable",
            "reducer",
            "scalar_fixture",
            "command_template",
            "embedded_metallib",
            "assets",
            "tensor_requirements",
            "tokenizer",
            "prompt",
            "carry_token",
            "expected_mask_token",
            "parser_caps",
            "host_predicate",
            "command",
            "environment",
        ],
        "inventory spec",
    )?;
    keys(
        &value["checkout"],
        &["path", "commit", "tree", "dirty"],
        "inventory checkout",
    )?;
    keys(
        &value["build"],
        &[
            "commit",
            "source_sha256",
            "dirty",
            "compiler",
            "compiler_version",
            "target",
            "profile",
            "features",
        ],
        "inventory build",
    )?;
    for name in [
        "executable",
        "reducer",
        "scalar_fixture",
        "command_template",
        "embedded_metallib",
    ] {
        keys(
            &value[name],
            &["path", "bytes", "sha256", "max_bytes"],
            name,
        )?;
    }
    for name in ["sources", "assets"] {
        for claim in value[name].as_array().context("inventory claims invalid")? {
            keys(
                claim,
                &["role", "path", "bytes", "sha256", "max_bytes"],
                name,
            )?;
        }
    }
    for requirement in value["tensor_requirements"]
        .as_array()
        .context("inventory tensor requirements invalid")?
    {
        keys(
            requirement,
            &[
                "role",
                "asset_role",
                "name",
                "dtype",
                "shape",
                "orientation",
                "row_domain",
            ],
            "inventory tensor requirement",
        )?;
        if !requirement["row_domain"].is_null() {
            keys(
                &requirement["row_domain"],
                &["first", "count"],
                "row domain",
            )?;
        }
    }
    keys(
        &value["tokenizer"],
        &[
            "vocab_size",
            "token_embd_name",
            "token_embd_shape",
            "token_embd_dtype",
            "token_count",
            "tokenizer_tokens_sha256",
        ],
        "inventory tokenizer",
    )?;
    keys(
        &value["prompt"],
        &[
            "utf8_hex",
            "token_ids",
            "token_ids_sha256_i32le",
            "tokenizer_identity_sha256",
        ],
        "inventory prompt",
    )?;
    keys(
        &value["parser_caps"],
        &[
            "header_bytes",
            "metadata",
            "tensors",
            "strings_bytes",
            "array_items",
            "objects",
        ],
        "inventory parser caps",
    )?;
    keys(
        &value["host_predicate"],
        &["os", "arch", "device_name"],
        "inventory host predicate",
    )?;
    Ok(())
}

fn semantic_references_are_exact(value: &SemanticReferences) -> bool {
    value
        == &SemanticReferences {
            mlx_model_mlx_py: SemanticReference {
                commit: "07ebd93db9f472af339b644bb70221ad8428328a".into(),
                sha256: "2f8598eaca4cb814e63ea69e791c1bdf55ba82280bfd299f9141906356b7cb87".into(),
            },
            vllm_qwen3_dflash2_py: SemanticReference {
                commit: "b389ac29465b33f9e9c534df221ea3c129e9793f".into(),
                sha256: "c141daa4b2059c0098224ac36471c2197b7052c100bef0a4dbc2ca79b627053f".into(),
            },
            vllm_speculator_py: SemanticReference {
                commit: "b389ac29465b33f9e9c534df221ea3c129e9793f".into(),
                sha256: "1f6ff5ca9c8f38ff417aafd43bfa3116b5387bf0f7b58721acb2185781879836".into(),
            },
            llama_cpp_dflash_cpp: SemanticReference {
                commit: "1deefcca395743049c3820ab8f9b15043f3e9446".into(),
                sha256: "3b31b1a6b888ec37013a4d6275fdaf1fdeb6a9e3ca25b933ab136acdbcd58c49".into(),
            },
            llama_cpp_speculative_cpp: SemanticReference {
                commit: "1deefcca395743049c3820ab8f9b15043f3e9446".into(),
                sha256: "e14da24022c4c16d2de7cb582257020a02893c5bbcc6bb2b052cf1e8a8525138".into(),
            },
        }
}

fn validate_static(
    manifest: &Manifest,
    args: &DflashK0sArgs,
    chains: &[StaticChain],
    build: &Value,
) -> Result<()> {
    ensure!(
        manifest.schema == MANIFEST_SCHEMA && manifest.schema_version == 1,
        "manifest schema/version mismatch"
    );
    ensure!(
        !manifest.run_id.is_empty() && manifest.run_id.len() <= 128 && manifest.run_id.is_ascii(),
        "run_id invalid"
    );
    ensure!(
        !manifest.attempt_id.is_empty()
            && manifest.attempt_id.len() <= 128
            && manifest.attempt_id.is_ascii()
            && manifest.attempt_id == args.attempt_id,
        "attempt_id invalid or differs from manifest"
    );
    ensure!(
        (1..=MAX_TRACE).contains(&manifest.trace_max_bytes),
        "trace cap invalid"
    );
    ensure!(
        (1..=MAX_SIDECAR).contains(&manifest.sidecar_max_bytes),
        "sidecar cap invalid"
    );
    ensure!(
        manifest
            .trace_max_bytes
            .checked_add(manifest.sidecar_max_bytes)
            .is_some_and(|n| n <= MAX_COMBINED),
        "combined cap invalid"
    );
    ensure!(
        args.temperature.is_finite() && args.temperature > 0.0,
        "temperature must be finite and positive"
    );
    ensure!(
        manifest.expected_request.request.temperature_f32_bits == bits(args.temperature.to_bits()),
        "temperature differs from static manifest"
    );
    ensure!(
        hex_decode(&manifest.expected_prompt.utf8_hex)
            .is_ok_and(|bytes| bytes == args.prompt.as_bytes())
            && !manifest.expected_prompt.token_ids.is_empty()
            && manifest.expected_prompt.token_ids.len() <= MAX_GGUF_ARRAY_ITEMS as usize
            && manifest.expected_prompt.token_ids_sha256_i32le
                == token_ids_sha256_i32le(&manifest.expected_prompt.token_ids)
            && valid_sha(&manifest.expected_prompt.tokenizer_identity_sha256),
        "prompt bytes/token identity differ from static manifest"
    );
    ensure!(
        manifest.expected_request.ignored_target_policy.variant_a
            != manifest.expected_request.ignored_target_policy.variant_b,
        "ignored policies must differ"
    );
    for policy in [
        &manifest.expected_request.ignored_target_policy.variant_a,
        &manifest.expected_request.ignored_target_policy.variant_b,
    ] {
        ensure!(
            policy.grammar.is_none() && policy.penalties.is_none(),
            "ignored policy grammar/penalties must be null"
        );
        for raw in [&policy.top_p_f32_bits, &policy.min_p_f32_bits] {
            ensure!(
                parse_bits(raw).is_some_and(f32::is_finite),
                "ignored policy bits invalid"
            );
        }
    }
    ensure!(
        manifest.expected_fixed_chains == chains && !chains.is_empty(),
        "fixed chains differ from static manifest"
    );
    validate_chain_set(chains, args.carry_token)?;
    ensure!(
        manifest.expected_continuation_carry_token == args.continuation_carry_token
            && (0..DFLASH_K0S_VOCAB as i32).contains(&args.continuation_carry_token),
        "continuation carry differs from static manifest or vocabulary"
    );
    ensure!(
        !manifest.expected_rng_domains.is_empty()
            && manifest.expected_rng_domains.len() <= 32
            && manifest
                .expected_rng_domains
                .iter()
                .all(|name| !name.is_empty() && name.len() <= 128 && name.is_ascii())
            && manifest
                .expected_rng_domains
                .iter()
                .collect::<HashSet<_>>()
                .len()
                == manifest.expected_rng_domains.len(),
        "expected RNG domains invalid"
    );
    let context = &manifest.expected_capture_context;
    ensure!(
        context.definition_version == CAPTURE_DEFINITION && context.carry_token == args.carry_token,
        "capture context/carry mismatch"
    );
    ensure!(
        context.target_context_len == context.context_hidden_watermark
            && context.target_context_len == context.kv_context_watermark,
        "capture watermarks must be synchronized"
    );
    ensure!(
        context.noise_start_position as u64 == context.target_context_len,
        "noise position must equal fixed prompt context length"
    );
    ensure!(
        context.noise_start_position <= u32::MAX - 7,
        "noise start position leaves no room for seven selector depths"
    );
    ensure!(
        manifest.expected_build.features == ["dflash-k0s-diagnostics"]
            && manifest.expected_build.profile == "release"
            && !manifest.expected_build.dirty,
        "build contract invalid"
    );
    ensure!(
        !cfg!(debug_assertions) && cfg!(feature = "dflash-k0s-diagnostics"),
        "K0-S requires a release binary with dflash-k0s-diagnostics enabled"
    );
    ensure!(
        manifest.expected_build.target == compiled_target(),
        "compiled target differs from static build metadata"
    );
    ensure!(
        valid_git_oid(&manifest.expected_build.commit)
            && valid_sha(&manifest.expected_build.source_sha256),
        "build digests invalid"
    );
    ensure!(
        manifest.expected_host.os == std::env::consts::OS
            && manifest.expected_host.arch == std::env::consts::ARCH,
        "host OS/architecture mismatch"
    );
    ensure!(
        !manifest.expected_host.device_name.is_empty()
            && manifest
                .expected_host
                .device_family
                .starts_with("mtl-gpu-family-v1:"),
        "host device identity/capability string invalid"
    );
    ensure!(
        !manifest.expected_build.compiler.is_empty()
            && manifest.expected_build.compiler.len() <= 256
            && !manifest.expected_build.compiler_version.is_empty()
            && manifest.expected_build.compiler_version.len() <= 256,
        "static compiler metadata must be bounded nonempty manifest data"
    );
    ensure!(
        manifest.selector_dispatch_predicate.tag == "dflash_k0s.selector_hidden_projection.v1"
            && manifest.selector_dispatch_predicate.kernel == K0S_SELECTOR_KERNEL
            && manifest.selector_dispatch_predicate.weight_dtype == K0S_SELECTOR_WEIGHT_DTYPE
            && manifest.selector_dispatch_predicate.weight_dtype_id == K0S_SELECTOR_WEIGHT_DTYPE_ID
            && manifest.selector_dispatch_predicate.n == DFLASH_K0S_BLOCK_SIZE as u64
            && manifest.selector_dispatch_predicate.h == DFLASH_K0S_HIDDEN as u64
            && manifest.selector_dispatch_predicate.r == DFLASH_K0S_RANK as u64
            && manifest.selector_dispatch_predicate.input_dtype == "F32"
            && manifest.selector_dispatch_predicate.output_dtype == "F32"
            && manifest.selector_dispatch_predicate.input_dtype_id == 0
            && manifest.selector_dispatch_predicate.output_dtype_id == 0
            && manifest.selector_dispatch_predicate.metallib_sha256
                == manifest.embedded_metallib_sha256
            && manifest.selector_dispatch_predicate.build_source_sha256
                == manifest.expected_build.source_sha256
            && manifest.selector_dispatch_predicate.allowed_environment
                == BTreeMap::from([("QWEN_METAL_LEASE_WAIT".into(), "1".into())]),
        "selector dispatch predicate invalid"
    );
    ensure!(
        !manifest.selector_dispatch_predicate.kernel.is_empty()
            && manifest
                .selector_dispatch_predicate
                .grid
                .iter()
                .all(|value| *value > 0)
            && manifest
                .selector_dispatch_predicate
                .threads
                .iter()
                .all(|value| *value > 0)
            && valid_sha(&manifest.selector_dispatch_predicate.metal_source_sha256)
            && valid_sha(&manifest.selector_dispatch_predicate.metallib_sha256)
            && valid_sha(&manifest.selector_dispatch_predicate.build_source_sha256),
        "selector dispatch predicate fields invalid"
    );
    ensure!(
        valid_sha(&manifest.embedded_metallib_sha256),
        "metallib digest invalid"
    );
    ensure!(
        manifest.selector_dispatch_predicate.metal_source_sha256
            == claim_by_role(&manifest.sources, "mat_mat_q4_k_metal")?.sha256,
        "selector predicate does not bind authenticated mat_mat_q4_k source"
    );
    ensure!(
        semantic_references_are_exact(&manifest.semantic_references),
        "semantic references mismatch"
    );
    ensure!(
        manifest.scalar_contract.fixture_domain == SCALAR_DOMAIN
            && manifest.scalar_contract.fixture_sha256 == SCALAR_DIGEST,
        "scalar fixture identity mismatch"
    );
    ensure!(
        manifest.scalar_contract.compiler == manifest.expected_build.compiler
            && manifest.scalar_contract.compiler_version
                == manifest.expected_build.compiler_version
            && manifest.scalar_contract.target == manifest.expected_build.target
            && manifest.scalar_contract.profile == manifest.expected_build.profile,
        "scalar/build identity mismatch"
    );
    let build_commit = build
        .get("build_commit")
        .and_then(Value::as_str)
        .context("compiled build commit absent")?;
    let source = build
        .get("build_source_state")
        .and_then(Value::as_str)
        .context("compiled source digest absent")?;
    let source = source
        .strip_prefix("git-source-sha256-v2:")
        .context("compiled source digest domain invalid")?;
    let clean = build.get("status").and_then(Value::as_str) == Some("match")
        && build.get("build_dirty").and_then(Value::as_bool) == Some(false)
        && build.get("runtime_dirty").and_then(Value::as_bool) == Some(false);
    ensure!(clean, "K0-S requires a clean matching build");
    ensure!(
        valid_git_oid(build_commit),
        "compiled build commit must be a canonical 40- or 64-hex Git object ID"
    );
    ensure!(
        manifest.expected_build.commit == build_commit
            && manifest.expected_build.source_sha256 == source,
        "compiled build identity differs from manifest"
    );
    Ok(())
}

fn parse_bits(value: &str) -> Option<f32> {
    (value.len() == 10
        && value.starts_with("0x")
        && value[2..]
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
    .then(|| u32::from_str_radix(&value[2..], 16).ok())
    .flatten()
    .map(f32::from_bits)
}

fn claim_by_role<'a>(claims: &'a [RoleClaim], role: &str) -> Result<&'a FileClaim> {
    let matches = claims
        .iter()
        .filter(|claim| claim.role == role)
        .collect::<Vec<_>>();
    ensure!(
        matches.len() == 1,
        "manifest requires exactly one {role} claim"
    );
    Ok(&matches[0].file)
}

fn validate_source_roles(sources: &[RoleClaim]) -> Result<()> {
    ensure!(
        sources.len() == REQUIRED_SOURCE_ROLES.len(),
        "source role list length mismatch"
    );
    let roles = sources
        .iter()
        .map(|claim| claim.role.as_str())
        .collect::<HashSet<_>>();
    ensure!(roles.len() == sources.len(), "source roles must be unique");
    ensure!(
        REQUIRED_SOURCE_ROLES
            .iter()
            .all(|role| roles.contains(role)),
        "source role set mismatch"
    );
    Ok(())
}

fn validate_claim_shape(manifest: &Manifest) -> Result<()> {
    validate_source_roles(&manifest.sources)?;
    let producer_source = manifest
        .sources
        .iter()
        .find(|claim| claim.role == "dflash_k0s_rs")
        .context("dflash_k0s_rs source claim absent")?;
    let expected_producer_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/dflash_k0s.rs")
        .canonicalize()
        .context("canonicalize compiled K0-S producer source")?;
    ensure!(
        Path::new(&producer_source.file.path) == expected_producer_path,
        "dflash_k0s_rs claim does not name the compiled producer source"
    );
    let asset_roles = manifest
        .assets
        .iter()
        .map(|v| v.role.as_str())
        .collect::<HashSet<_>>();
    ensure!(
        asset_roles == HashSet::from(["target", "drafter"]),
        "asset role set mismatch"
    );
    ensure!(
        manifest.tensors.len() == 3,
        "exactly three tensor claims required"
    );
    let tensor_roles = manifest
        .tensors
        .iter()
        .map(|v| v.role.as_str())
        .collect::<HashSet<_>>();
    ensure!(
        tensor_roles == HashSet::from(["selector_hidden", "predecessor", "successor"]),
        "tensor role set mismatch"
    );
    ensure!(
        manifest
            .tensors
            .iter()
            .find(|tensor| tensor.role == "selector_hidden")
            .is_some_and(|tensor| tensor.dtype == K0S_SELECTOR_WEIGHT_DTYPE),
        "K0-S v1 selector-hidden manifest tensor must be Q4_K"
    );
    ensure!(
        manifest
            .tensors
            .iter()
            .filter(|tensor| matches!(tensor.role.as_str(), "predecessor" | "successor"))
            .all(|tensor| tensor.dtype == "Q4_K"),
        "K0-S producer requires Q4_K predecessor/successor drafter codebooks"
    );
    for claim in manifest
        .sources
        .iter()
        .map(|v| &v.file)
        .chain(manifest.assets.iter().map(|v| &v.file))
        .chain([
            &manifest.reducer,
            &manifest.executable,
            &manifest.fixture,
            &manifest.command,
            &manifest.preparation_binding.inventory,
            &manifest.preparation_binding.inventory_spec,
            &manifest.preparation_binding.preparation_spec,
        ])
    {
        ensure!(
            valid_sha(&claim.sha256) && claim.max_bytes > 0 && claim.bytes <= claim.max_bytes,
            "file claim invalid"
        );
    }
    for claim in [
        &manifest.preparation_binding.inventory,
        &manifest.preparation_binding.inventory_spec,
        &manifest.preparation_binding.preparation_spec,
    ] {
        ensure!(
            Path::new(&claim.path).is_absolute()
                && lexical_absolute(Path::new(&claim.path))?.as_path() == Path::new(&claim.path),
            "preparation binding claim path is not canonical absolute"
        );
    }
    let seal = canonical_output(
        Path::new(&manifest.preparation_binding.seal_path),
        "preparation seal path",
    )?;
    ensure!(
        seal.as_path() == Path::new(&manifest.preparation_binding.seal_path),
        "preparation seal path is not canonical absolute"
    );
    Ok(())
}

fn open_and_verify_claim(path: &Path, claim: &FileClaim, label: &str) -> Result<OpenedInput> {
    let opened = OpenedInput::open(path, label, Some(claim.max_bytes))?;
    opened.verify(claim, label)?;
    Ok(opened)
}

fn opened_bytes(input: &OpenedInput) -> Result<Vec<u8>> {
    let mut file = input.file.try_clone()?;
    file.rewind()?;
    let capacity = usize::try_from(input.bytes).context("input byte count exceeds usize")?;
    let mut bytes = Vec::with_capacity(capacity);
    file.read_to_end(&mut bytes)?;
    ensure!(bytes.len() == capacity, "short read from verified input");
    Ok(bytes)
}

fn reject_input_aliases(inputs: &[(&str, &OpenedInput)], outputs: &ReservedOutputs) -> Result<()> {
    let mut paths = HashSet::new();
    let mut inodes = HashSet::new();
    for (label, input) in inputs {
        ensure!(
            paths.insert(input.path.clone()),
            "input path alias at {label}"
        );
        ensure!(
            inodes.insert((input.device, input.inode)),
            "input inode alias at {label}"
        );
        ensure!(
            input.path != outputs.sidecar_path && input.path != outputs.trace_path,
            "input aliases output at {label}"
        );
    }
    Ok(())
}

struct CommandBindings<'a> {
    executable: &'a Path,
    attempt_id: &'a str,
    model: &'a Path,
    drafter: &'a Path,
    prompt: &'a str,
    carry_token: i32,
    continuation_carry_token: i32,
    manifest: &'a Path,
    command_manifest: &'a Path,
    fixture: &'a Path,
    temperature_bits: u32,
    fixed_chains: &'a [StaticChain],
    trace_output: &'a Path,
    sidecar_output: &'a Path,
}

fn fixed_chain_arg(chain: &StaticChain) -> String {
    format!(
        "{}:{}:{}",
        chain.name,
        chain.initial_carry,
        chain
            .slots
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn validate_command_argv(
    bytes: &[u8],
    actual: &[String],
    supplied_manifest_sha256: &str,
    bindings: &CommandBindings<'_>,
) -> Result<()> {
    let command: CommandManifest =
        serde_json::from_slice(bytes).context("parse command manifest")?;
    let ordered: Value = serde_json::from_slice(bytes).context("parse ordered command manifest")?;
    keys(&ordered, &["argv"], "command manifest")?;
    ensure!(
        command.argv.len() == 28 + 2 * bindings.fixed_chains.len(),
        "command argv length mismatch"
    );
    let temperature_text = command
        .argv
        .get(23)
        .context("command temperature absent")?
        .clone();
    let temperature = temperature_text
        .parse::<f64>()
        .context("command temperature is not a number")? as f32;
    ensure!(
        temperature.to_bits() == bindings.temperature_bits,
        "command temperature differs from static request"
    );
    let mut index = 0usize;
    let mut take = |expected: &str, label: &str| -> Result<()> {
        ensure!(
            command.argv.get(index).map(String::as_str) == Some(expected),
            "command argv {label} mismatch"
        );
        index += 1;
        Ok(())
    };
    take(&bindings.executable.to_string_lossy(), "executable")?;
    take("dflash-k0s-lattice", "subcommand")?;
    take("--attempt-id", "attempt flag")?;
    take(bindings.attempt_id, "attempt value")?;
    take("--model", "model flag")?;
    take(&bindings.model.to_string_lossy(), "target path")?;
    take("--drafter", "drafter flag")?;
    take(&bindings.drafter.to_string_lossy(), "drafter path")?;
    take("--prompt", "prompt flag")?;
    take(bindings.prompt, "prompt")?;
    take("--carry-token", "carry flag")?;
    take(&bindings.carry_token.to_string(), "carry value")?;
    take("--continuation-carry-token", "continuation carry flag")?;
    take(
        &bindings.continuation_carry_token.to_string(),
        "continuation carry value",
    )?;
    take("--manifest", "manifest flag")?;
    take(&bindings.manifest.to_string_lossy(), "manifest path")?;
    take("--manifest-sha256", "manifest digest flag")?;
    take(COMMAND_MANIFEST_PLACEHOLDER, "manifest digest placeholder")?;
    take("--command-manifest", "command manifest flag")?;
    take(
        &bindings.command_manifest.to_string_lossy(),
        "command manifest path",
    )?;
    take("--fixture", "fixture flag")?;
    take(&bindings.fixture.to_string_lossy(), "fixture path")?;
    take("--temperature", "temperature flag")?;
    take(&temperature_text, "temperature value")?;
    for chain in bindings.fixed_chains {
        take("--fixed-chain", "fixed-chain flag")?;
        take(&fixed_chain_arg(chain), "fixed-chain value")?;
    }
    take("--trace-output", "trace output flag")?;
    take(
        &bindings.trace_output.to_string_lossy(),
        "trace output path",
    )?;
    take("--sidecar-output", "sidecar output flag")?;
    take(
        &bindings.sidecar_output.to_string_lossy(),
        "sidecar output path",
    )?;
    ensure!(
        index == command.argv.len(),
        "command argv has trailing values"
    );
    ensure!(
        actual.first().is_some_and(
            |value| canonical_input(Path::new(value), "command executable")
                .ok()
                .as_deref()
                == Some(bindings.executable)
        ),
        "argv executable differs from authenticated executable"
    );
    ensure!(
        actual
            .iter()
            .filter(|arg| arg.as_str() == supplied_manifest_sha256)
            .count()
            == 1,
        "actual argv must contain the supplied manifest digest exactly once"
    );
    ensure!(
        command
            .argv
            .iter()
            .filter(|arg| arg.as_str() == COMMAND_MANIFEST_PLACEHOLDER)
            .count()
            == 1,
        "command manifest must contain exactly one literal manifest digest placeholder"
    );
    let placeholder_index = command
        .argv
        .iter()
        .position(|arg| arg == COMMAND_MANIFEST_PLACEHOLDER)
        .context("command manifest placeholder absent")?;
    let mut substituted = command.argv.clone();
    substituted[placeholder_index] = supplied_manifest_sha256.to_owned();
    ensure!(
        substituted == actual,
        "running argv differs from exact digest-substituted canonical command"
    );
    Ok(())
}

fn validate_command(
    bytes: &[u8],
    supplied_manifest_sha256: &str,
    bindings: &CommandBindings<'_>,
) -> Result<()> {
    let actual = std::env::args_os()
        .map(|arg| {
            arg.into_string()
                .map_err(|_| anyhow!("non-UTF8 argv is forbidden"))
        })
        .collect::<Result<Vec<_>>>()?;
    validate_command_argv(bytes, &actual, supplied_manifest_sha256, bindings)
}

fn scalar_vectors() -> Vec<ScalarVector> {
    dflash_k0s_scalar_contract_fixture()
        .cases
        .into_iter()
        .map(|case| ScalarVector {
            name: case.name.into(),
            a_f32_bits: case.a_bits.into_iter().map(bits).collect(),
            z_f32_bits: case.z_bits.into_iter().map(bits).collect(),
            successor_f32_bits: case.successor_bits.into_iter().map(bits).collect(),
            unary_f32_bits: bits(case.unary_bits),
            score_f32_bits: bits(case.score_bits),
        })
        .collect()
}

fn validate_scalar_fixture(bytes: &[u8], manifest: &Manifest) -> Result<()> {
    let file: ScalarFixtureFile = serde_json::from_slice(bytes).context("parse scalar fixture")?;
    let ordered: Value = serde_json::from_slice(bytes).context("parse ordered scalar fixture")?;
    keys(
        &ordered,
        &[
            "schema",
            "schema_version",
            "fixture_domain",
            "fixture_sha256",
            "vectors",
        ],
        "scalar fixture",
    )?;
    for vector in ordered["vectors"]
        .as_array()
        .context("scalar fixture vectors invalid")?
    {
        keys(
            vector,
            &[
                "name",
                "a_f32_bits",
                "z_f32_bits",
                "successor_f32_bits",
                "unary_f32_bits",
                "score_f32_bits",
            ],
            "scalar fixture vector",
        )?;
    }
    let fixture = dflash_k0s_scalar_contract_fixture();
    ensure!(
        file.schema == "qwen.dflash_k0s_scalar_fixture"
            && file.schema_version == 1
            && file.fixture_domain == SCALAR_DOMAIN,
        "scalar fixture schema mismatch"
    );
    ensure!(
        hex(&fixture.fixture_sha256) == SCALAR_DIGEST,
        "compiled scalar fixture digest mismatch"
    );
    let vectors = scalar_vectors();
    ensure!(
        file.fixture_sha256 == SCALAR_DIGEST
            && file.vectors == vectors
            && manifest.scalar_contract.vectors == vectors,
        "scalar fixture vectors differ from compiled Rust v2 fixture"
    );
    ensure!(
        manifest.scalar_contract.artifact == manifest.fixture,
        "scalar artifact identity differs from fixture claim"
    );
    Ok(())
}

fn dtype_name(dtype: qwen_llm::tensor::GgmlType) -> Result<&'static str> {
    use qwen_llm::tensor::GgmlType;
    let (name, id) = match dtype {
        GgmlType::F32 => ("F32", 0),
        GgmlType::F16 => ("F16", 1),
        GgmlType::Q8_0 => ("Q8_0", 8),
        GgmlType::Q4_K => (K0S_SELECTOR_WEIGHT_DTYPE, K0S_SELECTOR_WEIGHT_DTYPE_ID),
        GgmlType::BF16 => ("BF16", 30),
        _ => return Err(anyhow!("dtype is outside the K0-S evidence schema")),
    };
    ensure!(
        dtype as i32 == id && dtype.wire_name() == name,
        "GGML dtype name/discriminant mismatch"
    );
    Ok(name)
}

fn require_q4_selector_dtype(dtype: qwen_llm::tensor::GgmlType) -> Result<()> {
    ensure!(
        dtype_name(dtype)? == K0S_SELECTOR_WEIGHT_DTYPE
            && dtype as i32 == K0S_SELECTOR_WEIGHT_DTYPE_ID,
        "K0-S v1 selector-hidden dtype must be Q4_K / GGML type 12"
    );
    Ok(())
}

fn require_q4_selector_name(dtype: &str) -> Result<()> {
    ensure!(
        dtype == K0S_SELECTOR_WEIGHT_DTYPE,
        "K0-S v1 selector-hidden dtype must be Q4_K"
    );
    Ok(())
}

fn validate_tensor_claims_before_metal(manifest: &Manifest, drafter: &GgufFile) -> Result<()> {
    for claim in &manifest.tensors {
        ensure!(
            claim.asset_role == "drafter",
            "all K0-S tensors must bind the drafter"
        );
        let descriptor = drafter
            .find(&claim.name)
            .with_context(|| format!("missing tensor {}", claim.name))?;
        if claim.role == "selector_hidden" {
            require_q4_selector_name(&claim.dtype)?;
            require_q4_selector_dtype(descriptor.dtype)?;
        }
        ensure!(
            descriptor.shape == claim.shape
                && dtype_name(descriptor.dtype)? == claim.dtype
                && descriptor.data_offset == claim.offset
                && descriptor.n_bytes == claim.bytes,
            "tensor descriptor mismatch for {}",
            claim.role
        );
        let bytes = drafter.try_slice(descriptor)?;
        ensure!(
            hex(&Sha256::digest(bytes)) == claim.sha256,
            "tensor hash mismatch for {}",
            claim.role
        );
        match claim.role.as_str() {
            "selector_hidden" => ensure!(
                claim.name == "selector_hidden.weight"
                    && claim.dtype == K0S_SELECTOR_WEIGHT_DTYPE
                    && claim.shape == [DFLASH_K0S_HIDDEN as u64, DFLASH_K0S_RANK as u64]
                    && claim.orientation == "gguf_ne0_hidden_ne1_rank"
                    && claim.row_domain.is_none(),
                "selector-hidden claim mismatch"
            ),
            "predecessor" => ensure!(
                claim.name == "selector_predecessor.weight"
                    && claim.shape == [DFLASH_K0S_RANK as u64, DFLASH_K0S_VOCAB as u64]
                    && claim.orientation == "gguf_ne0_rank_ne1_token"
                    && claim.row_domain
                        == Some(RowDomain {
                            first: 0,
                            count: DFLASH_K0S_VOCAB as u64
                        }),
                "predecessor claim mismatch"
            ),
            "successor" => ensure!(
                claim.name == "selector_successor.weight"
                    && claim.shape == [DFLASH_K0S_RANK as u64, DFLASH_K0S_VOCAB as u64]
                    && claim.orientation == "gguf_ne0_rank_ne1_token"
                    && claim.row_domain
                        == Some(RowDomain {
                            first: 0,
                            count: DFLASH_K0S_VOCAB as u64
                        }),
                "successor claim mismatch"
            ),
            _ => return Err(anyhow!("unknown tensor role")),
        };
    }
    Ok(())
}

#[derive(Serialize)]
struct Record<T> {
    schema: &'static str,
    schema_version: u32,
    run_id: String,
    attempt_id: String,
    event: &'static str,
    payload: T,
}

#[derive(Serialize)]
struct Geometry {
    block_size: usize,
    depths: usize,
    top_k: usize,
    rank: usize,
    hidden: usize,
    vocab: usize,
    rows: usize,
}
#[derive(Serialize)]
struct Abstention {
    enabled: bool,
    p_min: Option<u8>,
    n_min: Option<u8>,
}
#[derive(Clone, Serialize, PartialEq, Eq)]
struct KernelTrace {
    encoders: u64,
    concurrent_encoders: u64,
    dispatches: u64,
}
#[derive(Serialize)]
struct Provenance {
    dispatch_census: Vec<DispatchClaim>,
    selector_hidden_dispatch: DispatchClaim,
    kernel_trace: KernelTrace,
    embedded_metallib_sha256: String,
    build: BuildClaim,
    host: HostClaim,
    environment: BTreeMap<String, String>,
}
#[derive(Serialize)]
struct CaptureState {
    target_context_len: usize,
    context_hidden_watermark: usize,
    kv_context_watermark: usize,
    noise_input_sha256: String,
    synchronized_event_sha256: String,
    diagnostic_state_sha256: String,
}
#[derive(Serialize)]
struct CaptureJson {
    definition_version: &'static str,
    noise_start_position: u32,
    carry_token: i32,
    synchronized_capture_sha256: String,
    draft_tokens: Vec<i32>,
    draft_token_bits: Vec<String>,
    draft_tokens_sha256_i32le: String,
    state: CaptureState,
}
#[derive(Serialize)]
struct Identities {
    sidecar: FileClaim,
    reducer: FileClaim,
    executable: FileClaim,
    fixture: FileClaim,
    command: FileClaim,
    sources: Vec<RoleClaim>,
    assets: Vec<RoleClaim>,
}
#[derive(Clone, Serialize)]
struct SidecarRange {
    id: String,
    kind: &'static str,
    dtype: String,
    shape: Vec<usize>,
    offset: u64,
    bytes: u64,
    sha256: String,
    tensor_role: Option<&'static str>,
    row: Option<i32>,
}
#[derive(Serialize)]
struct ChainJson {
    name: String,
    initial_carry: i32,
    slots: Vec<usize>,
    events: Vec<ChainEventJson>,
    tokens: Vec<i32>,
    terminated: bool,
}
#[derive(Serialize)]
struct ChainEventJson {
    kind: &'static str,
    depth: usize,
    token: i32,
    slot: Option<usize>,
}
#[derive(Serialize)]
struct RunPayload {
    authority: &'static str,
    attempt_id: String,
    geometry: Geometry,
    request: RequestClaim,
    proposal_abstention: Abstention,
    ignored_target_policy: IgnoredPolicies,
    binding: Binding,
    provenance: Provenance,
    capture: CaptureJson,
    diagnostic_nonperturbation_parity: ParityJson,
    on_b_projection: OnBProjection,
    identities: Identities,
    semantic_references: SemanticReferences,
    tensors: Vec<TensorClaim>,
    sidecar_registry: Vec<SidecarRange>,
    production_chain: ChainJson,
    fixed_chains: Vec<ChainJson>,
}
#[derive(Clone, Serialize, PartialEq, Eq)]
struct ArmPhaseJson {
    carry_token: i32,
    noise_start_position: u32,
    target_sha256: String,
    dflash_sha256: String,
    state_sha256: String,
    draft_tokens_count: usize,
    draft_tokens_sha256_i32le: String,
    full_logits_count: usize,
    full_logits_sha256_f32le: String,
    topk_count: usize,
    topk_sha256_i32le: String,
    unary_count: usize,
    unary_sha256_f32le: String,
    z_count: usize,
    z_sha256_f32le: String,
    dispatch_census: Vec<DispatchClaim>,
    kernel_trace: KernelTrace,
    runtime_selector_contract: SelectorDispatchPredicate,
}
#[derive(Clone, Serialize, PartialEq, Eq)]
struct ObserverBaselineJson {
    before_sha256: String,
    after_sha256: String,
    restored: bool,
}
#[derive(Clone, Serialize, PartialEq, Eq)]
struct ArmSummaryJson {
    domain: &'static str,
    first: ArmPhaseJson,
    continuation: ArmPhaseJson,
    observer_baseline: ObserverBaselineJson,
    common_production_content_sha256: String,
    capture_content_sha256: Option<String>,
}
#[derive(Clone, Serialize, PartialEq, Eq)]
struct ArmJson {
    name: &'static str,
    diagnostic: bool,
    session_id: String,
    first_event: ArmEventJson,
    continuation_event: ArmEventJson,
    summary: ArmSummaryJson,
    rng_domains: Vec<RngDomainJson>,
    capture_projection_sha256: Option<String>,
    arm_envelope_sha256: String,
}
#[derive(Clone, Serialize, PartialEq, Eq)]
struct ArmEventJson {
    kind: &'static str,
    library_sequence: Option<u64>,
    library_event_envelope_sha256: Option<String>,
    session_binding_sha256: Option<String>,
    draft_tokens: Option<Vec<i32>>,
    wrapper_binding_sha256: String,
}
#[derive(Clone, Serialize, PartialEq, Eq)]
struct RngDomainJson {
    domain: String,
    scope: &'static str,
    absent_state_sha256: String,
    before_counter: u64,
    after_counter: u64,
}
#[derive(Serialize)]
struct ParityJson {
    status: &'static str,
    arm_order: [&'static str; 4],
    selected_arm: &'static str,
    arms: Vec<ArmJson>,
    comparison_fields: [&'static str; 5],
}
#[derive(Serialize)]
struct OnBProjection {
    exclusion_allowlist: [&'static str; 3],
    exclusion_content_sha256: String,
    depths: Vec<Value>,
    lattices: Vec<Value>,
    capture: Value,
    production_chain: Value,
    fixed_chains: Value,
    provenance: Value,
    projection_sha256: String,
}
#[derive(Serialize)]
struct FailureIdentities {
    reducer: FileClaim,
    executable: FileClaim,
    fixture: FileClaim,
    command: FileClaim,
    sources: Vec<RoleClaim>,
    assets: Vec<RoleClaim>,
    build: BuildClaim,
    host: HostClaim,
    embedded_metallib_sha256: String,
}
#[derive(Serialize)]
struct ParityFailure {
    completed_arms: Vec<ArmJson>,
    failed_arm: Option<&'static str>,
    failed_stage: Option<&'static str>,
    first_mismatch: Option<String>,
    observer_cleanup: bool,
    identities: FailureIdentities,
    authority: &'static str,
    status: &'static str,
}
#[derive(Serialize)]
struct DepthPayload {
    depth: usize,
    position: u32,
    production_call_id: String,
    drafter_checkpoint_sha256: String,
    proposal_construction_id: String,
    noise_input_sha256: String,
    synchronized_capture_sha256: String,
    draft_tokens_sha256_i32le: String,
    diagnostic_state_sha256: String,
    z_f32_bits: Vec<String>,
    full_logits_range_id: String,
    top16_ids: Vec<i32>,
    unary_f32_bits: Vec<String>,
    topk_issues: Vec<TopKIssueJson>,
}
#[derive(Serialize)]
struct LatticePayload {
    depth: usize,
    rows: Vec<RowJson>,
}
#[derive(Serialize)]
struct RowJson {
    row_index: usize,
    predecessor_token: i32,
    predecessor_slot: Option<usize>,
    predecessor_raw_range_id: Option<String>,
    slots: Vec<SlotJson>,
    issues: Vec<RowIssueJson>,
    choice_slot: usize,
}
#[derive(Serialize)]
struct SlotJson {
    slot: usize,
    token: i32,
    unary_f32_bits: String,
    successor_raw_range_id: Option<String>,
    score_f32_bits: Option<String>,
    issues: Vec<SlotIssueJson>,
}
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TopKIssueJson {
    #[serde(rename = "nonfinite_logit")]
    NonFiniteLogit { token: usize, bits: String },
    IdMismatch {
        slot: usize,
        expected: i32,
        observed: i32,
    },
    UnaryMismatch {
        slot: usize,
        expected_bits: String,
        observed_bits: String,
    },
}
#[derive(Serialize, PartialEq, Eq, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SlotIssueJson {
    DuplicateId {
        slot: usize,
        token: i32,
        first_slot: usize,
    },
    Sentinel {
        slot: usize,
        token: i32,
    },
    NonfiniteScore {
        slot: usize,
        token: i32,
        classification: &'static str,
    },
}
#[derive(Serialize)]
struct RowIssueJson {
    kind: &'static str,
}
#[derive(Serialize)]
struct EndPayload {
    producer_status: &'static str,
    authority: &'static str,
}

fn dispatch(value: &DFlashK0sDispatchCensusRow) -> DispatchClaim {
    DispatchClaim {
        family: value.family.clone(),
        tag: value.tag.clone(),
        encoder_ordinal: value.encoder_ordinal,
        encoder_concurrent: value.encoder_concurrent,
        kernel: value.kernel.clone(),
        grid: value.grid,
        threads: value.threads,
        grid_threadgroups: value.grid_threadgroups,
        threadgroup_threads: value.threadgroup_threads,
    }
}

fn chain_event(value: &DFlashK0sChainEvent) -> ChainEventJson {
    match value {
        DFlashK0sChainEvent::InvalidCarry { depth, token_id } => ChainEventJson {
            kind: "invalid_carry",
            depth: *depth,
            token: *token_id,
            slot: None,
        },
        DFlashK0sChainEvent::MissingPredecessorRow {
            depth,
            token_id,
            predecessor_slot,
        } => ChainEventJson {
            kind: "missing_predecessor_row",
            depth: *depth,
            token: *token_id,
            slot: *predecessor_slot,
        },
        DFlashK0sChainEvent::SlotZeroTermination {
            depth,
            token_id,
            slot,
        } => ChainEventJson {
            kind: "slot_zero_termination",
            depth: *depth,
            token: *token_id,
            slot: Some(*slot),
        },
    }
}

fn chain_json(name: String, carry: i32, chain: &DFlashK0sChain) -> ChainJson {
    ChainJson {
        name,
        initial_carry: carry,
        slots: chain.requested_slots.clone(),
        events: chain.event.iter().map(chain_event).collect(),
        tokens: chain.tokens.clone(),
        terminated: chain.terminated,
    }
}

fn topk_issues(values: &[DFlashK0sTopKIssue]) -> Vec<TopKIssueJson> {
    values
        .iter()
        .map(|issue| match issue {
            DFlashK0sTopKIssue::NonFiniteLogit {
                token_id,
                bits: raw,
            } => TopKIssueJson::NonFiniteLogit {
                token: *token_id,
                bits: bits(*raw),
            },
            DFlashK0sTopKIssue::IdMismatch {
                slot,
                expected,
                observed,
            } => TopKIssueJson::IdMismatch {
                slot: *slot,
                expected: *expected,
                observed: *observed,
            },
            DFlashK0sTopKIssue::UnaryMismatch {
                slot,
                expected_bits,
                observed_bits,
            } => TopKIssueJson::UnaryMismatch {
                slot: *slot,
                expected_bits: bits(*expected_bits),
                observed_bits: bits(*observed_bits),
            },
        })
        .collect()
}

fn class(value: DFlashK0sNonFiniteClass) -> &'static str {
    match value {
        DFlashK0sNonFiniteClass::PositiveInfinity => "positive_infinity",
        DFlashK0sNonFiniteClass::NegativeInfinity => "negative_infinity",
        DFlashK0sNonFiniteClass::Nan => "nan",
    }
}

fn split_issues(
    issues: &[DFlashK0sSlotIssue],
    slot_count: usize,
) -> Result<(Vec<Vec<SlotIssueJson>>, Vec<RowIssueJson>)> {
    let mut slots = (0..slot_count).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut rows = Vec::new();
    for issue in issues {
        match issue {
            DFlashK0sSlotIssue::DuplicateId {
                candidate_slot,
                first_slot,
                token_id,
            } => slots
                .get_mut(*candidate_slot)
                .context("duplicate issue slot out of range")?
                .push(SlotIssueJson::DuplicateId {
                    slot: *candidate_slot,
                    token: *token_id,
                    first_slot: *first_slot,
                }),
            DFlashK0sSlotIssue::Sentinel {
                candidate_slot,
                token_id,
            } => slots
                .get_mut(*candidate_slot)
                .context("sentinel issue slot out of range")?
                .push(SlotIssueJson::Sentinel {
                    slot: *candidate_slot,
                    token: *token_id,
                }),
            DFlashK0sSlotIssue::NonFiniteScore {
                candidate_slot,
                token_id,
                class: category,
                ..
            } => slots
                .get_mut(*candidate_slot)
                .context("nonfinite issue slot out of range")?
                .push(SlotIssueJson::NonfiniteScore {
                    slot: *candidate_slot,
                    token: *token_id,
                    classification: class(*category),
                }),
            DFlashK0sSlotIssue::NoValidChoice => rows.push(RowIssueJson {
                kind: "no_valid_choice",
            }),
        }
    }
    Ok((slots, rows))
}

fn raw_key(raw: &DFlashK0sRawRow) -> (usize, Option<usize>, Option<usize>, u8) {
    (
        raw.depth,
        raw.predecessor_slot,
        raw.candidate_slot,
        match raw.side {
            DFlashK0sCodebookSide::Predecessor => 0,
            DFlashK0sCodebookSide::Successor => 1,
        },
    )
}

struct SidecarLayout {
    bytes: Vec<u8>,
    ranges: Vec<SidecarRange>,
    ids: HashMap<(usize, Option<usize>, Option<usize>, u8), String>,
    full_logits: Vec<String>,
}

fn append_range(
    layout: &mut SidecarLayout,
    id: String,
    kind: &'static str,
    dtype: String,
    shape: Vec<usize>,
    material: &[u8],
    tensor_role: Option<&'static str>,
    row: Option<i32>,
) -> Result<()> {
    ensure!(
        !layout.ranges.iter().any(|range| range.id == id),
        "duplicate sidecar occurrence id"
    );
    let offset = u64::try_from(layout.bytes.len())?;
    let count = u64::try_from(material.len())?;
    let end = offset
        .checked_add(count)
        .context("sidecar offset overflow")?;
    ensure!(end <= MAX_SIDECAR, "sidecar exceeds authorization cap");
    layout.bytes.extend_from_slice(material);
    layout.ranges.push(SidecarRange {
        id,
        kind,
        dtype,
        shape,
        offset,
        bytes: count,
        sha256: hex(&Sha256::digest(material)),
        tensor_role,
        row,
    });
    Ok(())
}

fn build_sidecar_prefixed(
    capture: &DFlashK0sCapture,
    tensors: &[TensorClaim],
    prefix: &str,
) -> Result<SidecarLayout> {
    ensure!(
        capture.depths.len() == 7 && capture.lattice.len() == DFLASH_K0S_LATTICE_ROWS,
        "capture geometry malformed"
    );
    let mut layout = SidecarLayout {
        bytes: Vec::new(),
        ranges: Vec::new(),
        ids: HashMap::new(),
        full_logits: Vec::new(),
    };
    for depth in &capture.depths {
        ensure!(
            depth.full_logits_bits.len() == DFLASH_K0S_VOCAB,
            "full logits geometry malformed"
        );
        let id = format!("{prefix}full-logits-d{:02}", depth.depth);
        let mut material = Vec::with_capacity(depth.full_logits_bits.len() * 4);
        for raw in &depth.full_logits_bits {
            material.extend_from_slice(&raw.to_le_bytes());
        }
        append_range(
            &mut layout,
            id.clone(),
            "full_logits",
            "F32".into(),
            vec![DFLASH_K0S_VOCAB],
            &material,
            None,
            None,
        )?;
        layout.full_logits.push(id);
    }
    let dtype = |role: &str| {
        tensors
            .iter()
            .find(|tensor| tensor.role == role)
            .map(|tensor| tensor.dtype.clone())
            .with_context(|| format!("missing {role} tensor"))
    };
    let mut raw_iter = capture.raw_rows.iter();
    for row in &capture.lattice {
        let pred_valid = (0..DFLASH_K0S_VOCAB as i32).contains(&row.predecessor_token);
        if pred_valid {
            let raw = raw_iter
                .next()
                .context("missing predecessor raw occurrence")?;
            ensure!(
                raw.side == DFlashK0sCodebookSide::Predecessor
                    && raw.depth == row.depth
                    && raw.predecessor_slot == row.predecessor_slot
                    && raw.candidate_slot.is_none()
                    && raw.token_id == row.predecessor_token,
                "predecessor raw occurrence association mismatch"
            );
            let id = format!("{prefix}predecessor-r{:03}", row.row_index);
            append_range(
                &mut layout,
                id.clone(),
                "predecessor_row",
                dtype("predecessor")?,
                vec![DFLASH_K0S_RANK],
                &raw.bytes,
                Some("predecessor"),
                Some(raw.token_id),
            )?;
            ensure!(
                layout.ids.insert(raw_key(raw), id).is_none(),
                "predecessor physical alias"
            );
        }
        for slot in &row.slots {
            let valid = pred_valid && (0..DFLASH_K0S_VOCAB as i32).contains(&slot.token_id);
            if valid {
                let raw = raw_iter
                    .next()
                    .context("missing successor raw occurrence")?;
                ensure!(
                    raw.side == DFlashK0sCodebookSide::Successor
                        && raw.depth == row.depth
                        && raw.predecessor_slot == row.predecessor_slot
                        && raw.candidate_slot == Some(slot.candidate_slot)
                        && raw.token_id == slot.token_id,
                    "successor raw occurrence association mismatch"
                );
                let id = format!(
                    "{prefix}successor-r{:03}-s{:02}",
                    row.row_index, slot.candidate_slot
                );
                append_range(
                    &mut layout,
                    id.clone(),
                    "successor_row",
                    dtype("successor")?,
                    vec![DFLASH_K0S_RANK],
                    &raw.bytes,
                    Some("successor"),
                    Some(raw.token_id),
                )?;
                ensure!(
                    layout.ids.insert(raw_key(raw), id).is_none(),
                    "successor physical alias"
                );
            }
        }
    }
    ensure!(
        raw_iter.next().is_none(),
        "capture contains trailing raw rows"
    );
    ensure!(
        layout.ranges.len() <= 4096 && layout.bytes.len() as u64 <= MAX_SIDECAR,
        "sidecar registry cap exceeded"
    );
    Ok(layout)
}

#[cfg(test)]
fn build_sidecar(capture: &DFlashK0sCapture, tensors: &[TensorClaim]) -> Result<SidecarLayout> {
    build_sidecar_prefixed(capture, tensors, "")
}

fn append_sidecar_layout(base: &mut SidecarLayout, mut extra: SidecarLayout) -> Result<()> {
    let offset = u64::try_from(base.bytes.len())?;
    for range in &mut extra.ranges {
        range.offset = range
            .offset
            .checked_add(offset)
            .context("sidecar offset overflow")?;
    }
    ensure!(
        base.bytes
            .len()
            .checked_add(extra.bytes.len())
            .is_some_and(|n| n as u64 <= MAX_SIDECAR),
        "combined sidecar exceeds cap"
    );
    base.bytes.extend(extra.bytes);
    base.ranges.extend(extra.ranges);
    Ok(())
}

fn lattice_json(
    capture: &DFlashK0sCapture,
    layout: &SidecarLayout,
    depth: usize,
) -> Result<LatticePayload> {
    let rows = capture
        .lattice
        .iter()
        .filter(|row| row.depth == depth)
        .map(|row| -> Result<RowJson> {
            let (mut issue_slots, row_issues) = split_issues(&row.issues, row.slots.len())?;
            let predecessor_raw_range_id = layout
                .ids
                .get(&(row.depth, row.predecessor_slot, None, 0))
                .cloned();
            let slots = row
                .slots
                .iter()
                .map(|slot| SlotJson {
                    slot: slot.candidate_slot,
                    token: slot.token_id,
                    unary_f32_bits: bits(slot.unary_bits),
                    successor_raw_range_id: layout
                        .ids
                        .get(&(
                            row.depth,
                            row.predecessor_slot,
                            Some(slot.candidate_slot),
                            1,
                        ))
                        .cloned(),
                    score_f32_bits: slot.score_bits.map(bits),
                    issues: std::mem::take(&mut issue_slots[slot.candidate_slot]),
                })
                .collect();
            Ok(RowJson {
                row_index: row.row_index,
                predecessor_token: row.predecessor_token,
                predecessor_slot: row.predecessor_slot,
                predecessor_raw_range_id,
                slots,
                issues: row_issues,
                choice_slot: row.greedy_slot,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        rows.len() == if depth == 1 { 1 } else { 16 },
        "lattice depth row count malformed"
    );
    Ok(LatticePayload { depth, rows })
}

fn local_capture_sha(capture: &DFlashK0sCapture) -> [u8; 32] {
    fn hash_bytes(hash: &mut Sha256, value: &[u8]) {
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value);
    }
    fn hash_dispatch(hash: &mut Sha256, row: &DFlashK0sDispatchCensusRow) {
        hash_bytes(hash, row.family.as_bytes());
        match &row.tag {
            Some(tag) => {
                hash.update([1]);
                hash_bytes(hash, tag.as_bytes());
            }
            None => hash.update([0]),
        }
        hash.update(row.encoder_ordinal.to_le_bytes());
        hash.update([u8::from(row.encoder_concurrent)]);
        hash_bytes(hash, row.kernel.as_bytes());
        for value in row.grid.into_iter().chain(row.threads) {
            hash.update(value.to_le_bytes());
        }
        hash.update(row.grid_threadgroups.to_le_bytes());
        hash.update(row.threadgroup_threads.to_le_bytes());
    }
    let mut hash = Sha256::new();
    hash.update(CAPTURE_DEFINITION.as_bytes());
    let state = &capture.state_identity;
    hash.update(state.carry_token.to_le_bytes());
    hash.update(state.noise_start_position.to_le_bytes());
    for value in [
        state.target_context_len,
        state.context_hidden_watermark,
        state.kv_context_watermark,
    ] {
        hash.update((value as u64).to_le_bytes());
    }
    hash.update(state.draft_tokens_sha256);
    hash.update(state.noise_input_sha256);
    hash.update(state.synchronized_event_sha256);
    hash.update(state.diagnostic_state_sha256);
    hash.update((capture.draft_token_bits.len() as u64).to_le_bytes());
    for raw in &capture.draft_token_bits {
        hash.update(raw.to_le_bytes());
    }
    hash.update((capture.depths.len() as u64).to_le_bytes());
    for depth in &capture.depths {
        hash.update((depth.depth as u64).to_le_bytes());
        hash.update(depth.position.to_le_bytes());
        hash.update((depth.full_logits_bits.len() as u64).to_le_bytes());
        for raw in &depth.full_logits_bits {
            hash.update(raw.to_le_bytes());
        }
        hash.update((depth.top_k_ids.len() as u64).to_le_bytes());
        for token in &depth.top_k_ids {
            hash.update(token.to_le_bytes());
        }
        hash.update((depth.unary_bits.len() as u64).to_le_bytes());
        for raw in &depth.unary_bits {
            hash.update(raw.to_le_bytes());
        }
        hash.update((depth.selector_hidden_bits.len() as u64).to_le_bytes());
        for raw in &depth.selector_hidden_bits {
            hash.update(raw.to_le_bytes());
        }
    }
    hash.update((capture.dispatch_census.len() as u64).to_le_bytes());
    for row in &capture.dispatch_census {
        hash_dispatch(&mut hash, row);
    }
    hash_dispatch(&mut hash, &capture.selector_hidden_dispatch);
    hash.update(capture.kernel_trace.encoders.to_le_bytes());
    hash.update(capture.kernel_trace.concurrent_encoders.to_le_bytes());
    hash.update(capture.kernel_trace.dispatches.to_le_bytes());
    hash.update(capture.provenance.selector_hidden.full_tensor_sha256);
    hash.update(capture.provenance.predecessor.full_tensor_sha256);
    hash.update(capture.provenance.successor.full_tensor_sha256);
    hash.update(capture.provenance.embedded_metallib_sha256);
    hash.finalize().into()
}

fn tensor_capture_matches(capture: &DFlashK0sCapture, tensors: &[TensorClaim]) -> bool {
    let values = [
        ("selector_hidden", &capture.provenance.selector_hidden),
        ("predecessor", &capture.provenance.predecessor),
        ("successor", &capture.provenance.successor),
    ];
    values.into_iter().all(|(role, value)| {
        tensors
            .iter()
            .find(|claim| claim.role == role)
            .is_some_and(|claim| {
                value.descriptor.name == claim.name
                    && value.descriptor.shape == claim.shape
                    && dtype_name(value.descriptor.dtype).is_ok_and(|name| name == claim.dtype)
                    && value.descriptor.data_offset == claim.offset
                    && value.descriptor.n_bytes == claim.bytes
                    && hex(&value.full_tensor_sha256) == claim.sha256
            })
    })
}

fn serialize_record<T: Serialize>(
    run_id: &str,
    attempt_id: &str,
    event: &'static str,
    payload: T,
) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(&Record {
        schema: SCHEMA,
        schema_version: SCHEMA_VERSION,
        run_id: run_id.to_owned(),
        attempt_id: attempt_id.to_owned(),
        event,
        payload,
    })?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn write_parity_failure(
    outputs: &mut ReservedOutputs,
    manifest: &Manifest,
    completed_arms: Vec<ArmJson>,
    failed_arm: Option<&'static str>,
    failed_stage: &'static str,
    error: &anyhow::Error,
) -> Result<()> {
    let decision = failure_disposition(&completed_arms, failed_arm, failed_stage)?;
    ensure!(
        outputs.sidecar.metadata()?.len() == 0,
        "failure sidecar is not empty"
    );
    ensure!(
        outputs.trace.metadata()?.len() == 0,
        "failure trace already contains a terminal record"
    );
    outputs.sidecar.flush()?;
    outputs.sidecar.sync_all()?;
    let row = serialize_record(
        &manifest.run_id,
        &manifest.attempt_id,
        "parity_failure",
        ParityFailure {
            completed_arms,
            failed_arm: decision.failed_arm,
            failed_stage: Some(decision.failed_stage),
            first_mismatch: Some(error.to_string().chars().take(1024).collect()),
            observer_cleanup: diagnostics_observer_active_counts() == [0, 0, 0],
            identities: FailureIdentities {
                reducer: manifest.reducer.clone(),
                executable: manifest.executable.clone(),
                fixture: manifest.fixture.clone(),
                command: manifest.command.clone(),
                sources: manifest.sources.clone(),
                assets: manifest.assets.clone(),
                build: manifest.expected_build.clone(),
                host: manifest.expected_host.clone(),
                embedded_metallib_sha256: manifest.embedded_metallib_sha256.clone(),
            },
            authority: AUTHORITY,
            status: "failed",
        },
    )?;
    ensure!(
        row.len() as u64 <= manifest.trace_max_bytes,
        "failure record exceeds trace cap"
    );
    outputs.trace.write_all(&row)?;
    outputs.trace.flush()?;
    outputs.trace.sync_all()?;
    ensure!(
        outputs.trace.metadata()?.len() == row.len() as u64,
        "failure trace size mismatch"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FailureDisposition {
    failed_arm: Option<&'static str>,
    failed_stage: &'static str,
}

fn failure_disposition(
    completed: &[ArmJson],
    failed_arm: Option<&'static str>,
    failed_stage: &'static str,
) -> Result<FailureDisposition> {
    validate_failure_shape(completed, failed_arm, failed_stage)?;
    Ok(FailureDisposition {
        failed_arm,
        failed_stage,
    })
}

fn validate_failure_shape(
    completed: &[ArmJson],
    failed_arm: Option<&str>,
    failed_stage: &str,
) -> Result<()> {
    validate_failure_names(
        &completed.iter().map(|arm| arm.name).collect::<Vec<_>>(),
        failed_arm,
        failed_stage,
    )
}

fn validate_failure_names(
    completed: &[&str],
    failed_arm: Option<&str>,
    failed_stage: &str,
) -> Result<()> {
    let order = ["off-A", "on-A", "on-B", "off-B"];
    const IN_ARM_STAGES: [&str; 6] = [
        "session_setup",
        "prompt_prefill",
        "first_block",
        "extraction",
        "continuation",
        "observer_cleanup",
    ];
    ensure!(
        completed.len() <= order.len()
            && completed
                .iter()
                .enumerate()
                .all(|(index, arm)| *arm == order[index]),
        "completed failure arms are not an exact order prefix"
    );
    if matches!(failed_stage, "comparison" | "serialization") {
        ensure!(
            completed.len() == 4 && failed_arm.is_none(),
            "post-four-arm failure shape invalid"
        );
    } else {
        ensure!(
            IN_ARM_STAGES.contains(&failed_stage),
            "unknown in-arm failure stage"
        );
        ensure!(
            completed.len() < 4 && failed_arm == Some(order[completed.len()]),
            "in-arm failure shape invalid"
        );
    }
    Ok(())
}

fn classify_arm_failure(detail: &str) -> &'static str {
    if detail.contains("observer") {
        "observer_cleanup"
    } else if detail.contains("[session_setup]") {
        "session_setup"
    } else if detail.contains("[prompt_prefill]") {
        "prompt_prefill"
    } else if detail.contains("[extraction]") {
        "extraction"
    } else if detail.contains("[continuation]") {
        "continuation"
    } else {
        "first_block"
    }
}

fn validate_artifact_caps(
    sidecar_bytes: u64,
    trace_bytes: u64,
    sidecar_max_bytes: u64,
    trace_max_bytes: u64,
) -> Result<()> {
    ensure!(
        sidecar_bytes <= sidecar_max_bytes && sidecar_bytes <= MAX_SIDECAR,
        "sidecar layout exceeds manifest or authorization cap"
    );
    ensure!(
        trace_bytes <= trace_max_bytes && trace_bytes <= MAX_TRACE,
        "trace exceeds manifest or authorization cap"
    );
    let combined = sidecar_bytes
        .checked_add(trace_bytes)
        .context("combined artifact size overflow")?;
    let manifest_combined = sidecar_max_bytes
        .checked_add(trace_max_bytes)
        .context("combined manifest cap overflow")?;
    ensure!(
        combined <= manifest_combined && combined <= MAX_COMBINED,
        "combined artifacts exceed manifest or authorization cap"
    );
    Ok(())
}

fn claim_shape(claim: &FileClaim, label: &str) -> Result<()> {
    ensure!(
        claim.max_bytes > 0
            && claim.bytes <= claim.max_bytes
            && valid_sha(&claim.sha256)
            && Path::new(&claim.path).is_absolute(),
        "{label} claim is invalid"
    );
    Ok(())
}

fn verify_inventory_build(build: &BuildClaim, identity: &Value) -> Result<()> {
    ensure!(
        build.profile == "release"
            && build.features == ["dflash-k0s-diagnostics"]
            && !build.dirty
            && build.target == compiled_target(),
        "inventory requires the frozen clean release build"
    );
    let commit = identity
        .get("build_commit")
        .and_then(Value::as_str)
        .context("compiled build commit absent")?;
    let source = identity
        .get("build_source_state")
        .and_then(Value::as_str)
        .and_then(|value| value.strip_prefix("git-source-sha256-v2:"))
        .context("compiled source identity absent")?;
    ensure!(
        identity.get("status").and_then(Value::as_str) == Some("match")
            && identity.get("build_dirty").and_then(Value::as_bool) == Some(false)
            && identity.get("runtime_dirty").and_then(Value::as_bool) == Some(false)
            && build.commit == commit
            && build.source_sha256 == source,
        "inventory compiled/runtime build identity mismatch"
    );
    Ok(())
}

fn validate_external_metallib_claim(claim: &FileClaim) -> Result<()> {
    let compiled = dflash_k0s_embedded_metallib_identity();
    let bytes = dflash_k0s_embedded_metallib_bytes();
    ensure!(
        compiled.byte_count == bytes.len()
            && compiled.sha256 == <[u8; 32]>::from(Sha256::digest(bytes))
            && claim.bytes == compiled.byte_count as u64
            && claim.sha256 == hex(&compiled.sha256),
        "external embedded-metallib identity differs from compiled bytes"
    );
    Ok(())
}

fn derive_checkout() -> Result<CheckoutClaim> {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = crate::source_identity::git_text(repo, &["rev-parse", "--show-toplevel"])
        .context("derive checkout root")?;
    let path = PathBuf::from(path).canonicalize()?;
    let commit = crate::source_identity::git_text(&path, &["rev-parse", "HEAD"])
        .context("derive checkout commit")?;
    let tree = crate::source_identity::git_text(&path, &["rev-parse", "HEAD^{tree}"])
        .context("derive checkout tree")?;
    let status = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=all"])
        .current_dir(&path)
        .output()
        .context("derive complete checkout status")?;
    ensure!(status.status.success(), "git status failed");
    let dirty = crate::source_identity::git_dirty(&path).context("derive checkout dirty state")?
        || !status.stdout.is_empty();
    ensure!(
        valid_git_oid(&commit) && valid_git_oid(&tree),
        "invalid Git identity"
    );
    Ok(CheckoutClaim {
        path: path.to_string_lossy().into_owned(),
        commit,
        tree,
        dirty,
    })
}

fn canonical_json_sha(value: &Value) -> Result<String> {
    Ok(hex(&Sha256::digest(serde_json::to_vec(value)?)))
}

fn canonical_domain_digest<T: Serialize>(domain: &str, value: &T) -> Result<String> {
    ensure!(domain.is_ascii(), "canonical digest domain must be ASCII");
    let payload = serde_json::to_vec(value)?;
    ensure!(
        payload.is_ascii(),
        "canonical digest JSON must be compact ASCII"
    );
    let mut digest = Sha256::new();
    digest.update(domain.as_bytes());
    digest.update([0]);
    digest.update((payload.len() as u64).to_le_bytes());
    digest.update(payload);
    Ok(hex(&digest.finalize()))
}

#[derive(Serialize)]
struct EventWrapperDigest<'a> {
    attempt_id: &'a str,
    arm: &'a str,
    session_id: &'a str,
    phase: &'a str,
    kind: &'a str,
    library_sequence: Option<u64>,
    library_event_envelope_sha256: Option<&'a str>,
    session_binding_sha256: Option<&'a str>,
    draft_tokens: Option<&'a [i32]>,
}

fn bind_event(
    attempt_id: &str,
    arm: &str,
    session_id: &str,
    phase: &'static str,
    kind: &'static str,
    library_sequence: Option<u64>,
    library_event_envelope_sha256: Option<String>,
    session_binding_sha256: Option<String>,
    draft_tokens: Option<Vec<i32>>,
) -> Result<ArmEventJson> {
    let observed = kind == "observed";
    ensure!(
        observed
            == (library_sequence.is_some()
                && library_event_envelope_sha256.is_some()
                && session_binding_sha256.is_some()
                && draft_tokens
                    .as_ref()
                    .is_some_and(|tokens| tokens.len() == 8)),
        "event observation identity is incomplete"
    );
    let wrapper = canonical_domain_digest(
        "qwen.dflash_k0s.event_wrapper.v1",
        &EventWrapperDigest {
            attempt_id,
            arm,
            session_id,
            phase,
            kind,
            library_sequence,
            library_event_envelope_sha256: library_event_envelope_sha256.as_deref(),
            session_binding_sha256: session_binding_sha256.as_deref(),
            draft_tokens: draft_tokens.as_deref(),
        },
    )?;
    Ok(ArmEventJson {
        kind,
        library_sequence,
        library_event_envelope_sha256,
        session_binding_sha256,
        draft_tokens,
        wrapper_binding_sha256: wrapper,
    })
}

#[derive(Serialize)]
struct RngAbsentDigest<'a> {
    attempt_id: &'a str,
    arm: &'a str,
    domain: &'a str,
    scope: &'static str,
}

fn arm_rng_domains(attempt_id: &str, arm: &str, domains: &[String]) -> Result<Vec<RngDomainJson>> {
    const SCOPE: &str = "statically_unreachable_no_rng_object_constructed";
    domains
        .iter()
        .map(|domain| {
            Ok(RngDomainJson {
                domain: domain.clone(),
                scope: SCOPE,
                absent_state_sha256: canonical_domain_digest(
                    "qwen.dflash_k0s.rng_absent.v1",
                    &RngAbsentDigest {
                        attempt_id,
                        arm,
                        domain,
                        scope: SCOPE,
                    },
                )?,
                before_counter: 0,
                after_counter: 0,
            })
        })
        .collect()
}

#[derive(Serialize)]
struct CommonProductionDigest<'a> {
    first: &'a ArmPhaseJson,
    continuation: &'a ArmPhaseJson,
}

#[derive(Serialize)]
struct ArmEnvelopeDigest<'a> {
    attempt_id: &'a str,
    name: &'a str,
    diagnostic: bool,
    session_id: &'a str,
    first_event: &'a ArmEventJson,
    continuation_event: &'a ArmEventJson,
    summary: &'a ArmSummaryJson,
    rng_domains: &'a [RngDomainJson],
    capture_projection_sha256: Option<&'a str>,
}

fn arm_envelope_sha256(arm: &ArmJson, attempt_id: &str) -> Result<String> {
    canonical_domain_digest(
        "qwen.dflash_k0s.cli_arm_envelope.v1",
        &ArmEnvelopeDigest {
            attempt_id,
            name: arm.name,
            diagnostic: arm.diagnostic,
            session_id: &arm.session_id,
            first_event: &arm.first_event,
            continuation_event: &arm.continuation_event,
            summary: &arm.summary,
            rng_domains: &arm.rng_domains,
            capture_projection_sha256: arm.capture_projection_sha256.as_deref(),
        },
    )
}

fn refresh_arm_envelope(arm: &mut ArmJson, attempt_id: &str) -> Result<()> {
    arm.arm_envelope_sha256 = arm_envelope_sha256(arm, attempt_id)?;
    Ok(())
}

fn validate_capture_presence(present: &[bool]) -> Result<()> {
    ensure!(
        present == [false, true, true, false],
        "four-arm extraction presence mismatch"
    );
    Ok(())
}

fn phase_event_envelope_sha256(
    phase: &ArmPhaseJson,
    session_binding_sha256: &str,
    sequence: u64,
    draft_tokens: &[i32],
) -> Result<String> {
    let digest = |value: &str| -> Result<[u8; 32]> {
        hex_decode(value)?
            .try_into()
            .map_err(|_| anyhow!("event digest is not 32 bytes"))
    };
    let contract = &phase.runtime_selector_contract;
    ensure!(
        contract.weight_dtype == K0S_SELECTOR_WEIGHT_DTYPE
            && contract.weight_dtype_id == K0S_SELECTOR_WEIGHT_DTYPE_ID
            && contract.input_dtype == "F32"
            && contract.input_dtype_id == 0
            && contract.output_dtype == "F32"
            && contract.output_dtype_id == 0,
        "event selector dtype contract invalid"
    );
    let summary = DFlashK0sParitySummary {
        draft_tokens: draft_tokens.to_vec(),
        dispatch_census: phase
            .dispatch_census
            .iter()
            .map(|row| DFlashK0sDispatchCensusRow {
                family: row.family.clone(),
                tag: row.tag.clone(),
                encoder_ordinal: row.encoder_ordinal,
                encoder_concurrent: row.encoder_concurrent,
                kernel: row.kernel.clone(),
                grid: row.grid,
                threads: row.threads,
                grid_threadgroups: row.grid_threadgroups,
                threadgroup_threads: row.threadgroup_threads,
            })
            .collect(),
        kernel_trace: [
            phase.kernel_trace.encoders,
            phase.kernel_trace.concurrent_encoders,
            phase.kernel_trace.dispatches,
        ],
        selector_inputs: qwen_llm::metal_dflash::DFlashK0sSelectorInputIdentity {
            full_logits_count: phase.full_logits_count,
            full_logits_sha256_f32le: digest(&phase.full_logits_sha256_f32le)?,
            top_k_ids_count: phase.topk_count,
            top_k_ids_sha256_i32le: digest(&phase.topk_sha256_i32le)?,
            unary_count: phase.unary_count,
            unary_sha256_f32le: digest(&phase.unary_sha256_f32le)?,
            selector_hidden_count: phase.z_count,
            selector_hidden_sha256_f32le: digest(&phase.z_sha256_f32le)?,
        },
        selector_dispatch: qwen_llm::metal_dflash::DFlashK0sSelectorDispatchIdentity {
            weight_dtype: qwen_llm::tensor::GgmlType::Q4_K,
            input_dtype: qwen_llm::tensor::GgmlType::F32,
            output_dtype: qwen_llm::tensor::GgmlType::F32,
            block_size: usize::try_from(contract.n)?,
            hidden_size: usize::try_from(contract.h)?,
            selector_rank: usize::try_from(contract.r)?,
        },
        diagnostic_state_sha256: digest(&phase.dflash_sha256)?,
        carry_token: phase.carry_token,
        noise_start_position: phase.noise_start_position,
        session_binding_sha256: digest(session_binding_sha256)?,
    };
    Ok(hex(&dflash_k0s_event_envelope_sha256(&summary, sequence)))
}

fn validate_four_arm_gate(
    arms: &[ArmJson],
    attempt_id: &str,
    associations_complete: bool,
) -> Result<&'static str> {
    const ORDER: [(&str, bool); 4] = [
        ("off-A", false),
        ("on-A", true),
        ("on-B", true),
        ("off-B", false),
    ];
    ensure!(
        arms.len() == ORDER.len(),
        "four-arm gate requires four arms"
    );
    for (arm, (name, diagnostic)) in arms.iter().zip(ORDER) {
        ensure!(
            arm.name == name && arm.diagnostic == diagnostic,
            "four-arm order or mode mismatch"
        );
    }
    ensure!(
        arms.windows(2)
            .all(|pair| pair[0].summary.first == pair[1].summary.first),
        "four-arm first phase mismatch"
    );
    ensure!(
        arms.windows(2)
            .all(|pair| pair[0].summary.continuation == pair[1].summary.continuation),
        "four-arm continuation mismatch"
    );
    ensure!(
        arms.windows(2)
            .all(|pair| pair[0].summary.observer_baseline == pair[1].summary.observer_baseline),
        "four-arm observer baseline mismatch"
    );
    ensure!(
        arms.iter().all(|arm| {
            arm.summary.common_production_content_sha256
                == arms[0].summary.common_production_content_sha256
        }),
        "four-arm common production content mismatch"
    );
    ensure!(
        arms.iter()
            .all(|arm| arm.summary.observer_baseline.restored),
        "four-arm observer baseline is not restored"
    );

    let expected_domains = arms[0]
        .rng_domains
        .iter()
        .map(|rng| rng.domain.as_str())
        .collect::<Vec<_>>();
    ensure!(
        !expected_domains.is_empty()
            && expected_domains.iter().collect::<HashSet<_>>().len() == expected_domains.len(),
        "four-arm RNG domain list invalid"
    );
    let mut expected = 1u64;
    let mut library_envelopes = HashSet::new();
    let mut wrappers = HashSet::new();
    let mut sessions = HashSet::new();
    let mut arm_envelopes = HashSet::new();
    for arm in arms {
        ensure!(
            arm.summary.domain == "qwen.dflash_k0s.parity_arm_summary.v1"
                && arm.summary.common_production_content_sha256
                    == canonical_domain_digest(
                        "qwen.dflash_k0s.common_production.v1",
                        &CommonProductionDigest {
                            first: &arm.summary.first,
                            continuation: &arm.summary.continuation,
                        },
                    )?,
            "arm summary domain or common production digest invalid"
        );
        ensure!(
            sessions.insert(arm.session_id.as_str()),
            "four-arm sessions are not distinct"
        );
        ensure!(
            arm.arm_envelope_sha256 == arm_envelope_sha256(arm, attempt_id)?
                && arm_envelopes.insert(arm.arm_envelope_sha256.as_str()),
            "arm envelope is invalid or duplicated"
        );
        ensure!(
            arm.rng_domains.len() == expected_domains.len(),
            "arm RNG domain count mismatch"
        );
        for (rng, domain) in arm.rng_domains.iter().zip(&expected_domains) {
            let expected_rng = arm_rng_domains(attempt_id, arm.name, &[(*domain).to_owned()])?;
            ensure!(rng == &expected_rng[0], "arm RNG binding mismatch");
        }
        for (phase, event) in [
            ("first", &arm.first_event),
            ("continuation", &arm.continuation_event),
        ] {
            let phase_summary = if phase == "first" {
                &arm.summary.first
            } else {
                &arm.summary.continuation
            };
            let expected_kind = if phase == "first" && !arm.diagnostic {
                "plain"
            } else {
                "observed"
            };
            ensure!(event.kind == expected_kind, "arm event mode mismatch");
            match event.library_sequence {
                Some(sequence) => {
                    ensure!(event.kind == "observed", "plain event has library sequence");
                    ensure!(
                        sequence == expected,
                        "library event sequence is not contiguous"
                    );
                    expected += 1;
                    ensure!(
                        event
                            .library_event_envelope_sha256
                            .as_ref()
                            .is_some_and(|value| valid_sha(value)
                                && library_envelopes.insert(value.as_str())),
                        "library event envelope is absent, invalid, or duplicated"
                    );
                    ensure!(
                        event.session_binding_sha256.as_deref() == Some(arm.session_id.as_str())
                            && event
                                .draft_tokens
                                .as_ref()
                                .is_some_and(|tokens| tokens.len() == 8),
                        "observed event session or draft-token binding is invalid"
                    );
                    let expected_library = phase_event_envelope_sha256(
                        phase_summary,
                        &arm.session_id,
                        sequence,
                        event.draft_tokens.as_deref().unwrap(),
                    )?;
                    ensure!(
                        event.library_event_envelope_sha256.as_deref()
                            == Some(expected_library.as_str()),
                        "observed event library envelope differs from phase"
                    );
                    ensure!(
                        hash_i32(event.draft_tokens.as_deref().unwrap())
                            == phase_summary.draft_tokens_sha256_i32le,
                        "observed event draft tokens differ from phase"
                    );
                }
                None => ensure!(
                    event.kind == "plain"
                        && event.library_event_envelope_sha256.is_none()
                        && event.session_binding_sha256.is_none()
                        && event.draft_tokens.is_none(),
                    "plain event has library identity"
                ),
            }
            let expected_wrapper = canonical_domain_digest(
                "qwen.dflash_k0s.event_wrapper.v1",
                &EventWrapperDigest {
                    attempt_id,
                    arm: arm.name,
                    session_id: &arm.session_id,
                    phase,
                    kind: event.kind,
                    library_sequence: event.library_sequence,
                    library_event_envelope_sha256: event.library_event_envelope_sha256.as_deref(),
                    session_binding_sha256: event.session_binding_sha256.as_deref(),
                    draft_tokens: event.draft_tokens.as_deref(),
                },
            )?;
            ensure!(
                event.wrapper_binding_sha256 == expected_wrapper
                    && wrappers.insert(event.wrapper_binding_sha256.as_str()),
                "event wrapper is invalid or duplicated"
            );
        }
    }
    ensure!(expected == 7, "observed event sequence count mismatch");

    if associations_complete {
        ensure!(
            arms[0].summary.capture_content_sha256.is_none()
                && arms[0].capture_projection_sha256.is_none()
                && arms[3].summary.capture_content_sha256.is_none()
                && arms[3].capture_projection_sha256.is_none(),
            "off-arm capture association must be absent"
        );
        let capture = arms[1]
            .summary
            .capture_content_sha256
            .as_ref()
            .context("on-A capture association absent")?;
        ensure!(
            valid_sha(capture)
                && arms[1].capture_projection_sha256.as_ref() == Some(capture)
                && arms[2].summary.capture_content_sha256.as_ref() == Some(capture)
                && arms[2].capture_projection_sha256.as_ref() == Some(capture),
            "on-arm capture associations mismatch"
        );
    } else {
        ensure!(
            arms.iter()
                .all(|arm| arm.summary.capture_content_sha256.is_none()
                    && arm.capture_projection_sha256.is_none()),
            "capture association appeared before projection completion"
        );
    }
    Ok("on-A")
}

pub fn run_inventory(args: DflashK0sInventoryArgs, build_identity: Value) -> Result<()> {
    let observed_environment = observed_behavior_environment()
        .context("inventory environment rejected before opening authenticated inputs")?;
    ensure!(
        valid_sha(&args.inventory_spec_sha256),
        "inventory spec SHA-256 invalid"
    );
    ensure!(
        !args.prompt.is_empty() && args.prompt.len() <= 1 << 20,
        "prompt is invalid"
    );
    ensure!(
        (0..DFLASH_K0S_VOCAB as i32).contains(&args.carry_token),
        "carry token outside vocabulary"
    );

    let spec_input = OpenedInput::open(&args.inventory_spec, "inventory spec", Some(MAX_TRACE))?;
    ensure!(
        args.inventory_spec == spec_input.path && spec_input.sha256 == args.inventory_spec_sha256,
        "inventory spec path/hash mismatch"
    );
    let spec_bytes = opened_bytes(&spec_input)?;
    let spec_value: Value = serde_json::from_slice(&spec_bytes).context("parse inventory spec")?;
    validate_inventory_spec_order(&spec_value)?;
    let spec: InventorySpec =
        serde_json::from_slice(&spec_bytes).context("decode inventory spec")?;
    ensure!(
        spec.schema == INVENTORY_SPEC_SCHEMA
            && spec.schema_version == 1
            && !spec.run_id.is_empty()
            && spec.run_id.len() <= 128
            && spec.inventory_max_bytes > 0
            && spec.inventory_max_bytes <= MAX_TRACE,
        "inventory spec schema/caps invalid"
    );
    ensure!(
        spec.parser_caps
            == ParserCaps {
                header_bytes: MAX_GGUF_HEADER_BYTES,
                metadata: MAX_GGUF_METADATA,
                tensors: MAX_GGUF_TENSORS,
                strings_bytes: MAX_GGUF_STRINGS_BYTES,
                array_items: MAX_GGUF_ARRAY_ITEMS,
                objects: MAX_GGUF_OBJECTS,
            },
        "inventory parser caps differ from schema authority"
    );
    ensure!(
        spec.environment == observed_environment,
        "inventory environment allowlist mismatch"
    );
    let checkout = derive_checkout()?;
    ensure!(
        checkout == spec.checkout && !checkout.dirty,
        "checkout identity is not exact/clean"
    );
    verify_inventory_build(&spec.build, &build_identity)?;
    ensure!(
        spec.build.commit == checkout.commit,
        "build/checkout commit mismatch"
    );
    validate_source_roles(&spec.sources)?;
    ensure!(
        spec.assets
            .iter()
            .map(|claim| claim.role.as_str())
            .collect::<HashSet<_>>()
            == HashSet::from(["target", "drafter"]),
        "inventory asset roles invalid"
    );
    for (label, claim) in [
        ("executable", &spec.executable),
        ("reducer", &spec.reducer),
        ("scalar fixture", &spec.scalar_fixture),
        ("command template", &spec.command_template),
        ("embedded metallib", &spec.embedded_metallib),
    ] {
        claim_shape(claim, label)?;
    }
    for claim in spec.sources.iter().chain(&spec.assets) {
        claim_shape(&claim.file, &claim.role)?;
    }

    let output_path = canonical_output(&args.output, "inventory output")?;
    ensure!(
        args.output == output_path,
        "inventory output path must be canonical absolute"
    );
    let mut output = create_new_nofollow(&output_path, "inventory output")?;

    let executable_path = std::env::current_exe()?.canonicalize()?;
    let executable = open_and_verify_claim(&executable_path, &spec.executable, "executable")?;
    let reducer = open_and_verify_claim(Path::new(&spec.reducer.path), &spec.reducer, "reducer")?;
    let fixture = open_and_verify_claim(
        Path::new(&spec.scalar_fixture.path),
        &spec.scalar_fixture,
        "scalar fixture",
    )?;
    let command_template = open_and_verify_claim(
        Path::new(&spec.command_template.path),
        &spec.command_template,
        "command template",
    )?;
    let metallib = open_and_verify_claim(
        Path::new(&spec.embedded_metallib.path),
        &spec.embedded_metallib,
        "embedded metallib",
    )?;
    validate_external_metallib_claim(&spec.embedded_metallib)?;
    let mut source_inputs = Vec::new();
    for claim in &spec.sources {
        source_inputs.push(open_and_verify_claim(
            Path::new(&claim.file.path),
            &claim.file,
            &format!("source {}", claim.role),
        )?);
    }
    let mut paths = HashSet::from([spec_input.path.clone(), output_path.clone()]);
    let mut inodes = HashSet::from([(spec_input.device, spec_input.inode)]);
    for input in [
        &executable,
        &reducer,
        &fixture,
        &command_template,
        &metallib,
    ]
    .into_iter()
    .chain(source_inputs.iter())
    {
        ensure!(
            paths.insert(input.path.clone()),
            "inventory input path alias"
        );
        ensure!(
            inodes.insert((input.device, input.inode)),
            "inventory input inode alias"
        );
    }

    let target_claim = claim_by_role(&spec.assets, "target")?;
    let drafter_claim = claim_by_role(&spec.assets, "drafter")?;
    let target_input = open_and_verify_claim(&args.model, target_claim, "target asset")?;
    let drafter_input = open_and_verify_claim(&args.drafter, drafter_claim, "drafter asset")?;
    ensure!(
        args.model == target_input.path && args.drafter == drafter_input.path,
        "asset argv paths must be canonical absolute"
    );
    for input in [&target_input, &drafter_input] {
        ensure!(
            paths.insert(input.path.clone()),
            "inventory asset path alias"
        );
        ensure!(
            inodes.insert((input.device, input.inode)),
            "inventory asset inode alias"
        );
    }

    let target_scan = scan_inventory_gguf(&target_input)?;
    let drafter_scan = scan_inventory_gguf(&drafter_input)?;
    let target_g =
        GgufFile::from_opened_file(target_input.file.try_clone()?, target_input.path.clone())?;
    let drafter_g =
        GgufFile::from_opened_file(drafter_input.file.try_clone()?, drafter_input.path.clone())?;
    ensure!(
        target_g.shards.len() == 1 && drafter_g.shards.len() == 1,
        "inventory requires single-file GGUFs"
    );
    ensure!(
        target_g.tensors.len() as u64 == target_scan.tensor_count
            && drafter_g.tensors.len() as u64 == drafter_scan.tensor_count,
        "independent GGUF tensor counts disagree"
    );

    ensure!(
        spec.tensor_requirements.len() == 3,
        "three tensor requirements required"
    );
    let mut tensors = Vec::new();
    let mut tensor_roles = HashSet::new();
    for requirement in &spec.tensor_requirements {
        ensure!(
            tensor_roles.insert(requirement.role.as_str()),
            "duplicate tensor role"
        );
        ensure!(
            requirement.asset_role == "drafter",
            "K0-S tensor must bind drafter"
        );
        match requirement.role.as_str() {
            "selector_hidden" => ensure!(
                requirement.name == "selector_hidden.weight"
                    && requirement.dtype == K0S_SELECTOR_WEIGHT_DTYPE
                    && requirement.shape == [DFLASH_K0S_HIDDEN as u64, DFLASH_K0S_RANK as u64]
                    && requirement.orientation == "gguf_ne0_hidden_ne1_rank"
                    && requirement.row_domain.is_none(),
                "selector-hidden inventory requirement invalid"
            ),
            "predecessor" | "successor" => ensure!(
                requirement.name == format!("selector_{}.weight", requirement.role)
                    && requirement.dtype == "Q4_K"
                    && requirement.shape == [DFLASH_K0S_RANK as u64, DFLASH_K0S_VOCAB as u64]
                    && requirement.orientation == "gguf_ne0_rank_ne1_token"
                    && requirement.row_domain
                        == Some(RowDomain {
                            first: 0,
                            count: DFLASH_K0S_VOCAB as u64,
                        }),
                "codebook inventory requirement invalid"
            ),
            _ => return Err(anyhow!("unexpected inventory tensor role")),
        }
        let descriptor = drafter_g
            .find(&requirement.name)
            .with_context(|| format!("missing inventory tensor {}", requirement.name))?;
        if requirement.role == "selector_hidden" {
            require_q4_selector_name(&requirement.dtype)?;
            require_q4_selector_dtype(descriptor.dtype)?;
        }
        ensure!(
            descriptor.shape == requirement.shape
                && dtype_name(descriptor.dtype)? == requirement.dtype,
            "inventory tensor static descriptor mismatch"
        );
        let material = drafter_g.try_slice(descriptor)?;
        tensors.push(TensorClaim {
            role: requirement.role.clone(),
            asset_role: requirement.asset_role.clone(),
            name: requirement.name.clone(),
            dtype: requirement.dtype.clone(),
            shape: requirement.shape.clone(),
            offset: descriptor.data_offset,
            bytes: descriptor.n_bytes,
            sha256: hex(&Sha256::digest(material)),
            orientation: requirement.orientation.clone(),
            row_domain: requirement.row_domain.clone(),
        });
    }
    ensure!(
        tensor_roles == HashSet::from(["selector_hidden", "predecessor", "successor"]),
        "inventory tensor role set mismatch"
    );

    let embedding = target_g
        .find("token_embd.weight")
        .context("target token embedding absent")?;
    ensure!(
        embedding.shape.len() == 2 && embedding.shape[1] == DFLASH_K0S_VOCAB as u64,
        "target vocabulary geometry mismatch"
    );
    let token_count = target_scan
        .tokenizer_count
        .context("target tokenizer count absent")?;
    let token_digest = target_scan.tokenizer_sha256.clone().unwrap_or_else(|| {
        canonical_json_sha(&serde_json::json!({
            "token_count": token_count,
            "token_embd": embedding.shape,
        }))
        .expect("canonical tokenizer fallback")
    });
    let tokenizer_facts = InventoryTokenizer {
        vocab_size: DFLASH_K0S_VOCAB as u64,
        token_embd_name: "token_embd.weight".into(),
        token_embd_shape: embedding.shape.clone(),
        token_embd_dtype: dtype_name(embedding.dtype)?.to_owned(),
        token_count,
        tokenizer_tokens_sha256: token_digest.clone(),
    };
    ensure!(
        tokenizer_facts == spec.tokenizer,
        "tokenizer identity differs from spec"
    );
    let tokenizer =
        NativeTokenizer::from_gguf(&target_g).context("construct inventory tokenizer")?;
    let prompt_ids = tokenizer
        .encode(&args.prompt, false)
        .context("tokenize inventory prompt")?;
    let prompt_facts = InventoryPrompt {
        utf8_hex: hex(args.prompt.as_bytes()),
        token_ids: prompt_ids.clone(),
        token_ids_sha256_i32le: token_ids_sha256_i32le(&prompt_ids),
        tokenizer_identity_sha256: token_digest,
    };
    ensure!(
        prompt_facts == spec.prompt,
        "prompt/tokenization differs from spec"
    );
    ensure!(
        args.prompt.as_bytes() == hex_decode(&spec.prompt.utf8_hex)?.as_slice(),
        "prompt bytes differ from spec"
    );

    ensure!(
        drafter_scan.masks.len() == 1,
        "drafter requires exactly one supported mask key"
    );
    let (mask_key, mask_raw) = &drafter_scan.masks[0];
    let mask_token = i32::try_from(*mask_raw).context("mask token exceeds i32")?;
    ensure!(
        mask_token == spec.expected_mask_token && args.carry_token == spec.carry_token,
        "mask/carry differs from inventory spec"
    );
    ensure!(
        drafter_g
            .get_u64("dflash.block_size")
            .or_else(|| drafter_g.get_u64("dflash-draft.dflash.block_size"))
            == Some(DFLASH_K0S_BLOCK_SIZE as u64)
            && drafter_g
                .get_u64("dflash.embedding_length")
                .or_else(|| drafter_g.get_u64("dflash-draft.embedding_length"))
                == Some(DFLASH_K0S_HIDDEN as u64)
            && drafter_g.get_u64("dflash.selector_rank") == Some(DFLASH_K0S_RANK as u64)
            && drafter_g.get_u64("dflash.selector_top_k") == Some(DFLASH_K0S_TOP_K as u64),
        "drafter K0-S geometry mismatch"
    );
    let noise_tokens = std::iter::once(args.carry_token)
        .chain(std::iter::repeat_n(mask_token, DFLASH_K0S_BLOCK_SIZE - 1))
        .collect::<Vec<_>>();
    let noise_sha = token_ids_sha256_i32le(&noise_tokens);

    let device = MTLCreateSystemDefaultDevice().context("no default Metal device")?;
    let device_claim = HostClaim {
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        device_name: device.name().to_string(),
        device_registry_id: device.registryID(),
        device_family: metal_family_capabilities(&device),
    };
    ensure!(
        device_claim.os == spec.host_predicate.os
            && device_claim.arch == spec.host_predicate.arch
            && device_claim.device_name == spec.host_predicate.device_name,
        "Metal device predicate mismatch"
    );

    let actual_command = std::env::args_os()
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow!("non-UTF8 argv is forbidden"))
        })
        .collect::<Result<Vec<_>>>()?;
    let expected_command = vec![
        executable.path.to_string_lossy().into_owned(),
        "dflash-k0s-inventory".into(),
        "--model".into(),
        target_input.path.to_string_lossy().into_owned(),
        "--drafter".into(),
        drafter_input.path.to_string_lossy().into_owned(),
        "--prompt".into(),
        args.prompt.clone(),
        "--carry-token".into(),
        args.carry_token.to_string(),
        "--inventory-spec".into(),
        spec_input.path.to_string_lossy().into_owned(),
        "--inventory-spec-sha256".into(),
        args.inventory_spec_sha256.clone(),
        "--output".into(),
        output_path.to_string_lossy().into_owned(),
    ];
    ensure!(
        actual_command == expected_command,
        "inventory argv is not canonical"
    );
    let mut expected_template = expected_command.clone();
    expected_template[13] = "${INVENTORY_SPEC_SHA256}".into();
    expected_template[15] = "${INVENTORY_OUTPUT}".into();
    ensure!(
        expected_template == spec.command,
        "inventory command template mismatch"
    );

    let artifact = InventoryArtifact {
        schema: INVENTORY_SCHEMA,
        schema_version: 1,
        authority: INVENTORY_AUTHORITY,
        inventory_spec_sha256: args.inventory_spec_sha256,
        run_id: spec.run_id,
        checkout,
        build: spec.build,
        sources: spec.sources,
        executable: spec.executable,
        reducer: spec.reducer,
        scalar_fixture: spec.scalar_fixture,
        command_template: spec.command_template,
        embedded_metallib: spec.embedded_metallib,
        device: device_claim,
        assets: spec.assets,
        gguf: vec![
            InventoryGguf {
                role: "target".into(),
                version: target_scan.version,
                tensor_count: target_scan.tensor_count,
                metadata_count: target_scan.metadata_count,
            },
            InventoryGguf {
                role: "drafter".into(),
                version: drafter_scan.version,
                tensor_count: drafter_scan.tensor_count,
                metadata_count: drafter_scan.metadata_count,
            },
        ],
        tensors,
        tokenizer: tokenizer_facts,
        prompt: prompt_facts,
        mask_noise: InventoryMaskNoise {
            metadata_key: mask_key.clone(),
            mask_token,
            noise_tokens,
            noise_sha256_i32le: noise_sha,
        },
        parser_caps: spec.parser_caps,
        command: expected_command,
        environment: observed_environment,
    };
    let bytes = serde_json::to_vec(&artifact)?;
    ensure!(
        bytes.len() as u64 <= spec.inventory_max_bytes,
        "inventory output exceeds spec cap"
    );
    for input in [
        &spec_input,
        &executable,
        &reducer,
        &fixture,
        &command_template,
        &metallib,
    ]
    .into_iter()
    .chain(source_inputs.iter())
    .chain([&target_input, &drafter_input])
    {
        input.final_custody_check("inventory input")?;
    }
    ensure!(
        target_g.revalidate_retained_shard_stamps()?.len() == 1
            && drafter_g.revalidate_retained_shard_stamps()?.len() == 1,
        "inventory GGUF custody changed"
    );
    output.write_all(&bytes)?;
    output.flush()?;
    output.sync_all()?;
    ensure!(
        output.metadata()?.len() == bytes.len() as u64,
        "inventory output size mismatch"
    );
    Ok(())
}

fn hex_decode(value: &str) -> Result<Vec<u8>> {
    ensure!(value.len().is_multiple_of(2), "hex length is odd");
    (0..value.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&value[index..index + 2], 16).map_err(Into::into))
        .collect()
}

fn hash_i32(values: &[i32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    hex(&hash.finalize())
}

fn library_vector_hash(
    domain: &[u8],
    count: usize,
    values: impl IntoIterator<Item = [u8; 4]>,
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update((count as u64).to_le_bytes());
    for value in values {
        hash.update(value);
    }
    hash.finalize().into()
}

fn capture_matches_summary(capture: &DFlashK0sCapture, summary: &DFlashK0sParitySummary) -> bool {
    let logits_count = capture
        .depths
        .iter()
        .map(|depth| depth.full_logits_bits.len())
        .sum();
    let ids_count = capture
        .depths
        .iter()
        .map(|depth| depth.top_k_ids.len())
        .sum();
    let unary_count = capture
        .depths
        .iter()
        .map(|depth| depth.unary_bits.len())
        .sum();
    let hidden_count = capture
        .depths
        .iter()
        .map(|depth| depth.selector_hidden_bits.len())
        .sum();
    capture.draft_tokens == summary.draft_tokens
        && capture.dispatch_census == summary.dispatch_census
        && [
            capture.kernel_trace.encoders,
            capture.kernel_trace.concurrent_encoders,
            capture.kernel_trace.dispatches,
        ] == summary.kernel_trace
        && capture.state_identity.diagnostic_state_sha256 == summary.diagnostic_state_sha256
        && capture.state_identity.carry_token == summary.carry_token
        && capture.state_identity.noise_start_position == summary.noise_start_position
        && summary.selector_inputs.full_logits_count == logits_count
        && summary.selector_inputs.full_logits_sha256_f32le
            == library_vector_hash(
                b"qwen.dflash_k0s.full_logits.f32le.v1",
                logits_count,
                capture.depths.iter().flat_map(|depth| {
                    depth
                        .full_logits_bits
                        .iter()
                        .map(|value| value.to_le_bytes())
                }),
            )
        && summary.selector_inputs.top_k_ids_count == ids_count
        && summary.selector_inputs.top_k_ids_sha256_i32le
            == library_vector_hash(
                b"qwen.dflash_k0s.top_k_ids.i32le.v1",
                ids_count,
                capture
                    .depths
                    .iter()
                    .flat_map(|depth| depth.top_k_ids.iter().map(|value| value.to_le_bytes())),
            )
        && summary.selector_inputs.unary_count == unary_count
        && summary.selector_inputs.unary_sha256_f32le
            == library_vector_hash(
                b"qwen.dflash_k0s.unary.f32le.v1",
                unary_count,
                capture
                    .depths
                    .iter()
                    .flat_map(|depth| depth.unary_bits.iter().map(|value| value.to_le_bytes())),
            )
        && summary.selector_inputs.selector_hidden_count == hidden_count
        && summary.selector_inputs.selector_hidden_sha256_f32le
            == library_vector_hash(
                b"qwen.dflash_k0s.selector_hidden.f32le.v1",
                hidden_count,
                capture.depths.iter().flat_map(|depth| {
                    depth
                        .selector_hidden_bits
                        .iter()
                        .map(|value| value.to_le_bytes())
                }),
            )
}

fn tensor_f32_sha(tensor: &MetalTensor) -> Result<String> {
    ensure!(
        tensor.dtype == qwen_llm::tensor::GgmlType::F32,
        "expected F32 tensor"
    );
    let bytes = usize::try_from(tensor.n_elements())?
        .checked_mul(4)
        .context("tensor byte count overflow")?;
    let offset = usize::try_from(tensor.offset)?;
    ensure!(
        offset
            .checked_add(bytes)
            .is_some_and(|end| end <= tensor.buffer.length()),
        "tensor read exceeds buffer"
    );
    let slice = unsafe {
        std::slice::from_raw_parts(
            (tensor.buffer.contents().as_ptr() as *const u8).add(offset),
            bytes,
        )
    };
    Ok(hex(&Sha256::digest(slice)))
}

fn snapshot_sha256(snapshot: &SessionSnapshot) -> String {
    fn bytes(hash: &mut Sha256, value: &[u8]) {
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value);
    }
    let mut hash = Sha256::new();
    hash.update(b"qwen.dflash_k0s.target_session_snapshot.v1");
    let identity = &snapshot.identity;
    hash.update(identity.model_id.to_le_bytes());
    hash.update(identity.tokenizer_id.to_le_bytes());
    hash.update(identity.layout_version.to_le_bytes());
    hash.update(identity.n_attn_layers.to_le_bytes());
    hash.update(identity.n_gdn_layers.to_le_bytes());
    hash.update(identity.kv_dim_elements.to_le_bytes());
    hash.update(identity.kv_bytes_per_token.to_le_bytes());
    hash.update((identity.kv_storage_kind as u32).to_le_bytes());
    hash.update(identity.gdn_state_elements_per_layer.to_le_bytes());
    hash.update(identity.gdn_conv_elements_per_layer.to_le_bytes());
    hash.update((snapshot.prefix_tokens.len() as u64).to_le_bytes());
    for token in &snapshot.prefix_tokens {
        hash.update(token.to_le_bytes());
    }
    match snapshot.pending_token {
        Some(token) => {
            hash.update([1]);
            hash.update(token.to_le_bytes());
        }
        None => hash.update([0]),
    }
    hash.update((snapshot.kv_n_pos.len() as u64).to_le_bytes());
    for value in &snapshot.kv_n_pos {
        hash.update((*value as u64).to_le_bytes());
    }
    for arena in [
        &snapshot.kv_k_arena,
        &snapshot.kv_v_arena,
        &snapshot.gdn_conv_arena,
        &snapshot.gdn_state_arena,
    ] {
        bytes(&mut hash, arena);
    }
    for values in [&snapshot.final_logits, &snapshot.capture_tail] {
        match values {
            Some(values) => {
                hash.update([1]);
                hash.update((values.len() as u64).to_le_bytes());
                for value in values {
                    hash.update(value.to_bits().to_le_bytes());
                }
            }
            None => hash.update([0]),
        }
    }
    hex(&hash.finalize())
}

fn observer_sha(counts: [usize; 3]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"qwen.dflash_k0s.observer_baseline.v1");
    for count in counts {
        hash.update((count as u64).to_le_bytes());
    }
    hex(&hash.finalize())
}

struct PlainObserverGuard {
    trace: Option<KernelTraceGuard>,
    cleaned: bool,
}

impl PlainObserverGuard {
    fn begin() -> Result<Self> {
        ensure!(
            diagnostics_observer_active_counts() == [0, 0, 0],
            "observer baseline is not zero"
        );
        dispatch_census_begin();
        Ok(Self {
            trace: Some(kernel_trace_begin()),
            cleaned: false,
        })
    }

    fn finish(mut self) -> Result<(Vec<DispatchCensusRow>, KernelTraceCounters)> {
        let counters = kernel_trace_snapshot();
        drop(self.trace.take());
        let rows = dispatch_census_take();
        self.cleaned = true;
        ensure!(
            diagnostics_observer_active_counts() == [0, 0, 0],
            "observer cleanup failed"
        );
        Ok((rows, counters))
    }
}

impl Drop for PlainObserverGuard {
    fn drop(&mut self) {
        if !self.cleaned {
            drop(self.trace.take());
            let _ = dispatch_census_take();
        }
    }
}

fn external_plain_draft(
    decoder: &mut DFlashDecoder<'_>,
    carry: i32,
    position: u32,
) -> Result<(
    Vec<i32>,
    Vec<DFlashK0sDispatchCensusRow>,
    KernelTraceCounters,
)> {
    let observer = PlainObserverGuard::begin()?;
    let result = decoder.draft_block(carry, position);
    let (rows, counters) = observer.finish()?;
    let tokens = result?;
    ensure!(
        rows.len() <= MAX_DISPATCH_ROWS,
        "dispatch census exceeds cap"
    );
    Ok((
        tokens,
        rows.into_iter()
            .map(|row| DFlashK0sDispatchCensusRow {
                family: row.family.into(),
                tag: row.tag,
                encoder_ordinal: row.encoder_ordinal,
                encoder_concurrent: row.encoder_concurrent,
                kernel: row.kernel,
                grid: [row.grid_width, row.grid_height, row.grid_depth],
                threads: [row.threads_width, row.threads_height, row.threads_depth],
                grid_threadgroups: row.grid_tgs,
                threadgroup_threads: row.tg_threads,
            })
            .collect(),
        counters,
    ))
}

fn predicate_matches(
    row: &DFlashK0sDispatchCensusRow,
    predicate: &SelectorDispatchPredicate,
) -> bool {
    row.tag.as_deref() == Some(predicate.tag.as_str())
        && row.kernel == predicate.kernel
        && row.grid == predicate.grid
        && row.threads == predicate.threads
}

struct VerifiedRuntimeBinding {
    metal_source_sha256: String,
    metallib_sha256: String,
    build_source_sha256: String,
    environment: BTreeMap<String, String>,
}

fn validate_phase_census(
    summary: &DFlashK0sParitySummary,
    predicate: &SelectorDispatchPredicate,
    binding: &VerifiedRuntimeBinding,
) -> Result<SelectorDispatchPredicate> {
    let tagged = summary
        .dispatch_census
        .iter()
        .filter(|row| row.tag.as_deref() == Some(predicate.tag.as_str()))
        .collect::<Vec<_>>();
    ensure!(
        tagged.len() == 1 && predicate_matches(tagged[0], predicate),
        "selector dispatch predicate mismatch"
    );
    for row in &summary.dispatch_census {
        let grid = row.grid.into_iter().try_fold(1u64, |product, value| {
            product
                .checked_mul(value)
                .context("dispatch grid product overflow")
        })?;
        let threads = row.threads.into_iter().try_fold(1u64, |product, value| {
            product
                .checked_mul(value)
                .context("dispatch thread product overflow")
        })?;
        ensure!(
            row.grid_threadgroups == grid && row.threadgroup_threads == threads,
            "dispatch flattened geometry mismatch"
        );
    }
    let ordinals = summary
        .dispatch_census
        .iter()
        .map(|row| row.encoder_ordinal)
        .collect::<HashSet<_>>();
    let concurrent = summary
        .dispatch_census
        .iter()
        .filter(|row| row.encoder_concurrent)
        .map(|row| row.encoder_ordinal)
        .collect::<HashSet<_>>();
    ensure!(
        summary.kernel_trace
            == [
                ordinals.len() as u64,
                concurrent.len() as u64,
                summary.dispatch_census.len() as u64
            ],
        "dispatch census/kernel counters mismatch"
    );
    let selector = summary.selector_dispatch;
    require_q4_selector_dtype(selector.weight_dtype)?;
    ensure!(
        selector.input_dtype == qwen_llm::tensor::GgmlType::F32
            && selector.output_dtype == qwen_llm::tensor::GgmlType::F32
            && selector.input_dtype as i32 == 0
            && selector.output_dtype as i32 == 0
            && tagged[0].kernel == K0S_SELECTOR_KERNEL,
        "runtime selector dispatch is not the authorized Q4_K/F32 kernel contract"
    );
    let runtime = SelectorDispatchPredicate {
        tag: predicate.tag.clone(),
        kernel: tagged[0].kernel.clone(),
        weight_dtype: dtype_name(selector.weight_dtype)?.to_owned(),
        input_dtype: dtype_name(selector.input_dtype)?.to_owned(),
        output_dtype: dtype_name(selector.output_dtype)?.to_owned(),
        weight_dtype_id: selector.weight_dtype as i32,
        input_dtype_id: selector.input_dtype as i32,
        output_dtype_id: selector.output_dtype as i32,
        n: selector.block_size as u64,
        h: selector.hidden_size as u64,
        r: selector.selector_rank as u64,
        grid: tagged[0].grid,
        threads: tagged[0].threads,
        metal_source_sha256: binding.metal_source_sha256.clone(),
        metallib_sha256: binding.metallib_sha256.clone(),
        build_source_sha256: binding.build_source_sha256.clone(),
        allowed_environment: binding.environment.clone(),
    };
    ensure!(
        runtime == *predicate,
        "runtime selector contract differs from manifest predicate"
    );
    Ok(runtime)
}

fn arm_phase(
    target_sha256: String,
    target_state_sha256: String,
    summary: &DFlashK0sParitySummary,
    runtime_selector_contract: SelectorDispatchPredicate,
) -> ArmPhaseJson {
    ArmPhaseJson {
        carry_token: summary.carry_token,
        noise_start_position: summary.noise_start_position,
        target_sha256,
        dflash_sha256: hex(&summary.diagnostic_state_sha256),
        state_sha256: target_state_sha256,
        draft_tokens_count: summary.draft_tokens.len(),
        draft_tokens_sha256_i32le: hash_i32(&summary.draft_tokens),
        full_logits_count: summary.selector_inputs.full_logits_count,
        full_logits_sha256_f32le: hex(&summary.selector_inputs.full_logits_sha256_f32le),
        topk_count: summary.selector_inputs.top_k_ids_count,
        topk_sha256_i32le: hex(&summary.selector_inputs.top_k_ids_sha256_i32le),
        unary_count: summary.selector_inputs.unary_count,
        unary_sha256_f32le: hex(&summary.selector_inputs.unary_sha256_f32le),
        z_count: summary.selector_inputs.selector_hidden_count,
        z_sha256_f32le: hex(&summary.selector_inputs.selector_hidden_sha256_f32le),
        dispatch_census: summary.dispatch_census.iter().map(dispatch).collect(),
        kernel_trace: KernelTrace {
            encoders: summary.kernel_trace[0],
            concurrent_encoders: summary.kernel_trace[1],
            dispatches: summary.kernel_trace[2],
        },
        runtime_selector_contract,
    }
}

fn target_output_sha(logits: &[f32], hidden: &MetalTensor) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(b"qwen.dflash_k0s.target_output.v1");
    hash.update((logits.len() as u64).to_le_bytes());
    for value in logits {
        hash.update(value.to_bits().to_le_bytes());
    }
    hash.update(hex_decode(&tensor_f32_sha(hidden)?)?);
    Ok(hex(&hash.finalize()))
}

fn projection_normalized(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, child)| {
                    (
                        key.clone(),
                        if key.ends_with("_range_id") && !child.is_null() {
                            Value::String("<range>".into())
                        } else {
                            projection_normalized(child)
                        },
                    )
                })
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(projection_normalized).collect()),
        other => other.clone(),
    }
}

fn capture_json(capture: &DFlashK0sCapture) -> CaptureJson {
    CaptureJson {
        definition_version: CAPTURE_DEFINITION,
        noise_start_position: capture.state_identity.noise_start_position,
        carry_token: capture.state_identity.carry_token,
        synchronized_capture_sha256: hex(&capture.capture_sha256),
        draft_tokens: capture.draft_tokens.clone(),
        draft_token_bits: capture.draft_token_bits.iter().copied().map(bits).collect(),
        draft_tokens_sha256_i32le: hex(&capture.state_identity.draft_tokens_sha256),
        state: CaptureState {
            target_context_len: capture.state_identity.target_context_len,
            context_hidden_watermark: capture.state_identity.context_hidden_watermark,
            kv_context_watermark: capture.state_identity.kv_context_watermark,
            noise_input_sha256: hex(&capture.state_identity.noise_input_sha256),
            synchronized_event_sha256: hex(&capture.state_identity.synchronized_event_sha256),
            diagnostic_state_sha256: hex(&capture.state_identity.diagnostic_state_sha256),
        },
    }
}

fn capture_provenance(
    capture: &DFlashK0sCapture,
    manifest: &Manifest,
    environment: &BTreeMap<String, String>,
) -> Result<Provenance> {
    let provenance = Provenance {
        dispatch_census: capture.dispatch_census.iter().map(dispatch).collect(),
        selector_hidden_dispatch: dispatch(&capture.selector_hidden_dispatch),
        kernel_trace: KernelTrace {
            encoders: capture.kernel_trace.encoders,
            concurrent_encoders: capture.kernel_trace.concurrent_encoders,
            dispatches: capture.kernel_trace.dispatches,
        },
        embedded_metallib_sha256: hex(&capture.provenance.embedded_metallib_sha256),
        build: manifest.expected_build.clone(),
        host: manifest.expected_host.clone(),
        environment: environment.clone(),
    };
    let tagged = capture
        .dispatch_census
        .iter()
        .filter(|row| row.tag.as_deref() == Some(manifest.selector_dispatch_predicate.tag.as_str()))
        .collect::<Vec<_>>();
    ensure!(
        tagged.len() == 1
            && tagged[0] == &capture.selector_hidden_dispatch
            && predicate_matches(tagged[0], &manifest.selector_dispatch_predicate),
        "capture selector dispatch differs from runtime predicate"
    );
    let ordinals = capture
        .dispatch_census
        .iter()
        .map(|row| row.encoder_ordinal)
        .collect::<HashSet<_>>();
    let concurrent = capture
        .dispatch_census
        .iter()
        .filter(|row| row.encoder_concurrent)
        .map(|row| row.encoder_ordinal)
        .collect::<HashSet<_>>();
    ensure!(
        capture.kernel_trace.encoders == ordinals.len() as u64
            && capture.kernel_trace.concurrent_encoders == concurrent.len() as u64
            && capture.kernel_trace.dispatches == capture.dispatch_census.len() as u64,
        "capture census/counter mismatch"
    );
    Ok(provenance)
}

struct ProjectionParts {
    depths: Vec<Value>,
    lattices: Vec<Value>,
    capture: Value,
    production_chain: Value,
    fixed_chains: Value,
    provenance: Value,
    digest: String,
}

struct ArmExecution {
    arm: ArmJson,
    capture: Option<DFlashK0sCapture>,
}

fn run_arm(
    name: &'static str,
    diagnostic: bool,
    attempt_id: &str,
    rng_domain_names: &[String],
    ctx: &MetalContext,
    mm: &MetalModel,
    mf: &MetalForward<'_>,
    mhead: &MetalDFlashHead,
    target_layer_ids: &[u32],
    prompt_ids: &[i32],
    carry_token: i32,
    continuation_carry_token: i32,
    predicate: &SelectorDispatchPredicate,
    runtime_binding: &VerifiedRuntimeBinding,
) -> Result<ArmExecution> {
    let baseline_before = diagnostics_observer_active_counts();
    ensure!(
        baseline_before == [0, 0, 0],
        "arm starts with active observer"
    );
    let capacity = prompt_ids
        .len()
        .checked_add(DFLASH_K0S_BLOCK_SIZE + 1)
        .context("arm session capacity overflow")?;
    let mut target_session = MetalSession::fresh(ctx, mm, capacity).context("[session_setup]")?;
    let dflash_session = MetalDFlashSession::fresh(
        ctx,
        mhead,
        DFLASH_K0S_HIDDEN as u64,
        DFLASH_K0S_VOCAB as u64,
        capacity,
    )
    .context("[session_setup]")?;
    let features = target_layer_ids
        .len()
        .checked_mul(DFLASH_K0S_HIDDEN)
        .context("arm feature count overflow")?;
    let prompt_hidden = MetalTensor::zeros_f32(
        ctx,
        vec![u64::try_from(
            prompt_ids
                .len()
                .checked_mul(features)
                .context("arm hidden allocation overflow")?,
        )?],
    )
    .context("[session_setup]")?;
    let mut scratch =
        MetalDFlashLayerMajorScratch::fresh_prefill(ctx, mm, DFLASH_K0S_BLOCK_SIZE as u32)
            .context("[session_setup]")?;
    let prompt_logits = prefill_tokens_with_multi_hidden(
        mf,
        prompt_ids,
        0,
        &mut target_session,
        &mut scratch,
        target_layer_ids,
        Some(&prompt_hidden),
    )
    .context("[prompt_prefill]")?;
    let first_target_sha = target_output_sha(&prompt_logits, &prompt_hidden)?;
    let first_identity = target_session.snapshot_identity(0, 0);
    let first_snapshot = target_session
        .snapshot(
            first_identity,
            prompt_ids.to_vec(),
            Some(prompt_logits.clone()),
        )
        .context("[prompt_prefill]")?;
    let first_state_sha = snapshot_sha256(&first_snapshot);

    let mut dflash_session = dflash_session;
    dflash_session
        .append_target_ctx_columns_contiguous_now(
            ctx,
            &prompt_hidden,
            0,
            prompt_ids.len(),
            features,
        )
        .context("[prompt_prefill]")?;
    let mut decoder = DFlashDecoder::new(mf, mhead, dflash_session);
    let noise_start = u32::try_from(prompt_ids.len())?;
    let (
        first_summary,
        first_sequence,
        first_library_envelope,
        first_event_session,
        first_event_tokens,
        capture,
    ) = if diagnostic {
        let observation = decoder
            .draft_block_with_k0s_observation(carry_token, noise_start)
            .context("[first_block]")?;
        let summary = observation.summary().clone();
        ensure!(
            dflash_k0s_event_envelope_sha256(&summary.parity, summary.event_sequence)
                == summary.event_envelope_sha256,
            "first library event envelope failed local verification"
        );
        let event_session = hex(&summary.parity.session_binding_sha256);
        let event_tokens = summary.parity.draft_tokens.clone();
        let capture = decoder
            .extract_k0s_observation(observation)
            .context("[extraction]")?;
        (
            summary.parity,
            Some(summary.event_sequence),
            Some(hex(&summary.event_envelope_sha256)),
            Some(event_session),
            Some(event_tokens),
            Some(capture),
        )
    } else {
        let (tokens, census, counters) =
            external_plain_draft(&mut decoder, carry_token, noise_start)
                .context("[first_block]")?;
        let summary = decoder
            .dflash_k0s_parity_summary(carry_token, noise_start, &tokens, &census, counters)
            .context("[first_block]")?;
        (summary, None, None, None, None, None)
    };
    if let Some(capture) = &capture {
        ensure!(
            capture_matches_summary(capture, &first_summary),
            "retained capture differs from first observed summary"
        );
    }
    let first_runtime = validate_phase_census(&first_summary, predicate, runtime_binding)?;
    let session_id = hex(&first_summary.session_binding_sha256);
    let first_event = bind_event(
        attempt_id,
        name,
        &session_id,
        "first",
        if diagnostic { "observed" } else { "plain" },
        first_sequence,
        first_library_envelope,
        first_event_session,
        first_event_tokens,
    )?;
    let first_phase = arm_phase(
        first_target_sha,
        first_state_sha,
        &first_summary,
        first_runtime,
    );

    let continuation_hidden =
        MetalTensor::zeros_f32(ctx, vec![features as u64]).context("[continuation]")?;
    let continuation_logits = mf
        .single_token_with_multi_hidden(
            carry_token,
            noise_start,
            &mut target_session,
            target_layer_ids,
            &continuation_hidden,
        )
        .context("[continuation]")?;
    decoder
        .session
        .append_target_ctx_column_now(
            ctx,
            &continuation_hidden,
            u32::try_from(prompt_ids.len())?,
            features,
        )
        .context("[continuation]")?;
    let continuation_target_sha = target_output_sha(&continuation_logits, &continuation_hidden)?;
    let mut continuation_prefix = prompt_ids.to_vec();
    continuation_prefix.push(carry_token);
    let continuation_identity = target_session.snapshot_identity(0, 0);
    let continuation_snapshot = target_session
        .snapshot(
            continuation_identity,
            continuation_prefix,
            Some(continuation_logits),
        )
        .context("[continuation]")?;
    let continuation_state_sha = snapshot_sha256(&continuation_snapshot);
    let continuation_position = noise_start
        .checked_add(1)
        .context("continuation position overflow")?;
    let continuation_observation = decoder
        .draft_block_with_k0s_observation(continuation_carry_token, continuation_position)
        .context("[continuation]")?;
    let cloned_continuation = continuation_observation.summary().clone();
    let consumed_continuation = decoder
        .finish_k0s_observation_without_extraction(continuation_observation)
        .context("[continuation]")?;
    ensure!(
        consumed_continuation == cloned_continuation,
        "continuation consumed summary differs from cloned summary"
    );
    ensure!(
        dflash_k0s_event_envelope_sha256(
            &consumed_continuation.parity,
            consumed_continuation.event_sequence,
        ) == consumed_continuation.event_envelope_sha256,
        "continuation library event envelope failed local verification"
    );
    let continuation_sequence = consumed_continuation.event_sequence;
    let continuation_library_envelope = hex(&consumed_continuation.event_envelope_sha256);
    let continuation_event_session = hex(&consumed_continuation.parity.session_binding_sha256);
    let continuation_event_tokens = consumed_continuation.parity.draft_tokens.clone();
    let continuation_summary = consumed_continuation.parity;
    ensure!(
        continuation_summary.session_binding_sha256 == first_summary.session_binding_sha256,
        "continuation session binding changed"
    );
    let continuation_runtime =
        validate_phase_census(&continuation_summary, predicate, runtime_binding)?;
    let continuation_event = bind_event(
        attempt_id,
        name,
        &session_id,
        "continuation",
        "observed",
        Some(continuation_sequence),
        Some(continuation_library_envelope),
        Some(continuation_event_session),
        Some(continuation_event_tokens),
    )?;
    let continuation_phase = arm_phase(
        continuation_target_sha,
        continuation_state_sha,
        &continuation_summary,
        continuation_runtime,
    );
    let baseline_after = diagnostics_observer_active_counts();
    let observer_baseline = ObserverBaselineJson {
        before_sha256: observer_sha(baseline_before),
        after_sha256: observer_sha(baseline_after),
        restored: baseline_before == baseline_after,
    };
    ensure!(
        observer_baseline.restored,
        "arm observer baseline was not restored"
    );
    let common_production_content_sha256 = canonical_domain_digest(
        "qwen.dflash_k0s.common_production.v1",
        &CommonProductionDigest {
            first: &first_phase,
            continuation: &continuation_phase,
        },
    )?;
    let rng_domains = arm_rng_domains(attempt_id, name, rng_domain_names)?;
    let mut arm = ArmJson {
        name,
        diagnostic,
        session_id,
        first_event,
        continuation_event,
        summary: ArmSummaryJson {
            domain: "qwen.dflash_k0s.parity_arm_summary.v1",
            first: first_phase,
            continuation: continuation_phase,
            observer_baseline,
            common_production_content_sha256,
            capture_content_sha256: None,
        },
        rng_domains,
        capture_projection_sha256: None,
        arm_envelope_sha256: String::new(),
    };
    refresh_arm_envelope(&mut arm, attempt_id)?;
    Ok(ArmExecution { arm, capture })
}

fn projection_parts(
    capture: &DFlashK0sCapture,
    layout: &SidecarLayout,
    manifest: &Manifest,
    chains: &[StaticChain],
    environment: &BTreeMap<String, String>,
) -> Result<ProjectionParts> {
    let capture_digest = hex(&capture.capture_sha256);
    let draft_digest = hex(&capture.state_identity.draft_tokens_sha256);
    let depths = (1..=7)
        .map(|depth| {
            let value = &capture.depths[depth - 1];
            serde_json::to_value(DepthPayload {
                depth,
                position: value.position,
                production_call_id: manifest.expected_binding.production_call_id.clone(),
                drafter_checkpoint_sha256: manifest
                    .expected_binding
                    .drafter_checkpoint_sha256
                    .clone(),
                proposal_construction_id: manifest
                    .expected_binding
                    .proposal_construction_id
                    .clone(),
                noise_input_sha256: manifest.expected_binding.noise_input_sha256.clone(),
                synchronized_capture_sha256: capture_digest.clone(),
                draft_tokens_sha256_i32le: draft_digest.clone(),
                diagnostic_state_sha256: hex(&capture.state_identity.diagnostic_state_sha256),
                z_f32_bits: value
                    .selector_hidden_bits
                    .iter()
                    .copied()
                    .map(bits)
                    .collect(),
                full_logits_range_id: layout.full_logits[depth - 1].clone(),
                top16_ids: value.top_k_ids.clone(),
                unary_f32_bits: value.unary_bits.iter().copied().map(bits).collect(),
                topk_issues: topk_issues(&value.top_k_issues),
            })
            .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    let lattices = (1..=7)
        .map(|depth| {
            serde_json::to_value(lattice_json(capture, layout, depth)?).map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    let production_chain = serde_json::to_value(chain_json(
        "production-greedy".into(),
        capture.state_identity.carry_token,
        &capture.production_chain,
    ))?;
    let fixed_chains = serde_json::to_value(
        chains
            .iter()
            .map(|item| {
                chain_json(
                    item.name.clone(),
                    item.initial_carry,
                    &dflash_k0s_traverse_slots(&capture.lattice, item.initial_carry, &item.slots),
                )
            })
            .collect::<Vec<_>>(),
    )?;
    let capture_value = serde_json::to_value(capture_json(capture))?;
    let provenance = serde_json::to_value(capture_provenance(capture, manifest, environment)?)?;
    let mut content = serde_json::Map::new();
    content.insert("depths".into(), Value::Array(depths.clone()));
    content.insert("lattices".into(), Value::Array(lattices.clone()));
    content.insert("capture".into(), capture_value.clone());
    content.insert("production_chain".into(), production_chain.clone());
    content.insert("fixed_chains".into(), fixed_chains.clone());
    content.insert("provenance".into(), provenance.clone());
    let digest = canonical_json_sha(&projection_normalized(&Value::Object(content)))?;
    Ok(ProjectionParts {
        depths,
        lattices,
        capture: capture_value,
        production_chain,
        fixed_chains,
        provenance,
        digest,
    })
}

pub fn run(args: DflashK0sArgs, build_identity: Value) -> Result<()> {
    let observed_environment = observed_behavior_environment()
        .context("acquisition environment rejected before opening authenticated inputs")?;
    ensure!(!args.prompt.is_empty(), "prompt must be nonempty");
    ensure!(
        !args.attempt_id.is_empty() && args.attempt_id.len() <= 128 && args.attempt_id.is_ascii(),
        "attempt_id must be bounded nonempty ASCII"
    );
    ensure!(
        (0..DFLASH_K0S_VOCAB as i32).contains(&args.carry_token),
        "carry token outside vocabulary"
    );
    ensure!(
        valid_sha(&args.manifest_sha256),
        "manifest SHA-256 must be canonical lowercase hex"
    );
    let chains = args
        .fixed_chain
        .iter()
        .map(|value| static_chain(value))
        .collect::<Result<Vec<_>>>()?;

    let manifest_open = OpenedInput::open(&args.manifest, "static manifest", Some(MAX_TRACE))?;
    ensure!(
        args.manifest == manifest_open.path,
        "manifest argv path must be canonical absolute"
    );
    ensure!(
        manifest_open.sha256 == args.manifest_sha256,
        "independent manifest SHA-256 mismatch"
    );
    let manifest_bytes = opened_bytes(&manifest_open)?;
    let manifest: Manifest =
        serde_json::from_slice(&manifest_bytes).context("decode static manifest")?;
    let manifest_value: Value =
        serde_json::from_slice(&manifest_bytes).context("parse static manifest JSON")?;
    validate_manifest_order(&manifest_value)?;
    validate_claim_shape(&manifest)?;
    validate_static(&manifest, &args, &chains, &build_identity)?;
    let target_path = canonical_input(&args.model, "target model")?;
    ensure!(
        args.model == target_path
            && target_path.as_path() == Path::new(&claim_by_role(&manifest.assets, "target")?.path),
        "target path differs from manifest"
    );
    let drafter_path = canonical_input(&args.drafter, "drafter")?;
    ensure!(
        args.drafter == drafter_path
            && drafter_path.as_path()
                == Path::new(&claim_by_role(&manifest.assets, "drafter")?.path),
        "drafter path differs from manifest"
    );
    let fixture_path = canonical_input(&args.fixture, "scalar fixture")?;
    ensure!(
        args.fixture == fixture_path && fixture_path.as_path() == Path::new(&manifest.fixture.path),
        "fixture path differs from manifest"
    );
    let command_path = canonical_input(&args.command_manifest, "command manifest")?;
    ensure!(
        args.command_manifest == command_path
            && command_path.as_path() == Path::new(&manifest.command.path),
        "command path differs from manifest"
    );
    let mut outputs = reserve_outputs(&args.sidecar_output, &args.trace_output)?;
    ensure!(
        args.sidecar_output == outputs.sidecar_path && args.trace_output == outputs.trace_path,
        "artifact argv paths must be canonical absolute"
    );

    let target_input = open_and_verify_claim(
        &args.model,
        claim_by_role(&manifest.assets, "target")?,
        "target asset",
    )?;
    let drafter_input = open_and_verify_claim(
        &args.drafter,
        claim_by_role(&manifest.assets, "drafter")?,
        "drafter asset",
    )?;
    let fixture_input = open_and_verify_claim(&args.fixture, &manifest.fixture, "scalar fixture")?;
    let command_input = open_and_verify_claim(
        &args.command_manifest,
        &manifest.command,
        "command manifest",
    )?;
    let reducer_input = open_and_verify_claim(
        Path::new(&manifest.reducer.path),
        &manifest.reducer,
        "reducer",
    )?;
    let executable_path = std::env::current_exe()?.canonicalize()?;
    let executable_input =
        open_and_verify_claim(&executable_path, &manifest.executable, "executable")?;
    let preparation_inventory = open_and_verify_claim(
        Path::new(&manifest.preparation_binding.inventory.path),
        &manifest.preparation_binding.inventory,
        "preparation inventory",
    )?;
    let preparation_inventory_spec = open_and_verify_claim(
        Path::new(&manifest.preparation_binding.inventory_spec.path),
        &manifest.preparation_binding.inventory_spec,
        "preparation inventory spec",
    )?;
    let preparation_spec = open_and_verify_claim(
        Path::new(&manifest.preparation_binding.preparation_spec.path),
        &manifest.preparation_binding.preparation_spec,
        "preparation spec",
    )?;
    let mut source_inputs = Vec::new();
    for claim in &manifest.sources {
        source_inputs.push((
            claim.role.clone(),
            open_and_verify_claim(
                Path::new(&claim.file.path),
                &claim.file,
                &format!("source {}", claim.role),
            )?,
        ));
    }
    let mut aliases = vec![
        ("manifest", &manifest_open),
        ("target", &target_input),
        ("drafter", &drafter_input),
        ("fixture", &fixture_input),
        ("command", &command_input),
        ("reducer", &reducer_input),
        ("executable", &executable_input),
        ("preparation inventory", &preparation_inventory),
        ("preparation inventory spec", &preparation_inventory_spec),
        ("preparation spec", &preparation_spec),
    ];
    aliases.extend(
        source_inputs
            .iter()
            .map(|(role, input)| (role.as_str(), input)),
    );
    reject_input_aliases(&aliases, &outputs)?;
    let seal_path = PathBuf::from(&manifest.preparation_binding.seal_path);
    ensure!(
        seal_path != outputs.sidecar_path
            && seal_path != outputs.trace_path
            && aliases.iter().all(|(_, input)| input.path != seal_path),
        "preparation seal path aliases an acquisition input or output"
    );
    let command_bytes = opened_bytes(&command_input)?;
    let command_bindings = CommandBindings {
        executable: &executable_input.path,
        attempt_id: &args.attempt_id,
        model: &target_input.path,
        drafter: &drafter_input.path,
        prompt: &args.prompt,
        carry_token: args.carry_token,
        continuation_carry_token: args.continuation_carry_token,
        manifest: &manifest_open.path,
        command_manifest: &command_input.path,
        fixture: &fixture_input.path,
        temperature_bits: args.temperature.to_bits(),
        fixed_chains: &manifest.expected_fixed_chains,
        trace_output: &outputs.trace_path,
        sidecar_output: &outputs.sidecar_path,
    };
    validate_command(&command_bytes, &args.manifest_sha256, &command_bindings)?;
    let fixture_bytes = opened_bytes(&fixture_input)?;
    validate_scalar_fixture(&fixture_bytes, &manifest)?;

    let compiled_metallib = dflash_k0s_embedded_metallib_identity();
    ensure!(
        hex(&compiled_metallib.sha256) == manifest.embedded_metallib_sha256
            && compiled_metallib.byte_count == dflash_k0s_embedded_metallib_bytes().len(),
        "context-free compiled metallib identity differs from manifest"
    );
    let selector_source = source_inputs
        .iter()
        .find(|(role, _)| role == "mat_mat_q4_k_metal")
        .map(|(_, input)| input)
        .context("authenticated mat_mat_q4_k source absent")?;
    let runtime_binding = VerifiedRuntimeBinding {
        metal_source_sha256: selector_source.sha256.clone(),
        metallib_sha256: hex(&compiled_metallib.sha256),
        build_source_sha256: manifest.expected_build.source_sha256.clone(),
        environment: observed_environment.clone(),
    };
    ensure!(
        runtime_binding.environment == manifest.selector_dispatch_predicate.allowed_environment,
        "observed environment differs from selector predicate"
    );

    let target_g =
        GgufFile::from_opened_file(target_input.file.try_clone()?, target_input.path.clone())
            .context("parse authenticated target GGUF handle")?;
    let drafter_g =
        GgufFile::from_opened_file(drafter_input.file.try_clone()?, drafter_input.path.clone())
            .context("parse authenticated drafter GGUF handle")?;
    ensure!(
        target_g.shards.len() == 1 && drafter_g.shards.len() == 1,
        "K0-S v1 requires manifest-bound single-file GGUF assets"
    );
    validate_tensor_claims_before_metal(&manifest, &drafter_g)?;
    let target_m = Model::from_gguf(&target_g).context("parse target architecture")?;
    let head = open_dflash_drafter(&drafter_g, &target_m).context("bind DFlash drafter")?;
    ensure!(
        head.config.block_size as usize == DFLASH_K0S_BLOCK_SIZE
            && head.config.hidden_size as usize == DFLASH_K0S_HIDDEN
            && head.config.selector_rank as usize == DFLASH_K0S_RANK
            && head.config.selector_top_k as usize == DFLASH_K0S_TOP_K
            && target_m.arch.vocab_size as usize == DFLASH_K0S_VOCAB,
        "DFlash geometry differs from K0-S v1"
    );
    let tokenizer = Tokenizer::from_gguf(&target_g).context("open tokenizer")?;
    let prompt_ids = tokenizer
        .encode(&args.prompt, false)
        .context("tokenize fixed prompt")?;
    ensure!(
        !prompt_ids.is_empty()
            && prompt_ids.len() as u64 == manifest.expected_capture_context.target_context_len,
        "fixed prompt token count differs from manifest context"
    );

    let ctx = MetalContext::new().context("initialize Metal")?;
    ensure!(
        ctx.device.name().to_string() == manifest.expected_host.device_name
            && ctx.device.registryID() == manifest.expected_host.device_registry_id
            && metal_family_capabilities(&ctx.device) == manifest.expected_host.device_family,
        "Metal host/device differs from static manifest"
    );
    let mm = MetalModel::load(&ctx, &target_g, &target_m).context("load target Metal model")?;
    let mhead =
        MetalDFlashHead::load(&ctx, &drafter_g, &head).context("load drafter Metal model")?;
    ensure!(
        mhead.selector.is_some(),
        "K0-S requires the DFlash 2 selector"
    );
    let mf = MetalForward::new(&ctx, &mm);
    ensure!(
        manifest.expected_prompt.utf8_hex == hex(args.prompt.as_bytes())
            && manifest.expected_prompt.token_ids == prompt_ids
            && manifest.expected_prompt.token_ids_sha256_i32le
                == token_ids_sha256_i32le(&prompt_ids),
        "runtime prompt differs from manifest"
    );
    ensure!(
        manifest.expected_binding.drafter_checkpoint_sha256 == drafter_input.sha256,
        "drafter checkpoint binding differs from asset"
    );
    let selector_tensor = manifest
        .tensors
        .iter()
        .find(|tensor| tensor.role == "selector_hidden")
        .context("selector-hidden tensor claim absent")?;
    ensure!(
        manifest.selector_dispatch_predicate.weight_dtype == selector_tensor.dtype
            && manifest.selector_dispatch_predicate.metal_source_sha256
                == claim_by_role(&manifest.sources, "mat_mat_q4_k_metal")?.sha256,
        "selector predicate source/tensor binding mismatch"
    );

    let arm_plan = [
        ("off-A", false),
        ("on-A", true),
        ("on-B", true),
        ("off-B", false),
    ];
    let mut executions = Vec::with_capacity(4);
    for (name, diagnostic) in arm_plan {
        match run_arm(
            name,
            diagnostic,
            &manifest.attempt_id,
            &manifest.expected_rng_domains,
            &ctx,
            &mm,
            &mf,
            &mhead,
            &head.target_layer_ids,
            &prompt_ids,
            args.carry_token,
            args.continuation_carry_token,
            &manifest.selector_dispatch_predicate,
            &runtime_binding,
        ) {
            Ok(execution) => executions.push(execution),
            Err(error) => {
                let detail = format!("{error:#}");
                let stage = classify_arm_failure(&detail);
                let completed = executions.into_iter().map(|value| value.arm).collect();
                write_parity_failure(
                    &mut outputs,
                    &manifest,
                    completed,
                    Some(name),
                    stage,
                    &error,
                )?;
                return Ok(());
            }
        }
    }

    let comparison = (|| -> Result<()> {
        validate_capture_presence(
            &executions
                .iter()
                .map(|value| value.capture.is_some())
                .collect::<Vec<_>>(),
        )?;
        let arms = executions
            .iter()
            .map(|value| value.arm.clone())
            .collect::<Vec<_>>();
        ensure!(
            validate_four_arm_gate(&arms, &manifest.attempt_id, false)? == "on-A",
            "four-arm selected arm mismatch"
        );
        Ok(())
    })();
    if let Err(error) = comparison {
        for execution in &mut executions {
            refresh_arm_envelope(&mut execution.arm, &manifest.attempt_id)?;
        }
        let completed = executions.into_iter().map(|value| value.arm).collect();
        write_parity_failure(
            &mut outputs,
            &manifest,
            completed,
            None,
            "comparison",
            &error,
        )?;
        return Ok(());
    }

    macro_rules! parity_try {
        ($expression:expr, $stage:literal) => {
            match $expression {
                Ok(value) => value,
                Err(error) => {
                    let error: anyhow::Error = error.into();
                    for execution in &mut executions {
                        refresh_arm_envelope(&mut execution.arm, &manifest.attempt_id)?;
                    }
                    let completed = executions.iter().map(|value| value.arm.clone()).collect();
                    write_parity_failure(&mut outputs, &manifest, completed, None, $stage, &error)?;
                    return Ok(());
                }
            }
        };
    }

    let on_a = parity_try!(
        executions[1].capture.take().context("on-A capture absent"),
        "extraction"
    );
    let on_b = parity_try!(
        executions[2].capture.take().context("on-B capture absent"),
        "extraction"
    );
    let validate_capture = |capture: &DFlashK0sCapture| -> Result<()> {
        ensure!(
            local_capture_sha(capture) == capture.capture_sha256,
            "capture digest mismatch"
        );
        ensure!(
            capture.state_identity.carry_token == args.carry_token
                && capture.state_identity.noise_start_position
                    == manifest.expected_capture_context.noise_start_position
                && [
                    capture.state_identity.target_context_len as u64,
                    capture.state_identity.context_hidden_watermark as u64,
                    capture.state_identity.kv_context_watermark as u64,
                ] == [
                    manifest.expected_capture_context.target_context_len,
                    manifest.expected_capture_context.context_hidden_watermark,
                    manifest.expected_capture_context.kv_context_watermark,
                ]
                && hex(&capture.state_identity.noise_input_sha256)
                    == manifest.expected_binding.noise_input_sha256
                && hex(&capture.provenance.embedded_metallib_sha256)
                    == manifest.embedded_metallib_sha256
                && tensor_capture_matches(capture, &manifest.tensors),
            "capture static/state binding mismatch"
        );
        ensure!(
            capture.draft_tokens.len() == DFLASH_K0S_BLOCK_SIZE
                && capture.depths.len() == 7
                && capture.lattice.len() == DFLASH_K0S_LATTICE_ROWS,
            "capture geometry malformed"
        );
        Ok(())
    };
    if let Err(error) = validate_capture(&on_a).and_then(|_| validate_capture(&on_b)) {
        for execution in &mut executions {
            refresh_arm_envelope(&mut execution.arm, &manifest.attempt_id)?;
        }
        let completed = executions.into_iter().map(|value| value.arm).collect();
        write_parity_failure(
            &mut outputs,
            &manifest,
            completed,
            None,
            "comparison",
            &error,
        )?;
        return Ok(());
    }

    let mut layout = parity_try!(
        build_sidecar_prefixed(&on_a, &manifest.tensors, ""),
        "serialization"
    );
    let on_b_layout = parity_try!(
        build_sidecar_prefixed(&on_b, &manifest.tensors, "on-b/"),
        "serialization"
    );
    let on_a_parts = parity_try!(
        projection_parts(&on_a, &layout, &manifest, &chains, &observed_environment,),
        "serialization"
    );
    let on_b_parts = parity_try!(
        projection_parts(
            &on_b,
            &on_b_layout,
            &manifest,
            &chains,
            &observed_environment,
        ),
        "serialization"
    );
    if on_a_parts.digest != on_b_parts.digest {
        let error = anyhow!("on-A/on-B complete canonical projection mismatch");
        for execution in &mut executions {
            refresh_arm_envelope(&mut execution.arm, &manifest.attempt_id)?;
        }
        let completed = executions.into_iter().map(|value| value.arm).collect();
        write_parity_failure(
            &mut outputs,
            &manifest,
            completed,
            None,
            "comparison",
            &error,
        )?;
        return Ok(());
    }
    executions[1].arm.summary.capture_content_sha256 = Some(on_a_parts.digest.clone());
    executions[1].arm.capture_projection_sha256 = Some(on_a_parts.digest.clone());
    executions[2].arm.summary.capture_content_sha256 = Some(on_b_parts.digest.clone());
    executions[2].arm.capture_projection_sha256 = Some(on_b_parts.digest.clone());
    parity_try!(
        refresh_arm_envelope(&mut executions[1].arm, &manifest.attempt_id),
        "serialization"
    );
    parity_try!(
        refresh_arm_envelope(&mut executions[2].arm, &manifest.attempt_id),
        "serialization"
    );
    parity_try!(
        validate_four_arm_gate(
            &executions
                .iter()
                .map(|value| value.arm.clone())
                .collect::<Vec<_>>(),
            &manifest.attempt_id,
            true,
        )
        .map(|_| ()),
        "comparison"
    );
    parity_try!(
        append_sidecar_layout(&mut layout, on_b_layout),
        "serialization"
    );
    parity_try!(
        validate_artifact_caps(
            layout.bytes.len() as u64,
            0,
            manifest.sidecar_max_bytes,
            manifest.trace_max_bytes,
        ),
        "serialization"
    );
    let sidecar_sha = hex(&Sha256::digest(&layout.bytes));
    let sidecar_claim = FileClaim {
        path: outputs.sidecar_path.to_string_lossy().into_owned(),
        bytes: layout.bytes.len() as u64,
        sha256: sidecar_sha,
        max_bytes: manifest.sidecar_max_bytes,
    };
    let production = chain_json(
        "production-greedy".into(),
        args.carry_token,
        &on_a.production_chain,
    );
    let fixed = chains
        .iter()
        .map(|item| {
            chain_json(
                item.name.clone(),
                item.initial_carry,
                &dflash_k0s_traverse_slots(&on_a.lattice, item.initial_carry, &item.slots),
            )
        })
        .collect::<Vec<_>>();
    let capture_digest = hex(&on_a.capture_sha256);
    let draft_digest = hex(&on_a.state_identity.draft_tokens_sha256);
    let provenance = parity_try!(
        capture_provenance(&on_a, &manifest, &observed_environment),
        "serialization"
    );
    let exclusion_allowlist = ["arm_name", "session_id", "event_envelope_sha256"];
    let exclusion_value = parity_try!(
        serde_json::to_value(exclusion_allowlist).map_err(anyhow::Error::from),
        "serialization"
    );
    let exclusion_content_sha256 =
        parity_try!(canonical_json_sha(&exclusion_value), "serialization");
    let on_b_projection = OnBProjection {
        exclusion_allowlist,
        exclusion_content_sha256,
        depths: on_b_parts.depths,
        lattices: on_b_parts.lattices,
        capture: on_b_parts.capture,
        production_chain: on_b_parts.production_chain,
        fixed_chains: on_b_parts.fixed_chains,
        provenance: on_b_parts.provenance,
        projection_sha256: on_b_parts.digest.clone(),
    };
    let parity = ParityJson {
        status: "passed",
        arm_order: ["off-A", "on-A", "on-B", "off-B"],
        selected_arm: "on-A",
        arms: executions.iter().map(|value| value.arm.clone()).collect(),
        comparison_fields: [
            "first",
            "continuation",
            "observer_baseline",
            "common_production_content_sha256",
            "on_capture_content_sha256",
        ],
    };
    let run_payload = RunPayload {
        authority: AUTHORITY,
        attempt_id: manifest.attempt_id.clone(),
        geometry: Geometry {
            block_size: 8,
            depths: 7,
            top_k: 16,
            rank: 256,
            hidden: 5120,
            vocab: 248320,
            rows: 97,
        },
        request: manifest.expected_request.request.clone(),
        proposal_abstention: Abstention {
            enabled: false,
            p_min: None,
            n_min: None,
        },
        ignored_target_policy: manifest.expected_request.ignored_target_policy.clone(),
        binding: manifest.expected_binding.clone(),
        provenance,
        capture: capture_json(&on_a),
        diagnostic_nonperturbation_parity: parity,
        on_b_projection,
        identities: Identities {
            sidecar: sidecar_claim,
            reducer: manifest.reducer.clone(),
            executable: manifest.executable.clone(),
            fixture: manifest.fixture.clone(),
            command: manifest.command.clone(),
            sources: manifest.sources.clone(),
            assets: manifest.assets.clone(),
        },
        semantic_references: manifest.semantic_references.clone(),
        tensors: manifest.tensors.clone(),
        sidecar_registry: layout.ranges.clone(),
        production_chain: production,
        fixed_chains: fixed,
    };
    let mut records = Vec::with_capacity(RECORDS);
    records.push(parity_try!(
        serialize_record(&manifest.run_id, &manifest.attempt_id, "run", run_payload),
        "serialization"
    ));
    for depth in 1..=7 {
        let value = &on_a.depths[depth - 1];
        records.push(parity_try!(
            serialize_record(
                &manifest.run_id,
                &manifest.attempt_id,
                "depth",
                DepthPayload {
                    depth,
                    position: value.position,
                    production_call_id: manifest.expected_binding.production_call_id.clone(),
                    drafter_checkpoint_sha256: manifest
                        .expected_binding
                        .drafter_checkpoint_sha256
                        .clone(),
                    proposal_construction_id: manifest
                        .expected_binding
                        .proposal_construction_id
                        .clone(),
                    noise_input_sha256: manifest.expected_binding.noise_input_sha256.clone(),
                    synchronized_capture_sha256: capture_digest.clone(),
                    draft_tokens_sha256_i32le: draft_digest.clone(),
                    diagnostic_state_sha256: hex(&on_a.state_identity.diagnostic_state_sha256),
                    z_f32_bits: value
                        .selector_hidden_bits
                        .iter()
                        .copied()
                        .map(bits)
                        .collect(),
                    full_logits_range_id: layout.full_logits[depth - 1].clone(),
                    top16_ids: value.top_k_ids.clone(),
                    unary_f32_bits: value.unary_bits.iter().copied().map(bits).collect(),
                    topk_issues: topk_issues(&value.top_k_issues),
                },
            ),
            "serialization"
        ));
        let lattice = parity_try!(lattice_json(&on_a, &layout, depth), "serialization");
        records.push(parity_try!(
            serialize_record(&manifest.run_id, &manifest.attempt_id, "lattice", lattice,),
            "serialization"
        ));
    }
    records.push(parity_try!(
        serialize_record(
            &manifest.run_id,
            &manifest.attempt_id,
            "end",
            EndPayload {
                producer_status: "complete",
                authority: AUTHORITY,
            },
        ),
        "serialization"
    ));
    ensure!(
        records.len() == RECORDS,
        "trace record count invariant failed"
    );
    let trace_bytes = parity_try!(
        records.iter().try_fold(0u64, |sum, row| {
            sum.checked_add(row.len() as u64)
                .context("trace size overflow")
        }),
        "serialization"
    );
    parity_try!(
        validate_artifact_caps(
            layout.bytes.len() as u64,
            trace_bytes,
            manifest.sidecar_max_bytes,
            manifest.trace_max_bytes,
        ),
        "serialization"
    );

    let custody = (|| -> Result<()> {
        ensure!(
            target_g.revalidate_retained_shard_stamps()?.len() == 1
                && drafter_g.revalidate_retained_shard_stamps()?.len() == 1,
            "final retained GGUF source validation failed"
        );
        for (label, input) in &aliases {
            input.final_custody_check(label)?;
        }
        Ok(())
    })();
    parity_try!(custody, "serialization");

    outputs.sidecar.write_all(&layout.bytes)?;
    outputs.sidecar.flush()?;
    outputs.sidecar.sync_all()?;
    ensure!(
        outputs.sidecar.metadata()?.len() == layout.bytes.len() as u64,
        "sidecar final size mismatch"
    );
    for row in &records {
        outputs.trace.write_all(row)?;
    }
    outputs.trace.flush()?;
    outputs.trace.sync_all()?;
    ensure!(
        outputs.trace.metadata()?.len() == trace_bytes,
        "trace final size mismatch"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen_llm::{
        metal::KernelTraceCounters,
        metal_dflash::{
            DFlashK0sDepth, DFlashK0sLatticeRow, DFlashK0sParitySummary, DFlashK0sProvenance,
            DFlashK0sSelectorDispatchIdentity, DFlashK0sSelectorInputIdentity, DFlashK0sSlot,
            DFlashK0sStateIdentity, DFlashK0sTensorProvenance,
        },
        tensor::{GgmlType, TensorDesc},
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "qwen-k0s-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&path).unwrap();
        path.canonicalize().unwrap()
    }

    fn mock_phase(seed: u8) -> ArmPhaseJson {
        let sha = |offset: u8| format!("{:064x}", u64::from(seed) + u64::from(offset));
        let selector = SelectorDispatchPredicate {
            tag: "dflash_k0s.selector_hidden_projection.v1".into(),
            kernel: K0S_SELECTOR_KERNEL.into(),
            weight_dtype: K0S_SELECTOR_WEIGHT_DTYPE.into(),
            input_dtype: "F32".into(),
            output_dtype: "F32".into(),
            weight_dtype_id: 12,
            input_dtype_id: 0,
            output_dtype_id: 0,
            n: 8,
            h: 5120,
            r: 256,
            grid: [1, 1, 1],
            threads: [1, 1, 1],
            metal_source_sha256: "a".repeat(64),
            metallib_sha256: "b".repeat(64),
            build_source_sha256: "d".repeat(64),
            allowed_environment: BTreeMap::from([("QWEN_METAL_LEASE_WAIT".into(), "1".into())]),
        };
        ArmPhaseJson {
            carry_token: if seed == 1 { 42 } else { 43 },
            noise_start_position: if seed == 1 { 2 } else { 3 },
            target_sha256: sha(1),
            dflash_sha256: sha(2),
            state_sha256: sha(3),
            draft_tokens_count: 8,
            draft_tokens_sha256_i32le: hash_i32(&(1..=8).collect::<Vec<_>>()),
            full_logits_count: 7 * DFLASH_K0S_VOCAB,
            full_logits_sha256_f32le: sha(4),
            topk_count: 7 * DFLASH_K0S_TOP_K,
            topk_sha256_i32le: sha(5),
            unary_count: 7 * DFLASH_K0S_TOP_K,
            unary_sha256_f32le: sha(6),
            z_count: 7 * DFLASH_K0S_RANK,
            z_sha256_f32le: sha(7),
            dispatch_census: vec![DispatchClaim {
                family: "dflash_tail".into(),
                tag: Some("dflash_k0s.selector_hidden_projection.v1".into()),
                encoder_ordinal: 0,
                encoder_concurrent: false,
                kernel: K0S_SELECTOR_KERNEL.into(),
                grid: [1, 1, 1],
                threads: [1, 1, 1],
                grid_threadgroups: 1,
                threadgroup_threads: 1,
            }],
            kernel_trace: KernelTrace {
                encoders: 1,
                concurrent_encoders: 0,
                dispatches: 1,
            },
            runtime_selector_contract: selector,
        }
    }

    fn mock_arms(associations_complete: bool) -> Vec<ArmJson> {
        let attempt = "mock-attempt";
        let first = mock_phase(1);
        let continuation = mock_phase(2);
        let common = canonical_domain_digest(
            "qwen.dflash_k0s.common_production.v1",
            &CommonProductionDigest {
                first: &first,
                continuation: &continuation,
            },
        )
        .unwrap();
        let capture = "c".repeat(64);
        let mut sequence = 1u64;
        [
            ("off-A", false),
            ("on-A", true),
            ("on-B", true),
            ("off-B", false),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, (name, diagnostic))| {
            let session = format!("{:064x}", index + 1);
            let event = |phase_name: &'static str,
                         phase: &ArmPhaseJson,
                         observed: bool,
                         sequence: &mut u64| {
                if observed {
                    let current = *sequence;
                    *sequence += 1;
                    let tokens = (1..=8).collect::<Vec<_>>();
                    bind_event(
                        attempt,
                        name,
                        &session,
                        phase_name,
                        "observed",
                        Some(current),
                        Some(
                            phase_event_envelope_sha256(phase, &session, current, &tokens).unwrap(),
                        ),
                        Some(session.clone()),
                        Some(tokens),
                    )
                    .unwrap()
                } else {
                    bind_event(
                        attempt, name, &session, phase_name, "plain", None, None, None, None,
                    )
                    .unwrap()
                }
            };
            let first_event = event("first", &first, diagnostic, &mut sequence);
            let continuation_event = event("continuation", &continuation, true, &mut sequence);
            let association = (associations_complete && diagnostic).then(|| capture.clone());
            let mut arm = ArmJson {
                name,
                diagnostic,
                session_id: session,
                first_event,
                continuation_event,
                summary: ArmSummaryJson {
                    domain: "qwen.dflash_k0s.parity_arm_summary.v1",
                    first: first.clone(),
                    continuation: continuation.clone(),
                    observer_baseline: ObserverBaselineJson {
                        before_sha256: observer_sha([0, 0, 0]),
                        after_sha256: observer_sha([0, 0, 0]),
                        restored: true,
                    },
                    common_production_content_sha256: common.clone(),
                    capture_content_sha256: association.clone(),
                },
                rng_domains: arm_rng_domains(
                    attempt,
                    name,
                    &["request_rng".into(), "proposal_rng".into()],
                )
                .unwrap(),
                capture_projection_sha256: association,
                arm_envelope_sha256: String::new(),
            };
            refresh_arm_envelope(&mut arm, attempt).unwrap();
            arm
        })
        .collect()
    }

    fn mock_manifest() -> Manifest {
        let claim = serde_json::json!({
            "path": "/tmp/mock", "bytes": 1, "sha256": "0".repeat(64), "max_bytes": 1
        });
        let reference = serde_json::json!({"commit": "0".repeat(40), "sha256": "0".repeat(64)});
        let sources = REQUIRED_SOURCE_ROLES
            .iter()
            .map(|role| serde_json::json!({"role": role, "path": format!("/tmp/{role}"), "bytes": 1, "sha256": if *role == "mat_mat_q4_k_metal" { "a".repeat(64) } else { "0".repeat(64) }, "max_bytes": 1}))
            .collect::<Vec<_>>();
        let assets = ["target", "drafter"]
            .into_iter()
            .map(|role| serde_json::json!({"role": role, "path": format!("/tmp/{role}"), "bytes": 1, "sha256": "0".repeat(64), "max_bytes": 1}))
            .collect::<Vec<_>>();
        serde_json::from_value(serde_json::json!({
            "schema": MANIFEST_SCHEMA,
            "schema_version": 1,
            "run_id": "mock-run",
            "attempt_id": "mock-attempt",
            "trace_max_bytes": MAX_TRACE,
            "sidecar_max_bytes": MAX_SIDECAR,
            "reducer": claim.clone(),
            "executable": claim.clone(),
            "fixture": claim.clone(),
            "command": claim.clone(),
            "sources": sources,
            "assets": assets,
            "semantic_references": {
                "mlx_model_mlx_py": reference.clone(),
                "vllm_qwen3_dflash2_py": reference.clone(),
                "vllm_speculator_py": reference.clone(),
                "llama_cpp_dflash_cpp": reference.clone(),
                "llama_cpp_speculative_cpp": reference
            },
            "tensors": [],
            "expected_request": {
                "request": {"temperature_f32_bits": "0x3f800000"},
                "ignored_target_policy": {
                    "variant_a": {"top_k": 1, "top_p_f32_bits": "0x3f800000", "min_p_f32_bits": "0x00000000", "grammar": null, "penalties": null},
                    "variant_b": {"top_k": 2, "top_p_f32_bits": "0x3f000000", "min_p_f32_bits": "0x00000000", "grammar": null, "penalties": null}
                }
            },
            "expected_prompt": {"utf8_hex": "61", "token_ids": [1], "token_ids_sha256_i32le": "0".repeat(64), "tokenizer_identity_sha256": "0".repeat(64)},
            "expected_binding": {"production_call_id": "p", "drafter_checkpoint_sha256": "0".repeat(64), "proposal_construction_id": "q", "noise_input_sha256": "0".repeat(64)},
            "expected_continuation_carry_token": 43,
            "expected_rng_domains": ["request_rng", "proposal_rng"],
            "expected_fixed_chains": [],
            "expected_capture_context": {"definition_version": CAPTURE_DEFINITION, "carry_token": 42, "noise_start_position": 2, "target_context_len": 2, "context_hidden_watermark": 2, "kv_context_watermark": 2},
            "expected_build": {"commit": "0".repeat(40), "source_sha256": "d".repeat(64), "dirty": false, "compiler": "rustc", "compiler_version": "mock", "target": "aarch64-apple-darwin", "profile": "release", "features": ["dflash-k0s-diagnostics"]},
            "expected_host": {"os": "macos", "arch": "aarch64", "device_name": "mock", "device_registry_id": 1, "device_family": "mtl-gpu-family-v1:mock"},
            "embedded_metallib_sha256": "b".repeat(64),
            "selector_dispatch_predicate": serde_json::to_value(mock_phase(1).runtime_selector_contract).unwrap(),
            "preparation_binding": {"inventory": claim.clone(), "inventory_spec": claim.clone(), "preparation_spec": claim.clone(), "seal_path": "/tmp/mock-seal"},
            "scalar_contract": {"artifact": claim, "compiler": "rustc", "compiler_version": "mock", "target": "aarch64-apple-darwin", "profile": "release", "fixture_domain": SCALAR_DOMAIN, "fixture_sha256": SCALAR_DIGEST, "vectors": []}
        }))
        .unwrap()
    }

    fn synthetic_capture() -> DFlashK0sCapture {
        let descriptor = |name: &str, shape: Vec<u64>| TensorDesc {
            name: name.into(),
            shape,
            dtype: GgmlType::F32,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: 4,
        };
        let dispatch = DFlashK0sDispatchCensusRow {
            family: "dflash_tail".into(),
            tag: Some("dflash_k0s.selector_hidden_projection.v1".into()),
            encoder_ordinal: 0,
            encoder_concurrent: false,
            kernel: "synthetic".into(),
            grid: [1, 1, 1],
            threads: [1, 1, 1],
            grid_threadgroups: 1,
            threadgroup_threads: 1,
        };
        let depths = (1..8)
            .map(|depth| DFlashK0sDepth {
                depth,
                position: 100 + depth as u32,
                full_logits_bits: vec![0; DFLASH_K0S_VOCAB],
                top_k_ids: (0..16).collect(),
                unary_bits: vec![0; 16],
                selector_hidden_bits: vec![0; 256],
                top_k_issues: Vec::new(),
            })
            .collect::<Vec<_>>();
        let mut lattice = Vec::with_capacity(97);
        let mut raw_rows = Vec::with_capacity(97 * 17);
        for depth in 1..8 {
            let count = if depth == 1 { 1 } else { 16 };
            for predecessor in 0..count {
                let predecessor_slot = (depth > 1).then_some(predecessor);
                let predecessor_token = if depth == 1 { 42 } else { predecessor as i32 };
                raw_rows.push(DFlashK0sRawRow {
                    side: DFlashK0sCodebookSide::Predecessor,
                    depth,
                    predecessor_slot,
                    candidate_slot: None,
                    token_id: predecessor_token,
                    bytes: vec![0; 1024],
                });
                let slots = (0..16)
                    .map(|slot| {
                        raw_rows.push(DFlashK0sRawRow {
                            side: DFlashK0sCodebookSide::Successor,
                            depth,
                            predecessor_slot,
                            candidate_slot: Some(slot),
                            token_id: slot as i32,
                            bytes: vec![0; 1024],
                        });
                        DFlashK0sSlot {
                            candidate_slot: slot,
                            token_id: slot as i32,
                            unary_bits: 0,
                            score_bits: Some((slot as f32).to_bits()),
                        }
                    })
                    .collect();
                lattice.push(DFlashK0sLatticeRow {
                    row_index: lattice.len(),
                    depth,
                    predecessor_slot,
                    predecessor_token,
                    slots,
                    issues: Vec::new(),
                    greedy_slot: 15,
                    has_valid_choice: true,
                });
            }
        }
        let production_chain = dflash_k0s_traverse_slots(&lattice, 42, &[15; 7]);
        let provenance = DFlashK0sProvenance {
            selector_hidden: DFlashK0sTensorProvenance {
                descriptor: descriptor("selector_hidden.weight", vec![5120, 256]),
                full_tensor_sha256: [1; 32],
            },
            predecessor: DFlashK0sTensorProvenance {
                descriptor: descriptor("selector_predecessor.weight", vec![256, 248320]),
                full_tensor_sha256: [2; 32],
            },
            successor: DFlashK0sTensorProvenance {
                descriptor: descriptor("selector_successor.weight", vec![256, 248320]),
                full_tensor_sha256: [3; 32],
            },
            embedded_metallib_sha256: [4; 32],
        };
        let mut capture = DFlashK0sCapture {
            draft_tokens: std::iter::once(42)
                .chain(std::iter::repeat_n(15, 7))
                .collect(),
            draft_token_bits: std::iter::once(42)
                .chain(std::iter::repeat_n(15, 7))
                .collect(),
            depths,
            lattice,
            production_chain,
            raw_rows,
            provenance,
            dispatch_census: vec![dispatch.clone()],
            selector_hidden_dispatch: dispatch,
            kernel_trace: KernelTraceCounters {
                encoders: 1,
                concurrent_encoders: 0,
                dispatches: 1,
            },
            state_identity: DFlashK0sStateIdentity {
                carry_token: 42,
                noise_start_position: 100,
                target_context_len: 100,
                context_hidden_watermark: 100,
                kv_context_watermark: 100,
                draft_tokens_sha256: [5; 32],
                noise_input_sha256: [6; 32],
                synchronized_event_sha256: [7; 32],
                diagnostic_state_sha256: [8; 32],
            },
            capture_sha256: [0; 32],
            content_sha256: [0; 32],
        };
        capture.capture_sha256 = local_capture_sha(&capture);
        capture.content_sha256 =
            qwen_llm::metal_dflash::dflash_k0s_capture_content_sha256(&capture);
        capture
    }

    fn synthetic_tensors() -> Vec<TensorClaim> {
        ["selector_hidden", "predecessor", "successor"]
            .into_iter()
            .map(|role| TensorClaim {
                role: role.into(),
                asset_role: "drafter".into(),
                name: role.into(),
                dtype: "F32".into(),
                shape: vec![1],
                offset: 0,
                bytes: 4,
                sha256: "0".repeat(64),
                orientation: "synthetic".into(),
                row_domain: None,
            })
            .collect()
    }

    #[test]
    fn fixed_chain_parser_is_bounded_and_positional() {
        let chain = static_chain("alternate:17:0,1,2,3,4,5,6").unwrap();
        assert_eq!(chain.initial_carry, 17);
        assert_eq!(chain.slots, [0, 1, 2, 3, 4, 5, 6]);
        assert!(static_chain("bad:17:0,1").is_err());
        assert!(static_chain("bad:17:0,1,2,3,4,5,16").is_err());
    }

    fn command_argv(executable: &Path, digest: &str) -> Vec<String> {
        [
            executable.to_string_lossy().into_owned(),
            "dflash-k0s-lattice".into(),
            "--attempt-id".into(),
            "attempt-1".into(),
            "--model".into(),
            "target.gguf".into(),
            "--drafter".into(),
            "drafter.gguf".into(),
            "--prompt".into(),
            "fixed prompt".into(),
            "--carry-token".into(),
            "42".into(),
            "--continuation-carry-token".into(),
            "43".into(),
            "--manifest".into(),
            "manifest.json".into(),
            "--manifest-sha256".into(),
            digest.into(),
            "--command-manifest".into(),
            "command.json".into(),
            "--fixture".into(),
            "fixture.json".into(),
            "--temperature".into(),
            "0.7".into(),
            "--fixed-chain".into(),
            "fixed:42:0,1,2,3,4,5,6".into(),
            "--trace-output".into(),
            "trace.jsonl".into(),
            "--sidecar-output".into(),
            "sidecar.bin".into(),
        ]
        .into()
    }

    #[test]
    fn command_placeholder_breaks_manifest_hash_cycle_and_rejects_mutations() {
        let executable = std::env::current_exe().unwrap().canonicalize().unwrap();
        let digest = "a".repeat(64);
        let actual = command_argv(&executable, &digest);
        assert_eq!(
            &actual[1..6],
            [
                "dflash-k0s-lattice",
                "--attempt-id",
                "attempt-1",
                "--model",
                "target.gguf"
            ]
        );
        let chains = [static_chain("fixed:42:0,1,2,3,4,5,6").unwrap()];
        let bindings = CommandBindings {
            executable: &executable,
            attempt_id: "attempt-1",
            model: Path::new("target.gguf"),
            drafter: Path::new("drafter.gguf"),
            prompt: "fixed prompt",
            carry_token: 42,
            continuation_carry_token: 43,
            manifest: Path::new("manifest.json"),
            command_manifest: Path::new("command.json"),
            fixture: Path::new("fixture.json"),
            temperature_bits: 0.7f32.to_bits(),
            fixed_chains: &chains,
            trace_output: Path::new("trace.jsonl"),
            sidecar_output: Path::new("sidecar.bin"),
        };
        let mut expected = actual.clone();
        let digest_index = expected
            .iter()
            .position(|arg| arg == "--manifest-sha256")
            .unwrap()
            + 1;
        expected[digest_index] = COMMAND_MANIFEST_PLACEHOLDER.into();
        let bytes = serde_json::to_vec(&CommandManifest {
            argv: expected.clone(),
        })
        .unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains(&digest));
        validate_command_argv(&bytes, &actual, &digest, &bindings).unwrap();

        let mut reordered_actual = actual.clone();
        reordered_actual[2..6].rotate_left(2);
        let mut reordered_static = reordered_actual.clone();
        reordered_static[digest_index] = COMMAND_MANIFEST_PLACEHOLDER.into();
        let reordered_bytes = serde_json::to_vec(&CommandManifest {
            argv: reordered_static,
        })
        .unwrap();
        assert!(
            validate_command_argv(&reordered_bytes, &reordered_actual, &digest, &bindings).is_err()
        );

        let mut duplicate_placeholder = expected.clone();
        duplicate_placeholder[7] = COMMAND_MANIFEST_PLACEHOLDER.into();
        let bytes = serde_json::to_vec(&CommandManifest {
            argv: duplicate_placeholder,
        })
        .unwrap();
        assert!(validate_command_argv(&bytes, &actual, &digest, &bindings).is_err());

        let mut missing_placeholder = expected.clone();
        missing_placeholder[digest_index] = "not-a-placeholder".into();
        let bytes = serde_json::to_vec(&CommandManifest {
            argv: missing_placeholder,
        })
        .unwrap();
        assert!(validate_command_argv(&bytes, &actual, &digest, &bindings).is_err());

        let mut duplicate_digest = actual.clone();
        duplicate_digest[7] = digest.clone();
        let bytes = serde_json::to_vec(&CommandManifest {
            argv: expected.clone(),
        })
        .unwrap();
        assert!(validate_command_argv(&bytes, &duplicate_digest, &digest, &bindings).is_err());

        let mut missing_digest = actual.clone();
        missing_digest[digest_index] = "b".repeat(64);
        assert!(validate_command_argv(&bytes, &missing_digest, &digest, &bindings).is_err());

        let mut mutated = actual;
        mutated[3] = "other.gguf".into();
        assert!(validate_command_argv(&bytes, &mutated, &digest, &bindings).is_err());
        let mut wrong_subcommand = command_argv(&executable, &digest);
        wrong_subcommand[1] = "decode".into();
        assert!(validate_command_argv(&bytes, &wrong_subcommand, &digest, &bindings).is_err());
        let mut wrong_executable = command_argv(&executable, &digest);
        wrong_executable[0] = "/bin/sh".into();
        assert!(validate_command_argv(&bytes, &wrong_executable, &digest, &bindings).is_err());
    }

    #[test]
    fn duplicate_json_fields_fail_before_order_validation() {
        let manifest_error = serde_json::from_slice::<Manifest>(br#"{"schema":"a","schema":"b"}"#)
            .unwrap_err()
            .to_string();
        assert!(manifest_error.contains("duplicate field"));
        let command_error = serde_json::from_slice::<CommandManifest>(br#"{"argv":[],"argv":[]}"#)
            .unwrap_err()
            .to_string();
        assert!(command_error.contains("duplicate field"));
        let scalar_error = serde_json::from_slice::<ScalarFixtureFile>(
                br#"{"schema":"a","schema":"b","schema_version":1,"fixture_domain":"x","fixture_sha256":"y","vectors":[]}"#
            )
            .unwrap_err()
            .to_string();
        assert!(scalar_error.contains("duplicate field"));
    }

    #[test]
    fn source_roles_caps_unique_chains_and_git_oid_bounds_are_fixed() {
        assert!(REQUIRED_SOURCE_ROLES.contains(&"dflash_k0s_rs"));
        assert!(REQUIRED_SOURCE_ROLES.contains(&"mat_mat_q4_k_metal"));
        assert_eq!(REQUIRED_SOURCE_ROLES.len(), 11);
        let file = || FileClaim {
            path: "/synthetic".into(),
            bytes: 1,
            sha256: "a".repeat(64),
            max_bytes: 1,
        };
        let mut sources = REQUIRED_SOURCE_ROLES
            .iter()
            .map(|role| RoleClaim {
                role: (*role).into(),
                file: file(),
            })
            .collect::<Vec<_>>();
        assert!(validate_source_roles(&sources).is_ok());
        sources[1].role = "dflash_k0s_rs".into();
        assert!(validate_source_roles(&sources).is_err());
        sources[1].role = REQUIRED_SOURCE_ROLES[1].into();
        sources.push(RoleClaim {
            role: "dflash_k0s_rs".into(),
            file: file(),
        });
        assert!(validate_source_roles(&sources).is_err());
        assert!(valid_git_oid(&"a".repeat(40)));
        assert!(valid_git_oid(&"b".repeat(64)));
        assert!(!valid_git_oid(&"c".repeat(39)));
        let chains = [
            static_chain("same:42:0,1,2,3,4,5,6").unwrap(),
            static_chain("same:42:6,5,4,3,2,1,0").unwrap(),
        ];
        assert!(validate_chain_set(&chains, 42).is_err());
        assert!(validate_chain_set(&chains[..1], 41).is_err());
        assert!(validate_artifact_caps(8, 8, 8, 8).is_ok());
        assert!(validate_artifact_caps(9, 8, 8, 8).is_err());
        assert!(validate_artifact_caps(8, 9, 8, 8).is_err());
    }

    #[test]
    fn final_custody_rehash_detects_same_vnode_mutation() {
        let root = temp_dir();
        let path = root.join("custody");
        std::fs::write(&path, b"first").unwrap();
        let opened = OpenedInput::open(&path, "custody", None).unwrap();
        opened.final_custody_check("custody").unwrap();
        std::fs::write(&path, b"other").unwrap();
        assert_eq!(opened.file.metadata().unwrap().ino(), opened.inode);
        assert!(opened.final_custody_check("custody").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scalar_fixture_is_exact_six_case_v2() {
        let fixture = dflash_k0s_scalar_contract_fixture();
        assert_eq!(fixture.cases.len(), 6);
        assert_eq!(hex(&fixture.fixture_sha256), SCALAR_DIGEST);
        assert!(fixture.cases.iter().all(|case| case.a_bits.len() == 256
            && case.z_bits.len() == 256
            && case.successor_bits.len() == 256));
    }

    #[test]
    fn inventory_scan_hashes_token_strings_and_mask_from_one_fd() {
        fn string(bytes: &mut Vec<u8>, value: &[u8]) {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value);
        }
        let root = temp_dir();
        let path = root.join("inventory.gguf");
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        string(&mut bytes, b"tokenizer.ggml.tokens");
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        string(&mut bytes, b"a");
        string(&mut bytes, b"bc");
        string(&mut bytes, b"tokenizer.ggml.mask_token_id");
        bytes.extend_from_slice(&10u32.to_le_bytes());
        bytes.extend_from_slice(&248070u64.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        let opened = OpenedInput::open(&path, "synthetic inventory GGUF", None).unwrap();
        let scan = scan_inventory_gguf(&opened).unwrap();
        let mut digest = Sha256::new();
        digest.update(1u64.to_le_bytes());
        digest.update(b"a");
        digest.update(2u64.to_le_bytes());
        digest.update(b"bc");
        assert_eq!(scan.version, 3);
        assert_eq!(scan.tokenizer_count, Some(2));
        assert_eq!(scan.tokenizer_sha256, Some(hex(&digest.finalize())));
        assert_eq!(
            scan.masks,
            vec![("tokenizer.ggml.mask_token_id".into(), 248070)]
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn parity_failure_payload_order_is_schema_exact() {
        let file = FileClaim {
            path: "/synthetic".into(),
            bytes: 1,
            sha256: "0".repeat(64),
            max_bytes: 1,
        };
        let payload = ParityFailure {
            completed_arms: Vec::new(),
            failed_arm: Some("off-A"),
            failed_stage: Some("session_setup"),
            first_mismatch: Some("synthetic".into()),
            observer_cleanup: true,
            identities: FailureIdentities {
                reducer: file.clone(),
                executable: file.clone(),
                fixture: file.clone(),
                command: file.clone(),
                sources: Vec::new(),
                assets: Vec::new(),
                build: BuildClaim {
                    commit: "0".repeat(40),
                    source_sha256: "0".repeat(64),
                    dirty: false,
                    compiler: "rustc".into(),
                    compiler_version: "synthetic".into(),
                    target: "aarch64-apple-darwin".into(),
                    profile: "release".into(),
                    features: vec!["dflash-k0s-diagnostics".into()],
                },
                host: HostClaim {
                    os: "macos".into(),
                    arch: "aarch64".into(),
                    device_name: "synthetic".into(),
                    device_registry_id: 1,
                    device_family: "synthetic".into(),
                },
                embedded_metallib_sha256: "0".repeat(64),
            },
            authority: AUTHORITY,
            status: "failed",
        };
        let value = serde_json::to_value(payload).unwrap();
        assert_eq!(
            value.as_object().unwrap().keys().collect::<Vec<_>>(),
            [
                "completed_arms",
                "failed_arm",
                "failed_stage",
                "first_mismatch",
                "observer_cleanup",
                "identities",
                "authority",
                "status",
            ]
        );
    }

    #[test]
    fn canonical_domain_digest_matches_independent_python_fixture() {
        #[derive(Serialize)]
        struct Fixture<'a> {
            a: &'a str,
            n: u64,
        }
        assert_eq!(
            canonical_domain_digest("test.domain", &Fixture { a: "b", n: 1 }).unwrap(),
            "819e15c7abda5e1e4b292b5e1b685381d313c787e78558b3fdb8076d9c147e91"
        );
    }

    #[test]
    fn event_and_rng_bindings_match_independent_python_fixtures() {
        let event = bind_event(
            "a1", "off-A", "s1", "first", "plain", None, None, None, None,
        )
        .unwrap();
        assert_eq!(
            event.wrapper_binding_sha256,
            "5fceb2fe7d13126f3d8e0e73ae39c708d5903daa200fbed90cd3a335f1c5ffcb"
        );
        let session = "11".repeat(32);
        let observed = bind_event(
            "a1",
            "on-A",
            &session,
            "first",
            "observed",
            Some(1),
            Some("22".repeat(32)),
            Some(session.clone()),
            Some((1..=8).collect()),
        )
        .unwrap();
        assert_eq!(
            observed.wrapper_binding_sha256,
            "d62936c01696e736c74a6f9eb1a257cc43a3e9d9f19cc813a3e249ef9c12e68d"
        );
        let rng = arm_rng_domains("a1", "off-A", &["request_rng".into()]).unwrap();
        assert_eq!(
            rng[0].scope,
            "statically_unreachable_no_rng_object_constructed"
        );
        assert_eq!(
            rng[0].absent_state_sha256,
            "732b2ffa723e427c2760a1e552d0c42f7a10638c18f2db5a22a5e7b9d834a711"
        );
        assert_eq!((rng[0].before_counter, rng[0].after_counter), (0, 0));
    }

    #[test]
    fn library_event_envelope_matches_independent_count_framing_vector() {
        let summary = DFlashK0sParitySummary {
            draft_tokens: (1..=8).collect(),
            dispatch_census: Vec::new(),
            kernel_trace: [0, 0, 0],
            selector_inputs: DFlashK0sSelectorInputIdentity {
                full_logits_count: 7 * DFLASH_K0S_VOCAB,
                full_logits_sha256_f32le: [1; 32],
                top_k_ids_count: 7 * DFLASH_K0S_TOP_K,
                top_k_ids_sha256_i32le: [2; 32],
                unary_count: 7 * DFLASH_K0S_TOP_K,
                unary_sha256_f32le: [3; 32],
                selector_hidden_count: 7 * DFLASH_K0S_RANK,
                selector_hidden_sha256_f32le: [4; 32],
            },
            selector_dispatch: DFlashK0sSelectorDispatchIdentity {
                weight_dtype: GgmlType::Q4_K,
                input_dtype: GgmlType::F32,
                output_dtype: GgmlType::F32,
                block_size: 8,
                hidden_size: 5120,
                selector_rank: 256,
            },
            diagnostic_state_sha256: [5; 32],
            carry_token: 42,
            noise_start_position: 2,
            session_binding_sha256: [6; 32],
        };
        assert_eq!(
            hex(&dflash_k0s_event_envelope_sha256(&summary, 9)),
            "46a221b28a75252e2bd4efcc1905bad1d0221f282f92971fe92be56bebc1f1f9"
        );
    }

    #[test]
    fn k0s_v1_selector_dtype_is_exact_q4_k_discriminant_twelve() {
        assert_eq!(dtype_name(GgmlType::Q4_K).unwrap(), "Q4_K");
        assert_eq!(GgmlType::Q4_K as i32, 12);
        require_q4_selector_dtype(GgmlType::Q4_K).unwrap();
        require_q4_selector_name("Q4_K").unwrap();
        for (dtype, name, id) in [
            (GgmlType::F32, "F32", 0),
            (GgmlType::F16, "F16", 1),
            (GgmlType::Q8_0, "Q8_0", 8),
            (GgmlType::BF16, "BF16", 30),
        ] {
            assert_eq!(dtype_name(dtype).unwrap(), name);
            assert_eq!(dtype as i32, id);
            assert!(require_q4_selector_dtype(dtype).is_err());
            assert!(require_q4_selector_name(name).is_err());
        }
        assert_eq!(K0S_SELECTOR_KERNEL, "kernel_mat_mat_q4_K_f32");
        assert!(REQUIRED_SOURCE_ROLES.contains(&"mat_mat_q4_k_metal"));
    }

    #[test]
    fn behavior_environment_rejects_all_selector_and_metal_overrides() {
        let clean = vec![
            (OsString::from("PATH"), OsString::from("/usr/bin")),
            (OsString::from("QWEN_METAL_LEASE_WAIT"), OsString::from("1")),
        ];
        assert_eq!(
            behavior_environment(clean).unwrap(),
            BTreeMap::from([("QWEN_METAL_LEASE_WAIT".into(), "1".into())])
        );
        for name in [
            "QWEN_DFLASH2_CAPTURE_SHIFT",
            "QWEN_DFLASH2_SELECTOR_ENABLED",
            "MTL_CAPTURE_ENABLED",
            "METAL_DEVICE_WRAPPER_TYPE",
            "GGML_METAL_NDEBUG",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_LIBRARY_PATH",
        ] {
            let variables = vec![
                (OsString::from("QWEN_METAL_LEASE_WAIT"), OsString::from("1")),
                (OsString::from(name), OsString::from("1")),
            ];
            assert!(behavior_environment(variables).is_err(), "accepted {name}");
        }
    }

    #[test]
    fn behavior_environment_rejects_malformed_unicode() {
        use std::os::unix::ffi::OsStringExt;
        let variables = vec![
            (OsString::from("QWEN_METAL_LEASE_WAIT"), OsString::from("1")),
            (OsString::from_vec(vec![0xff]), OsString::from("x")),
        ];
        assert!(behavior_environment(variables).is_err());
    }

    #[test]
    fn plain_observer_guard_restores_baseline_on_unwind() {
        assert_eq!(diagnostics_observer_active_counts(), [0, 0, 0]);
        let unwind = std::panic::catch_unwind(|| {
            let _guard = PlainObserverGuard::begin().unwrap();
            panic!("synthetic unwind");
        });
        assert!(unwind.is_err());
        assert_eq!(diagnostics_observer_active_counts(), [0, 0, 0]);
        let source = include_str!("dflash_k0s.rs");
        let plain = source
            .split("fn external_plain_draft(")
            .nth(1)
            .unwrap()
            .split("fn predicate_matches(")
            .next()
            .unwrap();
        assert_eq!(plain.matches("decoder.draft_block(").count(), 1);
        assert!(plain.find("observer.finish()").unwrap() < plain.find("result?").unwrap());
    }

    #[test]
    fn inventory_metallib_claim_binds_compiled_context_free_bytes() {
        let identity = dflash_k0s_embedded_metallib_identity();
        let mut claim = FileClaim {
            path: "/synthetic/metallib".into(),
            bytes: identity.byte_count as u64,
            sha256: hex(&identity.sha256),
            max_bytes: identity.byte_count as u64,
        };
        validate_external_metallib_claim(&claim).unwrap();
        claim.sha256.replace_range(0..1, "0");
        if identity.sha256[0] >> 4 == 0 {
            claim.sha256.replace_range(0..1, "1");
        }
        assert!(validate_external_metallib_claim(&claim).is_err());
    }

    #[test]
    fn continuation_source_is_observed_and_finished_without_extraction() {
        let source = include_str!("dflash_k0s.rs");
        let arm = source
            .split("fn run_arm(")
            .nth(1)
            .unwrap()
            .split("fn projection_parts(")
            .next()
            .unwrap();
        assert_eq!(arm.matches("draft_block_with_k0s_observation(").count(), 2);
        assert_eq!(
            arm.matches("finish_k0s_observation_without_extraction(")
                .count(),
            1
        );
        assert_eq!(arm.matches("extract_k0s_observation(").count(), 1);
        assert_eq!(arm.matches("external_plain_draft(").count(), 1);
        let live = arm
            .split("let continuation_observation")
            .nth(1)
            .unwrap()
            .split("finish_k0s_observation_without_extraction")
            .next()
            .unwrap();
        assert!(!live.contains(".draft_block("));
        assert!(arm.contains("mm: &MetalModel"));
        assert!(arm.contains("mhead: &MetalDFlashHead"));
        assert!(!arm.contains("mm: &mut MetalModel"));
        assert!(!arm.contains("mhead: &mut MetalDFlashHead"));
        assert_eq!(arm.matches("MetalSession::fresh(").count(), 1);
        assert_eq!(arm.matches("MetalDFlashSession::fresh(").count(), 1);
        assert_eq!(arm.matches("fresh_prefill(").count(), 1);
        assert!(!arm.contains("warmup"));
        assert!(!arm.contains("probe"));
    }

    #[test]
    fn event_and_preparation_binding_key_order_is_schema_exact() {
        let event =
            bind_event("a", "off-A", "s", "first", "plain", None, None, None, None).unwrap();
        let value = serde_json::to_value(event).unwrap();
        assert_eq!(
            value.as_object().unwrap().keys().collect::<Vec<_>>(),
            [
                "kind",
                "library_sequence",
                "library_event_envelope_sha256",
                "session_binding_sha256",
                "draft_tokens",
                "wrapper_binding_sha256",
            ]
        );
        let claim = FileClaim {
            path: "/tmp/a".into(),
            bytes: 1,
            sha256: "0".repeat(64),
            max_bytes: 1,
        };
        let binding = PreparationBinding {
            inventory: claim.clone(),
            inventory_spec: claim.clone(),
            preparation_spec: claim,
            seal_path: "/tmp/seal".into(),
        };
        let value = serde_json::to_value(binding).unwrap();
        assert_eq!(
            value.as_object().unwrap().keys().collect::<Vec<_>>(),
            [
                "inventory",
                "inventory_spec",
                "preparation_spec",
                "seal_path"
            ]
        );
        let predicate = SelectorDispatchPredicate {
            tag: "tag".into(),
            kernel: "kernel".into(),
            weight_dtype: "Q4_K".into(),
            input_dtype: "F32".into(),
            output_dtype: "F32".into(),
            weight_dtype_id: GgmlType::Q4_K as i32,
            input_dtype_id: GgmlType::F32 as i32,
            output_dtype_id: GgmlType::F32 as i32,
            n: 8,
            h: 5120,
            r: 256,
            grid: [1; 3],
            threads: [1; 3],
            metal_source_sha256: "1".repeat(64),
            metallib_sha256: "2".repeat(64),
            build_source_sha256: "3".repeat(64),
            allowed_environment: BTreeMap::from([("QWEN_METAL_LEASE_WAIT".into(), "1".into())]),
        };
        let value = serde_json::to_value(predicate).unwrap();
        assert_eq!(
            value.as_object().unwrap().keys().collect::<Vec<_>>(),
            [
                "tag",
                "kernel",
                "weight_dtype",
                "input_dtype",
                "output_dtype",
                "weight_dtype_id",
                "input_dtype_id",
                "output_dtype_id",
                "n",
                "h",
                "r",
                "grid",
                "threads",
                "metal_source_sha256",
                "metallib_sha256",
                "build_source_sha256",
                "allowed_environment",
            ]
        );
    }

    #[test]
    fn post_four_arm_failure_requires_null_failed_arm() {
        let all = ["off-A", "on-A", "on-B", "off-B"];
        assert!(validate_failure_names(&all, None, "comparison").is_ok());
        assert!(validate_failure_names(&all, Some("off-B"), "comparison").is_err());
        assert!(validate_failure_names(&all[..2], Some("on-B"), "continuation").is_ok());
        assert!(validate_failure_names(&all[..2], None, "continuation").is_err());
    }

    #[test]
    fn pure_four_arm_mock_gate_passes_and_selects_on_a() {
        let arms = mock_arms(true);
        assert_eq!(
            validate_four_arm_gate(&arms, "mock-attempt", true).unwrap(),
            "on-A"
        );
        assert_eq!(
            validate_four_arm_gate(&mock_arms(false), "mock-attempt", false).unwrap(),
            "on-A"
        );
        validate_capture_presence(&[false, true, true, false]).unwrap();
    }

    #[test]
    fn pure_four_arm_mock_gate_rejects_every_comparison_and_binding_mutation() {
        let baseline = mock_arms(true);
        let rejects = |mut arms: Vec<ArmJson>, mutate: fn(&mut Vec<ArmJson>)| {
            mutate(&mut arms);
            assert!(validate_four_arm_gate(&arms, "mock-attempt", true).is_err());
        };
        rejects(baseline.clone(), |arms| {
            arms[3].summary.first.target_sha256 = "f".repeat(64)
        });
        rejects(baseline.clone(), |arms| {
            arms[3].summary.continuation.state_sha256 = "f".repeat(64)
        });
        rejects(baseline.clone(), |arms| {
            arms[2].summary.observer_baseline.after_sha256 = "f".repeat(64)
        });
        rejects(baseline.clone(), |arms| {
            arms[1].summary.common_production_content_sha256 = "f".repeat(64)
        });
        rejects(baseline.clone(), |arms| arms.swap(0, 1));
        rejects(baseline.clone(), |arms| arms[0].diagnostic = true);
        rejects(baseline.clone(), |arms| {
            arms[3].session_id = arms[0].session_id.clone()
        });
        rejects(baseline.clone(), |arms| {
            arms[2].first_event.library_sequence = arms[1].first_event.library_sequence
        });
        rejects(baseline.clone(), |arms| {
            arms[2].first_event.library_event_envelope_sha256 =
                arms[1].first_event.library_event_envelope_sha256.clone()
        });
        rejects(baseline.clone(), |arms| {
            arms[1].arm_envelope_sha256 = "f".repeat(64)
        });
        rejects(baseline.clone(), |arms| {
            arms[2].rng_domains[0].after_counter = 1
        });
        rejects(baseline.clone(), |arms| {
            arms[0].summary.capture_content_sha256 = Some("c".repeat(64))
        });
        rejects(baseline.clone(), |arms| {
            arms[1].capture_projection_sha256 = Some("e".repeat(64))
        });
        rejects(baseline.clone(), |arms| {
            arms[2].summary.capture_content_sha256 = None
        });
        for index in 0..4 {
            let mut presence = [false, true, true, false];
            presence[index] = !presence[index];
            assert!(validate_capture_presence(&presence).is_err());
        }
        let incomplete = mock_arms(false);
        assert!(validate_four_arm_gate(&incomplete, "mock-attempt", true).is_err());
        assert!(validate_four_arm_gate(&baseline, "mock-attempt", false).is_err());
    }

    #[test]
    fn pure_failure_gate_covers_every_arm_and_stage() {
        let arms = mock_arms(true);
        for stage in [
            "session_setup",
            "prompt_prefill",
            "first_block",
            "extraction",
            "continuation",
            "observer_cleanup",
        ] {
            for failed_index in 0..4 {
                let decision = failure_disposition(
                    &arms[..failed_index],
                    Some(arms[failed_index].name),
                    stage,
                )
                .unwrap();
                assert_eq!(decision.failed_arm, Some(arms[failed_index].name));
                assert_eq!(decision.failed_stage, stage);
                assert!(
                    validate_failure_shape(
                        &arms[..failed_index],
                        Some(arms[(failed_index + 1) % 4].name),
                        stage,
                    )
                    .is_err()
                );
            }
        }
        for stage in ["comparison", "serialization"] {
            let decision = failure_disposition(&arms, None, stage).unwrap();
            assert_eq!(decision.failed_arm, None);
            assert_eq!(decision.failed_stage, stage);
            assert!(validate_failure_shape(&arms, Some("off-B"), stage).is_err());
            assert!(validate_failure_shape(&arms[..3], None, stage).is_err());
        }
        assert_eq!(
            classify_arm_failure("observer cleanup failed"),
            "observer_cleanup"
        );
    }

    #[test]
    fn parity_failure_writer_is_single_terminal_record_with_empty_sidecar() {
        let root = temp_dir();
        let sidecar_path = root.join("sidecar");
        let trace_path = root.join("trace");
        let mut outputs = reserve_outputs(&sidecar_path, &trace_path).unwrap();
        let manifest = mock_manifest();
        let arms = mock_arms(true);
        write_parity_failure(
            &mut outputs,
            &manifest,
            arms.clone(),
            None,
            "comparison",
            &anyhow!("mock mismatch"),
        )
        .unwrap();
        assert_eq!(outputs.sidecar.metadata().unwrap().len(), 0);
        let first = std::fs::read(&trace_path).unwrap();
        assert_eq!(first.iter().filter(|byte| **byte == b'\n').count(), 1);
        let record: Value = serde_json::from_slice(&first).unwrap();
        assert_eq!(record["event"], "parity_failure");
        assert_eq!(record["payload"]["failed_arm"], Value::Null);
        assert_eq!(
            record["payload"]["completed_arms"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
        assert!(
            write_parity_failure(
                &mut outputs,
                &manifest,
                arms,
                None,
                "comparison",
                &anyhow!("retry"),
            )
            .is_err()
        );
        assert_eq!(std::fs::read(&trace_path).unwrap(), first);
        assert_eq!(outputs.sidecar.metadata().unwrap().len(), 0);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn success_artifact_writes_follow_comparison_projection_and_custody() {
        let source = include_str!("dflash_k0s.rs");
        let run = source.split("pub fn run(args:").nth(1).unwrap();
        let sidecar_write = run
            .find("outputs.sidecar.write_all(&layout.bytes)")
            .unwrap();
        let trace_write = run.find("outputs.trace.write_all(row)").unwrap();
        for marker in [
            "validate_four_arm_gate(",
            "projection_parts(",
            "let custody = (|| -> Result<()> {",
            "records.len() == RECORDS",
        ] {
            let position = run.find(marker).unwrap();
            assert!(
                position < sidecar_write && position < trace_write,
                "{marker}"
            );
        }
        let before_success_writes = &run[..sidecar_write];
        assert!(!before_success_writes.contains("outputs.trace.write_all"));
        assert!(!before_success_writes.contains("outputs.sidecar.write_all"));
    }

    #[test]
    fn issue_mapping_partitions_slots_and_row_issue() {
        let issues = vec![
            DFlashK0sSlotIssue::DuplicateId {
                candidate_slot: 2,
                first_slot: 0,
                token_id: 7,
            },
            DFlashK0sSlotIssue::Sentinel {
                candidate_slot: 2,
                token_id: -1,
            },
            DFlashK0sSlotIssue::NoValidChoice,
        ];
        let (slots, rows) = split_issues(&issues, 16).unwrap();
        assert_eq!(
            slots[2],
            [
                SlotIssueJson::DuplicateId {
                    slot: 2,
                    token: 7,
                    first_slot: 0
                },
                SlotIssueJson::Sentinel { slot: 2, token: -1 }
            ]
        );
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn global_row_formula_covers_zero_through_ninety_six() {
        let rows = (1..=7)
            .flat_map(|depth| {
                (0..if depth == 1 { 1 } else { 16 }).map(move |slot| {
                    if depth == 1 {
                        0
                    } else {
                        1 + (depth - 2) * 16 + slot
                    }
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(rows, (0..97).collect::<Vec<_>>());
    }

    #[test]
    fn forbidden_schema_words_are_absent_from_serialized_key_source() {
        let source = include_str!("dflash_k0s.rs");
        for forbidden in [
            ["target", "_logits:"].concat(),
            ["target", "_distribution:"].concat(),
            ["proposal", "_rng:"].concat(),
            ["proposal", "_temperature:"].concat(),
            ["projection", "_distance:"].concat(),
            ["trace", "_sha256:"].concat(),
        ] {
            assert!(!source.contains(&forbidden), "forbidden key {forbidden}");
        }
    }

    #[test]
    fn paths_reject_symlinks_aliases_and_overwrite() {
        use std::os::unix::fs::symlink;
        let root = temp_dir();
        let input = root.join("input");
        std::fs::write(&input, b"x").unwrap();
        let link = root.join("link");
        symlink(&input, &link).unwrap();
        assert!(canonical_input(&link, "link").is_err());
        let hardlink = root.join("hardlink");
        std::fs::hard_link(&input, &hardlink).unwrap();
        assert!(OpenedInput::open(&input, "hard-linked input", None).is_err());
        std::fs::remove_file(&hardlink).unwrap();
        let opened_a = OpenedInput::open(&input, "input a", None).unwrap();
        let opened_b = OpenedInput::open(&input, "input b", None).unwrap();
        let sidecar_path = root.join("sidecar");
        let trace_path = root.join("trace");
        let outputs = reserve_outputs(&sidecar_path, &trace_path).unwrap();
        assert!(reject_input_aliases(&[("a", &opened_a), ("b", &opened_b)], &outputs).is_err());
        let output = root.join("out");
        let first = create_new_nofollow(&output, "out").unwrap();
        drop(first);
        assert!(create_new_nofollow(&output, "out").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sidecar_ranges_are_contiguous_unique_and_capped() {
        let mut layout = SidecarLayout {
            bytes: Vec::new(),
            ranges: Vec::new(),
            ids: HashMap::new(),
            full_logits: Vec::new(),
        };
        append_range(
            &mut layout,
            "a".into(),
            "predecessor_row",
            "F32".into(),
            vec![1],
            &[1, 2, 3, 4],
            Some("predecessor"),
            Some(0),
        )
        .unwrap();
        append_range(
            &mut layout,
            "b".into(),
            "successor_row",
            "F32".into(),
            vec![1],
            &[5, 6, 7, 8],
            Some("successor"),
            Some(1),
        )
        .unwrap();
        assert_eq!(layout.ranges[0].offset, 0);
        assert_eq!(layout.ranges[1].offset, 4);
        assert_eq!(layout.bytes.len(), 8);
        assert!(
            append_range(
                &mut layout,
                "a".into(),
                "successor_row",
                "F32".into(),
                vec![1],
                &[0; 4],
                Some("successor"),
                Some(2)
            )
            .is_err()
        );
    }

    #[test]
    fn synthetic_capture_sidecar_has_complete_contiguous_occurrences() {
        let capture = synthetic_capture();
        assert_eq!(
            capture
                .lattice
                .iter()
                .map(|row| row.row_index)
                .collect::<Vec<_>>(),
            (0..97).collect::<Vec<_>>()
        );
        assert_eq!(capture.production_chain.tokens, vec![15; 7]);
        let layout = build_sidecar(&capture, &synthetic_tensors()).unwrap();
        assert_eq!(layout.ranges.len(), 7 + 97 * 17);
        assert_eq!(
            layout.ranges.last().unwrap().offset + layout.ranges.last().unwrap().bytes,
            layout.bytes.len() as u64
        );
        assert!(
            layout
                .ranges
                .windows(2)
                .all(|pair| pair[0].offset + pair[0].bytes == pair[1].offset)
        );
        assert!(layout.bytes.len() as u64 <= MAX_SIDECAR);
    }

    #[test]
    fn record_field_order_and_count_contract_are_fixed() {
        let row = serialize_record(
            "run",
            "attempt-1",
            "end",
            EndPayload {
                producer_status: "complete",
                authority: AUTHORITY,
            },
        )
        .unwrap();
        let value: Value = serde_json::from_slice(&row).unwrap();
        assert_eq!(
            value.as_object().unwrap().keys().collect::<Vec<_>>(),
            [
                "schema",
                "schema_version",
                "run_id",
                "attempt_id",
                "event",
                "payload",
            ]
        );
        assert_eq!(RECORDS, 1 + 7 * 2 + 1);
    }

    #[test]
    fn static_and_dynamic_manifest_fields_are_separated() {
        let manifest_keys = [
            "attempt_id",
            "expected_request",
            "expected_prompt",
            "expected_binding",
            "expected_continuation_carry_token",
            "expected_rng_domains",
            "expected_fixed_chains",
            "expected_capture_context",
            "expected_build",
            "expected_host",
            "embedded_metallib_sha256",
            "selector_dispatch_predicate",
        ];
        for forbidden in [
            "synchronized_capture_sha256",
            "draft_tokens_sha256_i32le",
            "diagnostic_state_sha256",
            "sidecar",
        ] {
            assert!(!manifest_keys.contains(&forbidden));
        }
    }
}
