//! Feature-gated, single-block DFlash K0-S evidence producer.

use anyhow::{Context, Result, anyhow, ensure};
use clap::Parser;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLGPUFamily};
use qwen_llm::{
    gguf::GgufFile,
    loader::{Model, open_dflash_drafter},
    metal::{MetalContext, MetalTensor},
    metal_dflash::{
        DFLASH_K0S_BLOCK_SIZE, DFLASH_K0S_HIDDEN, DFLASH_K0S_LATTICE_ROWS, DFLASH_K0S_RANK,
        DFLASH_K0S_TOP_K, DFLASH_K0S_VOCAB, DFlashDecoder, DFlashK0sCapture, DFlashK0sChain,
        DFlashK0sChainEvent, DFlashK0sCodebookSide, DFlashK0sDispatchCensusRow,
        DFlashK0sNonFiniteClass, DFlashK0sRawRow, DFlashK0sSlotIssue, DFlashK0sTopKIssue,
        MetalDFlashHead, MetalDFlashLayerMajorScratch, MetalDFlashSession,
        dflash_k0s_scalar_contract_fixture, dflash_k0s_traverse_slots,
        prefill_tokens_with_multi_hidden,
    },
    metal_forward::{MetalForward, MetalModel, MetalSession},
    tokenizer::Tokenizer,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions},
    io::{Read, Seek, Write},
    os::unix::{
        ffi::OsStrExt,
        fs::{MetadataExt, OpenOptionsExt},
    },
    path::{Component, Path, PathBuf},
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
const REQUIRED_SOURCE_ROLES: [&str; 10] = [
    "metal_dflash_rs",
    "bench_rs",
    "dflash_k0s_rs",
    "qwen_llm_cargo_toml",
    "qwen_cli_cargo_toml",
    "metal_rs",
    "metal_forward_rs",
    "dflash2_metal",
    "mat_mat_mma8_metal",
    "build_rs",
];

#[derive(Parser, Debug)]
pub struct DflashK0sArgs {
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
    expected_binding: Binding,
    expected_fixed_chains: Vec<StaticChain>,
    expected_capture_context: CaptureContext,
    expected_build: BuildClaim,
    expected_host: HostClaim,
    embedded_metallib_sha256: String,
    expected_selector_dispatch: DispatchClaim,
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
            "expected_binding",
            "expected_fixed_chains",
            "expected_capture_context",
            "expected_build",
            "expected_host",
            "embedded_metallib_sha256",
            "expected_selector_dispatch",
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
        &value["expected_selector_dispatch"],
        &[
            "family",
            "tag",
            "encoder_ordinal",
            "encoder_concurrent",
            "kernel",
            "grid",
            "threads",
            "grid_threadgroups",
            "threadgroup_threads",
        ],
        "selector dispatch",
    )?;
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
        manifest.expected_selector_dispatch.tag.as_deref()
            == Some("dflash_k0s.selector_hidden_projection.v1"),
        "selector dispatch tag invalid"
    );
    ensure!(
        valid_sha(&manifest.embedded_metallib_sha256),
        "metallib digest invalid"
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
        ])
    {
        ensure!(
            valid_sha(&claim.sha256) && claim.max_bytes > 0 && claim.bytes <= claim.max_bytes,
            "file claim invalid"
        );
    }
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
    model: &'a Path,
    drafter: &'a Path,
    prompt: &'a str,
    carry_token: i32,
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
        command.argv.len() == 24 + 2 * bindings.fixed_chains.len(),
        "command argv length mismatch"
    );
    let temperature_text = command
        .argv
        .get(19)
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
    take("--model", "model flag")?;
    take(&bindings.model.to_string_lossy(), "target path")?;
    take("--drafter", "drafter flag")?;
    take(&bindings.drafter.to_string_lossy(), "drafter path")?;
    take("--prompt", "prompt flag")?;
    take(bindings.prompt, "prompt")?;
    take("--carry-token", "carry flag")?;
    take(&bindings.carry_token.to_string(), "carry value")?;
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

fn dtype_name(dtype: qwen_llm::tensor::GgmlType) -> String {
    dtype.wire_name().to_owned()
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
        ensure!(
            descriptor.shape == claim.shape
                && dtype_name(descriptor.dtype) == claim.dtype
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
#[derive(Serialize)]
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
    geometry: Geometry,
    request: RequestClaim,
    proposal_abstention: Abstention,
    ignored_target_policy: IgnoredPolicies,
    binding: Binding,
    provenance: Provenance,
    capture: CaptureJson,
    identities: Identities,
    semantic_references: SemanticReferences,
    tensors: Vec<TensorClaim>,
    sidecar_registry: Vec<SidecarRange>,
    production_chain: ChainJson,
    fixed_chains: Vec<ChainJson>,
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

fn build_sidecar(capture: &DFlashK0sCapture, tensors: &[TensorClaim]) -> Result<SidecarLayout> {
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
        let id = format!("full-logits-d{:02}", depth.depth);
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
            let id = format!("predecessor-r{:03}", row.row_index);
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
                    "successor-r{:03}-s{:02}",
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
                    && dtype_name(value.descriptor.dtype) == claim.dtype
                    && value.descriptor.data_offset == claim.offset
                    && value.descriptor.n_bytes == claim.bytes
                    && hex(&value.full_tensor_sha256) == claim.sha256
            })
    })
}

fn serialize_record<T: Serialize>(
    run_id: &str,
    event: &'static str,
    payload: T,
) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(&Record {
        schema: SCHEMA,
        schema_version: SCHEMA_VERSION,
        run_id: run_id.to_owned(),
        event,
        payload,
    })?;
    bytes.push(b'\n');
    Ok(bytes)
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

pub fn run(args: DflashK0sArgs, build_identity: Value) -> Result<()> {
    ensure!(
        std::env::var("QWEN_METAL_LEASE_WAIT").as_deref() == Ok("1"),
        "dflash-k0s-lattice requires literal QWEN_METAL_LEASE_WAIT=1 before Metal initialization"
    );
    ensure!(!args.prompt.is_empty(), "prompt must be nonempty");
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
    ];
    aliases.extend(
        source_inputs
            .iter()
            .map(|(role, input)| (role.as_str(), input)),
    );
    reject_input_aliases(&aliases, &outputs)?;
    let command_bytes = opened_bytes(&command_input)?;
    let command_bindings = CommandBindings {
        executable: &executable_input.path,
        model: &target_input.path,
        drafter: &drafter_input.path,
        prompt: &args.prompt,
        carry_token: args.carry_token,
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
    let capacity = prompt_ids
        .len()
        .checked_add(DFLASH_K0S_BLOCK_SIZE)
        .context("session capacity overflow")?;
    let mut target_session = MetalSession::fresh(&ctx, &mm, capacity)?;
    let mut dflash_session = MetalDFlashSession::fresh(
        &ctx,
        &mhead,
        DFLASH_K0S_HIDDEN as u64,
        DFLASH_K0S_VOCAB as u64,
        capacity,
    )?;
    let features = head
        .target_layer_ids
        .len()
        .checked_mul(DFLASH_K0S_HIDDEN)
        .context("feature count overflow")?;
    let hidden_elements = prompt_ids
        .len()
        .checked_mul(features)
        .context("prompt hidden allocation overflow")?;
    let prefill_hidden = MetalTensor::zeros_f32(&ctx, vec![u64::try_from(hidden_elements)?])?;
    let mut scratch =
        MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, head.config.block_size)?;
    let _prompt_logits = prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids,
        0,
        &mut target_session,
        &mut scratch,
        &head.target_layer_ids,
        Some(&prefill_hidden),
    )?;
    dflash_session.append_target_ctx_columns_contiguous_now(
        &ctx,
        &prefill_hidden,
        0,
        prompt_ids.len(),
        features,
    )?;
    let mut decoder = DFlashDecoder::new(&mf, &mhead, dflash_session);
    let capture = decoder.draft_block_with_k0s_diagnostic(
        args.carry_token,
        manifest.expected_capture_context.noise_start_position,
    )?;

    ensure!(
        local_capture_sha(&capture) == capture.capture_sha256,
        "local capture digest recomputation mismatch"
    );
    ensure!(
        capture.state_identity.carry_token == args.carry_token
            && capture.state_identity.noise_start_position
                == manifest.expected_capture_context.noise_start_position,
        "capture carry/position mismatch"
    );
    ensure!(
        [
            capture.state_identity.target_context_len as u64,
            capture.state_identity.context_hidden_watermark as u64,
            capture.state_identity.kv_context_watermark as u64
        ] == [
            manifest.expected_capture_context.target_context_len,
            manifest.expected_capture_context.context_hidden_watermark,
            manifest.expected_capture_context.kv_context_watermark
        ],
        "capture state differs from manifest"
    );
    ensure!(
        hex(&capture.state_identity.noise_input_sha256)
            == manifest.expected_binding.noise_input_sha256,
        "capture noise digest differs from binding"
    );
    ensure!(
        hex(&capture.provenance.embedded_metallib_sha256) == manifest.embedded_metallib_sha256,
        "embedded metallib differs from manifest"
    );
    ensure!(
        dispatch(&capture.selector_hidden_dispatch) == manifest.expected_selector_dispatch,
        "selector dispatch differs from manifest"
    );
    ensure!(
        tensor_capture_matches(&capture, &manifest.tensors),
        "capture tensor provenance differs from manifest"
    );
    ensure!(
        manifest.expected_binding.drafter_checkpoint_sha256 == drafter_input.sha256,
        "drafter checkpoint binding differs from asset"
    );
    ensure!(
        capture.draft_tokens.len() == 8
            && capture.depths.iter().all(|d| d.top_k_ids.len() == 16
                && d.unary_bits.len() == 16
                && d.selector_hidden_bits.len() == 256),
        "capture dimensions malformed"
    );
    ensure!(
        (1..=MAX_DISPATCH_ROWS).contains(&capture.dispatch_census.len()),
        "dispatch census exceeds reducer cap"
    );
    let tagged = capture
        .dispatch_census
        .iter()
        .filter(|row| row.tag.as_deref() == Some("dflash_k0s.selector_hidden_projection.v1"))
        .collect::<Vec<_>>();
    ensure!(
        tagged.len() == 1 && tagged[0] == &capture.selector_hidden_dispatch,
        "dispatch census must contain the exact selector tag once"
    );

    let layout = build_sidecar(&capture, &manifest.tensors)?;
    validate_artifact_caps(
        layout.bytes.len() as u64,
        0,
        manifest.sidecar_max_bytes,
        manifest.trace_max_bytes,
    )?;
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
        &capture.production_chain,
    );
    let fixed = chains
        .iter()
        .map(|item| {
            chain_json(
                item.name.clone(),
                item.initial_carry,
                &dflash_k0s_traverse_slots(&capture.lattice, item.initial_carry, &item.slots),
            )
        })
        .collect::<Vec<_>>();
    let capture_digest = hex(&capture.capture_sha256);
    let draft_digest = hex(&capture.state_identity.draft_tokens_sha256);
    let provenance = Provenance {
        dispatch_census: capture.dispatch_census.iter().map(dispatch).collect(),
        selector_hidden_dispatch: dispatch(&capture.selector_hidden_dispatch),
        kernel_trace: KernelTrace {
            encoders: capture.kernel_trace.encoders,
            concurrent_encoders: capture.kernel_trace.concurrent_encoders,
            dispatches: capture.kernel_trace.dispatches,
        },
        embedded_metallib_sha256: hex(&capture.provenance.embedded_metallib_sha256),
        // Compiler name/version are authenticated static manifest metadata; the
        // observable profile, feature, target, commit, source, and executable
        // facts are checked independently before this copy.
        build: manifest.expected_build.clone(),
        host: manifest.expected_host.clone(),
    };
    let encoder_ordinals = provenance
        .dispatch_census
        .iter()
        .map(|row| row.encoder_ordinal)
        .collect::<HashSet<_>>();
    let concurrent_ordinals = provenance
        .dispatch_census
        .iter()
        .filter(|row| row.encoder_concurrent)
        .map(|row| row.encoder_ordinal)
        .collect::<HashSet<_>>();
    ensure!(
        provenance.kernel_trace.dispatches as usize == provenance.dispatch_census.len()
            && provenance.kernel_trace.encoders as usize == encoder_ordinals.len()
            && provenance.kernel_trace.concurrent_encoders as usize == concurrent_ordinals.len(),
        "kernel counters/complete dispatch census mismatch"
    );
    let run_payload = RunPayload {
        authority: AUTHORITY,
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
        capture: CaptureJson {
            definition_version: CAPTURE_DEFINITION,
            noise_start_position: capture.state_identity.noise_start_position,
            carry_token: capture.state_identity.carry_token,
            synchronized_capture_sha256: capture_digest.clone(),
            draft_tokens: capture.draft_tokens.clone(),
            draft_token_bits: capture.draft_token_bits.iter().copied().map(bits).collect(),
            draft_tokens_sha256_i32le: draft_digest.clone(),
            state: CaptureState {
                target_context_len: capture.state_identity.target_context_len,
                context_hidden_watermark: capture.state_identity.context_hidden_watermark,
                kv_context_watermark: capture.state_identity.kv_context_watermark,
                noise_input_sha256: hex(&capture.state_identity.noise_input_sha256),
                synchronized_event_sha256: hex(&capture.state_identity.synchronized_event_sha256),
                diagnostic_state_sha256: hex(&capture.state_identity.diagnostic_state_sha256),
            },
        },
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
    records.push(serialize_record(&manifest.run_id, "run", run_payload)?);
    for depth in 1..=7 {
        let value = &capture.depths[depth - 1];
        records.push(serialize_record(
            &manifest.run_id,
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
            },
        )?);
        records.push(serialize_record(
            &manifest.run_id,
            "lattice",
            lattice_json(&capture, &layout, depth)?,
        )?);
    }
    records.push(serialize_record(
        &manifest.run_id,
        "end",
        EndPayload {
            producer_status: "complete",
            authority: AUTHORITY,
        },
    )?);
    ensure!(
        records.len() == RECORDS,
        "trace record count invariant failed"
    );
    let trace_bytes = records.iter().try_fold(0u64, |sum, row| {
        sum.checked_add(row.len() as u64)
            .context("trace size overflow")
    })?;
    validate_artifact_caps(
        layout.bytes.len() as u64,
        trace_bytes,
        manifest.sidecar_max_bytes,
        manifest.trace_max_bytes,
    )?;

    ensure!(
        target_g.revalidate_retained_shard_stamps()?.len() == 1
            && drafter_g.revalidate_retained_shard_stamps()?.len() == 1,
        "final retained GGUF source validation failed"
    );
    for (label, input) in &aliases {
        input.final_custody_check(label)?;
    }

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
            DFlashK0sDepth, DFlashK0sLatticeRow, DFlashK0sProvenance, DFlashK0sSlot,
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
        };
        capture.capture_sha256 = local_capture_sha(&capture);
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
            "--model".into(),
            "target.gguf".into(),
            "--drafter".into(),
            "drafter.gguf".into(),
            "--prompt".into(),
            "fixed prompt".into(),
            "--carry-token".into(),
            "42".into(),
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
        let chains = [static_chain("fixed:42:0,1,2,3,4,5,6").unwrap()];
        let bindings = CommandBindings {
            executable: &executable,
            model: Path::new("target.gguf"),
            drafter: Path::new("drafter.gguf"),
            prompt: "fixed prompt",
            carry_token: 42,
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
        assert_eq!(REQUIRED_SOURCE_ROLES.len(), 10);
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
            ["schema", "schema_version", "run_id", "event", "payload"]
        );
        assert_eq!(RECORDS, 1 + 7 * 2 + 1);
    }

    #[test]
    fn static_and_dynamic_manifest_fields_are_separated() {
        let manifest_keys = [
            "expected_request",
            "expected_binding",
            "expected_fixed_chains",
            "expected_capture_context",
            "expected_build",
            "expected_host",
            "embedded_metallib_sha256",
            "expected_selector_dispatch",
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
