//! Published .pt transport import: profiles, manifests, payload extraction and verification.

use super::*;

pub(super) const FULL_MANIFEST_NAME: &str = "lens.json";

pub(super) const FULL_PAYLOAD_NAME: &str = "transport.f16le";

pub(super) const PAYLOAD_BYTES: u64 = MATRIX_BYTES * (SOURCE_LAYER_COUNT as u64);

pub(super) const TRACE_FULL_BATCH_MANIFEST_NAME: &str = "manifest.json";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PublishedProfileId {
    Qwen38J,
    Qwen36NeuronpediaJ1000,
    Qwen36J,
    Qwen36R,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PublishedProfile {
    pub(super) id: PublishedProfileId,
    pub(super) method: &'static str,
    pub(super) source_repository: &'static str,
    pub(super) source_revision: &'static str,
    pub(super) source_filename: &'static str,
    pub(super) source_bytes: u64,
    pub(super) source_sha256: &'static str,
    pub(super) data_pickle_sha256: &'static str,
    pub(super) expected_payload_blake3: &'static str,
    pub(super) archive_root: &'static str,
    pub(super) target_layer: u32,
    pub(super) identity_anchor_layer: Option<u32>,
    pub(super) base_model: &'static str,
    pub(super) fitted_checkpoint: &'static str,
    pub(super) fitted_checkpoint_revision: &'static str,
    pub(super) model_name_fragment: &'static str,
    pub(super) license: &'static str,
}

pub(super) const PUBLISHED_PROFILES: [PublishedProfile; 4] = [
    PublishedProfile {
        id: PublishedProfileId::Qwen38J,
        method: "j",
        source_repository: "eyes-ml/Qwen3.8-27B_jacobian-lens",
        source_revision: "f8608c19b441f605d87ce46b80184f3774d75f2c",
        source_filename: "Qwen3.8-27B_jacobian_lens.pt",
        source_bytes: 3_303_033_664,
        source_sha256: "6b51f369e45a68b7eb775081ba5d195bb41360fb2abd15b0ab5b49881b638d49",
        data_pickle_sha256: "3e58341435e2178dc78af9689fab7dd872661063e5087848b0099f436aa7d448",
        expected_payload_blake3: "4a75d250d754d6e02f7d865bf84253a4804df3f49a7f5ccd6059c74f0f26b9e3",
        archive_root: "Qwen3.8-27B_jacobian_lens",
        target_layer: 63,
        identity_anchor_layer: None,
        base_model: "Qwen/Qwen3.8-27B",
        fitted_checkpoint: "eyes-ml/Qwen3.8-27B",
        fitted_checkpoint_revision: FITTED_CHECKPOINT_REVISION,
        model_name_fragment: "qwen3.8",
        license: "Apache-2.0",
    },
    PublishedProfile {
        id: PublishedProfileId::Qwen36NeuronpediaJ1000,
        method: "j",
        source_repository: "neuronpedia/jacobian-lens",
        source_revision: "0731326edff4ae730ffc5356fe1a4728c748b3a6",
        source_filename: "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
        source_bytes: 3_303_032_772,
        source_sha256: "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1",
        data_pickle_sha256: "3e58341435e2178dc78af9689fab7dd872661063e5087848b0099f436aa7d448",
        expected_payload_blake3: "2251a5872df8ddb53bfd72339f6f13440836f071a1bbeb8033b4f6c2be5244c7",
        archive_root: "jacobian_lens",
        target_layer: 63,
        identity_anchor_layer: None,
        base_model: "Qwen/Qwen3.6-27B",
        fitted_checkpoint: "Qwen/Qwen3.6-27B",
        fitted_checkpoint_revision: "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        model_name_fragment: "qwen3.6",
        license: "MIT",
    },
    PublishedProfile {
        id: PublishedProfileId::Qwen36J,
        method: "j",
        source_repository: "camilablank/workspace-lenses",
        source_revision: "d740106d1e0f95456dc8718fba2895e9c8ffd6ef",
        source_filename: "qwen3.6-27b/j-lens/lens.pt",
        source_bytes: 3_303_028_503,
        source_sha256: "a036b35843d389b6655df721711917436fb79c83358c1861a4d58ad103a02724",
        data_pickle_sha256: "4e0c9b7e0b2f362d3711813314073a66eb76657d84875f038d3688705a3a3f70",
        expected_payload_blake3: "a76c67d0c977696970511bbaf23baeb9e02f6a526c873069b4bb18c92317965e",
        archive_root: "lens",
        target_layer: 62,
        identity_anchor_layer: Some(62),
        base_model: "Qwen/Qwen3.6-27B",
        fitted_checkpoint: "Qwen/Qwen3.6-27B",
        fitted_checkpoint_revision: "not_recorded_in_published_artifact",
        model_name_fragment: "qwen3.6",
        license: "MIT",
    },
    PublishedProfile {
        id: PublishedProfileId::Qwen36R,
        method: "r",
        source_repository: "camilablank/workspace-lenses",
        source_revision: "d740106d1e0f95456dc8718fba2895e9c8ffd6ef",
        source_filename: "qwen3.6-27b/r-lens/lens.pt",
        source_bytes: 3_303_028_567,
        source_sha256: "fe4d0b6a17318760e67e7a5ed417fc5dbb8e66129c6944656f506b2f10ce6192",
        data_pickle_sha256: "fcb9a42587adf9069e1fe5189d88c37ff3f505c5e62825ad5ce2da764ad4e424",
        expected_payload_blake3: "0be52938f7a3b6f9e419017aa03036b70fe97e18d304193116b8263fac69f58e",
        archive_root: "lens",
        target_layer: 62,
        identity_anchor_layer: Some(62),
        base_model: "Qwen/Qwen3.6-27B",
        fitted_checkpoint: "Qwen/Qwen3.6-27B",
        fitted_checkpoint_revision: "not_recorded_in_published_artifact",
        model_name_fragment: "qwen3.6",
        license: "MIT",
    },
];

#[derive(Debug, Args)]
pub(crate) struct ImportFullArgs {
    /// One supported exact pinned published .pt transport asset.
    #[arg(long)]
    pub(super) source: PathBuf,

    /// New immutable full-lens artifact directory.
    #[arg(long)]
    pub(super) output: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FullLensManifest {
    pub(super) schema: String,
    pub(super) schema_version: u32,
    pub(super) status: String,
    pub(super) transport: FullTransport,
    pub(super) model: FullModel,
    pub(super) fit: PublishedFit,
    pub(super) source: PublishedSource,
    pub(super) payload: FullPayload,
    pub(super) transfer: TransferPolicy,
    pub(super) provenance: ImportProvenance,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FullPayload {
    pub(super) path: String,
    pub(super) dtype: String,
    pub(super) shape: [usize; 3],
    pub(super) byte_length: u64,
    pub(super) blake3: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ImportProvenance {
    pub(super) build_commit: String,
    pub(super) build_dirty: String,
    pub(super) build_source_state: String,
    pub(super) build_stamp_source: String,
    pub(super) build_stamp_error: String,
    pub(super) pickle_execution: String,
}

#[derive(Debug, Serialize)]
pub(super) struct TraceFullBatchManifest {
    pub(super) schema: &'static str,
    pub(super) schema_version: u32,
    pub(super) producer: TraceFullProducer,
    pub(super) execution_mode: &'static str,
    pub(super) requests_jsonl: PathBuf,
    pub(super) deployed_model: TraceFullModel,
    pub(super) lens: TraceFullLens,
    pub(super) selected_layers: Vec<u32>,
    pub(super) top_k: usize,
    pub(super) request_count: usize,
    pub(super) aggregate_rows: usize,
    pub(super) artifacts: Vec<TraceFullBatchArtifact>,
    pub(super) timing: TraceFullBatchTiming,
}

pub(crate) fn import_full(mut args: ImportFullArgs) -> Result<()> {
    validate_token_build_identity(
        env!("QWEN_BUILD_SOURCE_STATE"),
        env!("QWEN_BUILD_STAMP_ERROR"),
    )?;
    args.output = resolve_output_path(&args.output)?;
    let (mut source, source_length) = open_regular_file(&args.source)?;
    ensure!(
        PUBLISHED_PROFILES
            .iter()
            .any(|profile| profile.source_bytes == source_length as u64),
        "{} length {} does not match any supported pinned full-lens asset",
        args.source.display(),
        source_length,
    );
    let source_metadata = source
        .metadata()
        .with_context(|| format!("inspect opened {}", args.source.display()))?;
    let source_modified = source_metadata.modified().ok();
    let source_sha256 = hash_sha256(&mut source, &args.source)?;
    let profile = profile_for_source(source_length as u64, &source_sha256).with_context(|| {
        format!(
            "{} SHA-256 {} does not identify a supported pinned full-lens asset",
            args.source.display(),
            source_sha256
        )
    })?;

    prepare_output_directory(&args.output)?;
    let manifest_path = args.output.join(FULL_MANIFEST_NAME);
    if manifest_path.exists() {
        let manifest: FullLensManifest = read_json_file(&manifest_path)?;
        validate_manifest(&manifest)?;
        ensure!(
            profile_for_manifest(&manifest)?.id == profile.id,
            "existing output artifact was imported from a different published lens"
        );
        verify_payload(&args.output, &manifest.payload)?;
        println!("{}", serde_json::to_string_pretty(&manifest)?);
        return Ok(());
    }

    source
        .seek(SeekFrom::Start(0))
        .with_context(|| format!("rewind pinned source {}", args.source.display()))?;
    let mut archive = ZipArchive::new(source)
        .with_context(|| format!("open pinned torch ZIP {}", args.source.display()))?;
    let spec = ArchiveSpec {
        root: profile.archive_root,
        layout: ArchiveLayout::LayerStorages,
        layer_count: SOURCE_LAYER_COUNT,
        hidden_size: HIDDEN_SIZE,
        matrix_bytes: MATRIX_BYTES,
        data_pickle_sha256: profile.data_pickle_sha256,
        serialization_id: None,
        identity_layer_index: profile
            .identity_anchor_layer
            .and_then(|layer| usize::try_from(layer).ok()),
    };
    validate_archive(&mut archive, spec)?;

    let staging = staging_path(&args.output, FULL_PAYLOAD_NAME)?;
    let import = (|| -> Result<FullPayload> {
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&staging)
            .with_context(|| format!("create full-lens staging payload {}", staging.display()))?;
        let payload = extract_payload(&mut archive, spec, &mut output)?;
        output
            .sync_all()
            .with_context(|| format!("sync full-lens staging payload {}", staging.display()))?;
        drop(output);
        ensure!(
            payload.blake3 == profile.expected_payload_blake3,
            "imported transport payload BLAKE3 does not match the pinned source payload"
        );

        let source = archive.into_inner();
        let final_metadata = source
            .metadata()
            .with_context(|| format!("reinspect opened {}", args.source.display()))?;
        ensure!(
            final_metadata.len() == profile.source_bytes
                && final_metadata.modified().ok() == source_modified,
            "pinned source changed while it was being imported"
        );
        publish_streamed_payload(&staging, &args.output.join(FULL_PAYLOAD_NAME), &payload)?;
        Ok(payload)
    })();
    if import.is_err() {
        let _ = std::fs::remove_file(&staging);
    }
    let payload = import?;

    let manifest = published_manifest(profile, payload);
    validate_manifest(&manifest)?;
    publish_immutable(
        &manifest_path,
        &serialize_json_pretty_bounded(&manifest, "full lens manifest")?,
    )?;
    sync_directory(&args.output)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

pub(super) fn validate_trace_full_manifest(manifest: &FullLensManifest) -> Result<()> {
    ensure!(manifest.schema == FULL_SCHEMA, "unknown full lens schema");
    ensure!(
        manifest.schema_version == FULL_SCHEMA_VERSION,
        "unsupported full lens schema version"
    );
    ensure!(manifest.status == "complete", "full lens is not complete");
    let profile = profile_for_manifest(manifest)?;
    ensure!(
        manifest.transport == canonical_transport(profile),
        "full lens transport contract is not canonical"
    );
    ensure!(
        manifest.model == canonical_model(profile),
        "full lens model geometry is not canonical"
    );
    ensure!(
        manifest.payload.path == FULL_PAYLOAD_NAME
            && manifest.payload.dtype == "f16_le"
            && manifest.payload.shape == [SOURCE_LAYER_COUNT, HIDDEN_SIZE, HIDDEN_SIZE]
            && manifest.payload.byte_length == PAYLOAD_BYTES
            && manifest.payload.blake3 == profile.expected_payload_blake3,
        "full lens payload descriptor is not canonical"
    );
    Ok(())
}

pub(super) fn model_metadata_matches_profile(gguf: &GgufFile, profile: PublishedProfile) -> bool {
    let named = [
        gguf.get_str("general.name"),
        gguf.get_str("general.base_model.0.name"),
    ]
    .into_iter()
    .flatten()
    .any(|value| {
        let value = value.to_ascii_lowercase();
        value.contains(profile.model_name_fragment) && value.contains("27b")
    });
    named
        && gguf.get_str("tokenizer.ggml.model") == Some("gpt2")
        && gguf.get_str("tokenizer.ggml.pre") == Some("qwen35")
}

pub(super) fn profile_for_source(byte_length: u64, sha256: &str) -> Option<PublishedProfile> {
    PUBLISHED_PROFILES
        .iter()
        .copied()
        .find(|profile| profile.source_bytes == byte_length && profile.source_sha256 == sha256)
}

pub(super) fn profile_for_manifest(manifest: &FullLensManifest) -> Result<PublishedProfile> {
    PUBLISHED_PROFILES
        .iter()
        .copied()
        .find(|profile| {
            manifest.transport.method == profile.method
                && manifest.source.repository == profile.source_repository
                && manifest.source.revision == profile.source_revision
                && manifest.source.filename == profile.source_filename
                && manifest.source.byte_length == profile.source_bytes
                && manifest.source.sha256 == profile.source_sha256
                && manifest.source.data_pickle_sha256 == profile.data_pickle_sha256
        })
        .context("full lens manifest does not identify a supported pinned published asset")
}

pub(super) fn published_manifest(
    profile: PublishedProfile,
    payload: FullPayload,
) -> FullLensManifest {
    FullLensManifest {
        schema: FULL_SCHEMA.into(),
        schema_version: FULL_SCHEMA_VERSION,
        status: "complete".into(),
        transport: canonical_transport(profile),
        model: canonical_model(profile),
        fit: canonical_fit(profile),
        source: canonical_source(profile),
        payload,
        transfer: canonical_transfer_policy(profile),
        provenance: ImportProvenance {
            build_commit: env!("QWEN_BUILD_COMMIT").into(),
            build_dirty: env!("QWEN_BUILD_DIRTY").into(),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
            build_stamp_source: env!("QWEN_BUILD_STAMP_SOURCE").into(),
            build_stamp_error: env!("QWEN_BUILD_STAMP_ERROR").into(),
            pickle_execution: "none_fixed_schema_pinned_zip_entries_only".into(),
        },
    }
}

pub(super) fn canonical_transport(profile: PublishedProfile) -> FullTransport {
    FullTransport {
        method: profile.method.into(),
        target_layer: profile.target_layer,
        source_layers: (0..SOURCE_LAYER_COUNT as u32).collect(),
        capture_site: "post_block_residual".into(),
        orientation: ORIENTATION.into(),
        hidden_size: HIDDEN_SIZE as u32,
        bias: "none".into(),
        storage_dtype: "f16_le".into(),
    }
}

pub(super) fn canonical_model(profile: PublishedProfile) -> FullModel {
    FullModel {
        base_model: profile.base_model.into(),
        fitted_checkpoint: profile.fitted_checkpoint.into(),
        fitted_checkpoint_revision: profile.fitted_checkpoint_revision.into(),
        architecture: "qwen3_hybrid_dense".into(),
        n_layers: N_LAYERS,
        hidden_size: HIDDEN_SIZE as u32,
        vocab_size: VOCAB_SIZE,
        output_norm: "final_rms_norm_epsilon_1e-6".into(),
        unembedding: "untied_bias_free_lm_head".into(),
    }
}

pub(super) fn canonical_fit(profile: PublishedProfile) -> PublishedFit {
    match profile.id {
        PublishedProfileId::Qwen38J => PublishedFit {
            fitter: "neuronpedia_utils/jlens/fit_lens.py".into(),
            fitter_revision: "7724688596eb734a0662f911bf183151a5c66b2f".into(),
            dataset: "Salesforce/wikitext:wikitext-103-raw-v1".into(),
            split: "train".into(),
            n_prompts: 1_000,
            max_sequence_length: 128,
            skip_first: 16,
            valid_positions_per_prompt: Some(111),
            dim_batch: Some(8),
            model_execution_dtype: Some("bfloat16".into()),
            accumulator_dtype: Some("float32".into()),
            serialized_dtype: "float16".into(),
            docs_consumed: None,
            n_positions: None,
            config_json: None,
            weighting: None,
            corpus_mode: None,
        },
        PublishedProfileId::Qwen36NeuronpediaJ1000 => PublishedFit {
            fitter: "anthropics/jacobian-lens".into(),
            fitter_revision: "not_recorded_in_published_artifact".into(),
            dataset: "Salesforce/wikitext".into(),
            split: "not_recorded_in_published_artifact".into(),
            n_prompts: 1_000,
            max_sequence_length: 128,
            skip_first: 16,
            valid_positions_per_prompt: Some(111),
            dim_batch: None,
            model_execution_dtype: Some("bfloat16".into()),
            accumulator_dtype: Some("float32".into()),
            serialized_dtype: "float16".into(),
            docs_consumed: None,
            n_positions: None,
            config_json: None,
            weighting: None,
            corpus_mode: None,
        },
        PublishedProfileId::Qwen36J | PublishedProfileId::Qwen36R => PublishedFit {
            fitter: "jlens.fit".into(),
            fitter_revision: "modal".into(),
            dataset: "NeelNanda/pile-10k".into(),
            split: "not_recorded_in_published_artifact".into(),
            n_prompts: 25,
            max_sequence_length: 128,
            skip_first: 4,
            valid_positions_per_prompt: None,
            dim_batch: None,
            model_execution_dtype: Some("bfloat16".into()),
            accumulator_dtype: None,
            serialized_dtype: "float16".into(),
            docs_consumed: Some(25),
            n_positions: Some("0.0".into()),
            config_json: Some(match profile.id {
                PublishedProfileId::Qwen36J => r#"{"estimator": "standard"}"#.into(),
                PublishedProfileId::Qwen36R => r#"{"estimator": "relp", "rules": {"ln_rule": true, "identity_rule": true, "half_rule": true, "include_qk_norms": false}}"#.into(),
                PublishedProfileId::Qwen38J | PublishedProfileId::Qwen36NeuronpediaJ1000 => {
                    unreachable!()
                }
            }),
            weighting: Some("uniform".into()),
            corpus_mode: Some("pretrain".into()),
        },
    }
}

pub(super) fn canonical_source(profile: PublishedProfile) -> PublishedSource {
    PublishedSource {
        repository: profile.source_repository.into(),
        revision: profile.source_revision.into(),
        filename: profile.source_filename.into(),
        byte_length: profile.source_bytes,
        sha256: profile.source_sha256.into(),
        data_pickle_sha256: profile.data_pickle_sha256.into(),
        license: profile.license.into(),
    }
}

pub(super) fn canonical_transfer_policy(profile: PublishedProfile) -> TransferPolicy {
    TransferPolicy {
        fitted_weight_precision: match profile.id {
            PublishedProfileId::Qwen38J => "bfloat16",
            PublishedProfileId::Qwen36NeuronpediaJ1000
            | PublishedProfileId::Qwen36J
            | PublishedProfileId::Qwen36R => "bfloat16_model_float16_serialized_transport",
        }
        .into(),
        deployed_checkpoint_policy: "geometry_preserving_transfer_requires_validation".into(),
        validation_status: "unvalidated".into(),
    }
}

pub(super) fn validate_manifest(manifest: &FullLensManifest) -> Result<()> {
    ensure!(manifest.schema == FULL_SCHEMA, "unknown full lens schema");
    ensure!(
        manifest.schema_version == FULL_SCHEMA_VERSION,
        "unsupported full lens schema version"
    );
    ensure!(manifest.status == "complete", "full lens is not complete");
    let profile = profile_for_manifest(manifest)?;
    ensure!(
        manifest.transport == canonical_transport(profile),
        "full lens transport contract is not canonical"
    );
    ensure!(
        manifest.model == canonical_model(profile),
        "full lens model contract is not canonical"
    );
    ensure!(
        manifest.fit == canonical_fit(profile),
        "full lens fit contract is not canonical"
    );
    ensure!(
        manifest.source == canonical_source(profile),
        "full lens source provenance is not canonical"
    );
    ensure!(
        manifest.payload.path == FULL_PAYLOAD_NAME
            && manifest.payload.dtype == "f16_le"
            && manifest.payload.shape == [SOURCE_LAYER_COUNT, HIDDEN_SIZE, HIDDEN_SIZE]
            && manifest.payload.byte_length == PAYLOAD_BYTES
            && manifest.payload.blake3 == profile.expected_payload_blake3,
        "full lens payload descriptor is not canonical"
    );
    ensure!(
        manifest.transfer == canonical_transfer_policy(profile),
        "full lens transfer policy is not fail-closed"
    );
    validate_token_build_identity(
        &manifest.provenance.build_source_state,
        &manifest.provenance.build_stamp_error,
    )?;
    ensure!(
        !manifest.provenance.build_commit.is_empty()
            && matches!(manifest.provenance.build_dirty.as_str(), "0" | "1")
            && !manifest.provenance.build_stamp_source.is_empty()
            && manifest.provenance.pickle_execution == "none_fixed_schema_pinned_zip_entries_only",
        "full lens import provenance is incomplete or permits pickle execution"
    );
    Ok(())
}

pub(super) fn extract_payload<R: Read + Seek, W: Write>(
    archive: &mut ZipArchive<R>,
    spec: ArchiveSpec<'_>,
    output: &mut W,
) -> Result<FullPayload> {
    let extracted = crate::published_pt::extract_payload(archive, spec, output)?;
    Ok(FullPayload {
        path: FULL_PAYLOAD_NAME.into(),
        dtype: "f16_le".into(),
        shape: [spec.layer_count, spec.hidden_size, spec.hidden_size],
        byte_length: extracted.byte_length,
        blake3: extracted.blake3,
    })
}

pub(super) fn prepare_output_directory(output: &Path) -> Result<()> {
    if output.exists() {
        let metadata = std::fs::symlink_metadata(output)
            .with_context(|| format!("inspect output {}", output.display()))?;
        ensure!(
            metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
            "output {} must be a real directory",
            output.display()
        );
        return Ok(());
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    DirBuilder::new()
        .mode(0o700)
        .create(output)
        .with_context(|| format!("create output directory {}", output.display()))?;
    sync_directory(parent)
}

pub(super) fn staging_path(directory: &Path, name: &str) -> Result<PathBuf> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?
        .as_nanos();
    Ok(directory.join(format!(".{name}.stage.{}.{}", std::process::id(), nonce)))
}

pub(super) fn publish_streamed_payload(
    staging: &Path,
    destination: &Path,
    payload: &FullPayload,
) -> Result<()> {
    match std::fs::hard_link(staging, destination) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            verify_payload(
                destination.parent().unwrap_or_else(|| Path::new(".")),
                payload,
            )?;
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "publish staging payload {} to {}",
                    staging.display(),
                    destination.display()
                )
            });
        }
    }
    std::fs::remove_file(staging)
        .with_context(|| format!("remove staging payload {}", staging.display()))?;
    sync_directory(destination.parent().unwrap_or_else(|| Path::new(".")))
}

pub(super) fn verify_payload(directory: &Path, payload: &FullPayload) -> Result<()> {
    ensure!(
        Path::new(&payload.path).components().count() == 1,
        "full lens payload path must be one relative filename"
    );
    let path = directory.join(&payload.path);
    let (mut file, length) = open_regular_file(&path)?;
    ensure!(
        length as u64 == payload.byte_length,
        "{} length {} != expected {}",
        path.display(),
        length,
        payload.byte_length
    );
    let mut hasher = Blake3Hasher::new();
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES];
    let hidden_size = payload.shape[1];
    ensure!(
        payload.dtype == "f16_le" && hidden_size > 0 && payload.shape[2] == hidden_size,
        "full lens payload shape is invalid"
    );
    let matrix_bytes = (hidden_size as u64)
        .checked_mul(hidden_size as u64)
        .and_then(|words| words.checked_mul(2))
        .context("full lens payload matrix byte count overflow")?;
    ensure!(
        matrix_bytes.checked_mul(payload.shape[0] as u64) == Some(payload.byte_length),
        "full lens payload matrix inventory does not match its byte length"
    );
    for layer in 0..payload.shape[0] {
        let mut remaining = matrix_bytes;
        while remaining > 0 {
            let matrix_byte_offset = matrix_bytes - remaining;
            let read = usize::try_from(remaining.min(buffer.len() as u64))
                .context("full lens verification chunk")?;
            file.read_exact(&mut buffer[..read])
                .with_context(|| format!("read full lens payload {}", path.display()))?;
            hasher.update(&buffer[..read]);
            ensure_finite_f16(&buffer[..read], layer, matrix_byte_offset as usize / 2)?;
            remaining -= read as u64;
        }
    }
    let mut extra = [0u8; 1];
    ensure!(
        file.read(&mut extra)
            .with_context(|| format!("check full lens payload end {}", path.display()))?
            == 0,
        "full lens payload contains trailing bytes"
    );
    ensure!(
        hasher.finalize().to_hex().as_str() == payload.blake3,
        "full lens payload BLAKE3 mismatch"
    );
    Ok(())
}
