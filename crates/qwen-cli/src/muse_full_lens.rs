use super::muse_full_lens_artifact as artifact;
use super::muse_lens_rows_artifact as rows;
use anyhow::{Context, Result, ensure};
use blake3::Hasher;
use clap::Args;
use half::f16;
use std::fs::{DirBuilder, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

#[derive(Debug, Args)]
pub(crate) struct AssembleMuseFullArgs {
    /// Directory whose immediate child directories are completed Muse row shards.
    #[arg(long)]
    shards_root: PathBuf,

    /// New assembled Muse full-transport directory, or an incomplete one with --resume.
    #[arg(long)]
    output: PathBuf,

    /// Reopen a complete artifact or restart an interrupted assembly.
    #[arg(long)]
    resume: bool,
}

struct ShardInput {
    directory: PathBuf,
    manifest: rows::Manifest,
}

pub(crate) fn assemble(mut args: AssembleMuseFullArgs) -> Result<()> {
    super::validate_token_build_identity(
        env!("QWEN_BUILD_SOURCE_STATE"),
        env!("QWEN_BUILD_STAMP_ERROR"),
    )?;
    let shards_root = canonical_real_directory(&args.shards_root, "Muse row-shard root")?;
    args.output = super::resolve_output_path(&args.output)?;
    ensure!(
        args.output != shards_root,
        "Muse full-transport output must differ from its shard root"
    );
    let shards = discover_shards(&shards_root)?;
    let first = shards.first().context("Muse row-shard root is empty")?;
    let config = artifact::config_from_row(&first.manifest.config);
    artifact::validate_config(&config)?;
    validate_shard_set(&shards, &config)?;
    let assembly = assembly_descriptor(&shards, config.geometry.hidden_size);
    let corpus = first.manifest.corpus.clone();
    let replay = aggregate_replay(&shards)?;
    let config_blake3 = super::digest_json(&config)?;

    if let Some(manifest) = open_output(
        &args.output,
        args.resume,
        &config,
        &corpus,
        &assembly,
        &config_blake3,
    )? {
        println!("{}", serde_json::to_string_pretty(&manifest)?);
        return Ok(());
    }

    let partial_path = args.output.join(artifact::PARTIAL_PAYLOAD_NAME);
    let payload_path = args.output.join(artifact::PAYLOAD_NAME);
    remove_regular_if_present(&partial_path)?;
    remove_regular_if_present(&payload_path)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&partial_path)
        .with_context(|| {
            format!(
                "create Muse transport staging file {}",
                partial_path.display()
            )
        })?;

    let hidden = config.geometry.hidden_size as usize;
    let mut shard_hashers = (0..shards.len()).map(|_| Hasher::new()).collect::<Vec<_>>();
    let mut payload_hasher = Hasher::new();
    let matrix_bytes = artifact::matrix_bytes(&config)?;
    let mut matrices = Vec::with_capacity(config.source_layers.len());
    let mut f32_bytes = Vec::new();
    let mut f16_bytes = Vec::new();
    let mut written = 0u64;

    for (source_slot, &source_layer) in config.source_layers.iter().enumerate() {
        eprintln!(
            "assemble Muse transport source {}/{} layer={source_layer}",
            source_slot + 1,
            config.source_layers.len()
        );
        let matrix_offset = written;
        let mut matrix_hasher = Hasher::new();
        for (shard_slot, shard) in shards.iter().enumerate() {
            let row_count =
                usize::try_from(shard.manifest.config.row_end - shard.manifest.config.row_start)
                    .context("Muse input shard row count")?;
            let source_slice_bytes = row_count
                .checked_mul(hidden)
                .and_then(|values| values.checked_mul(4))
                .context("Muse input source slice byte count overflow")?;
            let source_offset = source_slot
                .checked_mul(source_slice_bytes)
                .and_then(|bytes| u64::try_from(bytes).ok())
                .context("Muse input source slice offset overflow")?;
            let payload_path = shard.directory.join(&shard.manifest.payload.path);
            let (mut input, input_length) = super::open_regular_file(&payload_path)?;
            ensure!(
                input_length as u64 == shard.manifest.payload.byte_length,
                "Muse input payload {} length changed",
                payload_path.display()
            );
            input
                .seek(SeekFrom::Start(source_offset))
                .with_context(|| format!("seek Muse input payload {}", payload_path.display()))?;
            f32_bytes.resize(source_slice_bytes, 0);
            input
                .read_exact(&mut f32_bytes)
                .with_context(|| format!("read Muse input payload {}", payload_path.display()))?;
            shard_hashers[shard_slot].update(&f32_bytes);
            convert_f32_slice_to_f16(&f32_bytes, &mut f16_bytes)?;
            output
                .write_all(&f16_bytes)
                .context("write assembled Muse transport")?;
            matrix_hasher.update(&f16_bytes);
            payload_hasher.update(&f16_bytes);
            written = written
                .checked_add(u64::try_from(f16_bytes.len()).context("Muse F16 slice length")?)
                .context("Muse assembled payload length overflow")?;
        }
        ensure!(
            written - matrix_offset == matrix_bytes,
            "assembled Muse source matrix has the wrong byte length"
        );
        matrices.push(artifact::MatrixDescriptor {
            source_layer,
            byte_offset: matrix_offset,
            byte_length: matrix_bytes,
            blake3: matrix_hasher.finalize().to_hex().to_string(),
        });
    }

    for (shard, hasher) in shards.iter().zip(shard_hashers) {
        ensure!(
            hasher.finalize().to_hex().as_str() == shard.manifest.payload.blake3,
            "Muse input row-shard payload digest mismatch for rows {}..{}",
            shard.manifest.config.row_start,
            shard.manifest.config.row_end
        );
    }
    let expected_bytes = matrix_bytes
        .checked_mul(u64::try_from(config.source_layers.len()).context("Muse source count")?)
        .context("Muse assembled payload size overflow")?;
    ensure!(
        written == expected_bytes,
        "assembled Muse payload length {written} differs from expected {expected_bytes}"
    );
    output.sync_all().context("sync assembled Muse transport")?;
    drop(output);
    std::fs::rename(&partial_path, &payload_path).with_context(|| {
        format!(
            "publish Muse transport {} to {}",
            partial_path.display(),
            payload_path.display()
        )
    })?;
    super::sync_directory(&args.output)?;

    let manifest = artifact::Manifest {
        schema: artifact::SCHEMA.into(),
        schema_version: artifact::SCHEMA_VERSION,
        status: "complete".into(),
        config_blake3,
        config,
        corpus,
        assembly,
        payload: artifact::Payload {
            path: artifact::PAYLOAD_NAME.into(),
            dtype: "f16_le".into(),
            shape: [config_source_count(&shards)?, hidden, hidden],
            byte_length: written,
            blake3: payload_hasher.finalize().to_hex().to_string(),
            matrices,
        },
        replay,
        provenance: artifact::Provenance {
            build_commit: env!("QWEN_BUILD_COMMIT").into(),
            build_dirty: env!("QWEN_BUILD_DIRTY").into(),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
            build_stamp_source: env!("QWEN_BUILD_STAMP_SOURCE").into(),
            build_stamp_error: env!("QWEN_BUILD_STAMP_ERROR").into(),
        },
    };
    artifact::validate_manifest(&manifest)?;
    super::publish_immutable(
        &args.output.join(artifact::MANIFEST_NAME),
        &super::serialize_json_pretty_bounded(&manifest, "Muse full-transport manifest")?,
    )?;
    super::sync_directory(&args.output)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn canonical_real_directory(path: &Path, label: &str) -> Result<PathBuf> {
    let lexical = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect {label} {}", path.display()))?;
    ensure!(
        lexical.file_type().is_dir() && !lexical.file_type().is_symlink(),
        "{label} {} must be a real directory, not a symlink",
        path.display()
    );
    let canonical = std::fs::canonicalize(path)
        .with_context(|| format!("resolve {label} {}", path.display()))?;
    let metadata = std::fs::symlink_metadata(&canonical)
        .with_context(|| format!("inspect {label} {}", canonical.display()))?;
    ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "{label} {} must be a real directory",
        canonical.display()
    );
    Ok(canonical)
}

fn discover_shards(root: &Path) -> Result<Vec<ShardInput>> {
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(root)
        .with_context(|| format!("read Muse row-shard root {}", root.display()))?
    {
        let entry = entry.with_context(|| format!("read entry in {}", root.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("inspect Muse shard candidate {}", entry.path().display()))?;
        ensure!(
            !file_type.is_symlink(),
            "Muse shard root contains symlink {}",
            entry.path().display()
        );
        if !file_type.is_dir() {
            continue;
        }
        let manifest_path = entry.path().join(rows::MANIFEST_NAME);
        if !manifest_path.exists() {
            continue;
        }
        let manifest: rows::Manifest = super::read_json_file(&manifest_path)?;
        rows::validate_stored(&manifest)
            .with_context(|| format!("validate Muse row shard {}", entry.path().display()))?;
        shards.push(ShardInput {
            directory: entry.path(),
            manifest,
        });
    }
    shards.sort_by_key(|shard| shard.manifest.config.row_start);
    ensure!(
        !shards.is_empty(),
        "Muse row-shard root has no complete shards"
    );
    Ok(shards)
}

fn validate_shard_set(shards: &[ShardInput], config: &artifact::Config) -> Result<()> {
    let first = shards.first().context("Muse row-shard set is empty")?;
    let mut cursor = 0u32;
    for shard in shards {
        ensure!(
            artifact::config_from_row(&shard.manifest.config) == *config,
            "Muse row shard {} has incompatible fit configuration",
            shard.directory.display()
        );
        ensure!(
            shard.manifest.corpus == first.manifest.corpus,
            "Muse row shard {} has incompatible corpus summary",
            shard.directory.display()
        );
        ensure!(
            shard.manifest.config.row_start == cursor,
            "Muse row-shard coverage has a gap or overlap at row {cursor}"
        );
        cursor = shard.manifest.config.row_end;
    }
    ensure!(
        cursor == config.geometry.hidden_size,
        "Muse row-shard coverage ends at {cursor}, expected {}",
        config.geometry.hidden_size
    );
    Ok(())
}

fn assembly_descriptor(shards: &[ShardInput], hidden_size: u32) -> artifact::Assembly {
    artifact::Assembly {
        input_schema: rows::SCHEMA.into(),
        input_schema_version: rows::SCHEMA_VERSION,
        input_dtype: "f32_le".into(),
        conversion: artifact::CONVERSION.into(),
        row_coverage: [0, hidden_size],
        shards: shards
            .iter()
            .map(|shard| artifact::InputShard {
                row_start: shard.manifest.config.row_start,
                row_end: shard.manifest.config.row_end,
                config_blake3: shard.manifest.config_blake3.clone(),
                payload_blake3: shard.manifest.payload.blake3.clone(),
            })
            .collect(),
    }
}

fn aggregate_replay(shards: &[ShardInput]) -> Result<Vec<rows::ReplayDiagnostic>> {
    let mut aggregate = shards
        .first()
        .context("Muse row-shard set is empty")?
        .manifest
        .diagnostics
        .clone();
    for shard in &shards[1..] {
        ensure!(
            shard.manifest.diagnostics.len() == aggregate.len(),
            "Muse row-shard replay schedules differ"
        );
        for (aggregate, current) in aggregate.iter_mut().zip(&shard.manifest.diagnostics) {
            ensure!(
                aggregate.block == current.block && aggregate.kind == current.kind,
                "Muse row-shard replay schedules differ at block {}",
                current.block
            );
            aggregate.post_attention_replay_max_abs_error = aggregate
                .post_attention_replay_max_abs_error
                .max(current.post_attention_replay_max_abs_error);
            aggregate.post_block_replay_max_abs_error = aggregate
                .post_block_replay_max_abs_error
                .max(current.post_block_replay_max_abs_error);
        }
    }
    Ok(aggregate)
}

fn open_output(
    output: &Path,
    resume: bool,
    config: &artifact::Config,
    corpus: &rows::CorpusSummary,
    assembly: &artifact::Assembly,
    config_blake3: &str,
) -> Result<Option<artifact::Manifest>> {
    super::validate_output_leaf(output)?;
    let assembly_state = artifact::AssemblyState {
        schema: artifact::ASSEMBLY_STATE_SCHEMA.into(),
        schema_version: artifact::SCHEMA_VERSION,
        config_blake3: config_blake3.into(),
        assembly_blake3: super::digest_json(assembly)?,
    };
    let mut created = false;
    if output.exists() {
        let metadata = std::fs::symlink_metadata(output)
            .with_context(|| format!("inspect Muse full output {}", output.display()))?;
        ensure!(
            metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
            "Muse full output {} must be a real directory",
            output.display()
        );
        ensure!(
            resume,
            "Muse full output {} already exists; pass --resume",
            output.display()
        );
    } else {
        let parent = output.parent().context("Muse full output has no parent")?;
        DirBuilder::new()
            .mode(0o700)
            .create(output)
            .with_context(|| format!("create Muse full output {}", output.display()))?;
        super::sync_directory(parent)?;
        created = true;
    }

    let manifest_path = output.join(artifact::MANIFEST_NAME);
    if !manifest_path.exists() {
        let state_path = output.join(artifact::ASSEMBLY_STATE_NAME);
        if created {
            super::publish_immutable(
                &state_path,
                &super::serialize_json_pretty_bounded(&assembly_state, "Muse full assembly state")?,
            )?;
        } else {
            let observed: artifact::AssemblyState = super::read_json_file(&state_path)
                .context("existing Muse output is not an owned interrupted assembly")?;
            ensure!(
                observed == assembly_state,
                "interrupted Muse assembly state differs from the requested shard set"
            );
        }
        return Ok(None);
    }
    let manifest: artifact::Manifest = super::read_json_file(&manifest_path)?;
    artifact::validate_manifest(&manifest)?;
    ensure!(
        &manifest.config == config
            && &manifest.corpus == corpus
            && &manifest.assembly == assembly
            && manifest.config_blake3 == config_blake3,
        "completed Muse full transport differs from requested shard set"
    );
    verify_complete_payload(output, &manifest.payload)?;
    Ok(Some(manifest))
}

fn verify_complete_payload(output: &Path, payload: &artifact::Payload) -> Result<()> {
    const BUFFER_BYTES: usize = 8 * 1024 * 1024;
    let path = output.join(&payload.path);
    let (mut file, length) = super::open_regular_file(&path)?;
    ensure!(
        length as u64 == payload.byte_length,
        "completed Muse full-transport payload length changed"
    );
    let mut buffer = vec![0u8; BUFFER_BYTES];
    let mut whole = Hasher::new();
    for matrix in &payload.matrices {
        file.seek(SeekFrom::Start(matrix.byte_offset))
            .with_context(|| format!("seek completed Muse payload {}", path.display()))?;
        let mut remaining = matrix.byte_length;
        let mut matrix_hasher = Hasher::new();
        while remaining > 0 {
            let count = usize::try_from(remaining.min(BUFFER_BYTES as u64))
                .context("Muse payload verification chunk")?;
            file.read_exact(&mut buffer[..count])
                .with_context(|| format!("verify completed Muse payload {}", path.display()))?;
            matrix_hasher.update(&buffer[..count]);
            whole.update(&buffer[..count]);
            remaining -= count as u64;
        }
        ensure!(
            matrix_hasher.finalize().to_hex().as_str() == matrix.blake3,
            "completed Muse source matrix {} digest mismatch",
            matrix.source_layer
        );
    }
    ensure!(
        whole.finalize().to_hex().as_str() == payload.blake3,
        "completed Muse full-transport payload digest mismatch"
    );
    Ok(())
}

fn remove_regular_if_present(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                "Muse assembly path {} must be a regular non-symlink file",
                path.display()
            );
            std::fs::remove_file(path)
                .with_context(|| format!("remove interrupted Muse assembly {}", path.display()))?;
            super::sync_directory(path.parent().unwrap_or_else(|| Path::new(".")))?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("inspect Muse assembly path {}", path.display()));
        }
    }
    Ok(())
}

fn convert_f32_slice_to_f16(input: &[u8], output: &mut Vec<u8>) -> Result<()> {
    ensure!(
        input.len().is_multiple_of(4),
        "Muse F32 transport slice is not word-aligned"
    );
    output.clear();
    let expected = input
        .len()
        .checked_div(2)
        .context("Muse F16 transport slice length")?;
    output
        .try_reserve_exact(expected)
        .context("allocate Muse F16 conversion buffer")?;
    for chunk in input.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().unwrap());
        ensure!(
            value.is_finite(),
            "Muse input transport contains non-finite F32"
        );
        let converted = f16::from_f32(value);
        ensure!(
            converted.is_finite(),
            "Muse input transport overflows finite F16 storage"
        );
        output.extend_from_slice(&converted.to_bits().to_le_bytes());
    }
    ensure!(
        output.len() == expected,
        "Muse F16 conversion produced the wrong byte count"
    );
    Ok(())
}

fn config_source_count(shards: &[ShardInput]) -> Result<usize> {
    Ok(shards
        .first()
        .context("Muse row-shard set is empty")?
        .manifest
        .config
        .source_layers
        .len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn f32_to_f16_conversion_is_little_endian_and_fails_closed() {
        let input = [0.0f32, 1.5, -2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut output = Vec::new();
        convert_f32_slice_to_f16(&input, &mut output).unwrap();
        let observed = output
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(observed, [0.0, 1.5, -2.0]);

        let nan = f32::NAN.to_le_bytes();
        assert!(convert_f32_slice_to_f16(&nan, &mut output).is_err());
        let overflow = f32::MAX.to_le_bytes();
        assert!(convert_f32_slice_to_f16(&overflow, &mut output).is_err());
    }

    #[test]
    fn completed_payload_verification_checks_matrix_and_whole_digests() {
        let root = std::env::temp_dir().join(format!(
            "qwen-muse-full-verify-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let bytes = b"abcdefgh";
        std::fs::write(root.join(artifact::PAYLOAD_NAME), bytes).unwrap();
        let payload = artifact::Payload {
            path: artifact::PAYLOAD_NAME.into(),
            dtype: "f16_le".into(),
            shape: [2, 1, 2],
            byte_length: bytes.len() as u64,
            blake3: blake3::hash(bytes).to_hex().to_string(),
            matrices: vec![
                artifact::MatrixDescriptor {
                    source_layer: 0,
                    byte_offset: 0,
                    byte_length: 4,
                    blake3: blake3::hash(&bytes[..4]).to_hex().to_string(),
                },
                artifact::MatrixDescriptor {
                    source_layer: 1,
                    byte_offset: 4,
                    byte_length: 4,
                    blake3: blake3::hash(&bytes[4..]).to_hex().to_string(),
                },
            ],
        };
        verify_complete_payload(&root, &payload).unwrap();
        std::fs::write(root.join(artifact::PAYLOAD_NAME), b"abcdEfgh").unwrap();
        assert!(verify_complete_payload(&root, &payload).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
