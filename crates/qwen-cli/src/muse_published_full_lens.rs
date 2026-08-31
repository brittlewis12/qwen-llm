use super::muse_published_full_lens_artifact as artifact;
use anyhow::{Context, Result, ensure};
use blake3::Hasher;
use clap::Args;
use std::fs::{DirBuilder, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use zip::ZipArchive;

const VERIFY_BUFFER_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Args)]
pub(crate) struct ImportMuseFullArgs {
    /// One supported exact pinned published Muse .pt transport asset.
    #[arg(long)]
    source: PathBuf,

    /// New immutable published Muse full-lens artifact directory.
    #[arg(long)]
    output: PathBuf,
}

pub(crate) fn import_full(mut args: ImportMuseFullArgs) -> Result<()> {
    super::validate_token_build_identity(
        env!("QWEN_BUILD_SOURCE_STATE"),
        env!("QWEN_BUILD_STAMP_ERROR"),
    )?;
    args.output = super::resolve_output_path(&args.output)?;
    let (mut source, source_length) = super::open_regular_file(&args.source)?;
    let source_length = u64::try_from(source_length).context("Muse source byte length")?;
    let source_metadata = source
        .metadata()
        .with_context(|| format!("inspect opened {}", args.source.display()))?;
    let source_modified = source_metadata.modified().ok();
    let source_sha256 = super::published_pt::hash_sha256(&mut source, &args.source)?;
    let profile = artifact::profile_for_source(source_length, &source_sha256).with_context(|| {
        format!(
            "{} length {} and SHA-256 {} do not identify a supported pinned Muse full-lens asset",
            args.source.display(),
            source_length,
            source_sha256
        )
    })?;

    prepare_output_directory(&args.output)?;
    let manifest_path = args.output.join(artifact::MANIFEST_NAME);
    if manifest_path.exists() {
        let manifest: artifact::Manifest = super::read_json_file(&manifest_path)?;
        artifact::validate_manifest(&manifest)?;
        ensure!(
            artifact::profile_for_manifest(&manifest)?.id == profile.id,
            "existing Muse output was imported from a different published lens"
        );
        verify_payload(&args.output, &manifest.payload, profile)?;
        println!("{}", serde_json::to_string_pretty(&manifest)?);
        return Ok(());
    }

    source
        .seek(SeekFrom::Start(0))
        .with_context(|| format!("rewind pinned Muse source {}", args.source.display()))?;
    let mut archive = ZipArchive::new(source)
        .with_context(|| format!("open pinned Muse torch ZIP {}", args.source.display()))?;
    let spec = profile.archive_spec();
    super::published_pt::validate_archive(&mut archive, spec)?;

    let staging = staging_path(&args.output)?;
    let import = (|| -> Result<artifact::Payload> {
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&staging)
            .with_context(|| format!("create Muse import staging file {}", staging.display()))?;
        let extracted = super::published_pt::extract_payload(&mut archive, spec, &mut output)?;
        output
            .sync_all()
            .with_context(|| format!("sync Muse import staging file {}", staging.display()))?;
        drop(output);
        let payload = artifact::payload_from_extracted(profile, extracted)?;

        let source = archive.into_inner();
        let final_metadata = source
            .metadata()
            .with_context(|| format!("reinspect opened {}", args.source.display()))?;
        ensure!(
            final_metadata.len() == profile.source_bytes()
                && final_metadata.modified().ok() == source_modified,
            "pinned Muse source changed while it was being imported"
        );
        publish_payload(&staging, &args.output, &payload, profile)?;
        Ok(payload)
    })();
    if import.is_err() {
        let _ = std::fs::remove_file(&staging);
    }
    let payload = import?;

    let manifest = artifact::canonical_manifest(profile, payload);
    artifact::validate_manifest(&manifest)?;
    super::publish_immutable(
        &manifest_path,
        &super::serialize_json_pretty_bounded(&manifest, "Muse published full-lens manifest")?,
    )?;
    super::sync_directory(&args.output)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn prepare_output_directory(output: &Path) -> Result<()> {
    if output.exists() {
        let metadata = std::fs::symlink_metadata(output)
            .with_context(|| format!("inspect Muse import output {}", output.display()))?;
        ensure!(
            metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
            "Muse import output {} must be a real directory",
            output.display()
        );
        return Ok(());
    }
    let parent = output
        .parent()
        .context("Muse import output has no parent")?;
    DirBuilder::new()
        .mode(0o700)
        .create(output)
        .with_context(|| format!("create Muse import output {}", output.display()))?;
    super::sync_directory(parent)
}

fn staging_path(output: &Path) -> Result<PathBuf> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?
        .as_nanos();
    Ok(output.join(format!(
        ".{}.stage.{}.{}",
        artifact::PAYLOAD_NAME,
        std::process::id(),
        nonce
    )))
}

fn publish_payload(
    staging: &Path,
    output: &Path,
    payload: &artifact::Payload,
    profile: artifact::Profile,
) -> Result<()> {
    let destination = output.join(&payload.path);
    match std::fs::hard_link(staging, &destination) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            verify_payload(output, payload, profile)?;
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "publish Muse staging payload {} to {}",
                    staging.display(),
                    destination.display()
                )
            });
        }
    }
    std::fs::remove_file(staging)
        .with_context(|| format!("remove Muse staging payload {}", staging.display()))?;
    super::sync_directory(output)
}

pub(crate) fn verify_payload(
    directory: &Path,
    payload: &artifact::Payload,
    profile: artifact::Profile,
) -> Result<()> {
    ensure!(
        Path::new(&payload.path).components().count() == 1,
        "Muse published payload path must be one relative filename"
    );
    let path = directory.join(&payload.path);
    let (mut file, length) = super::open_regular_file(&path)?;
    ensure!(
        length as u64 == payload.byte_length,
        "Muse published payload length changed"
    );
    let mut buffer = vec![0u8; VERIFY_BUFFER_BYTES];
    let mut whole = Hasher::new();
    let archive_spec = profile.archive_spec();
    for matrix in &payload.matrices {
        file.seek(SeekFrom::Start(matrix.byte_offset))
            .with_context(|| format!("seek Muse published matrix {}", matrix.source_layer))?;
        let mut remaining = matrix.byte_length;
        let mut matrix_hasher = Hasher::new();
        while remaining > 0 {
            let matrix_offset = matrix.byte_length - remaining;
            let count = usize::try_from(remaining.min(VERIFY_BUFFER_BYTES as u64))
                .context("Muse published verification chunk")?;
            file.read_exact(&mut buffer[..count])
                .with_context(|| format!("read Muse published payload {}", path.display()))?;
            super::published_pt::ensure_finite_f16(
                &buffer[..count],
                matrix.source_layer as usize,
                matrix_offset as usize / 2,
            )?;
            if profile.identity_layer_index() == Some(matrix.source_layer as usize) {
                super::published_pt::ensure_identity_f16(
                    &buffer[..count],
                    matrix_offset as usize / 2,
                    archive_spec.hidden_size,
                    matrix.source_layer as usize,
                )?;
            }
            matrix_hasher.update(&buffer[..count]);
            whole.update(&buffer[..count]);
            remaining -= count as u64;
        }
        ensure!(
            matrix_hasher.finalize().to_hex().as_str() == matrix.blake3,
            "Muse published source matrix {} digest mismatch",
            matrix.source_layer
        );
    }
    let mut extra = [0u8; 1];
    ensure!(
        file.read(&mut extra)
            .with_context(|| format!("check Muse published payload end {}", path.display()))?
            == 0,
        "Muse published payload contains trailing bytes"
    );
    ensure!(
        whole.finalize().to_hex().as_str() == payload.blake3,
        "Muse published payload BLAKE3 mismatch"
    );
    Ok(())
}
