use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::json;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::{
    ffi::OsStrExt,
    fs::{DirBuilderExt, OpenOptionsExt},
};
use std::path::{Path, PathBuf};

pub(super) struct Bundle {
    staging: PathBuf,
    output: PathBuf,
    file: File,
    layers: Vec<u32>,
    vocab: usize,
    written: Vec<bool>,
    bytes: usize,
}

impl Bundle {
    pub(super) fn optional(
        path: Option<&Path>,
        layers: &[u32],
        vocab: usize,
    ) -> Result<Option<Self>> {
        path.map(|path| Self::new(path, layers, vocab)).transpose()
    }

    fn new(output: &Path, layers: &[u32], vocab: usize) -> Result<Self> {
        ensure!(
            !layers.is_empty() && layers.len() <= 4096 && vocab > 0 && vocab <= 4_194_304,
            "full-output dimensions exceed bounds"
        );
        ensure!(
            layers
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == layers.len(),
            "duplicate full-output layers"
        );
        let bytes = layers
            .len()
            .checked_mul(vocab)
            .and_then(|n| n.checked_mul(4))
            .context("full-output dimensions overflow")?;
        ensure!(
            bytes <= 16 * 1024 * 1024 * 1024usize,
            "full-output exceeds 16 GiB"
        );
        let output = super::full_lens::resolve_output_file(output)?;
        ensure!(
            std::fs::symlink_metadata(&output).is_err(),
            "full-output already exists"
        );
        let staging = output.with_file_name(format!(
            ".full-output-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::DirBuilder::new().mode(0o700).create(&staging)?;
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(staging.join("logits.f32le"))
        {
            Ok(file) => file,
            Err(error) => {
                let _ = std::fs::remove_dir_all(&staging);
                return Err(error.into());
            }
        };
        Ok(Self {
            staging,
            output,
            file,
            layers: layers.to_vec(),
            vocab,
            written: vec![false; layers.len()],
            bytes,
        })
    }

    pub(super) fn row(&mut self, slot: usize, logits: &[f32]) -> Result<()> {
        ensure!(
            slot < self.layers.len() && !self.written[slot],
            "invalid or duplicate full-output row"
        );
        ensure!(
            logits.len() == self.vocab && logits.iter().all(|v| v.is_finite()),
            "full-output row must contain exactly V finite logits"
        );
        self.file
            .seek(SeekFrom::Start((slot * self.vocab * 4) as u64))?;
        let mut writer = std::io::BufWriter::new(&mut self.file);
        for value in logits {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.flush()?;
        self.written[slot] = true;
        Ok(())
    }

    pub(super) fn publish(mut self, provenance: &serde_json::Value) -> Result<()> {
        ensure!(
            self.written.iter().all(|v| *v),
            "full-output has missing rows"
        );
        ensure!(
            self.file.metadata()?.len() == self.bytes as u64,
            "full-output byte count mismatch"
        );
        self.file.sync_all()?;
        self.file.rewind()?;
        let mut hasher = blake3::Hasher::new();
        let mut buffer = [0u8; 65536];
        loop {
            let n = self.file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        let mut identity_hasher = blake3::Hasher::new();
        serde_json::to_writer(
            super::ByteLimitedWriter::new(&mut identity_hasher, super::JSON_FILE_MAX_BYTES),
            provenance,
        )
        .context("hash bounded readout metadata")?;
        let identity = identity_hasher.finalize().to_hex().to_string();
        let observer = json!({"readout": provenance["readout"], "source_site": provenance["source_site"], "observer": provenance["observer"], "artifact": provenance["artifact"], "transfer": provenance["transfer"], "source_layers": self.layers, "output_tail": provenance["deployed_model"]["output_tail"], "scoring": provenance["scoring"]});
        let identities = json!({"scheme": "blake3_canonical_json_v1", "model_content_blake3": provenance["deployed_model"]["content_blake3"], "input_blake3": super::digest_json(&provenance["input"])?, "reader_blake3": super::digest_json(&provenance["reader"])?, "observer_blake3": super::digest_json(&observer)?, "observer": observer});
        let metadata = json!({
            "schema": "llm.lens.full_vocabulary_bundle", "schema_version": 1,
            "score_semantics": "deployed_pre_softmax_logits_after_architectural_output_norm_scaling_and_softcap",
            "payload": {"path": "logits.f32le", "dtype": "f32", "endianness": "little", "shape": [self.layers.len(), self.vocab], "axes": ["source_layer", "token_id"], "source_layers": self.layers, "token_ids": {"start": 0, "step": 1, "count": self.vocab}, "byte_count": self.bytes, "blake3": hasher.finalize().to_hex().to_string(), "all_finite": true},
            "identities": identities, "readout_metadata_blake3": identity,
            "execution_provenance": execution_provenance()
        });
        #[derive(Serialize)]
        struct Metadata<'a> {
            #[serde(flatten)]
            header: serde_json::Value,
            readout: &'a serde_json::Value,
        }
        let metadata = Metadata {
            header: metadata,
            readout: provenance,
        };
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(self.staging.join("metadata.json"))?;
        {
            let mut buffered = std::io::BufWriter::new(&mut file);
            let mut writer =
                super::ByteLimitedWriter::new(&mut buffered, super::JSON_FILE_MAX_BYTES);
            serde_json::to_writer_pretty(&mut writer, &metadata)
                .context("serialize bounded full-output metadata")?;
            writer.flush()?;
        }
        file.sync_all()?;
        File::open(&self.staging)?.sync_all()?;
        let old = std::ffi::CString::new(self.staging.as_os_str().as_bytes())?;
        let new = std::ffi::CString::new(self.output.as_os_str().as_bytes())?;
        let result = unsafe {
            libc::renameatx_np(
                libc::AT_FDCWD,
                old.as_ptr(),
                libc::AT_FDCWD,
                new.as_ptr(),
                libc::RENAME_EXCL,
            )
        };
        if result != 0 {
            return Err(std::io::Error::last_os_error())
                .context("publish full-output without overwrite");
        }
        if let Some(parent) = self.output.parent() {
            if let Err(error) = File::open(parent).and_then(|file| file.sync_all()) {
                eprintln!("warning: full-output published but parent sync failed: {error}");
            }
        }
        Ok(())
    }
}

impl Drop for Bundle {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.staging);
    }
}

pub(super) fn ranked(logits: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
    ensure!(
        k > 0 && k <= logits.len() && logits.iter().all(|v| v.is_finite()),
        "invalid full vocabulary logits or top-k"
    );
    let mut best = Vec::with_capacity(k + 1);
    for (id, &value) in logits.iter().enumerate() {
        let pair = (id as u32, value);
        let at = best.partition_point(|&(old_id, old): &(u32, f32)| {
            old.total_cmp(&value).is_gt() || (old.total_cmp(&value).is_eq() && old_id < pair.0)
        });
        if at < k {
            best.insert(at, pair);
            best.truncate(k);
        }
    }
    Ok(best)
}

pub(super) fn execution_provenance() -> serde_json::Value {
    json!({
        "reader_identity_scope": "build_and_source_identity_only_not_effective_numerical_execution",
        "effective_kernel_settings_recorded": false,
        "reproducibility_caveat": "Native runtime kernel selection, environment-controlled numerical settings, device and driver are not comprehensively recorded or bound by reader/build hashes; equal identities do not guarantee bitwise-identical logits."
    })
}

pub(super) fn retained_metadata_budget(
    layers: usize,
    hidden: usize,
    tokens: usize,
    top_k: usize,
    include_vector: bool,
) -> Result<()> {
    // Price retained JSON values and pretty-printed vectors before capture. The
    // writer remains the exact byte limit (token-piece lengths are data-dependent).
    let vectors = if include_vector {
        layers.checked_mul(hidden).and_then(|n| n.checked_mul(32))
    } else {
        Some(0)
    }
    .context("retained vector metadata budget overflow")?;
    let budget = layers
        .checked_mul(top_k)
        .and_then(|n| n.checked_mul(4096))
        .and_then(|n| n.checked_add(layers.checked_mul(4096)?))
        .and_then(|n| n.checked_add(tokens.checked_mul(32)?))
        .and_then(|n| n.checked_add(vectors))
        .and_then(|n| n.checked_add(1024 * 1024))
        .context("retained metadata budget overflow")?;
    ensure!(
        budget <= super::JSON_FILE_MAX_BYTES,
        "retained readout metadata estimate {budget} exceeds {} bytes; select fewer layers/top-k entries or omit --include-vector",
        super::JSON_FILE_MAX_BYTES
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_metadata_admits_normal_full_depth_and_rejects_large_vectors() {
        retained_metadata_budget(64, 5120, 256, 10, true).unwrap();
        assert!(retained_metadata_budget(4096, 32768, 256, 10, true).is_err());
        assert!(retained_metadata_budget(usize::MAX, usize::MAX, 1, 1, true).is_err());
        assert!(retained_metadata_budget(1, 1, usize::MAX, 1, false).is_err());
    }

    #[test]
    fn full_depth_vectors_fit_compact_and_bundle_metadata_limits() {
        let output = output();
        let layers: Vec<u32> = (0..64).collect();
        let mut bundle = Bundle::new(&output, &layers, 10).unwrap();
        let values = vec![f32::MIN_POSITIVE; 5120];
        let results: Vec<_> = layers.iter().enumerate().map(|(slot, layer)| {
            bundle.row(slot, &[1.0; 10]).unwrap();
            json!({"source_layer": layer, "transported_vector": {"operation": "identity", "shape": [5120], "values": values}, "top_k": []})
        }).collect();
        let document = json!({"results": results});
        assert!(
            crate::serialize_json_pretty_bounded(&document, "full depth")
                .unwrap()
                .len()
                <= crate::JSON_FILE_MAX_BYTES
        );
        bundle.publish(&document).unwrap();
        assert!(
            std::fs::metadata(output.join("metadata.json"))
                .unwrap()
                .len()
                <= crate::JSON_FILE_MAX_BYTES as u64
        );
        std::fs::remove_dir_all(output).unwrap();
    }

    #[test]
    fn bounded_pretty_serializer_stops_before_consuming_large_sequence() {
        use serde::ser::SerializeSeq;
        struct Large(std::cell::Cell<usize>);
        impl Serialize for Large {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut seq = serializer.serialize_seq(Some(1_000_000))?;
                for i in 0..1_000_000 {
                    self.0.set(i + 1);
                    seq.serialize_element(&"x".repeat(128))?;
                }
                seq.end()
            }
        }
        let value = Large(std::cell::Cell::new(0));
        assert!(crate::serialize_json_pretty_bounded(&value, "oversized test").is_err());
        assert!(value.0.get() < 150_000);
    }

    #[test]
    fn execution_identity_explicitly_excludes_unrecorded_kernel_settings() {
        let provenance = execution_provenance();
        assert_eq!(provenance["effective_kernel_settings_recorded"], false);
        assert!(
            provenance["reader_identity_scope"]
                .as_str()
                .unwrap()
                .contains("not_effective_numerical_execution")
        );
        assert!(
            provenance["reproducibility_caveat"]
                .as_str()
                .unwrap()
                .contains("bitwise")
        );
    }

    #[test]
    fn metadata_overflow_never_publishes_and_cleans_staging() {
        let output = output();
        let mut bundle = Bundle::new(&output, &[0], 1).unwrap();
        bundle.row(0, &[1.0]).unwrap();
        let staging = bundle.staging.clone();
        let document = json!({"large": "x".repeat(crate::JSON_FILE_MAX_BYTES)});
        assert!(bundle.publish(&document).is_err());
        assert!(!output.exists());
        assert!(!staging.exists());
    }

    fn output() -> PathBuf {
        std::env::temp_dir().join(format!(
            "qwen-full-output-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn full_output_shape_order_hash_and_metadata() {
        let output = output();
        let mut bundle = Bundle::new(&output, &[7, 2], 3).unwrap();
        bundle.row(1, &[4.0, 5.0, 6.0]).unwrap();
        bundle.row(0, &[1.0, -2.0, 3.0]).unwrap();
        let document = json!({"readout": "native_plain_logit_lens", "observer": {"transport": "identity"}, "input": {"token_ids": [12], "selected_position": 0}, "deployed_model": {"content_blake3": "strong-model-id"}, "reader": {"build_source_state": "source-id"}});
        bundle.publish(&document).unwrap();
        let payload = std::fs::read(output.join("logits.f32le")).unwrap();
        let expected: Vec<u8> = [1.0f32, -2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(payload, expected);
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(metadata["payload"]["shape"], json!([2, 3]));
        assert_eq!(metadata["payload"]["source_layers"], json!([7, 2]));
        assert_eq!(
            metadata["payload"]["axes"],
            json!(["source_layer", "token_id"])
        );
        assert_eq!(metadata["payload"]["endianness"], "little");
        assert_eq!(metadata["payload"]["byte_count"], 24);
        assert_eq!(
            metadata["payload"]["blake3"],
            blake3::hash(&payload).to_hex().to_string()
        );
        assert_eq!(
            metadata["identities"]["input_blake3"],
            crate::digest_json(&document["input"]).unwrap()
        );
        assert_eq!(metadata["readout"], document);
        assert!(
            metadata["score_semantics"]
                .as_str()
                .unwrap()
                .contains("scaling_and_softcap")
        );
        assert!(Bundle::new(&output, &[0], 1).is_err());
        assert_eq!(std::fs::read(output.join("logits.f32le")).unwrap(), payload);
        std::fs::remove_dir_all(output).unwrap();
    }

    #[test]
    fn full_output_finite_dimensions_and_cleanup() {
        let output = output();
        assert!(Bundle::new(&output, &[], 3).is_err());
        assert!(Bundle::new(&output, &[0, 0], 3).is_err());
        assert!(Bundle::new(&output, &[0], usize::MAX).is_err());
        let mut bundle = Bundle::new(&output, &[0], 2).unwrap();
        let staging = bundle.staging.clone();
        assert!(bundle.row(0, &[f32::NAN, 1.0]).is_err());
        assert!(bundle.row(0, &[f32::INFINITY, 1.0]).is_err());
        assert!(bundle.row(0, &[1.0]).is_err());
        assert!(bundle.row(1, &[1.0, 2.0]).is_err());
        assert!(bundle.publish(&json!({})).is_err());
        assert!(!staging.exists());
        assert!(!output.exists());
    }

    #[test]
    fn full_output_publish_race_does_not_clobber() {
        let output = output();
        let mut bundle = Bundle::new(&output, &[0], 1).unwrap();
        let staging = bundle.staging.clone();
        bundle.row(0, &[1.0]).unwrap();
        assert!(bundle.row(0, &[2.0]).is_err());
        std::fs::create_dir(&output).unwrap();
        assert!(bundle.publish(&json!({})).is_err());
        assert!(!staging.exists());
        assert!(output.is_dir());
        assert_eq!(std::fs::read_dir(&output).unwrap().count(), 0);
        std::fs::remove_dir(output).unwrap();
    }

    #[test]
    fn full_output_ranking_is_bounded_and_deterministic() {
        assert_eq!(
            ranked(&[2.0, 1.0, 2.0, 3.0], 3).unwrap(),
            vec![(3, 3.0), (0, 2.0), (2, 2.0)]
        );
        assert!(ranked(&[f32::NAN], 1).is_err());
    }
}
