use anyhow::{Context, Result};
use qwen_llm::checkpoint_identity::verified_checkpoint_content_identity;
use qwen_llm::gguf::GgufFile;
use qwen_llm::muse_glimmer::MuseGlimmerModel;
use serde_json::{Value, json};
use std::path::Path;

pub(crate) struct GenericAccess {
    pub(crate) data: crate::linear_transport::VerifiedTransport,
    pub(crate) runtime_binding: Value,
}

impl GenericAccess {
    pub(crate) fn open(
        directory: &Path,
        gguf: &GgufFile,
        _cache: Option<&Path>,
        allow: bool,
    ) -> Result<Self> {
        let data = crate::linear_transport::VerifiedTransport::open(directory)?;
        let bound = MuseGlimmerModel::from_gguf(gguf)
            .context("bind generic transport to actual Muse GGUF")?;
        let config = &bound.config;
        let architecture = gguf
            .get_str("general.architecture")
            .context("missing runtime architecture")?;
        let tokenizer = format!(
            "{:016x}",
            qwen_llm::runtime::tokenizer_metadata_identity(gguf)
        );
        let locator = retained_file_locator(gguf)?;
        let (content, content_bytes_hashed) = if data.manifest().model.exact_binding.is_some() {
            crate::shutdown::checkpoint()?;
            let identity = verified_checkpoint_content_identity(gguf)?;
            crate::shutdown::checkpoint()?;
            (
                Some(crate::hex(&identity.content_id)),
                identity.bytes_hashed,
            )
        } else {
            (None, 0)
        };
        let status = data.bind(
            architecture,
            config.layer_count,
            config.hidden_size,
            config.vocab_size,
            content.as_deref().unwrap_or(""),
            &tokenizer,
            true,
            allow,
        )?;
        let runtime_binding = json!({"status":status, "architecture":architecture,
            "n_layers":config.layer_count, "hidden_size":config.hidden_size, "vocab_size":config.vocab_size,
            "gguf_content_blake3":content, "tokenizer_metadata_id":tokenizer,
            "content_identity_provenance": if content.is_some() { "retained_bytes_hashed" } else { "not_verified" },
            "content_bytes_hashed":content_bytes_hashed,
            "locator_scheme":"ordered_gguf_retained_file_stamps_blake3_v1", "locator_id":locator,
            "qualification":"producer_claims_only"});
        Ok(Self {
            data,
            runtime_binding,
        })
    }

    pub(crate) fn summary(&self, directory: &Path) -> Result<Value> {
        let mut canonical = self.data.original_manifest().clone();
        canonical.sort_all_objects();
        let t = &self.data.manifest().transport;
        Ok(
            json!({"kind":"linear_transport", "manifest":directory.join("lens.json"),
            "manifest_canonical_json_blake3":crate::digest_json(&canonical)?,
            "method":t.method, "target_layer":t.target_layer, "orientation":t.orientation,
            "source_site":"post_block_residual", "payload_blake3":self.data.payload_blake3(),
            "producer_contract":self.data.original_manifest(), "runtime_binding":self.runtime_binding,
            "scoring":"transport_then_deployed_output_tail"}),
        )
    }

    pub(crate) fn trace_locator(&self) -> Result<(&'static str, String, bool)> {
        if let Some(content) = self.runtime_binding["gguf_content_blake3"].as_str() {
            Ok(("ordered_gguf_content_blake3_v1", content.into(), true))
        } else {
            let locator = self.runtime_binding["locator_id"]
                .as_str()
                .context("missing retained GGUF locator")?;
            Ok((
                "ordered_gguf_retained_file_stamps_blake3_v1",
                locator.into(),
                false,
            ))
        }
    }
}

fn retained_file_locator(gguf: &GgufFile) -> Result<String> {
    let stamps = gguf.revalidate_retained_shard_stamps()?;
    let mut hash = blake3::Hasher::new();
    hash.update(b"ordered_gguf_retained_file_stamps_blake3_v1\0");
    hash.update(&(stamps.len() as u64).to_le_bytes());
    for stamp in stamps {
        hash.update(&(stamp.shard_idx as u64).to_le_bytes());
        hash.update(&stamp.device.to_le_bytes());
        hash.update(&stamp.inode.to_le_bytes());
        hash.update(&stamp.size.to_le_bytes());
        hash.update(&stamp.mtime_sec.to_le_bytes());
        hash.update(&stamp.mtime_nsec.to_le_bytes());
        hash.update(&stamp.ctime_sec.to_le_bytes());
        hash.update(&stamp.ctime_nsec.to_le_bytes());
    }
    Ok(hash.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_transport::{
        VerifiedTransport,
        tests::{fixture, modify},
    };
    use std::io::{Seek, SeekFrom, Write};

    #[test]
    fn exact_identity_matches_native_bytes_and_ignores_declared_cached_roots() {
        use qwen_llm::checkpoint_identity::{
            CheckpointIdentityCache, checkpoint_content_identity,
            checkpoint_content_identity_without_weight_hashing,
        };
        let f = fixture("future-fit", 2, 123);
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.resize(32, 0);
        let path = f.0.join("model.gguf");
        std::fs::write(&path, &bytes).unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let native = checkpoint_content_identity(
            &gguf,
            &CheckpointIdentityCache::new(f.0.join("native-cache")),
        )
        .unwrap();
        let report = verified_checkpoint_content_identity(&gguf).unwrap();
        let verified = crate::hex(&report.content_id);
        let n = report.bytes_hashed;
        assert_eq!(verified, crate::hex(&native.content_id));
        assert_eq!(n, bytes.len() as u64);
        let declarations = f.0.join(".cache/huggingface/download");
        std::fs::create_dir_all(&declarations).unwrap();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
            + 1.0;
        std::fs::write(
            declarations.join("model.gguf.metadata"),
            format!("{}\n{}\n{timestamp}\n", "a".repeat(40), "b".repeat(64)),
        )
        .unwrap();
        let cache = CheckpointIdentityCache::new(f.0.join("declaration-cache"));
        let declared = checkpoint_content_identity_without_weight_hashing(&gguf, &cache).unwrap();
        assert_eq!(declared.bytes_hashed, 0);
        assert_ne!(verified, crate::hex(&declared.content_id));
        let hit = checkpoint_content_identity_without_weight_hashing(&gguf, &cache).unwrap();
        assert_eq!(hit.content_id, declared.content_id);
        assert_eq!(hit.bytes_hashed, 0);
        let repeated = verified_checkpoint_content_identity(&gguf).unwrap();
        assert_eq!(crate::hex(&repeated.content_id), verified);
        assert_eq!(repeated.bytes_hashed, n);
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[0])
            .unwrap();
        assert!(verified_checkpoint_content_identity(&gguf).is_err());
    }

    #[test]
    fn unbound_trace_uses_retained_file_locator_without_authentication() {
        let f = fixture("future-fit", 2, 123);
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.resize(32, 0);
        let first = f.0.join("first.gguf");
        let second = f.0.join("second.gguf");
        std::fs::write(&first, &bytes).unwrap();
        std::fs::write(&second, &bytes).unwrap();
        let gguf = GgufFile::open(&first).unwrap();
        let locator = retained_file_locator(&gguf).unwrap();
        assert_eq!(locator.len(), 64);
        assert_eq!(locator, retained_file_locator(&gguf).unwrap());
        assert_ne!(
            locator,
            retained_file_locator(&GgufFile::open(&second).unwrap()).unwrap()
        );
        let access = GenericAccess {
            data: VerifiedTransport::open(&f.0).unwrap(),
            runtime_binding: json!({"gguf_content_blake3":null,"locator_id":locator}),
        };
        let (scheme, id, authenticated) = access.trace_locator().unwrap();
        assert_eq!(scheme, "ordered_gguf_retained_file_stamps_blake3_v1");
        assert_eq!(id, locator);
        assert!(!authenticated);
    }

    #[test]
    fn muse_generic_claims_do_not_become_published_fit_metadata() {
        let f = fixture("future-muse-fit", 2, 123);
        modify(&f, |m| m["model"]["architecture"] = "muse-glimmer".into());
        let data = VerifiedTransport::open(&f.0).unwrap();
        assert!(
            data.bind("muse-glimmer", 3, 2, 32, "", "", true, false)
                .is_err()
        );
        let status = data
            .bind("muse-glimmer", 3, 2, 32, "", "", true, true)
            .unwrap();
        let access = GenericAccess {
            data,
            runtime_binding: json!({"status":status}),
        };
        let summary = access.summary(&f.0).unwrap();
        assert_eq!(summary["method"], "future-muse-fit");
        assert_eq!(
            summary["producer_contract"]["qualification"]["validated"],
            true
        );
        assert_eq!(
            summary["runtime_binding"]["status"],
            "source_deployment_equivalence_unverified"
        );
        for key in [
            "fit_n_prompts",
            "fit_max_sequence_length",
            "fitted_checkpoint",
            "source_repository",
            "claims_basis",
        ] {
            assert!(summary.get(key).is_none(), "fabricated {key}");
        }
    }

    #[test]
    fn muse_generic_binding_rejects_wrong_geometry_and_exact_ids_even_with_ack() {
        let f = fixture("not-a-registry-entry", 2, 123);
        modify(&f, |m| {
            m["model"]["architecture"] = "muse-glimmer".into();
            m["model"]["exact_binding"] = json!({"gguf_content_blake3":"a".repeat(64), "tokenizer_metadata_id":"b".repeat(16)});
        });
        let data = VerifiedTransport::open(&f.0).unwrap();
        let content = "a".repeat(64);
        let tokenizer = "b".repeat(16);
        assert_eq!(
            data.bind("muse-glimmer", 3, 2, 32, &content, &tokenizer, true, false)
                .unwrap(),
            "exact_deployment_binding_matched"
        );
        for (arch, n, h, v, c, t) in [
            ("qwen35", 3, 2, 32, content.as_str(), tokenizer.as_str()),
            (
                "muse-glimmer",
                4,
                2,
                32,
                content.as_str(),
                tokenizer.as_str(),
            ),
            (
                "muse-glimmer",
                3,
                3,
                32,
                content.as_str(),
                tokenizer.as_str(),
            ),
            (
                "muse-glimmer",
                3,
                2,
                33,
                content.as_str(),
                tokenizer.as_str(),
            ),
            ("muse-glimmer", 3, 2, 32, "", tokenizer.as_str()),
            ("muse-glimmer", 3, 2, 32, content.as_str(), ""),
        ] {
            assert!(data.bind(arch, n, h, v, c, t, true, true).is_err());
        }
    }

    #[test]
    fn muse_generic_matrix_access_reorders_and_reverifies_retained_payload() {
        let f = fixture("future-fit", 2, 123);
        let mut access = GenericAccess {
            data: VerifiedTransport::open(&f.0).unwrap(),
            runtime_binding: Value::Null,
        };
        assert_eq!(
            access.data.read_matrix(0).unwrap(),
            access.data.read_matrix(2).unwrap()
        );
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(f.0.join("transport.f16le"))
            .unwrap();
        file.seek(SeekFrom::Start(8)).unwrap();
        file.write_all(&0u16.to_le_bytes()).unwrap();
        assert!(access.data.read_matrix(0).is_err());
        assert!(access.data.read_matrix(2).is_ok());
    }
}
