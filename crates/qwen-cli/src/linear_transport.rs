//! Strict data-only transport opener; qualification is producer metadata only.
use anyhow::{Result, ensure};
use serde::{
    Deserialize, Serialize,
    de::{self, MapAccess, SeqAccess, Visitor},
};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fmt,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

#[derive(Debug, clap::Args)]
pub struct VerifyFullArgs {
    #[arg(long)]
    pub full_lens: PathBuf,
}
macro_rules! structure {
    ($name:ident { $($field:ident: $ty:ty),* $(,)? }) => {
        #[derive(Clone, Debug, Deserialize, Serialize)]
        #[serde(deny_unknown_fields)]
        pub struct $name { $(pub $field: $ty),* }
    };
}
structure!(Manifest {
    schema: String, schema_version: u32, status: String,
    transport: Transport, model: Model, payload: Payload,
    provenance: serde_json::Map<String, serde_json::Value>,
    qualification: serde_json::Map<String, serde_json::Value>,
});
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Transport {
    pub operator: String,
    pub method: String,
    pub source_layers: Vec<u32>,
    pub target_layer: u32,
    pub orientation: String,
    pub bias: String,
    pub output: String,
    #[serde(default)]
    pub identity_layers: Vec<u32>,
}
structure!(Descriptor {
    id: String,
    revision: String
});
structure!(ExactBinding {
    gguf_content_blake3: String,
    tokenizer_metadata_id: String
});
structure!(Model {
    architecture: String, n_layers: u32, hidden_size: u32, vocab_size: u32,
    source_checkpoint: Option<Descriptor>, source_tokenizer: Option<Descriptor>,
    exact_binding: Option<ExactBinding>,
});
structure!(Payload {
    path: String, dtype: String, shape: [u64; 3], byte_length: u64,
    sha256: String, matrix_sha256: Vec<String>,
});

// Value's normal decoder silently replaces duplicate keys, including metadata.
struct Unique(serde_json::Value);
impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Unique;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("JSON without duplicate keys")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Unique, A::Error> {
                let mut m = serde_json::Map::new();
                while let Some((k, v)) = a.next_entry::<String, Unique>()? {
                    if m.insert(k, v.0).is_some() {
                        return Err(de::Error::custom("duplicate JSON key"));
                    }
                }
                Ok(Unique(m.into()))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Unique, A::Error> {
                let mut v = Vec::new();
                while let Some(x) = a.next_element::<Unique>()? {
                    v.push(x.0);
                }
                Ok(Unique(v.into()))
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_unit<E: de::Error>(self) -> Result<Unique, E> {
                Ok(Unique(serde_json::Value::Null))
            }
        }
        d.deserialize_any(V)
    }
}
fn hex_digest(s: &str, length: usize) -> bool {
    s.len() == length
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub struct VerifiedTransport {
    manifest: Manifest,
    original: serde_json::Value,
    file: File,
    matrix_bytes: u64,
    payload_blake3: String,
}

/// Runtime-owned geometry/site policy, checked before opening or scanning payloads.
pub struct ExpectedProfile<'a> {
    pub architecture: &'a str,
    pub n_layers: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub target_layer: u32,
}

impl VerifiedTransport {
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    pub fn original_manifest(&self) -> &serde_json::Value {
        &self.original
    }
    pub fn payload_blake3(&self) -> &str {
        &self.payload_blake3
    }
    pub fn open(directory: &Path) -> Result<Self> {
        Self::open_inner(directory, None)
    }

    pub fn open_with_expected_profile(
        directory: &Path,
        profile: ExpectedProfile<'_>,
    ) -> Result<Self> {
        Self::open_inner(directory, Some(profile))
    }

    fn open_inner(directory: &Path, profile: Option<ExpectedProfile<'_>>) -> Result<Self> {
        for component in directory.ancestors().filter(|p| !p.as_os_str().is_empty()) {
            ensure!(
                !std::fs::symlink_metadata(component)?
                    .file_type()
                    .is_symlink(),
                "symlink artifact directory"
            );
        }
        let (mut file, length) = super::open_regular_file(&directory.join("lens.json"))?;
        ensure!(length <= 1024 * 1024, "manifest exceeds 1 MiB limit");
        let mut bytes = Vec::new();
        (&mut file).take(1024 * 1024 + 1).read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= 1024 * 1024, "manifest grew beyond limit");
        let original = serde_json::from_slice::<Unique>(&bytes)?.0;
        for key in ["exact_binding", "source_checkpoint", "source_tokenizer"] {
            ensure!(
                original
                    .get("model")
                    .and_then(|m| m.get(key))
                    .is_none_or(serde_json::Value::is_object),
                "model.{key} must be an object when present"
            );
        }
        let m: Manifest = serde_json::from_value(original.clone())?;
        ensure!(
            m.schema == "llm.lens.linear_transport"
                && m.schema_version == 1
                && m.status == "complete",
            "unsupported contract or incomplete artifact"
        );
        let t = &m.transport;
        ensure!(
            t.operator == "post_block_linear"
                && t.orientation == "target_source"
                && t.bias == "none"
                && t.output == "deployed_native",
            "unsupported transport operation"
        );
        ensure!(
            !t.method.trim().is_empty() && t.method.len() <= 1024,
            "invalid method label"
        );
        ensure!(
            !m.model.architecture.is_empty() && m.model.architecture.len() <= 128,
            "invalid architecture"
        );
        ensure!(
            (1..=4096).contains(&m.model.n_layers)
                && (1..=65536).contains(&m.model.hidden_size)
                && (1..=4194304).contains(&m.model.vocab_size),
            "model dimensions exceed bounds"
        );
        if let Some(profile) = profile {
            ensure!(
                m.model.architecture == profile.architecture
                    && m.model.n_layers == profile.n_layers
                    && m.model.hidden_size == profile.hidden_size
                    && m.model.vocab_size == profile.vocab_size
                    && t.target_layer == profile.target_layer,
                "transport manifest does not match expected runtime profile/target layer"
            );
        }
        let sources: BTreeSet<_> = t.source_layers.iter().copied().collect();
        ensure!(
            !sources.is_empty()
                && sources.len() == t.source_layers.len()
                && sources.iter().all(|&l| l < m.model.n_layers)
                && t.target_layer < m.model.n_layers,
            "invalid source/target layers"
        );
        let identities: BTreeSet<_> = t.identity_layers.iter().copied().collect();
        ensure!(
            identities.len() == t.identity_layers.len() && identities.is_subset(&sources),
            "invalid identity layers"
        );
        let h = u64::from(m.model.hidden_size);
        let matrix_bytes = h
            .checked_mul(h)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("matrix overflow"))?;
        let total = matrix_bytes
            .checked_mul(sources.len() as u64)
            .ok_or_else(|| anyhow::anyhow!("payload overflow"))?;
        ensure!(
            total <= 1024 * 1024 * 1024 * 1024,
            "payload exceeds 1 TiB limit"
        );
        let p = &m.payload;
        ensure!(
            p.path == "transport.f16le"
                && p.dtype == "f16_le"
                && p.shape == [sources.len() as u64, h, h]
                && p.byte_length == total,
            "invalid payload layout"
        );
        ensure!(
            hex_digest(&p.sha256, 64)
                && p.matrix_sha256.len() == sources.len()
                && p.matrix_sha256.iter().all(|s| hex_digest(s, 64)),
            "invalid SHA256 list"
        );
        if let Some(b) = &m.model.exact_binding {
            ensure!(
                hex_digest(&b.gguf_content_blake3, 64) && hex_digest(&b.tokenizer_metadata_id, 16),
                "invalid exact binding"
            );
        }
        let (file, length) = super::open_regular_file(&directory.join("transport.f16le"))?;
        ensure!(length as u64 == total, "payload length mismatch");
        let mut result = Self {
            manifest: m,
            original,
            file,
            matrix_bytes,
            payload_blake3: String::new(),
        };
        let mut sha = Sha256::new();
        let mut blake = blake3::Hasher::new();
        for i in 0..sources.len() {
            result.scan(i, |chunk| {
                sha.update(chunk);
                blake.update(chunk);
            })?;
        }
        ensure!(
            format!("{:x}", sha.finalize()) == result.manifest.payload.sha256,
            "whole payload SHA256 mismatch"
        );
        result.payload_blake3 = blake.finalize().to_hex().to_string();
        Ok(result)
    }
    fn scan(&mut self, index: usize, mut consume: impl FnMut(&[u8])) -> Result<()> {
        ensure!(
            self.file.metadata()?.len() == self.manifest.payload.byte_length,
            "payload length changed"
        );
        self.file
            .seek(SeekFrom::Start(index as u64 * self.matrix_bytes))?;
        let identity = self
            .manifest
            .transport
            .identity_layers
            .contains(&self.manifest.transport.source_layers[index]);
        let h = u64::from(self.manifest.model.hidden_size);
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut offset = 0;
        let mut sha = Sha256::new();
        while offset < self.matrix_bytes {
            let n = (self.matrix_bytes - offset).min(buffer.len() as u64) as usize;
            let chunk = &mut buffer[..n];
            self.file.read_exact(chunk)?;
            for (i, pair) in chunk.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([pair[0], pair[1]]);
                ensure!(bits & 0x7c00 != 0x7c00, "nonfinite F16 coefficient");
                if identity {
                    let cell = offset / 2 + i as u64;
                    ensure!(
                        bits == if cell / h == cell % h { 0x3c00 } else { 0 },
                        "declared identity matrix is not exact"
                    );
                }
            }
            sha.update(&*chunk);
            consume(chunk);
            offset += n as u64;
        }
        ensure!(
            format!("{:x}", sha.finalize()) == self.manifest.payload.matrix_sha256[index],
            "matrix SHA256 mismatch"
        );
        Ok(())
    }
    /// Bytes are returned only after rehashing; partial scan output never executes.
    pub fn read_matrix(&mut self, layer: u32) -> Result<Vec<u8>> {
        let index = self
            .manifest
            .transport
            .source_layers
            .iter()
            .position(|&l| l == layer)
            .ok_or_else(|| anyhow::anyhow!("source layer absent"))?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(usize::try_from(self.matrix_bytes)?)?;
        self.scan(index, |chunk| bytes.extend_from_slice(chunk))?;
        Ok(bytes)
    }
    pub fn bind(
        &self,
        architecture: &str,
        n_layers: u32,
        hidden_size: u32,
        vocab_size: u32,
        gguf_content_blake3: &str,
        tokenizer_metadata_id: &str,
        native_output_supported: bool,
        allow_unvalidated_transfer: bool,
    ) -> Result<&'static str> {
        let m = &self.manifest.model;
        ensure!(
            native_output_supported,
            "runtime lacks native capture/transport/output capability"
        );
        ensure!(
            m.architecture == architecture
                && m.n_layers == n_layers
                && m.hidden_size == hidden_size
                && m.vocab_size == vocab_size,
            "deployment geometry mismatch"
        );
        if let Some(b) = &m.exact_binding {
            ensure!(
                b.gguf_content_blake3 == gguf_content_blake3
                    && b.tokenizer_metadata_id == tokenizer_metadata_id,
                "deployment exact binding mismatch"
            );
            Ok("exact_deployment_binding_matched")
        } else {
            ensure!(
                allow_unvalidated_transfer,
                "source/deployment equivalence unverified; require --allow-unvalidated-transfer"
            );
            Ok("source_deployment_equivalence_unverified")
        }
    }
}
pub fn verify_full(args: VerifyFullArgs) -> Result<()> {
    let artifact = VerifiedTransport::open(&args.full_lens)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "contract": "llm.lens.linear_transport", "schema_version": 1,
            "integrity": "verified", "runtime_binding": "not_checked",
            "qualification": "producer_claims_only", "payload_blake3": artifact.payload_blake3(),
            "manifest": artifact.original_manifest(),
        }))?
    );
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;
    pub(crate) struct Fixture(pub(crate) PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    pub(crate) fn fixture(method: &str, h: usize, seed: u16) -> Fixture {
        let dir = std::env::temp_dir().join(format!(
            "linear-contract-{}-{}-{}",
            std::process::id(),
            h,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let matrix: Vec<u8> = (0..h * h)
            .flat_map(|i| (seed + i as u16).to_le_bytes())
            .collect();
        let payload = [matrix.clone(), matrix.clone()].concat();
        let sha = |b: &[u8]| format!("{:x}", Sha256::digest(b));
        let manifest = serde_json::json!({
            "schema":"llm.lens.linear_transport", "schema_version":1, "status":"complete",
            "transport":{"operator":"post_block_linear", "method":method, "source_layers":[2,0], "target_layer":1, "orientation":"target_source", "bias":"none", "output":"deployed_native"},
            "model":{"architecture":"qwen35", "n_layers":3, "hidden_size":h, "vocab_size":32},
            "payload":{"path":"transport.f16le", "dtype":"f16_le", "shape":[2,h,h], "byte_length":payload.len(), "sha256":sha(&payload), "matrix_sha256":[sha(&matrix),sha(&matrix)]},
            "provenance":{"seed":seed}, "qualification":{"validated":true}
        });
        std::fs::write(
            dir.join("lens.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join("transport.f16le"), payload).unwrap();
        Fixture(dir.canonicalize().unwrap())
    }
    pub(crate) fn modify(f: &Fixture, edit: impl FnOnce(&mut serde_json::Value)) {
        let path = f.0.join("lens.json");
        let mut value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        edit(&mut value);
        std::fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
    }
    #[test]
    fn expected_runtime_profile_rejects_before_payload_access() {
        let f = fixture("profile", 2, 123);
        let profile = || ExpectedProfile {
            architecture: "qwen35",
            n_layers: 3,
            hidden_size: 2,
            vocab_size: 32,
            target_layer: 1,
        };
        assert!(VerifiedTransport::open_with_expected_profile(&f.0, profile()).is_ok());
        std::fs::remove_file(f.0.join("transport.f16le")).unwrap();
        for wrong in [
            ExpectedProfile {
                architecture: "k2-horizon",
                ..profile()
            },
            ExpectedProfile {
                n_layers: 36,
                ..profile()
            },
            ExpectedProfile {
                hidden_size: 4096,
                ..profile()
            },
            ExpectedProfile {
                vocab_size: 250624,
                ..profile()
            },
            ExpectedProfile {
                target_layer: 2,
                ..profile()
            },
        ] {
            let error = VerifiedTransport::open_with_expected_profile(&f.0, wrong)
                .err()
                .unwrap();
            assert!(
                error
                    .to_string()
                    .contains("expected runtime profile/target layer")
            );
        }
    }

    #[test]
    fn unknown_methods_dynamic_geometry_and_claims() {
        for (method, h, seed) in [("independent-recipe-a", 2, 123), ("future-fit-b", 5, 512)] {
            let f = fixture(method, h, seed);
            let mut a = VerifiedTransport::open(&f.0).unwrap();
            assert_eq!(a.read_matrix(0).unwrap().len(), h * h * 2);
            assert!(
                a.bind("qwen35", 3, h as u32, 32, "", "", true, false)
                    .is_err()
            );
            assert_eq!(
                a.bind("qwen35", 3, h as u32, 32, "", "", true, true)
                    .unwrap(),
                "source_deployment_equivalence_unverified"
            );
            assert!(
                a.bind("qwen35", 3, h as u32 + 1, 32, "", "", true, true)
                    .is_err()
            );
            assert!(
                a.bind("muse_glimmer", 3, h as u32, 32, "", "", true, true)
                    .is_err()
            );
        }
    }
    #[test]
    fn corrupt_unselected_and_mutate_after_open() {
        let f = fixture("mutation", 2, 12);
        let mut a = VerifiedTransport::open(&f.0).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(f.0.join("transport.f16le"))
            .unwrap();
        file.seek(SeekFrom::Start(8)).unwrap();
        file.write_all(&[0, 0]).unwrap();
        assert!(a.read_matrix(0).is_err());
        assert!(VerifiedTransport::open(&f.0).is_err());
    }

    #[test]
    fn retained_handle_does_not_follow_payload_replacement() {
        let f = fixture("retained-file", 2, 77);
        let mut artifact = VerifiedTransport::open(&f.0).unwrap();
        let expected = artifact.read_matrix(2).unwrap();
        std::fs::rename(f.0.join("transport.f16le"), f.0.join("original.f16le")).unwrap();
        std::fs::write(f.0.join("transport.f16le"), [0u8; 16]).unwrap();
        assert_eq!(artifact.read_matrix(2).unwrap(), expected);
        assert!(VerifiedTransport::open(&f.0).is_err());
    }

    #[test]
    fn source_descriptors_never_authorize_transfer_or_native_capability() {
        let f = fixture("unregistered-source", 2, 19);
        modify(&f, |v| {
            for key in ["source_checkpoint", "source_tokenizer"] {
                v["model"][key] = serde_json::json!({"id":"matching-producer-source", "revision":"claimed-release"});
            }
        });
        let artifact = VerifiedTransport::open(&f.0).unwrap();
        assert!(
            artifact
                .bind("qwen35", 3, 2, 32, "", "", true, false)
                .is_err()
        );
        assert!(
            artifact
                .bind("qwen35", 3, 2, 32, "", "", false, true)
                .is_err()
        );
    }
    #[test]
    fn exact_binding_override_never_bypasses() {
        let f = fixture("binding", 2, 17);
        modify(
            &f,
            |v| v["model"]["exact_binding"] = serde_json::json!({"gguf_content_blake3":"a".repeat(64),"tokenizer_metadata_id":"b".repeat(16)}),
        );
        let a = VerifiedTransport::open(&f.0).unwrap();
        assert!(
            a.bind(
                "qwen35",
                3,
                2,
                32,
                &"a".repeat(64),
                &"b".repeat(16),
                true,
                false
            )
            .is_ok()
        );
        assert!(
            a.bind(
                "qwen35",
                3,
                2,
                32,
                &"a".repeat(64),
                &"c".repeat(16),
                true,
                true
            )
            .is_err()
        );
        assert!(
            a.bind(
                "qwen35",
                3,
                2,
                32,
                &"c".repeat(64),
                &"b".repeat(16),
                true,
                true
            )
            .is_err()
        );
    }
    #[test]
    fn duplicate_metadata_unknown_fields_traversal_and_symlink() {
        assert!(serde_json::from_str::<Unique>(r#"{"qualification":{"x":1,"x":2}}"#).is_err());
        let f = fixture("paths", 2, 1);
        modify(&f, |v| v["payload"]["path"] = "../transport.f16le".into());
        assert!(VerifiedTransport::open(&f.0).is_err());
        modify(&f, |v| {
            v["payload"]["path"] = "transport.f16le".into();
            v["model"]["surprise"] = true.into();
        });
        assert!(VerifiedTransport::open(&f.0).is_err());
        modify(&f, |v| {
            v["model"].as_object_mut().unwrap().remove("surprise");
        });
        std::fs::rename(f.0.join("transport.f16le"), f.0.join("real")).unwrap();
        std::os::unix::fs::symlink("real", f.0.join("transport.f16le")).unwrap();
        assert!(VerifiedTransport::open(&f.0).is_err());
    }

    fn replace_payload(f: &Fixture, payload: &[u8]) {
        std::fs::write(f.0.join("transport.f16le"), payload).unwrap();
        modify(f, |v| {
            v["payload"]["sha256"] = format!("{:x}", Sha256::digest(payload)).into();
            v["payload"]["matrix_sha256"] = payload
                .chunks_exact(payload.len() / 2)
                .map(|matrix| serde_json::Value::from(format!("{:x}", Sha256::digest(matrix))))
                .collect::<Vec<_>>()
                .into();
        });
    }

    #[test]
    fn authenticated_nonfinite_unused_matrix_and_false_identity_fail() {
        let f = fixture("finite-and-identity", 2, 1);
        let mut bytes = std::fs::read(f.0.join("transport.f16le")).unwrap();
        bytes[..8].copy_from_slice(&[0, 0x3c, 0, 0, 0, 0, 0, 0x3c]);
        replace_payload(&f, &bytes);
        modify(&f, |v| {
            v["transport"]["identity_layers"] = serde_json::json!([2])
        });
        VerifiedTransport::open(&f.0).unwrap();
        bytes[1] = 0;
        replace_payload(&f, &bytes);
        assert!(
            VerifiedTransport::open(&f.0)
                .err()
                .unwrap()
                .to_string()
                .contains("identity")
        );
        bytes[1] = 0x3c;
        bytes[9] = 0x7c;
        replace_payload(&f, &bytes);
        assert!(
            VerifiedTransport::open(&f.0)
                .err()
                .unwrap()
                .to_string()
                .contains("nonfinite")
        );
    }

    #[test]
    fn strict_bounds_binding_presence_and_duplicate_metadata_are_not_claim_overrides() {
        for edit in [
            ("hidden_size", serde_json::json!(0)),
            ("hidden_size", serde_json::json!(65537)),
            ("n_layers", serde_json::json!(0)),
            ("vocab_size", serde_json::json!(0)),
            ("exact_binding", serde_json::Value::Null),
        ] {
            let f = fixture("bounds", 2, 1);
            modify(&f, |v| v["model"][edit.0] = edit.1);
            assert!(VerifiedTransport::open(&f.0).is_err());
        }
        for (key, value) in [
            ("target_layer", serde_json::json!(3)),
            ("source_layers", serde_json::json!([2, 2])),
            ("identity_layers", serde_json::json!([1])),
        ] {
            let f = fixture("layers", 2, 1);
            modify(&f, |v| v["transport"][key] = value);
            assert!(VerifiedTransport::open(&f.0).is_err());
        }
        let f = fixture("duplicates", 2, 1);
        let path = f.0.join("lens.json");
        let text = std::fs::read_to_string(&path).unwrap().replace(
            "\"validated\":true",
            "\"validated\":true,\"validated\":false",
        );
        std::fs::write(path, text).unwrap();
        assert!(VerifiedTransport::open(&f.0).is_err());
        std::os::unix::fs::symlink(".", f.0.join("linked")).unwrap();
        assert!(VerifiedTransport::open(&f.0.join("linked")).is_err());
    }
}
