use super::muse_full_lens_artifact::MatrixDescriptor;
use super::muse_lens_artifact;
use super::published_pt::{ArchiveSpec, ExtractedPayload};
use anyhow::{Context, Result, ensure};
use qwen_llm::muse_glimmer::{ARCHITECTURE_NAME, MuseGlimmerConfig};
use serde::{Deserialize, Serialize};

pub(crate) const SCHEMA: &str = "muse_glimmer.published_full_transport";
pub(crate) const SCHEMA_VERSION: u32 = 1;
pub(crate) const MANIFEST_NAME: &str = "lens.json";
pub(crate) const PAYLOAD_NAME: &str = "transport.f16le";
pub(crate) const PROFILE_NAME: &str = "eyes_ml_muse_glimmer_30b_j_v1";

const SOURCE_REPOSITORY: &str = "eyes-ml/Muse-Glimmer-30B_jacobian-lens";
const SOURCE_REVISION: &str = "71d8434fbd38c8b5d70e1ff1ff2095d5da926c34";
const SOURCE_FILENAME: &str = "Muse-Glimmer-30B_jacobian_lens.pt";
const SOURCE_BYTES: u64 = 4_518_854_369;
const SOURCE_SHA256: &str = "397dc00807a8f72d9d21feaeedeba7fb9e7c50e5c80efc317f37e8cd9c833300";
const DATA_PICKLE_SHA256: &str = "f187e6221a2c5540769e3af89d4e0ea81159a1c98a7a1d405c22384f7bd605a7";
const SERIALIZATION_ID: &str = "1371680345541892666311410627619629120835";
const ARCHIVE_ROOT: &str = "Muse-Glimmer-30B_jacobian_lens";
const PAYLOAD_BLAKE3: &str = "64f50f387a56a4533631e62789a896e759f0ebbdb3a9fdbae45898a5dc8a2794";
const HIDDEN_SIZE: usize = 6_656;
const SOURCE_LAYER_COUNT: usize = 51;
const TARGET_LAYER: u32 = 51;
const MATRIX_BYTES: u64 = HIDDEN_SIZE as u64 * HIDDEN_SIZE as u64 * 2;
const PAYLOAD_BYTES: u64 = MATRIX_BYTES * SOURCE_LAYER_COUNT as u64;

const MATRIX_BLAKE3: [&str; SOURCE_LAYER_COUNT] = [
    "fa2be6849e83601a404dd6f059e55c2fc716ad681537d3ed7c586fb2fbc4b89c",
    "e62902ac559c99c97a5e6f9f4d7737eb2e44ac282b8ae7e78c7bdd097e1bbe22",
    "b2909e16eb89c3a7953942f9194ef0bb1c7b42b6fbbce2fb7f772a95d0917aea",
    "7da82020557cd9b050c6777ec5f79324423e5d63e8525385c3378d5447d89ea2",
    "d8f85c10d9988f43023b8efe81242b429e9f4c4fda79eb21cadb49dd9c070d0c",
    "f089f65149aff08b6c26e7d7a8b617baebfe0f9212b84f14ac3a4d5ddeabb865",
    "98bdd514b651e7c6f2c2acb02d55e48359d792cc4aa4efc4036b9edbc17d9531",
    "1bbfe43e6a2da8527710b9b850cdae24ca1fb18f6d129f6be4799bd766d3af39",
    "8400445201c043bdb08a95bba3f0254da7155c3b9fa57c43d4a22cf216130cb6",
    "8d7487f34fe099b0c8c1e5877aeb3a73661648e4e2d3f8aab57d074c851190ce",
    "82367272d1b5706b0d7f2251d061a04f35dc303b0ff0494a71118fa76f86600c",
    "45096e86ce9edaff80f68e55f7419e808f576a9631767ae1e14191361641860d",
    "68fed655ec9616adbf171bcbe251661b05a1a1d878fef8390c3387199fe39906",
    "a95ed92955e2490e4660653bde31a19ccc7afe174e5b43cb4fab2ff14fc98b9f",
    "dc43f3fa355dbf65543e3a04e218a7effd05713fd5107dd0b53fd11f41ab5149",
    "171cb9b18b2d5e969e85eaa28970f8f08a7945987c2b5bc699dc726b07f372e3",
    "64ef67d57084615ff9a18179869166a97ff34865efd2b3f0d8212a5c847cd01f",
    "e2ada606a901bc3aa42c26aa74a3fe424663f3574512a2ea2fd57dac19ca35d4",
    "6f06e143023feaafcf6fca7ba11918d8efcb65fb202b9ea4c56ca018f8a4967d",
    "7ae96a62ea8c3e3bed4abfa47ba3f93364ed6fe88728b2dfe02372e2298d7856",
    "1ae281bc8146eff33c5849dd01ab96cf18a13b9ba4f24dfba6e142541a321f17",
    "65044e6625383f23514d7bd188f69a0ff52db2f7dca2a0ddece96382f00af49b",
    "fd864a577466ac8ed8a975e6a52f4286ed675910ab1c72006b80388bbcabb76a",
    "d83a36e7f5e3d5bd3e050c1686a3ac4058041756d708e373fccd6121e97f3811",
    "6709d82cbf805557613f603bccc84af2a0bcedce85d83223136debfc3bc0ff96",
    "7701f2097dbefebcb3758007f907fad067e735d12a9841e704a8f47ea8b5d687",
    "1a8673b3b7214cbab246bfccb27a9611077f6d165926c79ef82a245732ec91ec",
    "99a144a3d2c20d7312016daf2f897abcd4a634d97c8dadb879fff2da29cb78c2",
    "014a14bb0ea0dc7903f5869b7cbb6b053c3b33368ed525d1deebec36a2177790",
    "9b37e96c9ba12a3ede4ef0df3b5f5d95c2e1353614772aa60641439d259c481d",
    "1ee9f7e678f352c8f138acf94a5cd8377faa3370cb328f42b070d6d32b97ca56",
    "ea372ede98df1899c45097460dc1fc32d3a76a5ce077318a6bca568962e7068b",
    "4dbd6cf5afe4229ac41ba88423b2ecf040a737cff015b90b9af89f4f7aa99966",
    "d260ea7f81e987d3a0631784351a972c138694accac1148bcb8f1adf52c1acf8",
    "a618dbbcc58066c0c4ac6be4adf191a1ea928330bdc9d9de42895d9a4634d52d",
    "853946be4070a949e3ddd64467a0f656c33ae49c5c117f0eb714cf9f7176e613",
    "f10c4fbfc608dd3563d9bdc94c28eb6547ed95c3f2c1e97d6d178cb7a6921907",
    "2a9afcaf27bcfc4389f46e36572f0147c0686081abd9e244f785073dcd38078a",
    "0443e63b4750a00715c815bce41a24c0a25c6e53d7d2e480e0ec26946fe37d80",
    "b8fd28883f1ae6d031878574a322e834369e6af1836c8e63f5f0ae76ff5c55ae",
    "aee6279d6d50a460179ab0ec8c968c0c4f07a7c0f67e4f5a4973b258b4a88eac",
    "afd8141e96c77f50d4ae348cd31f888e6cd924d987c106c529d763dc2ab17862",
    "4adef04732bf1273151bce460f055d6010eeef9656a13e005c944ba483e84abc",
    "03d4dab334f3b2f65db0dc84090b21bce113d4a680afcb8801585c708193d05f",
    "ce51bed2f5bdbd39130280c4f894421e5438ac74d88ae0dddb894c40f3983405",
    "5b1e3c045861b6e48b5c2bd5362c264185df0e584fcf225cac5e464e36fed2f2",
    "789a7ecf6bc1edd5305a889f9d53d13b6abda4fd4e12fcb3e5f464aaa676b53d",
    "466750a526bf395f5bd3904a21509b54297aee5da713390be057fd6cb5e4d521",
    "2942d708884d6ccf5f1d38744a1ff988cbcbfe23b286379d0cc6a067f2115157",
    "2cc225637559772b512f9d09fc227049d8e6d376e15c0991bbed47ccb3d23fe7",
    "ca84caa1dbcf0af8bdf9242b56ebb2eba548201cf60b8fad460c5d88b9e36027",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProfileId {
    EyesMlMuseGlimmer30bJ,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Profile {
    pub(crate) id: ProfileId,
}

const PROFILES: [Profile; 1] = [Profile {
    id: ProfileId::EyesMlMuseGlimmer30bJ,
}];

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub(crate) schema: String,
    pub(crate) schema_version: u32,
    pub(crate) status: String,
    pub(crate) profile: String,
    pub(crate) transport: Transport,
    pub(crate) model: Model,
    pub(crate) fit: Fit,
    pub(crate) source: Source,
    pub(crate) payload: Payload,
    pub(crate) transfer: Transfer,
    pub(crate) provenance: Provenance,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Transport {
    pub(crate) method: String,
    pub(crate) rule_contract: String,
    pub(crate) target_layer: u32,
    pub(crate) source_layers: Vec<u32>,
    pub(crate) coordinate: String,
    pub(crate) orientation: String,
    pub(crate) storage_dtype: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Model {
    pub(crate) architecture: String,
    pub(crate) geometry: muse_lens_artifact::Geometry,
    pub(crate) base_model: String,
    pub(crate) fitted_checkpoint: String,
    pub(crate) fitted_checkpoint_revision: String,
    pub(crate) output_rmsnorm_epsilon: f64,
    pub(crate) output_multiplier: f64,
    pub(crate) final_logit_softcap: f64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Fit {
    pub(crate) claims_basis: String,
    pub(crate) embedded_provenance: bool,
    pub(crate) fitter: String,
    pub(crate) fitter_revision: String,
    pub(crate) transformers_revision: String,
    pub(crate) dataset: String,
    pub(crate) split: String,
    pub(crate) corpus_preparation: String,
    pub(crate) n_prompts: u64,
    pub(crate) max_sequence_length: u32,
    pub(crate) skip_first: u32,
    pub(crate) valid_positions_per_prompt: u32,
    pub(crate) dim_batch: u32,
    pub(crate) model_execution_dtype: String,
    pub(crate) serialized_dtype: String,
    pub(crate) stop_rule: String,
    pub(crate) convergence_status: String,
    pub(crate) modality: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Source {
    pub(crate) repository: String,
    pub(crate) revision: String,
    pub(crate) filename: String,
    pub(crate) byte_length: u64,
    pub(crate) sha256: String,
    pub(crate) data_pickle_sha256: String,
    pub(crate) serialization_id: String,
    pub(crate) license: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Payload {
    pub(crate) path: String,
    pub(crate) dtype: String,
    pub(crate) shape: [usize; 3],
    pub(crate) byte_length: u64,
    pub(crate) blake3: String,
    pub(crate) matrices: Vec<MatrixDescriptor>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Transfer {
    pub(crate) binding: String,
    pub(crate) deployed_checkpoint_policy: String,
    pub(crate) validation_status: String,
    pub(crate) image_token_status: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Provenance {
    pub(crate) build_commit: String,
    pub(crate) build_dirty: String,
    pub(crate) build_source_state: String,
    pub(crate) build_stamp_source: String,
    pub(crate) build_stamp_error: String,
    pub(crate) pickle_execution: String,
}

pub(crate) fn profile_for_source(byte_length: u64, sha256: &str) -> Option<Profile> {
    PROFILES
        .iter()
        .copied()
        .find(|profile| profile.source_bytes() == byte_length && profile.source_sha256() == sha256)
}

pub(crate) fn profile_for_manifest(manifest: &Manifest) -> Result<Profile> {
    PROFILES
        .iter()
        .copied()
        .find(|profile| {
            manifest.profile == profile.name()
                && manifest.source.repository == profile.source_repository()
                && manifest.source.revision == profile.source_revision()
                && manifest.source.filename == profile.source_filename()
                && manifest.source.byte_length == profile.source_bytes()
                && manifest.source.sha256 == profile.source_sha256()
        })
        .context("Muse published manifest does not identify a supported pinned profile")
}

impl Profile {
    pub(crate) const fn name(self) -> &'static str {
        match self.id {
            ProfileId::EyesMlMuseGlimmer30bJ => PROFILE_NAME,
        }
    }

    pub(crate) const fn source_repository(self) -> &'static str {
        SOURCE_REPOSITORY
    }

    pub(crate) const fn source_revision(self) -> &'static str {
        SOURCE_REVISION
    }

    pub(crate) const fn source_filename(self) -> &'static str {
        SOURCE_FILENAME
    }

    pub(crate) const fn source_bytes(self) -> u64 {
        SOURCE_BYTES
    }

    pub(crate) const fn source_sha256(self) -> &'static str {
        SOURCE_SHA256
    }

    pub(crate) const fn expected_payload_blake3(self) -> &'static str {
        PAYLOAD_BLAKE3
    }

    pub(crate) const fn archive_spec(self) -> ArchiveSpec<'static> {
        ArchiveSpec {
            root: ARCHIVE_ROOT,
            layer_count: SOURCE_LAYER_COUNT,
            hidden_size: HIDDEN_SIZE,
            matrix_bytes: MATRIX_BYTES,
            data_pickle_sha256: DATA_PICKLE_SHA256,
            serialization_id: Some(SERIALIZATION_ID),
            identity_storage_index: None,
        }
    }
}

pub(crate) fn payload_from_extracted(
    profile: Profile,
    extracted: ExtractedPayload,
) -> Result<Payload> {
    ensure!(
        extracted.byte_length == PAYLOAD_BYTES
            && extracted.blake3 == profile.expected_payload_blake3()
            && extracted.matrices.len() == SOURCE_LAYER_COUNT,
        "imported Muse published payload does not match the pinned profile"
    );
    let matrices = extracted
        .matrices
        .into_iter()
        .enumerate()
        .map(|(slot, matrix)| {
            ensure!(
                matrix.storage_index == slot
                    && matrix.byte_offset == slot as u64 * MATRIX_BYTES
                    && matrix.byte_length == MATRIX_BYTES
                    && matrix.blake3 == MATRIX_BLAKE3[slot],
                "imported Muse published matrix {slot} does not match the pinned profile"
            );
            Ok(MatrixDescriptor {
                source_layer: slot as u32,
                byte_offset: matrix.byte_offset,
                byte_length: matrix.byte_length,
                blake3: matrix.blake3,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Payload {
        path: PAYLOAD_NAME.into(),
        dtype: "f16_le".into(),
        shape: [SOURCE_LAYER_COUNT, HIDDEN_SIZE, HIDDEN_SIZE],
        byte_length: extracted.byte_length,
        blake3: extracted.blake3,
        matrices,
    })
}

pub(crate) fn canonical_manifest(profile: Profile, payload: Payload) -> Manifest {
    Manifest {
        schema: SCHEMA.into(),
        schema_version: SCHEMA_VERSION,
        status: "complete".into(),
        profile: profile.name().into(),
        transport: canonical_transport(profile),
        model: canonical_model(profile),
        fit: canonical_fit(profile),
        source: canonical_source(profile),
        payload,
        transfer: canonical_transfer(profile),
        provenance: Provenance {
            build_commit: env!("QWEN_BUILD_COMMIT").into(),
            build_dirty: env!("QWEN_BUILD_DIRTY").into(),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
            build_stamp_source: env!("QWEN_BUILD_STAMP_SOURCE").into(),
            build_stamp_error: env!("QWEN_BUILD_STAMP_ERROR").into(),
            pickle_execution: "none_fixed_schema_pinned_zip_entries_only".into(),
        },
    }
}

pub(crate) fn validate_manifest(manifest: &Manifest) -> Result<()> {
    ensure!(
        manifest.schema == SCHEMA
            && manifest.schema_version == SCHEMA_VERSION
            && manifest.status == "complete",
        "artifact is not a complete Muse published full transport"
    );
    let profile = profile_for_manifest(manifest)?;
    ensure!(
        manifest.transport == canonical_transport(profile)
            && manifest.model == canonical_model(profile)
            && manifest.fit == canonical_fit(profile)
            && manifest.source == canonical_source(profile)
            && manifest.transfer == canonical_transfer(profile),
        "Muse published full-transport claims are not canonical"
    );
    validate_payload(&manifest.payload)?;
    super::validate_token_build_identity(
        &manifest.provenance.build_source_state,
        &manifest.provenance.build_stamp_error,
    )?;
    ensure!(
        !manifest.provenance.build_commit.is_empty()
            && matches!(manifest.provenance.build_dirty.as_str(), "0" | "1")
            && !manifest.provenance.build_stamp_source.is_empty()
            && manifest.provenance.pickle_execution == "none_fixed_schema_pinned_zip_entries_only",
        "Muse published import provenance is incomplete or permits pickle execution"
    );
    Ok(())
}

fn validate_payload(payload: &Payload) -> Result<()> {
    ensure!(
        payload.path == PAYLOAD_NAME
            && payload.dtype == "f16_le"
            && payload.shape == [SOURCE_LAYER_COUNT, HIDDEN_SIZE, HIDDEN_SIZE]
            && payload.byte_length == PAYLOAD_BYTES
            && payload.blake3 == PAYLOAD_BLAKE3
            && payload.matrices.len() == SOURCE_LAYER_COUNT,
        "Muse published payload descriptor is not canonical"
    );
    for (slot, matrix) in payload.matrices.iter().enumerate() {
        ensure!(
            matrix.source_layer == slot as u32
                && matrix.byte_offset == slot as u64 * MATRIX_BYTES
                && matrix.byte_length == MATRIX_BYTES
                && matrix.blake3 == MATRIX_BLAKE3[slot],
            "Muse published matrix descriptor {slot} is not canonical"
        );
    }
    Ok(())
}

fn canonical_transport(_profile: Profile) -> Transport {
    Transport {
        method: "J".into(),
        rule_contract: "published_standard_jacobian_lens_v1".into(),
        target_layer: TARGET_LAYER,
        source_layers: (0..SOURCE_LAYER_COUNT as u32).collect(),
        coordinate: muse_lens_artifact::COORDINATE.into(),
        orientation: "source_layer_target_output_coordinate_source_coordinate".into(),
        storage_dtype: "f16_le".into(),
    }
}

fn canonical_model(_profile: Profile) -> Model {
    let reference = MuseGlimmerConfig::release_reference();
    Model {
        architecture: ARCHITECTURE_NAME.into(),
        geometry: muse_lens_artifact::geometry(&reference),
        base_model: "meta-models/Muse-Glimmer-30B".into(),
        fitted_checkpoint: "eyes-ml/Muse-Glimmer-30B".into(),
        fitted_checkpoint_revision: "97e6fe0a8d8d221b100cd67f53fccf0744950abf".into(),
        output_rmsnorm_epsilon: 1e-5,
        output_multiplier: 0.19611613513818404,
        final_logit_softcap: 20.0,
    }
}

fn canonical_fit(_profile: Profile) -> Fit {
    Fit {
        claims_basis: "pinned_repository_model_card_not_embedded_in_pt".into(),
        embedded_provenance: false,
        fitter: "neuronpedia_utils/jlens/fit_lens.py".into(),
        fitter_revision: "7724688596eb734a0662f911bf183151a5c66b2f".into(),
        transformers_revision: "a61d9a57c1ca1018fd84acabbf2104fdf468e143".into(),
        dataset: "Salesforce/wikitext:wikitext-103-raw-v1".into(),
        split: "train".into(),
        corpus_preparation: "streamed_rechunked_approximately_2000_char_prompts".into(),
        n_prompts: 900,
        max_sequence_length: 128,
        skip_first: 16,
        valid_positions_per_prompt: 111,
        dim_batch: 8,
        model_execution_dtype: "bfloat16".into(),
        serialized_dtype: "float16".into(),
        stop_rule: "smoothed_delta_mean_below_1e-3_after_at_least_100_prompts".into(),
        convergence_status: "not_reached_at_900_prompts_final_smoothed_delta_approximately_1.5e-3"
            .into(),
        modality: "text_only".into(),
    }
}

fn canonical_source(_profile: Profile) -> Source {
    Source {
        repository: SOURCE_REPOSITORY.into(),
        revision: SOURCE_REVISION.into(),
        filename: SOURCE_FILENAME.into(),
        byte_length: SOURCE_BYTES,
        sha256: SOURCE_SHA256.into(),
        data_pickle_sha256: DATA_PICKLE_SHA256.into(),
        serialization_id: SERIALIZATION_ID.into(),
        license: "Apache-2.0".into(),
    }
}

fn canonical_transfer(_profile: Profile) -> Transfer {
    Transfer {
        binding: "published_checkpoint_geometry_transfer".into(),
        deployed_checkpoint_policy:
            "supported_muse_release_geometry_and_output_contract_requires_explicit_acknowledgement"
                .into(),
        validation_status: "unvalidated".into(),
        image_token_status: "unvalidated_text_only_fit".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical_payload() -> Payload {
        Payload {
            path: PAYLOAD_NAME.into(),
            dtype: "f16_le".into(),
            shape: [SOURCE_LAYER_COUNT, HIDDEN_SIZE, HIDDEN_SIZE],
            byte_length: PAYLOAD_BYTES,
            blake3: PAYLOAD_BLAKE3.into(),
            matrices: MATRIX_BLAKE3
                .iter()
                .enumerate()
                .map(|(slot, digest)| MatrixDescriptor {
                    source_layer: slot as u32,
                    byte_offset: slot as u64 * MATRIX_BYTES,
                    byte_length: MATRIX_BYTES,
                    blake3: (*digest).into(),
                })
                .collect(),
        }
    }

    #[test]
    fn canonical_profile_binds_publication_geometry_and_digests() {
        let profile = PROFILES[0];
        let manifest = canonical_manifest(profile, canonical_payload());
        validate_manifest(&manifest).unwrap();
        assert_eq!(manifest.transport.method, "J");
        assert_eq!(manifest.transport.target_layer, 51);
        assert_eq!(
            manifest.transport.source_layers,
            (0..51).collect::<Vec<_>>()
        );
        assert_eq!(manifest.model.geometry.hidden_size, 6_656);
        assert_eq!(manifest.fit.n_prompts, 900);

        let mut changed = manifest.clone();
        changed.payload.matrices[17].blake3 = "00".repeat(32);
        assert!(validate_manifest(&changed).is_err());
    }

    #[test]
    fn extracted_payload_requires_every_pinned_matrix_digest() {
        let profile = PROFILES[0];
        let extracted = ExtractedPayload {
            byte_length: PAYLOAD_BYTES,
            blake3: PAYLOAD_BLAKE3.into(),
            matrices: MATRIX_BLAKE3
                .iter()
                .enumerate()
                .map(
                    |(slot, digest)| super::super::published_pt::ExtractedMatrix {
                        storage_index: slot,
                        byte_offset: slot as u64 * MATRIX_BYTES,
                        byte_length: MATRIX_BYTES,
                        blake3: (*digest).into(),
                    },
                )
                .collect(),
        };
        let payload = payload_from_extracted(profile, extracted.clone()).unwrap();
        assert_eq!(payload.blake3, PAYLOAD_BLAKE3);

        let mut changed = extracted;
        changed.matrices[50].blake3 = "11".repeat(32);
        assert!(payload_from_extracted(profile, changed).is_err());
    }
}
