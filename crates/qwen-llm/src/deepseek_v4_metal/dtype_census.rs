//! Descriptor-only tensor-role dtype inventory and path-local admission helpers.

use crate::deepseek_v4_census::{DeepSeekV4CensusV1, DeepSeekV4TensorRole};
use crate::gguf::GgufFile;
use crate::tensor::GgmlType;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RoleDtype {
    pub role: DeepSeekV4TensorRole,
    pub dtype: GgmlType,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct QualifiedRoleDtypes {
    pub role: DeepSeekV4TensorRole,
    pub dtypes: &'static [GgmlType],
}

pub(super) fn role_dtype_census(gguf: &GgufFile) -> Result<Vec<RoleDtype>, String> {
    let census = DeepSeekV4CensusV1::from_gguf_flash_0731(gguf)
        .map_err(|error| format!("build DS4 role dtype census: {error}"))?;
    role_dtype_census_from_manifest(&census)
}

fn role_dtype_census_from_manifest(census: &DeepSeekV4CensusV1) -> Result<Vec<RoleDtype>, String> {
    let mut rows = Vec::new();
    for role_row in &census.roles {
        let role = DeepSeekV4TensorRole::all()
            .iter()
            .copied()
            .find(|role| role.as_str() == role_row.role)
            .ok_or_else(|| format!("unknown DS4 tensor role {:?}", role_row.role))?;
        for storage in &role_row.storage {
            let dtype = u32::try_from(storage.dtype_tag)
                .map(GgmlType::from_raw)
                .map_err(|_| {
                    format!(
                        "invalid dtype tag {} for role {}",
                        storage.dtype_tag, role_row.role
                    )
                })?;
            rows.push(RoleDtype { role, dtype });
        }
    }
    rows.sort_by_key(|row| (row.role, row.dtype as i32));
    rows.dedup();
    Ok(rows)
}

pub(super) fn role_dtype_declines(
    census: &[RoleDtype],
    path: &str,
    qualified: &[QualifiedRoleDtypes],
) -> Vec<String> {
    qualified
        .iter()
        .filter_map(|requirement| {
            let found = census
                .iter()
                .filter(|entry| entry.role == requirement.role)
                .map(|entry| entry.dtype)
                .collect::<Vec<_>>();
            let matches =
                !found.is_empty() && found.iter().all(|dtype| requirement.dtypes.contains(dtype));
            (!matches).then(|| {
                let found = if found.is_empty() {
                    "missing".to_string()
                } else {
                    found
                        .iter()
                        .map(|dtype| format!("{dtype:?}"))
                        .collect::<Vec<_>>()
                        .join(",")
                };
                let qualified = requirement
                    .dtypes
                    .iter()
                    .map(|dtype| format!("{dtype:?}"))
                    .collect::<Vec<_>>()
                    .join(",");
                format!(
                    "path={path} role={} found=[{found}] qualified=[{qualified}]",
                    requirement.role.as_str()
                )
            })
        })
        .collect()
}

pub(super) fn path_role_dtypes_qualified(
    census: &[RoleDtype],
    path: &str,
    qualified: &[QualifiedRoleDtypes],
) -> bool {
    let declines = role_dtype_declines(census, path, qualified);
    if !declines.is_empty() {
        static REPORTED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
            std::sync::OnceLock::new();
        let reported = REPORTED.get_or_init(Default::default);
        if let Ok(mut reported) = reported.lock() {
            for decline in declines {
                if reported.insert(decline.clone()) {
                    eprintln!("deepseek_v4: fast path dtype declined {decline}");
                }
            }
        }
        false
    } else {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPRESSOR_ROLES: &[QualifiedRoleDtypes] = &[
        QualifiedRoleDtypes {
            role: DeepSeekV4TensorRole::AttentionCompressorKv,
            dtypes: &[GgmlType::Q8_0],
        },
        QualifiedRoleDtypes {
            role: DeepSeekV4TensorRole::AttentionCompressorGate,
            dtypes: &[GgmlType::Q8_0],
        },
    ];

    #[test]
    fn consumed_roles_must_match_and_decline_names_path_role_and_dtype() {
        let census = [
            RoleDtype {
                role: DeepSeekV4TensorRole::AttentionCompressorKv,
                dtype: GgmlType::Q8_0,
            },
            RoleDtype {
                role: DeepSeekV4TensorRole::AttentionCompressorGate,
                dtype: GgmlType::F16,
            },
        ];
        let declines =
            role_dtype_declines(&census, "packed_q8_compressor_matrix", COMPRESSOR_ROLES);
        assert_eq!(declines.len(), 1);
        assert!(declines[0].contains("path=packed_q8_compressor_matrix"));
        assert!(declines[0].contains("role=attention_compressor_gate"));
        assert!(declines[0].contains("found=[F16]"));
    }

    #[test]
    fn changes_to_unconsumed_roles_do_not_change_admission() {
        let before = [
            RoleDtype {
                role: DeepSeekV4TensorRole::AttentionCompressorKv,
                dtype: GgmlType::Q8_0,
            },
            RoleDtype {
                role: DeepSeekV4TensorRole::AttentionCompressorGate,
                dtype: GgmlType::Q8_0,
            },
        ];
        let after = [
            before[0],
            before[1],
            RoleDtype {
                role: DeepSeekV4TensorRole::TokenEmbedding,
                dtype: GgmlType::F32,
            },
        ];
        assert!(path_role_dtypes_qualified(
            &before,
            "packed_q8_compressor_matrix",
            COMPRESSOR_ROLES
        ));
        assert!(path_role_dtypes_qualified(
            &after,
            "packed_q8_compressor_matrix",
            COMPRESSOR_ROLES
        ));
    }

    #[test]
    fn manifest_storage_byte_changes_do_not_change_role_dtypes() {
        let fixture: crate::deepseek_v4_census::PinnedDeepSeekV4AssetV1 =
            serde_json::from_str(include_str!(
                "../../tests/fixtures/deepseek_v4_flash_0731_ud_iq3_xxs_current_2026_08_04_census_v1.json"
            ))
            .unwrap();
        let before = role_dtype_census_from_manifest(&fixture.census).unwrap();
        let mut changed = fixture.census;
        for role in &mut changed.roles {
            for storage in &mut role.storage {
                storage.storage_bytes += 1;
            }
        }
        changed.totals.tensor_bytes += 1;
        let after = role_dtype_census_from_manifest(&changed).unwrap();
        assert_eq!(before, after);
    }
}
