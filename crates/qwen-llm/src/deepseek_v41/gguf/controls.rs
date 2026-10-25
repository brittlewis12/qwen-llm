use super::*;

const MAX_CONTROL_BYTES: u64 = 16 * 1024 * 1024;

/// CPU-only serialization work. These bytes are not Metal weight residency.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ControlAccounting {
    pub payload_bytes_read: u64,
    /// Conservative cold normalization bound, including raw and integer copies.
    pub cold_cpu_bytes: u64,
    /// Retained token map and hash vectors/config, excluding container headers.
    pub retained_cpu_bytes: u64,
    /// Source/table descriptors, stamp copies and small allocation slack.
    /// Excludes shared GGUF parser/header storage and virtual file mappings.
    pub source_bookkeeping_bytes: u64,
}

pub(super) fn cold_bound(g: &GgufFile) -> Result<u64, LoadError> {
    let mut bytes = 0u64;
    for t in g.tensors.iter().filter(|t| {
        matches!(
            t.name.as_str(),
            "engram_token_map.weight"
                | "engram_primes.weight"
                | "engram_offsets.weight"
                | "engram_multipliers.weight"
        )
    }) {
        if t.n_bytes > MAX_CONTROL_BYTES {
            return Err(invalid("Engram control exceeds bounded CPU staging"));
        }
        bytes = add(
            bytes,
            t.n_bytes
                .checked_mul(6)
                .ok_or_else(|| invalid("cold control bytes overflow"))?,
        )?;
    }
    Ok(bytes)
}

pub(super) fn merge(
    g: &GgufFile,
    name: &str,
    dtype: GgmlType,
    shape: &[u64],
    metadata: Option<Vec<u64>>,
    accounting: &mut ControlAccounting,
) -> Result<Option<Vec<u64>>, LoadError> {
    let mut matches = g.tensors.iter().filter(|t| t.name == name);
    let Some(t) = matches.next() else {
        return Ok(metadata);
    };
    if matches.next().is_some() {
        return Err(invalid(format!("duplicate Engram control {name}")));
    }
    if t.dtype != dtype || !shape_matches(&t.shape, shape) {
        return Err(invalid(format!(
            "invalid Engram control dtype/shape: {name}"
        )));
    }
    validate_storage(t)?;
    if t.n_bytes > MAX_CONTROL_BYTES {
        return Err(invalid("Engram control exceeds bounded CPU staging"));
    }
    let count = crate::tensor::checked_shape_elements(shape)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| invalid("Engram control count overflow"))?;
    let bytes = usize::try_from(t.n_bytes).map_err(|_| invalid("control bytes overflow"))?;
    let mut raw = Vec::new();
    raw.try_reserve_exact(bytes)
        .map_err(|e| invalid(format!("control allocation: {e}")))?;
    raw.resize(bytes, 0);
    g.read_shard_exact_at(t.shard_idx, t.data_offset, &mut raw)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|e| invalid(format!("control normalization allocation: {e}")))?;
    let width = if dtype == GgmlType::I32 { 4 } else { 8 };
    for chunk in raw.chunks_exact(width) {
        let value = if width == 4 {
            i64::from(i32::from_le_bytes(chunk.try_into().unwrap()))
        } else {
            i64::from_le_bytes(chunk.try_into().unwrap())
        };
        values.push(u64::try_from(value).map_err(|_| invalid(format!("negative control {name}")))?);
    }
    if metadata.as_ref().is_some_and(|old| old != &values) {
        return Err(invalid(format!(
            "conflicting metadata/tensor control {name}"
        )));
    }
    accounting.payload_bytes_read = add(accounting.payload_bytes_read, t.n_bytes)?;
    // Sum, rather than max, deliberately overbounds overlapping normalization
    // phases, including a possible inline u64 copy and the final u32 token map.
    let normalized = (count as u64)
        .checked_mul(20)
        .ok_or_else(|| invalid("control CPU accounting overflow"))?;
    accounting.cold_cpu_bytes = add(accounting.cold_cpu_bytes, add(t.n_bytes, normalized)?)?;
    Ok(Some(values))
}
