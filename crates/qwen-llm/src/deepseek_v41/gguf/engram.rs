use super::*;

const MAX_OUTPUT_BYTES: usize = 32 * 1024 * 1024;
const ROW_WIDTH: usize = 256;
pub(super) fn is_table(name: &str) -> bool {
    name.starts_with("blk.") && name.ends_with(".engram_embd.weight")
}
pub(super) fn validate(g: &GgufFile, t: &TensorDesc) -> Result<(), LoadError> {
    validate_encoding(g)?;
    let valid = t.shape.len() == 2 && t.shape[1] > 0 && t.shape[0] == ROW_WIDTH as u64;
    if !valid {
        return Err(invalid(format!(
            "unsupported Engram table: {} {:?} {:?}; expected a codec-supported [256, rows] matrix",
            t.name, t.dtype, t.shape
        )));
    }
    validate_storage(t)?;
    row_stride(t)?;
    if let Some(encoding) = g.get_str("deepseek41.engram.encoding") {
        if !encoding.eq_ignore_ascii_case(t.dtype.wire_name()) {
            return Err(invalid(format!(
                "Engram encoding {encoding:?} conflicts with {} {:?}",
                t.name, t.dtype
            )));
        }
    }
    // A zero-row descriptor checks the linked codec and block traits without
    // reading, allocating, or decoding any table payload.
    let probe = TensorDesc {
        name: t.name.clone(),
        shape: vec![ROW_WIDTH as u64, 0],
        dtype: t.dtype,
        shard_idx: t.shard_idx,
        data_offset: 0,
        n_bytes: 0,
    };
    crate::codec::validate_dequantization(&probe, &[]).map_err(|e| {
        invalid(format!(
            "unsupported Engram codec for {} {:?}: {e}",
            t.name, t.dtype
        ))
    })?;
    Ok(())
}
fn row_stride(t: &TensorDesc) -> Result<usize, LoadError> {
    let (block_elements, block_bytes) = t
        .dtype
        .storage_layout()
        .ok_or_else(|| invalid(format!("unsupported Engram type: {} {:?}", t.name, t.dtype)))?;
    if block_elements == 0 || ROW_WIDTH as u64 % block_elements != 0 {
        return Err(invalid(format!(
            "Engram row is not block-aligned: {} {:?}",
            t.name, t.dtype
        )));
    }
    let stride = (ROW_WIDTH as u64 / block_elements)
        .checked_mul(block_bytes)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| invalid("Engram row stride overflow"))?;
    if stride == 0 || stride > MAX_OUTPUT_BYTES {
        return Err(invalid("Engram packed row exceeds bounded staging"));
    }
    Ok(stride)
}
pub(super) fn validate_encoding(g: &GgufFile) -> Result<(), LoadError> {
    let key = "deepseek41.engram.encoding";
    let standard_name = g.get_str(key).is_some_and(|name| {
        (0..llama_cpp_sys_2::GGML_TYPE_COUNT)
            .map(GgmlType::from_raw)
            .any(|dtype| {
                dtype.storage_layout().is_some() && name.eq_ignore_ascii_case(dtype.wire_name())
            })
    });
    if g.model.metadata().get(key).is_some() && !standard_name {
        return Err(invalid(format!(
            "unsupported {key}: {:?}; expected a standard GGML type name, not custom packed FP8",
            g.get_str(key)
        )));
    }
    Ok(())
}

/// Retains the parsed file handles. Reads only requested rows, never maps a
/// table into Metal or materializes a full table in host memory.
pub struct DiskEngram {
    gguf: Arc<GgufFile>,
    tables: BTreeMap<usize, TensorDesc>,
}
impl DiskEngram {
    pub fn new(gguf: Arc<GgufFile>) -> Result<Self, LoadError> {
        let layers = metadata::array(&gguf, &["engram.layers", "engram.layer_ids"])?
            .unwrap_or_else(|| vec![1, 14]);
        Self::for_layers(gguf, &layers)
    }
    pub(super) fn for_layers(gguf: Arc<GgufFile>, layers: &[u64]) -> Result<Self, LoadError> {
        validate_encoding(&gguf)?;
        let mut tables = BTreeMap::new();
        for &layer in layers {
            let name = format!("blk.{layer}.engram_embd.weight");
            let t = gguf
                .find(&name)
                .ok_or_else(|| invalid(format!("missing Engram table {name}")))?;
            validate(&gguf, t)?;
            tables.insert(
                usize::try_from(layer).map_err(|_| invalid("Engram layer overflow"))?,
                t.clone(),
            );
        }
        Ok(Self { gguf, tables })
    }
    pub fn decode_rows(&self, layer: usize, ids: &[u64]) -> Result<Vec<f32>, String> {
        self.read_rows(layer, ids).map_err(|e| e.to_string())
    }
    pub fn read_rows(&self, layer: usize, ids: &[u64]) -> Result<Vec<f32>, LoadError> {
        if ids.len() > MAX_OUTPUT_BYTES / (ROW_WIDTH * std::mem::size_of::<f32>()) {
            return Err(invalid("Engram request exceeds bounded output"));
        }
        let t = self
            .tables
            .get(&layer)
            .ok_or_else(|| invalid("missing Engram layer"))?;
        if ids.iter().any(|&id| id >= t.shape[1]) {
            return Err(invalid("Engram row outside table"));
        }
        let stride = row_stride(t)?;
        let mut raw = vec![0u8; stride];
        let row = TensorDesc {
            name: t.name.clone(),
            shape: vec![ROW_WIDTH as u64, 1],
            dtype: t.dtype,
            shard_idx: t.shard_idx,
            data_offset: 0,
            n_bytes: stride as u64,
        };
        let mut output = Vec::new();
        output
            .try_reserve_exact(ids.len() * ROW_WIDTH)
            .map_err(|e| invalid(format!("Engram output allocation: {e}")))?;
        output.resize(ids.len() * ROW_WIDTH, 0.0);
        for (&id, destination) in ids.iter().zip(output.chunks_exact_mut(ROW_WIDTH)) {
            let offset = id
                .checked_mul(stride as u64)
                .and_then(|n| t.data_offset.checked_add(n))
                .ok_or_else(|| invalid("Engram row offset overflow"))?;
            self.gguf
                .read_shard_exact_at(t.shard_idx, offset, &mut raw)?;
            crate::codec::dequant_to_f32_in_place(&row, &raw, destination)
                .map_err(|e| invalid(format!("Engram row decode {} {:?}: {e}", t.name, t.dtype)))?;
            if destination.iter().any(|v| !v.is_finite()) {
                return Err(invalid(format!(
                    "non-finite Engram row: {} row {id}",
                    t.name
                )));
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn q8_signed_codes_and_half_scales() {
        let mut raw = [0u8; 272];
        for chunk in raw.chunks_exact_mut(34) {
            chunk[..2].copy_from_slice(&0x3800u16.to_le_bytes());
            chunk[2..].fill(0xfe);
        }
        let desc = TensorDesc {
            name: "q8_row".into(),
            shape: vec![256, 1],
            dtype: GgmlType::Q8_0,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: 272,
        };
        let mut out = vec![0.0; 256];
        crate::codec::dequant_to_f32_in_place(&desc, &raw, &mut out).unwrap();
        assert_eq!(out, vec![-1.0; 256]);
    }
}
