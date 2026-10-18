use super::*;

const MAX_OUTPUT_BYTES: usize = 32 * 1024 * 1024;
pub(super) fn is_table(name: &str) -> bool {
    name.starts_with("blk.") && name.ends_with(".engram_embd.weight")
}
pub(super) fn validate(g: &GgufFile, t: &TensorDesc) -> Result<(), LoadError> {
    validate_encoding(g)?;
    let valid =
        t.shape.len() == 2 && t.shape[1] > 0 && t.dtype == GgmlType::Q8_0 && t.shape[0] == 256;
    if !valid {
        return Err(invalid(format!(
            "unsupported Engram table: {} {:?} {:?}; expected Q8_0 [256, rows]",
            t.name, t.dtype, t.shape
        )));
    }
    Ok(())
}
pub(super) fn validate_encoding(g: &GgufFile) -> Result<(), LoadError> {
    let key = "deepseek41.engram.encoding";
    if g.model.metadata().get(key).is_some() && !matches!(g.get_str(key), Some("q8_0" | "Q8_0")) {
        return Err(invalid(format!(
            "unsupported {key}: {:?}; standard Q8_0 is required",
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
        if ids.len() > MAX_OUTPUT_BYTES / (256 * 4) {
            return Err(invalid("Engram request exceeds bounded output"));
        }
        let t = self
            .tables
            .get(&layer)
            .ok_or_else(|| invalid("missing Engram layer"))?;
        if ids.iter().any(|&id| id >= t.shape[1]) {
            return Err(invalid("Engram row outside table"));
        }
        let stride = 272;
        let mut raw = [0u8; 272];
        let mut output = Vec::with_capacity(ids.len() * 256);
        for &id in ids {
            let offset = id
                .checked_mul(stride as u64)
                .and_then(|n| t.data_offset.checked_add(n))
                .ok_or_else(|| invalid("Engram row offset overflow"))?;
            self.gguf
                .read_shard_exact_at(t.shard_idx, offset, &mut raw[..stride])?;
            decode(&raw, &mut output)?;
        }
        Ok(output)
    }
}
fn decode(bytes: &[u8; 272], output: &mut Vec<f32>) -> Result<(), LoadError> {
    for col in 0..256 {
        let v = {
            let base = (col / 32) * 34;
            let scale =
                half::f16::from_bits(u16::from_le_bytes([bytes[base], bytes[base + 1]])).to_f32();
            scale * bytes[base + 2 + col % 32] as i8 as f32
        };
        if !v.is_finite() {
            return Err(invalid("non-finite Engram row"));
        }
        output.push(v);
    }
    Ok(())
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
        let mut out = Vec::new();
        decode(&raw, &mut out).unwrap();
        assert_eq!(out, vec![-1.0; 256]);
    }
}
