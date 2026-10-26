use super::*;

const MAX_OUTPUT_BYTES: usize = 32 * 1024 * 1024;
const ROW_WIDTH: usize = 256;
const COMBINED_TABLE: &str = "per_layer_token_embd.weight";

#[derive(Debug)]
pub(super) struct TableView {
    pub(super) layer: usize,
    pub(super) tensor: TensorDesc,
    pub(super) row_base: u64,
    pub(super) row_count: u64,
}

pub(super) fn is_table(name: &str) -> bool {
    name == COMBINED_TABLE || (name.starts_with("blk.") && name.ends_with(".engram_embd.weight"))
}

pub(super) fn resolve_tables(
    g: &GgufFile,
    hash: &HashConfig,
    main_layers: usize,
) -> Result<Vec<TableView>, LoadError> {
    let mut combined = None;
    let mut separate = BTreeMap::new();
    for t in g.tensors.iter().filter(|t| is_table(&t.name)) {
        if t.name == COMBINED_TABLE {
            if combined.replace(t).is_some() {
                return Err(invalid("duplicate combined Engram table"));
            }
        } else {
            let layer: usize = t
                .name
                .strip_prefix("blk.")
                .and_then(|name| name.strip_suffix(".engram_embd.weight"))
                .and_then(|layer| layer.parse().ok())
                .ok_or_else(|| invalid(format!("invalid Engram table name: {}", t.name)))?;
            if layer >= main_layers {
                continue;
            }
            if separate.insert(layer, t).is_some() {
                return Err(invalid(format!("duplicate Engram table for layer {layer}")));
            }
        }
    }
    let mut tables = Vec::with_capacity(hash.layers.len());
    if let Some(t) = combined {
        if !separate.is_empty() {
            return Err(invalid("ambiguous combined and separate Engram tables"));
        }
        if hash.layers.is_empty() {
            return Err(invalid("undeclared combined Engram table"));
        }
        validate(g, t)?;
        let mut end = 0u64;
        for layer in &hash.layers {
            let row_base = end;
            end = end
                .checked_add(layer.rows)
                .ok_or_else(|| invalid("combined Engram row extent overflow"))?;
            tables.push(TableView {
                layer: layer.layer,
                tensor: t.clone(),
                row_base,
                row_count: layer.rows,
            });
        }
        if end != t.shape[1] {
            return Err(invalid("combined Engram rows differ from bucket total"));
        }
    } else {
        for &layer in separate.keys() {
            if !hash.layers.iter().any(|l| l.layer == layer) {
                return Err(invalid(format!(
                    "undeclared Engram table for layer {layer}"
                )));
            }
        }
        for layer in &hash.layers {
            let t = separate.get(&layer.layer).ok_or_else(|| {
                invalid(format!("missing Engram table for layer {}", layer.layer))
            })?;
            validate(g, t)?;
            if t.shape[1] != layer.rows {
                return Err(invalid(format!(
                    "Engram table rows differ from bucket total: {}",
                    t.name
                )));
            }
            tables.push(TableView {
                layer: layer.layer,
                tensor: (*t).clone(),
                row_base: 0,
                row_count: layer.rows,
            });
        }
    }
    Ok(tables)
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
pub(super) fn row_stride(t: &TensorDesc) -> Result<usize, LoadError> {
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
    context: companion::TableContext,
    tables: BTreeMap<usize, TableView>,
}
impl DiskEngram {
    pub fn new(gguf: Arc<GgufFile>) -> Result<Self, LoadError> {
        gguf.revalidate_retained_shard_stamps()?;
        let p = metadata::parse(&gguf)?;
        p.engram_context.revalidate(&gguf)?;
        Ok(Self::for_tables(gguf, p.engram_context))
    }
    pub(super) fn for_tables(gguf: Arc<GgufFile>, mut context: companion::TableContext) -> Self {
        let tables = std::mem::take(&mut context.tables);
        Self {
            gguf,
            context,
            tables: tables
                .into_iter()
                .map(|table| (table.layer, table))
                .collect(),
        }
    }
    pub fn source_groups(&self) -> &SourceGroups {
        &self.context.stamps
    }
    pub fn revalidate_sources(&self) -> Result<(), LoadError> {
        self.context.revalidate(&self.gguf)
    }
    pub fn decode_rows(&self, layer: usize, ids: &[u64]) -> Result<Vec<f32>, String> {
        self.read_rows(layer, ids).map_err(|e| e.to_string())
    }
    /// Like [`Self::decode_rows`] for rows already passed to
    /// [`Self::advise_rows`]: reads without advising again.
    pub fn decode_advised_rows(&self, layer: usize, ids: &[u64]) -> Result<Vec<f32>, String> {
        self.read_rows_with(layer, ids, false)
            .map_err(|e| e.to_string())
    }
    /// Start asynchronous page-ins for rows that will be read soon, so a
    /// caller can overlap several layers' misses. Advisory only: invalid
    /// layers or IDs are left for the exact read to reject.
    pub fn advise_rows(&self, layer: usize, ids: &[u64]) {
        if !concurrent_reads() {
            return;
        }
        let Some(table) = self.tables.get(&layer) else {
            return;
        };
        let Ok(stride) = row_stride(&table.tensor) else {
            return;
        };
        for &id in ids.iter().filter(|&&id| id < table.row_count) {
            if let Ok(offset) = row_offset(table, stride, id) {
                self.source()
                    .advise_shard_read(table.tensor.shard_idx, offset, stride);
            }
        }
    }
    /// Any number of rows (prefill chunks), read in order in bounded blocks.
    /// Cached rows take the plain serial path: advising every row, inline or
    /// from a helper thread, added 2-3% to a warm 512-token prefill. When a
    /// block reads slower than cached rows can (uncached rows are SSD round
    /// trips), read-ahead is queued for every remaining row at once.
    pub fn decode_rows_batched(&self, layer: usize, ids: &[u64]) -> Result<Vec<f32>, String> {
        let mut output = Vec::new();
        output
            .try_reserve_exact(ids.len() * ROW_WIDTH)
            .map_err(|e| format!("Engram output allocation: {e}"))?;
        let mut advised = !concurrent_reads();
        let mut done = 0;
        while done < ids.len() {
            // A short first block detects uncached rows early.
            let block = if done == 0 { PROBE_ROWS } else { BLOCK_ROWS };
            let block = &ids[done..(done + block).min(ids.len())];
            let started = std::time::Instant::now();
            output.extend(
                self.read_rows_with(layer, block, false)
                    .map_err(|e| e.to_string())?,
            );
            done += block.len();
            if !advised
                && started.elapsed() > COLD_ROW * u32::try_from(block.len()).unwrap_or(u32::MAX)
            {
                self.advise_rows(layer, &ids[done..]);
                advised = true;
            }
        }
        Ok(output)
    }
    /// Rows are independent small positional reads. An uncached row is an
    /// SSD round trip (~70 us for one 16 KiB page), and the 103 GiB table
    /// cannot stay cached next to the resident model, so every row is
    /// advised first (asynchronous read-ahead) and then read: 48 uncached
    /// rows cost ~0.24 ms instead of ~3.4 ms, cached rows ~0.5 us more each.
    pub fn read_rows(&self, layer: usize, ids: &[u64]) -> Result<Vec<f32>, LoadError> {
        self.read_rows_with(layer, ids, true)
    }
    fn source(&self) -> &GgufFile {
        self.context.auxiliary.as_ref().unwrap_or(&self.gguf)
    }
    fn read_rows_with(
        &self,
        layer: usize,
        ids: &[u64],
        advise: bool,
    ) -> Result<Vec<f32>, LoadError> {
        self.revalidate_sources()?;
        if ids.len() > MAX_OUTPUT_BYTES / (ROW_WIDTH * std::mem::size_of::<f32>()) {
            return Err(invalid("Engram request exceeds bounded output"));
        }
        let table = self
            .tables
            .get(&layer)
            .ok_or_else(|| invalid("missing Engram layer"))?;
        if ids.iter().any(|&id| id >= table.row_count) {
            return Err(invalid("Engram row outside table"));
        }
        let t = &table.tensor;
        let stride = row_stride(t)?;
        let source = self.source();
        let offsets = ids
            .iter()
            .map(|&id| row_offset(table, stride, id))
            .collect::<Result<Vec<_>, _>>()?;
        if advise && concurrent_reads() {
            for &offset in &offsets {
                source.advise_shard_read(t.shard_idx, offset, stride);
            }
        }
        let mut raw = Vec::new();
        raw.try_reserve_exact(stride)
            .map_err(|e| invalid(format!("Engram row allocation: {e}")))?;
        raw.resize(stride, 0);
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
        for ((&id, &offset), destination) in ids
            .iter()
            .zip(&offsets)
            .zip(output.as_chunks_mut::<ROW_WIDTH>().0)
        {
            source.read_shard_exact_at(t.shard_idx, offset, &mut raw)?;
            crate::codec::dequant_to_f32_in_place(&row, &raw, destination)
                .map_err(|e| invalid(format!("Engram row decode {} {:?}: {e}", t.name, t.dtype)))?;
            if destination.iter().any(|v| !v.is_finite()) {
                return Err(invalid(format!(
                    "non-finite Engram row: {} row {id}",
                    t.name
                )));
            }
        }
        self.revalidate_sources()?;
        Ok(output)
    }
}

/// Batched reads: first (probe) block, later blocks, and the mean per-row
/// read time above which rows are treated as uncached. Cached rows read and
/// decode in under 1 us; an uncached row is ~60-70 us.
const PROBE_ROWS: usize = 64;
const BLOCK_ROWS: usize = 4096;
const COLD_ROW: std::time::Duration = std::time::Duration::from_micros(8);

/// `QWEN_V41_ENGRAM_IO=serial` restores one blocking read per row with no
/// read-ahead advice (A/B control).
fn concurrent_reads() -> bool {
    static CONCURRENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CONCURRENT.get_or_init(|| std::env::var("QWEN_V41_ENGRAM_IO").as_deref() != Ok("serial"))
}

fn row_offset(table: &TableView, stride: usize, id: u64) -> Result<u64, LoadError> {
    id.checked_add(table.row_base)
        .and_then(|n| n.checked_mul(stride as u64))
        .and_then(|n| table.tensor.data_offset.checked_add(n))
        .ok_or_else(|| invalid("Engram row offset overflow"))
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
