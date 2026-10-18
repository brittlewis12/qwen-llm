use super::*;
use crate::metal::{MetalContext, MetalTensor};
use crate::tensor::GgmlType;
use objc2_metal::{MTLBuffer, MTLResource};

#[derive(Clone, Copy, Debug, Default)]
pub struct ByteCounters {
    pub packed_experts: u64,
    pub dense_f32: u64,
    pub disk_engram: u64,
    pub skipped_source: u64,
    /// Upper bound on simultaneously live payload scratch (excludes headers,
    /// descriptors, allocator bookkeeping, and the resident output buffers).
    pub peak_staging: u64,
    pub source_bytes_read: u64,
}

impl ByteCounters {
    pub fn resident_bytes(&self) -> u64 {
        self.packed_experts + self.dense_f32
    }
}

/// Caller supplies an explicit weight budget, leaving room for forward/KV.
/// Loading never changes iogpu limits or overrides system memory admission.
pub struct LoadLimits {
    pub resident_bytes: u64,
}

pub struct PackedBank {
    pub layer: usize,
    /// Old source projection name: w1 (gate), w3 (up), w2 (down).
    pub projection: &'static str,
    /// Native expert ABI: [K, M, 272], GGML MXFP4, offset zero.
    pub tensor: MetalTensor,
}

pub struct ResidentWeights {
    pub banks: Vec<PackedBank>,
    /// GGML dimension order (reverse of the source); read-only mapped F32 storage.
    pub dense: BTreeMap<String, MetalTensor>,
    pub engram: DiskEngram,
    pub bytes: ByteCounters,
}

pub(super) fn residency_admission(
    metal_bytes: u64,
    staging_bytes: u64,
    signals: crate::metal::MetalMemorySignals,
) -> crate::metal::MetalMemoryAdmission {
    // A reported zero omits the process budget on macOS, as in other resident loaders.
    crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(
        metal_bytes,
        staging_bytes,
        0,
        signals,
        true,
    )
}

fn dimensions(part: &str) -> (usize, usize) {
    if part == "w2" {
        (HIDDEN, INTERMEDIATE)
    } else {
        (INTERMEDIATE, HIDDEN)
    }
}

fn expert_key(layer: usize, expert: usize, part: &str, suffix: &str) -> String {
    format!("layers.{layer}.ffn.experts.{expert}.{part}.{suffix}")
}

fn main_tensor(name: &str) -> bool {
    if let Some(rest) = name.strip_prefix("layers.") {
        return rest
            .split('.')
            .next()
            .and_then(|v| v.parse::<usize>().ok())
            .is_some_and(|l| l < LAYERS);
    }
    name.starts_with("embed.") || name.starts_with("norm.") || name.starts_with("head.")
}

fn engram_table(name: &str) -> bool {
    name.contains(".engram.embed.")
}

fn dense_entries<'a>(
    source: &'a SourceStore,
    engram: &'a SourceStore,
) -> Vec<(&'a str, &'a StoredTensor, &'a SourceStore)> {
    let mut entries: BTreeMap<&str, (&StoredTensor, &SourceStore)> = source
        .descriptors()
        .filter(|(key, _)| main_tensor(key) && !engram_table(key) && !key.contains(".ffn.experts."))
        .map(|(key, tensor)| (key, (tensor, source)))
        .collect();
    for (key, tensor) in engram.descriptors().filter(|(key, _)| {
        (key.starts_with("layers.1.engram.") || key.starts_with("layers.14.engram."))
            && !engram_table(key)
    }) {
        entries.entry(key).or_insert((tensor, engram));
    }
    entries
        .into_iter()
        .map(|(key, (tensor, store))| (key, tensor, store))
        .collect()
}

fn validate_config(config: &serde_json::Value) -> Result<(), String> {
    let text = &config["text_config"];
    for (key, value) in [
        ("hidden_size", HIDDEN),
        ("moe_intermediate_size", INTERMEDIATE),
        ("num_hidden_layers", LAYERS),
        ("n_routed_experts", EXPERTS),
        ("vocab_size", 129280),
    ] {
        if text[key].as_u64() != Some(value as u64) {
            return Err(format!("not old REAP-272E geometry: {key}"));
        }
    }
    let q = &config["quantization_config"];
    if q["weight_block_size"] != serde_json::json!([32, 32])
        || q["scale_fmt"] != "ue8m0"
        || q["expert_dtype"] != "fp4"
    {
        return Err("unsupported old REAP quantization".into());
    }
    Ok(())
}

impl ResidentWeights {
    /// Header-only preflight; binds all 40*272*3 expert pairs, including pairs
    /// split across shards. Does not use the incompatible strict TextConfig.
    pub fn plan(
        source: &SourceStore,
        engram: &SourceStore,
        config: &serde_json::Value,
    ) -> Result<ByteCounters, String> {
        validate_config(config)?;
        for (key, shard) in &source.missing_tensors {
            if main_tensor(key)
                && !engram_table(key)
                && !(key.contains(".engram.") && engram.tensor(key).is_ok())
            {
                return Err(format!("required tensor {key} still downloading: {shard}"));
            }
        }
        let mut bytes = ByteCounters::default();
        for layer in 0..LAYERS {
            for part in ["w1", "w3", "w2"] {
                let (rows, columns) = dimensions(part);
                let (codes, scales, packed) =
                    super::super::mxfp4::storage_sizes(rows, columns).map_err(|e| e.to_string())?;
                bytes.packed_experts += (packed * EXPERTS) as u64;
                bytes.peak_staging = bytes.peak_staging.max((codes + scales + packed) as u64);
                for expert in 0..EXPERTS {
                    for (suffix, dtype, shape) in [
                        ("weight", SourceDtype::I8, [rows as u64, columns as u64 / 2]),
                        (
                            "scale",
                            SourceDtype::F8E8M0,
                            [rows as u64, columns as u64 / 32],
                        ),
                    ] {
                        let key = expert_key(layer, expert, part, suffix);
                        let d = &source.tensor(&key)?.descriptor;
                        if d.dtype != dtype || d.shape != shape {
                            return Err(format!("invalid old expert dtype/geometry: {key}"));
                        }
                    }
                }
            }
        }
        for (key, tensor) in source.descriptors() {
            let d = &tensor.descriptor;
            if engram_table(key) || !main_tensor(key) {
                bytes.skipped_source += d.file_bytes.end - d.file_bytes.start;
                continue;
            }
            if key.contains(".ffn.experts.") {
                let parts: Vec<_> = key.split('.').collect();
                if parts.len() != 7
                    || parts[4].parse::<usize>().ok().is_none_or(|e| e >= EXPERTS)
                    || !["w1", "w2", "w3"].contains(&parts[5])
                    || !["weight", "scale"].contains(&parts[6])
                {
                    return Err(format!("unexpected main expert tensor: {key}"));
                }
                continue;
            }
        }
        let dense = dense_entries(source, engram);
        for (key, tensor, store) in &dense {
            let d = &tensor.descriptor;
            if key.ends_with(".scale") {
                let weight = format!("{}weight", key.strip_suffix("scale").unwrap());
                if store.tensor(&weight)?.descriptor.dtype != SourceDtype::F8E4M3 {
                    return Err(format!("orphan scale: {key}"));
                }
                continue;
            }
            let (rows, columns) = row_geometry(d)?;
            if !matches!(
                d.dtype,
                SourceDtype::Bf16 | SourceDtype::F32 | SourceDtype::F8E4M3
            ) {
                return Err(format!("unsupported dense dtype: {key}"));
            }
            if columns.checked_mul(4).ok_or("dense row extent overflow")? > MAX_READ_BYTES as u64 {
                return Err(format!("dense row exceeds staging bound: {key}"));
            }
            if d.dtype == SourceDtype::F8E4M3 {
                let scale = format!(
                    "{}scale",
                    key.strip_suffix("weight")
                        .ok_or("FP8 without weight suffix")?
                );
                let sd = &store.tensor(&scale)?.descriptor;
                if d.shape.len() != 2
                    || sd.dtype != SourceDtype::F8E8M0
                    || sd.shape != [rows.div_ceil(32), columns.div_ceil(32)]
                {
                    return Err(format!("invalid dense scale: {scale}"));
                }
            }
            bytes.dense_f32 = bytes
                .dense_f32
                .checked_add(
                    rows.checked_mul(columns)
                        .and_then(|n| n.checked_mul(4))
                        .ok_or("dense extent overflow")?,
                )
                .ok_or("resident extent overflow")?;
            let batch = rows.min(32).min(MAX_READ_BYTES as u64 / (columns * 4));
            bytes.peak_staging = bytes
                .peak_staging
                .max(batch * columns * (4 + d.dtype.byte_width()) + 2 * columns.div_ceil(32));
        }
        for (key, shape) in [
            ("embed.weight", vec![129280, 5120]),
            ("norm.weight", vec![5120]),
            ("head.weight", vec![129280, 5120]),
        ] {
            if source.tensor(key)?.descriptor.shape != shape {
                return Err(format!("invalid old top-level geometry: {key}"));
            }
        }
        for layer in [1, 14] {
            for (suffix, shape) in [
                ("q_weight", vec![4, 5120]),
                ("k_weight", vec![4, 5120]),
                ("wkv.weight", vec![25600, 6144]),
            ] {
                let key = format!("layers.{layer}.engram.{suffix}");
                let (_, tensor, _) = dense
                    .iter()
                    .find(|(name, _, _)| *name == key)
                    .ok_or_else(|| format!("missing Engram projection: {key}"))?;
                if tensor.descriptor.shape != shape {
                    return Err(format!("invalid Engram projection: {key}"));
                }
            }
        }
        for (layer, rows) in [(1, 384006168), (14, 384016682)] {
            for (suffix, dtype, width) in [
                ("weight", SourceDtype::F8E4M3, 256),
                ("scale", SourceDtype::F8E8M0, 8),
            ] {
                let key = format!("layers.{layer}.engram.embed.{suffix}");
                let d = &engram.tensor(&key)?.descriptor;
                if d.dtype != dtype || d.shape != [rows, width] {
                    return Err(format!("invalid original-base Engram geometry: {key}"));
                }
                bytes.disk_engram += d.file_bytes.end - d.file_bytes.start;
            }
        }
        Ok(bytes)
    }

    /// The caller creates and validates `backing_dir`; files are unlinked immediately.
    pub fn load(
        ctx: &MetalContext,
        source: &SourceStore,
        engram: Arc<SourceStore>,
        config: &serde_json::Value,
        limits: LoadLimits,
        backing_dir: &Path,
    ) -> Result<Self, String> {
        let bytes = Self::plan(source, &engram, config)?;
        if bytes.resident_bytes() > limits.resident_bytes {
            return Err(format!(
                "weight budget exceeded: {} > {}",
                bytes.resident_bytes(),
                limits.resident_bytes
            ));
        }
        // Price each actual allocation, including page/alignment overhead, before
        // creating any buffer. Respect both recommended and process budgets.
        let mut priced = 0u64;
        let mut price = |n: u64| -> Result<(), String> {
            let n = backing::padded_bytes(n)?;
            if n > ctx.max_buffer_length() as u64 {
                return Err("weight exceeds Metal max buffer length".into());
            }
            priced = priced
                .checked_add(
                    ctx.price_shared_buffer_upper(n)
                        .map_err(|e| e.to_string())?
                        .priced_upper_bytes,
                )
                .ok_or("allocation pricing overflow")?;
            Ok(())
        };
        for _ in 0..LAYERS * 3 {
            price(bytes.packed_experts / (LAYERS * 3) as u64)?;
        }
        for (_, tensor, _) in dense_entries(source, &engram)
            .into_iter()
            .filter(|(key, _, _)| !key.ends_with(".scale"))
        {
            let (r, c) = row_geometry(&tensor.descriptor)?;
            price(r * c * 4)?;
        }
        let admission = residency_admission(priced, bytes.peak_staging, ctx.memory_signals());
        if let Some(refusal) = admission.refusal() {
            return Err(refusal.to_string());
        }
        let disk_required = priced.checked_add(2 << 30).ok_or("backing size overflow")?;
        let disk_available = crate::checkpoint_fs::volume_free_bytes(backing_dir)
            .ok_or("cannot determine free space for weight backing")?;
        if disk_available < disk_required {
            return Err(format!(
                "temporary weight backing needs {disk_required} bytes free, found {disk_available}"
            ));
        }
        // Do not retain a file-cache copy alongside the resident weight buffers.
        source.stream_without_cache()?;
        let mut loaded = Self {
            banks: Vec::with_capacity(LAYERS * 3),
            dense: BTreeMap::new(),
            engram: DiskEngram::new(engram.clone())?,
            bytes,
        };
        for layer in 0..LAYERS {
            for part in ["w1", "w3", "w2"] {
                let (rows, columns) = dimensions(part);
                let (nc, ns, _) =
                    super::super::mxfp4::storage_sizes(rows, columns).map_err(|e| e.to_string())?;
                let tensor = materialize_bank(
                    ctx,
                    source,
                    layer,
                    part,
                    rows,
                    columns,
                    EXPERTS,
                    backing_dir,
                )?;
                loaded.bytes.source_bytes_read += ((nc + ns) * EXPERTS) as u64;
                loaded.banks.push(PackedBank {
                    layer,
                    projection: part,
                    tensor,
                });
            }
            eprintln!("[dsv41] loaded expert layer {}/{}", layer + 1, LAYERS);
        }
        for (key, _, store) in dense_entries(source, &engram)
            .into_iter()
            .filter(|(key, _, _)| !key.ends_with(".scale"))
        {
            let (tensor, read) = materialize_dense(ctx, store, key, backing_dir)?;
            loaded.bytes.source_bytes_read += read;
            loaded.dense.insert(key.to_owned(), tensor);
        }
        Ok(loaded)
    }

    pub fn bank(&self, layer: usize, projection: &str) -> Option<&MetalTensor> {
        self.banks
            .iter()
            .find(|b| b.layer == layer && b.projection == projection)
            .map(|b| &b.tensor)
    }

    /// Copy selected rows from the resident F32 projection, in source order.
    pub fn dense_rows(&self, name: &str, rows: Range<u64>) -> Result<Vec<f32>, String> {
        let tensor = self
            .dense
            .get(name)
            .ok_or_else(|| format!("missing resident projection: {name}"))?;
        let height = *tensor.shape.last().ok_or("missing dense shape")?;
        let width = tensor.shape[..tensor.shape.len() - 1]
            .iter()
            .try_fold(1u64, |a, b| {
                a.checked_mul(*b).ok_or("dense row width overflow")
            })?;
        let count = rows.end.checked_sub(rows.start).ok_or("reversed rows")?;
        if rows.end > height
            || count
                .checked_mul(width)
                .and_then(|n| n.checked_mul(4))
                .ok_or("row overflow")?
                > MAX_READ_BYTES as u64
        {
            return Err("dense row request out of bounds".into());
        }
        let end = rows
            .end
            .checked_mul(width)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(tensor.offset))
            .ok_or("dense buffer extent overflow")?;
        if tensor.dtype != GgmlType::F32
            || tensor.buffer.storageMode() != objc2_metal::MTLStorageMode::Shared
            || tensor.offset % 4 != 0
            || end > tensor.buffer.length() as u64
        {
            return Err("invalid resident dense buffer view".into());
        }
        // Published weights are read-only by contract; no GPU writer is allowed.
        Ok(unsafe {
            std::slice::from_raw_parts(
                tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<f32>()
                    .add((tensor.offset / 4 + rows.start * width) as usize),
                (count * width) as usize,
            )
        }
        .to_vec())
    }
}

pub(super) fn materialize_bank(
    ctx: &MetalContext,
    source: &SourceStore,
    layer: usize,
    part: &str,
    rows: usize,
    columns: usize,
    experts: usize,
    backing_dir: &Path,
) -> Result<MetalTensor, String> {
    let (nc, ns, np) =
        super::super::mxfp4::storage_sizes(rows, columns).map_err(|e| e.to_string())?;
    if np > MAX_READ_BYTES || nc > MAX_READ_BYTES || ns > MAX_READ_BYTES {
        return Err("expert exceeds staging bound".into());
    }
    let extent = np.checked_mul(experts).ok_or("bank extent overflow")?;
    let mut writer = backing::BackingWriter::new(backing_dir, extent as u64)?;
    let mut output = vec![0; np];
    for expert in 0..experts {
        let codes =
            source.bounded_read(&expert_key(layer, expert, part, "weight"), 0..nc as u64)?;
        let scales =
            source.bounded_read(&expert_key(layer, expert, part, "scale"), 0..ns as u64)?;
        super::super::mxfp4::repack_into(rows, columns, &codes, &scales, &mut output)
            .map_err(|e| e.to_string())?;
        writer.write(&output)?;
    }
    writer.tensor(
        ctx,
        vec![columns as u64, rows as u64, experts as u64],
        GgmlType::MXFP4,
    )
}

pub(super) fn materialize_dense(
    ctx: &MetalContext,
    store: &SourceStore,
    key: &str,
    backing_dir: &Path,
) -> Result<(MetalTensor, u64), String> {
    let d = &store.tensor(key)?.descriptor;
    let (rows, columns) = row_geometry(d)?;
    let extent = rows
        .checked_mul(columns)
        .and_then(|n| n.checked_mul(4))
        .ok_or("dense extent overflow")?;
    let batch = rows.min(32).min(MAX_READ_BYTES as u64 / (columns * 4));
    if batch == 0 {
        return Err("dense row exceeds staging bound".into());
    }
    let mut writer = backing::BackingWriter::new(backing_dir, extent)?;
    let mut read = 0;
    for row in (0..rows).step_by(batch as usize) {
        let end = (row + batch).min(rows);
        let values = store.decode_rows(key, row..end, false)?;
        writer.write(bytemuck::cast_slice(&values))?;
        read += (end - row) * columns * d.dtype.byte_width()
            + if d.dtype == SourceDtype::F8E4M3 {
                (end.div_ceil(32) - row / 32) * columns.div_ceil(32)
            } else {
                0
            };
    }
    Ok((
        writer.tensor(ctx, d.shape.iter().rev().copied().collect(), GgmlType::F32)?,
        read,
    ))
}
