//! Metadata-driven V4.1 GGUF loading. Planning never probes hardware or weights.
//! Tensor-serialized Engram controls require bounded CPU payload reads.
//! Non-upstream GGML type 67 (MXFP8) is not supported; no fork layout is inferred.
use super::forward::Config;
use crate::{
    gguf::GgufFile,
    metal::{self, MetalContext, MetalTensor},
    tensor::{GgmlType, TensorDesc},
};
use std::{collections::BTreeMap, sync::Arc};

mod companion;
mod controls;
mod engram;
mod metadata;
mod names;
mod resident_storage;
mod retained_resources;
mod storage;
pub use super::engram::HashConfig;
pub use companion::SourceGroups;
pub use controls::ControlAccounting;
pub use engram::DiskEngram;
pub use resident_storage::{
    ResidentMappingKind, ResidentStorageAudit, ResidentStoragePlan, ResidentStorageWindow,
    ResidentTensorAudit, ResidentTensorExtent, resident_storage_audit, resident_storage_plan,
};
pub(crate) use retained_resources::{RetainedSummaryBookkeeping, retained_summary_bookkeeping};
pub use retained_resources::{RetainedWeightResource, RetainedWeightResourceSummary};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub layers: usize,
    pub kv_source_layers: Vec<usize>,
    pub index_source_layers: Vec<usize>,
    /// Main-layer ratios only; auxiliary metadata is retained in Plan.
    pub compression_ratios: Vec<usize>,
    pub candidate_source_layer: Option<usize>,
}
impl Default for Schedule {
    fn default() -> Self {
        Self {
            layers: 40,
            kv_source_layers: vec![2, 8, 14, 20],
            index_source_layers: vec![2, 8, 14, 20, 24, 28, 32, 36],
            compression_ratios: (0..40)
                .map(|l| {
                    if l < 2 {
                        0
                    } else if l < 20 {
                        2
                    } else {
                        1
                    }
                })
                .collect(),
            candidate_source_layer: Some(20),
        }
    }
}
impl Schedule {
    /// HF shared-KV ownership omitted by the upstream GGUF converter.
    fn deepseek41_shared_kv_layout(layers: usize, ratios: &[u64]) -> Option<Self> {
        if layers != 40 || ratios.len() < layers {
            return None;
        }
        let mut layout = Self::default();
        if ratios[..layers].iter().enumerate().any(|(l, &ratio)| {
            ratio
                != if l < 2 {
                    0
                } else if l < 20 {
                    2
                } else {
                    1
                }
        }) {
            return None;
        }
        layout.layers = layers;
        layout.compression_ratios = ratios[..layers].iter().map(|&n| n as usize).collect();
        Some(layout)
    }

    pub fn owns_kv(&self, layer: usize) -> bool {
        self.kv_source_layers.contains(&layer)
    }
    pub fn produces_index(&self, layer: usize) -> bool {
        self.index_source_layers.contains(&layer)
    }
    pub fn owner_slot(&self, layer: usize) -> Option<usize> {
        self.kv_source_layers
            .iter()
            .rposition(|&owner| owner <= layer)
    }
    pub fn owner(&self, layer: usize) -> Option<usize> {
        self.owner_slot(layer).map(|i| self.kv_source_layers[i])
    }
    pub fn ratio(&self, layer: usize) -> usize {
        self.compression_ratios[layer]
    }
    pub fn validate(&self) -> Result<(), LoadError> {
        if self.layers == 0 || self.compression_ratios.len() != self.layers {
            return Err(invalid("schedule block count"));
        }
        for ids in [&self.kv_source_layers, &self.index_source_layers] {
            if ids.iter().any(|&l| l >= self.layers) || ids.windows(2).any(|w| w[0] >= w[1]) {
                return Err(invalid("schedule source layers"));
            }
        }
        for layer in 0..self.layers {
            let ratio = self.ratio(layer);
            if ratio > 2 {
                return Err(invalid(
                    "attention kernels support compression ratios 0, 1 and 2",
                ));
            }
            match self.owner(layer) {
                None if ratio != 0 || self.produces_index(layer) => {
                    return Err(invalid("compressed attention has no KV owner"));
                }
                Some(owner) if ratio == 0 || ratio != self.ratio(owner) => {
                    return Err(invalid("reuse layer compression differs from KV owner"));
                }
                _ => (),
            }
            if self.owns_kv(layer) && !self.produces_index(layer) {
                return Err(invalid("KV owner must produce index keys and selection"));
            }
        }
        if self
            .candidate_source_layer
            .is_some_and(|l| !self.owns_kv(l) || self.ratio(l) != 1)
        {
            return Err(invalid("candidate source must own uncompressed KV"));
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("invalid V4.1 GGUF: {0}")]
    Invalid(String),
    #[error(transparent)]
    Gguf(#[from] crate::gguf::GgufError),
    #[error(transparent)]
    Metal(#[from] metal::MetalError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    MemoryAdmissionDenied(#[from] metal::MemoryAdmissionDenied),
}
impl LoadError {
    pub fn as_memory_admission(&self) -> Option<&metal::MemoryAdmissionDenied> {
        match self {
            Self::MemoryAdmissionDenied(e) => Some(e),
            Self::Metal(metal::MetalError::MemoryAdmission(e)) => Some(e),
            _ => None,
        }
    }
}
fn invalid(message: impl Into<String>) -> LoadError {
    LoadError::Invalid(message.into())
}
fn add(a: u64, b: u64) -> Result<u64, LoadError> {
    a.checked_add(b)
        .ok_or_else(|| invalid("byte count overflow"))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ByteCounters {
    pub packed_experts: u64,
    pub dense_bytes: u64,
    pub disk_engram: u64,
}
impl ByteCounters {
    pub fn resident_bytes(&self) -> u64 {
        self.packed_experts + self.dense_bytes
    }
}
pub struct PackedBank {
    pub layer: usize,
    pub projection: &'static str,
    pub tensor: MetalTensor,
}
pub struct ResidentWeights {
    pub config: Config,
    pub schedule: Schedule,
    pub engram_hash: HashConfig,
    pub banks: Vec<PackedBank>,
    pub dense: BTreeMap<String, MetalTensor>,
    pub engram: DiskEngram,
    pub token_map: Vec<u32>,
    pub raw_pad: u32,
    pub bytes: ByteCounters,
    /// Captured once from owning views after load; no physical residency claim.
    pub retained_resources: RetainedWeightResourceSummary,
}
#[derive(Debug)]
pub struct Plan {
    pub config: Config,
    pub schedule: Schedule,
    pub engram_hash: HashConfig,
    /// Serialized bytes by GGML dtype, including auxiliary and disk tensors.
    pub dtype_inventory: BTreeMap<String, u64>,
    pub bytes: ByteCounters,
    pub token_map: Vec<u32>,
    pub raw_pad: u32,
    pub layers: usize,
    pub engram_layers: Vec<u64>,
    pub engram_primes: Vec<u64>,
    pub engram_offsets: Vec<u64>,
    pub engram_multipliers: Vec<u64>,
    pub kv_source_layers: Vec<u64>,
    pub index_source_layers: Vec<u64>,
    pub compression_ratios: Vec<u64>,
    pub controls: ControlAccounting,
    engram_context: companion::TableContext,
}

impl Plan {
    pub fn source_groups(&self) -> &SourceGroups {
        &self.engram_context.stamps
    }
}

pub fn plan(g: &GgufFile) -> Result<Plan, LoadError> {
    g.revalidate_retained_shard_stamps()?;
    let mut p = metadata::parse(g)?;
    let mut banks = BTreeMap::new();
    let mut dense = BTreeMap::new();
    for t in &g.tensors {
        let count = p
            .dtype_inventory
            .entry(format!("{:?}", t.dtype))
            .or_default();
        *count = add(*count, t.n_bytes)?;
        if p.engram_context.auxiliary.is_none()
            && p.engram_context
                .tables
                .iter()
                .any(|table| table.tensor.name == t.name)
        {
            p.bytes.disk_engram = add(p.bytes.disk_engram, t.n_bytes)?;
        } else if let Some((layer, part)) = names::bank(&t.name) {
            if layer >= p.layers {
                continue;
            }
            let c = &p.config;
            let shape = if part == "w2" {
                [c.intermediate as u64, c.hidden as u64, c.experts as u64]
            } else {
                [c.hidden as u64, c.intermediate as u64, c.experts as u64]
            };
            if !shape_matches(&t.shape, &shape) || !metal::all_slots_down_native_supported(t.dtype)
            {
                return Err(invalid(format!(
                    "unsupported bank geometry/type: {} {:?} {:?}",
                    t.name, t.shape, t.dtype
                )));
            }
            validate_storage(t)?;
            banks.insert((layer, part), ());
            p.bytes.packed_experts = add(p.bytes.packed_experts, t.n_bytes)?;
        } else if let Some(key) = names::dense(&t.name, p.layers) {
            if !dense_supported(t.dtype) {
                return Err(invalid(format!(
                    "unsupported dense type: {} {:?}",
                    t.name, t.dtype
                )));
            }
            validate_storage(t)?;
            if dense.insert(key, ()).is_some() {
                return Err(invalid("duplicate internal dense key"));
            }
            names::validate_shape(t, &p.config)?;
            p.bytes.dense_bytes = add(p.bytes.dense_bytes, t.n_bytes)?;
        }
    }
    for layer in 0..p.layers {
        for part in ["w1", "w3", "w2"] {
            if !banks.contains_key(&(layer, part)) {
                return Err(invalid(format!("missing bank {layer}/{part}")));
            }
        }
    }
    add(p.bytes.packed_experts, p.bytes.dense_bytes)?;
    names::validate_inventory(&p, &dense)?;
    if let Some(aux) = &p.engram_context.auxiliary {
        for table in &aux.tensors {
            p.bytes.disk_engram = add(p.bytes.disk_engram, table.n_bytes)?;
            let count = p
                .dtype_inventory
                .entry(format!("{:?}", table.dtype))
                .or_default();
            *count = add(*count, table.n_bytes)?;
        }
    }
    p.engram_context.revalidate(g)?;
    Ok(p)
}
fn dense_supported(t: GgmlType) -> bool {
    super::forward::dense_linear_supported(t)
}
fn resident_tensor(t: &TensorDesc, layers: usize) -> bool {
    !engram::is_table(&t.name)
        && (names::bank(&t.name).is_some_and(|(layer, _)| layer < layers)
            || names::dense(&t.name, layers).is_some())
}
fn shape_matches(actual: &[u64], expected: &[u64]) -> bool {
    fn trimmed(mut shape: &[u64]) -> &[u64] {
        while shape.len() > 1 && shape.last() == Some(&1) {
            shape = &shape[..shape.len() - 1];
        }
        shape
    }
    trimmed(actual) == trimmed(expected)
}
fn validate_storage(t: &TensorDesc) -> Result<(), LoadError> {
    if t.shape.is_empty() || t.shape.contains(&0) {
        return Err(invalid(format!("empty tensor {}", t.name)));
    }
    let n = crate::tensor::checked_shape_elements(&t.shape)
        .ok_or_else(|| invalid("tensor extent overflow"))?;
    let (block, block_bytes) = t
        .dtype
        .storage_layout()
        .ok_or_else(|| invalid("unknown tensor type"))?;
    if block == 0 || t.shape[0] % block != 0 {
        return Err(invalid("partial quantization row"));
    }
    let bytes = (n / block)
        .checked_mul(block_bytes)
        .ok_or_else(|| invalid("tensor byte overflow"))?;
    if bytes != t.n_bytes {
        return Err(invalid(format!("storage size mismatch: {}", t.name)));
    }
    add(t.data_offset, bytes)?;
    Ok(())
}
fn prepared_load_plan(
    g: &GgufFile,
    page: usize,
    max_buffer_length: usize,
    soft_target: Option<usize>,
) -> Result<(Plan, ResidentStoragePlan), LoadError> {
    let p = plan(g)?;
    p.engram_context.revalidate(g)?;
    let storage =
        resident_storage::for_plan_with_target(g, &p, page, max_buffer_length, soft_target)?;
    Ok((p, storage))
}

impl ResidentWeights {
    pub fn bank(&self, layer: usize, part: &str) -> Option<&MetalTensor> {
        self.banks
            .iter()
            .find(|b| b.layer == layer && b.projection == part)
            .map(|b| &b.tensor)
    }
    pub fn plan(g: &GgufFile) -> Result<Plan, LoadError> {
        plan(g)
    }
    pub fn load(ctx: &MetalContext, g: Arc<GgufFile>) -> Result<Self, LoadError> {
        Self::load_planned(ctx, g, resident_storage::SOFT_COALESCING_TARGET)
    }

    fn load_planned(
        ctx: &MetalContext,
        g: Arc<GgufFile>,
        soft_target: Option<usize>,
    ) -> Result<Self, LoadError> {
        let cold_cpu = controls::cold_bound(&g)?;
        if cold_cpu > 0
            && let Some(e) = metal::evaluate_metal_memory_admission_with_cpu_bytes(
                0,
                cold_cpu,
                0,
                ctx.memory_signals(),
                true,
            )
            .refusal()
        {
            return Err(e.into());
        }
        let page = metal::host_page_size_bytes()?;
        let (p, storage) = prepared_load_plan(&g, page, ctx.max_buffer_length(), soft_target)?;
        let bookkeeping: RetainedSummaryBookkeeping =
            retained_summary_bookkeeping(&g, &p, &storage)?;
        let windows = &storage.windows;
        let requests: Vec<_> = g
            .tensors
            .iter()
            .filter(|t| resident_tensor(t, p.layers))
            .collect();
        let _allocation_transaction = ctx.begin_allocation_transaction();
        if let Some(e) = metal::evaluate_metal_memory_admission_with_cpu_bytes(
            storage.metal_view_bytes,
            bookkeeping.load_cpu_bytes(&p.controls, storage.private_shadow_bytes)?,
            0,
            ctx.memory_signals(),
            true,
        )
        .refusal()
        {
            return Err(e.into());
        }
        let mut backings = Vec::new();
        for planned in windows {
            let w = &planned.window;
            let (mmap, local) = storage::mapping(&g, w, page)?;
            let ownership = match planned.mapping_kind {
                ResidentMappingKind::OriginalShared | ResidentMappingKind::SharedEof => {
                    metal::GgufMappingOwnership::Shared
                }
                ResidentMappingKind::PrivateBoundary { bytes } => {
                    metal::GgufMappingOwnership::PrivateCow {
                        shadow_bytes: bytes,
                    }
                }
            };
            backings.push(
                ctx.gguf_no_copy_window_with_ownership(
                    mmap,
                    w.shard_idx,
                    if local { 0 } else { w.mmap_offset as usize },
                    w.length,
                    16,
                    ownership,
                )
                .map(|backing| (backing, local))?,
            );
        }
        let mut banks = Vec::new();
        let mut dense = BTreeMap::new();
        for t in requests {
            let mut tensor = None;
            for (planned, (backing, local)) in windows.iter().zip(&backings) {
                let w = &planned.window;
                if w.shard_idx == t.shard_idx
                    && t.data_offset >= w.mmap_offset
                    && t.data_offset + t.n_bytes <= w.mmap_offset + w.length as u64
                {
                    let mut descriptor = t.clone();
                    if *local {
                        descriptor.data_offset -= w.mmap_offset;
                    }
                    tensor = backing.tensor(&descriptor)?.1;
                    break;
                }
            }
            let mut tensor =
                tensor.ok_or_else(|| invalid(format!("no direct view for {}", t.name)))?;
            if let Some((layer, projection)) = names::bank(&t.name) {
                // The shared GGUF parser drops trailing unit axes. Restore the
                // logical bank rank without changing the backing or its bytes.
                tensor.shape = if projection == "w2" {
                    vec![
                        p.config.intermediate as u64,
                        p.config.hidden as u64,
                        p.config.experts as u64,
                    ]
                } else {
                    vec![
                        p.config.hidden as u64,
                        p.config.intermediate as u64,
                        p.config.experts as u64,
                    ]
                };
                banks.push(PackedBank {
                    layer,
                    projection,
                    tensor,
                });
            } else if let Some(key) = names::dense(&t.name, p.layers) {
                tensor.shape = names::expected_shape(&t.name, &p.config)
                    .ok_or_else(|| invalid("missing dense geometry"))?;
                dense.insert(key, tensor);
            }
        }
        p.engram_context.revalidate(&g)?;
        let retained_resources = retained_resources::summarize(&banks, &dense, bookkeeping)?;
        Ok(Self {
            config: p.config,
            schedule: p.schedule,
            engram_hash: p.engram_hash,
            banks,
            dense,
            engram: DiskEngram::for_tables(g, p.engram_context),
            token_map: p.token_map,
            raw_pad: p.raw_pad,
            bytes: p.bytes,
            retained_resources,
        })
    }
}

#[cfg(test)]
mod tests;
