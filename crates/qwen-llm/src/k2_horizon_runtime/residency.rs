use super::*;
use crate::gguf::GgufShardStamp;
use crate::k2_horizon::K2HorizonModel;
use crate::metal::{
    MetalGgufBacking, RetainedStorageDisposition, RetainedStorageFallback, RetainedStoragePlan,
    plan_retained_storage,
};
use crate::tensor::TensorDesc;
use std::collections::BTreeMap;

/// CPU-only inspection/planning. No checkpoint hash whitelist and no Metal
/// context is required. The source borrow prevents safe descriptor mutation.
pub struct K2RuntimePlan<'a> {
    source: &'a GgufFile,
    model: K2HorizonModel<'a>,
    retained: RetainedStoragePlan,
    stamps: Vec<GgufShardStamp>,
    capacity: u32,
    pub(super) session: SessionMemoryPlan,
}

impl<'a> K2RuntimePlan<'a> {
    pub fn inspect(
        source: &'a GgufFile,
        capacity: u32,
        page_size: usize,
        max_buffer_bytes: usize,
    ) -> Result<Self> {
        Self::inspect_with_storage(
            source,
            capacity,
            page_size,
            max_buffer_bytes,
            K2KvStorage::F16,
        )
    }

    pub(super) fn inspect_with_storage(
        source: &'a GgufFile,
        capacity: u32,
        page_size: usize,
        max_buffer_bytes: usize,
        storage: K2KvStorage,
    ) -> Result<Self> {
        let stamps = source.revalidate_retained_shard_stamps()?;
        let model = K2HorizonModel::from_gguf(source)?;
        validate_embedding(model.token_embedding.dtype)?;
        let session = SessionMemoryPlan::new(&model.config, capacity, storage)?;
        if session
            .buffer_bytes()
            .iter()
            .any(|&bytes| bytes > max_buffer_bytes as u64)
        {
            return Err(invalid("session buffer exceeds device maximum"));
        }
        let requests = source.tensors.iter().collect::<Vec<_>>();
        let retained = plan_retained_storage(
            &source.shard_mapped_lengths(),
            &requests,
            page_size,
            max_buffer_bytes,
            32,
        )?;
        retained_buffer_bytes(&retained)?;
        if retained.entries.len() != source.tensors.len() {
            return Err(invalid("retained tensor census mismatch"));
        }
        let plan = Self {
            source,
            model,
            retained,
            stamps,
            capacity,
            session,
        };
        plan.revalidate_source()?;
        Ok(plan)
    }

    pub fn config(&self) -> &K2HorizonConfig {
        &self.model.config
    }
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
    pub fn weight_payload_bytes(&self) -> u64 {
        self.model.weight_payload_bytes
    }
    pub fn weight_buffer_bytes(&self) -> Result<Vec<u64>> {
        retained_buffer_bytes(&self.retained)
    }
    pub fn session_buffer_bytes(&self) -> Vec<u64> {
        self.session.buffer_bytes()
    }

    pub(super) fn revalidate_source(&self) -> Result<()> {
        check_stamps(
            &self.stamps,
            &self.source.revalidate_retained_shard_stamps()?,
        )
    }
}

fn check_stamps(expected: &[GgufShardStamp], actual: &[GgufShardStamp]) -> Result<()> {
    if expected != actual {
        return Err(invalid(
            "GGUF source stamps changed; file mutation is unsupported",
        ));
    }
    Ok(())
}

fn validate_embedding(dtype: GgmlType) -> Result<()> {
    if !matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q4_K
            | GgmlType::Q6_K
            | GgmlType::Q8_0
    ) {
        return Err(invalid(format!(
            "native K2 embedding gather does not support {dtype:?}"
        )));
    }
    Ok(())
}

fn retained_buffer_bytes(plan: &RetainedStoragePlan) -> Result<Vec<u64>> {
    let mut bytes = plan
        .windows
        .iter()
        .map(|window| window.length as u64)
        .collect::<Vec<_>>();
    for entry in &plan.entries {
        match entry.disposition {
            RetainedStorageDisposition::View { .. } => {}
            RetainedStorageDisposition::CopyFallback {
                reason: RetainedStorageFallback::FinalPartialPage,
            } => bytes.push(entry.n_bytes),
            other => {
                return Err(invalid(format!(
                    "{} requires disallowed retained disposition {other:?}",
                    entry.name
                )));
            }
        }
    }
    Ok(bytes)
}

pub(super) struct LayerWeights {
    pub attention_norm: MetalTensor,
    pub query: MetalTensor,
    pub key: MetalTensor,
    pub value: MetalTensor,
    pub attention_output: MetalTensor,
    pub feed_forward_norm: MetalTensor,
    pub gate: MetalTensor,
    pub up: MetalTensor,
    pub down: MetalTensor,
}

pub(super) struct ResidentWeights {
    pub embedding: MetalTensor,
    pub output_norm: MetalTensor,
    pub output: MetalTensor,
    pub layers: Vec<LayerWeights>,
    // Retain explicit backing owners in addition to the tensor buffer handles.
    _backings: Vec<MetalGgufBacking>,
}

impl ResidentWeights {
    pub fn realize(ctx: &MetalContext, plan: &K2RuntimePlan<'_>) -> Result<Self> {
        let source = plan.source;
        let mut backings = Vec::with_capacity(plan.retained.windows.len());
        for window in &plan.retained.windows {
            let mmap = source
                .retained_shard_mmap(window.shard_idx)
                .ok_or_else(|| invalid("missing retained shard"))?;
            backings.push(
                ctx.gguf_no_copy_window(
                    mmap,
                    window.shard_idx,
                    usize::try_from(window.mmap_offset)
                        .map_err(|_| invalid("window offset exceeds usize"))?,
                    window.length,
                    32,
                )?,
            );
        }
        let mut tensors = BTreeMap::new();
        for (index, (entry, desc)) in plan
            .retained
            .entries
            .iter()
            .zip(&source.tensors)
            .enumerate()
        {
            if entry.request_index != index
                || entry.name != desc.name
                || entry.shard_idx != desc.shard_idx
                || entry.data_offset != desc.data_offset
                || entry.n_bytes != desc.n_bytes
            {
                return Err(invalid("retained descriptor drift"));
            }
            let tensor = match entry.disposition {
                RetainedStorageDisposition::View {
                    window_index,
                    buffer_offset,
                } => {
                    let backing = backings
                        .get(window_index)
                        .ok_or_else(|| invalid("missing retained window"))?;
                    let (eligibility, tensor) = backing.tensor(desc)?;
                    let tensor = tensor.ok_or_else(|| {
                        invalid(format!("retained view rejected: {eligibility:?}"))
                    })?;
                    if tensor.offset != buffer_offset {
                        return Err(invalid("retained offset drift"));
                    }
                    tensor
                }
                RetainedStorageDisposition::CopyFallback {
                    reason: RetainedStorageFallback::FinalPartialPage,
                } => MetalTensor::copied_gguf_weight(ctx, desc, source.try_slice(desc)?)?,
                _ => return Err(invalid("disallowed retained fallback or alias")),
            };
            validate_realized(desc, &tensor)?;
            if tensors.insert(desc.name.as_str(), tensor).is_some() {
                return Err(invalid("duplicate tensor role"));
            }
        }
        let mut take = |desc: &TensorDesc| {
            tensors
                .remove(desc.name.as_str())
                .ok_or_else(|| invalid(format!("missing realized {}", desc.name)))
        };
        let model = &plan.model;
        let embedding = take(model.token_embedding)?;
        let output_norm = take(model.output_norm)?;
        let output = take(model.output)?;
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            layers.push(LayerWeights {
                attention_norm: take(layer.attention_norm)?,
                query: take(layer.query)?,
                key: take(layer.key)?,
                value: take(layer.value)?,
                attention_output: take(layer.attention_output)?,
                feed_forward_norm: take(layer.feed_forward_norm)?,
                gate: take(layer.feed_forward_gate)?,
                up: take(layer.feed_forward_up)?,
                down: take(layer.feed_forward_down)?,
            });
        }
        if !tensors.is_empty() {
            return Err(invalid("unused realized weight roles"));
        }
        Ok(Self {
            embedding,
            output_norm,
            output,
            layers,
            _backings: backings,
        })
    }
}

fn validate_realized(desc: &TensorDesc, tensor: &MetalTensor) -> Result<()> {
    if tensor.dtype != desc.dtype
        || tensor.shape != desc.shape
        || tensor.n_bytes() != desc.n_bytes
        || tensor.is_writable()
        || !tensor.offset.is_multiple_of(32)
        || tensor
            .offset
            .checked_add(desc.n_bytes)
            .is_none_or(|end| end > tensor.buffer.length() as u64)
    {
        return Err(invalid(format!("realized tensor drift for {}", desc.name)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_gather_admission_is_narrower_than_structural_matrix_admission() {
        for dtype in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::Q8_0,
            GgmlType::Q4_K,
            GgmlType::Q6_K,
        ] {
            assert!(validate_embedding(dtype).is_ok());
        }
        for dtype in [GgmlType::Q5_K, GgmlType::Q4_0, GgmlType::IQ4_NL] {
            assert!(validate_embedding(dtype).is_err());
        }
    }

    #[test]
    fn retained_accounting_prices_windows_and_only_final_page_fallback() {
        let desc = |name: &str, offset| TensorDesc {
            name: name.into(),
            shape: vec![16],
            dtype: GgmlType::F32,
            shard_idx: 0,
            data_offset: offset,
            n_bytes: 64,
        };
        let a = desc("a", 0);
        let b = desc("b", 96);
        let mut plan = plan_retained_storage(&[160], &[&a, &b], 64, 128, 32).unwrap();
        assert_eq!(retained_buffer_bytes(&plan).unwrap(), vec![64, 64]);
        plan.entries[1].disposition = RetainedStorageDisposition::CopyFallback {
            reason: RetainedStorageFallback::BindingMisalignment,
        };
        assert!(retained_buffer_bytes(&plan).is_err());
        plan.entries[1].disposition = RetainedStorageDisposition::Alias {
            source_request_index: 0,
        };
        assert!(retained_buffer_bytes(&plan).is_err());
    }

    #[test]
    fn source_stamp_comparison_is_exact_but_not_called_authentication() {
        let stamp = GgufShardStamp {
            shard_idx: 0,
            path: "test.gguf".into(),
            device: 1,
            inode: 2,
            size: 1024,
            mtime_sec: 10,
            mtime_nsec: 11,
            ctime_sec: 12,
            ctime_nsec: 13,
        };
        assert!(check_stamps(&[stamp.clone()], &[stamp.clone()]).is_ok());
        for modify in [
            |s: &mut GgufShardStamp| s.inode += 1,
            |s: &mut GgufShardStamp| s.size += 1,
            |s: &mut GgufShardStamp| s.mtime_nsec += 1,
            |s: &mut GgufShardStamp| s.ctime_sec += 1,
        ] {
            let mut changed = stamp.clone();
            modify(&mut changed);
            assert!(check_stamps(&[stamp.clone()], &[changed]).is_err());
        }
        assert!(check_stamps(&[stamp], &[]).is_err());
    }
}
