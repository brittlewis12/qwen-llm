use super::*;
use crate::k2_horizon_metal::checked_slice;
use crate::k2_horizon_plan::PACKED_CHUNK_TOKENS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PrefillMode {
    Serial,
    BatchQ8,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct K2PrefillInfo {
    pub mode: &'static str,
    pub chunk_tokens: usize,
    pub commands: usize,
    /// Logical additional activation storage, excluding page pricing and reserve.
    pub temporary_activation_bytes: u64,
}

fn eligible(types: impl Iterator<Item = GgmlType>, lcpp: bool) -> bool {
    let (count, all_q8) = types.fold((0, true), |(count, q8), dtype| {
        (count + 1, q8 && dtype == GgmlType::Q8_0)
    });
    lcpp && count == 36 * 7 && all_q8
}

impl PrefillMode {
    pub(super) fn for_weights(weights: &ResidentWeights) -> Self {
        let types = weights.layers.iter().flat_map(|w| {
            [
                &w.query,
                &w.key,
                &w.value,
                &w.attention_output,
                &w.gate,
                &w.up,
                &w.down,
            ]
            .map(|t| t.dtype)
        });
        if eligible(types, crate::metal::mat_vec_q8_0_lcpp_enabled()) {
            Self::BatchQ8
        } else {
            Self::Serial
        }
    }

    pub(super) fn info(self, tokens: usize) -> K2PrefillInfo {
        let chunk_tokens = tokens.min(if self == Self::BatchQ8 {
            PACKED_CHUNK_TOKENS
        } else {
            1
        });
        K2PrefillInfo {
            mode: if chunk_tokens > 1 {
                "q8_lcpp_token_batch"
            } else {
                "serial_single_token"
            },
            chunk_tokens,
            commands: if tokens == 0 {
                0
            } else {
                tokens.div_ceil(chunk_tokens)
            },
            temporary_activation_bytes: if chunk_tokens > 1 {
                237572 * chunk_tokens as u64
            } else {
                0
            },
        }
    }

    pub(super) fn chunk(self, weights: &ResidentWeights, tokens: usize) -> Result<usize> {
        match self {
            Self::Serial => {
                let _ = (weights, tokens);
                Ok(1)
            }
            Self::BatchQ8 => {
                if tokens <= 1 {
                    return Ok(1);
                }
                if Self::for_weights(weights) != Self::BatchQ8 {
                    return Err(invalid(
                        "packed K2 requires Q8_0 block projections and lcpp matvec",
                    ));
                }
                Ok(tokens.min(PACKED_CHUNK_TOKENS))
            }
        }
    }
}

fn scratch_specs(rows: usize) -> Result<Vec<(GgmlType, Vec<u64>)>> {
    if !(2..=PACKED_CHUNK_TOKENS).contains(&rows) {
        return Err(invalid("packed scratch rows must fit 2..=32"));
    }
    Ok(SessionMemoryPlan {
        cache_bytes: 0,
        storage: K2KvStorage::F16,
    }
    .specs()
    .into_iter()
    .take(11)
    .map(|(dtype, shape)| (dtype, vec![shape.iter().product::<u64>() * rows as u64]))
    .collect())
}

pub(super) struct PackedScratch {
    pub(super) buffers: SessionBuffers,
}

impl PackedScratch {
    pub(super) fn allocate(ctx: &MetalContext, base: &SessionBuffers, rows: usize) -> Result<Self> {
        let specs = scratch_specs(rows)?;
        let bytes = specs
            .iter()
            .map(|(_, shape)| shape[0] * 4)
            .collect::<Vec<_>>();
        let price = price_buffers(ctx, &bytes)?;
        let _transaction = ctx.begin_allocation_transaction();
        admit(ctx, price)?;
        let before = ctx.current_allocated_size();
        let mut tensors = Vec::new();
        for (dtype, shape) in specs {
            let tensor = MetalTensor::zeros_dtype_unstaged(ctx, shape.clone(), dtype)?;
            if tensor.dtype != dtype || tensor.shape != shape || !tensor.is_writable() {
                return Err(invalid("packed scratch descriptor drift"));
            }
            validate_cpu_layout(
                tensor.buffer.storageMode() == MTLStorageMode::Shared,
                tensor.offset,
                tensor.n_bytes(),
                tensor.buffer.length() as u64,
            )?;
            tensors.push(tensor);
        }
        tensors.push(base.logits.view_subrange(0, base.logits.shape.clone()));
        tensors.push(base.cache.view_bytes(0, base.cache.shape.clone()));
        reconcile(ctx, before, price)?;
        Ok(Self {
            buffers: SessionBuffers::from_tensors(tensors.into_iter().map(Ok))?,
        })
    }
}

impl SessionBuffers {
    pub(super) fn rows(&self, first: usize, count: usize) -> Result<Self> {
        if count == 0 || count > PACKED_CHUNK_TOKENS {
            return Err(invalid("invalid activation row count"));
        }
        let slice = |tensor: &MetalTensor, width: u64, singleton: &[u64], dtype| {
            let offset = (first as u64)
                .checked_mul(width)
                .ok_or_else(|| invalid("row offset overflow"))?;
            let shape = if count == 1 {
                singleton.to_vec()
            } else {
                vec![width * count as u64]
            };
            Ok(checked_slice(tensor, offset, shape, dtype)?)
        };
        let tensors = [
            slice(&self.id, 1, &[1], GgmlType::I32),
            slice(&self.residual, 4096, &[4096], GgmlType::F32),
            slice(&self.norm, 4096, &[4096], GgmlType::F32),
            slice(&self.query, 4096, &[128, 32], GgmlType::F32),
            slice(&self.key, 1024, &[128, 8], GgmlType::F32),
            slice(&self.value, 1024, &[128, 8], GgmlType::F32),
            slice(&self.attention, 4096, &[128, 32], GgmlType::F32),
            slice(&self.projection, 4096, &[4096], GgmlType::F32),
            slice(&self.gate, 12288, &[12288], GgmlType::F32),
            slice(&self.up, 12288, &[12288], GgmlType::F32),
            slice(&self.gated, 12288, &[12288], GgmlType::F32),
            Ok(self.logits.view_subrange(0, self.logits.shape.clone())),
            Ok(self.cache.view_bytes(0, self.cache.shape.clone())),
        ];
        Self::from_tensors(tensors.into_iter())
    }

    pub(super) fn upload_ids(&self, tokens: &[u32]) -> Result<()> {
        if self.id.offset != 0 || self.id.n_elements() != tokens.len() as u64 {
            return Err(invalid(
                "token upload requires exact zero-offset ID storage",
            ));
        }
        for (index, &token) in tokens.iter().enumerate() {
            unsafe {
                self.id
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<i32>()
                    .add(index)
                    .write(token as i32);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
