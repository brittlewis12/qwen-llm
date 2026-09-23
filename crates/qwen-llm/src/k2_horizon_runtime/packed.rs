use super::*;
use crate::k2_horizon_metal::checked_slice;
use crate::k2_horizon_plan::{GENERAL_CHUNK_TOKENS, MAX_CHUNK_TOKENS, PACKED_CHUNK_TOKENS};

/// Prefill topology override: `general` (default), `q8_lcpp`, or `serial`.
/// An explicit request that the loaded weights cannot satisfy is a load
/// error, never a silent substitution.
pub(super) const PREFILL_ENV: &str = "QWEN_K2_PREFILL";

/// Block projections per checkpoint: Q, K, V, O, gate, up, down x 36 layers.
const PROJECTIONS: usize = 36 * 7;

/// Logical temporary activation bytes per packed row (see `scratch_specs`).
pub(super) const ACTIVATION_BYTES_PER_ROW: u64 = 237572;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PrefillMode {
    /// One command per token; the decode path.
    Serial,
    /// Q8_0-only specialization: lcpp token-batched GEMV projections with
    /// per-row norm/RoPE/cache/attention. Bitwise-equal to `Serial` for any
    /// partition; retained as an explicit opt-in lineage, not the default.
    BatchQ8,
    /// General batched prefill for every admitted projection dtype: tiled
    /// mat-mat projections and row-parallel norm/RoPE/KV store/attention.
    General { chunk: usize },
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
    lcpp && count == PROJECTIONS && all_q8
}

/// Weight dtypes with a tiled mat-mat kernel in `encode_mat_mat_dispatch`
/// that the K2 structural admission can produce.
pub(super) fn general_projection_dtype(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_0
    )
}

pub(super) fn projection_dtypes(weights: &ResidentWeights) -> Vec<GgmlType> {
    weights
        .layers
        .iter()
        .flat_map(|w| {
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
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PrefillRequest {
    General,
    Q8Lcpp,
    Serial,
}

impl PrefillRequest {
    pub(super) fn parse(value: Option<&str>) -> Result<Self> {
        match value.map(str::trim) {
            None | Some("") | Some("general") => Ok(Self::General),
            Some("q8_lcpp") => Ok(Self::Q8Lcpp),
            Some("serial") => Ok(Self::Serial),
            Some(other) => Err(invalid(format!(
                "{PREFILL_ENV}={other:?} is not one of general, q8_lcpp, serial"
            ))),
        }
    }

    fn from_env() -> Result<Self> {
        Self::parse(std::env::var(PREFILL_ENV).ok().as_deref())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PrefillSelection {
    pub mode: PrefillMode,
    pub reason: String,
}

/// Largest-first chunk candidates; each halving keeps the tiled kernels fed
/// while shrinking temporary activations before falling back to serial.
const GENERAL_CANDIDATES: [usize; 8] = [GENERAL_CHUNK_TOKENS, 128, 64, 32, 16, 8, 4, 2];

/// Pure selection policy. `admits(rows)` answers whether scratch for a
/// `rows`-token chunk (plus one session) passes memory admission.
pub(super) fn select(
    request: PrefillRequest,
    dtypes: &[GgmlType],
    lcpp: bool,
    mut admits: impl FnMut(usize) -> Result<bool>,
) -> Result<PrefillSelection> {
    match request {
        PrefillRequest::Serial => Ok(PrefillSelection {
            mode: PrefillMode::Serial,
            reason: format!("requested by {PREFILL_ENV}=serial"),
        }),
        PrefillRequest::Q8Lcpp => {
            if !eligible(dtypes.iter().copied(), lcpp) {
                return Err(invalid(format!(
                    "{PREFILL_ENV}=q8_lcpp requires all {PROJECTIONS} block projections Q8_0 \
                     and lcpp matvec enabled"
                )));
            }
            if !admits(PACKED_CHUNK_TOKENS)? {
                return Err(invalid(format!(
                    "{PREFILL_ENV}=q8_lcpp scratch for {PACKED_CHUNK_TOKENS} rows is not admitted"
                )));
            }
            Ok(PrefillSelection {
                mode: PrefillMode::BatchQ8,
                reason: format!("requested by {PREFILL_ENV}=q8_lcpp"),
            })
        }
        PrefillRequest::General => {
            if dtypes.len() != PROJECTIONS {
                return Err(invalid(format!(
                    "expected {PROJECTIONS} block projections, found {}",
                    dtypes.len()
                )));
            }
            if let Some(dtype) = dtypes.iter().find(|&&d| !general_projection_dtype(d)) {
                return Ok(PrefillSelection {
                    mode: PrefillMode::Serial,
                    reason: format!("projection dtype {dtype:?} has no tiled mat-mat kernel"),
                });
            }
            for chunk in GENERAL_CANDIDATES {
                if admits(chunk)? {
                    return Ok(PrefillSelection {
                        mode: PrefillMode::General { chunk },
                        reason: if chunk == GENERAL_CHUNK_TOKENS {
                            "default".to_string()
                        } else {
                            format!(
                                "memory admission reduced chunk from {GENERAL_CHUNK_TOKENS} to {chunk}"
                            )
                        },
                    });
                }
            }
            Ok(PrefillSelection {
                mode: PrefillMode::Serial,
                reason: format!(
                    "memory admission denied scratch for every chunk down to 2 rows \
                     ({} bytes/row)",
                    ACTIVATION_BYTES_PER_ROW
                ),
            })
        }
    }
}

impl PrefillMode {
    /// Load-time selection: env request, projection inventory, and scratch
    /// admission priced together with one session. Logged once per load.
    pub(super) fn for_model(
        ctx: &MetalContext,
        weights: &ResidentWeights,
        session_bytes: &[u64],
    ) -> Result<Self> {
        let session_price = price_buffers(ctx, session_bytes)?;
        let dtypes = projection_dtypes(weights);
        let selection = select(
            PrefillRequest::from_env()?,
            &dtypes,
            crate::metal::mat_vec_q8_0_lcpp_enabled(),
            |rows| {
                let total = price_buffers(ctx, &scratch_bytes(rows)?)?
                    .checked_add(session_price)
                    .ok_or_else(|| invalid("prefill scratch price overflow"))?;
                Ok(evaluate_metal_memory_admission(
                    total,
                    RESERVE_BYTES,
                    ctx.memory_signals(),
                    true,
                )
                .admitted)
            },
        )?;
        let info = selection.mode.info(usize::MAX);
        tracing::info!(
            target: "qwen_diag",
            "k2 prefill: mode={} chunk_tokens={} temporary_activation_bytes={} reason={}",
            info.mode,
            info.chunk_tokens,
            info.temporary_activation_bytes,
            selection.reason,
        );
        Ok(selection.mode)
    }

    /// The pre-general default: a partition-invariant (bitwise) lineage, i.e.
    /// Q8 lcpp batching when eligible, otherwise serial. For tests that pin
    /// split/whole bitwise identity rather than tolerance.
    #[cfg(test)]
    pub(super) fn bitwise_lineage(weights: &ResidentWeights) -> Self {
        if eligible(
            projection_dtypes(weights).into_iter(),
            crate::metal::mat_vec_q8_0_lcpp_enabled(),
        ) {
            Self::BatchQ8
        } else {
            Self::Serial
        }
    }

    pub(super) fn is_general(self) -> bool {
        matches!(self, Self::General { .. })
    }

    fn max_chunk(self) -> usize {
        match self {
            Self::Serial => 1,
            Self::BatchQ8 => PACKED_CHUNK_TOKENS,
            Self::General { chunk } => chunk,
        }
    }

    pub(super) fn info(self, tokens: usize) -> K2PrefillInfo {
        let chunk_tokens = tokens.min(self.max_chunk());
        K2PrefillInfo {
            mode: match self {
                _ if chunk_tokens <= 1 => "serial_single_token",
                Self::BatchQ8 => "q8_lcpp_token_batch",
                Self::General { .. } => "general_matmat_batch",
                Self::Serial => "serial_single_token",
            },
            chunk_tokens,
            commands: if tokens == 0 {
                0
            } else {
                tokens.div_ceil(chunk_tokens)
            },
            temporary_activation_bytes: if chunk_tokens > 1 {
                ACTIVATION_BYTES_PER_ROW * chunk_tokens as u64
            } else {
                0
            },
        }
    }

    pub(super) fn chunk(self, weights: &ResidentWeights, tokens: usize) -> Result<usize> {
        match self {
            Self::Serial => Ok(1),
            Self::BatchQ8 => {
                if tokens <= 1 {
                    return Ok(1);
                }
                if !eligible(
                    projection_dtypes(weights).into_iter(),
                    crate::metal::mat_vec_q8_0_lcpp_enabled(),
                ) {
                    return Err(invalid(
                        "packed K2 requires Q8_0 block projections and lcpp matvec",
                    ));
                }
                Ok(tokens.min(PACKED_CHUNK_TOKENS))
            }
            Self::General { chunk } => {
                if !(2..=MAX_CHUNK_TOKENS).contains(&chunk) {
                    return Err(invalid("general K2 prefill chunk outside 2..=256"));
                }
                Ok(tokens.clamp(1, chunk))
            }
        }
    }
}

fn scratch_bytes(rows: usize) -> Result<Vec<u64>> {
    Ok(scratch_specs(rows)?
        .iter()
        .map(|(_, shape)| shape[0] * 4)
        .collect())
}

fn scratch_specs(rows: usize) -> Result<Vec<(GgmlType, Vec<u64>)>> {
    if !(2..=MAX_CHUNK_TOKENS).contains(&rows) {
        return Err(invalid("packed scratch rows must fit 2..=256"));
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
        let price = price_buffers(ctx, &scratch_bytes(rows)?)?;
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
        if count == 0 || count > MAX_CHUNK_TOKENS {
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
