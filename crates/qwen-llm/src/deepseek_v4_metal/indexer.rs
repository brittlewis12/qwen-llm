//! Lightning indexer, top-k selection, and compressed sparse attention.

use super::*;

#[cfg(feature = "dsv4-diagnostics")]
pub(super) const DEEPSEEK_V4_FP4_STATUS_UNAVAILABLE: i32 = i32::MIN;

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MultigroupSelectorGeometry {
    pub(super) forward_limit: usize,
    pub(super) physical_capacity_rows: usize,
    pub(super) max_visible_rows: usize,
}

impl DeepSeekV4MultigroupSelectorGeometry {
    pub fn forward_limit(self) -> usize {
        self.forward_limit
    }

    pub fn physical_capacity_rows(self) -> usize {
        self.physical_capacity_rows
    }

    pub fn max_visible_rows(self) -> usize {
        self.max_visible_rows
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MultigroupSelectorTelemetry {
    pub(super) sealed: bool,
    pub(super) multigroup_invocations: u64,
    pub(super) ineligible_radix4_invocations: u64,
}

impl DeepSeekV4MultigroupSelectorTelemetry {
    pub fn sealed(self) -> bool {
        self.sealed
    }

    pub fn multigroup_invocations(self) -> u64 {
        self.multigroup_invocations
    }

    pub fn ineligible_radix4_invocations(self) -> u64 {
        self.ineligible_radix4_invocations
    }
}

pub(super) const DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS: usize =
    DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS * 3;

pub(super) const DEEPSEEK_V4_HCA_HISTORY_CAPACITY_ROWS: usize =
    DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS * 2;

pub(super) const DEEPSEEK_V4_CSA_TOP_K: usize = 512;

pub(super) const DEEPSEEK_V4_HCA_TILE_ROWS: usize = 512;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepSeekV4IndexerContract {
    LlamaCppB10222F16HadamardV1,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DeepSeekV4Fp4ScorePlan {
    F16Only,
    Fp4Only,
    Paired {
        consume: DeepSeekV4Fp4SelectionSource,
    },
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4Fp4ScorePlan {
    pub(super) fn runs_f16(self) -> bool {
        matches!(self, Self::F16Only | Self::Paired { .. })
    }

    pub(super) fn runs_fp4(self) -> bool {
        matches!(self, Self::Fp4Only | Self::Paired { .. })
    }

    pub(super) fn consumes_fp4(self) -> bool {
        matches!(
            self,
            Self::Fp4Only
                | Self::Paired {
                    consume: DeepSeekV4Fp4SelectionSource::Fp4,
                }
        )
    }

    pub(super) fn kind(self) -> DeepSeekV4Fp4ScorePlanKind {
        match self {
            Self::F16Only => DeepSeekV4Fp4ScorePlanKind::F16Only,
            Self::Fp4Only => DeepSeekV4Fp4ScorePlanKind::Fp4Only,
            Self::Paired { .. } => DeepSeekV4Fp4ScorePlanKind::Paired,
        }
    }

    pub(super) fn consumed_source(self) -> DeepSeekV4Fp4SelectionSource {
        match self {
            Self::F16Only
            | Self::Paired {
                consume: DeepSeekV4Fp4SelectionSource::F16,
            } => DeepSeekV4Fp4SelectionSource::F16,
            Self::Fp4Only
            | Self::Paired {
                consume: DeepSeekV4Fp4SelectionSource::Fp4,
            } => DeepSeekV4Fp4SelectionSource::Fp4,
        }
    }
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum DeepSeekV4Fp4SessionMode {
    #[default]
    F16Authoritative,
    PairedCounterfactual,
    Fp4OnlyExperimental,
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4Fp4SessionMode {
    pub(super) fn score_plan(self, audit_active: bool) -> DeepSeekV4Fp4ScorePlan {
        match (self, audit_active) {
            (Self::F16Authoritative, false) => DeepSeekV4Fp4ScorePlan::F16Only,
            (Self::F16Authoritative, true) => DeepSeekV4Fp4ScorePlan::Paired {
                consume: DeepSeekV4Fp4SelectionSource::F16,
            },
            (Self::PairedCounterfactual, _) => DeepSeekV4Fp4ScorePlan::Paired {
                consume: DeepSeekV4Fp4SelectionSource::Fp4,
            },
            (Self::Fp4OnlyExperimental, false) => DeepSeekV4Fp4ScorePlan::Fp4Only,
            (Self::Fp4OnlyExperimental, true) => DeepSeekV4Fp4ScorePlan::Paired {
                consume: DeepSeekV4Fp4SelectionSource::Fp4,
            },
        }
    }

    pub(super) fn is_counterfactual(self) -> bool {
        self != Self::F16Authoritative
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DeepSeekV4CompressorPublication {
    Attention,
    IndexerHadamard,
}

#[cfg(feature = "dsv4-diagnostics")]
pub(super) struct DeepSeekV4IndexerFp4Sidecar {
    pub(super) enabled: bool,
    pub(super) capacity_rows: usize,
    pub(super) values: MetalTensor,
    pub(super) scales: MetalTensor,
    pub(super) status: MetalTensor,
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4IndexerFp4Sidecar {
    pub(super) fn new(
        ctx: &MetalContext,
        capacity_rows: usize,
    ) -> Result<Self, DeepSeekV4MetalError> {
        if capacity_rows == 0 || u32::try_from(capacity_rows).is_err() {
            return invalid("indexer FP4 sidecar capacity must be nonzero and fit u32");
        }
        let unavailable = vec![DEEPSEEK_V4_FP4_STATUS_UNAVAILABLE; capacity_rows];
        Ok(Self {
            enabled: false,
            capacity_rows,
            values: MetalTensor::zeros_dtype(
                ctx,
                vec![
                    crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES as u64,
                    capacity_rows as u64,
                ],
                GgmlType::I8,
            )?,
            scales: MetalTensor::zeros_dtype(
                ctx,
                vec![
                    crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES as u64,
                    capacity_rows as u64,
                ],
                GgmlType::I8,
            )?,
            status: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&unavailable),
                vec![capacity_rows as u64],
                GgmlType::I32,
            )?,
        })
    }

    pub(super) fn enable(&mut self) {
        self.enabled = true;
    }

    pub(super) fn disable(&mut self) {
        self.enabled = false;
    }

    pub(super) fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub(super) fn invalidate(&self) -> Result<(), DeepSeekV4MetalError> {
        host_write_i32(
            &self.status,
            &vec![DEEPSEEK_V4_FP4_STATUS_UNAVAILABLE; self.capacity_rows],
            "indexer FP4 sidecar status",
        )
    }

    pub(super) fn encode_row(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        source: &MetalTensor,
        row: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        self.encode_rows(ctx, enc, source, row, 1)
    }

    pub(super) fn encode_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        source: &MetalTensor,
        first_row: usize,
        row_count: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        if !self.enabled {
            return Ok(());
        }
        if row_count == 0
            || first_row
                .checked_add(row_count)
                .is_none_or(|end| end > self.capacity_rows)
        {
            return invalid(format!(
                "indexer FP4 sidecar rows {first_row}..{} exceed capacity {}",
                first_row.saturating_add(row_count),
                self.capacity_rows,
            ));
        }
        let mut value_shape = source.shape.clone();
        let mut scale_shape = source.shape.clone();
        if value_shape.is_empty() {
            return invalid("indexer FP4 sidecar source has no row width");
        }
        value_shape[0] = crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES as u64;
        scale_shape[0] = crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES as u64;
        let values = raw_i8_subview(
            &self.values,
            checked_mul(
                first_row,
                crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES,
                "indexer FP4 sidecar value offset",
            )?,
            value_shape,
            "indexer FP4 sidecar value rows",
        )?;
        let scales = raw_i8_subview(
            &self.scales,
            checked_mul(
                first_row,
                crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES,
                "indexer FP4 sidecar scale offset",
            )?,
            scale_shape,
            "indexer FP4 sidecar scale rows",
        )?;
        let status = self
            .status
            .view_subrange(first_row as u64, vec![row_count as u64]);
        encode_pack_indexer_fp4_rows_shadow(ctx, enc, source, &values, &scales, &status, row_count)
    }
}

pub(super) struct DeepSeekV4CompressorFrontier {
    pub(super) ratio: usize,
    pub(super) head_dim: usize,
    pub(super) width: usize,
    pub(super) rows: usize,
    pub(super) capacity_rows: usize,
    pub(super) publication: DeepSeekV4CompressorPublication,
    pub(super) kv_state: MetalTensor,
    pub(super) score_state: MetalTensor,
    pub(super) projected_kv: MetalTensor,
    pub(super) projected_score: MetalTensor,
    pub(super) pooled: MetalTensor,
    pub(super) normalized: MetalTensor,
    pub(super) published: MetalTensor,
    #[cfg(feature = "dsv4-diagnostics")]
    pub(super) fp4_sidecar: Option<DeepSeekV4IndexerFp4Sidecar>,
}

impl DeepSeekV4CompressorFrontier {
    pub(super) fn new(
        ctx: &MetalContext,
        ratio: usize,
        head_dim: usize,
        publication: DeepSeekV4CompressorPublication,
        capacity_rows: usize,
    ) -> Result<Self, DeepSeekV4MetalError> {
        if !matches!(ratio, 4 | 128) || head_dim == 0 {
            return invalid(format!(
                "compressor frontier requires ratio 4 or 128 and a nonzero head dimension, got ratio={ratio} head_dim={head_dim}"
            ));
        }
        if publication == DeepSeekV4CompressorPublication::IndexerHadamard && head_dim != 128 {
            return invalid("indexer publication requires exactly 128 dimensions");
        }
        if capacity_rows == 0
            || !capacity_rows.is_multiple_of(DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS)
        {
            return invalid(format!(
                "compressor publication capacity {capacity_rows} is not a nonzero slab multiple"
            ));
        }
        let (width, rows, state_elements) = compressor_frontier_geometry(ratio, head_dim)?;
        let zeros = vec![0.0f32; state_elements];
        let negative_infinity = vec![f32::NEG_INFINITY; state_elements];
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_sidecar = (publication == DeepSeekV4CompressorPublication::IndexerHadamard)
            .then(|| DeepSeekV4IndexerFp4Sidecar::new(ctx, capacity_rows))
            .transpose()?;
        Ok(Self {
            ratio,
            head_dim,
            width,
            rows,
            capacity_rows,
            publication,
            kv_state: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&zeros),
                vec![width as u64, rows as u64],
                GgmlType::F32,
            )?,
            score_state: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&negative_infinity),
                vec![width as u64, rows as u64],
                GgmlType::F32,
            )?,
            projected_kv: MetalTensor::zeros_f32(ctx, vec![width as u64])?,
            projected_score: MetalTensor::zeros_f32(ctx, vec![width as u64])?,
            pooled: MetalTensor::zeros_f32(ctx, vec![head_dim as u64])?,
            normalized: MetalTensor::zeros_f32(ctx, vec![head_dim as u64])?,
            published: MetalTensor::zeros_f16(ctx, vec![head_dim as u64, capacity_rows as u64])?,
            #[cfg(feature = "dsv4-diagnostics")]
            fp4_sidecar,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        input: &MetalTensor,
        kv_weight: &MetalTensor,
        score_weight: &MetalTensor,
        ape: &MetalTensor,
        norm_weight: &MetalTensor,
        position: u32,
        hidden_size: usize,
        rope: DeepSeekV4RopeParameters,
        rms_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_compressor_frontier")?;
        validate_f32(input, &[hidden_size as u64], false, "compressor input")?;
        validate_matvec_weight(kv_weight, hidden_size, self.width, "compressor KV weight")?;
        validate_matvec_weight(
            score_weight,
            hidden_size,
            self.width,
            "compressor score weight",
        )?;
        validate_f32(
            ape,
            &[self.width as u64, self.ratio as u64],
            false,
            "compressor APE",
        )?;
        validate_f32(
            norm_weight,
            &[self.head_dim as u64],
            false,
            "compressor norm weight",
        )?;
        validate_f32(
            &self.kv_state,
            &[self.width as u64, self.rows as u64],
            true,
            "compressor KV state",
        )?;
        validate_f32(
            &self.score_state,
            &[self.width as u64, self.rows as u64],
            true,
            "compressor score state",
        )?;
        for (tensor, name) in [
            (&self.projected_kv, "projected compressor KV"),
            (&self.projected_score, "projected compressor score"),
        ] {
            validate_f32(tensor, &[self.width as u64], true, name)?;
        }
        validate_f32(
            &self.pooled,
            &[self.head_dim as u64],
            true,
            "pooled compressor row",
        )?;
        validate_f32(
            &self.normalized,
            &[self.head_dim as u64],
            true,
            "normalized compressor row",
        )?;
        validate_f16(
            &self.published,
            &[self.head_dim as u64, self.capacity_rows as u64],
            true,
            "published compressor rows",
        )?;

        let fused = deepseek_v4_decode_compressor_fused_enabled()
            && kv_weight.dtype == GgmlType::Q8_0
            && score_weight.dtype == GgmlType::Q8_0
            && crate::metal::mat_vec_q8_0_lcpp_enabled();
        if fused {
            validate_ds4_rope(rope, self.head_dim, rope.rotary_dim)?;
            validate_eps(rms_eps, "compressor RMSNorm epsilon")?;
            let (following_position, state_row, published_row) = self.step(position)?;
            let ape_row = ape.view_subrange(
                ((position as usize % self.ratio) * self.width) as u64,
                vec![self.width as u64],
            );
            let state_offset = checked_mul(
                state_row,
                self.width,
                "fused compressor frontier row offset",
            )?;
            let kv_state_row = self
                .kv_state
                .view_subrange(state_offset as u64, vec![self.width as u64]);
            let score_state_row = self
                .score_state
                .view_subrange(state_offset as u64, vec![self.width as u64]);
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| {
                eprintln!(
                    "deepseek_v4: decode Q8 compressor projections and frontier write run one fused dispatch; rollback=QWEN_DSV4_DECODE_COMPRESSOR_FUSED=0"
                );
            });
            crate::metal::encode_ds4_compressor_pair_q8_0_f32(
                ctx,
                enc,
                kv_weight,
                score_weight,
                input,
                &self.projected_score,
                &ape_row,
                &kv_state_row,
                &score_state_row,
                hidden_size,
                self.width,
            )?;
            return self.encode_after_frontier_write(
                ctx,
                enc,
                norm_weight,
                following_position,
                published_row,
                rope,
                rms_eps,
            );
        }

        encode_projection(
            ctx,
            enc,
            kv_weight,
            input,
            &self.projected_kv,
            hidden_size,
            self.width,
            "compressor KV",
        )?;
        encode_projection(
            ctx,
            enc,
            score_weight,
            input,
            &self.projected_score,
            hidden_size,
            self.width,
            "compressor score",
        )?;
        self.encode_projected(
            ctx,
            enc,
            &self.projected_kv,
            &self.projected_score,
            ape,
            norm_weight,
            position,
            rope,
            rms_eps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_projected(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        projected_kv: &MetalTensor,
        projected_score: &MetalTensor,
        ape: &MetalTensor,
        norm_weight: &MetalTensor,
        position: u32,
        rope: DeepSeekV4RopeParameters,
        rms_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_compressor_frontier_projected")?;
        validate_f32(
            projected_kv,
            &[self.width as u64],
            false,
            "projected compressor KV",
        )?;
        validate_f32(
            projected_score,
            &[self.width as u64],
            false,
            "projected compressor score",
        )?;
        validate_f32(
            ape,
            &[self.width as u64, self.ratio as u64],
            false,
            "compressor APE",
        )?;
        validate_f32(
            norm_weight,
            &[self.head_dim as u64],
            false,
            "compressor norm weight",
        )?;
        validate_f32(
            &self.kv_state,
            &[self.width as u64, self.rows as u64],
            true,
            "compressor KV state",
        )?;
        validate_f32(
            &self.score_state,
            &[self.width as u64, self.rows as u64],
            true,
            "compressor score state",
        )?;
        validate_f32(
            &self.pooled,
            &[self.head_dim as u64],
            true,
            "pooled compressor row",
        )?;
        validate_f32(
            &self.normalized,
            &[self.head_dim as u64],
            true,
            "normalized compressor row",
        )?;
        validate_f16(
            &self.published,
            &[self.head_dim as u64, self.capacity_rows as u64],
            true,
            "published compressor rows",
        )?;
        validate_ds4_rope(rope, self.head_dim, rope.rotary_dim)?;
        validate_eps(rms_eps, "compressor RMSNorm epsilon")?;

        let (following_position, state_row, published_row) = self.step(position)?;
        let ape_row = ape.view_subrange(
            ((position as usize % self.ratio) * self.width) as u64,
            vec![self.width as u64],
        );
        encode_compressor_frontier_write(
            ctx,
            enc,
            projected_kv,
            projected_score,
            &ape_row,
            &self.kv_state,
            &self.score_state,
            self.width,
            state_row,
        )?;
        self.encode_after_frontier_write(
            ctx,
            enc,
            norm_weight,
            following_position,
            published_row,
            rope,
            rms_eps,
        )
    }

    pub(super) fn step(
        &self,
        position: u32,
    ) -> Result<(u32, usize, Option<usize>), DeepSeekV4MetalError> {
        let following_position = position
            .checked_add(1)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("compressor position overflow".into()))?;
        let boundary = (following_position as usize).is_multiple_of(self.ratio);
        let published_row = if boundary {
            let row = following_position as usize / self.ratio - 1;
            if row >= self.capacity_rows {
                return invalid(format!(
                    "compressor published row {row} exceeds the allocated {}-row history",
                    self.capacity_rows
                ));
            }
            Some(row)
        } else {
            None
        };
        let state_row = if self.ratio == 4 {
            self.ratio + position as usize % self.ratio
        } else {
            position as usize % self.ratio
        };
        Ok((following_position, state_row, published_row))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_after_frontier_write(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        norm_weight: &MetalTensor,
        following_position: u32,
        published_row: Option<usize>,
        rope: DeepSeekV4RopeParameters,
        rms_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let Some(published_row) = published_row else {
            return Ok(());
        };
        encode_compressor_pool(
            ctx,
            enc,
            &self.kv_state,
            &self.score_state,
            &self.pooled,
            self.ratio,
            self.head_dim,
            self.width,
            self.rows,
        )?;
        encode_rms_norm_mul_f32(
            ctx,
            enc,
            &self.pooled,
            norm_weight,
            &self.normalized,
            rms_eps,
        )?;
        let start_position = following_position - self.ratio as u32;
        encode_ds4_rope_tail_adjacent_in_place(
            ctx,
            enc,
            &self.normalized,
            start_position,
            rope,
            false,
        )?;
        if self.publication == DeepSeekV4CompressorPublication::IndexerHadamard {
            encode_hadamard_128_in_place(ctx, enc, &self.normalized)?;
            #[cfg(feature = "dsv4-diagnostics")]
            self.fp4_sidecar
                .as_ref()
                .expect("indexer publication requires an FP4 diagnostics sidecar")
                .encode_row(ctx, enc, &self.normalized, published_row)?;
        }
        encode_scatter_offset_f32_to_f16(
            ctx,
            enc,
            &self.normalized,
            &self.published,
            published_row * self.head_dim,
            self.head_dim,
        )?;
        if self.ratio == 4 {
            encode_compressor_roll_ratio4(ctx, enc, &self.kv_state, &self.score_state, self.width)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_projected_chunk(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        projected_kv: &MetalTensor,
        projected_score: &MetalTensor,
        ape: &MetalTensor,
        norm_weight: &MetalTensor,
        pooled_scratch: &MetalTensor,
        normalized_scratch: &MetalTensor,
        start_position: u32,
        row_count: usize,
        rope: DeepSeekV4RopeParameters,
        rms_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_compressor_frontier_projected_chunk")?;
        if row_count == 0 || row_count > DEEPSEEK_V4_PREFILL_MAX_TOKENS {
            return invalid(format!(
                "compressor chunk requires 1..={DEEPSEEK_V4_PREFILL_MAX_TOKENS} rows, got {row_count}"
            ));
        }
        validate_f32(
            projected_kv,
            &[self.width as u64, row_count as u64],
            false,
            "projected compressor KV chunk",
        )?;
        validate_f32(
            projected_score,
            &[self.width as u64, row_count as u64],
            false,
            "projected compressor score chunk",
        )?;
        validate_f32(
            ape,
            &[self.width as u64, self.ratio as u64],
            false,
            "compressor chunk APE",
        )?;
        validate_f32(
            norm_weight,
            &[self.head_dim as u64],
            false,
            "compressor chunk norm weight",
        )?;
        validate_f16(
            &self.published,
            &[self.head_dim as u64, self.capacity_rows as u64],
            true,
            "published compressor rows",
        )?;
        validate_ds4_rope(rope, self.head_dim, rope.rotary_dim)?;
        validate_eps(rms_eps, "compressor RMSNorm epsilon")?;

        let end_position = start_position
            .checked_add(u32::try_from(row_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("compressor chunk rows exceed u32".into())
            })?)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("compressor chunk position overflow".into())
            })?;
        let first_published_row = start_position as usize / self.ratio;
        let end_published_row = end_position as usize / self.ratio;
        let published_rows = end_published_row - first_published_row;
        if end_published_row > self.capacity_rows {
            return invalid(format!(
                "compressor chunk publication end {end_published_row} exceeds the allocated {}-row history",
                self.capacity_rows
            ));
        }
        for (scratch, name) in [
            (pooled_scratch, "pooled compressor chunk scratch"),
            (normalized_scratch, "normalized compressor chunk scratch"),
        ] {
            if scratch.dtype != GgmlType::F32
                || !scratch.is_writable()
                || scratch.n_elements() < self.head_dim as u64 * published_rows as u64
            {
                return invalid(format!(
                    "{name} cannot hold {published_rows} rows of {} values",
                    self.head_dim
                ));
            }
        }
        let pooled_capacity = pooled_scratch.n_elements() as usize / self.head_dim;
        let normalized_capacity = normalized_scratch.n_elements() as usize / self.head_dim;
        let pooled_backing =
            pooled_scratch.view_subrange(0, vec![self.head_dim as u64, pooled_capacity as u64]);
        let normalized_backing = normalized_scratch
            .view_subrange(0, vec![self.head_dim as u64, normalized_capacity as u64]);

        encode_compressor_frontier_chunk(
            ctx,
            enc,
            projected_kv,
            projected_score,
            ape,
            &self.kv_state,
            &self.score_state,
            &pooled_backing,
            self.ratio,
            self.head_dim,
            self.width,
            row_count,
            start_position,
            published_rows,
        )?;
        if published_rows == 0 {
            return Ok(());
        }

        let pooled =
            pooled_backing.view_subrange(0, vec![self.head_dim as u64, published_rows as u64]);
        let normalized =
            normalized_backing.view_subrange(0, vec![self.head_dim as u64, published_rows as u64]);
        validate_f32(
            &pooled,
            &[self.head_dim as u64, published_rows as u64],
            true,
            "pooled compressor chunk rows",
        )?;
        validate_f32(
            &normalized,
            &[self.head_dim as u64, published_rows as u64],
            true,
            "normalized compressor chunk rows",
        )?;
        encode_rms_norm_mul_rows_f32(
            ctx,
            enc,
            &pooled,
            norm_weight,
            &normalized,
            published_rows,
            self.head_dim,
            rms_eps,
        )?;
        let rope_start = u32::try_from(checked_mul(
            first_published_row,
            self.ratio,
            "compressor chunk RoPE start",
        )?)
        .map_err(|_| {
            DeepSeekV4MetalError::Invalid("compressor chunk RoPE start exceeds u32".into())
        })?;
        encode_ds4_rope_tail_adjacent_batch_in_place(
            ctx,
            enc,
            &normalized,
            rope_start,
            published_rows,
            self.ratio as u32,
            rope,
            false,
        )?;
        if self.publication == DeepSeekV4CompressorPublication::IndexerHadamard {
            encode_hadamard_128_rows_in_place(ctx, enc, &normalized, published_rows)?;
            #[cfg(feature = "dsv4-diagnostics")]
            self.fp4_sidecar
                .as_ref()
                .expect("indexer publication requires an FP4 diagnostics sidecar")
                .encode_rows(ctx, enc, &normalized, first_published_row, published_rows)?;
        }
        encode_scatter_offset_f32_to_f16(
            ctx,
            enc,
            &normalized,
            &self.published,
            checked_mul(
                first_published_row,
                self.head_dim,
                "compressor chunk publication offset",
            )?,
            checked_mul(
                published_rows,
                self.head_dim,
                "compressor chunk publication elements",
            )?,
        )?;
        Ok(())
    }

    pub(super) fn published_count(&self, position: u32) -> usize {
        (position as usize + 1) / self.ratio
    }
}

pub(super) fn compressor_frontier_geometry(
    ratio: usize,
    head_dim: usize,
) -> Result<(usize, usize, usize), DeepSeekV4MetalError> {
    if !matches!(ratio, 4 | 128) || head_dim == 0 {
        return invalid(format!(
            "compressor frontier requires ratio 4 or 128 and a nonzero head dimension, got ratio={ratio} head_dim={head_dim}"
        ));
    }
    let coefficient = if ratio == 4 { 2 } else { 1 };
    let width = checked_mul(coefficient, head_dim, "compressor frontier width")?;
    let rows = checked_mul(coefficient, ratio, "compressor frontier rows")?;
    let state_elements = checked_mul(width, rows, "compressor frontier elements")?;
    Ok((width, rows, state_elements))
}

pub(super) enum DeepSeekV4LayerCompressorFrontiers {
    SlidingWindow,
    CompressedSparse {
        attention: Box<DeepSeekV4CompressorFrontier>,
        indexer: Box<DeepSeekV4CompressorFrontier>,
    },
    HeavilyCompressed {
        attention: Box<DeepSeekV4CompressorFrontier>,
    },
}

pub(super) struct DeepSeekV4CompressorFrontiers {
    pub(super) hidden_size: usize,
    pub(super) layers: Vec<DeepSeekV4LayerCompressorFrontiers>,
}

impl DeepSeekV4CompressorFrontiers {
    pub(super) fn new(
        ctx: &MetalContext,
        config: &DeepSeekV4Config,
        capacity: DeepSeekV4SessionCapacity,
    ) -> Result<Self, DeepSeekV4MetalError> {
        let hidden_size = config.hidden_size as usize;
        let attention_dim = config.key_length as usize;
        let indexer_dim = config.indexer_key_length as usize;
        let mut layers = Vec::with_capacity(config.attention_kinds.len());
        for kind in config.attention_kinds.iter().copied() {
            layers.push(match kind {
                AttentionKind::SlidingWindow => DeepSeekV4LayerCompressorFrontiers::SlidingWindow,
                AttentionKind::CompressedSparse => {
                    DeepSeekV4LayerCompressorFrontiers::CompressedSparse {
                        attention: Box::new(DeepSeekV4CompressorFrontier::new(
                            ctx,
                            4,
                            attention_dim,
                            DeepSeekV4CompressorPublication::Attention,
                            capacity.csa_physical_rows(),
                        )?),
                        indexer: Box::new(DeepSeekV4CompressorFrontier::new(
                            ctx,
                            4,
                            indexer_dim,
                            DeepSeekV4CompressorPublication::IndexerHadamard,
                            capacity.csa_physical_rows(),
                        )?),
                    }
                }
                AttentionKind::HeavilyCompressed => {
                    DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed {
                        attention: Box::new(DeepSeekV4CompressorFrontier::new(
                            ctx,
                            128,
                            attention_dim,
                            DeepSeekV4CompressorPublication::Attention,
                            capacity.hca_physical_rows(),
                        )?),
                    }
                }
            });
        }
        Ok(Self {
            hidden_size,
            layers,
        })
    }

    #[cfg(feature = "dsv4-diagnostics")]
    pub(super) fn enable_fp4_shadow_lineage(&mut self) -> Result<(), DeepSeekV4MetalError> {
        let mut enabled = 0usize;
        for layer in &mut self.layers {
            if let DeepSeekV4LayerCompressorFrontiers::CompressedSparse { indexer, .. } = layer {
                indexer
                    .fp4_sidecar
                    .as_mut()
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(
                            "CSA indexer frontier is missing its FP4 diagnostics sidecar".into(),
                        )
                    })?
                    .enable();
                enabled += 1;
            }
        }
        if enabled != diagnostics::CSA_LAYER_COUNT {
            return invalid(format!(
                "enabled {enabled} FP4 indexer sidecars, expected {}",
                diagnostics::CSA_LAYER_COUNT
            ));
        }
        Ok(())
    }

    #[cfg(feature = "dsv4-diagnostics")]
    pub(super) fn invalidate_fp4_shadow_lineage(&self) -> Result<(), DeepSeekV4MetalError> {
        for layer in &self.layers {
            if let DeepSeekV4LayerCompressorFrontiers::CompressedSparse { indexer, .. } = layer {
                indexer
                    .fp4_sidecar
                    .as_ref()
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(
                            "CSA indexer frontier is missing its FP4 diagnostics sidecar".into(),
                        )
                    })?
                    .invalidate()?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "dsv4-diagnostics")]
    pub(super) fn disable_fp4_shadow_lineage(&mut self) -> Result<(), DeepSeekV4MetalError> {
        for layer in &mut self.layers {
            if let DeepSeekV4LayerCompressorFrontiers::CompressedSparse { indexer, .. } = layer {
                indexer
                    .fp4_sidecar
                    .as_mut()
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(
                            "CSA indexer frontier is missing its FP4 diagnostics sidecar".into(),
                        )
                    })?
                    .disable();
            }
        }
        Ok(())
    }

    pub(super) fn encode_layer(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residency: &DeepSeekV4MetalResidency,
        layer: usize,
        position: u32,
        normalized_input: &MetalTensor,
        rope: DeepSeekV4RopeParameters,
        rms_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let frontiers = self.layers.get(layer).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!("compressor layer {layer} is out of range"))
        })?;
        let tensor = |suffix: &str| residency.require_tensor(&format!("blk.{layer}.{suffix}"));
        match frontiers {
            DeepSeekV4LayerCompressorFrontiers::SlidingWindow => Ok(()),
            DeepSeekV4LayerCompressorFrontiers::CompressedSparse { attention, indexer } => {
                attention.encode(
                    ctx,
                    enc,
                    normalized_input,
                    tensor("attn_compressor_kv.weight")?,
                    tensor("attn_compressor_gate.weight")?,
                    tensor("attn_compressor_ape.weight")?,
                    tensor("attn_compressor_norm.weight")?,
                    position,
                    self.hidden_size,
                    rope,
                    rms_eps,
                )?;
                indexer.encode(
                    ctx,
                    enc,
                    normalized_input,
                    tensor("indexer_compressor_kv.weight")?,
                    tensor("indexer_compressor_gate.weight")?,
                    tensor("indexer_compressor_ape.weight")?,
                    tensor("indexer_compressor_norm.weight")?,
                    position,
                    self.hidden_size,
                    rope,
                    rms_eps,
                )
            }
            DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed { attention } => attention
                .encode(
                    ctx,
                    enc,
                    normalized_input,
                    tensor("attn_compressor_kv.weight")?,
                    tensor("attn_compressor_gate.weight")?,
                    tensor("attn_compressor_ape.weight")?,
                    tensor("attn_compressor_norm.weight")?,
                    position,
                    self.hidden_size,
                    rope,
                    rms_eps,
                ),
        }
    }

    pub(super) fn attention_rows(
        &self,
        layer: usize,
        position: u32,
    ) -> Result<Option<DeepSeekV4PublishedRows<'_>>, DeepSeekV4MetalError> {
        let frontier = match self.layers.get(layer).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!("compressor layer {layer} is out of range"))
        })? {
            DeepSeekV4LayerCompressorFrontiers::SlidingWindow => return Ok(None),
            DeepSeekV4LayerCompressorFrontiers::CompressedSparse { attention, .. }
            | DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed { attention } => attention,
        };
        let count = frontier.published_count(position);
        if count == 0 {
            return Ok(None);
        }
        if count > frontier.capacity_rows {
            return invalid(format!(
                "visible compressed rows {count} exceed the allocated {}-row history",
                frontier.capacity_rows
            ));
        }
        Ok(Some(DeepSeekV4PublishedRows {
            cache: &frontier.published,
            count,
            capacity_rows: frontier.capacity_rows,
        }))
    }

    pub(super) fn csa_rows(
        &self,
        layer: usize,
        position: u32,
    ) -> Result<Option<DeepSeekV4CsaRows<'_>>, DeepSeekV4MetalError> {
        let Some(DeepSeekV4LayerCompressorFrontiers::CompressedSparse { attention, indexer }) =
            self.layers.get(layer)
        else {
            return Ok(None);
        };
        let attention_count = attention.published_count(position);
        let indexer_count = indexer.published_count(position);
        if attention_count != indexer_count || attention.capacity_rows != indexer.capacity_rows {
            return invalid(format!(
                "CSA layer {layer} histories are misaligned: attention={attention_count}/{} indexer={indexer_count}/{}",
                attention.capacity_rows, indexer.capacity_rows
            ));
        }
        if attention_count == 0 {
            return Ok(None);
        }
        Ok(Some(DeepSeekV4CsaRows {
            attention_cache: &attention.published,
            indexer_cache: &indexer.published,
            #[cfg(feature = "dsv4-diagnostics")]
            indexer_fp4_sidecar: indexer.fp4_sidecar.as_ref(),
            count: attention_count,
            capacity_rows: attention.capacity_rows,
        }))
    }
}

#[derive(Clone, Copy)]
pub(super) struct DeepSeekV4CsaRows<'a> {
    pub(super) attention_cache: &'a MetalTensor,
    pub(super) indexer_cache: &'a MetalTensor,
    #[cfg(feature = "dsv4-diagnostics")]
    pub(super) indexer_fp4_sidecar: Option<&'a DeepSeekV4IndexerFp4Sidecar>,
    pub(super) count: usize,
    pub(super) capacity_rows: usize,
}

pub(super) const DEEPSEEK_V4_SELECTION_RECORD_I32_WIDTH: usize = 3;

#[cfg(feature = "dsv4-diagnostics")]
pub(super) const DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH: usize = 6;

#[derive(Clone, Copy)]
pub(super) struct DeepSeekV4CsaSelectionView<'a> {
    pub(super) cache_order_ids: &'a MetalTensor,
    pub(super) selected_count: &'a MetalTensor,
    pub(super) visible_count: &'a MetalTensor,
}

impl DeepSeekV4CsaSelectionView<'_> {
    pub(super) fn validate(&self) -> Result<(), DeepSeekV4MetalError> {
        validate_i32(
            self.cache_order_ids,
            &[DEEPSEEK_V4_CSA_TOP_K as u64, 1],
            false,
            "CSA selection-view IDs",
        )?;
        validate_i32(self.selected_count, &[1], false, "CSA selection-view count")?;
        validate_i32(
            self.visible_count,
            &[1],
            false,
            "CSA selection-view visibility",
        )
    }
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone)]
pub(super) struct DeepSeekV4LayerFp4SelectionRecord {
    pub(super) cache_order_ids: MetalTensor,
    pub(super) eligible_visible: MetalTensor,
    pub(super) eligibility_record: MetalTensor,
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4LayerFp4SelectionRecord {
    pub(super) fn output<'a>(
        &'a self,
        selection_record: &'a DeepSeekV4SelectionRecord,
    ) -> DeepSeekV4Fp4SelectionOutput<'a> {
        DeepSeekV4Fp4SelectionOutput {
            eligible_visible: &self.eligible_visible,
            eligibility_record: &self.eligibility_record,
            cache_order_ids: &self.cache_order_ids,
            selected_count: &selection_record.selected_count,
            status: &selection_record.status,
        }
    }

    pub(super) fn selection_view<'a>(
        &'a self,
        selection_record: &'a DeepSeekV4SelectionRecord,
    ) -> DeepSeekV4CsaSelectionView<'a> {
        DeepSeekV4CsaSelectionView {
            cache_order_ids: &self.cache_order_ids,
            selected_count: &selection_record.selected_count,
            visible_count: &self.eligible_visible,
        }
    }
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy)]
pub(super) struct DeepSeekV4Fp4SelectionOutput<'a> {
    pub(super) eligible_visible: &'a MetalTensor,
    pub(super) eligibility_record: &'a MetalTensor,
    pub(super) cache_order_ids: &'a MetalTensor,
    pub(super) selected_count: &'a MetalTensor,
    pub(super) status: &'a MetalTensor,
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4Fp4SelectionOutput<'_> {
    pub(super) fn validate(&self) -> Result<(), DeepSeekV4MetalError> {
        validate_i32(
            self.eligible_visible,
            &[1],
            true,
            "FP4 selection eligible visibility",
        )?;
        validate_i32(
            self.eligibility_record,
            &[3],
            true,
            "FP4 selection eligibility record",
        )?;
        validate_i32(
            self.cache_order_ids,
            &[DEEPSEEK_V4_CSA_TOP_K as u64, 1],
            true,
            "FP4 selection cache-order IDs",
        )?;
        validate_i32(self.selected_count, &[1], true, "FP4 selection count")?;
        validate_i32(self.status, &[1], true, "FP4 selection status")
    }
}

#[cfg(feature = "dsv4-diagnostics")]
pub(super) struct DeepSeekV4LayerFp4SelectionRecords {
    pub(super) cache_order_ids: MetalTensor,
    pub(super) integers: MetalTensor,
}

#[cfg(feature = "dsv4-diagnostics")]
pub(super) struct DeepSeekV4CompletedLayerFp4SelectionRecords {
    pub(super) cache_order_ids: Vec<i32>,
    pub(super) integers: Vec<i32>,
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4LayerFp4SelectionRecords {
    pub(super) fn new(ctx: &MetalContext) -> Result<Self, DeepSeekV4MetalError> {
        Ok(Self {
            cache_order_ids: MetalTensor::zeros_i32(
                ctx,
                vec![DEEPSEEK_V4_CSA_TOP_K as u64, DEEPSEEK_V4_LAYER_COUNT as u64],
            )?,
            integers: MetalTensor::zeros_i32(
                ctx,
                vec![
                    DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH as u64,
                    DEEPSEEK_V4_LAYER_COUNT as u64,
                ],
            )?,
        })
    }

    pub(super) fn layer(
        &self,
        layer: usize,
    ) -> Result<DeepSeekV4LayerFp4SelectionRecord, DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid(format!(
                "collapsed FP4 selection-record layer {layer} is out of range"
            ));
        }
        validate_i32(
            &self.cache_order_ids,
            &[DEEPSEEK_V4_CSA_TOP_K as u64, DEEPSEEK_V4_LAYER_COUNT as u64],
            true,
            "collapsed FP4 layer-selection IDs",
        )?;
        validate_i32(
            &self.integers,
            &[
                DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH as u64,
                DEEPSEEK_V4_LAYER_COUNT as u64,
            ],
            true,
            "collapsed FP4 layer-selection records",
        )?;
        let ids_base = checked_mul(
            layer,
            DEEPSEEK_V4_CSA_TOP_K,
            "collapsed FP4 selection-ID layer offset",
        )? as u64;
        let record_base = checked_mul(
            layer,
            DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH,
            "collapsed FP4 completion-record layer offset",
        )? as u64;
        Ok(DeepSeekV4LayerFp4SelectionRecord {
            cache_order_ids: self
                .cache_order_ids
                .view_subrange(ids_base, vec![DEEPSEEK_V4_CSA_TOP_K as u64, 1]),
            eligible_visible: self.integers.view_subrange(record_base, vec![1]),
            eligibility_record: self.integers.view_subrange(record_base + 1, vec![3]),
        })
    }

    pub(super) fn reset_for_token(
        &self,
        execution: DeepSeekV4Fp4ShadowExecution,
        source: DeepSeekV4Fp4SelectionSource,
    ) -> Result<(), DeepSeekV4MetalError> {
        let mut integers =
            vec![-1; DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH * DEEPSEEK_V4_LAYER_COUNT];
        for record in integers.chunks_exact_mut(DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH) {
            record[4] = i32::from(source.domain_code());
            record[5] = i32::from(execution.domain_code());
        }
        host_write_i32(
            &self.cache_order_ids,
            &vec![-1; DEEPSEEK_V4_CSA_TOP_K * DEEPSEEK_V4_LAYER_COUNT],
            "reset collapsed FP4 layer-selection IDs",
        )?;
        host_write_i32(
            &self.integers,
            &integers,
            "reset collapsed FP4 layer-selection records",
        )
    }

    pub(super) fn read_completed(
        &self,
    ) -> Result<DeepSeekV4CompletedLayerFp4SelectionRecords, DeepSeekV4MetalError> {
        validate_i32(
            &self.cache_order_ids,
            &[DEEPSEEK_V4_CSA_TOP_K as u64, DEEPSEEK_V4_LAYER_COUNT as u64],
            false,
            "completed collapsed FP4 layer-selection IDs",
        )?;
        validate_i32(
            &self.integers,
            &[
                DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH as u64,
                DEEPSEEK_V4_LAYER_COUNT as u64,
            ],
            false,
            "completed collapsed FP4 layer-selection records",
        )?;
        Ok(DeepSeekV4CompletedLayerFp4SelectionRecords {
            cache_order_ids: host_read_i32(
                &self.cache_order_ids,
                "completed collapsed FP4 layer-selection IDs",
            )?,
            integers: host_read_i32(
                &self.integers,
                "completed collapsed FP4 layer-selection records",
            )?,
        })
    }
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4CompletedLayerFp4SelectionRecords {
    pub(super) fn record_and_ids(
        &self,
        layer: usize,
    ) -> Result<(&[i32], &[i32]), DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid(format!(
                "completed collapsed FP4 selection layer {layer} is out of range"
            ));
        }
        let record_base = checked_mul(
            layer,
            DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH,
            "completed collapsed FP4 record layer offset",
        )?;
        let record_end = record_base + DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH;
        let record = self.integers.get(record_base..record_end).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "completed collapsed FP4 record layer {layer} is truncated"
            ))
        })?;
        let ids_base = checked_mul(
            layer,
            DEEPSEEK_V4_CSA_TOP_K,
            "completed collapsed FP4 ID layer offset",
        )?;
        let ids_end = ids_base + DEEPSEEK_V4_CSA_TOP_K;
        let ids = self.cache_order_ids.get(ids_base..ids_end).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "completed collapsed FP4 IDs for layer {layer} are truncated"
            ))
        })?;
        Ok((record, ids))
    }

    pub(super) fn validate_layer(
        &self,
        layer: usize,
        expected_visible_count: usize,
        expected_execution: DeepSeekV4Fp4ShadowExecution,
        expected_source: DeepSeekV4Fp4SelectionSource,
    ) -> Result<&[i32], DeepSeekV4MetalError> {
        let (record, ids) = self.record_and_ids(layer)?;
        let expected_record = [
            expected_visible_count as i32,
            0,
            -1,
            0,
            i32::from(expected_source.domain_code()),
            i32::from(expected_execution.domain_code()),
        ];
        if record != expected_record {
            return invalid(format!(
                "layer {layer} collapsed FP4 completion record {record:?} differs from {expected_record:?}"
            ));
        }
        if ids.windows(2).any(|pair| pair[0] >= pair[1])
            || ids
                .iter()
                .any(|&id| id < 0 || id >= expected_visible_count as i32)
        {
            return invalid(format!(
                "layer {layer} collapsed FP4 IDs are not sorted, unique, and in range"
            ));
        }
        Ok(ids)
    }

    pub(super) fn validate_inactive_layer(
        &self,
        layer: usize,
        expected_execution: DeepSeekV4Fp4ShadowExecution,
        expected_source: DeepSeekV4Fp4SelectionSource,
    ) -> Result<(), DeepSeekV4MetalError> {
        let (record, ids) = self.record_and_ids(layer)?;
        let expected_record = [
            -1,
            -1,
            -1,
            -1,
            i32::from(expected_source.domain_code()),
            i32::from(expected_execution.domain_code()),
        ];
        if record != expected_record || ids.iter().any(|&id| id != -1) {
            return invalid(format!(
                "inactive layer {layer} collapsed FP4 slice was modified: record={record:?}"
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "dsv4-diagnostics")]
pub(super) struct DeepSeekV4Fp4ShadowScratch {
    pub(super) capacity_rows: usize,
    pub(super) query_values: MetalTensor,
    pub(super) query_scales: MetalTensor,
    pub(super) query_status: MetalTensor,
    pub(super) query_units: MetalTensor,
    pub(super) eligible_visible: MetalTensor,
    pub(super) eligibility_record: MetalTensor,
    pub(super) scores: MetalTensor,
    pub(super) selected_mask: MetalTensor,
    pub(super) cache_order_ids: MetalTensor,
    pub(super) selected_count: MetalTensor,
    pub(super) status: MetalTensor,
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4Fp4ShadowScratch {
    pub(super) fn new(
        ctx: &MetalContext,
        capacity_rows: usize,
    ) -> Result<Self, DeepSeekV4MetalError> {
        if capacity_rows < DEEPSEEK_V4_CSA_TOP_K
            || !capacity_rows.is_multiple_of(DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS)
        {
            return invalid(format!(
                "FP4 shadow scratch capacity {capacity_rows} is not an aligned top-k superset"
            ));
        }
        Ok(Self {
            capacity_rows,
            query_values: MetalTensor::zeros_dtype(
                ctx,
                vec![
                    crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES as u64,
                    64,
                    1,
                ],
                GgmlType::I8,
            )?,
            query_scales: MetalTensor::zeros_dtype(
                ctx,
                vec![
                    crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES as u64,
                    64,
                    1,
                ],
                GgmlType::I8,
            )?,
            query_status: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&[DEEPSEEK_V4_FP4_STATUS_UNAVAILABLE; 64]),
                vec![64],
                GgmlType::I32,
            )?,
            query_units: MetalTensor::zeros_f16(ctx, vec![128, 64, 1])?,
            eligible_visible: MetalTensor::zeros_i32(ctx, vec![1])?,
            eligibility_record: MetalTensor::zeros_i32(ctx, vec![3])?,
            scores: MetalTensor::zeros_f32(ctx, vec![capacity_rows as u64, 1])?,
            selected_mask: MetalTensor::zeros_i32(ctx, vec![capacity_rows as u64, 1])?,
            cache_order_ids: MetalTensor::zeros_i32(ctx, vec![DEEPSEEK_V4_CSA_TOP_K as u64, 1])?,
            selected_count: MetalTensor::zeros_i32(ctx, vec![1])?,
            status: MetalTensor::zeros_i32(ctx, vec![1])?,
        })
    }

    pub(super) fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        index_queries: &MetalTensor,
        head_weights: &MetalTensor,
        rows: DeepSeekV4CsaRows<'_>,
        requested_visible: &MetalTensor,
    ) -> Result<(), DeepSeekV4MetalError> {
        let output = DeepSeekV4Fp4SelectionOutput {
            eligible_visible: &self.eligible_visible,
            eligibility_record: &self.eligibility_record,
            cache_order_ids: &self.cache_order_ids,
            selected_count: &self.selected_count,
            status: &self.status,
        };
        self.encode_into(
            ctx,
            enc,
            index_queries,
            head_weights,
            rows,
            requested_visible,
            output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_into(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        index_queries: &MetalTensor,
        head_weights: &MetalTensor,
        rows: DeepSeekV4CsaRows<'_>,
        requested_visible: &MetalTensor,
        output: DeepSeekV4Fp4SelectionOutput<'_>,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_fp4_shadow")?;
        if rows.count <= DEEPSEEK_V4_CSA_TOP_K
            || rows.count > rows.capacity_rows
            || rows.capacity_rows != self.capacity_rows
        {
            return invalid(format!(
                "FP4 shadow requires 513..={} rows, got {}/{}",
                self.capacity_rows, rows.count, rows.capacity_rows
            ));
        }
        let sidecar = rows.indexer_fp4_sidecar.ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("FP4 shadow query has no indexer sidecar".into())
        })?;
        if !sidecar.is_enabled() {
            return invalid("FP4 shadow lineage is not enabled");
        }
        validate_f32(
            index_queries,
            &[128, 64, 1],
            false,
            "FP4 shadow index queries",
        )?;
        validate_f32(head_weights, &[64, 1], false, "FP4 shadow head weights")?;
        validate_i32(
            requested_visible,
            &[1],
            false,
            "FP4 shadow requested visibility",
        )?;
        output.validate()?;
        encode_pack_indexer_fp4_rows_shadow(
            ctx,
            enc,
            index_queries,
            &self.query_values,
            &self.query_scales,
            &self.query_status,
            64,
        )?;
        encode_unpack_indexer_fp4_units_shadow(
            ctx,
            enc,
            &self.query_values,
            &self.query_status,
            &self.query_units,
            64,
        )?;
        encode_indexer_fp4_shadow_preflight(
            ctx,
            enc,
            &self.query_status,
            &sidecar.status,
            requested_visible,
            output.eligible_visible,
            output.eligibility_record,
            rows.capacity_rows,
            rows.count,
        )?;
        encode_lightning_indexer_scores_fp4_matrix_shadow(
            ctx,
            enc,
            &self.query_units,
            &self.query_scales,
            head_weights,
            &sidecar.values,
            &sidecar.scales,
            output.eligible_visible,
            &self.scores,
            rows.capacity_rows,
            1,
        )?;
        encode_select_top_k_f32_with_policy(
            ctx,
            enc,
            &self.scores,
            output.eligible_visible,
            &self.selected_mask,
            None,
            output.cache_order_ids,
            output.selected_count,
            output.status,
            rows.capacity_rows,
            rows.count,
            DEEPSEEK_V4_CSA_TOP_K,
            1,
            DeepSeekV4SelectorDispatchPolicy::Production,
            true,
        )
    }

    pub(super) fn selection_view(&self) -> DeepSeekV4CsaSelectionView<'_> {
        DeepSeekV4CsaSelectionView {
            cache_order_ids: &self.cache_order_ids,
            selected_count: &self.selected_count,
            visible_count: &self.eligible_visible,
        }
    }

    pub(super) fn validate_completed(&self) -> Result<(), DeepSeekV4MetalError> {
        let eligibility = host_read_i32(
            &self.eligibility_record,
            "completed FP4 shadow eligibility record",
        )?;
        let selected_count =
            host_read_i32(&self.selected_count, "completed FP4 shadow selected count")?;
        let status = host_read_i32(&self.status, "completed FP4 shadow selection status")?;
        if eligibility.as_slice() != [0, -1, 0]
            || selected_count.as_slice() != [DEEPSEEK_V4_CSA_TOP_K as i32]
            || status.as_slice() != [0]
        {
            return invalid(format!(
                "FP4 shadow selection failed with eligibility={eligibility:?} count={selected_count:?} status={status:?}"
            ));
        }
        Ok(())
    }

    pub(super) fn record_counterfactual_selection(
        &self,
        trace: &mut diagnostics::DeepSeekV4Fp4CounterfactualTrace,
        execution: DeepSeekV4Fp4ShadowExecution,
        source: DeepSeekV4Fp4SelectionSource,
        position: u32,
        layer: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        let visible = host_read_i32(
            &self.eligible_visible,
            "consumed FP4 counterfactual visibility",
        )?;
        let ids = host_read_i32(&self.cache_order_ids, "consumed FP4 counterfactual IDs")?;
        trace.record(execution, source, position, layer, visible[0], &ids)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn capture_layer(
        &self,
        layer: usize,
        position: u32,
        rows: DeepSeekV4CsaRows<'_>,
        authoritative_scores: &MetalTensor,
        authoritative_mask: &MetalTensor,
        authoritative_ids: &MetalTensor,
        authoritative_count: &MetalTensor,
        authoritative_status: &MetalTensor,
    ) -> Result<DeepSeekV4Fp4ShadowLayer, DeepSeekV4MetalError> {
        let sidecar = rows.indexer_fp4_sidecar.ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("FP4 shadow report has no indexer sidecar".into())
        })?;
        let visible_key_status = sidecar.status.view_subrange(0, vec![rows.count as u64]);
        Ok(diagnostics::build_fp4_shadow_layer(
            diagnostics::DeepSeekV4Fp4ShadowLayerInputs {
                layer,
                position,
                visible_count: rows.count,
                capacity_rows: rows.capacity_rows,
                query_statuses: host_read_i32(&self.query_status, "FP4 shadow query statuses")?,
                visible_key_statuses: host_read_i32(
                    &visible_key_status,
                    "FP4 shadow visible key statuses",
                )?,
                eligibility_record: host_read_i32(
                    &self.eligibility_record,
                    "FP4 shadow eligibility record",
                )?,
                authoritative_scores: host_read_f32(
                    authoritative_scores,
                    "FP4 authoritative scores",
                )?,
                authoritative_mask: host_read_i32(authoritative_mask, "FP4 authoritative mask")?,
                authoritative_ids: host_read_i32(authoritative_ids, "FP4 authoritative IDs")?,
                authoritative_count: host_read_i32(
                    authoritative_count,
                    "FP4 authoritative selected count",
                )?,
                authoritative_status: host_read_i32(
                    authoritative_status,
                    "FP4 authoritative selection status",
                )?,
                shadow_scores: host_read_f32(&self.scores, "FP4 shadow scores")?,
                shadow_mask: host_read_i32(&self.selected_mask, "FP4 shadow mask")?,
                shadow_ids: host_read_i32(&self.cache_order_ids, "FP4 shadow IDs")?,
                shadow_count: host_read_i32(&self.selected_count, "FP4 shadow selected count")?,
                shadow_status: host_read_i32(&self.status, "FP4 shadow selection status")?,
            },
        )?)
    }
}

pub(super) const DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS: usize = 20;

pub(super) const DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS: usize = 10;

pub(super) const DEEPSEEK_V4_MULTIGROUP_SELECTOR_DIGITS: usize = 8;

pub(super) const DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS: usize = 5;

pub(super) const DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS: usize = 32;

#[cfg(not(test))]
pub(super) const DEEPSEEK_V4_MULTIGROUP_SELECTOR_QUALIFIED_DEVICE: &str = "Apple M4 Max";

#[doc(hidden)]
pub const DEEPSEEK_V4_MULTIGROUP_SELECTOR_MIN_VISIBLE_ROWS: usize = 196_608;

#[doc(hidden)]
pub const DEEPSEEK_V4_MULTIGROUP_SELECTOR_MAX_CAPACITY_ROWS: usize = 262_144;

pub(super) fn deepseek_v4_multigroup_selector_capacity_supported(capacity_rows: usize) -> bool {
    (DEEPSEEK_V4_MULTIGROUP_SELECTOR_MIN_VISIBLE_ROWS
        ..=DEEPSEEK_V4_MULTIGROUP_SELECTOR_MAX_CAPACITY_ROWS)
        .contains(&capacity_rows)
}

pub(super) fn deepseek_v4_multigroup_selector_eligible(
    capacity_rows: usize,
    visible_rows: usize,
) -> bool {
    deepseek_v4_multigroup_selector_capacity_supported(capacity_rows)
        && visible_rows <= capacity_rows
        && visible_rows >= DEEPSEEK_V4_MULTIGROUP_SELECTOR_MIN_VISIBLE_ROWS
        && visible_rows >= capacity_rows - capacity_rows / 4
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DeepSeekV4SparseSelectorMode {
    Radix4,
    MultigroupProduction,
    MultigroupExperimental,
}

#[derive(Debug, Default)]
pub(super) struct DeepSeekV4MultigroupSelectorInvocationCounters {
    pub(super) multigroup: Cell<u64>,
    pub(super) ineligible_radix4: Cell<u64>,
}

impl DeepSeekV4MultigroupSelectorInvocationCounters {
    pub(super) fn next_multigroup(&self) -> Result<u64, DeepSeekV4MetalError> {
        self.multigroup.get().checked_add(1).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "DeepSeek V4 multi-group selector invocation count overflowed".into(),
            )
        })
    }

    pub(super) fn commit_multigroup(&self, next: u64) {
        self.multigroup.set(next);
    }

    pub(super) fn next_ineligible_radix4(&self) -> Result<u64, DeepSeekV4MetalError> {
        self.ineligible_radix4.get().checked_add(1).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "DeepSeek V4 ineligible radix4 selector invocation count overflowed".into(),
            )
        })
    }

    pub(super) fn commit_ineligible_radix4(&self, next: u64) {
        self.ineligible_radix4.set(next);
    }

    pub(super) fn telemetry(&self, sealed: bool) -> DeepSeekV4MultigroupSelectorTelemetry {
        DeepSeekV4MultigroupSelectorTelemetry {
            sealed,
            multigroup_invocations: self.multigroup.get(),
            ineligible_radix4_invocations: self.ineligible_radix4.get(),
        }
    }
}

#[derive(Debug)]
pub(super) struct DeepSeekV4MultigroupSelectorGeneration {
    pub(super) next: Cell<Option<NonZeroU32>>,
}

impl DeepSeekV4MultigroupSelectorGeneration {
    pub(super) fn new() -> Self {
        Self::from_next(NonZeroU32::MIN)
    }

    pub(super) fn from_next(next: NonZeroU32) -> Self {
        Self {
            next: Cell::new(Some(next)),
        }
    }

    pub(super) fn take(&self) -> Result<u32, DeepSeekV4MetalError> {
        let generation = self.next.get().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "DeepSeek V4 multi-group selector generation space is exhausted".into(),
            )
        })?;
        self.next
            .set(generation.get().checked_add(1).and_then(NonZeroU32::new));
        Ok(generation.get())
    }
}

pub(super) struct DeepSeekV4MultigroupSelectorScratch {
    pub(super) records: MetalTensor,
    pub(super) partition_plan: MetalTensor,
    pub(super) state: MetalTensor,
    pub(super) private_mask: MetalTensor,
    pub(super) private_ids: MetalTensor,
    pub(super) generation: DeepSeekV4MultigroupSelectorGeneration,
}

impl DeepSeekV4MultigroupSelectorScratch {
    pub(super) fn new(
        ctx: &MetalContext,
        capacity_rows: usize,
    ) -> Result<Self, DeepSeekV4MetalError> {
        if !deepseek_v4_multigroup_selector_capacity_supported(capacity_rows) {
            return invalid(format!(
                "multi-group selector scratch does not support capacity {capacity_rows}"
            ));
        }
        Ok(Self {
            records: MetalTensor::zeros_i32(
                ctx,
                vec![
                    DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS as u64,
                    DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS as u64,
                ],
            )?,
            partition_plan: MetalTensor::zeros_i32(
                ctx,
                vec![
                    DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS as u64,
                    DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS as u64,
                ],
            )?,
            state: MetalTensor::zeros_i32(
                ctx,
                vec![DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS as u64],
            )?,
            private_mask: MetalTensor::zeros_dtype(
                ctx,
                vec![capacity_rows as u64, 1],
                GgmlType::I8,
            )?,
            private_ids: MetalTensor::zeros_i32(ctx, vec![DEEPSEEK_V4_CSA_TOP_K as u64, 1])?,
            generation: DeepSeekV4MultigroupSelectorGeneration::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        scores: &MetalTensor,
        visible_counts: &MetalTensor,
        selected_mask: &MetalTensor,
        cache_order_ids: &MetalTensor,
        selected_counts: &MetalTensor,
        status: &MetalTensor,
        capacity_rows: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        encode_select_top_k_multigroup_full_f32(
            ctx,
            enc,
            scores,
            visible_counts,
            &self.records,
            &self.partition_plan,
            &self.state,
            &self.private_mask,
            &self.private_ids,
            selected_mask,
            cache_order_ids,
            selected_counts,
            status,
            capacity_rows,
            DEEPSEEK_V4_CSA_TOP_K,
            self.generation.take()?,
            None,
            false,
        )
    }
}

pub(super) struct DeepSeekV4SparseCsaScratch {
    pub(super) capacity_rows: usize,
    pub(super) index_queries: MetalTensor,
    pub(super) matrix_queries_f16: MetalTensor,
    pub(super) head_weights: MetalTensor,
    pub(super) visible_counts: MetalTensor,
    pub(super) scores: MetalTensor,
    pub(super) selected_mask: MetalTensor,
    pub(super) cache_order_ids: MetalTensor,
    pub(super) selected_counts: MetalTensor,
    pub(super) status: MetalTensor,
    pub(super) selector_mode: DeepSeekV4SparseSelectorMode,
    pub(super) multigroup: Option<DeepSeekV4MultigroupSelectorScratch>,
    pub(super) multigroup_invocations: DeepSeekV4MultigroupSelectorInvocationCounters,
    #[cfg(test)]
    pub(super) score_test_policy: DeepSeekV4IndexerScoreTestPolicy,
    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) selector_test_policy: DeepSeekV4SelectorTestPolicy,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DeepSeekV4IndexerScoreTestPolicy {
    Production,
    ScalarOracle,
    MatrixF16,
}

impl DeepSeekV4SparseCsaScratch {
    pub(super) fn new(
        ctx: &MetalContext,
        capacity_rows: usize,
    ) -> Result<Self, DeepSeekV4MetalError> {
        if capacity_rows < DEEPSEEK_V4_CSA_TOP_K
            || !capacity_rows.is_multiple_of(DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS)
        {
            return invalid(format!(
                "sparse CSA scratch capacity {capacity_rows} is not an aligned top-k superset"
            ));
        }
        let multigroup = deepseek_v4_multigroup_selector_capacity_supported(capacity_rows)
            .then(|| DeepSeekV4MultigroupSelectorScratch::new(ctx, capacity_rows))
            .transpose()?;
        #[cfg(test)]
        let selector_mode = DeepSeekV4SparseSelectorMode::Radix4;
        #[cfg(not(test))]
        let selector_mode = if multigroup.is_some()
            && ctx.device.name().to_string() == DEEPSEEK_V4_MULTIGROUP_SELECTOR_QUALIFIED_DEVICE
            && deepseek_v4_multigroup_selector_enabled()
        {
            DeepSeekV4SparseSelectorMode::MultigroupProduction
        } else {
            DeepSeekV4SparseSelectorMode::Radix4
        };
        Ok(Self {
            capacity_rows,
            index_queries: MetalTensor::zeros_f32(ctx, vec![128, 64, 1])?,
            matrix_queries_f16: MetalTensor::zeros_f16(ctx, vec![128, 64, 1])?,
            head_weights: MetalTensor::zeros_f32(ctx, vec![64, 1])?,
            visible_counts: MetalTensor::zeros_i32(ctx, vec![1])?,
            scores: MetalTensor::zeros_f32(ctx, vec![capacity_rows as u64, 1])?,
            selected_mask: MetalTensor::zeros_i32(ctx, vec![capacity_rows as u64, 1])?,
            cache_order_ids: MetalTensor::zeros_i32(ctx, vec![DEEPSEEK_V4_CSA_TOP_K as u64, 1])?,
            selected_counts: MetalTensor::zeros_i32(ctx, vec![1])?,
            status: MetalTensor::zeros_i32(ctx, vec![1])?,
            selector_mode,
            multigroup,
            multigroup_invocations: DeepSeekV4MultigroupSelectorInvocationCounters::default(),
            #[cfg(test)]
            score_test_policy: DeepSeekV4IndexerScoreTestPolicy::Production,
            #[cfg(all(test, feature = "dsv4-diagnostics"))]
            selector_test_policy: DeepSeekV4SelectorTestPolicy::Production,
        })
    }

    pub(super) fn enable_multigroup_selector_experiment(
        &mut self,
    ) -> Result<(), DeepSeekV4MetalError> {
        if self.selector_mode == DeepSeekV4SparseSelectorMode::MultigroupExperimental {
            return invalid("DeepSeek V4 sparse selector experiment is already sealed");
        }
        if self.multigroup.is_none() {
            return invalid(format!(
                "DeepSeek V4 session CSA capacity {} cannot enter the measured multi-group selector band",
                self.capacity_rows
            ));
        }
        self.selector_mode = DeepSeekV4SparseSelectorMode::MultigroupExperimental;
        Ok(())
    }

    pub(super) fn disable_multigroup_selector(&mut self) -> Result<(), DeepSeekV4MetalError> {
        if self.selector_mode == DeepSeekV4SparseSelectorMode::MultigroupExperimental {
            return invalid("DeepSeek V4 sparse selector experiment is already sealed");
        }
        self.selector_mode = DeepSeekV4SparseSelectorMode::Radix4;
        Ok(())
    }

    pub(super) fn multigroup_selector_telemetry(&self) -> DeepSeekV4MultigroupSelectorTelemetry {
        self.multigroup_invocations
            .telemetry(self.selector_mode == DeepSeekV4SparseSelectorMode::MultigroupExperimental)
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn set_score_test_policy(&mut self, policy: DeepSeekV4IndexerScoreTestPolicy) {
        self.score_test_policy = policy;
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn set_selector_test_policy(&mut self, policy: DeepSeekV4SelectorTestPolicy) {
        self.selector_test_policy = policy;
    }

    pub(super) fn force_scalar_score_kernel(&self) -> bool {
        #[cfg(test)]
        {
            self.score_test_policy == DeepSeekV4IndexerScoreTestPolicy::ScalarOracle
        }
        #[cfg(not(test))]
        {
            false
        }
    }

    pub(super) fn use_f16_matrix_score(&self, ctx: &MetalContext, visible_rows: usize) -> bool {
        #[cfg(test)]
        {
            let _ = (ctx, visible_rows);
            self.score_test_policy == DeepSeekV4IndexerScoreTestPolicy::MatrixF16
        }
        #[cfg(not(test))]
        {
            visible_rows >= DEEPSEEK_V4_F16_MATRIX_SCORER_MIN_VISIBLE_ROWS
                && ctx.device.name().to_string() == DEEPSEEK_V4_F16_MATRIX_SCORER_QUALIFIED_DEVICE
                && deepseek_v4_f16_matrix_scorer_enabled()
        }
    }

    pub(super) fn use_radix4_selector(&self) -> bool {
        #[cfg(all(test, feature = "dsv4-diagnostics"))]
        {
            self.selector_test_policy == DeepSeekV4SelectorTestPolicy::Production
        }
        #[cfg(any(not(test), all(test, not(feature = "dsv4-diagnostics"))))]
        {
            true
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        q_lora: &MetalTensor,
        normalized_input: &MetalTensor,
        indexer_q_weight: &MetalTensor,
        indexer_projection: &MetalTensor,
        rows: DeepSeekV4CsaRows<'_>,
        position: u32,
        rope: DeepSeekV4RopeParameters,
        record: &DeepSeekV4SelectionRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        self.encode_prepare(
            ctx,
            enc,
            q_lora,
            normalized_input,
            indexer_q_weight,
            indexer_projection,
            rows,
            position,
            rope,
            record,
        )?;
        self.encode_f16_score_and_select(ctx, enc, rows, record)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_prepare(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        q_lora: &MetalTensor,
        normalized_input: &MetalTensor,
        indexer_q_weight: &MetalTensor,
        indexer_projection: &MetalTensor,
        rows: DeepSeekV4CsaRows<'_>,
        position: u32,
        rope: DeepSeekV4RopeParameters,
        record: &DeepSeekV4SelectionRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_sparse_csa_indexer")?;
        if rows.count <= DEEPSEEK_V4_CSA_TOP_K
            || rows.count > rows.capacity_rows
            || rows.capacity_rows != self.capacity_rows
        {
            return invalid(format!(
                "sparse CSA requires 513..={} aligned rows, got count={} capacity={}",
                self.capacity_rows, rows.count, rows.capacity_rows
            ));
        }
        validate_f32(q_lora, &[1_024], false, "sparse CSA Q-LoRA input")?;
        validate_f32(
            normalized_input,
            &[DEEPSEEK_V4_HIDDEN_SIZE as u64],
            false,
            "sparse CSA normalized input",
        )?;
        validate_matvec_weight(indexer_q_weight, 1_024, 64 * 128, "indexer Q weight")?;
        validate_matvec_weight(
            indexer_projection,
            DEEPSEEK_V4_HIDDEN_SIZE,
            64,
            "indexer projection weight",
        )?;
        validate_f16(
            rows.indexer_cache,
            &[128, rows.capacity_rows as u64],
            false,
            "sparse CSA indexer cache",
        )?;
        record.validate()?;
        host_write_i32(
            &record.visible_count,
            &[rows.count as i32],
            "sparse CSA visible count",
        )?;
        encode_projection(
            ctx,
            enc,
            indexer_q_weight,
            q_lora,
            &self.index_queries,
            1_024,
            64 * 128,
            "indexer Q",
        )?;
        encode_ds4_rope_tail_adjacent_in_place(
            ctx,
            enc,
            &self.index_queries,
            position,
            rope,
            false,
        )?;
        encode_hadamard_128_rows_in_place(ctx, enc, &self.index_queries, 64)?;
        encode_projection(
            ctx,
            enc,
            indexer_projection,
            normalized_input,
            &self.head_weights,
            DEEPSEEK_V4_HIDDEN_SIZE,
            64,
            "indexer head weights",
        )?;
        encode_scale_f32_in_place(
            ctx,
            enc,
            &self.head_weights,
            1.0 / (64.0f32 * 128.0).sqrt(),
            "indexer head weights",
        )
    }

    pub(super) fn encode_f16_score_and_select(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        rows: DeepSeekV4CsaRows<'_>,
        record: &DeepSeekV4SelectionRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        if self.use_f16_matrix_score(ctx, rows.count) {
            #[cfg(not(test))]
            {
                static REPORTED: std::sync::Once = std::sync::Once::new();
                REPORTED.call_once(|| {
                    eprintln!(
                        "deepseek_v4: forced F16-staged Lightning matrix scorer active for far singleton rows; disable=QWEN_DSV4_LIGHTNING_F16_MATRIX=0"
                    );
                });
            }
            encode_scatter_offset_f32_to_f16(
                ctx,
                enc,
                &self.index_queries,
                &self.matrix_queries_f16,
                0,
                64 * 128,
            )?;
            encode_lightning_indexer_scores_f16_matrix(
                ctx,
                enc,
                &self.matrix_queries_f16,
                &self.head_weights,
                rows.indexer_cache,
                &record.visible_count,
                &self.scores,
                64,
                128,
                rows.capacity_rows,
                rows.count,
                1,
            )?;
        } else {
            encode_lightning_indexer_scores_f16_with_policy(
                ctx,
                enc,
                &self.index_queries,
                &self.head_weights,
                rows.indexer_cache,
                &record.visible_count,
                &self.scores,
                64,
                128,
                rows.capacity_rows,
                1,
                self.force_scalar_score_kernel(),
            )?;
        }
        self.encode_scored_rows(ctx, enc, rows.capacity_rows, rows.count, record)
    }

    pub(super) fn encode_scored_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        capacity_rows: usize,
        visible_rows: usize,
        record: &DeepSeekV4SelectionRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        if self.selector_mode != DeepSeekV4SparseSelectorMode::Radix4
            && deepseek_v4_multigroup_selector_eligible(capacity_rows, visible_rows)
        {
            if self.selector_mode == DeepSeekV4SparseSelectorMode::MultigroupProduction {
                static REPORTED: std::sync::Once = std::sync::Once::new();
                REPORTED.call_once(|| {
                    eprintln!(
                        "deepseek_v4: exact multi-group selector owns the qualified far-context band; rollback=QWEN_DSV4_MULTIGROUP_SELECTOR=0"
                    );
                });
            }
            let next_invocations = self.multigroup_invocations.next_multigroup()?;
            self.multigroup
                .as_ref()
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "eligible multi-group selector has no session-owned scratch".into(),
                    )
                })?
                .encode(
                    ctx,
                    enc,
                    &self.scores,
                    &record.visible_count,
                    &self.selected_mask,
                    &self.cache_order_ids,
                    &record.selected_count,
                    &record.status,
                    capacity_rows,
                )?;
            self.multigroup_invocations
                .commit_multigroup(next_invocations);
            return Ok(());
        }
        let next_ineligible = (self.selector_mode != DeepSeekV4SparseSelectorMode::Radix4)
            .then(|| self.multigroup_invocations.next_ineligible_radix4())
            .transpose()?;
        encode_select_top_k_f32_with_policy(
            ctx,
            enc,
            &self.scores,
            &record.visible_count,
            &self.selected_mask,
            None,
            &self.cache_order_ids,
            &record.selected_count,
            &record.status,
            capacity_rows,
            visible_rows,
            DEEPSEEK_V4_CSA_TOP_K,
            1,
            DeepSeekV4SelectorDispatchPolicy::Production,
            self.use_radix4_selector(),
        )?;
        if let Some(next) = next_ineligible {
            self.multigroup_invocations.commit_ineligible_radix4(next);
        }
        Ok(())
    }

    pub(super) fn default_record(&self) -> DeepSeekV4SelectionRecord {
        DeepSeekV4SelectionRecord {
            visible_count: self.visible_counts.clone(),
            selected_count: self.selected_counts.clone(),
            status: self.status.clone(),
        }
    }

    pub(super) fn selection_view<'a>(
        &'a self,
        record: &'a DeepSeekV4SelectionRecord,
    ) -> DeepSeekV4CsaSelectionView<'a> {
        DeepSeekV4CsaSelectionView {
            cache_order_ids: &self.cache_order_ids,
            selected_count: &record.selected_count,
            visible_count: &record.visible_count,
        }
    }

    pub(super) fn validate_completed(
        &self,
        record: &DeepSeekV4SelectionRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        record.validate()?;
        let status = host_read_i32(&record.status, "sparse CSA selection status")?;
        let count = host_read_i32(&record.selected_count, "sparse CSA selected count")?;
        if status.as_slice() != [0] || count.as_slice() != [DEEPSEEK_V4_CSA_TOP_K as i32] {
            return invalid(format!(
                "sparse CSA selection failed with status={status:?} count={count:?}"
            ));
        }
        Ok(())
    }

    #[cfg(feature = "dsv4-diagnostics")]
    pub(super) fn capture_decision(
        &self,
        visible_count: usize,
        record: &DeepSeekV4SelectionRecord,
    ) -> Result<DeepSeekV4CsaDecision, DeepSeekV4MetalError> {
        record.validate()?;
        let scores = host_read_f32(&self.scores, "diagnostic sparse CSA scores")?;
        let selected_ids = host_read_i32(
            &self.cache_order_ids,
            "diagnostic sparse CSA cache-order IDs",
        )?;
        let selected_count = host_read_i32(
            &record.selected_count,
            "diagnostic sparse CSA selected count",
        )?;
        let status = host_read_i32(&record.status, "diagnostic sparse CSA status")?;
        Ok(diagnostics::build_csa_decision(
            scores,
            visible_count,
            selected_ids,
            selected_count,
            status,
        )?)
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DeepSeekV4HcaTestPolicy {
    Production,
    #[cfg(feature = "dsv4-diagnostics")]
    GroupedOnline,
    LegacyTiled,
}

pub(super) fn encode_position_zero_sink_attention(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    kv: &MetalTensor,
    sinks: &MetalTensor,
    output: &MetalTensor,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        scale: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_position_zero_sink_attention")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: config.head_count as u32,
            head_dim: config.head_dim as u32,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, kv);
    enc.set_tensor(3, sinks);
    enc.set_tensor(4, output);
    let width = checked_mul(
        config.head_count,
        config.head_dim,
        "attention dispatch width",
    )?;
    enc.dispatch(
        MTLSize {
            width: width.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
pub(super) fn encode_local_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    sinks: &MetalTensor,
    output: &MetalTensor,
    position: u32,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    encode_dense_sink_attention_f16(
        ctx, enc, queries, raw_cache, None, sinks, output, position, config,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dense_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    compressed: Option<DeepSeekV4PublishedRows<'_>>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    position: u32,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    let query_width = checked_mul(config.head_count, config.head_dim, "local query width")?;
    validate_f32(
        queries,
        &[config.head_dim as u64, config.head_count as u64],
        false,
        "local attention queries",
    )?;
    validate_f16(
        raw_cache,
        &[config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        false,
        "local attention cache",
    )?;
    validate_f32(
        sinks,
        &[config.head_count as u64],
        false,
        "local attention sinks",
    )?;
    validate_f32(
        output,
        &[config.head_dim as u64, config.head_count as u64],
        true,
        "local attention output",
    )?;
    let (compressed_cache, compressed_count) = if let Some(rows) = compressed {
        validate_f16(
            rows.cache,
            &[config.head_dim as u64, rows.capacity_rows as u64],
            false,
            "dense compressed attention cache",
        )?;
        if rows.count == 0 || rows.count > DEEPSEEK_V4_CSA_TOP_K || rows.count > rows.capacity_rows
        {
            return invalid(format!(
                "dense compressed attention count {} is out of range",
                rows.count
            ));
        }
        (rows.cache, rows.count)
    } else {
        (raw_cache, 0)
    };

    let visible_end = u64::from(position) + 1;
    let raw_count = visible_end.min(DEEPSEEK_V4_LOCAL_WINDOW as u64) as u32;
    let raw_start = visible_end - u64::from(raw_count);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        window: u32,
        raw_count: u32,
        raw_start: u32,
        compressed_count: u32,
        scale: f32,
    }
    let raw_start = u32::try_from(raw_start)
        .map_err(|_| DeepSeekV4MetalError::Invalid("local attention start exceeds u32".into()))?;
    let pso = ctx.pipeline("kernel_deepseek_v4_dense_sink_attention_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: config.head_count as u32,
            head_dim: config.head_dim as u32,
            window: DEEPSEEK_V4_LOCAL_WINDOW as u32,
            raw_count,
            raw_start,
            compressed_count: compressed_count as u32,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, raw_cache);
    enc.set_tensor(3, compressed_cache);
    enc.set_tensor(4, sinks);
    enc.set_tensor(5, output);
    enc.dispatch(
        MTLSize {
            width: query_width.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(super) const DEEPSEEK_V4_SPLITK_HCA_PARTITIONS: usize = 8;

#[cfg(not(test))]
pub(super) const DEEPSEEK_V4_LONG_HCA_QUALIFIED_DEVICE: &str = "Apple M4 Max";

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_cooperative_dense_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    raw_cache_layout: DeepSeekV4RawCacheLayout,
    compressed: Option<DeepSeekV4PublishedRows<'_>>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    kind: AttentionKind,
    start_position: u32,
    token_count: usize,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    encode_dense_sink_attention_f16_with_kernel(
        ctx,
        enc,
        queries,
        raw_cache,
        raw_cache_before_chunk,
        raw_cache_layout,
        compressed,
        sinks,
        output,
        kind,
        start_position,
        token_count,
        config,
        DeepSeekV4DenseAttentionKernel::Cooperative,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_grouped_online_dense_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    raw_cache_layout: DeepSeekV4RawCacheLayout,
    compressed: Option<DeepSeekV4PublishedRows<'_>>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    kind: AttentionKind,
    start_position: u32,
    token_count: usize,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    encode_dense_sink_attention_f16_with_kernel(
        ctx,
        enc,
        queries,
        raw_cache,
        raw_cache_before_chunk,
        raw_cache_layout,
        compressed,
        sinks,
        output,
        kind,
        start_position,
        token_count,
        config,
        DeepSeekV4DenseAttentionKernel::GroupedOnline,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_grouped_splitk_hca_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    raw_cache_layout: DeepSeekV4RawCacheLayout,
    compressed: DeepSeekV4PublishedRows<'_>,
    sinks: &MetalTensor,
    partial_output: &MetalTensor,
    partial_ml: &MetalTensor,
    output: &MetalTensor,
    position: u32,
    partitions: usize,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    const MAX_PARTITIONS: usize = 16;
    require_serial(enc, "deepseek_v4_grouped_splitk_hca_f16")?;
    validate_deepseek_v4_online_hca_request_geometry(config, 128, 1, 0, 1)?;
    if raw_cache_layout != DeepSeekV4RawCacheLayout::Ring {
        return invalid("grouped split-K HCA requires the singleton ring raw cache");
    }
    if !matches!(partitions, 4 | 8 | 16) {
        return invalid("grouped split-K HCA requires 4, 8, or 16 partitions");
    }
    let query_width = checked_mul(
        config.head_count,
        config.head_dim,
        "split-K HCA query width",
    )?;
    validate_f32(
        queries,
        &[query_width as u64, 1],
        false,
        "split-K HCA queries",
    )?;
    validate_raw_attention_caches(
        raw_cache,
        raw_cache_before_chunk,
        raw_cache_layout,
        config.head_dim,
        1,
        "split-K HCA",
    )?;
    validate_f16(
        compressed.cache,
        &[config.head_dim as u64, compressed.capacity_rows as u64],
        false,
        "split-K HCA compressed cache",
    )?;
    validate_f32(
        sinks,
        &[config.head_count as u64],
        false,
        "split-K HCA sinks",
    )?;
    validate_f32(
        partial_output,
        &[
            config.head_dim as u64,
            config.head_count as u64,
            partitions as u64,
        ],
        true,
        "split-K HCA partial output",
    )?;
    validate_f32(
        partial_ml,
        &[2, config.head_count as u64, partitions as u64],
        true,
        "split-K HCA partial max/mass",
    )?;
    validate_f32(output, &[query_width as u64, 1], true, "split-K HCA output")?;
    let expected_rows = (position as usize + 1) / 128;
    if compressed.count != expected_rows || compressed.count > compressed.capacity_rows {
        return invalid(format!(
            "split-K HCA expected {expected_rows} compressed rows, got {}/{}",
            compressed.count, compressed.capacity_rows
        ));
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        compression_ratio: u32,
        start_position: u32,
        window: u32,
        raw_cache_is_chunk: u32,
        partitions: u32,
        scale: f32,
    }
    let args = Args {
        head_count: config.head_count as u32,
        head_dim: config.head_dim as u32,
        compression_ratio: 128,
        start_position: position,
        window: DEEPSEEK_V4_LOCAL_WINDOW as u32,
        raw_cache_is_chunk: raw_cache_layout.is_chunk(),
        partitions: partitions as u32,
        scale: 1.0 / (config.head_dim as f32).sqrt(),
    };

    let main = ctx.pipeline("kernel_deepseek_v4_grouped_splitk_hca_main_f16")?;
    if main.threadExecutionWidth() != 32
        || main.maxTotalThreadsPerThreadgroup() < DEEPSEEK_V4_GROUPED_DENSE_THREADS
        || ctx.device.maxThreadgroupMemoryLength() < DEEPSEEK_V4_GROUPED_DENSE_THREADGROUP_BYTES
    {
        return invalid("grouped split-K HCA main pipeline does not support its launch geometry");
    }
    enc.set_pipeline(&main);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, queries);
    enc.set_tensor(2, raw_cache);
    enc.set_tensor(3, raw_cache_before_chunk);
    enc.set_tensor(4, compressed.cache);
    enc.set_tensor(5, partial_output);
    enc.set_tensor(6, partial_ml);
    enc.set_threadgroup_memory(0, DEEPSEEK_V4_GROUPED_DENSE_THREADGROUP_BYTES);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: config.head_count / DEEPSEEK_V4_GROUPED_DENSE_HEADS,
            depth: partitions,
        },
        MTLSize {
            width: 32,
            height: DEEPSEEK_V4_GROUPED_DENSE_HEADS,
            depth: 1,
        },
    );

    let reduce = ctx.pipeline("kernel_deepseek_v4_grouped_splitk_hca_reduce_f32")?;
    if reduce.threadExecutionWidth() != 32 || reduce.maxTotalThreadsPerThreadgroup() < 32 {
        return invalid("grouped split-K HCA reducer does not support one SIMDgroup");
    }
    enc.set_pipeline(&reduce);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, partial_output);
    enc.set_tensor(2, partial_ml);
    enc.set_tensor(3, sinks);
    enc.set_tensor(4, output);
    enc.set_threadgroup_memory(0, MAX_PARTITIONS * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: config.head_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_dense_sink_attention_f16_with_kernel(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    raw_cache_layout: DeepSeekV4RawCacheLayout,
    compressed: Option<DeepSeekV4PublishedRows<'_>>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    kind: AttentionKind,
    start_position: u32,
    token_count: usize,
    config: DeepSeekV4PositionZeroAttentionConfig,
    kernel: DeepSeekV4DenseAttentionKernel,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "deepseek_v4_cooperative_dense_attention")?;
    let dims = config.checked()?;
    if token_count == 0 {
        return invalid("cooperative dense attention requires at least one token");
    }
    if token_count > DEEPSEEK_V4_PREFILL_MAX_TOKENS {
        return invalid(format!(
            "cooperative dense attention token count {token_count} exceeds retained chunk limit {DEEPSEEK_V4_PREFILL_MAX_TOKENS}"
        ));
    }
    let token_count_u32 = u32::try_from(token_count).map_err(|_| {
        DeepSeekV4MetalError::Invalid("cooperative dense token count exceeds u32".into())
    })?;
    validate_f32(
        queries,
        &[dims.query_width as u64, token_count as u64],
        false,
        "cooperative dense attention queries",
    )?;
    validate_raw_attention_caches(
        raw_cache,
        raw_cache_before_chunk,
        raw_cache_layout,
        config.head_dim,
        token_count,
        "cooperative dense attention",
    )?;
    validate_f32(
        sinks,
        &[config.head_count as u64],
        false,
        "cooperative dense attention sinks",
    )?;
    validate_f32(
        output,
        &[dims.query_width as u64, token_count as u64],
        true,
        "cooperative dense attention output",
    )?;

    let ratio = match kind {
        AttentionKind::SlidingWindow => 0,
        AttentionKind::CompressedSparse => 4,
        AttentionKind::HeavilyCompressed => 128,
    };
    let end_position = start_position.checked_add(token_count_u32).ok_or_else(|| {
        DeepSeekV4MetalError::Invalid("cooperative dense position overflow".into())
    })?;
    let expected_rows = (end_position as usize).checked_div(ratio).unwrap_or(0);
    let grouped_long_hca = kernel == DeepSeekV4DenseAttentionKernel::GroupedOnline
        && kind == AttentionKind::HeavilyCompressed
        && raw_cache_layout == DeepSeekV4RawCacheLayout::Ring
        && token_count == 1;
    let compressed_cache = match compressed {
        None if expected_rows == 0 => raw_cache,
        Some(rows) if rows.count == expected_rows => {
            validate_f16(
                rows.cache,
                &[config.head_dim as u64, rows.capacity_rows as u64],
                false,
                "cooperative dense compressed cache",
            )?;
            if expected_rows > rows.capacity_rows
                || (!grouped_long_hca && expected_rows > DEEPSEEK_V4_CSA_TOP_K)
            {
                return invalid(format!(
                    "cooperative dense attention cannot consume {expected_rows} rows from capacity {}",
                    rows.capacity_rows
                ));
            }
            rows.cache
        }
        rows => {
            return invalid(format!(
                "cooperative dense attention expected {expected_rows} compressed rows, got {}",
                rows.map_or(0, |rows| rows.count)
            ));
        }
    };
    let final_raw_rows = (end_position as usize).min(DEEPSEEK_V4_LOCAL_WINDOW);
    let maximum_rows = final_raw_rows
        .checked_add(expected_rows)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid("cooperative dense row overflow".into()))?;
    if maximum_rows == 0
        || (!grouped_long_hca && maximum_rows > DEEPSEEK_V4_LOCAL_WINDOW + DEEPSEEK_V4_CSA_TOP_K)
    {
        return invalid(format!(
            "cooperative dense attention row count {maximum_rows} is out of range"
        ));
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        token_count: u32,
        compression_ratio: u32,
        start_position: u32,
        window: u32,
        raw_cache_is_chunk: u32,
        scale: f32,
    }
    let threadgroup_width = config.head_dim.max(maximum_rows);
    let (kernel_name, threadgroup_bytes, grid, threads) = match kernel {
        DeepSeekV4DenseAttentionKernel::Cooperative => (
            "kernel_deepseek_v4_packed_dense_sink_attention_f16",
            (maximum_rows + 1) * std::mem::size_of::<f32>(),
            MTLSize {
                width: token_count,
                height: config.head_count,
                depth: 1,
            },
            MTLSize {
                width: threadgroup_width,
                height: 1,
                depth: 1,
            },
        ),
        DeepSeekV4DenseAttentionKernel::GroupedOnline => {
            if config.head_count != 64 || config.head_dim != DEEPSEEK_V4_HCA_TILE_ROWS {
                return invalid("grouped online dense attention requires 64 heads of width 512");
            }
            (
                "kernel_deepseek_v4_grouped_online_dense_sink_attention_f16",
                DEEPSEEK_V4_GROUPED_DENSE_THREADGROUP_BYTES,
                MTLSize {
                    width: token_count,
                    height: config.head_count / DEEPSEEK_V4_GROUPED_DENSE_HEADS,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: DEEPSEEK_V4_GROUPED_DENSE_HEADS,
                    depth: 1,
                },
            )
        }
    };
    let pso = ctx.pipeline(kernel_name)?;
    match kernel {
        DeepSeekV4DenseAttentionKernel::Cooperative => {
            if pso.maxTotalThreadsPerThreadgroup() < threadgroup_width {
                return invalid(format!(
                    "cooperative dense attention pipeline supports {} threads, requires {threadgroup_width}",
                    pso.maxTotalThreadsPerThreadgroup()
                ));
            }
        }
        DeepSeekV4DenseAttentionKernel::GroupedOnline => {
            if pso.threadExecutionWidth() != 32
                || pso.maxTotalThreadsPerThreadgroup() < DEEPSEEK_V4_GROUPED_DENSE_THREADS
                || ctx.device.maxThreadgroupMemoryLength()
                    < DEEPSEEK_V4_GROUPED_DENSE_THREADGROUP_BYTES
            {
                return invalid(format!(
                    "grouped online dense attention requires SIMD width 32, {} threads, and {} threadgroup bytes; pipeline width={} max_threads={} device_bytes={}",
                    DEEPSEEK_V4_GROUPED_DENSE_THREADS,
                    DEEPSEEK_V4_GROUPED_DENSE_THREADGROUP_BYTES,
                    pso.threadExecutionWidth(),
                    pso.maxTotalThreadsPerThreadgroup(),
                    ctx.device.maxThreadgroupMemoryLength(),
                ));
            }
        }
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: config.head_count as u32,
            head_dim: config.head_dim as u32,
            token_count: token_count_u32,
            compression_ratio: ratio as u32,
            start_position,
            window: DEEPSEEK_V4_LOCAL_WINDOW as u32,
            raw_cache_is_chunk: raw_cache_layout.is_chunk(),
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, raw_cache);
    enc.set_tensor(3, raw_cache_before_chunk);
    enc.set_tensor(4, compressed_cache);
    enc.set_tensor(5, sinks);
    enc.set_tensor(6, output);
    enc.set_threadgroup_memory(0, threadgroup_bytes);
    enc.dispatch(grid, threads);
    Ok(())
}

pub(super) const DEEPSEEK_V4_ONLINE_HCA_THREADS: usize = 32;

pub(super) const DEEPSEEK_V4_ONLINE_HCA_THREADGROUP_BYTES: usize =
    DEEPSEEK_V4_HCA_TILE_ROWS * std::mem::size_of::<half::f16>();

pub(super) fn validate_deepseek_v4_online_hca_request_geometry(
    config: DeepSeekV4PositionZeroAttentionConfig,
    compression_ratio: usize,
    token_count: usize,
    query_token_offset: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if config.head_count != 64
        || config.head_dim != DEEPSEEK_V4_HCA_TILE_ROWS
        || compression_ratio != 128
        || token_count != 1
        || query_token_offset != 0
        || query_count != 1
    {
        return invalid(
            "online tiled HCA requires one complete 64-head x 512-dimension ratio-128 singleton query",
        );
    }
    Ok(())
}

pub(super) fn validate_deepseek_v4_online_hca_launch_geometry(
    thread_execution_width: usize,
    max_threads_per_group: usize,
    max_threadgroup_bytes: usize,
    required_threadgroup_bytes: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if thread_execution_width != DEEPSEEK_V4_ONLINE_HCA_THREADS
        || max_threads_per_group < DEEPSEEK_V4_ONLINE_HCA_THREADS
    {
        return invalid(format!(
            "online tiled HCA requires SIMD width {} and {} threads, got width {thread_execution_width} max {max_threads_per_group}",
            DEEPSEEK_V4_ONLINE_HCA_THREADS, DEEPSEEK_V4_ONLINE_HCA_THREADS,
        ));
    }
    if max_threadgroup_bytes < required_threadgroup_bytes {
        return invalid(format!(
            "online tiled HCA requires {} threadgroup bytes, device allows {max_threadgroup_bytes}",
            required_threadgroup_bytes,
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_tiled_dense_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    raw_cache_layout: DeepSeekV4RawCacheLayout,
    compressed: DeepSeekV4PublishedRows<'_>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    chunk_start_position: u32,
    query_token_offset: usize,
    query_count: usize,
    compression_ratio: usize,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    encode_tiled_dense_sink_attention_f16_with_mode(
        ctx,
        enc,
        queries,
        raw_cache,
        raw_cache_before_chunk,
        raw_cache_layout,
        compressed,
        sinks,
        output,
        chunk_start_position,
        query_token_offset,
        query_count,
        compression_ratio,
        config,
        false,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_online_dense_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    raw_cache_layout: DeepSeekV4RawCacheLayout,
    compressed: DeepSeekV4PublishedRows<'_>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    chunk_start_position: u32,
    query_token_offset: usize,
    query_count: usize,
    compression_ratio: usize,
    direct_load: bool,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    encode_tiled_dense_sink_attention_f16_with_mode(
        ctx,
        enc,
        queries,
        raw_cache,
        raw_cache_before_chunk,
        raw_cache_layout,
        compressed,
        sinks,
        output,
        chunk_start_position,
        query_token_offset,
        query_count,
        compression_ratio,
        config,
        true,
        direct_load,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_tiled_dense_sink_attention_f16_with_mode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    raw_cache_layout: DeepSeekV4RawCacheLayout,
    compressed: DeepSeekV4PublishedRows<'_>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    chunk_start_position: u32,
    query_token_offset: usize,
    query_count: usize,
    compression_ratio: usize,
    config: DeepSeekV4PositionZeroAttentionConfig,
    online: bool,
    direct_load: bool,
) -> Result<(), DeepSeekV4MetalError> {
    let query_width = checked_mul(config.head_count, config.head_dim, "tiled query width")?;
    if config.head_dim != DEEPSEEK_V4_HCA_TILE_ROWS || compression_ratio != 128 || query_count == 0
    {
        return invalid("tiled dense attention requires 512-wide ratio-128 HCA queries");
    }
    if queries.shape.len() != 2 || output.shape.len() != 2 {
        return invalid("tiled dense attention requires token-major rank-2 queries and output");
    }
    let token_count = usize::try_from(queries.shape[1])
        .map_err(|_| DeepSeekV4MetalError::Invalid("tiled query count exceeds usize".into()))?;
    if direct_load && !online {
        return invalid("direct row loading requires online dense attention");
    }
    if online {
        validate_deepseek_v4_online_hca_request_geometry(
            config,
            compression_ratio,
            token_count,
            query_token_offset,
            query_count,
        )?;
        require_serial(enc, "deepseek_v4_online_dense_sink_attention_f16")?;
    }
    let query_end = query_token_offset
        .checked_add(query_count)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid("tiled query range overflow".into()))?;
    if query_end > token_count || output.shape != queries.shape {
        return invalid(format!(
            "tiled dense attention query range {query_token_offset}..{query_end} exceeds {token_count} tokens or output shape differs"
        ));
    }
    validate_f32(
        queries,
        &[query_width as u64, token_count as u64],
        false,
        "tiled attention queries",
    )?;
    validate_f32(
        output,
        &[query_width as u64, token_count as u64],
        true,
        "tiled attention output",
    )?;
    validate_raw_attention_caches(
        raw_cache,
        raw_cache_before_chunk,
        raw_cache_layout,
        config.head_dim,
        token_count,
        "tiled attention",
    )?;
    validate_f16(
        compressed.cache,
        &[config.head_dim as u64, compressed.capacity_rows as u64],
        false,
        "tiled compressed attention cache",
    )?;
    validate_f32(
        sinks,
        &[config.head_count as u64],
        false,
        "tiled attention sinks",
    )?;

    let first_position =
        chunk_start_position
            .checked_add(u32::try_from(query_token_offset).map_err(|_| {
                DeepSeekV4MetalError::Invalid("tiled query offset exceeds u32".into())
            })?)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("tiled first position overflow".into()))?;
    let end_position = chunk_start_position
        .checked_add(u32::try_from(query_end).map_err(|_| {
            DeepSeekV4MetalError::Invalid("tiled query endpoint exceeds u32".into())
        })?)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid("tiled end position overflow".into()))?;
    let first_visible = (u64::from(first_position) + 1) as usize / compression_ratio;
    let final_visible = end_position as usize / compression_ratio;
    if first_visible == 0
        || compressed.count != final_visible
        || compressed.count > compressed.capacity_rows
    {
        return invalid(format!(
            "tiled HCA requires 1..={} visible rows, first={first_visible} final={final_visible} stored={}/{}",
            compressed.capacity_rows, compressed.count, compressed.capacity_rows
        ));
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        query_count: u32,
        query_token_offset: u32,
        chunk_start_position: u32,
        window: u32,
        compression_ratio: u32,
        raw_cache_is_chunk: u32,
        scale: f32,
    }
    let args = Args {
        head_count: u32::try_from(config.head_count).map_err(|_| {
            DeepSeekV4MetalError::Invalid("tiled HCA head count exceeds u32".into())
        })?,
        head_dim: DEEPSEEK_V4_HCA_TILE_ROWS as u32,
        query_count: u32::try_from(query_count).map_err(|_| {
            DeepSeekV4MetalError::Invalid("tiled HCA query count exceeds u32".into())
        })?,
        query_token_offset: u32::try_from(query_token_offset).map_err(|_| {
            DeepSeekV4MetalError::Invalid("tiled HCA query offset exceeds u32".into())
        })?,
        chunk_start_position,
        window: DEEPSEEK_V4_LOCAL_WINDOW as u32,
        compression_ratio: compression_ratio as u32,
        raw_cache_is_chunk: raw_cache_layout.is_chunk(),
        scale: 1.0 / (config.head_dim as f32).sqrt(),
    };
    let (kernel, threadgroup_width, threadgroup_bytes) = if online {
        (
            if direct_load {
                "kernel_deepseek_v4_online_dense_sink_attention_f16_direct"
            } else {
                "kernel_deepseek_v4_online_dense_sink_attention_f16"
            },
            DEEPSEEK_V4_ONLINE_HCA_THREADS,
            if direct_load {
                0
            } else {
                DEEPSEEK_V4_ONLINE_HCA_THREADGROUP_BYTES
            },
        )
    } else {
        (
            "kernel_deepseek_v4_tiled_dense_sink_attention_f16",
            DEEPSEEK_V4_HCA_TILE_ROWS,
            (2 * DEEPSEEK_V4_HCA_TILE_ROWS + 1) * std::mem::size_of::<f32>(),
        )
    };
    let pso = ctx.pipeline(kernel)?;
    if online {
        validate_deepseek_v4_online_hca_launch_geometry(
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup(),
            ctx.device.maxThreadgroupMemoryLength(),
            threadgroup_bytes,
        )?;
    } else if pso.maxTotalThreadsPerThreadgroup() < threadgroup_width {
        return invalid(format!(
            "tiled HCA pipeline supports {} threads, requires {threadgroup_width}",
            pso.maxTotalThreadsPerThreadgroup()
        ));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, queries);
    enc.set_tensor(2, raw_cache);
    enc.set_tensor(3, raw_cache_before_chunk);
    enc.set_tensor(4, compressed.cache);
    enc.set_tensor(5, sinks);
    enc.set_tensor(6, output);
    if threadgroup_bytes != 0 {
        enc.set_threadgroup_memory(0, threadgroup_bytes);
    }
    enc.dispatch(
        MTLSize {
            width: query_count,
            height: config.head_count,
            depth: 1,
        },
        MTLSize {
            width: threadgroup_width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_compressor_frontier_write(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    projected_kv: &MetalTensor,
    projected_score: &MetalTensor,
    ape: &MetalTensor,
    kv_state: &MetalTensor,
    score_state: &MetalTensor,
    width: usize,
    row: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let row_offset = checked_mul(row, width, "compressor frontier row offset")?;
    for (tensor, name) in [
        (projected_kv, "projected compressor KV"),
        (projected_score, "projected compressor score"),
        (ape, "compressor APE row"),
    ] {
        validate_f32(tensor, &[width as u64], false, name)?;
    }
    validate_f32(kv_state, &kv_state.shape, true, "compressor KV state")?;
    validate_f32(
        score_state,
        &score_state.shape,
        true,
        "compressor score state",
    )?;
    if kv_state.shape != score_state.shape
        || row_offset
            .checked_add(width)
            .is_none_or(|end| end as u64 > kv_state.n_elements())
    {
        return invalid("compressor frontier row exceeds aligned KV/score state");
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        width: u32,
        row_offset: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_compressor_frontier_write")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            width: width as u32,
            row_offset: row_offset as u32,
        },
    );
    enc.set_tensor(1, projected_kv);
    enc.set_tensor(2, projected_score);
    enc.set_tensor(3, ape);
    enc.set_tensor(4, kv_state);
    enc.set_tensor(5, score_state);
    enc.dispatch(
        MTLSize {
            width: width.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_compressor_frontier_chunk(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    projected_kv: &MetalTensor,
    projected_score: &MetalTensor,
    ape: &MetalTensor,
    kv_state: &MetalTensor,
    score_state: &MetalTensor,
    pooled_rows: &MetalTensor,
    ratio: usize,
    head_dim: usize,
    width: usize,
    row_count: usize,
    start_position: u32,
    output_rows: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if !matches!(ratio, 4 | 128) || row_count == 0 {
        return invalid("compressor chunk requires ratio 4 or 128 and at least one row");
    }
    let coefficient = if ratio == 4 { 2 } else { 1 };
    if width != coefficient * head_dim {
        return invalid("compressor chunk width differs from its ratio geometry");
    }
    let state_rows = checked_mul(coefficient, ratio, "compressor chunk state rows")?;
    validate_f32(
        projected_kv,
        &[width as u64, row_count as u64],
        false,
        "projected compressor KV chunk",
    )?;
    validate_f32(
        projected_score,
        &[width as u64, row_count as u64],
        false,
        "projected compressor score chunk",
    )?;
    validate_f32(
        ape,
        &[width as u64, ratio as u64],
        false,
        "compressor chunk APE",
    )?;
    for (state, name) in [
        (kv_state, "compressor chunk KV state"),
        (score_state, "compressor chunk score state"),
    ] {
        validate_f32(state, &[width as u64, state_rows as u64], true, name)?;
    }
    if pooled_rows.dtype != GgmlType::F32
        || !pooled_rows.is_writable()
        || pooled_rows.shape.len() != 2
        || pooled_rows.shape[0] != head_dim as u64
        || pooled_rows.shape[1] < output_rows as u64
    {
        return invalid(format!(
            "compressor pooled scratch must hold [{head_dim}, >={output_rows}], got {:?} {:?}",
            pooled_rows.dtype, pooled_rows.shape
        ));
    }
    let end_position = start_position
        .checked_add(u32::try_from(row_count).map_err(|_| {
            DeepSeekV4MetalError::Invalid("compressor chunk row count exceeds u32".into())
        })?)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("compressor chunk position overflow".into())
        })?;
    let expected_rows = end_position as usize / ratio - start_position as usize / ratio;
    if output_rows != expected_rows {
        return invalid(format!(
            "compressor chunk expected {expected_rows} pooled rows, got {output_rows}"
        ));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ratio: u32,
        head_dim: u32,
        width: u32,
        row_count: u32,
        start_position: u32,
        output_rows: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_compressor_frontier_chunk")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            ratio: u32::try_from(ratio).map_err(|_| {
                DeepSeekV4MetalError::Invalid("compressor chunk ratio exceeds u32".into())
            })?,
            head_dim: u32::try_from(head_dim).map_err(|_| {
                DeepSeekV4MetalError::Invalid("compressor chunk head dimension exceeds u32".into())
            })?,
            width: u32::try_from(width).map_err(|_| {
                DeepSeekV4MetalError::Invalid("compressor chunk width exceeds u32".into())
            })?,
            row_count: u32::try_from(row_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("compressor chunk rows exceed u32".into())
            })?,
            start_position,
            output_rows: u32::try_from(output_rows).map_err(|_| {
                DeepSeekV4MetalError::Invalid("compressor output rows exceed u32".into())
            })?,
        },
    );
    enc.set_tensor(1, projected_kv);
    enc.set_tensor(2, projected_score);
    enc.set_tensor(3, ape);
    enc.set_tensor(4, kv_state);
    enc.set_tensor(5, score_state);
    enc.set_tensor(6, pooled_rows);
    enc.dispatch(
        MTLSize {
            width: head_dim.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_compressor_pool(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    kv_state: &MetalTensor,
    score_state: &MetalTensor,
    output: &MetalTensor,
    ratio: usize,
    head_dim: usize,
    width: usize,
    rows: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if !matches!(ratio, 4 | 128) {
        return invalid(format!("unsupported compressor pooling ratio {ratio}"));
    }
    let coefficient = if ratio == 4 { 2 } else { 1 };
    if width != coefficient * head_dim || rows != coefficient * ratio {
        return invalid("compressor pooling geometry is inconsistent");
    }
    validate_f32(
        kv_state,
        &[width as u64, rows as u64],
        false,
        "compressor pooling KV state",
    )?;
    validate_f32(
        score_state,
        &[width as u64, rows as u64],
        false,
        "compressor pooling score state",
    )?;
    validate_f32(output, &[head_dim as u64], true, "compressor pooled output")?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ratio: u32,
        head_dim: u32,
        width: u32,
        rows: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_compressor_pool")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            ratio: ratio as u32,
            head_dim: head_dim as u32,
            width: width as u32,
            rows: rows as u32,
        },
    );
    enc.set_tensor(1, kv_state);
    enc.set_tensor(2, score_state);
    enc.set_tensor(3, output);
    enc.dispatch(
        MTLSize {
            width: head_dim.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(super) fn encode_compressor_roll_ratio4(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    kv_state: &MetalTensor,
    score_state: &MetalTensor,
    width: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let shape = [width as u64, 8];
    validate_f32(kv_state, &shape, true, "ratio-4 KV frontier")?;
    validate_f32(score_state, &shape, true, "ratio-4 score frontier")?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        width: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_compressor_roll_ratio4")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            width: width as u32,
        },
    );
    enc.set_tensor(1, kv_state);
    enc.set_tensor(2, score_state);
    let elements = checked_mul(4, width, "ratio-4 roll elements")?;
    enc.dispatch(
        MTLSize {
            width: elements.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_lightning_indexer_scores_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    head_weights: &MetalTensor,
    keys: &MetalTensor,
    visible_counts: &MetalTensor,
    scores: &MetalTensor,
    head_count: usize,
    head_dim: usize,
    row_capacity: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    encode_lightning_indexer_scores_f16_with_policy(
        ctx,
        enc,
        queries,
        head_weights,
        keys,
        visible_counts,
        scores,
        head_count,
        head_dim,
        row_capacity,
        query_count,
        false,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DeepSeekV4LightningScoreKernel {
    Scalar,
    Cooperative,
    TiledF32,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_lightning_indexer_scores_f16_with_limit(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    head_weights: &MetalTensor,
    keys: &MetalTensor,
    visible_counts: &MetalTensor,
    scores: &MetalTensor,
    head_count: usize,
    head_dim: usize,
    row_capacity: usize,
    max_dispatched_rows: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    encode_lightning_indexer_scores_f16_with_limit_and_kernel(
        ctx,
        enc,
        queries,
        head_weights,
        keys,
        visible_counts,
        scores,
        head_count,
        head_dim,
        row_capacity,
        max_dispatched_rows,
        query_count,
        if head_count == 64 && head_dim == 128 {
            DeepSeekV4LightningScoreKernel::Cooperative
        } else {
            DeepSeekV4LightningScoreKernel::Scalar
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_lightning_indexer_scores_f16_tiled_f32_with_limit(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    head_weights: &MetalTensor,
    keys: &MetalTensor,
    visible_counts: &MetalTensor,
    scores: &MetalTensor,
    head_count: usize,
    head_dim: usize,
    row_capacity: usize,
    max_dispatched_rows: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    encode_lightning_indexer_scores_f16_with_limit_and_kernel(
        ctx,
        enc,
        queries,
        head_weights,
        keys,
        visible_counts,
        scores,
        head_count,
        head_dim,
        row_capacity,
        max_dispatched_rows,
        query_count,
        DeepSeekV4LightningScoreKernel::TiledF32,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_lightning_indexer_scores_f16_with_policy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    head_weights: &MetalTensor,
    keys: &MetalTensor,
    visible_counts: &MetalTensor,
    scores: &MetalTensor,
    head_count: usize,
    head_dim: usize,
    row_capacity: usize,
    query_count: usize,
    force_scalar: bool,
) -> Result<(), DeepSeekV4MetalError> {
    encode_lightning_indexer_scores_f16_with_limit_and_kernel(
        ctx,
        enc,
        queries,
        head_weights,
        keys,
        visible_counts,
        scores,
        head_count,
        head_dim,
        row_capacity,
        row_capacity,
        query_count,
        if !force_scalar && head_count == 64 && head_dim == 128 {
            DeepSeekV4LightningScoreKernel::Cooperative
        } else {
            DeepSeekV4LightningScoreKernel::Scalar
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_lightning_indexer_scores_f16_with_limit_and_kernel(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    head_weights: &MetalTensor,
    keys: &MetalTensor,
    visible_counts: &MetalTensor,
    scores: &MetalTensor,
    head_count: usize,
    head_dim: usize,
    row_capacity: usize,
    max_dispatched_rows: usize,
    query_count: usize,
    kernel: DeepSeekV4LightningScoreKernel,
) -> Result<(), DeepSeekV4MetalError> {
    for (name, value) in [
        ("indexer head count", head_count),
        ("indexer head dimension", head_dim),
        ("indexer row capacity", row_capacity),
        ("indexer maximum dispatched rows", max_dispatched_rows),
        ("indexer query count", query_count),
    ] {
        if value == 0 || u32::try_from(value).is_err() {
            return invalid(format!("{name} must be nonzero and fit u32"));
        }
    }
    if max_dispatched_rows > row_capacity {
        return invalid(format!(
            "indexer maximum dispatched rows {max_dispatched_rows} exceed row capacity {row_capacity}"
        ));
    }
    if kernel == DeepSeekV4LightningScoreKernel::TiledF32 && (head_count != 64 || head_dim != 128) {
        return invalid(format!(
            "tiled F32 indexer scoring requires 64 heads of width 128, got {head_count}x{head_dim}"
        ));
    }
    validate_lightning_indexer_score_offsets(head_count, head_dim, row_capacity, query_count)?;
    validate_f32(
        queries,
        &[head_dim as u64, head_count as u64, query_count as u64],
        false,
        "indexer queries",
    )?;
    validate_f32(
        head_weights,
        &[head_count as u64, query_count as u64],
        false,
        "indexer head weights",
    )?;
    validate_f16(
        keys,
        &[head_dim as u64, row_capacity as u64],
        false,
        "indexer keys",
    )?;
    validate_i32(
        visible_counts,
        &[query_count as u64],
        false,
        "indexer visible counts",
    )?;
    validate_f32(
        scores,
        &[row_capacity as u64, query_count as u64],
        true,
        "indexer scores",
    )?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        row_capacity: u32,
        query_count: u32,
    }
    let kernel_name = match kernel {
        DeepSeekV4LightningScoreKernel::Scalar => "kernel_deepseek_v4_lightning_indexer_scores_f16",
        DeepSeekV4LightningScoreKernel::Cooperative => {
            "kernel_deepseek_v4_lightning_indexer_scores_f16_cooperative"
        }
        DeepSeekV4LightningScoreKernel::TiledF32 => {
            "kernel_deepseek_v4_lightning_indexer_scores_f16_tiled_f32"
        }
    };
    let pso = ctx.pipeline(kernel_name)?;
    match kernel {
        DeepSeekV4LightningScoreKernel::Scalar => {}
        DeepSeekV4LightningScoreKernel::Cooperative => {
            validate_cooperative_lightning_score_geometry(
                kernel_name,
                pso.threadExecutionWidth(),
                pso.maxTotalThreadsPerThreadgroup(),
                ctx.device.maxThreadgroupMemoryLength(),
            )?;
        }
        DeepSeekV4LightningScoreKernel::TiledF32 => {
            validate_tiled_f32_lightning_score_geometry(
                kernel_name,
                pso.threadExecutionWidth(),
                pso.maxTotalThreadsPerThreadgroup(),
                ctx.device.maxThreadgroupMemoryLength(),
            )?;
        }
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: head_count as u32,
            head_dim: head_dim as u32,
            row_capacity: row_capacity as u32,
            query_count: query_count as u32,
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, head_weights);
    enc.set_tensor(3, keys);
    enc.set_tensor(4, visible_counts);
    enc.set_tensor(5, scores);
    match kernel {
        DeepSeekV4LightningScoreKernel::Scalar => {
            enc.dispatch(
                MTLSize {
                    width: max_dispatched_rows.div_ceil(256),
                    height: query_count,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        DeepSeekV4LightningScoreKernel::Cooperative => {
            enc.set_threadgroup_memory(0, 8 * 128 * std::mem::size_of::<u16>());
            enc.set_threadgroup_memory(1, 8 * 64 * std::mem::size_of::<f32>());
            enc.dispatch(
                MTLSize {
                    width: max_dispatched_rows.div_ceil(8),
                    height: query_count,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        DeepSeekV4LightningScoreKernel::TiledF32 => {
            enc.set_threadgroup_memory(0, 5_376 * std::mem::size_of::<f32>());
            enc.dispatch(
                MTLSize {
                    width: max_dispatched_rows.div_ceil(32),
                    height: query_count.div_ceil(8),
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_lightning_indexer_scores_f16_matrix(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    head_weights: &MetalTensor,
    keys: &MetalTensor,
    visible_counts: &MetalTensor,
    scores: &MetalTensor,
    head_count: usize,
    head_dim: usize,
    row_capacity: usize,
    max_dispatched_rows: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const KERNEL: &str = "kernel_deepseek_v4_lightning_indexer_scores_f16_matrix_ceiling";
    if head_count != 64 || head_dim != 128 {
        return invalid(format!(
            "{KERNEL} requires 64 heads of width 128, got {head_count}x{head_dim}"
        ));
    }
    for (name, value) in [
        ("indexer row capacity", row_capacity),
        ("indexer maximum dispatched rows", max_dispatched_rows),
        ("indexer query count", query_count),
    ] {
        if value == 0 || u32::try_from(value).is_err() {
            return invalid(format!("{name} must be nonzero and fit u32"));
        }
    }
    if max_dispatched_rows > row_capacity {
        return invalid(format!(
            "matrix-ceiling indexer maximum dispatched rows {max_dispatched_rows} exceed row capacity {row_capacity}"
        ));
    }
    validate_lightning_indexer_score_offsets(head_count, head_dim, row_capacity, query_count)?;
    validate_f16(
        queries,
        &[head_dim as u64, head_count as u64, query_count as u64],
        false,
        "matrix-ceiling indexer queries",
    )?;
    validate_f32(
        head_weights,
        &[head_count as u64, query_count as u64],
        false,
        "matrix-ceiling indexer head weights",
    )?;
    validate_f16(
        keys,
        &[head_dim as u64, row_capacity as u64],
        false,
        "matrix-ceiling indexer keys",
    )?;
    validate_i32(
        visible_counts,
        &[query_count as u64],
        false,
        "matrix-ceiling indexer visible counts",
    )?;
    validate_f32(
        scores,
        &[row_capacity as u64, query_count as u64],
        true,
        "matrix-ceiling indexer scores",
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        row_capacity: u32,
        query_count: u32,
    }

    let pso = ctx.pipeline(KERNEL)?;
    validate_cooperative_lightning_score_geometry(
        KERNEL,
        pso.threadExecutionWidth(),
        pso.maxTotalThreadsPerThreadgroup(),
        ctx.device.maxThreadgroupMemoryLength(),
    )?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: head_count as u32,
            head_dim: head_dim as u32,
            row_capacity: row_capacity as u32,
            query_count: query_count as u32,
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, head_weights);
    enc.set_tensor(3, keys);
    enc.set_tensor(4, visible_counts);
    enc.set_tensor(5, scores);
    enc.set_threadgroup_memory(0, 8 * 128 * std::mem::size_of::<u16>());
    enc.set_threadgroup_memory(1, 8 * 64 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: max_dispatched_rows.div_ceil(8),
            height: query_count,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
pub(super) fn encode_indexer_fp4_contract_primitives(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    e2m1_values: &MetalTensor,
    scale_maxima: &MetalTensor,
    e2m1_codes: &MetalTensor,
    scale_codes: &MetalTensor,
    scale_status: &MetalTensor,
    e2m1_count: usize,
    scale_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if e2m1_count == 0
        || scale_count == 0
        || u32::try_from(e2m1_count).is_err()
        || u32::try_from(scale_count).is_err()
    {
        return invalid("indexer FP4 primitive counts must be nonzero and fit u32");
    }
    validate_f32(
        e2m1_values,
        &[e2m1_count as u64],
        false,
        "indexer FP4 E2M1 primitive inputs",
    )?;
    validate_f32(
        scale_maxima,
        &[scale_count as u64],
        false,
        "indexer FP4 scale primitive inputs",
    )?;
    validate_i8(
        e2m1_codes,
        &[e2m1_count as u64],
        true,
        "indexer FP4 E2M1 primitive codes",
    )?;
    validate_i8(
        scale_codes,
        &[scale_count as u64],
        true,
        "indexer FP4 scale primitive codes",
    )?;
    validate_i32(
        scale_status,
        &[scale_count as u64],
        true,
        "indexer FP4 scale primitive status",
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        e2m1_count: u32,
        scale_count: u32,
    }

    let pso = ctx.pipeline("kernel_deepseek_v4_indexer_fp4_contract_primitives")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            e2m1_count: e2m1_count as u32,
            scale_count: scale_count as u32,
        },
    );
    enc.set_tensor(1, e2m1_values);
    enc.set_tensor(2, scale_maxima);
    enc.set_tensor(3, e2m1_codes);
    enc.set_tensor(4, scale_codes);
    enc.set_tensor(5, scale_status);
    enc.dispatch(
        MTLSize {
            width: e2m1_count.max(scale_count).div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
pub(super) fn encode_pack_indexer_fp4_rows_shadow(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    packed_values: &MetalTensor,
    packed_scales: &MetalTensor,
    status: &MetalTensor,
    row_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const VALUES_PER_ROW: usize = crate::deepseek_v4_oracle::INDEXER_FP4_VALUES_PER_ROW;
    const VALUE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES;
    const SCALE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES;
    if row_count == 0 || u32::try_from(row_count).is_err() {
        return invalid("indexer FP4 row count must be nonzero and fit u32");
    }
    for (name, elements) in [
        (
            "indexer FP4 input elements",
            checked_mul(row_count, VALUES_PER_ROW, "indexer FP4 input elements")?,
        ),
        (
            "indexer FP4 value bytes",
            checked_mul(row_count, VALUE_BYTES, "indexer FP4 value bytes")?,
        ),
        (
            "indexer FP4 scale bytes",
            checked_mul(row_count, SCALE_BYTES, "indexer FP4 scale bytes")?,
        ),
    ] {
        if u32::try_from(elements).is_err() {
            return invalid(format!("{name} exceed u32 shader offsets"));
        }
    }
    let trailing_rows = input
        .shape
        .get(1..)
        .and_then(crate::tensor::checked_shape_elements)
        .and_then(|rows| usize::try_from(rows).ok());
    if input.dtype != GgmlType::F32
        || input.shape.first().copied() != Some(VALUES_PER_ROW as u64)
        || trailing_rows != Some(row_count)
    {
        return invalid(format!(
            "indexer FP4 pack input must be F32 with width {VALUES_PER_ROW} and {row_count} rows, got {:?} {:?}",
            input.dtype, input.shape
        ));
    }
    let mut value_shape = input.shape.clone();
    value_shape[0] = VALUE_BYTES as u64;
    let mut scale_shape = input.shape.clone();
    scale_shape[0] = SCALE_BYTES as u64;
    validate_f32(input, &input.shape, false, "indexer FP4 pack input")?;
    validate_i8(
        packed_values,
        &value_shape,
        true,
        "indexer FP4 packed values",
    )?;
    validate_i8(
        packed_scales,
        &scale_shape,
        true,
        "indexer FP4 packed scales",
    )?;
    validate_i32(status, &[row_count as u64], true, "indexer FP4 pack status")?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_count: u32,
    }

    let pso = ctx.pipeline("kernel_deepseek_v4_pack_indexer_fp4_rows_shadow")?;
    validate_fp4_pack_geometry(
        pso.threadExecutionWidth(),
        pso.maxTotalThreadsPerThreadgroup(),
        ctx.device.maxThreadgroupMemoryLength(),
    )?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            row_count: row_count as u32,
        },
    );
    enc.set_tensor(1, input);
    enc.set_tensor(2, packed_values);
    enc.set_tensor(3, packed_scales);
    enc.set_tensor(4, status);
    enc.dispatch(
        MTLSize {
            width: row_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
pub(super) fn encode_unpack_indexer_fp4_units_shadow(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    packed_values: &MetalTensor,
    status: &MetalTensor,
    units: &MetalTensor,
    row_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const VALUE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES;
    const VALUES_PER_ROW: usize = crate::deepseek_v4_oracle::INDEXER_FP4_VALUES_PER_ROW;
    if row_count == 0 || u32::try_from(row_count).is_err() {
        return invalid("indexer FP4 unpack row count must be nonzero and fit u32");
    }
    let trailing_rows = packed_values
        .shape
        .get(1..)
        .and_then(crate::tensor::checked_shape_elements)
        .and_then(|rows| usize::try_from(rows).ok());
    if packed_values.dtype != GgmlType::I8
        || packed_values.shape.first().copied() != Some(VALUE_BYTES as u64)
        || trailing_rows != Some(row_count)
    {
        return invalid(format!(
            "indexer FP4 unpack values must be raw I8 width {VALUE_BYTES} with {row_count} rows, got {:?} {:?}",
            packed_values.dtype, packed_values.shape
        ));
    }
    let mut unit_shape = packed_values.shape.clone();
    unit_shape[0] = VALUES_PER_ROW as u64;
    validate_i8(
        packed_values,
        &packed_values.shape,
        false,
        "indexer FP4 unpack values",
    )?;
    validate_i32(
        status,
        &[row_count as u64],
        false,
        "indexer FP4 unpack status",
    )?;
    validate_f16(units, &unit_shape, true, "indexer FP4 unpacked units")?;
    let element_count = checked_mul(row_count, VALUES_PER_ROW, "indexer FP4 unpack elements")?;
    if u32::try_from(element_count).is_err() {
        return invalid("indexer FP4 unpack elements exceed u32 shader offsets");
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_count: u32,
    }

    let pso = ctx.pipeline("kernel_deepseek_v4_unpack_indexer_fp4_units_shadow")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            row_count: row_count as u32,
        },
    );
    enc.set_tensor(1, packed_values);
    enc.set_tensor(2, status);
    enc.set_tensor(3, units);
    enc.dispatch(
        MTLSize {
            width: element_count.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(feature = "dsv4-diagnostics")]
pub(super) fn encode_indexer_fp4_shadow_preflight(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query_status: &MetalTensor,
    key_status: &MetalTensor,
    requested_visible: &MetalTensor,
    eligible_visible: &MetalTensor,
    eligibility_record: &MetalTensor,
    row_capacity: usize,
    expected_visible: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const QUERY_ROWS: usize = 64;
    if row_capacity == 0
        || expected_visible == 0
        || expected_visible > row_capacity
        || u32::try_from(row_capacity).is_err()
        || u32::try_from(expected_visible).is_err()
    {
        return invalid(
            "indexer FP4 preflight capacity/visibility must be nonzero, ordered, and fit u32",
        );
    }
    validate_i32(
        query_status,
        &[QUERY_ROWS as u64],
        false,
        "indexer FP4 query status",
    )?;
    validate_i32(
        key_status,
        &[row_capacity as u64],
        false,
        "indexer FP4 key status",
    )?;
    validate_i32(
        requested_visible,
        &[1],
        false,
        "indexer FP4 requested visibility",
    )?;
    validate_i32(
        eligible_visible,
        &[1],
        true,
        "indexer FP4 eligible visibility",
    )?;
    validate_i32(
        eligibility_record,
        &[3],
        true,
        "indexer FP4 eligibility record",
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        query_rows: u32,
        row_capacity: u32,
        expected_visible: u32,
    }

    let pso = ctx.pipeline("kernel_deepseek_v4_indexer_fp4_shadow_preflight")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            query_rows: QUERY_ROWS as u32,
            row_capacity: row_capacity as u32,
            expected_visible: expected_visible as u32,
        },
    );
    enc.set_tensor(1, query_status);
    enc.set_tensor(2, key_status);
    enc.set_tensor(3, requested_visible);
    enc.set_tensor(4, eligible_visible);
    enc.set_tensor(5, eligibility_record);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
pub(super) fn encode_validate_indexer_fp4_rows_shadow(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    packed_values: &MetalTensor,
    packed_scales: &MetalTensor,
    status: &MetalTensor,
    row_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const VALUE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES;
    const SCALE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES;
    if row_count == 0 || u32::try_from(row_count).is_err() {
        return invalid("indexer FP4 validation row count must be nonzero and fit u32");
    }
    validate_i8(
        packed_values,
        &[VALUE_BYTES as u64, row_count as u64],
        false,
        "indexer FP4 validation values",
    )?;
    validate_i8(
        packed_scales,
        &[SCALE_BYTES as u64, row_count as u64],
        false,
        "indexer FP4 validation scales",
    )?;
    validate_i32(
        status,
        &[row_count as u64],
        true,
        "indexer FP4 validation status",
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_count: u32,
    }

    let pso = ctx.pipeline("kernel_deepseek_v4_validate_indexer_fp4_rows_shadow")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            row_count: row_count as u32,
        },
    );
    enc.set_tensor(1, packed_values);
    enc.set_tensor(2, packed_scales);
    enc.set_tensor(3, status);
    enc.dispatch(
        MTLSize {
            width: row_count.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_lightning_indexer_scores_fp4_matrix_shadow(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query_units: &MetalTensor,
    query_scales: &MetalTensor,
    head_weights: &MetalTensor,
    key_values: &MetalTensor,
    key_scales: &MetalTensor,
    visible_counts: &MetalTensor,
    scores: &MetalTensor,
    row_capacity: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const KERNEL: &str = "kernel_deepseek_v4_lightning_indexer_scores_fp4_matrix_shadow";
    const HEAD_COUNT: usize = 64;
    const HEAD_DIM: usize = 128;
    const VALUE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES;
    const SCALE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES;
    if row_capacity == 0
        || query_count == 0
        || u32::try_from(row_capacity).is_err()
        || u32::try_from(query_count).is_err()
    {
        return invalid("FP4 matrix score row and query counts must be nonzero and fit u32");
    }
    validate_fp4_lightning_score_offsets(row_capacity, query_count)?;
    validate_f16(
        query_units,
        &[HEAD_DIM as u64, HEAD_COUNT as u64, query_count as u64],
        false,
        "FP4 matrix indexer query units",
    )?;
    validate_i8(
        query_scales,
        &[SCALE_BYTES as u64, HEAD_COUNT as u64, query_count as u64],
        false,
        "FP4 matrix indexer query scales",
    )?;
    validate_f32(
        head_weights,
        &[HEAD_COUNT as u64, query_count as u64],
        false,
        "FP4 matrix indexer head weights",
    )?;
    validate_i8(
        key_values,
        &[VALUE_BYTES as u64, row_capacity as u64],
        false,
        "FP4 matrix indexer key values",
    )?;
    validate_i8(
        key_scales,
        &[SCALE_BYTES as u64, row_capacity as u64],
        false,
        "FP4 matrix indexer key scales",
    )?;
    validate_i32(
        visible_counts,
        &[query_count as u64],
        false,
        "FP4 matrix indexer visible counts",
    )?;
    validate_f32(
        scores,
        &[row_capacity as u64, query_count as u64],
        true,
        "FP4 matrix indexer scores",
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        row_capacity: u32,
        query_count: u32,
    }

    let pso = ctx.pipeline(KERNEL)?;
    validate_fp4_matrix_score_geometry(
        KERNEL,
        pso.threadExecutionWidth(),
        pso.maxTotalThreadsPerThreadgroup(),
        ctx.device.maxThreadgroupMemoryLength(),
    )?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: HEAD_COUNT as u32,
            head_dim: HEAD_DIM as u32,
            row_capacity: row_capacity as u32,
            query_count: query_count as u32,
        },
    );
    enc.set_tensor(1, query_units);
    enc.set_tensor(2, query_scales);
    enc.set_tensor(3, head_weights);
    enc.set_tensor(4, key_values);
    enc.set_tensor(5, key_scales);
    enc.set_tensor(6, visible_counts);
    enc.set_tensor(7, scores);
    enc.set_threadgroup_memory(0, 8 * 32 * std::mem::size_of::<u16>());
    enc.set_threadgroup_memory(1, 64 * 8 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: row_capacity.div_ceil(8),
            height: query_count,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
pub(super) fn validate_fp4_lightning_score_offsets(
    row_capacity: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const HEAD_COUNT: usize = 64;
    const VALUE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES;
    const SCALE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES;
    for (name, elements) in [
        (
            "FP4 indexer score elements",
            checked_mul(row_capacity, query_count, "FP4 indexer score elements")?,
        ),
        (
            "FP4 indexer weight elements",
            checked_mul(HEAD_COUNT, query_count, "FP4 indexer weight elements")?,
        ),
        (
            "FP4 indexer query unit elements",
            checked_mul(
                checked_mul(HEAD_COUNT, query_count, "FP4 indexer query rows")?,
                crate::deepseek_v4_oracle::INDEXER_FP4_VALUES_PER_ROW,
                "FP4 indexer query unit elements",
            )?,
        ),
        (
            "FP4 indexer query scale bytes",
            checked_mul(
                checked_mul(HEAD_COUNT, query_count, "FP4 indexer query rows")?,
                SCALE_BYTES,
                "FP4 indexer query scale bytes",
            )?,
        ),
        (
            "FP4 indexer key value bytes",
            checked_mul(row_capacity, VALUE_BYTES, "FP4 indexer key value bytes")?,
        ),
        (
            "FP4 indexer key scale bytes",
            checked_mul(row_capacity, SCALE_BYTES, "FP4 indexer key scale bytes")?,
        ),
    ] {
        if u32::try_from(elements).is_err() {
            return invalid(format!("{name} exceed u32 shader offsets"));
        }
    }
    Ok(())
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
pub(super) fn validate_fp4_matrix_score_geometry(
    kernel: &str,
    thread_execution_width: usize,
    max_threads_per_group: usize,
    max_threadgroup_bytes: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const THREADS: usize = 256;
    const KEY_BYTES: usize = 8 * 32 * std::mem::size_of::<u16>();
    const DOT_BYTES: usize = 64 * 8 * std::mem::size_of::<f32>();
    const THREADGROUP_BYTES: usize = KEY_BYTES + DOT_BYTES;
    if thread_execution_width != 32 || max_threads_per_group < THREADS {
        return invalid(format!(
            "{kernel} requires SIMD width 32 and {THREADS} threads, got width {thread_execution_width} max {max_threads_per_group}"
        ));
    }
    if max_threadgroup_bytes < THREADGROUP_BYTES {
        return invalid(format!(
            "{kernel} requires {THREADGROUP_BYTES} threadgroup bytes, device allows {max_threadgroup_bytes}"
        ));
    }
    Ok(())
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
pub(super) fn validate_fp4_pack_geometry(
    thread_execution_width: usize,
    max_threads_per_group: usize,
    max_threadgroup_bytes: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const THREADS: usize = 128;
    const THREADGROUP_BYTES: usize = 128 * std::mem::size_of::<f32>()
        + 64 * std::mem::size_of::<u8>()
        + 4 * std::mem::size_of::<u8>()
        + 4 * std::mem::size_of::<u32>();
    if thread_execution_width != 32 || max_threads_per_group < THREADS {
        return invalid(format!(
            "FP4 row pack requires SIMD width 32 and {THREADS} threads, got width {thread_execution_width} max {max_threads_per_group}"
        ));
    }
    if max_threadgroup_bytes < THREADGROUP_BYTES {
        return invalid(format!(
            "FP4 row pack requires {THREADGROUP_BYTES} threadgroup bytes, device allows {max_threadgroup_bytes}"
        ));
    }
    Ok(())
}

pub(super) fn validate_lightning_indexer_score_offsets(
    head_count: usize,
    head_dim: usize,
    row_capacity: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let query_width = checked_mul(head_count, head_dim, "indexer score query width")?;
    let products = [
        (
            "indexer score elements",
            checked_mul(row_capacity, query_count, "indexer score elements")?,
        ),
        (
            "indexer score weight elements",
            checked_mul(head_count, query_count, "indexer score weight elements")?,
        ),
        (
            "indexer score query elements",
            checked_mul(query_width, query_count, "indexer score query elements")?,
        ),
        (
            "indexer score key elements",
            checked_mul(row_capacity, head_dim, "indexer score key elements")?,
        ),
    ];
    for (name, elements) in products {
        if u32::try_from(elements).is_err() {
            return invalid(format!("{name} exceed u32 shader offsets"));
        }
    }
    Ok(())
}

pub(super) fn validate_cooperative_lightning_score_geometry(
    kernel: &str,
    thread_execution_width: usize,
    max_threads_per_group: usize,
    max_threadgroup_bytes: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const THREADS: usize = 256;
    const KEY_BYTES: usize = 8 * 128 * std::mem::size_of::<u16>();
    const DOT_BYTES: usize = 8 * 64 * std::mem::size_of::<f32>();
    const THREADGROUP_BYTES: usize = KEY_BYTES + DOT_BYTES;

    if thread_execution_width != 32 || max_threads_per_group < THREADS {
        return invalid(format!(
            "{kernel} requires SIMD width 32 and {THREADS} threads, got width {thread_execution_width} max {max_threads_per_group}"
        ));
    }
    if max_threadgroup_bytes < THREADGROUP_BYTES {
        return invalid(format!(
            "{kernel} requires {THREADGROUP_BYTES} threadgroup bytes, device allows {max_threadgroup_bytes}"
        ));
    }
    Ok(())
}

pub(super) fn validate_tiled_f32_lightning_score_geometry(
    kernel: &str,
    thread_execution_width: usize,
    max_threads_per_group: usize,
    max_threadgroup_bytes: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const THREADS: usize = 128;
    const THREADGROUP_BYTES: usize = 5_376 * std::mem::size_of::<f32>();
    if thread_execution_width != 32 || max_threads_per_group < THREADS {
        return invalid(format!(
            "{kernel} requires SIMD width 32 and {THREADS} threads, got width {thread_execution_width} max {max_threads_per_group}"
        ));
    }
    if max_threadgroup_bytes < THREADGROUP_BYTES {
        return invalid(format!(
            "{kernel} requires {THREADGROUP_BYTES} threadgroup bytes, device allows {max_threadgroup_bytes}"
        ));
    }
    Ok(())
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_select_top_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    scores: &MetalTensor,
    visible_counts: &MetalTensor,
    selected_mask: &MetalTensor,
    ranked_ids: Option<&MetalTensor>,
    cache_order_ids: &MetalTensor,
    selected_counts: &MetalTensor,
    status: &MetalTensor,
    row_capacity: usize,
    max_visible_rows: usize,
    top_k: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    encode_select_top_k_f32_with_policy(
        ctx,
        enc,
        scores,
        visible_counts,
        selected_mask,
        ranked_ids,
        cache_order_ids,
        selected_counts,
        status,
        row_capacity,
        max_visible_rows,
        top_k,
        query_count,
        DeepSeekV4SelectorDispatchPolicy::Production,
        true,
    )
}

#[cfg(test)]
pub(super) const DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_GREATER: usize = 0;

#[cfg(test)]
pub(super) const DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_EQUAL: usize = 1;

#[cfg(test)]
pub(super) const DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_TIE_QUOTA: usize = 2;

#[cfg(test)]
pub(super) const DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_SELECTED: usize = 3;

#[cfg(test)]
pub(super) const DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_ID_OFFSET: usize = 4;

#[cfg(test)]
pub(super) const DEEPSEEK_V4_MULTIGROUP_SELECTOR_COMPACT_PHASE: i32 = 8;

#[cfg(test)]
pub(super) fn deepseek_v4_multigroup_selector_record_completion(
    generation: u32,
    digit: usize,
    group: usize,
) -> u32 {
    0xd541_0000 ^ generation ^ ((digit as u32) << 12) ^ group as u32
}

#[cfg(test)]
pub(super) fn deepseek_v4_multigroup_selector_state_completion(
    generation: u32,
    digit: usize,
) -> u32 {
    0xd542_0000 ^ generation ^ ((digit as u32) << 12)
}

#[cfg(test)]
pub(super) fn deepseek_v4_multigroup_selector_compact_completion(
    generation: u32,
    group: usize,
) -> u32 {
    0xd543_0000 ^ generation ^ group as u32
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_select_top_k_multigroup_threshold_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    scores: &MetalTensor,
    visible_counts: &MetalTensor,
    records: &MetalTensor,
    partition_plan: &MetalTensor,
    state: &MetalTensor,
    row_capacity: usize,
    top_k: usize,
    group_count: usize,
    generation: u32,
    fault_digit: Option<usize>,
) -> Result<(), DeepSeekV4MetalError> {
    const THREADGROUP_WIDTH: usize = 256;
    const SIMD_GROUPS: usize = THREADGROUP_WIDTH / 32;
    const HISTOGRAM_WORDS: usize = SIMD_GROUPS * 16;
    const ERROR_WORDS: usize = SIMD_GROUPS;
    require_serial(enc, "deepseek_v4_multigroup_selector_threshold")?;
    if row_capacity == 0
        || top_k == 0
        || top_k > row_capacity
        || !(2..=THREADGROUP_WIDTH).contains(&group_count)
        || generation == 0
        || fault_digit.is_some_and(|digit| digit >= DEEPSEEK_V4_MULTIGROUP_SELECTOR_DIGITS)
        || [row_capacity, top_k, group_count]
            .into_iter()
            .any(|value| u32::try_from(value).is_err())
    {
        return invalid("multi-group selector threshold geometry is invalid");
    }
    let score_elements = checked_mul(row_capacity, 1, "multi-group selector score elements")?;
    let record_elements = checked_mul(
        DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS,
        group_count,
        "multi-group selector record elements",
    )?;
    let partition_elements = checked_mul(
        DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS,
        group_count,
        "multi-group selector partition plan",
    )?;
    if [score_elements, record_elements, partition_elements]
        .into_iter()
        .any(|value| u32::try_from(value).is_err())
    {
        return invalid("multi-group selector threshold offsets exceed u32");
    }
    validate_f32(
        scores,
        &[row_capacity as u64, 1],
        false,
        "multi-group selector scores",
    )?;
    validate_i32(
        visible_counts,
        &[1],
        false,
        "multi-group selector visibility",
    )?;
    validate_i32(
        records,
        &[
            DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS as u64,
            group_count as u64,
        ],
        true,
        "multi-group selector records",
    )?;
    validate_i32(
        partition_plan,
        &[
            DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS as u64,
            group_count as u64,
        ],
        true,
        "multi-group selector partition plan",
    )?;
    validate_i32(
        state,
        &[DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS as u64],
        true,
        "multi-group selector state",
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_capacity: u32,
        top_k: u32,
        group_count: u32,
        generation: u32,
        digit: u32,
        shift: u32,
    }

    let producer = ctx.pipeline("kernel_deepseek_v4_select_top_k_multigroup_histogram_f32")?;
    validate_parallel_selector_pipeline(
        producer.threadExecutionWidth(),
        producer.maxTotalThreadsPerThreadgroup(),
    )?;
    let reducer = ctx.pipeline("kernel_deepseek_v4_select_top_k_multigroup_reduce_f32")?;
    for digit in 0..DEEPSEEK_V4_MULTIGROUP_SELECTOR_DIGITS {
        let args = Args {
            row_capacity: row_capacity as u32,
            top_k: top_k as u32,
            group_count: group_count as u32,
            generation,
            digit: digit as u32,
            shift: 28 - digit as u32 * 4,
        };
        enc.set_pipeline(&producer);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, scores);
        enc.set_tensor(2, visible_counts);
        enc.set_tensor(3, state);
        enc.set_tensor(4, records);
        enc.set_threadgroup_memory(
            0,
            (HISTOGRAM_WORDS + ERROR_WORDS) * std::mem::size_of::<u32>(),
        );
        let producer_groups = if fault_digit == Some(digit) {
            group_count - 1
        } else {
            group_count
        };
        enc.dispatch(
            MTLSize {
                width: producer_groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: THREADGROUP_WIDTH,
                height: 1,
                depth: 1,
            },
        );

        enc.set_pipeline(&reducer);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, visible_counts);
        enc.set_tensor(2, records);
        enc.set_tensor(3, partition_plan);
        enc.set_tensor(4, state);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_select_top_k_multigroup_full_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    scores: &MetalTensor,
    visible_counts: &MetalTensor,
    records: &MetalTensor,
    partition_plan: &MetalTensor,
    state: &MetalTensor,
    private_mask: &MetalTensor,
    private_ids: &MetalTensor,
    selected_mask: &MetalTensor,
    cache_order_ids: &MetalTensor,
    selected_counts: &MetalTensor,
    status: &MetalTensor,
    row_capacity: usize,
    top_k: usize,
    generation: u32,
    fault_digit: Option<usize>,
    omit_last_compactor: bool,
) -> Result<(), DeepSeekV4MetalError> {
    const THREADGROUP_WIDTH: usize = 256;
    const COMPACT_SCRATCH_WORDS: usize = 3 * THREADGROUP_WIDTH + 20;
    const PUBLISH_SCRATCH_WORDS: usize = 20;
    let group_count = DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS;
    require_serial(enc, "deepseek_v4_multigroup_selector_full")?;
    validate_i8(
        private_mask,
        &[row_capacity as u64, 1],
        true,
        "multi-group selector private mask",
    )?;
    for (tensor, shape, name) in [
        (
            private_ids,
            vec![top_k as u64, 1],
            "multi-group selector private IDs",
        ),
        (
            selected_mask,
            vec![row_capacity as u64, 1],
            "multi-group selector published mask",
        ),
        (
            cache_order_ids,
            vec![top_k as u64, 1],
            "multi-group selector published IDs",
        ),
        (
            selected_counts,
            vec![1],
            "multi-group selector published count",
        ),
        (status, vec![1], "multi-group selector published status"),
    ] {
        validate_i32(tensor, &shape, true, name)?;
    }
    let compactor = ctx.pipeline("kernel_deepseek_v4_select_top_k_multigroup_compact_f32")?;
    validate_parallel_selector_pipeline(
        compactor.threadExecutionWidth(),
        compactor.maxTotalThreadsPerThreadgroup(),
    )?;
    let publisher = ctx.pipeline("kernel_deepseek_v4_select_top_k_multigroup_publish_f32")?;
    validate_parallel_selector_pipeline(
        publisher.threadExecutionWidth(),
        publisher.maxTotalThreadsPerThreadgroup(),
    )?;

    encode_select_top_k_multigroup_threshold_f32(
        ctx,
        enc,
        scores,
        visible_counts,
        records,
        partition_plan,
        state,
        row_capacity,
        top_k,
        group_count,
        generation,
        fault_digit,
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_capacity: u32,
        top_k: u32,
        group_count: u32,
        generation: u32,
        digit: u32,
        shift: u32,
    }
    let args = Args {
        row_capacity: row_capacity as u32,
        top_k: top_k as u32,
        group_count: group_count as u32,
        generation,
        digit: DEEPSEEK_V4_MULTIGROUP_SELECTOR_DIGITS as u32,
        shift: 0,
    };
    enc.set_pipeline(&compactor);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, scores);
    enc.set_tensor(2, visible_counts);
    enc.set_tensor(3, state);
    enc.set_tensor(4, partition_plan);
    enc.set_tensor(5, private_mask);
    enc.set_tensor(6, private_ids);
    enc.set_tensor(7, records);
    enc.set_threadgroup_memory(0, COMPACT_SCRATCH_WORDS * std::mem::size_of::<u32>());
    enc.dispatch(
        MTLSize {
            width: group_count - usize::from(omit_last_compactor),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        },
    );

    enc.set_pipeline(&publisher);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, visible_counts);
    enc.set_tensor(2, state);
    enc.set_tensor(3, partition_plan);
    enc.set_tensor(4, records);
    enc.set_tensor(5, private_mask);
    enc.set_tensor(6, private_ids);
    enc.set_tensor(7, selected_mask);
    enc.set_tensor(8, cache_order_ids);
    enc.set_tensor(9, selected_counts);
    enc.set_tensor(10, status);
    enc.set_threadgroup_memory(0, PUBLISH_SCRATCH_WORDS * std::mem::size_of::<u32>());
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_select_top_k_multigroup_publish_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    visible_counts: &MetalTensor,
    records: &MetalTensor,
    partition_plan: &MetalTensor,
    state: &MetalTensor,
    private_mask: &MetalTensor,
    private_ids: &MetalTensor,
    selected_mask: &MetalTensor,
    cache_order_ids: &MetalTensor,
    selected_counts: &MetalTensor,
    status: &MetalTensor,
    row_capacity: usize,
    top_k: usize,
    generation: u32,
) -> Result<(), DeepSeekV4MetalError> {
    const THREADGROUP_WIDTH: usize = 256;
    const PUBLISH_SCRATCH_WORDS: usize = 20;
    let group_count = DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS;
    require_serial(enc, "deepseek_v4_multigroup_selector_publish_only")?;
    if row_capacity == 0
        || top_k == 0
        || top_k > row_capacity
        || generation == 0
        || [row_capacity, top_k]
            .into_iter()
            .any(|value| u32::try_from(value).is_err())
    {
        return invalid("multi-group selector publisher geometry is invalid");
    }
    validate_i32(
        visible_counts,
        &[1],
        false,
        "multi-group selector publisher visibility",
    )?;
    validate_i32(
        records,
        &[
            DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS as u64,
            group_count as u64,
        ],
        false,
        "multi-group selector publisher records",
    )?;
    validate_i32(
        partition_plan,
        &[
            DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS as u64,
            group_count as u64,
        ],
        false,
        "multi-group selector publisher plan",
    )?;
    validate_i32(
        state,
        &[DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS as u64],
        false,
        "multi-group selector publisher state",
    )?;
    validate_i8(
        private_mask,
        &[row_capacity as u64, 1],
        false,
        "multi-group selector publisher private mask",
    )?;
    for (tensor, shape, writable, name) in [
        (
            private_ids,
            vec![top_k as u64, 1],
            false,
            "multi-group selector publisher private IDs",
        ),
        (
            selected_mask,
            vec![row_capacity as u64, 1],
            true,
            "multi-group selector publisher mask",
        ),
        (
            cache_order_ids,
            vec![top_k as u64, 1],
            true,
            "multi-group selector publisher IDs",
        ),
        (
            selected_counts,
            vec![1],
            true,
            "multi-group selector publisher count",
        ),
        (
            status,
            vec![1],
            true,
            "multi-group selector publisher status",
        ),
    ] {
        validate_i32(tensor, &shape, writable, name)?;
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_capacity: u32,
        top_k: u32,
        group_count: u32,
        generation: u32,
        digit: u32,
        shift: u32,
    }
    let publisher = ctx.pipeline("kernel_deepseek_v4_select_top_k_multigroup_publish_f32")?;
    validate_parallel_selector_pipeline(
        publisher.threadExecutionWidth(),
        publisher.maxTotalThreadsPerThreadgroup(),
    )?;
    enc.set_pipeline(&publisher);
    enc.set_bytes(
        0,
        &Args {
            row_capacity: row_capacity as u32,
            top_k: top_k as u32,
            group_count: group_count as u32,
            generation,
            digit: DEEPSEEK_V4_MULTIGROUP_SELECTOR_DIGITS as u32,
            shift: 0,
        },
    );
    enc.set_tensor(1, visible_counts);
    enc.set_tensor(2, state);
    enc.set_tensor(3, partition_plan);
    enc.set_tensor(4, records);
    enc.set_tensor(5, private_mask);
    enc.set_tensor(6, private_ids);
    enc.set_tensor(7, selected_mask);
    enc.set_tensor(8, cache_order_ids);
    enc.set_tensor(9, selected_counts);
    enc.set_tensor(10, status);
    enc.set_threadgroup_memory(0, PUBLISH_SCRATCH_WORDS * std::mem::size_of::<u32>());
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_select_top_k_f32_with_policy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    scores: &MetalTensor,
    visible_counts: &MetalTensor,
    selected_mask: &MetalTensor,
    ranked_ids: Option<&MetalTensor>,
    cache_order_ids: &MetalTensor,
    selected_counts: &MetalTensor,
    status: &MetalTensor,
    row_capacity: usize,
    max_visible_rows: usize,
    top_k: usize,
    query_count: usize,
    dispatch_policy: DeepSeekV4SelectorDispatchPolicy,
    radix4: bool,
) -> Result<(), DeepSeekV4MetalError> {
    if row_capacity == 0
        || max_visible_rows == 0
        || max_visible_rows > row_capacity
        || top_k == 0
        || top_k > row_capacity
        || query_count == 0
        || [row_capacity, top_k, query_count]
            .into_iter()
            .any(|value| u32::try_from(value).is_err())
    {
        return invalid("indexer selection geometry is invalid");
    }
    let score_elements = checked_mul(
        row_capacity,
        query_count,
        "indexer selection score elements",
    )?;
    let id_elements = checked_mul(top_k, query_count, "indexer selection ID elements")?;
    if u32::try_from(score_elements).is_err() || u32::try_from(id_elements).is_err() {
        return invalid("indexer selection buffer offsets exceed u32");
    }
    validate_f32(
        scores,
        &[row_capacity as u64, query_count as u64],
        false,
        "indexer selection scores",
    )?;
    validate_i32(
        visible_counts,
        &[query_count as u64],
        false,
        "indexer selection visible counts",
    )?;
    validate_i32(
        selected_mask,
        &[row_capacity as u64, query_count as u64],
        true,
        "indexer selection mask",
    )?;
    if let Some(ranked_ids) = ranked_ids {
        validate_i32(
            ranked_ids,
            &[top_k as u64, query_count as u64],
            true,
            "ranked indexer IDs",
        )?;
    }
    validate_i32(
        cache_order_ids,
        &[top_k as u64, query_count as u64],
        true,
        "cache-order indexer IDs",
    )?;
    validate_i32(
        selected_counts,
        &[query_count as u64],
        true,
        "indexer selected counts",
    )?;
    validate_i32(
        status,
        &[query_count as u64],
        true,
        "indexer selection status",
    )?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_capacity: u32,
        top_k: u32,
        query_count: u32,
        emit_ranked: u32,
    }
    let emit_ranked = ranked_ids.is_some();
    let ranked_ids = ranked_ids.unwrap_or(cache_order_ids);
    let parallel = use_parallel_selector(dispatch_policy, max_visible_rows, top_k);
    let radix4 = radix4 && parallel;
    let pso = ctx.pipeline(if radix4 {
        "kernel_deepseek_v4_select_top_k_radix4_f32"
    } else if parallel {
        "kernel_deepseek_v4_select_top_k_parallel_f32"
    } else {
        "kernel_deepseek_v4_select_top_k_f32"
    })?;
    if parallel {
        validate_parallel_selector_pipeline(
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup(),
        )?;
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            row_capacity: row_capacity as u32,
            top_k: top_k as u32,
            query_count: query_count as u32,
            emit_ranked: u32::from(emit_ranked),
        },
    );
    enc.set_tensor(1, scores);
    enc.set_tensor(2, visible_counts);
    enc.set_tensor(3, selected_mask);
    enc.set_tensor(4, ranked_ids);
    enc.set_tensor(5, cache_order_ids);
    enc.set_tensor(6, selected_counts);
    enc.set_tensor(7, status);
    if parallel {
        const THREADGROUP_WIDTH: usize = 256;
        enc.set_threadgroup_memory(0, THREADGROUP_WIDTH * std::mem::size_of::<u32>());
        enc.set_threadgroup_memory(1, THREADGROUP_WIDTH * std::mem::size_of::<u32>());
        enc.dispatch(
            MTLSize {
                width: query_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: THREADGROUP_WIDTH,
                height: 1,
                depth: 1,
            },
        );
    } else {
        enc.dispatch(
            MTLSize {
                width: query_count.div_ceil(64),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(feature = "dsv4-diagnostics", allow(dead_code))]
pub(super) fn encode_select_top_k_radix4_ids_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    scores: &MetalTensor,
    visible_counts: &MetalTensor,
    cache_order_ids: &MetalTensor,
    selected_counts: &MetalTensor,
    status: &MetalTensor,
    row_capacity: usize,
    max_visible_rows: usize,
    top_k: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if row_capacity == 0
        || max_visible_rows <= top_k
        || max_visible_rows > row_capacity
        || top_k == 0
        || top_k > row_capacity
        || query_count == 0
        || [row_capacity, top_k, query_count]
            .into_iter()
            .any(|value| u32::try_from(value).is_err())
    {
        return invalid("maskless radix4 selector requires parallel selection geometry");
    }
    let score_elements = checked_mul(
        row_capacity,
        query_count,
        "maskless radix4 selector score elements",
    )?;
    let id_elements = checked_mul(top_k, query_count, "maskless radix4 selector ID elements")?;
    if u32::try_from(score_elements).is_err() || u32::try_from(id_elements).is_err() {
        return invalid("maskless radix4 selector buffer offsets exceed u32");
    }
    validate_f32(
        scores,
        &[row_capacity as u64, query_count as u64],
        false,
        "maskless radix4 selector scores",
    )?;
    validate_i32(
        visible_counts,
        &[query_count as u64],
        false,
        "maskless radix4 selector visible counts",
    )?;
    validate_i32(
        cache_order_ids,
        &[top_k as u64, query_count as u64],
        true,
        "maskless radix4 selector cache-order IDs",
    )?;
    validate_i32(
        selected_counts,
        &[query_count as u64],
        true,
        "maskless radix4 selector counts",
    )?;
    validate_i32(
        status,
        &[query_count as u64],
        true,
        "maskless radix4 selector status",
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_capacity: u32,
        top_k: u32,
        query_count: u32,
        emit_ranked: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_select_top_k_radix4_ids_f32")?;
    validate_parallel_selector_pipeline(
        pso.threadExecutionWidth(),
        pso.maxTotalThreadsPerThreadgroup(),
    )?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            row_capacity: row_capacity as u32,
            top_k: top_k as u32,
            query_count: query_count as u32,
            emit_ranked: 0,
        },
    );
    enc.set_tensor(1, scores);
    enc.set_tensor(2, visible_counts);
    enc.set_tensor(3, cache_order_ids);
    enc.set_tensor(4, selected_counts);
    enc.set_tensor(5, status);
    const THREADGROUP_WIDTH: usize = 256;
    enc.set_threadgroup_memory(0, THREADGROUP_WIDTH * std::mem::size_of::<u32>());
    enc.set_threadgroup_memory(1, THREADGROUP_WIDTH * std::mem::size_of::<u32>());
    enc.dispatch(
        MTLSize {
            width: query_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_selected_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    compressed_cache: &MetalTensor,
    selected_ids: &MetalTensor,
    sinks: &MetalTensor,
    output: &MetalTensor,
    position: u32,
    compressed_count: usize,
    selected_slots: usize,
    compressed_capacity: usize,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    let query_width = checked_mul(config.head_count, config.head_dim, "selected query width")?;
    validate_f32(
        queries,
        &[config.head_dim as u64, config.head_count as u64],
        false,
        "selected attention queries",
    )?;
    validate_f16(
        raw_cache,
        &[config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        false,
        "selected attention raw cache",
    )?;
    validate_f16(
        compressed_cache,
        &[config.head_dim as u64, compressed_capacity as u64],
        false,
        "selected attention compressed cache",
    )?;
    validate_i32(
        selected_ids,
        &[selected_slots as u64],
        false,
        "selected attention row IDs",
    )?;
    validate_f32(
        sinks,
        &[config.head_count as u64],
        false,
        "selected attention sinks",
    )?;
    validate_f32(
        output,
        &[config.head_dim as u64, config.head_count as u64],
        true,
        "selected attention output",
    )?;
    if compressed_count == 0
        || compressed_count > compressed_capacity
        || selected_slots == 0
        || selected_slots > compressed_count
        || [compressed_count, selected_slots, compressed_capacity]
            .into_iter()
            .any(|value| u32::try_from(value).is_err())
    {
        return invalid("selected attention compressed geometry is invalid");
    }
    let visible_end = u64::from(position) + 1;
    let raw_count = visible_end.min(DEEPSEEK_V4_LOCAL_WINDOW as u64) as u32;
    let raw_start = u32::try_from(visible_end - u64::from(raw_count)).map_err(|_| {
        DeepSeekV4MetalError::Invalid("selected attention raw start exceeds u32".into())
    })?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        window: u32,
        raw_count: u32,
        raw_start: u32,
        compressed_count: u32,
        selected_slots: u32,
        scale: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_selected_sink_attention_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: config.head_count as u32,
            head_dim: config.head_dim as u32,
            window: DEEPSEEK_V4_LOCAL_WINDOW as u32,
            raw_count,
            raw_start,
            compressed_count: compressed_count as u32,
            selected_slots: selected_slots as u32,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, raw_cache);
    enc.set_tensor(3, compressed_cache);
    enc.set_tensor(4, selected_ids);
    enc.set_tensor(5, sinks);
    enc.set_tensor(6, output);
    enc.dispatch(
        MTLSize {
            width: query_width.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_cooperative_selected_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    raw_cache_layout: DeepSeekV4RawCacheLayout,
    compressed_cache: &MetalTensor,
    compressed_capacity: usize,
    selected_ids: &MetalTensor,
    selected_counts: &MetalTensor,
    visible_counts: &MetalTensor,
    sinks: &MetalTensor,
    output: &MetalTensor,
    chunk_start_position: u32,
    query_token_offset: usize,
    query_count: usize,
    token_count: usize,
    selected_slots: usize,
    online: bool,
    direct_load: bool,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "deepseek_v4_cooperative_selected_attention")?;
    let query_width = checked_mul(
        config.head_count,
        config.head_dim,
        "cooperative selected query width",
    )?;
    let query_end = query_token_offset.checked_add(query_count).ok_or_else(|| {
        DeepSeekV4MetalError::Invalid("cooperative selected query range overflow".into())
    })?;
    if compressed_capacity == 0
        || selected_slots == 0
        || selected_slots > compressed_capacity
        || query_count == 0
        || token_count == 0
        || query_token_offset >= token_count
        || query_end > token_count
        || [
            config.head_count,
            config.head_dim,
            query_width,
            compressed_capacity,
            selected_slots,
            query_token_offset,
            query_count,
            token_count,
        ]
        .into_iter()
        .any(|value| u32::try_from(value).is_err())
    {
        return invalid("cooperative selected attention geometry is invalid");
    }
    let final_token = query_end - 1;
    chunk_start_position
        .checked_add(u32::try_from(final_token).map_err(|_| {
            DeepSeekV4MetalError::Invalid("cooperative selected final token exceeds u32".into())
        })?)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("cooperative selected absolute position overflow".into())
        })?;
    validate_f32(
        queries,
        &[query_width as u64, token_count as u64],
        false,
        "cooperative selected attention queries",
    )?;
    validate_raw_attention_caches(
        raw_cache,
        raw_cache_before_chunk,
        raw_cache_layout,
        config.head_dim,
        token_count,
        "cooperative selected attention",
    )?;
    validate_f16(
        compressed_cache,
        &[config.head_dim as u64, compressed_capacity as u64],
        false,
        "cooperative selected compressed cache",
    )?;
    validate_i32(
        selected_ids,
        &[selected_slots as u64, query_count as u64],
        false,
        "cooperative selected row IDs",
    )?;
    for (tensor, name) in [
        (selected_counts, "cooperative selected row counts"),
        (visible_counts, "cooperative selected visible counts"),
    ] {
        validate_i32(tensor, &[query_count as u64], false, name)?;
    }
    validate_f32(
        sinks,
        &[config.head_count as u64],
        false,
        "cooperative selected attention sinks",
    )?;
    validate_f32(
        output,
        &[query_width as u64, token_count as u64],
        true,
        "cooperative selected attention output",
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        query_count: u32,
        query_token_offset: u32,
        chunk_start_position: u32,
        window: u32,
        selected_slots: u32,
        compressed_capacity: u32,
        raw_cache_is_chunk: u32,
        scale: f32,
    }
    let maximum_rows = DEEPSEEK_V4_LOCAL_WINDOW
        .checked_add(selected_slots)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("cooperative selected maximum row count overflow".into())
        })?;
    if direct_load && !online {
        return invalid("direct row loading requires online selected attention");
    }
    let (kernel, threadgroup_width, threadgroup_bytes) = if online {
        if config.head_count != 64
            || config.head_dim != DEEPSEEK_V4_HCA_TILE_ROWS
            || selected_slots != DEEPSEEK_V4_CSA_TOP_K
        {
            return invalid(
                "online selected attention requires 64 heads x 512 dimensions and top-512 rows",
            );
        }
        (
            if direct_load {
                "kernel_deepseek_v4_online_packed_selected_sink_attention_f16_direct"
            } else {
                "kernel_deepseek_v4_online_packed_selected_sink_attention_f16"
            },
            DEEPSEEK_V4_ONLINE_HCA_THREADS,
            if direct_load {
                0
            } else {
                DEEPSEEK_V4_ONLINE_HCA_THREADGROUP_BYTES
            },
        )
    } else {
        (
            "kernel_deepseek_v4_packed_selected_sink_attention_f16",
            config.head_dim.max(maximum_rows),
            (maximum_rows + 1) * std::mem::size_of::<f32>(),
        )
    };
    let pso = ctx.pipeline(kernel)?;
    if online {
        validate_deepseek_v4_online_hca_launch_geometry(
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup(),
            ctx.device.maxThreadgroupMemoryLength(),
            threadgroup_bytes,
        )?;
    } else if pso.maxTotalThreadsPerThreadgroup() < threadgroup_width {
        return invalid(format!(
            "cooperative selected attention pipeline supports {} threads, requires {threadgroup_width}",
            pso.maxTotalThreadsPerThreadgroup()
        ));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: config.head_count as u32,
            head_dim: config.head_dim as u32,
            query_count: query_count as u32,
            query_token_offset: query_token_offset as u32,
            chunk_start_position,
            window: DEEPSEEK_V4_LOCAL_WINDOW as u32,
            selected_slots: selected_slots as u32,
            compressed_capacity: compressed_capacity as u32,
            raw_cache_is_chunk: raw_cache_layout.is_chunk(),
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, raw_cache);
    enc.set_tensor(3, raw_cache_before_chunk);
    enc.set_tensor(4, compressed_cache);
    enc.set_tensor(5, selected_ids);
    enc.set_tensor(6, selected_counts);
    enc.set_tensor(7, visible_counts);
    enc.set_tensor(8, sinks);
    enc.set_tensor(9, output);
    if threadgroup_bytes != 0 {
        enc.set_threadgroup_memory(0, threadgroup_bytes);
    }
    enc.dispatch(
        MTLSize {
            width: query_count,
            height: config.head_count,
            depth: 1,
        },
        MTLSize {
            width: threadgroup_width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(super) fn append_compressor_frontier_allocations(
    requests: &mut Vec<DeepSeekV4SessionAllocationRequest>,
    prefix: &str,
    ratio: usize,
    head_dim: usize,
    publication: DeepSeekV4CompressorPublication,
    capacity_rows: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if publication == DeepSeekV4CompressorPublication::IndexerHadamard && head_dim != 128 {
        return invalid("indexer publication requires exactly 128 dimensions");
    }
    let (width, _, state_elements) = compressor_frontier_geometry(ratio, head_dim)?;
    for suffix in ["kv_state", "score_state"] {
        push_session_allocation(
            requests,
            format!("{prefix}.{suffix}"),
            state_elements,
            std::mem::size_of::<f32>(),
        )?;
    }
    for suffix in ["projected_kv", "projected_score"] {
        push_session_allocation(
            requests,
            format!("{prefix}.{suffix}"),
            width,
            std::mem::size_of::<f32>(),
        )?;
    }
    for suffix in ["pooled", "normalized"] {
        push_session_allocation(
            requests,
            format!("{prefix}.{suffix}"),
            head_dim,
            std::mem::size_of::<f32>(),
        )?;
    }
    push_session_allocation(
        requests,
        format!("{prefix}.published"),
        checked_mul(
            head_dim,
            capacity_rows,
            "published compressor history elements",
        )?,
        std::mem::size_of::<u16>(),
    )?;
    #[cfg(feature = "dsv4-diagnostics")]
    if publication == DeepSeekV4CompressorPublication::IndexerHadamard {
        push_session_allocation(
            requests,
            format!("{prefix}.fp4_values"),
            checked_mul(
                crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES,
                capacity_rows,
                "indexer FP4 sidecar value bytes",
            )?,
            std::mem::size_of::<u8>(),
        )?;
        push_session_allocation(
            requests,
            format!("{prefix}.fp4_scales"),
            checked_mul(
                crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES,
                capacity_rows,
                "indexer FP4 sidecar scale bytes",
            )?,
            std::mem::size_of::<u8>(),
        )?;
        push_session_allocation(
            requests,
            format!("{prefix}.fp4_status"),
            capacity_rows,
            std::mem::size_of::<i32>(),
        )?;
    }
    Ok(())
}
