//! Dense/sink attention, RoPE, norms, and hyper-connection scratch.

use super::*;

pub const DEEPSEEK_V4_HC_PARAMETER_COUNT: usize = 24;

/// Numerical cache contract used by a native DeepSeek V4 execution path.
///
/// The maintained b10222 oracle leaves llama.cpp's K-cache type at its default
/// F16. The mixed contract remains available to the isolated position-zero
/// attention differential; it must not be silently substituted for an F16
/// continuing session because the second token observes the stored first row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepSeekV4AttentionCacheContract {
    LlamaCppB10222F16,
    MixedFp8NopeBf16RopeOracle,
}

/// Compatibility name retained for the position-zero live differential.
pub type DeepSeekV4PositionZeroForward = DeepSeekV4Session;

pub(super) fn deepseek_v4_session_attention_config() -> DeepSeekV4PositionZeroAttentionConfig {
    DeepSeekV4PositionZeroAttentionConfig {
        hidden_size: DEEPSEEK_V4_HIDDEN_SIZE,
        q_lora_rank: 1_024,
        head_count: 64,
        head_dim: 512,
        rotary_dim: 64,
        group_count: 8,
        output_rank: 1_024,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepSeekV4RopeParameters {
    pub rotary_dim: usize,
    pub theta: f32,
    pub scaling_factor: f32,
    pub original_context_length: u32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

pub(super) fn deepseek_v4_layer_rope(
    config: &DeepSeekV4Config,
    layer: usize,
) -> Result<DeepSeekV4RopeParameters, DeepSeekV4MetalError> {
    let kind = config.attention_kinds.get(layer).copied().ok_or_else(|| {
        DeepSeekV4MetalError::Invalid(format!("RoPE layer {layer} is out of range"))
    })?;
    let compressed = kind != AttentionKind::SlidingWindow;
    Ok(DeepSeekV4RopeParameters {
        rotary_dim: config.rope_dimension_count as usize,
        theta: if compressed {
            config.compress_rope_freq_base
        } else {
            config.rope_freq_base
        },
        scaling_factor: if compressed {
            config.rope_scaling_factor
        } else {
            1.0
        },
        original_context_length: if compressed {
            config.rope_original_context_length
        } else {
            0
        },
        beta_fast: config.rope_yarn_beta_fast,
        beta_slow: config.rope_yarn_beta_slow,
    })
}

/// Dimensions for the native, position-zero shared-KV attention body.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4PositionZeroAttentionConfig {
    pub hidden_size: usize,
    pub q_lora_rank: usize,
    pub head_count: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub group_count: usize,
    pub output_rank: usize,
}

impl DeepSeekV4PositionZeroAttentionConfig {
    pub(super) fn checked(self) -> Result<CheckedAttentionDims, DeepSeekV4MetalError> {
        let values = [
            ("hidden size", self.hidden_size),
            ("Q LoRA rank", self.q_lora_rank),
            ("head count", self.head_count),
            ("head dimension", self.head_dim),
            ("rotary dimension", self.rotary_dim),
            ("group count", self.group_count),
            ("output rank", self.output_rank),
        ];
        for (name, value) in values {
            if value == 0 {
                return invalid(format!("position-zero attention {name} must be nonzero"));
            }
            u32::try_from(value).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!("position-zero attention {name} exceeds u32"))
            })?;
        }
        if !self.head_count.is_multiple_of(self.group_count) {
            return invalid("position-zero attention group count must divide head count");
        }
        if self.rotary_dim > self.head_dim {
            return invalid("position-zero attention rotary dimension exceeds head dimension");
        }
        let nope_dim = self.head_dim - self.rotary_dim;
        if !nope_dim.is_multiple_of(64) {
            return invalid("position-zero attention NoPE dimension must be divisible by 64");
        }
        let query_width = checked_mul(self.head_count, self.head_dim, "query width")?;
        let group_width = query_width / self.group_count;
        let low_rank_width = checked_mul(self.group_count, self.output_rank, "low-rank width")?;
        u32::try_from(query_width)
            .map_err(|_| DeepSeekV4MetalError::Invalid("query width exceeds u32".into()))?;
        u32::try_from(group_width)
            .map_err(|_| DeepSeekV4MetalError::Invalid("group width exceeds u32".into()))?;
        u32::try_from(low_rank_width)
            .map_err(|_| DeepSeekV4MetalError::Invalid("low-rank width exceeds u32".into()))?;
        Ok(CheckedAttentionDims {
            query_width,
            group_width,
            low_rank_width,
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct CheckedAttentionDims {
    pub(super) query_width: usize,
    pub(super) group_width: usize,
    pub(super) low_rank_width: usize,
}

/// Reusable F32 activation storage for one native DS4 position-zero attention
/// body. The intermediate accessors are intended for differential inspection.
pub struct DeepSeekV4PositionZeroAttentionScratch {
    pub(super) config: DeepSeekV4PositionZeroAttentionConfig,
    pub(super) normalized_input: MetalTensor,
    pub(super) q_lora_raw: MetalTensor,
    pub(super) q_lora: MetalTensor,
    pub(super) queries_raw: MetalTensor,
    pub(super) queries: MetalTensor,
    pub(super) kv_raw: MetalTensor,
    pub(super) kv: MetalTensor,
    pub(super) cached_kv: MetalTensor,
    pub(super) attention: MetalTensor,
    pub(super) hca_partial_output: MetalTensor,
    pub(super) hca_partial_ml: MetalTensor,
    pub(super) low_rank: MetalTensor,
    pub(super) output: MetalTensor,
    pub(super) head_norm_ones: MetalTensor,
    pub(super) paired_prepare_capabilities: DeepSeekV4PairedPrepareCapabilities,
    #[cfg(test)]
    pub(super) hca_test_policy: DeepSeekV4HcaTestPolicy,
    #[cfg(test)]
    pub(super) prepare_test_policy: DeepSeekV4PrepareTestPolicy,
}

impl DeepSeekV4PositionZeroAttentionScratch {
    pub fn new(
        ctx: &MetalContext,
        config: DeepSeekV4PositionZeroAttentionConfig,
    ) -> Result<Self, DeepSeekV4MetalError> {
        let dims = config.checked()?;
        let ones = vec![1.0f32; config.head_dim];
        #[cfg(test)]
        let probe_paired_prepare = true;
        #[cfg(not(test))]
        let probe_paired_prepare = deepseek_v4_decode_prepare_paired_enabled()
            && crate::metal::mat_vec_q8_0_lcpp_enabled();
        let paired_prepare_capabilities = if probe_paired_prepare {
            DeepSeekV4PairedPrepareCapabilities::probe(ctx)
        } else {
            DeepSeekV4PairedPrepareCapabilities::default()
        };
        Ok(Self {
            config,
            normalized_input: MetalTensor::zeros_f32(ctx, vec![config.hidden_size as u64])?,
            q_lora_raw: MetalTensor::zeros_f32(ctx, vec![config.q_lora_rank as u64])?,
            q_lora: MetalTensor::zeros_f32(ctx, vec![config.q_lora_rank as u64])?,
            queries_raw: MetalTensor::zeros_f32(ctx, vec![dims.query_width as u64])?,
            queries: MetalTensor::zeros_f32(
                ctx,
                vec![config.head_dim as u64, config.head_count as u64],
            )?,
            kv_raw: MetalTensor::zeros_f32(ctx, vec![config.head_dim as u64])?,
            kv: MetalTensor::zeros_f32(ctx, vec![config.head_dim as u64])?,
            cached_kv: MetalTensor::zeros_f32(ctx, vec![config.head_dim as u64])?,
            attention: MetalTensor::zeros_f32(
                ctx,
                vec![config.head_dim as u64, config.head_count as u64],
            )?,
            hca_partial_output: MetalTensor::zeros_f32(
                ctx,
                vec![
                    config.head_dim as u64,
                    config.head_count as u64,
                    DEEPSEEK_V4_SPLITK_HCA_PARTITIONS as u64,
                ],
            )?,
            hca_partial_ml: MetalTensor::zeros_f32(
                ctx,
                vec![
                    2,
                    config.head_count as u64,
                    DEEPSEEK_V4_SPLITK_HCA_PARTITIONS as u64,
                ],
            )?,
            low_rank: MetalTensor::zeros_f32(ctx, vec![dims.low_rank_width as u64])?,
            output: MetalTensor::zeros_f32(ctx, vec![config.hidden_size as u64])?,
            head_norm_ones: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&ones),
                vec![config.head_dim as u64],
                GgmlType::F32,
            )?,
            paired_prepare_capabilities,
            #[cfg(test)]
            hca_test_policy: DeepSeekV4HcaTestPolicy::Production,
            #[cfg(test)]
            prepare_test_policy: DeepSeekV4PrepareTestPolicy::Production,
        })
    }

    pub fn config(&self) -> DeepSeekV4PositionZeroAttentionConfig {
        self.config
    }

    pub fn normalized_input(&self) -> &MetalTensor {
        &self.normalized_input
    }

    pub fn q_lora_raw(&self) -> &MetalTensor {
        &self.q_lora_raw
    }

    pub fn q_lora(&self) -> &MetalTensor {
        &self.q_lora
    }

    pub fn queries(&self) -> &MetalTensor {
        &self.queries
    }

    pub fn kv_raw(&self) -> &MetalTensor {
        &self.kv_raw
    }

    pub fn kv(&self) -> &MetalTensor {
        &self.kv
    }

    pub fn cached_kv(&self) -> &MetalTensor {
        &self.cached_kv
    }

    pub fn attention_heads(&self) -> &MetalTensor {
        &self.attention
    }

    pub fn low_rank(&self) -> &MetalTensor {
        &self.low_rank
    }

    pub fn output(&self) -> &MetalTensor {
        &self.output
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn set_hca_test_policy(&mut self, policy: DeepSeekV4HcaTestPolicy) {
        self.hca_test_policy = policy;
    }

    #[cfg(test)]
    pub(super) fn set_prepare_test_policy(&mut self, policy: DeepSeekV4PrepareTestPolicy) {
        self.prepare_test_policy = policy;
    }

    pub(super) fn use_paired_prepare(&self, q_a: &MetalTensor, kv_weight: &MetalTensor) -> bool {
        #[cfg(test)]
        if self.prepare_test_policy == DeepSeekV4PrepareTestPolicy::Composed {
            return false;
        }
        #[cfg(not(test))]
        if !deepseek_v4_decode_prepare_paired_enabled() {
            return false;
        }
        #[cfg(test)]
        if self.prepare_test_policy == DeepSeekV4PrepareTestPolicy::Production
            && !deepseek_v4_decode_prepare_paired_enabled()
        {
            return false;
        }
        kv_weight.dtype == GgmlType::Q8_0
            && matches!(q_a.dtype, GgmlType::Q6_K | GgmlType::Q8_0)
            && crate::metal::mat_vec_q8_0_lcpp_enabled()
            && self.paired_prepare_capabilities.supports(q_a.dtype)
    }

    pub(super) fn use_online_hca(&self) -> bool {
        #[cfg(test)]
        {
            self.hca_test_policy != DeepSeekV4HcaTestPolicy::LegacyTiled
        }
        #[cfg(not(test))]
        {
            true
        }
    }

    pub(super) fn use_splitk_hca(&self, ctx: &MetalContext) -> bool {
        #[cfg(test)]
        {
            let _ = ctx;
            self.hca_test_policy == DeepSeekV4HcaTestPolicy::Production
        }
        #[cfg(not(test))]
        {
            ctx.device.name().to_string() == DEEPSEEK_V4_LONG_HCA_QUALIFIED_DEVICE
        }
    }

    pub(super) fn use_grouped_long_hca(&self, ctx: &MetalContext) -> bool {
        #[cfg(test)]
        {
            let _ = ctx;
            true
        }
        #[cfg(not(test))]
        {
            ctx.device.name().to_string() == DEEPSEEK_V4_LONG_HCA_QUALIFIED_DEVICE
        }
    }

    /// Encode exactly the position-zero DS4 attention body. Forward and inverse
    /// RoPE are omitted because both are identity at position zero.
    #[allow(clippy::too_many_arguments)]
    pub fn encode<'a>(
        &'a self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        input: &MetalTensor,
        attention_norm: &MetalTensor,
        q_a: &MetalTensor,
        q_a_norm: &MetalTensor,
        q_b: &MetalTensor,
        kv_weight: &MetalTensor,
        kv_norm: &MetalTensor,
        sinks: &MetalTensor,
        output_a: &MetalTensor,
        output_b: &MetalTensor,
        rms_eps: f32,
    ) -> Result<&'a MetalTensor, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_position_zero_attention")?;
        validate_eps(rms_eps, "RMSNorm epsilon")?;
        let c = self.config;
        let dims = c.checked()?;
        self.validate_scratch(dims)?;

        validate_f32(input, &[c.hidden_size as u64], false, "attention input")?;
        validate_f32(
            attention_norm,
            &[c.hidden_size as u64],
            false,
            "attention norm weight",
        )?;
        validate_matvec_weight(q_a, c.hidden_size, c.q_lora_rank, "Q A weight")?;
        validate_f32(q_a_norm, &[c.q_lora_rank as u64], false, "Q A norm weight")?;
        validate_matvec_weight(q_b, c.q_lora_rank, dims.query_width, "Q B weight")?;
        validate_matvec_weight(kv_weight, c.hidden_size, c.head_dim, "KV weight")?;
        validate_f32(kv_norm, &[c.head_dim as u64], false, "KV norm weight")?;
        validate_f32(sinks, &[c.head_count as u64], false, "attention sinks")?;

        encode_rms_norm_mul_f32(
            ctx,
            enc,
            input,
            attention_norm,
            &self.normalized_input,
            rms_eps,
        )?;
        encode_projection(
            ctx,
            enc,
            q_a,
            &self.normalized_input,
            &self.q_lora_raw,
            c.hidden_size,
            c.q_lora_rank,
            "Q A",
        )?;
        encode_rms_norm_mul_f32(ctx, enc, &self.q_lora_raw, q_a_norm, &self.q_lora, rms_eps)?;
        encode_projection(
            ctx,
            enc,
            q_b,
            &self.q_lora,
            &self.queries_raw,
            c.q_lora_rank,
            dims.query_width,
            "Q B",
        )?;
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            &self.queries_raw,
            &self.head_norm_ones,
            &self.queries,
            c.head_count,
            c.head_dim,
            rms_eps,
        )?;
        encode_projection(
            ctx,
            enc,
            kv_weight,
            &self.normalized_input,
            &self.kv_raw,
            c.hidden_size,
            c.head_dim,
            "KV",
        )?;
        encode_rms_norm_mul_f32(ctx, enc, &self.kv_raw, kv_norm, &self.kv, rms_eps)?;
        encode_attention_cache_roundtrip(ctx, enc, &self.kv, &self.cached_kv, c)?;
        encode_position_zero_sink_attention(
            ctx,
            enc,
            &self.queries,
            &self.cached_kv,
            sinks,
            &self.attention,
            c,
        )?;

        for group in 0..c.group_count {
            let input_view = self.attention.view_subrange(
                (group * dims.group_width) as u64,
                vec![dims.group_width as u64],
            );
            let output_view = self
                .low_rank
                .view_subrange((group * c.output_rank) as u64, vec![c.output_rank as u64]);
            let weight_view = group_weight_view(output_a, dims.group_width, c.output_rank, group)?;
            encode_projection(
                ctx,
                enc,
                &weight_view,
                &input_view,
                &output_view,
                dims.group_width,
                c.output_rank,
                "grouped output A",
            )?;
        }
        encode_projection(
            ctx,
            enc,
            output_b,
            &self.low_rank,
            &self.output,
            dims.low_rank_width,
            c.hidden_size,
            "output B",
        )?;
        Ok(&self.output)
    }

    /// Project and rotate Q/shared-KV, then publish the current raw F16 row.
    /// Compressor publication is ordered between this preparation and
    /// `encode_finish_dense_f16` so a boundary token can see its own row.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_prepare_local_f16(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        input: &MetalTensor,
        attention_norm: &MetalTensor,
        q_a: &MetalTensor,
        q_a_norm: &MetalTensor,
        q_b: &MetalTensor,
        kv_weight: &MetalTensor,
        kv_norm: &MetalTensor,
        raw_cache: &MetalTensor,
        position: u32,
        rope: DeepSeekV4RopeParameters,
        rms_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_attention_prepare")?;
        validate_eps(rms_eps, "RMSNorm epsilon")?;
        let c = self.config;
        let dims = c.checked()?;
        self.validate_scratch(dims)?;
        validate_ds4_rope(rope, c.head_dim, c.rotary_dim)?;

        validate_f32(input, &[c.hidden_size as u64], false, "attention input")?;
        validate_f32(
            attention_norm,
            &[c.hidden_size as u64],
            false,
            "attention norm weight",
        )?;
        validate_matvec_weight(q_a, c.hidden_size, c.q_lora_rank, "Q A weight")?;
        validate_f32(q_a_norm, &[c.q_lora_rank as u64], false, "Q A norm weight")?;
        validate_matvec_weight(q_b, c.q_lora_rank, dims.query_width, "Q B weight")?;
        validate_matvec_weight(kv_weight, c.hidden_size, c.head_dim, "KV weight")?;
        validate_f32(kv_norm, &[c.head_dim as u64], false, "KV norm weight")?;
        validate_f16(
            raw_cache,
            &[c.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            true,
            "local raw cache",
        )?;

        encode_rms_norm_mul_f32(
            ctx,
            enc,
            input,
            attention_norm,
            &self.normalized_input,
            rms_eps,
        )?;
        let paired = self.use_paired_prepare(q_a, kv_weight);
        if paired {
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| {
                eprintln!(
                    "deepseek_v4: decode Q/KV projections, norms, and RoPE run as paired dispatches; rollback=QWEN_DSV4_DECODE_PREPARE_PAIRED=0"
                );
            });
            encode_ds4_prepare_projection_pair(
                ctx,
                enc,
                q_a,
                kv_weight,
                &self.normalized_input,
                &self.q_lora_raw,
                &self.kv_raw,
                c.hidden_size,
                c.q_lora_rank,
                c.head_dim,
            )?;
            encode_ds4_prepare_norm_pair(
                ctx,
                enc,
                &self.q_lora_raw,
                q_a_norm,
                &self.q_lora,
                &self.kv_raw,
                kv_norm,
                &self.kv,
                rms_eps,
            )?;
        } else {
            encode_projection(
                ctx,
                enc,
                q_a,
                &self.normalized_input,
                &self.q_lora_raw,
                c.hidden_size,
                c.q_lora_rank,
                "Q A",
            )?;
            encode_rms_norm_mul_f32(ctx, enc, &self.q_lora_raw, q_a_norm, &self.q_lora, rms_eps)?;
        }
        encode_projection(
            ctx,
            enc,
            q_b,
            &self.q_lora,
            &self.queries_raw,
            c.q_lora_rank,
            dims.query_width,
            "Q B",
        )?;
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            &self.queries_raw,
            &self.head_norm_ones,
            &self.queries,
            c.head_count,
            c.head_dim,
            rms_eps,
        )?;
        if paired {
            encode_ds4_rope_pair_in_place(ctx, enc, &self.queries, &self.kv, position, rope)?;
        } else {
            encode_projection(
                ctx,
                enc,
                kv_weight,
                &self.normalized_input,
                &self.kv_raw,
                c.hidden_size,
                c.head_dim,
                "KV",
            )?;
            encode_rms_norm_mul_f32(ctx, enc, &self.kv_raw, kv_norm, &self.kv, rms_eps)?;
            encode_ds4_rope_tail_adjacent_in_place(ctx, enc, &self.queries, position, rope, false)?;
            encode_ds4_rope_tail_adjacent_in_place(ctx, enc, &self.kv, position, rope, false)?;
        }
        let cache_slot = position as usize % DEEPSEEK_V4_LOCAL_WINDOW;
        encode_scatter_offset_f32_to_f16(
            ctx,
            enc,
            &self.kv,
            raw_cache,
            cache_slot * c.head_dim,
            c.head_dim,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_dense_attention_f16(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        raw_cache: &MetalTensor,
        compressed: Option<DeepSeekV4PublishedRows<'_>>,
        kind: AttentionKind,
        sinks: &MetalTensor,
        position: u32,
        rope: DeepSeekV4RopeParameters,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_attention_finish")?;
        let c = self.config;
        let dims = c.checked()?;
        self.validate_scratch(dims)?;
        validate_ds4_rope(rope, c.head_dim, c.rotary_dim)?;
        validate_f16(
            raw_cache,
            &[c.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            false,
            "local raw cache",
        )?;
        validate_f32(sinks, &[c.head_count as u64], false, "attention sinks")?;
        if let Some(rows) = compressed {
            validate_f16(
                rows.cache,
                &[c.head_dim as u64, rows.capacity_rows as u64],
                false,
                "compressed attention cache",
            )?;
            if rows.count == 0 || rows.count > rows.capacity_rows {
                return invalid(format!(
                    "compressed attention row count {} is out of range",
                    rows.count
                ));
            }
        }

        if let Some(rows) = compressed.filter(|rows| rows.count > DEEPSEEK_V4_HCA_TILE_ROWS) {
            if kind != AttentionKind::HeavilyCompressed {
                return invalid("tiled dense attention is only valid for HCA layers");
            }
            let queries = self
                .queries
                .view_subrange(0, vec![dims.query_width as u64, 1]);
            let output = self
                .attention
                .view_subrange(0, vec![dims.query_width as u64, 1]);
            if self.use_online_hca() {
                if self.use_splitk_hca(ctx) && deepseek_v4_splitk_hca_enabled() {
                    static REPORTED: std::sync::Once = std::sync::Once::new();
                    REPORTED.call_once(|| {
                        eprintln!(
                            "deepseek_v4: grouped split-K HCA runs eight independent history partitions; rollback=QWEN_DSV4_SPLITK_HCA=0"
                        );
                    });
                    encode_grouped_splitk_hca_f16(
                        ctx,
                        enc,
                        &queries,
                        raw_cache,
                        raw_cache,
                        DeepSeekV4RawCacheLayout::Ring,
                        rows,
                        sinks,
                        &self.hca_partial_output,
                        &self.hca_partial_ml,
                        &output,
                        position,
                        DEEPSEEK_V4_SPLITK_HCA_PARTITIONS,
                        c,
                    )?;
                } else if self.use_grouped_long_hca(ctx) && deepseek_v4_grouped_long_hca_enabled() {
                    static REPORTED: std::sync::Once = std::sync::Once::new();
                    REPORTED.call_once(|| {
                        eprintln!(
                            "deepseek_v4: grouped online HCA shares long-history rows across eight heads; rollback=QWEN_DSV4_GROUPED_LONG_HCA=0"
                        );
                    });
                    encode_grouped_online_dense_sink_attention_f16(
                        ctx,
                        enc,
                        &queries,
                        raw_cache,
                        raw_cache,
                        DeepSeekV4RawCacheLayout::Ring,
                        Some(rows),
                        sinks,
                        &output,
                        kind,
                        position,
                        1,
                        c,
                    )?;
                } else {
                    let direct_load = deepseek_v4_online_direct_load_enabled();
                    if direct_load {
                        static REPORTED: std::sync::Once = std::sync::Once::new();
                        REPORTED.call_once(|| {
                            eprintln!(
                                "deepseek_v4: online HCA loads rows directly; rollback=QWEN_DSV4_ONLINE_DIRECT_LOAD=0"
                            );
                        });
                    }
                    encode_online_dense_sink_attention_f16(
                        ctx,
                        enc,
                        &queries,
                        raw_cache,
                        raw_cache,
                        DeepSeekV4RawCacheLayout::Ring,
                        rows,
                        sinks,
                        &output,
                        position,
                        0,
                        1,
                        128,
                        direct_load,
                        c,
                    )?;
                }
            } else {
                encode_tiled_dense_sink_attention_f16(
                    ctx,
                    enc,
                    &queries,
                    raw_cache,
                    raw_cache,
                    DeepSeekV4RawCacheLayout::Ring,
                    rows,
                    sinks,
                    &output,
                    position,
                    0,
                    1,
                    128,
                    c,
                )?;
            }
        } else if position == 0 {
            if compressed.is_some() {
                return invalid("position-zero dense attention cannot have compressed rows");
            }
            encode_dense_sink_attention_f16(
                ctx,
                enc,
                &self.queries,
                raw_cache,
                None,
                sinks,
                &self.attention,
                position,
                c,
            )?;
        } else {
            let queries = self
                .queries
                .view_subrange(0, vec![dims.query_width as u64, 1]);
            let output = self
                .attention
                .view_subrange(0, vec![dims.query_width as u64, 1]);
            encode_cooperative_dense_sink_attention_f16(
                ctx,
                enc,
                &queries,
                raw_cache,
                raw_cache,
                DeepSeekV4RawCacheLayout::Ring,
                compressed,
                sinks,
                &output,
                kind,
                position,
                1,
                c,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_selected_attention_f16(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        raw_cache: &MetalTensor,
        rows: DeepSeekV4CsaRows<'_>,
        selection: DeepSeekV4CsaSelectionView<'_>,
        sinks: &MetalTensor,
        position: u32,
        rope: DeepSeekV4RopeParameters,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_selected_attention_finish")?;
        let c = self.config;
        let dims = c.checked()?;
        self.validate_scratch(dims)?;
        validate_ds4_rope(rope, c.head_dim, c.rotary_dim)?;
        selection.validate()?;
        let queries = self
            .queries
            .view_subrange(0, vec![dims.query_width as u64, 1]);
        let output = self
            .attention
            .view_subrange(0, vec![dims.query_width as u64, 1]);
        encode_cooperative_selected_sink_attention_f16(
            ctx,
            enc,
            &queries,
            raw_cache,
            raw_cache,
            DeepSeekV4RawCacheLayout::Ring,
            rows.attention_cache,
            rows.capacity_rows,
            selection.cache_order_ids,
            selection.selected_count,
            selection.visible_count,
            sinks,
            &output,
            position,
            0,
            1,
            1,
            DEEPSEEK_V4_CSA_TOP_K,
            false,
            false,
            c,
        )?;
        Ok(())
    }

    pub(super) fn encode_attention_output<'a>(
        &'a self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        output_a: &MetalTensor,
        output_b: &MetalTensor,
        position: u32,
        rope: DeepSeekV4RopeParameters,
    ) -> Result<&'a MetalTensor, DeepSeekV4MetalError> {
        let c = self.config;
        let dims = c.checked()?;
        validate_matvec_weight(
            output_a,
            dims.group_width,
            dims.low_rank_width,
            "output A weight",
        )?;
        validate_matvec_weight(
            output_b,
            dims.low_rank_width,
            c.hidden_size,
            "output B weight",
        )?;
        encode_ds4_rope_tail_adjacent_in_place(ctx, enc, &self.attention, position, rope, true)?;

        let grouped_output_a = deepseek_v4_decode_output_grouped_enabled()
            && output_a.dtype == GgmlType::Q8_0
            && crate::metal::mat_vec_q8_0_lcpp_enabled();
        if grouped_output_a {
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| {
                eprintln!(
                    "deepseek_v4: decode output A runs one grouped GEMV; rollback=QWEN_DSV4_DECODE_OUTPUT_GROUPED=0"
                );
            });
            crate::metal::encode_mat_vec_q8_0_grouped_f32(
                ctx,
                enc,
                output_a,
                &self.attention,
                &self.low_rank,
                dims.group_width,
                c.output_rank,
                c.group_count,
            )
            .map_err(DeepSeekV4MetalError::Metal)?;
        } else {
            for group in 0..c.group_count {
                let input_view = self.attention.view_subrange(
                    (group * dims.group_width) as u64,
                    vec![dims.group_width as u64],
                );
                let output_view = self
                    .low_rank
                    .view_subrange((group * c.output_rank) as u64, vec![c.output_rank as u64]);
                let weight_view =
                    group_weight_view(output_a, dims.group_width, c.output_rank, group)?;
                encode_projection(
                    ctx,
                    enc,
                    &weight_view,
                    &input_view,
                    &output_view,
                    dims.group_width,
                    c.output_rank,
                    "grouped output A",
                )?;
            }
        }
        encode_projection(
            ctx,
            enc,
            output_b,
            &self.low_rank,
            &self.output,
            dims.low_rank_width,
            c.hidden_size,
            "output B",
        )?;
        Ok(&self.output)
    }

    pub(super) fn validate_scratch(
        &self,
        dims: CheckedAttentionDims,
    ) -> Result<(), DeepSeekV4MetalError> {
        let c = self.config;
        validate_f32(
            &self.normalized_input,
            &[c.hidden_size as u64],
            true,
            "normalized input scratch",
        )?;
        validate_f32(
            &self.q_lora_raw,
            &[c.q_lora_rank as u64],
            true,
            "raw Q LoRA scratch",
        )?;
        validate_f32(
            &self.q_lora,
            &[c.q_lora_rank as u64],
            true,
            "Q LoRA scratch",
        )?;
        validate_f32(
            &self.queries_raw,
            &[dims.query_width as u64],
            true,
            "raw query scratch",
        )?;
        validate_f32(
            &self.queries,
            &[c.head_dim as u64, c.head_count as u64],
            true,
            "query scratch",
        )?;
        validate_f32(&self.kv_raw, &[c.head_dim as u64], true, "raw KV scratch")?;
        validate_f32(&self.kv, &[c.head_dim as u64], true, "KV scratch")?;
        validate_f32(
            &self.cached_kv,
            &[c.head_dim as u64],
            true,
            "cached KV scratch",
        )?;
        validate_f32(
            &self.attention,
            &[c.head_dim as u64, c.head_count as u64],
            true,
            "attention scratch",
        )?;
        validate_f32(
            &self.low_rank,
            &[dims.low_rank_width as u64],
            true,
            "low-rank scratch",
        )?;
        validate_f32(
            &self.output,
            &[c.hidden_size as u64],
            true,
            "attention output scratch",
        )?;
        validate_f32(
            &self.head_norm_ones,
            &[c.head_dim as u64],
            false,
            "head norm ones scratch",
        )
    }
}

/// Session-owned storage for DeepSeek V4's four-stream manifold-constrained
/// hyper-connections. The all-ones tensor makes the existing weighted RMSNorm
/// kernel implement the required unweighted norm over the flattened `4H` row.
pub struct DeepSeekV4HyperConnectionScratch {
    pub(super) hidden_size: usize,
    pub(super) ones: MetalTensor,
    pub(super) normalized: MetalTensor,
    pub(super) mixes: MetalTensor,
    pub(super) pre: MetalTensor,
    pub(super) post: MetalTensor,
    pub(super) combination: MetalTensor,
    pub(super) collapsed: MetalTensor,
    pub(super) head_mixes: MetalTensor,
    pub(super) head_gates: MetalTensor,
}

impl DeepSeekV4HyperConnectionScratch {
    pub fn new(ctx: &MetalContext, hidden_size: usize) -> Result<Self, DeepSeekV4MetalError> {
        let residual_len = residual_len(hidden_size)?;
        let ones_values = vec![1.0f32; residual_len];
        Ok(Self {
            hidden_size,
            ones: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&ones_values),
                vec![residual_len as u64],
                GgmlType::F32,
            )?,
            normalized: MetalTensor::zeros_f32(ctx, vec![residual_len as u64])?,
            mixes: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_HC_PARAMETER_COUNT as u64])?,
            pre: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_CONNECTION_COUNT as u64])?,
            post: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_CONNECTION_COUNT as u64])?,
            combination: MetalTensor::zeros_f32(
                ctx,
                vec![
                    DEEPSEEK_V4_CONNECTION_COUNT as u64,
                    DEEPSEEK_V4_CONNECTION_COUNT as u64,
                ],
            )?,
            collapsed: MetalTensor::zeros_f32(ctx, vec![hidden_size as u64])?,
            head_mixes: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_CONNECTION_COUNT as u64])?,
            head_gates: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_CONNECTION_COUNT as u64])?,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn normalized(&self) -> &MetalTensor {
        &self.normalized
    }

    pub fn mixes(&self) -> &MetalTensor {
        &self.mixes
    }

    pub fn pre_gates(&self) -> &MetalTensor {
        &self.pre
    }

    pub fn post_gates(&self) -> &MetalTensor {
        &self.post
    }

    /// Source-major `[source, destination]` matrix.
    pub fn combination(&self) -> &MetalTensor {
        &self.combination
    }

    pub fn collapsed_input(&self) -> &MetalTensor {
        &self.collapsed
    }

    pub fn head_mixes(&self) -> &MetalTensor {
        &self.head_mixes
    }

    pub fn head_gates(&self) -> &MetalTensor {
        &self.head_gates
    }

    /// Repeat one embedding into four stream-major residual rows.
    pub fn encode_initial_repeat(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        embedding: &MetalTensor,
        residual: &MetalTensor,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_hc_repeat")?;
        validate_f32(embedding, &[self.hidden_size as u64], false, "embedding")?;
        validate_f32(
            residual,
            &[self.hidden_size as u64, DEEPSEEK_V4_CONNECTION_COUNT as u64],
            true,
            "residual",
        )?;
        let pso = ctx.pipeline("kernel_deepseek_v4_hc_repeat")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &u32_hidden(self.hidden_size)?);
        enc.set_tensor(1, embedding);
        enc.set_tensor(2, residual);
        enc.dispatch(
            MTLSize {
                width: residual_len(self.hidden_size)?.div_ceil(256),
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

    /// Flattened `4H` RMSNorm, `[4H,24]` projection, exact split-Sinkhorn
    /// controls, and weighted stream collapse.
    pub fn encode_pre(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residual: &MetalTensor,
        function: &MetalTensor,
        scale: &MetalTensor,
        base: &MetalTensor,
        rms_eps: f32,
        hc_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_hc_pre")?;
        validate_eps(rms_eps, "RMSNorm epsilon")?;
        validate_eps(hc_eps, "hyper-connection epsilon")?;
        let residual_shape = [self.hidden_size as u64, DEEPSEEK_V4_CONNECTION_COUNT as u64];
        validate_f32(residual, &residual_shape, false, "residual")?;
        validate_matvec_weight(
            function,
            residual_len(self.hidden_size)?,
            DEEPSEEK_V4_HC_PARAMETER_COUNT,
            "function",
        )?;
        validate_f32(scale, &[3], false, "scale")?;
        validate_f32(
            base,
            &[DEEPSEEK_V4_HC_PARAMETER_COUNT as u64],
            false,
            "base",
        )?;
        self.validate_scratch()?;

        encode_rms_norm_mul_f32(ctx, enc, residual, &self.ones, &self.normalized, rms_eps)?;
        if function.dtype == GgmlType::F32 {
            encode_f32_projection(
                ctx,
                enc,
                function,
                &self.normalized,
                &self.mixes,
                residual_len(self.hidden_size)?,
                DEEPSEEK_V4_HC_PARAMETER_COUNT,
            )?;
        } else {
            encode_projection(
                ctx,
                enc,
                function,
                &self.normalized,
                &self.mixes,
                residual_len(self.hidden_size)?,
                DEEPSEEK_V4_HC_PARAMETER_COUNT,
                "hyper-connection function",
            )?;
        }

        let pso = ctx.pipeline("kernel_deepseek_v4_hc_controls")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &hc_eps);
        enc.set_tensor(1, &self.mixes);
        enc.set_tensor(2, scale);
        enc.set_tensor(3, base);
        enc.set_tensor(4, &self.pre);
        enc.set_tensor(5, &self.post);
        enc.set_tensor(6, &self.combination);
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

        let pso = ctx.pipeline("kernel_deepseek_v4_hc_collapse")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &u32_hidden(self.hidden_size)?);
        enc.set_tensor(1, residual);
        enc.set_tensor(2, &self.pre);
        enc.set_tensor(3, &self.collapsed);
        enc.dispatch(
            MTLSize {
                width: self.hidden_size.div_ceil(256),
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

    /// Apply the most recently encoded pre controls to a block output and its
    /// source residual. The matrix is consumed as `source*4 + destination`.
    pub fn encode_post(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        block_output: &MetalTensor,
        residual: &MetalTensor,
        output: &MetalTensor,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_hc_post")?;
        validate_f32(
            block_output,
            &[self.hidden_size as u64],
            false,
            "block output",
        )?;
        let shape = [self.hidden_size as u64, DEEPSEEK_V4_CONNECTION_COUNT as u64];
        validate_f32(residual, &shape, false, "residual")?;
        validate_f32(output, &shape, true, "post residual")?;
        self.validate_scratch()?;
        let pso = ctx.pipeline("kernel_deepseek_v4_hc_post")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &u32_hidden(self.hidden_size)?);
        enc.set_tensor(1, block_output);
        enc.set_tensor(2, residual);
        enc.set_tensor(3, &self.post);
        enc.set_tensor(4, &self.combination);
        enc.set_tensor(5, output);
        enc.dispatch(
            MTLSize {
                width: residual_len(self.hidden_size)?.div_ceil(256),
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

    /// Flattened norm and `[4H,4]` projection followed by sigmoid+epsilon
    /// gates and four-stream collapse for the final output head.
    pub fn encode_head(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residual: &MetalTensor,
        function: &MetalTensor,
        scale: &MetalTensor,
        base: &MetalTensor,
        output: &MetalTensor,
        rms_eps: f32,
        hc_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_hc_head")?;
        validate_eps(rms_eps, "RMSNorm epsilon")?;
        validate_eps(hc_eps, "hyper-connection epsilon")?;
        let shape = [self.hidden_size as u64, DEEPSEEK_V4_CONNECTION_COUNT as u64];
        validate_f32(residual, &shape, false, "residual")?;
        validate_matvec_weight(
            function,
            residual_len(self.hidden_size)?,
            DEEPSEEK_V4_CONNECTION_COUNT,
            "head function",
        )?;
        validate_f32(scale, &[1], false, "head scale")?;
        validate_f32(
            base,
            &[DEEPSEEK_V4_CONNECTION_COUNT as u64],
            false,
            "head base",
        )?;
        validate_f32(output, &[self.hidden_size as u64], true, "head output")?;
        self.validate_scratch()?;

        encode_rms_norm_mul_f32(ctx, enc, residual, &self.ones, &self.normalized, rms_eps)?;
        if function.dtype == GgmlType::F32 {
            encode_f32_projection(
                ctx,
                enc,
                function,
                &self.normalized,
                &self.head_mixes,
                residual_len(self.hidden_size)?,
                DEEPSEEK_V4_CONNECTION_COUNT,
            )?;
        } else {
            encode_projection(
                ctx,
                enc,
                function,
                &self.normalized,
                &self.head_mixes,
                residual_len(self.hidden_size)?,
                DEEPSEEK_V4_CONNECTION_COUNT,
                "hyper-connection head function",
            )?;
        }
        let pso = ctx.pipeline("kernel_deepseek_v4_hc_head")?;
        enc.set_pipeline(&pso);
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Args {
            hidden_size: u32,
            eps: f32,
        }
        enc.set_bytes(
            0,
            &Args {
                hidden_size: u32_hidden(self.hidden_size)?,
                eps: hc_eps,
            },
        );
        enc.set_tensor(1, residual);
        enc.set_tensor(2, &self.head_mixes);
        enc.set_tensor(3, scale);
        enc.set_tensor(4, base);
        enc.set_tensor(5, &self.head_gates);
        enc.set_tensor(6, output);
        enc.dispatch(
            MTLSize {
                width: self
                    .hidden_size
                    .max(DEEPSEEK_V4_CONNECTION_COUNT)
                    .div_ceil(256),
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

    pub(super) fn validate_scratch(&self) -> Result<(), DeepSeekV4MetalError> {
        let residual_len = residual_len(self.hidden_size)? as u64;
        validate_f32(&self.ones, &[residual_len], false, "ones scratch")?;
        validate_f32(
            &self.normalized,
            &[residual_len],
            true,
            "normalized scratch",
        )?;
        validate_f32(&self.mixes, &[24], true, "mix scratch")?;
        validate_f32(&self.pre, &[4], true, "pre scratch")?;
        validate_f32(&self.post, &[4], true, "post scratch")?;
        validate_f32(&self.combination, &[4, 4], true, "combination scratch")?;
        validate_f32(
            &self.collapsed,
            &[self.hidden_size as u64],
            true,
            "collapse scratch",
        )?;
        validate_f32(&self.head_mixes, &[4], true, "head mix scratch")?;
        validate_f32(&self.head_gates, &[4], true, "head gate scratch")
    }
}

#[cfg(feature = "dsv4-diagnostics")]
pub(super) fn host_l2_norms_f32_rows(
    tensor: &MetalTensor,
    rows: usize,
    width: usize,
    name: &str,
) -> Result<Vec<f64>, DeepSeekV4MetalError> {
    let values = host_read_f32(tensor, name)?;
    let required = checked_mul(rows, width, name)?;
    if required > values.len() {
        return invalid(format!(
            "{name} requires {required} values for {rows}x{width} rows, tensor has {}",
            values.len()
        ));
    }
    values[..required]
        .chunks_exact(width)
        .map(|row| {
            let mut squared_norm = 0.0f64;
            for &value in row {
                if !value.is_finite() {
                    return invalid(format!("{name} contains a non-finite value"));
                }
                let value = f64::from(value).abs();
                let square = next_up_nonnegative(value * value)?;
                squared_norm = next_up_nonnegative(squared_norm + square)?;
            }
            next_up_nonnegative(squared_norm.sqrt())
        })
        .collect()
}

#[cfg(feature = "dsv4-diagnostics")]
pub(super) fn host_l2_norms_f16_rows(
    tensor: &MetalTensor,
    rows: usize,
    width: usize,
    name: &str,
) -> Result<Vec<f64>, DeepSeekV4MetalError> {
    validate_f16(tensor, &tensor.shape, false, name)?;
    let required = checked_mul(rows, width, name)?;
    if required > tensor.n_elements() as usize {
        return invalid(format!(
            "{name} requires {required} values for {rows}x{width} rows, tensor has {}",
            tensor.n_elements()
        ));
    }
    let source = unsafe {
        tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<u16>()
    };
    (0..rows)
        .map(|row| {
            let mut squared_norm = 0.0f64;
            for column in 0..width {
                let value =
                    half::f16::from_bits(unsafe { *source.add(row * width + column) }).to_f32();
                if !value.is_finite() {
                    return invalid(format!("{name} contains a non-finite value"));
                }
                let value = f64::from(value).abs();
                let square = next_up_nonnegative(value * value)?;
                squared_norm = next_up_nonnegative(squared_norm + square)?;
            }
            next_up_nonnegative(squared_norm.sqrt())
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_ds4_prepare_norm_pair(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_input: &MetalTensor,
    q_weight: &MetalTensor,
    q_output: &MetalTensor,
    kv_input: &MetalTensor,
    kv_weight: &MetalTensor,
    kv_output: &MetalTensor,
    eps: f32,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "paired prepare RMSNorm")?;
    let q_dim = usize::try_from(q_input.n_elements())
        .map_err(|_| DeepSeekV4MetalError::Invalid("paired Q norm width exceeds usize".into()))?;
    let kv_dim = usize::try_from(kv_input.n_elements())
        .map_err(|_| DeepSeekV4MetalError::Invalid("paired KV norm width exceeds usize".into()))?;
    validate_eps(eps, "paired prepare RMSNorm epsilon")?;
    validate_f32(q_input, &[q_dim as u64], false, "paired Q norm input")?;
    validate_f32(q_weight, &[q_dim as u64], false, "paired Q norm weight")?;
    validate_f32(q_output, &[q_dim as u64], true, "paired Q norm output")?;
    validate_f32(kv_input, &[kv_dim as u64], false, "paired KV norm input")?;
    validate_f32(kv_weight, &[kv_dim as u64], false, "paired KV norm weight")?;
    validate_f32(kv_output, &[kv_dim as u64], true, "paired KV norm output")?;
    if q_dim == 0
        || kv_dim == 0
        || metal_tensor_ranges_overlap(q_output, kv_output)
        || [q_output, kv_output].iter().any(|output| {
            [q_input, q_weight, kv_input, kv_weight]
                .iter()
                .any(|input| metal_tensor_ranges_overlap(output, input))
        })
        || u32::try_from(q_dim).is_err()
        || u32::try_from(kv_dim).is_err()
    {
        return invalid("paired prepare RMSNorm requires nonzero, distinct F32 input/output rows");
    }
    let reference = ctx.pipeline("kernel_rms_norm_mul_f32")?;
    let pso = ctx.pipeline("kernel_ds4_prepare_norm_pair_f32")?;
    let tg_threads = reference.maxTotalThreadsPerThreadgroup().min(1024);
    if tg_threads == 0 || pso.maxTotalThreadsPerThreadgroup() < tg_threads {
        return invalid(format!(
            "paired prepare RMSNorm pipeline supports {} threads, exact reference requires {tg_threads}",
            pso.maxTotalThreadsPerThreadgroup()
        ));
    }
    enc.note_read(q_input);
    enc.note_read(q_weight);
    enc.note_read(kv_input);
    enc.note_read(kv_weight);
    enc.note_write(q_output);
    enc.note_write(kv_output);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        q_dim: u32,
        kv_dim: u32,
        eps: f32,
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            q_dim: q_dim as u32,
            kv_dim: kv_dim as u32,
            eps,
        },
    );
    enc.set_tensor(1, q_input);
    enc.set_tensor(2, q_weight);
    enc.set_tensor(3, q_output);
    enc.set_tensor(4, kv_input);
    enc.set_tensor(5, kv_weight);
    enc.set_tensor(6, kv_output);
    let simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (simdgroups * std::mem::size_of::<f32>()).max(32));
    enc.dispatch(
        MTLSize {
            width: 2,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(super) fn encode_attention_cache_roundtrip(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    output: &MetalTensor,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        width: u32,
        rotary_dim: u32,
    }

    validate_f32(input, &[config.head_dim as u64], false, "pre-cache KV")?;
    validate_f32(output, &[config.head_dim as u64], true, "cached KV")?;
    let pso = ctx.pipeline("kernel_deepseek_v4_attention_cache_roundtrip")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            width: config.head_dim as u32,
            rotary_dim: config.rotary_dim as u32,
        },
    );
    enc.set_tensor(1, input);
    enc.set_tensor(2, output);
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

pub(super) fn ds4_rope_correction_bounds(rope: DeepSeekV4RopeParameters) -> (f32, f32) {
    let (correction_low, correction_high) = if rope.scaling_factor > 1.0 {
        let correction = |rotations: f32| {
            rope.rotary_dim as f32
                * (rope.original_context_length as f32 / (rotations * 2.0 * std::f32::consts::PI))
                    .ln()
                / (2.0 * rope.theta.ln())
        };
        (
            correction(rope.beta_fast).floor().max(0.0),
            correction(rope.beta_slow)
                .ceil()
                .min((rope.rotary_dim - 1) as f32),
        )
    } else {
        (0.0, 0.0)
    };
    (correction_low, correction_high)
}

pub(super) fn encode_ds4_rope_tail_adjacent_in_place(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    tensor: &MetalTensor,
    position: u32,
    rope: DeepSeekV4RopeParameters,
    inverse: bool,
) -> Result<(), DeepSeekV4MetalError> {
    validate_ds4_rope(
        rope,
        tensor.shape.first().copied().unwrap_or(0) as usize,
        rope.rotary_dim,
    )?;
    validate_f32(tensor, &tensor.shape, true, "DS4 RoPE tensor")?;
    let head_dim = usize::try_from(*tensor.shape.first().ok_or_else(|| {
        DeepSeekV4MetalError::Invalid("DS4 RoPE tensor has no head dimension".into())
    })?)
    .map_err(|_| DeepSeekV4MetalError::Invalid("DS4 RoPE head dimension exceeds usize".into()))?;
    if head_dim == 0 || !tensor.n_elements().is_multiple_of(head_dim as u64) {
        return invalid("DS4 RoPE tensor is not a complete set of heads");
    }
    let head_count = usize::try_from(tensor.n_elements() / head_dim as u64)
        .map_err(|_| DeepSeekV4MetalError::Invalid("DS4 RoPE head count exceeds usize".into()))?;
    if position == 0 {
        return Ok(());
    }

    let (correction_low, correction_high) = ds4_rope_correction_bounds(rope);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        rotary_dim: u32,
        position: u32,
        inverse: u32,
        yarn: u32,
        theta: f32,
        frequency_scale: f32,
        correction_low: f32,
        correction_high: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_rope_tail_adjacent_in_place")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: head_count as u32,
            head_dim: head_dim as u32,
            rotary_dim: rope.rotary_dim as u32,
            position,
            inverse: u32::from(inverse),
            yarn: u32::from(rope.scaling_factor > 1.0),
            theta: rope.theta,
            frequency_scale: 1.0 / rope.scaling_factor,
            correction_low,
            correction_high,
        },
    );
    enc.set_tensor(1, tensor);
    let pair_count = checked_mul(head_count, rope.rotary_dim / 2, "DS4 RoPE pair count")?;
    enc.dispatch(
        MTLSize {
            width: pair_count.div_ceil(256),
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

pub(super) fn encode_ds4_rope_pair_in_place(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    kv: &MetalTensor,
    position: u32,
    rope: DeepSeekV4RopeParameters,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "paired prepare RoPE")?;
    validate_ds4_rope(
        rope,
        q.shape.first().copied().unwrap_or(0) as usize,
        rope.rotary_dim,
    )?;
    validate_f32(q, &q.shape, true, "paired Q RoPE tensor")?;
    validate_f32(kv, &kv.shape, true, "paired KV RoPE tensor")?;
    let head_dim = usize::try_from(*q.shape.first().ok_or_else(|| {
        DeepSeekV4MetalError::Invalid("paired Q RoPE tensor has no head dimension".into())
    })?)
    .map_err(|_| {
        DeepSeekV4MetalError::Invalid("paired RoPE head dimension exceeds usize".into())
    })?;
    if head_dim == 0
        || kv.shape.first().copied() != Some(head_dim as u64)
        || !q.n_elements().is_multiple_of(head_dim as u64)
        || !kv.n_elements().is_multiple_of(head_dim as u64)
        || metal_tensor_ranges_overlap(q, kv)
    {
        return invalid(
            "paired RoPE requires distinct complete Q/KV head sets with one head width",
        );
    }
    if position == 0 {
        return Ok(());
    }
    let q_head_count = usize::try_from(q.n_elements() / head_dim as u64)
        .map_err(|_| DeepSeekV4MetalError::Invalid("paired Q head count exceeds usize".into()))?;
    let kv_head_count = usize::try_from(kv.n_elements() / head_dim as u64)
        .map_err(|_| DeepSeekV4MetalError::Invalid("paired KV head count exceeds usize".into()))?;
    u32::try_from(q.n_elements())
        .map_err(|_| DeepSeekV4MetalError::Invalid("paired Q elements exceed u32".into()))?;
    u32::try_from(kv.n_elements())
        .map_err(|_| DeepSeekV4MetalError::Invalid("paired KV elements exceed u32".into()))?;
    let pairs_per_head = rope.rotary_dim / 2;
    let q_pair_count = checked_mul(q_head_count, pairs_per_head, "paired Q RoPE pair count")?;
    let kv_pair_count = checked_mul(kv_head_count, pairs_per_head, "paired KV RoPE pair count")?;
    let pair_count = q_pair_count
        .checked_add(kv_pair_count)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid("paired RoPE pair count overflow".into()))?;
    let (correction_low, correction_high) = ds4_rope_correction_bounds(rope);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        q_pair_count: u32,
        pair_count: u32,
        head_dim: u32,
        rotary_dim: u32,
        position: u32,
        inverse: u32,
        yarn: u32,
        theta: f32,
        frequency_scale: f32,
        correction_low: f32,
        correction_high: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_rope_pair_in_place")?;
    enc.note_write(q);
    enc.note_write(kv);
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            q_pair_count: u32::try_from(q_pair_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("paired Q RoPE pairs exceed u32".into())
            })?,
            pair_count: u32::try_from(pair_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("paired RoPE pairs exceed u32".into())
            })?,
            head_dim: u32::try_from(head_dim)
                .map_err(|_| DeepSeekV4MetalError::Invalid("paired head dim exceeds u32".into()))?,
            rotary_dim: rope.rotary_dim as u32,
            position,
            inverse: 0,
            yarn: u32::from(rope.scaling_factor > 1.0),
            theta: rope.theta,
            frequency_scale: 1.0 / rope.scaling_factor,
            correction_low,
            correction_high,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, kv);
    enc.dispatch(
        MTLSize {
            width: pair_count.div_ceil(256),
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

pub(super) fn encode_ds4_rope_tail_adjacent_batch_in_place(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    tensor: &MetalTensor,
    start_position: u32,
    row_count: usize,
    position_stride: u32,
    rope: DeepSeekV4RopeParameters,
    inverse: bool,
) -> Result<(), DeepSeekV4MetalError> {
    if row_count == 0 {
        return invalid("DS4 batched RoPE requires at least one row");
    }
    if position_stride == 0 {
        return invalid("DS4 batched RoPE requires a nonzero position stride");
    }
    let head_dim = usize::try_from(*tensor.shape.first().ok_or_else(|| {
        DeepSeekV4MetalError::Invalid("DS4 batched RoPE tensor has no head dimension".into())
    })?)
    .map_err(|_| {
        DeepSeekV4MetalError::Invalid("DS4 batched RoPE head dimension exceeds usize".into())
    })?;
    validate_ds4_rope(rope, head_dim, rope.rotary_dim)?;
    validate_f32(tensor, &tensor.shape, true, "DS4 batched RoPE tensor")?;
    let row_width = tensor
        .n_elements()
        .checked_div(row_count as u64)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid("DS4 batched RoPE row overflow".into()))?;
    if head_dim == 0
        || row_width == 0
        || row_width * row_count as u64 != tensor.n_elements()
        || !row_width.is_multiple_of(head_dim as u64)
    {
        return invalid("DS4 batched RoPE tensor is not a complete row-major head set");
    }
    start_position
        .checked_add(
            u32::try_from(row_count - 1)
                .map_err(|_| {
                    DeepSeekV4MetalError::Invalid("DS4 batched RoPE row count exceeds u32".into())
                })?
                .checked_mul(position_stride)
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid("DS4 batched RoPE position span overflow".into())
                })?,
        )
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("DS4 batched RoPE position overflow".into())
        })?;
    let head_count = usize::try_from(row_width / head_dim as u64).map_err(|_| {
        DeepSeekV4MetalError::Invalid("DS4 batched RoPE head count exceeds usize".into())
    })?;
    let (correction_low, correction_high) = ds4_rope_correction_bounds(rope);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        rotary_dim: u32,
        start_position: u32,
        row_count: u32,
        position_stride: u32,
        inverse: u32,
        yarn: u32,
        theta: f32,
        frequency_scale: f32,
        correction_low: f32,
        correction_high: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_rope_tail_adjacent_batch_in_place")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: u32::try_from(head_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("DS4 batched RoPE heads exceed u32".into())
            })?,
            head_dim: u32::try_from(head_dim).map_err(|_| {
                DeepSeekV4MetalError::Invalid("DS4 batched RoPE head dimension exceeds u32".into())
            })?,
            rotary_dim: u32::try_from(rope.rotary_dim).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "DS4 batched RoPE rotary dimension exceeds u32".into(),
                )
            })?,
            start_position,
            row_count: u32::try_from(row_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("DS4 batched RoPE rows exceed u32".into())
            })?,
            position_stride,
            inverse: u32::from(inverse),
            yarn: u32::from(rope.scaling_factor > 1.0),
            theta: rope.theta,
            frequency_scale: 1.0 / rope.scaling_factor,
            correction_low,
            correction_high,
        },
    );
    enc.set_tensor(1, tensor);
    let pair_count = checked_mul(
        checked_mul(row_count, head_count, "DS4 batched RoPE row heads")?,
        rope.rotary_dim / 2,
        "DS4 batched RoPE pair count",
    )?;
    enc.dispatch(
        MTLSize {
            width: pair_count.div_ceil(256),
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

pub(super) fn validate_raw_attention_caches(
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    layout: DeepSeekV4RawCacheLayout,
    head_dim: usize,
    token_count: usize,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    validate_f16(
        raw_cache,
        &[head_dim as u64, layout.rows(token_count) as u64],
        false,
        &format!("{name} current raw cache"),
    )?;
    validate_f16(
        raw_cache_before_chunk,
        &[head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        false,
        &format!("{name} preserved raw cache"),
    )?;
    if layout == DeepSeekV4RawCacheLayout::Chunk
        && Retained::as_ptr(&raw_cache.buffer) == Retained::as_ptr(&raw_cache_before_chunk.buffer)
    {
        let raw_end = raw_cache
            .offset
            .checked_add(raw_cache.n_bytes())
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("raw cache range overflow".into()))?;
        let preserved_end = raw_cache_before_chunk
            .offset
            .checked_add(raw_cache_before_chunk.n_bytes())
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("preserved raw cache range overflow".into())
            })?;
        if raw_cache.offset < preserved_end && raw_cache_before_chunk.offset < raw_end {
            return invalid(format!(
                "{name} requires disjoint current and preserved raw caches"
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DeepSeekV4DenseAttentionKernel {
    Cooperative,
    GroupedOnline,
}

pub(super) fn validate_ds4_rope(
    rope: DeepSeekV4RopeParameters,
    head_dim: usize,
    expected_rotary_dim: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if head_dim == 0
        || rope.rotary_dim != expected_rotary_dim
        || rope.rotary_dim == 0
        || rope.rotary_dim > head_dim
        || !rope.rotary_dim.is_multiple_of(2)
    {
        return invalid(format!(
            "invalid DS4 RoPE dimensions: head={head_dim} rotary={} expected={expected_rotary_dim}",
            rope.rotary_dim
        ));
    }
    if !rope.theta.is_finite() || rope.theta <= 1.0 {
        return invalid(format!("invalid DS4 RoPE theta {}", rope.theta));
    }
    if !rope.scaling_factor.is_finite() || rope.scaling_factor < 1.0 {
        return invalid(format!(
            "invalid DS4 RoPE scaling factor {}",
            rope.scaling_factor
        ));
    }
    if rope.scaling_factor > 1.0
        && (rope.original_context_length == 0
            || !rope.beta_fast.is_finite()
            || rope.beta_fast <= 0.0
            || !rope.beta_slow.is_finite()
            || rope.beta_slow <= 0.0)
    {
        return invalid("scaled DS4 RoPE requires an original context and positive YaRN betas");
    }
    Ok(())
}
