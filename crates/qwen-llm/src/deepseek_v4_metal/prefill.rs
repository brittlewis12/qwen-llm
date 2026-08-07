use super::*;
use std::sync::OnceLock;

pub const DEEPSEEK_V4_PREFILL_MAX_TOKENS: usize = 2_048;

const QUERY_WIDTH: usize = 64 * 512;
const GROUP_WIDTH: usize = QUERY_WIDTH / 8;
const LOW_RANK_WIDTH: usize = 8 * 1_024;
const COMPRESSOR_ATTENTION_WIDTH: usize = 2 * 512;
const COMPRESSOR_INDEXER_WIDTH: usize = 2 * 128;
const INDEXER_HEAD_COUNT: usize = 64;
const INDEXER_HEAD_DIM: usize = 128;
const INDEXER_QUERY_WIDTH: usize = INDEXER_HEAD_COUNT * INDEXER_HEAD_DIM;
const MOE_FFN_SIZE: usize = 2_048;
const MOE_EXPERT_COUNT: usize = 256;
const MOE_TOP_K: usize = 6;
#[cfg(any(test, feature = "dsv4-diagnostics"))]
const PACKED_ROUTE_RECORD_WIDTH: usize = 4;
const PACKED_COMPACT_ROUTE_HEADER_WIDTH: usize = 8;
const PACKED_COMPACT_ROUTE_STATUS_READY: i32 = 1;
#[cfg(test)]
const PACKED_COMPACT_ROUTE_STATUS_INVALID_ROUTE: i32 = -301;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedRouteArgs {
    expert_count: u32,
    top_k: u32,
    n_tokens: u32,
    produced_tokens: u32,
    vocab_size: u32,
    generation: u32,
    routed_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedRouteScheduleArgs {
    expert_count: u32,
    top_k: u32,
    n_tokens: u32,
    generation: u32,
    max_tiles32: u32,
    max_tiles16: u32,
}

struct PackedGpuRouteBuffers<'a> {
    logits: &'a MetalTensor,
    token_ids: &'a MetalTensor,
    expert_ids: &'a MetalTensor,
    weights: &'a MetalTensor,
    route_generations: &'a MetalTensor,
    route_status: &'a MetalTensor,
    counts: &'a MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    slot_ids: &'a MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    schedule_generations: &'a MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    aggregate: &'a MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    signature: &'a MetalTensor,
    compact_header: &'a MetalTensor,
}

fn packed_route_args(
    n_tokens: usize,
    produced_tokens: usize,
    vocab_size: usize,
    generation: NonZeroU32,
    routed_scale: f32,
) -> Result<PackedRouteArgs, DeepSeekV4MetalError> {
    if !routed_scale.is_finite() || routed_scale <= 0.0 {
        return invalid("packed GPU route scale must be finite and positive");
    }
    Ok(PackedRouteArgs {
        expert_count: MOE_EXPERT_COUNT as u32,
        top_k: MOE_TOP_K as u32,
        n_tokens: checked_token_count(n_tokens)?,
        produced_tokens: u32::try_from(produced_tokens).map_err(|_| {
            DeepSeekV4MetalError::Invalid("packed GPU route producer count exceeds u32".into())
        })?,
        vocab_size: u32::try_from(vocab_size).map_err(|_| {
            DeepSeekV4MetalError::Invalid("packed GPU route vocabulary exceeds u32".into())
        })?,
        generation: generation.get(),
        routed_scale,
    })
}

fn packed_route_schedule_args(
    n_tokens: usize,
    generation: NonZeroU32,
) -> Result<PackedRouteScheduleArgs, DeepSeekV4MetalError> {
    Ok(PackedRouteScheduleArgs {
        expert_count: MOE_EXPERT_COUNT as u32,
        top_k: MOE_TOP_K as u32,
        n_tokens: checked_token_count(n_tokens)?,
        generation: generation.get(),
        max_tiles32: PACKED_GROUPED_EXPERT_MAX_TILES as u32,
        max_tiles16: PACKED_GROUPED_IQ2_MMA16_MAX_TILES as u32,
    })
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
fn packed_route_completion(generation: u32, n_tokens: usize) -> u32 {
    0xd551_0000 ^ generation ^ ((n_tokens as u32) << 8)
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
fn packed_route_signature_completion(generation: u32, n_tokens: usize) -> u32 {
    0xd552_0000 ^ generation ^ ((n_tokens as u32) << 8)
}

fn packed_route_compact_completion(generation: u32, n_tokens: usize) -> u32 {
    0xd553_0000 ^ generation ^ ((n_tokens as u32) << 8)
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
fn packed_route_signature_mix(hash: u32, value: u32) -> u32 {
    (hash ^ value).wrapping_mul(16_777_619)
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
fn packed_route_signature_hash(
    expert_ids: &[i32],
    weights: &[f32],
    counts: &[i32],
    slot_ids: &[i32],
    n_tokens: usize,
) -> Result<u32, DeepSeekV4MetalError> {
    if expert_ids.len() != n_tokens * MOE_TOP_K
        || weights.len() != expert_ids.len()
        || counts.len() != MOE_EXPERT_COUNT
        || slot_ids.len() != n_tokens * MOE_EXPERT_COUNT
    {
        return invalid("packed GPU route signature payload has invalid geometry");
    }
    let mut partials = [0u32; 256];
    for tid in 0..256 {
        let mut hash = 2_166_136_261 ^ tid as u32;
        for index in (tid..expert_ids.len()).step_by(256) {
            hash = packed_route_signature_mix(hash, expert_ids[index] as u32);
            hash = packed_route_signature_mix(hash, weights[index].to_bits());
        }
        hash = packed_route_signature_mix(hash, counts[tid] as u32);
        let base = tid * n_tokens;
        for &slot in &slot_ids[base..base + n_tokens] {
            hash = packed_route_signature_mix(hash, slot as u32);
        }
        partials[tid] = hash;
    }
    Ok(partials
        .into_iter()
        .fold(2_166_136_261, packed_route_signature_mix))
}

impl PackedGpuRouteBuffers<'_> {
    #[allow(clippy::too_many_arguments)]
    fn encode_learned(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        bias: &MetalTensor,
        n_tokens: usize,
        produced_tokens: usize,
        generation: NonZeroU32,
        routed_scale: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "packed learned GPU route")?;
        if produced_tokens == 0 || produced_tokens > n_tokens {
            return invalid(format!(
                "packed learned route producer count {produced_tokens} is outside 1..={n_tokens}"
            ));
        }
        validate_f32(
            bias,
            &[MOE_EXPERT_COUNT as u64],
            false,
            "packed learned route bias",
        )?;
        let pso = ctx.pipeline("kernel_deepseek_v4_packed_route_learned")?;
        if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 256 {
            return invalid("packed learned route requires SIMD width 32 and 256 threads");
        }
        enc.set_pipeline(&pso);
        enc.set_bytes(
            0,
            &packed_route_args(n_tokens, produced_tokens, 0, generation, routed_scale)?,
        );
        enc.set_tensor(1, self.logits);
        enc.set_tensor(2, bias);
        enc.set_tensor(3, self.expert_ids);
        enc.set_tensor(4, self.weights);
        enc.set_tensor(5, self.route_generations);
        enc.set_tensor(6, self.route_status);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: produced_tokens,
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
    fn encode_hash(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        token_to_expert: &MetalTensor,
        n_tokens: usize,
        produced_tokens: usize,
        generation: NonZeroU32,
        routed_scale: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "packed hash GPU route")?;
        if produced_tokens == 0 || produced_tokens > n_tokens {
            return invalid(format!(
                "packed hash route producer count {produced_tokens} is outside 1..={n_tokens}"
            ));
        }
        validate_i32_bank(token_to_expert, MOE_TOP_K, "packed hash route map")?;
        let vocab_size = usize::try_from(token_to_expert.shape[1]).map_err(|_| {
            DeepSeekV4MetalError::Invalid("packed hash vocabulary exceeds usize".into())
        })?;
        let pso = ctx.pipeline("kernel_deepseek_v4_packed_route_hash")?;
        if pso.maxTotalThreadsPerThreadgroup() < 256 {
            return invalid("packed hash route requires 256 threads per threadgroup");
        }
        enc.set_pipeline(&pso);
        enc.set_bytes(
            0,
            &packed_route_args(
                n_tokens,
                produced_tokens,
                vocab_size,
                generation,
                routed_scale,
            )?,
        );
        enc.set_tensor(1, self.logits);
        enc.set_tensor(2, self.token_ids);
        enc.set_tensor(3, token_to_expert);
        enc.set_tensor(4, self.expert_ids);
        enc.set_tensor(5, self.weights);
        enc.set_tensor(6, self.route_generations);
        enc.set_tensor(7, self.route_status);
        enc.dispatch(
            MTLSize {
                width: produced_tokens.div_ceil(256),
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
    fn encode_schedule(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        n_tokens: usize,
        produced_experts: usize,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "packed GPU route schedule")?;
        if produced_experts == 0 || produced_experts > MOE_EXPERT_COUNT {
            return invalid(format!(
                "packed schedule producer count {produced_experts} is outside 1..={MOE_EXPERT_COUNT}"
            ));
        }
        let pso = ctx.pipeline("kernel_deepseek_v4_packed_route_schedule")?;
        if pso.maxTotalThreadsPerThreadgroup() < produced_experts {
            return invalid("packed schedule exceeds pipeline threadgroup capacity");
        }
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &packed_route_schedule_args(n_tokens, generation)?);
        enc.set_tensor(1, self.expert_ids);
        enc.set_tensor(2, self.counts);
        enc.set_tensor(3, self.slot_ids);
        enc.set_tensor(4, self.schedule_generations);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: produced_experts,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    fn encode_validate(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        n_tokens: usize,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "packed GPU route validator")?;
        let pso = ctx.pipeline("kernel_deepseek_v4_packed_route_validate")?;
        if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 256 {
            return invalid("packed route validator requires SIMD width 32 and 256 threads");
        }
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &packed_route_schedule_args(n_tokens, generation)?);
        enc.set_tensor(1, self.route_generations);
        enc.set_tensor(2, self.route_status);
        enc.set_tensor(3, self.expert_ids);
        enc.set_tensor(4, self.weights);
        enc.set_tensor(5, self.counts);
        enc.set_tensor(6, self.slot_ids);
        enc.set_tensor(7, self.schedule_generations);
        enc.set_tensor(8, self.aggregate);
        enc.dispatch(
            MTLSize {
                width: 1,
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
    fn encode_signature(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        n_tokens: usize,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "packed GPU route signature")?;
        let pso = ctx.pipeline("kernel_deepseek_v4_packed_route_signature")?;
        if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 256 {
            return invalid("packed route signature requires SIMD width 32 and 256 threads");
        }
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &packed_route_schedule_args(n_tokens, generation)?);
        enc.set_tensor(1, self.aggregate);
        enc.set_tensor(2, self.expert_ids);
        enc.set_tensor(3, self.weights);
        enc.set_tensor(4, self.counts);
        enc.set_tensor(5, self.slot_ids);
        enc.set_tensor(6, self.signature);
        enc.dispatch(
            MTLSize {
                width: 1,
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

    fn encode_compact(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        bucket_rows: &MetalTensor,
        bucket_slots: &MetalTensor,
        tiles32: &MetalTensor,
        tiles16: &MetalTensor,
        n_tokens: usize,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "packed GPU route compaction")?;
        let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed GPU compact routes")?;
        validate_i32(
            bucket_rows,
            &[route_count as u64],
            true,
            "packed GPU compact rows",
        )?;
        validate_i32(
            bucket_slots,
            &[route_count as u64],
            true,
            "packed GPU compact slots",
        )?;
        validate_i32(
            tiles32,
            &[PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS as u64],
            true,
            "packed GPU compact 32-row tiles",
        )?;
        validate_i32(
            tiles16,
            &[PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS as u64],
            true,
            "packed GPU compact 16-row tiles",
        )?;
        validate_i32(
            self.compact_header,
            &[PACKED_COMPACT_ROUTE_HEADER_WIDTH as u64],
            true,
            "packed GPU compact header",
        )?;
        let pso = ctx.pipeline("kernel_deepseek_v4_packed_route_compact")?;
        if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 256 {
            return invalid("packed route compaction requires SIMD width 32 and 256 threads");
        }
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &packed_route_schedule_args(n_tokens, generation)?);
        enc.set_tensor(1, self.route_generations);
        enc.set_tensor(2, self.route_status);
        enc.set_tensor(3, self.expert_ids);
        enc.set_tensor(4, self.weights);
        enc.set_tensor(5, self.counts);
        enc.set_tensor(6, bucket_rows);
        enc.set_tensor(7, bucket_slots);
        enc.set_tensor(8, tiles32);
        enc.set_tensor(9, tiles16);
        enc.set_tensor(10, self.compact_header);
        enc.dispatch(
            MTLSize {
                width: 1,
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
}

pub(super) struct DeepSeekV4PrefillScratch {
    token_ids: MetalTensor,
    embedding: MetalTensor,
    residual_primary: MetalTensor,
    residual_secondary: MetalTensor,
    hyper: PrefillHyperScratch,
    attention: PrefillAttentionScratch,
    compressor: PrefillCompressorScratch,
    moe: PrefillMoeScratch,
}

struct PrefillHyperScratch {
    ones: MetalTensor,
    normalized: MetalTensor,
    mixes: MetalTensor,
    pre: MetalTensor,
    post: MetalTensor,
    combination: MetalTensor,
    collapsed: MetalTensor,
}

struct PrefillAttentionScratch {
    raw_cache_before_chunk: MetalTensor,
    raw_chunk: MetalTensor,
    normalized_input: MetalTensor,
    q_lora_raw: MetalTensor,
    q_lora: MetalTensor,
    queries_raw: MetalTensor,
    queries: MetalTensor,
    kv_raw: MetalTensor,
    kv: MetalTensor,
    attention: MetalTensor,
    low_rank: MetalTensor,
    output: MetalTensor,
    head_norm_ones: MetalTensor,
    group_input: MetalTensor,
    group_output: MetalTensor,
    sparse_csa: PrefillSparseCsaScratch,
}

struct PrefillSparseCsaScratch {
    capacity_rows: usize,
    index_queries: MetalTensor,
    head_weights: MetalTensor,
    visible_counts: MetalTensor,
    scores: MetalTensor,
    selected_mask: MetalTensor,
    cache_order_ids: MetalTensor,
    selected_counts: MetalTensor,
    status: MetalTensor,
}

struct PrefillCompressorScratch {
    attention_kv: MetalTensor,
    attention_score: MetalTensor,
    indexer_kv: MetalTensor,
    indexer_score: MetalTensor,
    hca_kv: MetalTensor,
    hca_score: MetalTensor,
    pooled_rows: MetalTensor,
    normalized_rows: MetalTensor,
    #[cfg(feature = "dsv4-diagnostics")]
    q8_matrix_invocations: Cell<u32>,
}

struct PrefillMoeScratch {
    normalized_input: MetalTensor,
    logits: MetalTensor,
    hash_ids: MetalTensor,
    expert_ids: MetalTensor,
    weights: MetalTensor,
    bucket_rows: MetalTensor,
    bucket_slots: MetalTensor,
    expert_input: MetalTensor,
    gate: MetalTensor,
    up: MetalTensor,
    inner: MetalTensor,
    bucket_output: MetalTensor,
    expert_outputs: MetalTensor,
    routed_output: MetalTensor,
    shared_output: MetalTensor,
    final_output: MetalTensor,
    grouped_tiles: MetalTensor,
    grouped_iq2_mma16_tiles: MetalTensor,
    grouped_inner: MetalTensor,
    gpu_route: PrefillGpuRouteScratch,
    #[cfg(feature = "dsv4-diagnostics")]
    grouped_iq2_invocations: Cell<u32>,
    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    grouped_iq3_invocations: Cell<u32>,
}

struct PrefillGpuRouteScratch {
    route_generations: MetalTensor,
    route_status: MetalTensor,
    counts: MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    slot_ids: MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    schedule_generations: MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    aggregate: MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    signature: MetalTensor,
    compact_header: MetalTensor,
    next_generation: Cell<u32>,
}

impl DeepSeekV4PrefillScratch {
    pub(super) fn new(
        ctx: &MetalContext,
        csa_capacity_rows: usize,
    ) -> Result<Self, DeepSeekV4MetalError> {
        if csa_capacity_rows < DEEPSEEK_V4_CSA_TOP_K
            || !csa_capacity_rows.is_multiple_of(DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS)
        {
            return invalid(format!(
                "packed sparse CSA capacity {csa_capacity_rows} is not an aligned top-k superset"
            ));
        }
        let n = DEEPSEEK_V4_PREFILL_MAX_TOKENS as u64;
        let compressed_rows = DEEPSEEK_V4_PREFILL_MAX_TOKENS.div_ceil(4) as u64;
        let h = DEEPSEEK_V4_HIDDEN_SIZE as u64;
        let residual = residual_len(DEEPSEEK_V4_HIDDEN_SIZE)? as u64;
        let ones = vec![1.0f32; residual as usize];
        Ok(Self {
            token_ids: MetalTensor::zeros_dtype(ctx, vec![n], GgmlType::I32)?,
            embedding: MetalTensor::zeros_f32(ctx, vec![h, n])?,
            residual_primary: MetalTensor::zeros_f32(
                ctx,
                vec![h, DEEPSEEK_V4_CONNECTION_COUNT as u64, n],
            )?,
            residual_secondary: MetalTensor::zeros_f32(
                ctx,
                vec![h, DEEPSEEK_V4_CONNECTION_COUNT as u64, n],
            )?,
            hyper: PrefillHyperScratch {
                ones: MetalTensor::from_bytes(
                    ctx,
                    bytemuck::cast_slice(&ones),
                    vec![residual],
                    GgmlType::F32,
                )?,
                normalized: MetalTensor::zeros_f32(ctx, vec![residual, n])?,
                mixes: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_HC_PARAMETER_COUNT as u64, n])?,
                pre: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_CONNECTION_COUNT as u64, n])?,
                post: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_CONNECTION_COUNT as u64, n])?,
                combination: MetalTensor::zeros_f32(
                    ctx,
                    vec![
                        DEEPSEEK_V4_CONNECTION_COUNT as u64,
                        DEEPSEEK_V4_CONNECTION_COUNT as u64,
                        n,
                    ],
                )?,
                collapsed: MetalTensor::zeros_f32(ctx, vec![h, n])?,
            },
            attention: PrefillAttentionScratch {
                raw_cache_before_chunk: MetalTensor::zeros_f16(
                    ctx,
                    vec![512, DEEPSEEK_V4_LOCAL_WINDOW as u64],
                )?,
                raw_chunk: MetalTensor::zeros_f16(ctx, vec![512, n])?,
                normalized_input: MetalTensor::zeros_f32(ctx, vec![h, n])?,
                q_lora_raw: MetalTensor::zeros_f32(ctx, vec![1_024, n])?,
                q_lora: MetalTensor::zeros_f32(ctx, vec![1_024, n])?,
                queries_raw: MetalTensor::zeros_f32(ctx, vec![QUERY_WIDTH as u64, n])?,
                queries: MetalTensor::zeros_f32(ctx, vec![QUERY_WIDTH as u64, n])?,
                kv_raw: MetalTensor::zeros_f32(ctx, vec![512, n])?,
                kv: MetalTensor::zeros_f32(ctx, vec![512, n])?,
                attention: MetalTensor::zeros_f32(ctx, vec![QUERY_WIDTH as u64, n])?,
                low_rank: MetalTensor::zeros_f32(ctx, vec![LOW_RANK_WIDTH as u64, n])?,
                output: MetalTensor::zeros_f32(ctx, vec![h, n])?,
                head_norm_ones: MetalTensor::from_bytes(
                    ctx,
                    bytemuck::cast_slice(&vec![1.0f32; 512]),
                    vec![512],
                    GgmlType::F32,
                )?,
                group_input: MetalTensor::zeros_f32(ctx, vec![GROUP_WIDTH as u64, n])?,
                group_output: MetalTensor::zeros_f32(ctx, vec![1_024, n])?,
                sparse_csa: PrefillSparseCsaScratch {
                    capacity_rows: csa_capacity_rows,
                    index_queries: MetalTensor::zeros_f32(
                        ctx,
                        vec![INDEXER_HEAD_DIM as u64, INDEXER_HEAD_COUNT as u64, n],
                    )?,
                    head_weights: MetalTensor::zeros_f32(ctx, vec![INDEXER_HEAD_COUNT as u64, n])?,
                    visible_counts: MetalTensor::zeros_i32(ctx, vec![n])?,
                    scores: MetalTensor::zeros_f32(ctx, vec![csa_capacity_rows as u64, n])?,
                    selected_mask: MetalTensor::zeros_i32(ctx, vec![csa_capacity_rows as u64, n])?,
                    cache_order_ids: MetalTensor::zeros_i32(
                        ctx,
                        vec![DEEPSEEK_V4_CSA_TOP_K as u64, n],
                    )?,
                    selected_counts: MetalTensor::zeros_i32(ctx, vec![n])?,
                    status: MetalTensor::zeros_i32(ctx, vec![n])?,
                },
            },
            compressor: PrefillCompressorScratch {
                attention_kv: MetalTensor::zeros_f32(
                    ctx,
                    vec![COMPRESSOR_ATTENTION_WIDTH as u64, n],
                )?,
                attention_score: MetalTensor::zeros_f32(
                    ctx,
                    vec![COMPRESSOR_ATTENTION_WIDTH as u64, n],
                )?,
                indexer_kv: MetalTensor::zeros_f32(ctx, vec![COMPRESSOR_INDEXER_WIDTH as u64, n])?,
                indexer_score: MetalTensor::zeros_f32(
                    ctx,
                    vec![COMPRESSOR_INDEXER_WIDTH as u64, n],
                )?,
                hca_kv: MetalTensor::zeros_f32(ctx, vec![512, n])?,
                hca_score: MetalTensor::zeros_f32(ctx, vec![512, n])?,
                pooled_rows: MetalTensor::zeros_f32(ctx, vec![512, compressed_rows])?,
                normalized_rows: MetalTensor::zeros_f32(ctx, vec![512, compressed_rows])?,
                #[cfg(feature = "dsv4-diagnostics")]
                q8_matrix_invocations: Cell::new(0),
            },
            moe: PrefillMoeScratch {
                normalized_input: MetalTensor::zeros_f32(ctx, vec![h, n])?,
                logits: MetalTensor::zeros_f32(ctx, vec![MOE_EXPERT_COUNT as u64, n])?,
                hash_ids: MetalTensor::zeros_dtype(ctx, vec![MOE_TOP_K as u64, n], GgmlType::I32)?,
                expert_ids: MetalTensor::zeros_dtype(
                    ctx,
                    vec![MOE_TOP_K as u64, n],
                    GgmlType::I32,
                )?,
                weights: MetalTensor::zeros_f32(ctx, vec![MOE_TOP_K as u64, n])?,
                bucket_rows: MetalTensor::zeros_dtype(
                    ctx,
                    vec![MOE_TOP_K as u64, n],
                    GgmlType::I32,
                )?,
                bucket_slots: MetalTensor::zeros_dtype(
                    ctx,
                    vec![MOE_TOP_K as u64, n],
                    GgmlType::I32,
                )?,
                expert_input: MetalTensor::zeros_f32(ctx, vec![h, n])?,
                gate: MetalTensor::zeros_f32(ctx, vec![MOE_FFN_SIZE as u64, n])?,
                up: MetalTensor::zeros_f32(ctx, vec![MOE_FFN_SIZE as u64, n])?,
                inner: MetalTensor::zeros_f32(ctx, vec![MOE_FFN_SIZE as u64, n])?,
                bucket_output: MetalTensor::zeros_f32(ctx, vec![h, n])?,
                expert_outputs: MetalTensor::zeros_f32(ctx, vec![h, MOE_TOP_K as u64, n])?,
                routed_output: MetalTensor::zeros_f32(ctx, vec![h, n])?,
                shared_output: MetalTensor::zeros_f32(ctx, vec![h, n])?,
                final_output: MetalTensor::zeros_f32(ctx, vec![h, n])?,
                grouped_tiles: MetalTensor::zeros_i32(
                    ctx,
                    vec![PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS as u64],
                )?,
                grouped_iq2_mma16_tiles: MetalTensor::zeros_i32(
                    ctx,
                    vec![PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS as u64],
                )?,
                grouped_inner: MetalTensor::zeros_f32(
                    ctx,
                    vec![
                        MOE_FFN_SIZE as u64,
                        MOE_TOP_K as u64,
                        PACKED_GROUPED_EXPERT_MAX_TOKENS as u64,
                    ],
                )?,
                gpu_route: PrefillGpuRouteScratch {
                    route_generations: MetalTensor::zeros_i32(
                        ctx,
                        vec![PACKED_GPU_ROUTE_MAX_TOKENS as u64],
                    )?,
                    route_status: MetalTensor::zeros_i32(
                        ctx,
                        vec![PACKED_GPU_ROUTE_MAX_TOKENS as u64],
                    )?,
                    counts: MetalTensor::zeros_i32(ctx, vec![MOE_EXPERT_COUNT as u64])?,
                    #[cfg(any(test, feature = "dsv4-diagnostics"))]
                    slot_ids: MetalTensor::zeros_i32(
                        ctx,
                        vec![PACKED_GPU_ROUTE_MAX_TOKENS as u64, MOE_EXPERT_COUNT as u64],
                    )?,
                    #[cfg(any(test, feature = "dsv4-diagnostics"))]
                    schedule_generations: MetalTensor::zeros_i32(
                        ctx,
                        vec![MOE_EXPERT_COUNT as u64],
                    )?,
                    #[cfg(any(test, feature = "dsv4-diagnostics"))]
                    aggregate: MetalTensor::zeros_i32(ctx, vec![PACKED_ROUTE_RECORD_WIDTH as u64])?,
                    #[cfg(any(test, feature = "dsv4-diagnostics"))]
                    signature: MetalTensor::zeros_i32(ctx, vec![PACKED_ROUTE_RECORD_WIDTH as u64])?,
                    compact_header: MetalTensor::zeros_i32(
                        ctx,
                        vec![PACKED_COMPACT_ROUTE_HEADER_WIDTH as u64],
                    )?,
                    next_generation: Cell::new(1),
                },
                #[cfg(feature = "dsv4-diagnostics")]
                grouped_iq2_invocations: Cell::new(0),
                #[cfg(all(test, feature = "dsv4-diagnostics"))]
                grouped_iq3_invocations: Cell::new(0),
            },
        })
    }
}

pub(super) fn append_session_allocation_requests(
    requests: &mut Vec<DeepSeekV4SessionAllocationRequest>,
    csa_capacity_rows: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let n = DEEPSEEK_V4_PREFILL_MAX_TOKENS;
    let h = DEEPSEEK_V4_HIDDEN_SIZE;
    let residual = residual_len(h)?;
    let f32_bytes = std::mem::size_of::<f32>();
    let f16_bytes = std::mem::size_of::<u16>();
    let i32_bytes = std::mem::size_of::<i32>();
    let mut push = |name: &str, elements: usize, element_bytes: usize| {
        push_session_allocation(requests, format!("prefill.{name}"), elements, element_bytes)
    };

    push("token_ids", n, i32_bytes)?;
    push(
        "embedding",
        checked_mul(n, h, "prefill embedding")?,
        f32_bytes,
    )?;
    for name in ["residual_primary", "residual_secondary"] {
        push(
            name,
            checked_mul(n, residual, "prefill residual")?,
            f32_bytes,
        )?;
    }
    push("hyper.ones", residual, f32_bytes)?;
    push(
        "hyper.normalized",
        checked_mul(n, residual, "prefill hyper normalized")?,
        f32_bytes,
    )?;
    push(
        "hyper.mixes",
        checked_mul(n, DEEPSEEK_V4_HC_PARAMETER_COUNT, "prefill hyper mixes")?,
        f32_bytes,
    )?;
    for name in ["hyper.pre", "hyper.post"] {
        push(
            name,
            checked_mul(n, DEEPSEEK_V4_CONNECTION_COUNT, "prefill hyper gates")?,
            f32_bytes,
        )?;
    }
    push(
        "hyper.combination",
        checked_mul(
            n,
            DEEPSEEK_V4_CONNECTION_COUNT * DEEPSEEK_V4_CONNECTION_COUNT,
            "prefill hyper combinations",
        )?,
        f32_bytes,
    )?;
    push(
        "hyper.collapsed",
        checked_mul(n, h, "prefill hyper collapsed")?,
        f32_bytes,
    )?;

    for (name, width) in [
        ("attention.normalized_input", h),
        ("attention.q_lora_raw", 1_024),
        ("attention.q_lora", 1_024),
        ("attention.queries_raw", QUERY_WIDTH),
        ("attention.queries", QUERY_WIDTH),
        ("attention.kv_raw", 512),
        ("attention.kv", 512),
        ("attention.attention", QUERY_WIDTH),
        ("attention.low_rank", LOW_RANK_WIDTH),
        ("attention.output", h),
        ("attention.group_input", GROUP_WIDTH),
        ("attention.group_output", 1_024),
    ] {
        push(name, checked_mul(n, width, name)?, f32_bytes)?;
    }
    push(
        "attention.raw_cache_before_chunk",
        checked_mul(512, DEEPSEEK_V4_LOCAL_WINDOW, "prefill raw-ring snapshot")?,
        f16_bytes,
    )?;
    push(
        "attention.raw_chunk",
        checked_mul(512, n, "prefill raw chunk")?,
        f16_bytes,
    )?;
    push("attention.head_norm_ones", 512, f32_bytes)?;
    push(
        "attention.sparse_csa.index_queries",
        checked_mul(n, INDEXER_QUERY_WIDTH, "packed sparse index queries")?,
        f32_bytes,
    )?;
    push(
        "attention.sparse_csa.head_weights",
        checked_mul(n, INDEXER_HEAD_COUNT, "packed sparse head weights")?,
        f32_bytes,
    )?;
    push("attention.sparse_csa.visible_counts", n, i32_bytes)?;
    for name in ["scores", "selected_mask"] {
        push(
            &format!("attention.sparse_csa.{name}"),
            checked_mul(n, csa_capacity_rows, "packed sparse row scratch")?,
            if name == "scores" {
                f32_bytes
            } else {
                i32_bytes
            },
        )?;
    }
    push(
        "attention.sparse_csa.cache_order_ids",
        checked_mul(n, DEEPSEEK_V4_CSA_TOP_K, "packed sparse selected IDs")?,
        i32_bytes,
    )?;
    for name in ["selected_counts", "status"] {
        push(&format!("attention.sparse_csa.{name}"), n, i32_bytes)?;
    }

    for name in ["compressor.attention_kv", "compressor.attention_score"] {
        push(
            name,
            checked_mul(n, COMPRESSOR_ATTENTION_WIDTH, name)?,
            f32_bytes,
        )?;
    }
    for name in ["compressor.indexer_kv", "compressor.indexer_score"] {
        push(
            name,
            checked_mul(n, COMPRESSOR_INDEXER_WIDTH, name)?,
            f32_bytes,
        )?;
    }
    for name in ["compressor.hca_kv", "compressor.hca_score"] {
        push(name, checked_mul(n, 512, name)?, f32_bytes)?;
    }
    let compressed_rows = n.div_ceil(4);
    for name in ["compressor.pooled_rows", "compressor.normalized_rows"] {
        push(name, checked_mul(compressed_rows, 512, name)?, f32_bytes)?;
    }

    for (name, width, element_bytes) in [
        ("moe.normalized_input", h, f32_bytes),
        ("moe.logits", MOE_EXPERT_COUNT, f32_bytes),
        ("moe.hash_ids", MOE_TOP_K, i32_bytes),
        ("moe.expert_ids", MOE_TOP_K, i32_bytes),
        ("moe.weights", MOE_TOP_K, f32_bytes),
        ("moe.bucket_rows", MOE_TOP_K, i32_bytes),
        ("moe.bucket_slots", MOE_TOP_K, i32_bytes),
        ("moe.expert_input", h, f32_bytes),
        ("moe.gate", MOE_FFN_SIZE, f32_bytes),
        ("moe.up", MOE_FFN_SIZE, f32_bytes),
        ("moe.inner", MOE_FFN_SIZE, f32_bytes),
        ("moe.bucket_output", h, f32_bytes),
        ("moe.expert_outputs", MOE_TOP_K * h, f32_bytes),
        ("moe.routed_output", h, f32_bytes),
        ("moe.shared_output", h, f32_bytes),
        ("moe.final_output", h, f32_bytes),
    ] {
        push(name, checked_mul(n, width, name)?, element_bytes)?;
    }
    push(
        "moe.grouped_tiles",
        PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS,
        i32_bytes,
    )?;
    push(
        "moe.grouped_iq2_mma16_tiles",
        PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS,
        i32_bytes,
    )?;
    push(
        "moe.grouped_inner",
        checked_mul(
            checked_mul(
                PACKED_GROUPED_EXPERT_MAX_TOKENS,
                MOE_TOP_K,
                "packed grouped slots",
            )?,
            MOE_FFN_SIZE,
            "packed grouped inner",
        )?,
        f32_bytes,
    )?;
    for (name, elements) in [
        (
            "moe.gpu_route.route_generations",
            PACKED_GPU_ROUTE_MAX_TOKENS,
        ),
        ("moe.gpu_route.route_status", PACKED_GPU_ROUTE_MAX_TOKENS),
        ("moe.gpu_route.counts", MOE_EXPERT_COUNT),
        (
            "moe.gpu_route.compact_header",
            PACKED_COMPACT_ROUTE_HEADER_WIDTH,
        ),
    ] {
        push(name, elements, i32_bytes)?;
    }
    #[cfg(feature = "dsv4-diagnostics")]
    for (name, elements) in [
        (
            "moe.gpu_route.slot_ids",
            checked_mul(
                PACKED_GPU_ROUTE_MAX_TOKENS,
                MOE_EXPERT_COUNT,
                "packed GPU route slots",
            )?,
        ),
        ("moe.gpu_route.schedule_generations", MOE_EXPERT_COUNT),
        ("moe.gpu_route.aggregate", PACKED_ROUTE_RECORD_WIDTH),
        ("moe.gpu_route.signature", PACKED_ROUTE_RECORD_WIDTH),
    ] {
        push(name, elements, i32_bytes)?;
    }
    Ok(())
}

fn checked_token_count(n_tokens: usize) -> Result<u32, DeepSeekV4MetalError> {
    if n_tokens == 0 || n_tokens > DEEPSEEK_V4_PREFILL_MAX_TOKENS {
        return invalid(format!(
            "DeepSeek V4 packed prefill requires 1..={DEEPSEEK_V4_PREFILL_MAX_TOKENS} tokens, got {n_tokens}"
        ));
    }
    u32::try_from(n_tokens)
        .map_err(|_| DeepSeekV4MetalError::Invalid("prefill token count exceeds u32".into()))
}

fn f32_prefix(
    tensor: &MetalTensor,
    shape: Vec<u64>,
    name: &str,
) -> Result<MetalTensor, DeepSeekV4MetalError> {
    let elements = shape.iter().try_fold(1_u64, |total, &dimension| {
        total.checked_mul(dimension).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!("{name} shape element count overflow"))
        })
    })?;
    if elements > tensor.n_elements() {
        return invalid(format!(
            "{name} prefix requires {elements} elements, backing has {}",
            tensor.n_elements()
        ));
    }
    let view = tensor.view_subrange(0, shape);
    validate_f32(&view, &view.shape, tensor.is_writable(), name)?;
    Ok(view)
}

fn f16_prefix(
    tensor: &MetalTensor,
    shape: Vec<u64>,
    name: &str,
) -> Result<MetalTensor, DeepSeekV4MetalError> {
    let elements = shape.iter().try_fold(1_u64, |total, &dimension| {
        total.checked_mul(dimension).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!("{name} shape element count overflow"))
        })
    })?;
    if elements > tensor.n_elements() {
        return invalid(format!(
            "{name} prefix requires {elements} elements, backing has {}",
            tensor.n_elements()
        ));
    }
    let view = tensor.view_subrange(0, shape);
    validate_f16(&view, &view.shape, tensor.is_writable(), name)?;
    Ok(view)
}

fn i32_prefix(
    tensor: &MetalTensor,
    shape: Vec<u64>,
    name: &str,
) -> Result<MetalTensor, DeepSeekV4MetalError> {
    let elements = shape.iter().try_fold(1_u64, |total, &dimension| {
        total.checked_mul(dimension).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!("{name} shape element count overflow"))
        })
    })?;
    if elements > tensor.n_elements() {
        return invalid(format!(
            "{name} prefix requires {elements} elements, backing has {}",
            tensor.n_elements()
        ));
    }
    let view = tensor.view_subrange(0, shape);
    validate_i32(&view, &view.shape, tensor.is_writable(), name)?;
    Ok(view)
}

fn f32_row(
    tensor: &MetalTensor,
    row: usize,
    width: usize,
    shape: Vec<u64>,
    name: &str,
) -> Result<MetalTensor, DeepSeekV4MetalError> {
    let offset = checked_mul(row, width, &format!("{name} row offset"))?;
    let view = tensor.view_subrange(offset as u64, shape);
    validate_f32(&view, &view.shape, tensor.is_writable(), name)?;
    Ok(view)
}

fn i32_slice(
    tensor: &MetalTensor,
    offset: usize,
    len: usize,
    name: &str,
) -> Result<MetalTensor, DeepSeekV4MetalError> {
    let view = tensor.view_subrange(offset as u64, vec![len as u64]);
    validate_i32(&view, &view.shape, tensor.is_writable(), name)?;
    Ok(view)
}

fn encode_batch_projection(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_tokens: usize,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    checked_token_count(n_tokens)?;
    validate_matvec_weight(weight, n_in, n_out, name)?;
    if weight.dtype == GgmlType::MXFP4 {
        return invalid(format!(
            "{name} requires MXFP4 matmat, which is not implemented"
        ));
    }
    if input.dtype != GgmlType::F32
        || output.dtype != GgmlType::F32
        || input.n_elements() != checked_mul(n_tokens, n_in, name)? as u64
        || output.n_elements() != checked_mul(n_tokens, n_out, name)? as u64
        || !output.is_writable()
    {
        return invalid(format!(
            "{name} packed projection requires F32 [{n_tokens},{n_in}] -> [{n_tokens},{n_out}]"
        ));
    }
    if weight.dtype == GgmlType::Q8_0 {
        return crate::metal::encode_mat_vec_q8_0_batch_f32(
            ctx, enc, weight, input, output, n_in, n_out, n_tokens,
        )
        .map_err(DeepSeekV4MetalError::Metal);
    }
    crate::metal_forward::encode_mat_mat_dispatch(
        ctx, enc, weight, input, output, n_in, n_out, n_tokens,
    )
    .map_err(|error| match error {
        crate::metal_forward::MfError::Metal(error) => DeepSeekV4MetalError::Metal(error),
        other => DeepSeekV4MetalError::Invalid(format!("{name} packed projection failed: {other}")),
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_q8_f32_mma_r2c4k64(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "packed Q8 F32 R2C4K64 projection")?;
    checked_token_count(n_tokens)?;
    validate_matvec_weight(weight, n_in, n_out, "packed Q8 F32 R2C4K64 weight")?;
    validate_f32(
        input,
        &[n_in as u64, n_tokens as u64],
        false,
        "packed Q8 F32 R2C4K64 input",
    )?;
    validate_f32(
        output,
        &[n_out as u64, n_tokens as u64],
        true,
        "packed Q8 F32 R2C4K64 output",
    )?;
    if weight.dtype != GgmlType::Q8_0
        || !n_in.is_multiple_of(64)
        || !n_out.is_multiple_of(16)
        || !(1..=DEEPSEEK_V4_PREFILL_MAX_TOKENS).contains(&n_tokens)
    {
        return invalid("packed Q8 F32 R2C4K64 projection has invalid geometry or storage");
    }
    let padded_tokens = n_tokens.div_ceil(32) * 32;
    let padded_elements = checked_mul(padded_tokens, n_in, "packed Q8 F32 R2C4K64 padded input")?;
    let padded_bytes = checked_mul(
        padded_elements,
        std::mem::size_of::<f32>(),
        "packed Q8 F32 R2C4K64 padded input bytes",
    )?;
    let padded_end = input
        .offset
        .checked_add(u64::try_from(padded_bytes).map_err(|_| {
            DeepSeekV4MetalError::Invalid("packed Q8 F32 R2C4K64 padded input exceeds u64".into())
        })?)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed Q8 F32 R2C4K64 padded input end overflow".into())
        })?;
    if padded_end > input.buffer.length() as u64 || ctx.device.maxThreadgroupMemoryLength() < 4_096
    {
        return invalid("packed Q8 F32 R2C4K64 requires padded input backing and 4 KiB TGM");
    }
    let output_end = output.offset.checked_add(output.n_bytes()).ok_or_else(|| {
        DeepSeekV4MetalError::Invalid("packed Q8 F32 R2C4K64 output end overflow".into())
    })?;
    let overlaps_padded_input = Retained::as_ptr(&input.buffer) == Retained::as_ptr(&output.buffer)
        && input.offset < output_end
        && output.offset < padded_end;
    if overlaps_padded_input || packed_grouped_tensor_ranges_overlap(weight, output) {
        return invalid("packed Q8 F32 R2C4K64 output overlaps an input");
    }
    let pso = ctx.pipeline("kernel_mat_mat_q8_0_f32_r2c4k64")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 32 {
        return invalid("packed Q8 F32 R2C4K64 requires one 32-thread SIMD group");
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    let row_bytes = checked_mul(n_in / 32, 34, "packed Q8 F32 R2C4K64 row bytes")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: u32::try_from(n_out).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed Q8 F32 output exceeds u32".into())
            })?,
            n: u32::try_from(n_tokens).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed Q8 F32 token count exceeds u32".into())
            })?,
            k: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed Q8 F32 input exceeds u32".into())
            })?,
            nb01: u32::try_from(row_bytes).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed Q8 F32 row bytes exceed u32".into())
            })?,
            stride_b: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed Q8 F32 stride exceeds u32".into())
            })?,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, input);
    enc.set_tensor(3, output);
    enc.set_threadgroup_memory(0, 4_096);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_out / 16,
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

fn encode_state_batch_projection(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_tokens: usize,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    if weight.dtype != GgmlType::Q8_0 {
        return encode_batch_projection(
            ctx, enc, weight, input, output, n_in, n_out, n_tokens, name,
        );
    }
    validate_matvec_weight(weight, n_in, n_out, name)?;
    crate::metal::encode_mat_vec_q8_0_batch_f32(
        ctx, enc, weight, input, output, n_in, n_out, n_tokens,
    )
    .map_err(DeepSeekV4MetalError::Metal)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HcBatchArgs {
    hidden_size: u32,
    n_tokens: u32,
}

impl PrefillHyperScratch {
    fn encode_initial_repeat(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        embeddings: &MetalTensor,
        residual: &MetalTensor,
        n_tokens: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_hc_repeat_batch")?;
        let n_tokens_u32 = checked_token_count(n_tokens)?;
        validate_f32(
            embeddings,
            &[DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            false,
            "packed embeddings",
        )?;
        validate_f32(
            residual,
            &[
                DEEPSEEK_V4_HIDDEN_SIZE as u64,
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
                n_tokens as u64,
            ],
            true,
            "packed initial residual",
        )?;
        let pso = ctx.pipeline("kernel_deepseek_v4_hc_repeat_batch")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(
            0,
            &HcBatchArgs {
                hidden_size: u32_hidden(DEEPSEEK_V4_HIDDEN_SIZE)?,
                n_tokens: n_tokens_u32,
            },
        );
        enc.set_tensor(1, embeddings);
        enc.set_tensor(2, residual);
        let total = checked_mul(
            n_tokens,
            residual_len(DEEPSEEK_V4_HIDDEN_SIZE)?,
            "packed repeated residual",
        )?;
        enc.dispatch(
            MTLSize {
                width: total.div_ceil(256),
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
    fn encode_pre(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residual: &MetalTensor,
        function: &MetalTensor,
        scale: &MetalTensor,
        base: &MetalTensor,
        n_tokens: usize,
        rms_eps: f32,
        hc_eps: f32,
    ) -> Result<MetalTensor, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_hc_pre_batch")?;
        let n_tokens_u32 = checked_token_count(n_tokens)?;
        validate_eps(rms_eps, "packed mHC RMSNorm epsilon")?;
        validate_eps(hc_eps, "packed mHC epsilon")?;
        let residual_width = residual_len(DEEPSEEK_V4_HIDDEN_SIZE)?;
        validate_f32(
            residual,
            &[
                DEEPSEEK_V4_HIDDEN_SIZE as u64,
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
                n_tokens as u64,
            ],
            false,
            "packed mHC residual",
        )?;
        validate_f32(
            function,
            &[residual_width as u64, DEEPSEEK_V4_HC_PARAMETER_COUNT as u64],
            false,
            "packed mHC function",
        )?;
        validate_f32(scale, &[3], false, "packed mHC scale")?;
        validate_f32(
            base,
            &[DEEPSEEK_V4_HC_PARAMETER_COUNT as u64],
            false,
            "packed mHC base",
        )?;
        let normalized = f32_prefix(
            &self.normalized,
            vec![residual_width as u64, n_tokens as u64],
            "packed mHC normalized",
        )?;
        let mixes = f32_prefix(
            &self.mixes,
            vec![DEEPSEEK_V4_HC_PARAMETER_COUNT as u64, n_tokens as u64],
            "packed mHC mixes",
        )?;
        let pre = f32_prefix(
            &self.pre,
            vec![DEEPSEEK_V4_CONNECTION_COUNT as u64, n_tokens as u64],
            "packed mHC pre gates",
        )?;
        let post = f32_prefix(
            &self.post,
            vec![DEEPSEEK_V4_CONNECTION_COUNT as u64, n_tokens as u64],
            "packed mHC post gates",
        )?;
        let combination = f32_prefix(
            &self.combination,
            vec![
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
                n_tokens as u64,
            ],
            "packed mHC combinations",
        )?;
        let collapsed = f32_prefix(
            &self.collapsed,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            "packed mHC collapsed",
        )?;
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            residual,
            &self.ones,
            &normalized,
            n_tokens,
            residual_width,
            rms_eps,
        )?;
        encode_batch_projection(
            ctx,
            enc,
            function,
            &normalized,
            &mixes,
            residual_width,
            DEEPSEEK_V4_HC_PARAMETER_COUNT,
            n_tokens,
            "packed mHC function",
        )?;
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct ControlsArgs {
            n_tokens: u32,
            eps: f32,
        }
        let pso = ctx.pipeline("kernel_deepseek_v4_hc_controls_batch")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(
            0,
            &ControlsArgs {
                n_tokens: n_tokens_u32,
                eps: hc_eps,
            },
        );
        enc.set_tensor(1, &mixes);
        enc.set_tensor(2, scale);
        enc.set_tensor(3, base);
        enc.set_tensor(4, &pre);
        enc.set_tensor(5, &post);
        enc.set_tensor(6, &combination);
        enc.dispatch(
            MTLSize {
                width: n_tokens,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );

        let pso = ctx.pipeline("kernel_deepseek_v4_hc_collapse_batch")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(
            0,
            &HcBatchArgs {
                hidden_size: u32_hidden(DEEPSEEK_V4_HIDDEN_SIZE)?,
                n_tokens: n_tokens_u32,
            },
        );
        enc.set_tensor(1, residual);
        enc.set_tensor(2, &pre);
        enc.set_tensor(3, &collapsed);
        let total = checked_mul(n_tokens, DEEPSEEK_V4_HIDDEN_SIZE, "packed mHC collapse")?;
        enc.dispatch(
            MTLSize {
                width: total.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(collapsed)
    }

    fn encode_post(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        block_output: &MetalTensor,
        residual: &MetalTensor,
        output: &MetalTensor,
        n_tokens: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_hc_post_batch")?;
        let n_tokens_u32 = checked_token_count(n_tokens)?;
        validate_f32(
            block_output,
            &[DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            false,
            "packed mHC block output",
        )?;
        let residual_shape = [
            DEEPSEEK_V4_HIDDEN_SIZE as u64,
            DEEPSEEK_V4_CONNECTION_COUNT as u64,
            n_tokens as u64,
        ];
        validate_f32(
            residual,
            &residual_shape,
            false,
            "packed mHC source residual",
        )?;
        validate_f32(output, &residual_shape, true, "packed mHC output residual")?;
        let post = f32_prefix(
            &self.post,
            vec![DEEPSEEK_V4_CONNECTION_COUNT as u64, n_tokens as u64],
            "packed mHC post gates",
        )?;
        let combination = f32_prefix(
            &self.combination,
            vec![
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
                n_tokens as u64,
            ],
            "packed mHC combinations",
        )?;
        let pso = ctx.pipeline("kernel_deepseek_v4_hc_post_batch")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(
            0,
            &HcBatchArgs {
                hidden_size: u32_hidden(DEEPSEEK_V4_HIDDEN_SIZE)?,
                n_tokens: n_tokens_u32,
            },
        );
        enc.set_tensor(1, block_output);
        enc.set_tensor(2, residual);
        enc.set_tensor(3, &post);
        enc.set_tensor(4, &combination);
        enc.set_tensor(5, output);
        let total = checked_mul(
            n_tokens,
            residual_len(DEEPSEEK_V4_HIDDEN_SIZE)?,
            "packed mHC post",
        )?;
        enc.dispatch(
            MTLSize {
                width: total.div_ceil(256),
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
}

struct PackedAttentionViews {
    normalized_input: MetalTensor,
    q_lora: MetalTensor,
    queries: MetalTensor,
    kv: MetalTensor,
    attention: MetalTensor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Q8PrecisionProjection {
    Exact,
    #[cfg(test)]
    HalfMatrix,
    F32Matrix,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PackedQ8MatrixPolicy {
    Auto,
    Exact,
    F32Matrix,
}

crate::env_flag!(
    default_on packed_q8_compressor_matrix_enabled,
    "QWEN_DSV4_PACKED_Q8_COMPRESSOR_MATRIX"
);

const PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE: &str = "Apple M4 Max";
const PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES: u64 = 104_202_502_492;

fn packed_q8_compressor_matrix_scope_qualified(
    device_name: &str,
    tensor_count: usize,
    source_bytes: u64,
    n_tokens: usize,
) -> bool {
    n_tokens == DEEPSEEK_V4_PREFILL_MAX_TOKENS
        && device_name == PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE
        && tensor_count == 1_328
        && source_bytes == PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES
}

fn packed_q8_compressor_matrix_for_chunk(
    ctx: &MetalContext,
    residency: &DeepSeekV4MetalResidency,
    n_tokens: usize,
) -> bool {
    packed_q8_compressor_matrix_enabled()
        && packed_q8_compressor_matrix_scope_qualified(
            &ctx.device.name().to_string(),
            residency.report().tensor_count,
            residency.report().source_bytes,
            n_tokens,
        )
}

fn parse_packed_q8_qb_policy(
    value: Option<&str>,
) -> Result<PackedQ8MatrixPolicy, DeepSeekV4MetalError> {
    match value {
        None | Some("auto") => Ok(PackedQ8MatrixPolicy::Auto),
        Some("exact") => Ok(PackedQ8MatrixPolicy::Exact),
        Some("f32_matrix") => Ok(PackedQ8MatrixPolicy::F32Matrix),
        Some(value) => invalid(format!(
            "QWEN_DSV4_PACKED_Q8_QB must be auto, exact, or f32_matrix, got {value:?}"
        )),
    }
}

fn parse_packed_q8_output_policy(
    value: Option<&str>,
) -> Result<PackedQ8MatrixPolicy, DeepSeekV4MetalError> {
    match value {
        None | Some("auto") => Ok(PackedQ8MatrixPolicy::Auto),
        Some("exact") => Ok(PackedQ8MatrixPolicy::Exact),
        Some("f32_matrix") => Ok(PackedQ8MatrixPolicy::F32Matrix),
        Some(value) => invalid(format!(
            "QWEN_DSV4_PACKED_Q8_OUTPUT must be auto, exact, or f32_matrix, got {value:?}"
        )),
    }
}

fn packed_q8_output_projection_for_chunk(
    ctx: &MetalContext,
    residency: &DeepSeekV4MetalResidency,
    n_tokens: usize,
) -> Result<Q8PrecisionProjection, DeepSeekV4MetalError> {
    let value = std::env::var("QWEN_DSV4_PACKED_Q8_OUTPUT").ok();
    let policy = parse_packed_q8_output_policy(value.as_deref())?;
    let profile_qualified = packed_q8_compressor_matrix_scope_qualified(
        &ctx.device.name().to_string(),
        residency.report().tensor_count,
        residency.report().source_bytes,
        n_tokens,
    );
    Ok(resolve_packed_q8_matrix_policy(policy, profile_qualified))
}

fn resolve_packed_q8_matrix_policy(
    policy: PackedQ8MatrixPolicy,
    profile_qualified: bool,
) -> Q8PrecisionProjection {
    match policy {
        PackedQ8MatrixPolicy::Auto if profile_qualified => Q8PrecisionProjection::F32Matrix,
        PackedQ8MatrixPolicy::Auto | PackedQ8MatrixPolicy::Exact => Q8PrecisionProjection::Exact,
        PackedQ8MatrixPolicy::F32Matrix => Q8PrecisionProjection::F32Matrix,
    }
}

fn packed_q8_qb_projection_for_chunk(
    ctx: &MetalContext,
    residency: &DeepSeekV4MetalResidency,
    n_tokens: usize,
) -> Result<Q8PrecisionProjection, DeepSeekV4MetalError> {
    let value = std::env::var("QWEN_DSV4_PACKED_Q8_QB").ok();
    let policy = parse_packed_q8_qb_policy(value.as_deref())?;
    let profile_qualified = packed_q8_compressor_matrix_scope_qualified(
        &ctx.device.name().to_string(),
        residency.report().tensor_count,
        residency.report().source_bytes,
        n_tokens,
    );
    Ok(resolve_packed_q8_matrix_policy(policy, profile_qualified))
}

impl Q8PrecisionProjection {
    fn uses_full_chunk_f32(self, n_tokens: usize) -> bool {
        self == Self::F32Matrix && n_tokens == DEEPSEEK_V4_PREFILL_MAX_TOKENS
    }
}

#[derive(Clone)]
struct PackedSparseCsaViews {
    query_offset: usize,
    query_count: usize,
    cache_order_ids: MetalTensor,
    selected_counts: MetalTensor,
    visible_counts: MetalTensor,
    index_queries: MetalTensor,
    head_weights: MetalTensor,
    scores: MetalTensor,
    selected_mask: MetalTensor,
    status: MetalTensor,
}

#[derive(Clone, Copy)]
struct PackedCsaSelectionView<'a> {
    query_offset: usize,
    query_count: usize,
    cache_order_ids: &'a MetalTensor,
    selected_counts: &'a MetalTensor,
    visible_counts: &'a MetalTensor,
}

impl PackedSparseCsaViews {
    fn selection_view(&self) -> PackedCsaSelectionView<'_> {
        PackedCsaSelectionView {
            query_offset: self.query_offset,
            query_count: self.query_count,
            cache_order_ids: &self.cache_order_ids,
            selected_counts: &self.selected_counts,
            visible_counts: &self.visible_counts,
        }
    }
}

fn csa_visible_rows(position: u32) -> usize {
    (u64::from(position) + 1) as usize / 4
}

fn sparse_csa_query_offset(start_position: u32, n_tokens: usize) -> Option<usize> {
    (0..n_tokens).find(|&token| {
        let position = u64::from(start_position) + token as u64;
        (position + 1) / 4 > DEEPSEEK_V4_CSA_TOP_K as u64
    })
}

#[cfg(feature = "dsv4-diagnostics")]
fn validate_fp4_selection_counterfactual_packed(
    start_position: u32,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let sparse_query_count = sparse_csa_query_offset(start_position, n_tokens)
        .map(|offset| n_tokens - offset)
        .unwrap_or(0);
    if sparse_query_count > 1 {
        return invalid(format!(
            "FP4 selection counterfactual requires at most one packed sparse query, got {sparse_query_count}"
        ));
    }
    Ok(())
}

fn packed_sparse_visible_counts(
    start_position: u32,
    query_offset: usize,
    n_tokens: usize,
    final_rows: usize,
) -> Result<Vec<i32>, DeepSeekV4MetalError> {
    if query_offset >= n_tokens || final_rows <= DEEPSEEK_V4_CSA_TOP_K {
        return invalid(format!(
            "packed sparse visibility geometry is invalid: offset={query_offset} tokens={n_tokens} final_rows={final_rows}"
        ));
    }
    let final_rows_i32 = i32::try_from(final_rows).map_err(|_| {
        DeepSeekV4MetalError::Invalid("packed sparse final row count exceeds i32".into())
    })?;
    let visible = (query_offset..n_tokens)
        .map(|token| {
            let position = start_position
                .checked_add(u32::try_from(token).map_err(|_| {
                    DeepSeekV4MetalError::Invalid(
                        "packed sparse token offset exceeds u32".into(),
                    )
                })?)
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid("packed sparse position overflow".into())
                })?;
            let count = csa_visible_rows(position);
            if count <= DEEPSEEK_V4_CSA_TOP_K || count > final_rows {
                return invalid(format!(
                    "packed sparse token {token} sees {count} rows outside 513..={final_rows} final rows"
                ));
            }
            i32::try_from(count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed sparse visible count exceeds i32".into())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if visible.last().copied() != Some(final_rows_i32) {
        return invalid(format!(
            "packed sparse final visibility {:?} differs from published row count {final_rows}",
            visible.last()
        ));
    }
    Ok(visible)
}

fn tiled_hca_query_offset(start_position: u32, n_tokens: usize) -> Option<usize> {
    (0..n_tokens).find(|&token| {
        let position = u64::from(start_position) + token as u64;
        (position + 1) / 128 > DEEPSEEK_V4_HCA_TILE_ROWS as u64
    })
}

impl PrefillSparseCsaScratch {
    #[cfg(not(feature = "dsv4-diagnostics"))]
    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        q_lora: &MetalTensor,
        normalized_input: &MetalTensor,
        indexer_q_weight: &MetalTensor,
        indexer_projection: &MetalTensor,
        rows: DeepSeekV4CsaRows<'_>,
        start_position: u32,
        query_offset: usize,
        n_tokens: usize,
        rope: DeepSeekV4RopeParameters,
    ) -> Result<PackedSparseCsaViews, DeepSeekV4MetalError> {
        let prepared = self.encode_prepare(
            ctx,
            enc,
            q_lora,
            normalized_input,
            indexer_q_weight,
            indexer_projection,
            rows,
            start_position,
            query_offset,
            n_tokens,
            rope,
        )?;
        self.encode_f16_score_and_select(ctx, enc, rows, &prepared)?;
        Ok(prepared)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_prepare(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        q_lora: &MetalTensor,
        normalized_input: &MetalTensor,
        indexer_q_weight: &MetalTensor,
        indexer_projection: &MetalTensor,
        rows: DeepSeekV4CsaRows<'_>,
        start_position: u32,
        query_offset: usize,
        n_tokens: usize,
        rope: DeepSeekV4RopeParameters,
    ) -> Result<PackedSparseCsaViews, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_packed_sparse_csa_indexer")?;
        checked_token_count(n_tokens)?;
        if query_offset >= n_tokens
            || rows.count <= DEEPSEEK_V4_CSA_TOP_K
            || rows.count > rows.capacity_rows
            || rows.capacity_rows != self.capacity_rows
        {
            return invalid(format!(
                "packed sparse CSA geometry is invalid: offset={query_offset} tokens={n_tokens} rows={}/{}",
                rows.count, rows.capacity_rows
            ));
        }
        validate_f32(
            q_lora,
            &[1_024, n_tokens as u64],
            false,
            "packed sparse CSA Q-LoRA input",
        )?;
        validate_f32(
            normalized_input,
            &[DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            false,
            "packed sparse CSA normalized input",
        )?;
        validate_matvec_weight(
            indexer_q_weight,
            1_024,
            INDEXER_QUERY_WIDTH,
            "packed indexer Q weight",
        )?;
        validate_matvec_weight(
            indexer_projection,
            DEEPSEEK_V4_HIDDEN_SIZE,
            INDEXER_HEAD_COUNT,
            "packed indexer projection weight",
        )?;
        validate_f16(
            rows.indexer_cache,
            &[INDEXER_HEAD_DIM as u64, rows.capacity_rows as u64],
            false,
            "packed sparse CSA indexer cache",
        )?;

        let query_count = n_tokens - query_offset;
        let normalized_suffix = normalized_input.view_subrange(
            checked_mul(
                query_offset,
                DEEPSEEK_V4_HIDDEN_SIZE,
                "packed sparse normalized-input offset",
            )? as u64,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, query_count as u64],
        );
        let index_query_storage = f32_prefix(
            &self.index_queries,
            vec![
                INDEXER_HEAD_DIM as u64,
                INDEXER_HEAD_COUNT as u64,
                n_tokens as u64,
            ],
            "packed sparse index query storage",
        )?;
        let head_weights = f32_prefix(
            &self.head_weights,
            vec![INDEXER_HEAD_COUNT as u64, query_count as u64],
            "packed sparse head weights",
        )?;
        let visible_counts = i32_prefix(
            &self.visible_counts,
            vec![query_count as u64],
            "packed sparse visible counts",
        )?;
        let scores = f32_prefix(
            &self.scores,
            vec![rows.capacity_rows as u64, query_count as u64],
            "packed sparse scores",
        )?;
        let selected_mask = i32_prefix(
            &self.selected_mask,
            vec![rows.capacity_rows as u64, query_count as u64],
            "packed sparse selection mask",
        )?;
        let cache_order_ids = i32_prefix(
            &self.cache_order_ids,
            vec![DEEPSEEK_V4_CSA_TOP_K as u64, query_count as u64],
            "packed sparse cache-order IDs",
        )?;
        let selected_counts = i32_prefix(
            &self.selected_counts,
            vec![query_count as u64],
            "packed sparse selected counts",
        )?;
        let status = i32_prefix(
            &self.status,
            vec![query_count as u64],
            "packed sparse selection status",
        )?;

        let visible =
            packed_sparse_visible_counts(start_position, query_offset, n_tokens, rows.count)?;
        host_write_i32(
            &visible_counts,
            &visible,
            "packed sparse CSA visible counts",
        )?;

        let index_queries = if packed_indexer_q_matrix_enabled() {
            let matrix_output = index_query_storage
                .view_subrange(0, vec![INDEXER_QUERY_WIDTH as u64, n_tokens as u64]);
            encode_q8_f32_mma_r2c4k64(
                ctx,
                enc,
                indexer_q_weight,
                q_lora,
                &matrix_output,
                1_024,
                INDEXER_QUERY_WIDTH,
                n_tokens,
            )?;
            index_query_storage.view_subrange(
                checked_mul(
                    query_offset,
                    INDEXER_QUERY_WIDTH,
                    "packed sparse matrix-query offset",
                )? as u64,
                vec![
                    INDEXER_HEAD_DIM as u64,
                    INDEXER_HEAD_COUNT as u64,
                    query_count as u64,
                ],
            )
        } else {
            let q_lora_suffix = q_lora.view_subrange(
                checked_mul(query_offset, 1_024, "packed sparse Q-LoRA offset")? as u64,
                vec![1_024, query_count as u64],
            );
            let index_queries = index_query_storage.view_subrange(
                0,
                vec![
                    INDEXER_HEAD_DIM as u64,
                    INDEXER_HEAD_COUNT as u64,
                    query_count as u64,
                ],
            );
            encode_batch_projection(
                ctx,
                enc,
                indexer_q_weight,
                &q_lora_suffix,
                &index_queries,
                1_024,
                INDEXER_QUERY_WIDTH,
                query_count,
                "packed indexer Q",
            )?;
            index_queries
        };
        let batched_indexer_rope = packed_indexer_batched_rope_enabled();
        static POLICY_LOGGED: std::sync::Once = std::sync::Once::new();
        POLICY_LOGGED.call_once(|| {
            eprintln!(
                "deepseek_v4: packed indexer query RoPE policy={}; rollback=QWEN_DSV4_PACKED_INDEXER_BATCHED_ROPE=0",
                if batched_indexer_rope {
                    "batched"
                } else {
                    "scalar"
                },
            );
        });
        if batched_indexer_rope {
            let first_position = start_position
                .checked_add(u32::try_from(query_offset).map_err(|_| {
                    DeepSeekV4MetalError::Invalid("packed sparse query offset exceeds u32".into())
                })?)
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "packed sparse query start position overflow".into(),
                    )
                })?;
            encode_ds4_rope_tail_adjacent_batch_in_place(
                ctx,
                enc,
                &index_queries,
                first_position,
                query_count,
                1,
                rope,
                false,
            )?;
        } else {
            for local in 0..query_count {
                let query = f32_row(
                    &index_queries,
                    local,
                    INDEXER_QUERY_WIDTH,
                    vec![INDEXER_HEAD_DIM as u64, INDEXER_HEAD_COUNT as u64],
                    "packed sparse index query row",
                )?;
                let token = query_offset + local;
                let position = start_position
                    .checked_add(u32::try_from(token).map_err(|_| {
                        DeepSeekV4MetalError::Invalid(
                            "packed sparse token offset exceeds u32".into(),
                        )
                    })?)
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid("packed sparse position overflow".into())
                    })?;
                encode_ds4_rope_tail_adjacent_in_place(ctx, enc, &query, position, rope, false)?;
            }
        }
        encode_hadamard_128_rows_in_place(
            ctx,
            enc,
            &index_queries,
            checked_mul(
                query_count,
                INDEXER_HEAD_COUNT,
                "packed indexer Hadamard rows",
            )?,
        )?;
        encode_batch_projection(
            ctx,
            enc,
            indexer_projection,
            &normalized_suffix,
            &head_weights,
            DEEPSEEK_V4_HIDDEN_SIZE,
            INDEXER_HEAD_COUNT,
            query_count,
            "packed indexer head weights",
        )?;
        encode_scale_f32_in_place(
            ctx,
            enc,
            &head_weights,
            1.0 / (INDEXER_HEAD_COUNT as f32 * INDEXER_HEAD_DIM as f32).sqrt(),
            "packed indexer head weights",
        )?;
        Ok(PackedSparseCsaViews {
            query_offset,
            query_count,
            cache_order_ids,
            selected_counts,
            visible_counts,
            index_queries,
            head_weights,
            scores,
            selected_mask,
            status,
        })
    }

    #[cfg(not(feature = "dsv4-diagnostics"))]
    fn encode_f16_score_and_select(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        rows: DeepSeekV4CsaRows<'_>,
        prepared: &PackedSparseCsaViews,
    ) -> Result<(), DeepSeekV4MetalError> {
        self.encode_f16_scores(ctx, enc, rows, prepared)?;
        self.encode_f16_selection(ctx, enc, rows, prepared)
    }

    fn encode_f16_scores(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        rows: DeepSeekV4CsaRows<'_>,
        prepared: &PackedSparseCsaViews,
    ) -> Result<(), DeepSeekV4MetalError> {
        encode_lightning_indexer_scores_f16(
            ctx,
            enc,
            &prepared.index_queries,
            &prepared.head_weights,
            rows.indexer_cache,
            &prepared.visible_counts,
            &prepared.scores,
            INDEXER_HEAD_COUNT,
            INDEXER_HEAD_DIM,
            rows.capacity_rows,
            prepared.query_count,
        )
    }

    fn encode_f16_selection(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        rows: DeepSeekV4CsaRows<'_>,
        prepared: &PackedSparseCsaViews,
    ) -> Result<(), DeepSeekV4MetalError> {
        encode_select_top_k_f32(
            ctx,
            enc,
            &prepared.scores,
            &prepared.visible_counts,
            &prepared.selected_mask,
            None,
            &prepared.cache_order_ids,
            &prepared.selected_counts,
            &prepared.status,
            rows.capacity_rows,
            rows.count,
            DEEPSEEK_V4_CSA_TOP_K,
            prepared.query_count,
        )
    }

    fn validate_completed(&self, query_count: usize) -> Result<(), DeepSeekV4MetalError> {
        checked_token_count(query_count)?;
        let status = i32_prefix(
            &self.status,
            vec![query_count as u64],
            "packed sparse selection status",
        )?;
        let selected_counts = i32_prefix(
            &self.selected_counts,
            vec![query_count as u64],
            "packed sparse selected counts",
        )?;
        let statuses = host_read_i32(&status, "packed sparse selection status")?;
        let counts = host_read_i32(&selected_counts, "packed sparse selected counts")?;
        if let Some(query) = statuses
            .iter()
            .zip(&counts)
            .position(|(&status, &count)| status != 0 || count != DEEPSEEK_V4_CSA_TOP_K as i32)
        {
            return invalid(format!(
                "packed sparse CSA query {query} selection failed with status={} count={}",
                statuses[query], counts[query]
            ));
        }
        Ok(())
    }
}

impl PrefillAttentionScratch {
    #[allow(clippy::too_many_arguments)]
    fn encode_prepare(
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
        n_tokens: usize,
        rms_eps: f32,
        q_b_projection: Q8PrecisionProjection,
    ) -> Result<PackedAttentionViews, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_attention_prepare_batch")?;
        checked_token_count(n_tokens)?;
        validate_eps(rms_eps, "packed attention RMSNorm epsilon")?;
        let config = deepseek_v4_session_attention_config();
        let dims = config.checked()?;
        validate_f32(
            input,
            &[config.hidden_size as u64, n_tokens as u64],
            false,
            "packed attention input",
        )?;
        validate_f32(
            attention_norm,
            &[config.hidden_size as u64],
            false,
            "packed attention norm weight",
        )?;
        validate_f32(
            q_a_norm,
            &[config.q_lora_rank as u64],
            false,
            "packed Q A norm weight",
        )?;
        validate_f32(
            kv_norm,
            &[config.head_dim as u64],
            false,
            "packed KV norm weight",
        )?;
        let normalized_input = f32_prefix(
            &self.normalized_input,
            vec![config.hidden_size as u64, n_tokens as u64],
            "packed attention normalized input",
        )?;
        let q_lora_raw = f32_prefix(
            &self.q_lora_raw,
            vec![config.q_lora_rank as u64, n_tokens as u64],
            "packed raw Q LoRA",
        )?;
        let q_lora = f32_prefix(
            &self.q_lora,
            vec![config.q_lora_rank as u64, n_tokens as u64],
            "packed Q LoRA",
        )?;
        let queries_raw = f32_prefix(
            &self.queries_raw,
            vec![dims.query_width as u64, n_tokens as u64],
            "packed raw queries",
        )?;
        let queries = f32_prefix(
            &self.queries,
            vec![dims.query_width as u64, n_tokens as u64],
            "packed queries",
        )?;
        let kv_raw = f32_prefix(
            &self.kv_raw,
            vec![config.head_dim as u64, n_tokens as u64],
            "packed raw KV",
        )?;
        let kv = f32_prefix(
            &self.kv,
            vec![config.head_dim as u64, n_tokens as u64],
            "packed KV",
        )?;
        let attention = f32_prefix(
            &self.attention,
            vec![dims.query_width as u64, n_tokens as u64],
            "packed attention heads",
        )?;

        encode_rms_norm_batched_f32(
            ctx,
            enc,
            input,
            attention_norm,
            &normalized_input,
            n_tokens,
            config.hidden_size,
            rms_eps,
        )?;
        encode_batch_projection(
            ctx,
            enc,
            q_a,
            &normalized_input,
            &q_lora_raw,
            config.hidden_size,
            config.q_lora_rank,
            n_tokens,
            "packed Q A",
        )?;
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            &q_lora_raw,
            q_a_norm,
            &q_lora,
            n_tokens,
            config.q_lora_rank,
            rms_eps,
        )?;
        if q_b_projection.uses_full_chunk_f32(n_tokens) {
            encode_q8_f32_mma_r2c4k64(
                ctx,
                enc,
                q_b,
                &q_lora,
                &queries_raw,
                config.q_lora_rank,
                dims.query_width,
                n_tokens,
            )?;
        } else {
            encode_batch_projection(
                ctx,
                enc,
                q_b,
                &q_lora,
                &queries_raw,
                config.q_lora_rank,
                dims.query_width,
                n_tokens,
                "packed Q B",
            )?;
        }
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            &queries_raw,
            &self.head_norm_ones,
            &queries,
            checked_mul(n_tokens, config.head_count, "packed query rows")?,
            config.head_dim,
            rms_eps,
        )?;
        encode_state_batch_projection(
            ctx,
            enc,
            kv_weight,
            &normalized_input,
            &kv_raw,
            config.hidden_size,
            config.head_dim,
            n_tokens,
            "packed KV",
        )?;
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            &kv_raw,
            kv_norm,
            &kv,
            n_tokens,
            config.head_dim,
            rms_eps,
        )?;
        Ok(PackedAttentionViews {
            normalized_input,
            q_lora,
            queries,
            kv,
            attention,
        })
    }

    fn encode_output(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        attention: &MetalTensor,
        output_a: &MetalTensor,
        output_b: &MetalTensor,
        n_tokens: usize,
    ) -> Result<MetalTensor, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_attention_output_batch")?;
        checked_token_count(n_tokens)?;
        let config = deepseek_v4_session_attention_config();
        let dims = config.checked()?;
        validate_f32(
            attention,
            &[dims.query_width as u64, n_tokens as u64],
            false,
            "packed attention heads",
        )?;
        validate_matvec_weight(
            output_a,
            dims.group_width,
            dims.low_rank_width,
            "packed output A",
        )?;
        validate_matvec_weight(
            output_b,
            dims.low_rank_width,
            config.hidden_size,
            "packed output B",
        )?;
        let low_rank = f32_prefix(
            &self.low_rank,
            vec![dims.low_rank_width as u64, n_tokens as u64],
            "packed low-rank attention",
        )?;
        let output = f32_prefix(
            &self.output,
            vec![config.hidden_size as u64, n_tokens as u64],
            "packed attention output",
        )?;
        let group_input = f32_prefix(
            &self.group_input,
            vec![dims.group_width as u64, n_tokens as u64],
            "packed attention group input",
        )?;
        let group_output = f32_prefix(
            &self.group_output,
            vec![config.output_rank as u64, n_tokens as u64],
            "packed attention group output",
        )?;
        for group in 0..config.group_count {
            encode_group_pack(
                ctx,
                enc,
                attention,
                &group_input,
                n_tokens,
                dims.query_width,
                dims.group_width,
                group,
                false,
            )?;
            let weight = group_weight_view(output_a, dims.group_width, config.output_rank, group)?;
            encode_batch_projection(
                ctx,
                enc,
                &weight,
                &group_input,
                &group_output,
                dims.group_width,
                config.output_rank,
                n_tokens,
                "packed grouped output A",
            )?;
            encode_group_pack(
                ctx,
                enc,
                &group_output,
                &low_rank,
                n_tokens,
                dims.low_rank_width,
                config.output_rank,
                group,
                true,
            )?;
        }
        encode_batch_projection(
            ctx,
            enc,
            output_b,
            &low_rank,
            &output,
            dims.low_rank_width,
            config.hidden_size,
            n_tokens,
            "packed output B",
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_output_q8_precision(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        attention: &MetalTensor,
        output_a: &MetalTensor,
        output_b: &MetalTensor,
        n_tokens: usize,
        output_a_projection: Q8PrecisionProjection,
        output_b_projection: Q8PrecisionProjection,
    ) -> Result<MetalTensor, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_Q8_precision_output_batch")?;
        checked_token_count(n_tokens)?;
        let config = deepseek_v4_session_attention_config();
        let dims = config.checked()?;
        validate_f32(
            attention,
            &[dims.query_width as u64, n_tokens as u64],
            false,
            "Q8 precision attention heads",
        )?;
        for (weight, n_in, n_out, name) in [
            (
                output_a,
                dims.group_width,
                dims.low_rank_width,
                "Q8 precision output A",
            ),
            (
                output_b,
                dims.low_rank_width,
                config.hidden_size,
                "Q8 precision output B",
            ),
        ] {
            validate_matvec_weight(weight, n_in, n_out, name)?;
            if weight.dtype != GgmlType::Q8_0 {
                return invalid(format!("{name} must be Q8_0, got {:?}", weight.dtype));
            }
        }
        let low_rank = f32_prefix(
            &self.low_rank,
            vec![dims.low_rank_width as u64, n_tokens as u64],
            "Q8 precision low rank",
        )?;
        let output = f32_prefix(
            &self.output,
            vec![config.hidden_size as u64, n_tokens as u64],
            "Q8 precision output",
        )?;
        let group_input = f32_prefix(
            &self.group_input,
            vec![dims.group_width as u64, n_tokens as u64],
            "Q8 precision group input",
        )?;
        let group_output = f32_prefix(
            &self.group_output,
            vec![config.output_rank as u64, n_tokens as u64],
            "Q8 precision group output",
        )?;
        let encode_projection = |weight: &MetalTensor,
                                 input: &MetalTensor,
                                 output: &MetalTensor,
                                 n_in: usize,
                                 n_out: usize,
                                 projection: Q8PrecisionProjection,
                                 name: &str| {
            match projection {
                Q8PrecisionProjection::Exact => encode_batch_projection(
                    ctx, enc, weight, input, output, n_in, n_out, n_tokens, name,
                ),
                #[cfg(test)]
                Q8PrecisionProjection::HalfMatrix => crate::metal::encode_mat_mat_q8_0_f32(
                    ctx, enc, weight, input, output, n_in, n_out, n_tokens,
                )
                .map_err(DeepSeekV4MetalError::Metal),
                Q8PrecisionProjection::F32Matrix => encode_q8_f32_mma_r2c4k64(
                    ctx, enc, weight, input, output, n_in, n_out, n_tokens,
                ),
            }
        };
        for group in 0..config.group_count {
            encode_group_pack(
                ctx,
                enc,
                attention,
                &group_input,
                n_tokens,
                dims.query_width,
                dims.group_width,
                group,
                false,
            )?;
            let weight = group_weight_view(output_a, dims.group_width, config.output_rank, group)?;
            encode_projection(
                &weight,
                &group_input,
                &group_output,
                dims.group_width,
                config.output_rank,
                output_a_projection,
                "Q8 precision grouped output A",
            )?;
            encode_group_pack(
                ctx,
                enc,
                &group_output,
                &low_rank,
                n_tokens,
                dims.low_rank_width,
                config.output_rank,
                group,
                true,
            )?;
        }
        encode_projection(
            output_b,
            &low_rank,
            &output,
            dims.low_rank_width,
            config.hidden_size,
            output_b_projection,
            "Q8 precision output B",
        )?;
        Ok(output)
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_group_pack(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    output: &MetalTensor,
    n_tokens: usize,
    row_width: usize,
    group_width: usize,
    group: usize,
    scatter: bool,
) -> Result<(), DeepSeekV4MetalError> {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_tokens: u32,
        row_width: u32,
        group_width: u32,
        group: u32,
    }
    let n_tokens_u32 = checked_token_count(n_tokens)?;
    let required_input = if scatter {
        checked_mul(n_tokens, group_width, "group scatter input")?
    } else {
        checked_mul(n_tokens, row_width, "group pack input")?
    };
    let required_output = if scatter {
        checked_mul(n_tokens, row_width, "group scatter output")?
    } else {
        checked_mul(n_tokens, group_width, "group pack output")?
    };
    if input.dtype != GgmlType::F32
        || output.dtype != GgmlType::F32
        || input.n_elements() != required_input as u64
        || output.n_elements() != required_output as u64
        || !output.is_writable()
        || group
            .checked_add(1)
            .and_then(|count| count.checked_mul(group_width))
            .is_none_or(|end| end > row_width)
    {
        return invalid("packed attention group copy has invalid geometry");
    }
    let kernel = if scatter {
        "kernel_deepseek_v4_scatter_low_rank_group"
    } else {
        "kernel_deepseek_v4_pack_attention_group"
    };
    let pso = ctx.pipeline(kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_tokens: n_tokens_u32,
            row_width: u32::try_from(row_width)
                .map_err(|_| DeepSeekV4MetalError::Invalid("group row width exceeds u32".into()))?,
            group_width: u32::try_from(group_width)
                .map_err(|_| DeepSeekV4MetalError::Invalid("group width exceeds u32".into()))?,
            group: u32::try_from(group)
                .map_err(|_| DeepSeekV4MetalError::Invalid("group index exceeds u32".into()))?,
        },
    );
    enc.set_tensor(1, input);
    enc.set_tensor(2, output);
    let total = checked_mul(n_tokens, group_width, "group copy elements")?;
    enc.dispatch(
        MTLSize {
            width: total.div_ceil(256),
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

enum PackedCompressorViews {
    SlidingWindow,
    CompressedSparse {
        attention_kv: MetalTensor,
        attention_score: MetalTensor,
        indexer_kv: MetalTensor,
        indexer_score: MetalTensor,
    },
    HeavilyCompressed {
        attention_kv: MetalTensor,
        attention_score: MetalTensor,
    },
}

impl PrefillCompressorScratch {
    #[cfg(feature = "dsv4-diagnostics")]
    fn reset_q8_matrix_invocations(&self) {
        self.q8_matrix_invocations.set(0);
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn q8_matrix_invocations(&self) -> u32 {
        self.q8_matrix_invocations.get()
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_projection(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        weight: &MetalTensor,
        input: &MetalTensor,
        output: &MetalTensor,
        n_out: usize,
        n_tokens: usize,
        name: &str,
        use_matrix: bool,
    ) -> Result<(), DeepSeekV4MetalError> {
        if !use_matrix || weight.dtype != GgmlType::Q8_0 {
            return encode_state_batch_projection(
                ctx,
                enc,
                weight,
                input,
                output,
                DEEPSEEK_V4_HIDDEN_SIZE,
                n_out,
                n_tokens,
                name,
            );
        }
        #[cfg(feature = "dsv4-diagnostics")]
        let next = self
            .q8_matrix_invocations
            .get()
            .checked_add(1)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "packed Q8 compressor matrix invocation count overflow".into(),
                )
            })?;
        crate::metal::encode_mat_mat_q8_0_f32(
            ctx,
            enc,
            weight,
            input,
            output,
            DEEPSEEK_V4_HIDDEN_SIZE,
            n_out,
            n_tokens,
        )
        .map_err(DeepSeekV4MetalError::Metal)?;
        #[cfg(feature = "dsv4-diagnostics")]
        self.q8_matrix_invocations.set(next);
        Ok(())
    }

    fn encode_layer_projections(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residency: &DeepSeekV4MetalResidency,
        layer: usize,
        normalized_input: &MetalTensor,
        n_tokens: usize,
        use_matrix: bool,
    ) -> Result<PackedCompressorViews, DeepSeekV4MetalError> {
        let tensor = |suffix: &str| residency.require_tensor(&format!("blk.{layer}.{suffix}"));
        match residency
            .config()
            .attention_kinds
            .get(layer)
            .copied()
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "packed compressor layer {layer} is out of range"
                ))
            })? {
            AttentionKind::SlidingWindow => Ok(PackedCompressorViews::SlidingWindow),
            AttentionKind::CompressedSparse => {
                let attention_kv = f32_prefix(
                    &self.attention_kv,
                    vec![COMPRESSOR_ATTENTION_WIDTH as u64, n_tokens as u64],
                    "packed CSA compressor KV",
                )?;
                let attention_score = f32_prefix(
                    &self.attention_score,
                    vec![COMPRESSOR_ATTENTION_WIDTH as u64, n_tokens as u64],
                    "packed CSA compressor score",
                )?;
                let indexer_kv = f32_prefix(
                    &self.indexer_kv,
                    vec![COMPRESSOR_INDEXER_WIDTH as u64, n_tokens as u64],
                    "packed indexer compressor KV",
                )?;
                let indexer_score = f32_prefix(
                    &self.indexer_score,
                    vec![COMPRESSOR_INDEXER_WIDTH as u64, n_tokens as u64],
                    "packed indexer compressor score",
                )?;
                for (weight, output, width, name) in [
                    (
                        tensor("attn_compressor_kv.weight")?,
                        &attention_kv,
                        COMPRESSOR_ATTENTION_WIDTH,
                        "packed CSA compressor KV",
                    ),
                    (
                        tensor("attn_compressor_gate.weight")?,
                        &attention_score,
                        COMPRESSOR_ATTENTION_WIDTH,
                        "packed CSA compressor score",
                    ),
                    (
                        tensor("indexer_compressor_kv.weight")?,
                        &indexer_kv,
                        COMPRESSOR_INDEXER_WIDTH,
                        "packed indexer compressor KV",
                    ),
                    (
                        tensor("indexer_compressor_gate.weight")?,
                        &indexer_score,
                        COMPRESSOR_INDEXER_WIDTH,
                        "packed indexer compressor score",
                    ),
                ] {
                    self.encode_projection(
                        ctx,
                        enc,
                        weight,
                        normalized_input,
                        output,
                        width,
                        n_tokens,
                        name,
                        use_matrix,
                    )?;
                }
                Ok(PackedCompressorViews::CompressedSparse {
                    attention_kv,
                    attention_score,
                    indexer_kv,
                    indexer_score,
                })
            }
            AttentionKind::HeavilyCompressed => {
                let attention_kv = f32_prefix(
                    &self.hca_kv,
                    vec![512, n_tokens as u64],
                    "packed HCA compressor KV",
                )?;
                let attention_score = f32_prefix(
                    &self.hca_score,
                    vec![512, n_tokens as u64],
                    "packed HCA compressor score",
                )?;
                self.encode_projection(
                    ctx,
                    enc,
                    tensor("attn_compressor_kv.weight")?,
                    normalized_input,
                    &attention_kv,
                    512,
                    n_tokens,
                    "packed HCA compressor KV",
                    use_matrix,
                )?;
                self.encode_projection(
                    ctx,
                    enc,
                    tensor("attn_compressor_gate.weight")?,
                    normalized_input,
                    &attention_score,
                    512,
                    n_tokens,
                    "packed HCA compressor score",
                    use_matrix,
                )?;
                Ok(PackedCompressorViews::HeavilyCompressed {
                    attention_kv,
                    attention_score,
                })
            }
        }
    }
}

impl DeepSeekV4CompressorFrontiers {
    #[allow(clippy::too_many_arguments)]
    fn encode_layer_projected_chunk(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residency: &DeepSeekV4MetalResidency,
        layer: usize,
        start_position: u32,
        row_count: usize,
        projected: &PackedCompressorViews,
        scratch: &PrefillCompressorScratch,
        rope: DeepSeekV4RopeParameters,
        rms_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let tensor = |suffix: &str| residency.require_tensor(&format!("blk.{layer}.{suffix}"));
        let frontier = self.layers.get(layer).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "packed compressor layer {layer} is out of range"
            ))
        })?;
        match (frontier, projected) {
            (
                DeepSeekV4LayerCompressorFrontiers::SlidingWindow,
                PackedCompressorViews::SlidingWindow,
            ) => Ok(()),
            (
                DeepSeekV4LayerCompressorFrontiers::CompressedSparse { attention, indexer },
                PackedCompressorViews::CompressedSparse {
                    attention_kv,
                    attention_score,
                    indexer_kv,
                    indexer_score,
                },
            ) => {
                attention.encode_projected_chunk(
                    ctx,
                    enc,
                    attention_kv,
                    attention_score,
                    tensor("attn_compressor_ape.weight")?,
                    tensor("attn_compressor_norm.weight")?,
                    &scratch.pooled_rows,
                    &scratch.normalized_rows,
                    start_position,
                    row_count,
                    rope,
                    rms_eps,
                )?;
                indexer.encode_projected_chunk(
                    ctx,
                    enc,
                    indexer_kv,
                    indexer_score,
                    tensor("indexer_compressor_ape.weight")?,
                    tensor("indexer_compressor_norm.weight")?,
                    &scratch.pooled_rows,
                    &scratch.normalized_rows,
                    start_position,
                    row_count,
                    rope,
                    rms_eps,
                )
            }
            (
                DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed { attention },
                PackedCompressorViews::HeavilyCompressed {
                    attention_kv,
                    attention_score,
                },
            ) => attention.encode_projected_chunk(
                ctx,
                enc,
                attention_kv,
                attention_score,
                tensor("attn_compressor_ape.weight")?,
                tensor("attn_compressor_norm.weight")?,
                &scratch.pooled_rows,
                &scratch.normalized_rows,
                start_position,
                row_count,
                rope,
                rms_eps,
            ),
            _ => invalid(format!(
                "packed compressor projection kind differs from layer {layer}"
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_layer_projected_row(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residency: &DeepSeekV4MetalResidency,
        layer: usize,
        row: usize,
        position: u32,
        projected: &PackedCompressorViews,
        rope: DeepSeekV4RopeParameters,
        rms_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let tensor = |suffix: &str| residency.require_tensor(&format!("blk.{layer}.{suffix}"));
        let frontier = self.layers.get(layer).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "packed compressor layer {layer} is out of range"
            ))
        })?;
        match (frontier, projected) {
            (
                DeepSeekV4LayerCompressorFrontiers::SlidingWindow,
                PackedCompressorViews::SlidingWindow,
            ) => Ok(()),
            (
                DeepSeekV4LayerCompressorFrontiers::CompressedSparse { attention, indexer },
                PackedCompressorViews::CompressedSparse {
                    attention_kv,
                    attention_score,
                    indexer_kv,
                    indexer_score,
                },
            ) => {
                let attention_kv = f32_row(
                    attention_kv,
                    row,
                    COMPRESSOR_ATTENTION_WIDTH,
                    vec![COMPRESSOR_ATTENTION_WIDTH as u64],
                    "packed CSA compressor KV row",
                )?;
                let attention_score = f32_row(
                    attention_score,
                    row,
                    COMPRESSOR_ATTENTION_WIDTH,
                    vec![COMPRESSOR_ATTENTION_WIDTH as u64],
                    "packed CSA compressor score row",
                )?;
                attention.encode_projected(
                    ctx,
                    enc,
                    &attention_kv,
                    &attention_score,
                    tensor("attn_compressor_ape.weight")?,
                    tensor("attn_compressor_norm.weight")?,
                    position,
                    rope,
                    rms_eps,
                )?;
                let indexer_kv = f32_row(
                    indexer_kv,
                    row,
                    COMPRESSOR_INDEXER_WIDTH,
                    vec![COMPRESSOR_INDEXER_WIDTH as u64],
                    "packed indexer compressor KV row",
                )?;
                let indexer_score = f32_row(
                    indexer_score,
                    row,
                    COMPRESSOR_INDEXER_WIDTH,
                    vec![COMPRESSOR_INDEXER_WIDTH as u64],
                    "packed indexer compressor score row",
                )?;
                indexer.encode_projected(
                    ctx,
                    enc,
                    &indexer_kv,
                    &indexer_score,
                    tensor("indexer_compressor_ape.weight")?,
                    tensor("indexer_compressor_norm.weight")?,
                    position,
                    rope,
                    rms_eps,
                )
            }
            (
                DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed { attention },
                PackedCompressorViews::HeavilyCompressed {
                    attention_kv,
                    attention_score,
                },
            ) => {
                let attention_kv = f32_row(
                    attention_kv,
                    row,
                    512,
                    vec![512],
                    "packed HCA compressor KV row",
                )?;
                let attention_score = f32_row(
                    attention_score,
                    row,
                    512,
                    vec![512],
                    "packed HCA compressor score row",
                )?;
                attention.encode_projected(
                    ctx,
                    enc,
                    &attention_kv,
                    &attention_score,
                    tensor("attn_compressor_ape.weight")?,
                    tensor("attn_compressor_norm.weight")?,
                    position,
                    rope,
                    rms_eps,
                )
            }
            _ => invalid(format!(
                "packed compressor projection kind differs from layer {layer}"
            )),
        }
    }
}

struct PackedMoeViews {
    normalized_input: MetalTensor,
    logits: MetalTensor,
    hash_ids: Option<MetalTensor>,
}

struct ExpertBucket {
    expert: usize,
    start: usize,
    len: usize,
}

const PACKED_GROUPED_EXPERT_TILE_ROWS: usize = 32;
const PACKED_GROUPED_EXPERT_MAX_TILES: usize = MOE_EXPERT_COUNT
    + (DEEPSEEK_V4_PREFILL_MAX_TOKENS * MOE_TOP_K - MOE_EXPERT_COUNT)
        / PACKED_GROUPED_EXPERT_TILE_ROWS;
const PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS: usize = PACKED_GROUPED_EXPERT_MAX_TILES * 3;
const PACKED_GROUPED_EXPERT_INLINE_MAX_BYTES: usize = 4_096;
const PACKED_GROUPED_EXPERT_INLINE_MAX_TILES: usize = PACKED_GROUPED_EXPERT_INLINE_MAX_BYTES / 12;
const PACKED_GROUPED_IQ2_MMA16_TILE_ROWS: usize = 16;
const PACKED_GROUPED_IQ2_MMA16_NARROW_TOKENS: usize = 128;
const PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS: usize = 2_048;
const PACKED_GROUPED_IQ2_MMA16_MAX_TILES: usize = MOE_EXPERT_COUNT
    + (PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS * MOE_TOP_K - MOE_EXPERT_COUNT)
        / PACKED_GROUPED_IQ2_MMA16_TILE_ROWS;
const PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS: usize = PACKED_GROUPED_IQ2_MMA16_MAX_TILES * 3;
const _: () = assert!(PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS <= DEEPSEEK_V4_PREFILL_MAX_TOKENS);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedGroupedExpertTile {
    expert: u32,
    start: u32,
    count: u32,
}

const _: () = assert!(std::mem::size_of::<PackedGroupedExpertTile>() == 12);
const _: () = assert!(
    std::mem::size_of::<PackedGroupedExpertTile>() * PACKED_GROUPED_EXPERT_INLINE_MAX_TILES
        <= PACKED_GROUPED_EXPERT_INLINE_MAX_BYTES
);
const _: () = assert!(
    std::mem::size_of::<PackedGroupedExpertTile>() * (PACKED_GROUPED_EXPERT_INLINE_MAX_TILES + 1)
        > PACKED_GROUPED_EXPERT_INLINE_MAX_BYTES
);
const _: () = assert!(
    std::mem::size_of::<PackedGroupedExpertTile>() * PACKED_GROUPED_EXPERT_MAX_TILES
        > PACKED_GROUPED_EXPERT_INLINE_MAX_BYTES
);
const _: () = assert!(
    std::mem::size_of::<PackedGroupedExpertTile>() * PACKED_GROUPED_IQ2_MMA16_MAX_TILES
        > PACKED_GROUPED_EXPERT_INLINE_MAX_BYTES
);

fn validate_packed_expert_schedule(
    n_tokens: usize,
    expert_ids: &[i32],
    bucket_rows: &[i32],
    bucket_slots: &[i32],
    schedule: &[ExpertBucket],
) -> Result<(), DeepSeekV4MetalError> {
    let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed expert route count")?;
    if expert_ids.len() != route_count
        || bucket_rows.len() != route_count
        || bucket_slots.len() != route_count
    {
        return invalid("packed expert schedule payload has invalid length");
    }
    let mut cursor = 0usize;
    let mut previous_expert = None;
    let mut seen_slots = vec![false; route_count];
    for bucket in schedule {
        if bucket.expert >= MOE_EXPERT_COUNT
            || previous_expert.is_some_and(|expert| bucket.expert <= expert)
            || bucket.start != cursor
            || bucket.len == 0
            || bucket.len > n_tokens
        {
            return invalid("packed expert schedule has invalid bucket geometry");
        }
        let end = bucket.start.checked_add(bucket.len).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed expert bucket end overflow".into())
        })?;
        if end > route_count {
            return invalid("packed expert bucket exceeds route assignments");
        }
        let mut previous_slot = None;
        for index in bucket.start..end {
            let row = usize::try_from(bucket_rows[index]).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed expert schedule has negative row".into())
            })?;
            let slot = usize::try_from(bucket_slots[index]).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed expert schedule has negative slot".into())
            })?;
            if row >= n_tokens
                || slot >= route_count
                || row != slot / MOE_TOP_K
                || expert_ids[slot] != bucket.expert as i32
                || previous_slot.is_some_and(|previous| slot <= previous)
                || std::mem::replace(&mut seen_slots[slot], true)
            {
                return invalid("packed expert schedule violates expert/token/slot order");
            }
            previous_slot = Some(slot);
        }
        cursor = end;
        previous_expert = Some(bucket.expert);
    }
    if cursor != route_count || seen_slots.iter().any(|seen| !seen) {
        return invalid("packed expert schedule does not cover every route exactly once");
    }
    Ok(())
}

fn packed_grouped_expert_tiles(
    n_tokens: usize,
    schedule: &[ExpertBucket],
) -> Result<Vec<PackedGroupedExpertTile>, DeepSeekV4MetalError> {
    let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed grouped route count")?;
    let mut cursor = 0usize;
    let mut previous_expert = None;
    let mut tiles = Vec::new();
    for bucket in schedule {
        if bucket.expert >= MOE_EXPERT_COUNT
            || previous_expert.is_some_and(|expert| bucket.expert <= expert)
            || bucket.start != cursor
            || bucket.len == 0
            || bucket.len > n_tokens
        {
            return invalid("packed grouped expert schedule has invalid bucket geometry");
        }
        for offset in (0..bucket.len).step_by(PACKED_GROUPED_EXPERT_TILE_ROWS) {
            let count = (bucket.len - offset).min(PACKED_GROUPED_EXPERT_TILE_ROWS);
            tiles.push(PackedGroupedExpertTile {
                expert: u32::try_from(bucket.expert).map_err(|_| {
                    DeepSeekV4MetalError::Invalid("packed grouped expert exceeds u32".into())
                })?,
                start: u32::try_from(bucket.start + offset).map_err(|_| {
                    DeepSeekV4MetalError::Invalid("packed grouped start exceeds u32".into())
                })?,
                count: count as u32,
            });
        }
        cursor = bucket.start + bucket.len;
        previous_expert = Some(bucket.expert);
    }
    if cursor != route_count || tiles.is_empty() || tiles.len() > PACKED_GROUPED_EXPERT_MAX_TILES {
        return invalid(format!(
            "packed grouped expert plan has {} assignments and {} tiles",
            cursor,
            tiles.len()
        ));
    }
    Ok(tiles)
}

struct PackedGroupedExpertPlan {
    tiles: Vec<PackedGroupedExpertTile>,
    buffer: Option<MetalTensor>,
    dispatch_tiles: usize,
}

impl PackedGroupedExpertPlan {
    fn new(
        n_tokens: usize,
        schedule: &[ExpertBucket],
        tile_buffer: Option<&MetalTensor>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        Self::from_tiles(
            packed_grouped_expert_tiles(n_tokens, schedule)?,
            tile_buffer,
        )
    }

    fn new_iq2_mma16(
        n_tokens: usize,
        schedule: &[ExpertBucket],
        tile_buffer: Option<&MetalTensor>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        Self::from_tiles(
            packed_grouped_iq2_mma16_tiles(n_tokens, schedule)?,
            tile_buffer,
        )
    }

    fn from_tiles(
        tiles: Vec<PackedGroupedExpertTile>,
        tile_buffer: Option<&MetalTensor>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        let bytes = std::mem::size_of_val(tiles.as_slice());
        let buffer = if bytes > PACKED_GROUPED_EXPERT_INLINE_MAX_BYTES {
            let tile_buffer = tile_buffer.ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "packed grouped expert plan requires {bytes} descriptor bytes"
                ))
            })?;
            let words = bytemuck::cast_slice::<PackedGroupedExpertTile, i32>(&tiles);
            let view = i32_prefix(
                tile_buffer,
                vec![words.len() as u64],
                "packed grouped expert descriptors",
            )?;
            host_write_i32(&view, words, "packed grouped expert descriptors")?;
            Some(view)
        } else {
            None
        };
        let dispatch_tiles = tiles.len();
        Ok(Self {
            tiles,
            buffer,
            dispatch_tiles,
        })
    }

    fn from_device(
        buffer: &MetalTensor,
        dispatch_tiles: usize,
    ) -> Result<Self, DeepSeekV4MetalError> {
        validate_i32(
            buffer,
            &[(dispatch_tiles * 3) as u64],
            false,
            "packed grouped device descriptors",
        )?;
        Ok(Self {
            tiles: Vec::new(),
            buffer: Some(buffer.clone()),
            dispatch_tiles,
        })
    }

    fn bind(&self, enc: &KernelEncoder, index: usize) {
        if let Some(buffer) = self.buffer.as_ref() {
            enc.set_tensor(index, buffer);
        } else {
            enc.set_bytes_slice(index, &self.tiles);
        }
    }
}

fn packed_grouped_iq2_mma16_tiles(
    n_tokens: usize,
    schedule: &[ExpertBucket],
) -> Result<Vec<PackedGroupedExpertTile>, DeepSeekV4MetalError> {
    let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed IQ2 MMA route count")?;
    let mut cursor = 0usize;
    let mut previous_expert = None;
    let mut tiles = Vec::new();
    for bucket in schedule {
        if bucket.expert >= MOE_EXPERT_COUNT
            || previous_expert.is_some_and(|expert| bucket.expert <= expert)
            || bucket.start != cursor
            || bucket.len == 0
            || bucket.len > n_tokens
        {
            return invalid("packed IQ2 MMA schedule has invalid bucket geometry");
        }
        for offset in (0..bucket.len).step_by(PACKED_GROUPED_IQ2_MMA16_TILE_ROWS) {
            tiles.push(PackedGroupedExpertTile {
                expert: u32::try_from(bucket.expert).map_err(|_| {
                    DeepSeekV4MetalError::Invalid("packed IQ2 MMA expert exceeds u32".into())
                })?,
                start: u32::try_from(bucket.start + offset).map_err(|_| {
                    DeepSeekV4MetalError::Invalid("packed IQ2 MMA start exceeds u32".into())
                })?,
                count: (bucket.len - offset).min(PACKED_GROUPED_IQ2_MMA16_TILE_ROWS) as u32,
            });
        }
        cursor = bucket.start + bucket.len;
        previous_expert = Some(bucket.expert);
    }
    if cursor != route_count || tiles.is_empty() || tiles.len() > PACKED_GROUPED_IQ2_MMA16_MAX_TILES
    {
        return invalid(format!(
            "packed IQ2 MMA plan has {} assignments and {} tiles",
            cursor,
            tiles.len()
        ));
    }
    Ok(tiles)
}

#[allow(clippy::too_many_arguments)]
fn encode_packed_grouped_swiglu_iq2_xs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    normalized_input: &MetalTensor,
    bucket_slots: &MetalTensor,
    plan: &PackedGroupedExpertPlan,
    inner: &MetalTensor,
    hidden: usize,
    ffn: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
    clamp: f32,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "packed grouped IQ2_XS gate/up")?;
    if !hidden.is_multiple_of(256)
        || expert_count != MOE_EXPERT_COUNT
        || top_k != MOE_TOP_K
        || gate_bank.dtype != GgmlType::IQ2_XS
        || up_bank.dtype != GgmlType::IQ2_XS
        || !clamp.is_finite()
        || clamp <= 0.0
    {
        return invalid("packed grouped gate/up requires aligned IQ2_XS banks and positive clamp");
    }
    validate_expert_bank(
        gate_bank,
        hidden,
        ffn,
        expert_count,
        "packed grouped gate bank",
    )?;
    validate_expert_bank(up_bank, hidden, ffn, expert_count, "packed grouped up bank")?;
    validate_f32(
        normalized_input,
        &[hidden as u64, n_tokens as u64],
        false,
        "packed grouped normalized input",
    )?;
    validate_i32(
        bucket_slots,
        &[(n_tokens * top_k) as u64],
        false,
        "packed grouped slots",
    )?;
    validate_f32(
        inner,
        &[ffn as u64, top_k as u64, n_tokens as u64],
        true,
        "packed grouped inner",
    )?;
    let tile_count = plan.dispatch_tiles;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        hidden: u32,
        ffn: u32,
        n_expert: u32,
        top_k: u32,
        n_tokens: u32,
        clamp: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_packed_grouped_swiglu_iq2_xs_f32")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 32 {
        return invalid("packed grouped IQ2_XS gate/up requires SIMD width 32");
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            hidden: u32::try_from(hidden).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped hidden exceeds u32".into())
            })?,
            ffn: u32::try_from(ffn).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped FFN exceeds u32".into())
            })?,
            n_expert: u32::try_from(expert_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped experts exceed u32".into())
            })?,
            top_k: u32::try_from(top_k).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped top-k exceeds u32".into())
            })?,
            n_tokens: checked_token_count(n_tokens)?,
            clamp,
        },
    );
    enc.set_tensor(1, gate_bank);
    enc.set_tensor(2, up_bank);
    enc.set_tensor(3, normalized_input);
    enc.set_tensor(4, bucket_slots);
    plan.bind(enc, 5);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 64 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: ffn,
            height: tile_count,
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
fn encode_packed_grouped_down_iq3_xxs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    down_bank: &MetalTensor,
    inner: &MetalTensor,
    bucket_slots: &MetalTensor,
    plan: &PackedGroupedExpertPlan,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "packed grouped IQ3_XXS down")?;
    if !crate::metal::matmat_iq3_xxs_mm_is_enabled() {
        return invalid("packed grouped IQ3_XXS down requires the SIMD-matrix policy");
    }
    if !n_in.is_multiple_of(256)
        || !n_out.is_multiple_of(64)
        || expert_count != MOE_EXPERT_COUNT
        || top_k != MOE_TOP_K
        || down_bank.dtype != GgmlType::IQ3_XXS
    {
        return invalid("packed grouped down requires aligned IQ3_XXS storage");
    }
    validate_expert_bank(
        down_bank,
        n_in,
        n_out,
        expert_count,
        "packed grouped down bank",
    )?;
    validate_f32(
        inner,
        &[n_in as u64, top_k as u64, n_tokens as u64],
        false,
        "packed grouped inner",
    )?;
    validate_i32(
        bucket_slots,
        &[(n_tokens * top_k) as u64],
        false,
        "packed grouped down slots",
    )?;
    validate_f32(
        output,
        &[n_out as u64, top_k as u64, n_tokens as u64],
        true,
        "packed grouped output",
    )?;
    let tile_count = plan.dispatch_tiles;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
        n_expert: u32,
        top_k: u32,
        n_tokens: u32,
    }
    let row_bytes = checked_mul(n_in / 256, 98, "packed grouped IQ3_XXS row bytes")?;
    let pso = ctx.pipeline("kernel_deepseek_v4_packed_grouped_down_iq3_xxs_f32_mm")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 128 {
        return invalid("packed grouped IQ3_XXS down requires four SIMD groups");
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: u32::try_from(n_out).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped output exceeds u32".into())
            })?,
            k: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped input exceeds u32".into())
            })?,
            nb01: u32::try_from(row_bytes).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped row bytes exceed u32".into())
            })?,
            stride_b: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped stride exceeds u32".into())
            })?,
            n_expert: u32::try_from(expert_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped experts exceed u32".into())
            })?,
            top_k: u32::try_from(top_k).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped top-k exceeds u32".into())
            })?,
            n_tokens: checked_token_count(n_tokens)?,
        },
    );
    enc.set_tensor(1, down_bank);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, bucket_slots);
    plan.bind(enc, 4);
    enc.set_tensor(5, output);
    enc.set_threadgroup_memory(0, 8_192);
    enc.dispatch(
        MTLSize {
            width: tile_count,
            height: n_out / 64,
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

#[allow(clippy::too_many_arguments)]
fn encode_packed_grouped_mapped_iq2_xs_f32_mma16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    source_rows: &MetalTensor,
    destination_slots: &MetalTensor,
    plan: &PackedGroupedExpertPlan,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
    source_count: usize,
    destination_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "packed grouped mapped IQ2_XS F32 MMA projection")?;
    if !n_in.is_multiple_of(256)
        || !n_out.is_multiple_of(16)
        || expert_count != MOE_EXPERT_COUNT
        || top_k != MOE_TOP_K
        || source_count == 0
        || destination_count == 0
        || bank.dtype != GgmlType::IQ2_XS
    {
        return invalid("packed grouped mapped IQ2_XS F32 MMA has invalid geometry or storage");
    }
    validate_expert_bank(
        bank,
        n_in,
        n_out,
        expert_count,
        "packed grouped mapped IQ2_XS F32 MMA bank",
    )?;
    validate_f32(
        input,
        &[n_in as u64, source_count as u64],
        false,
        "packed grouped mapped IQ2_XS F32 MMA input",
    )?;
    let map_count = checked_mul(n_tokens, top_k, "packed grouped mapped IQ2 MMA rows")?;
    validate_i32(
        source_rows,
        &[map_count as u64],
        false,
        "packed grouped mapped IQ2_XS F32 MMA source rows",
    )?;
    validate_i32(
        destination_slots,
        &[map_count as u64],
        false,
        "packed grouped mapped IQ2_XS F32 MMA destination slots",
    )?;
    validate_f32(
        output,
        &[n_out as u64, destination_count as u64],
        true,
        "packed grouped mapped IQ2_XS F32 MMA output",
    )?;
    let tile_count = plan.dispatch_tiles;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
        n_expert: u32,
        map_count: u32,
        source_count: u32,
        destination_count: u32,
    }
    let row_bytes = checked_mul(n_in / 256, 74, "packed grouped mapped IQ2 MMA row")?;
    let pso = ctx.pipeline("kernel_deepseek_v4_packed_grouped_mapped_iq2_xs_f32_mma16")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 32 {
        return invalid("packed grouped mapped IQ2_XS F32 MMA requires one SIMD group");
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: u32::try_from(n_out).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed IQ2 MMA output exceeds u32".into())
            })?,
            k: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed IQ2 MMA input exceeds u32".into())
            })?,
            nb01: u32::try_from(row_bytes).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed IQ2 MMA row bytes exceed u32".into())
            })?,
            stride_b: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed IQ2 MMA stride exceeds u32".into())
            })?,
            n_expert: u32::try_from(expert_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed IQ2 MMA experts exceed u32".into())
            })?,
            map_count: u32::try_from(map_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed IQ2 MMA map count exceeds u32".into())
            })?,
            source_count: u32::try_from(source_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed IQ2 MMA source count exceeds u32".into())
            })?,
            destination_count: u32::try_from(destination_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed IQ2 MMA destination count exceeds u32".into())
            })?,
        },
    );
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, source_rows);
    enc.set_tensor(4, destination_slots);
    plan.bind(enc, 5);
    enc.set_tensor(6, output);
    enc.set_threadgroup_memory(0, 4_096);
    enc.dispatch(
        MTLSize {
            width: tile_count,
            height: n_out / 16,
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
fn encode_packed_grouped_mapped_iq2_xs_swiglu_f32_mma16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    input: &MetalTensor,
    source_rows: &MetalTensor,
    destination_slots: &MetalTensor,
    plan: &PackedGroupedExpertPlan,
    gate: &MetalTensor,
    up: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
    source_count: usize,
    destination_count: usize,
    clamp: f32,
) -> Result<(), DeepSeekV4MetalError> {
    for (bank, projection) in [(gate_bank, gate), (up_bank, up)] {
        encode_packed_grouped_mapped_iq2_xs_f32_mma16(
            ctx,
            enc,
            bank,
            input,
            source_rows,
            destination_slots,
            plan,
            projection,
            n_in,
            n_out,
            expert_count,
            top_k,
            n_tokens,
            source_count,
            destination_count,
        )?;
    }
    let projected_elements = checked_mul(
        n_out,
        destination_count,
        "packed BM16 IQ2 projected elements",
    )?;
    encode_ds4_clamped_swiglu(
        ctx,
        enc,
        &gate.view_subrange(0, vec![projected_elements as u64]),
        &up.view_subrange(0, vec![projected_elements as u64]),
        &output.view_subrange(0, vec![projected_elements as u64]),
        clamp,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_packed_grouped_mapped_iq3_xxs_f32_plan(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    source_rows: &MetalTensor,
    destination_slots: &MetalTensor,
    plan: &PackedGroupedExpertPlan,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
    source_count: usize,
    destination_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "packed grouped mapped IQ3_XXS projection")?;
    if !crate::metal::matmat_iq3_xxs_mm_is_enabled() {
        return invalid("packed grouped mapped IQ3_XXS requires the SIMD-matrix policy");
    }
    if !n_in.is_multiple_of(256)
        || !n_out.is_multiple_of(64)
        || expert_count != MOE_EXPERT_COUNT
        || top_k != MOE_TOP_K
        || source_count == 0
        || destination_count == 0
        || bank.dtype != GgmlType::IQ3_XXS
    {
        return invalid("packed grouped mapped IQ3_XXS has invalid geometry or storage");
    }
    validate_expert_bank(
        bank,
        n_in,
        n_out,
        expert_count,
        "packed grouped mapped IQ3_XXS bank",
    )?;
    validate_f32(
        input,
        &[n_in as u64, source_count as u64],
        false,
        "packed grouped mapped IQ3_XXS input",
    )?;
    let map_count = checked_mul(n_tokens, top_k, "packed grouped mapped row count")?;
    validate_i32(
        source_rows,
        &[map_count as u64],
        false,
        "packed grouped mapped IQ3_XXS source rows",
    )?;
    validate_i32(
        destination_slots,
        &[map_count as u64],
        false,
        "packed grouped mapped IQ3_XXS destination slots",
    )?;
    validate_f32(
        output,
        &[n_out as u64, destination_count as u64],
        true,
        "packed grouped mapped IQ3_XXS output",
    )?;
    let tile_count = plan.dispatch_tiles;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
        n_expert: u32,
        map_count: u32,
        source_count: u32,
        destination_count: u32,
    }
    let row_bytes = checked_mul(n_in / 256, 98, "packed grouped mapped IQ3_XXS row")?;
    let pso = ctx.pipeline("kernel_deepseek_v4_packed_grouped_mapped_iq3_xxs_f32_mm")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 128 {
        return invalid("packed grouped mapped IQ3_XXS requires four SIMD groups");
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: u32::try_from(n_out).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped mapped output exceeds u32".into())
            })?,
            k: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped mapped input exceeds u32".into())
            })?,
            nb01: u32::try_from(row_bytes).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped mapped row bytes exceed u32".into())
            })?,
            stride_b: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped mapped stride exceeds u32".into())
            })?,
            n_expert: u32::try_from(expert_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped mapped experts exceed u32".into())
            })?,
            map_count: u32::try_from(map_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped mapped count exceeds u32".into())
            })?,
            source_count: u32::try_from(source_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped source count exceeds u32".into())
            })?,
            destination_count: u32::try_from(destination_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped destination count exceeds u32".into())
            })?,
        },
    );
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, source_rows);
    enc.set_tensor(4, destination_slots);
    plan.bind(enc, 5);
    enc.set_tensor(6, output);
    enc.set_threadgroup_memory(0, 8_192);
    enc.dispatch(
        MTLSize {
            width: tile_count,
            height: n_out / 64,
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

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn encode_packed_grouped_mapped_iq3_xxs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    source_rows: &MetalTensor,
    destination_slots: &MetalTensor,
    schedule: &[ExpertBucket],
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
    source_count: usize,
    destination_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let plan = PackedGroupedExpertPlan::new(n_tokens, schedule, None)?;
    encode_packed_grouped_mapped_iq3_xxs_f32_plan(
        ctx,
        enc,
        bank,
        input,
        source_rows,
        destination_slots,
        &plan,
        output,
        n_in,
        n_out,
        expert_count,
        top_k,
        n_tokens,
        source_count,
        destination_count,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_packed_grouped_all_iq3(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    down_bank: &MetalTensor,
    input: &MetalTensor,
    source_rows: &MetalTensor,
    destination_slots: &MetalTensor,
    plan: &PackedGroupedExpertPlan,
    output: &MetalTensor,
    inner: &MetalTensor,
    hidden: usize,
    ffn: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
    clamp: f32,
) -> Result<(), DeepSeekV4MetalError> {
    let route_count = checked_mul(n_tokens, top_k, "packed grouped all-IQ3 routes")?;
    let projected_elements = checked_mul(ffn, route_count, "packed grouped all-IQ3 projection")?;
    let (gate, up) = packed_grouped_gate_up_views(output, hidden, ffn, route_count)?;
    for (bank, projection) in [(gate_bank, &gate), (up_bank, &up)] {
        encode_packed_grouped_mapped_iq3_xxs_f32_plan(
            ctx,
            enc,
            bank,
            input,
            source_rows,
            destination_slots,
            plan,
            projection,
            hidden,
            ffn,
            expert_count,
            top_k,
            n_tokens,
            n_tokens,
            route_count,
        )?;
    }
    encode_ds4_clamped_swiglu(
        ctx,
        enc,
        &gate.view_subrange(0, vec![projected_elements as u64]),
        &up.view_subrange(0, vec![projected_elements as u64]),
        &inner.view_subrange(0, vec![projected_elements as u64]),
        clamp,
    )?;
    encode_packed_grouped_mapped_iq3_xxs_f32_plan(
        ctx,
        enc,
        down_bank,
        inner,
        destination_slots,
        destination_slots,
        plan,
        output,
        ffn,
        hidden,
        expert_count,
        top_k,
        n_tokens,
        route_count,
        route_count,
    )
}

fn packed_grouped_tensor_ranges_overlap(left: &MetalTensor, right: &MetalTensor) -> bool {
    if Retained::as_ptr(&left.buffer) != Retained::as_ptr(&right.buffer) {
        return false;
    }
    let left_end = left.offset + left.n_bytes();
    let right_end = right.offset + right.n_bytes();
    left.offset < right_end && right.offset < left_end
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    input: &MetalTensor,
    source_rows: &MetalTensor,
    destination_slots: &MetalTensor,
    schedule: &[ExpertBucket],
    gate_up_arena: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
    source_count: usize,
    destination_count: usize,
    clamp: f32,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "packed grouped mapped IQ3_XXS gate/up/SwiGLU")?;
    if !crate::metal::matmat_iq3_xxs_mm_is_enabled() {
        return invalid("packed grouped mapped IQ3_XXS SwiGLU requires the SIMD-matrix policy");
    }
    if !n_in.is_multiple_of(256)
        || !n_out.is_multiple_of(64)
        || n_in != 2 * n_out
        || expert_count != MOE_EXPERT_COUNT
        || top_k != MOE_TOP_K
        || source_count == 0
        || destination_count == 0
        || gate_bank.dtype != GgmlType::IQ3_XXS
        || up_bank.dtype != GgmlType::IQ3_XXS
        || !clamp.is_finite()
        || clamp <= 0.0
    {
        return invalid("packed grouped mapped IQ3_XXS SwiGLU has invalid geometry or storage");
    }
    validate_expert_bank(
        gate_bank,
        n_in,
        n_out,
        expert_count,
        "packed grouped mapped IQ3_XXS gate bank",
    )?;
    validate_expert_bank(
        up_bank,
        n_in,
        n_out,
        expert_count,
        "packed grouped mapped IQ3_XXS up bank",
    )?;
    validate_f32(
        input,
        &[n_in as u64, source_count as u64],
        false,
        "packed grouped mapped IQ3_XXS SwiGLU input",
    )?;
    let map_count = checked_mul(n_tokens, top_k, "packed grouped mapped SwiGLU row count")?;
    validate_i32(
        source_rows,
        &[map_count as u64],
        false,
        "packed grouped mapped IQ3_XXS SwiGLU source rows",
    )?;
    validate_i32(
        destination_slots,
        &[map_count as u64],
        false,
        "packed grouped mapped IQ3_XXS SwiGLU destination slots",
    )?;
    let (gate, up) = packed_grouped_gate_up_views(gate_up_arena, n_in, n_out, destination_count)?;
    validate_f32(
        inner,
        &[n_out as u64, destination_count as u64],
        true,
        "packed grouped mapped IQ3_XXS SwiGLU inner",
    )?;
    if packed_grouped_tensor_ranges_overlap(&gate, inner)
        || packed_grouped_tensor_ranges_overlap(&up, inner)
    {
        return invalid("packed grouped mapped IQ3_XXS SwiGLU inner overlaps gate/up arena");
    }
    let tiles = packed_grouped_expert_tiles(n_tokens, schedule)?;
    let tile_count = tiles.len();
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
        n_expert: u32,
        map_count: u32,
        source_count: u32,
        destination_count: u32,
        clamp: f32,
    }
    let row_bytes = checked_mul(n_in / 256, 98, "packed grouped mapped IQ3_XXS SwiGLU row")?;
    let pso = ctx.pipeline("kernel_deepseek_v4_packed_grouped_mapped_swiglu_iq3_xxs_f32_mm")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 128 {
        return invalid("packed grouped mapped IQ3_XXS SwiGLU requires four SIMD groups");
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: u32::try_from(n_out).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "packed grouped mapped SwiGLU output exceeds u32".into(),
                )
            })?,
            k: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "packed grouped mapped SwiGLU input exceeds u32".into(),
                )
            })?,
            nb01: u32::try_from(row_bytes).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "packed grouped mapped SwiGLU row bytes exceed u32".into(),
                )
            })?,
            stride_b: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "packed grouped mapped SwiGLU stride exceeds u32".into(),
                )
            })?,
            n_expert: u32::try_from(expert_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "packed grouped mapped SwiGLU experts exceed u32".into(),
                )
            })?,
            map_count: u32::try_from(map_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "packed grouped mapped SwiGLU count exceeds u32".into(),
                )
            })?,
            source_count: u32::try_from(source_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "packed grouped mapped SwiGLU source count exceeds u32".into(),
                )
            })?,
            destination_count: u32::try_from(destination_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "packed grouped mapped SwiGLU destination count exceeds u32".into(),
                )
            })?,
            clamp,
        },
    );
    enc.set_tensor(1, gate_bank);
    enc.set_tensor(2, up_bank);
    enc.set_tensor(3, input);
    enc.set_tensor(4, source_rows);
    enc.set_tensor(5, destination_slots);
    enc.set_bytes_slice(6, &tiles);
    enc.set_tensor(7, &gate);
    enc.set_tensor(8, &up);
    enc.set_tensor(9, inner);
    enc.set_threadgroup_memory(0, 8_192);
    enc.dispatch(
        MTLSize {
            width: tile_count,
            height: n_out / 64,
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

fn packed_grouped_gate_up_views(
    arena: &MetalTensor,
    hidden: usize,
    ffn: usize,
    destination_count: usize,
) -> Result<(MetalTensor, MetalTensor), DeepSeekV4MetalError> {
    let arena_elements = checked_mul(
        destination_count,
        hidden,
        "packed grouped IQ3 gate/up arena",
    )?;
    let half_elements = checked_mul(destination_count, ffn, "packed grouped IQ3 gate/up half")?;
    if hidden != 2 * ffn
        || checked_mul(half_elements, 2, "packed grouped IQ3 gate/up split")? != arena_elements
    {
        return invalid("packed grouped IQ3 arena does not split into gate/up halves");
    }
    validate_f32(
        arena,
        &[hidden as u64, destination_count as u64],
        true,
        "packed grouped IQ3 gate/up arena",
    )?;
    let gate = arena.view_subrange(0, vec![ffn as u64, destination_count as u64]);
    let up = arena.view_subrange(
        half_elements as u64,
        vec![ffn as u64, destination_count as u64],
    );
    if Retained::as_ptr(&gate.buffer) != Retained::as_ptr(&up.buffer)
        || Retained::as_ptr(&gate.buffer) != Retained::as_ptr(&arena.buffer)
        || gate.offset + gate.n_bytes() != up.offset
        || up.offset + up.n_bytes() != arena.offset + arena.n_bytes()
    {
        return invalid("packed grouped IQ3 gate/up arena views are not exact and disjoint");
    }
    Ok((gate, up))
}

fn packed_grouped_expert_kernels_supported(ctx: &MetalContext) -> bool {
    if !crate::metal::matmat_iq3_xxs_mm_is_enabled()
        || ctx.device.maxThreadgroupMemoryLength() < 8_192
    {
        return false;
    }
    let Ok(gate_up) = ctx.pipeline("kernel_deepseek_v4_packed_grouped_swiglu_iq2_xs_f32") else {
        return false;
    };
    let Ok(down) = ctx.pipeline("kernel_deepseek_v4_packed_grouped_down_iq3_xxs_f32_mm") else {
        return false;
    };
    gate_up.threadExecutionWidth() == 32
        && gate_up.maxTotalThreadsPerThreadgroup() >= 32
        && down.threadExecutionWidth() == 32
        && down.maxTotalThreadsPerThreadgroup() >= 128
}

fn packed_grouped_iq2_mma16_candidate_supported(ctx: &MetalContext) -> bool {
    if ctx.device.maxThreadgroupMemoryLength() < 4_096 {
        return false;
    }
    let Ok(projection) = ctx.pipeline("kernel_deepseek_v4_packed_grouped_mapped_iq2_xs_f32_mma16")
    else {
        return false;
    };
    projection.threadExecutionWidth() == 32 && projection.maxTotalThreadsPerThreadgroup() >= 32
}

crate::env_flag!(
    default_on packed_grouped_iq2_mma16_enabled,
    "QWEN_DSV4_PACKED_BM16_IQ2"
);

crate::env_flag!(
    default_on packed_batched_rope_enabled,
    "QWEN_DSV4_BATCHED_ROPE"
);

crate::env_flag!(
    default_on packed_batched_compressor_enabled,
    "QWEN_DSV4_BATCHED_COMPRESSOR"
);

crate::env_flag!(
    default_on packed_selected_online_enabled,
    "QWEN_DSV4_PACKED_SELECTED_ONLINE"
);

crate::env_flag!(
    default_on packed_grouped_dense_attention_enabled,
    "QWEN_DSV4_PACKED_GROUP8_DENSE"
);

crate::env_flag!(
    default_on packed_indexer_batched_rope_enabled,
    "QWEN_DSV4_PACKED_INDEXER_BATCHED_ROPE"
);

crate::env_flag!(
    default_off packed_indexer_q_matrix_enabled,
    "QWEN_DSV4_PACKED_INDEXER_Q_MATRIX"
);

crate::env_flag!(
    default_off packed_gpu_route_compact_enabled,
    "QWEN_DSV4_PACKED_GPU_ROUTE_COMPACT"
);

crate::env_flag!(
    default_on packed_grouped_iq3_enabled,
    "QWEN_DSV4_PACKED_GROUPED_IQ3"
);

crate::env_flag!(
    default_off packed_gpu_route_iq3_enabled,
    "QWEN_DSV4_PACKED_GPU_ROUTE_IQ3"
);

fn packed_grouped_iq3_candidate_supported(ctx: &MetalContext) -> bool {
    if !crate::metal::matmat_iq3_xxs_mm_is_enabled()
        || ctx.device.maxThreadgroupMemoryLength() < 8_192
    {
        return false;
    }
    let Ok(projection) = ctx.pipeline("kernel_deepseek_v4_packed_grouped_mapped_iq3_xxs_f32_mm")
    else {
        return false;
    };
    projection.threadExecutionWidth() == 32 && projection.maxTotalThreadsPerThreadgroup() >= 128
}

#[cfg(test)]
fn packed_grouped_iq3_fused_candidate_supported(ctx: &MetalContext) -> bool {
    if !crate::metal::matmat_iq3_xxs_mm_is_enabled()
        || ctx.device.maxThreadgroupMemoryLength() < 8_192
    {
        return false;
    }
    let Ok(fused) = ctx.pipeline("kernel_deepseek_v4_packed_grouped_mapped_swiglu_iq3_xxs_f32_mm")
    else {
        return false;
    };
    fused.threadExecutionWidth() == 32 && fused.maxTotalThreadsPerThreadgroup() >= 128
}

const PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE: &str = "Apple M4 Max";
const PACKED_GROUPED_EXPERT_MAX_TOKENS: usize = DEEPSEEK_V4_PREFILL_MAX_TOKENS;
const PACKED_GPU_ROUTE_MAX_TOKENS: usize = DEEPSEEK_V4_PREFILL_MAX_TOKENS;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PackedGroupedExpertMode {
    Auto,
    ForceOn,
    ForceOff,
}

fn parse_packed_grouped_expert_mode(value: Option<&str>) -> PackedGroupedExpertMode {
    match value {
        None | Some("auto" | "AUTO") => PackedGroupedExpertMode::Auto,
        Some("1" | "true" | "TRUE" | "yes" | "YES") => PackedGroupedExpertMode::ForceOn,
        Some("0" | "false" | "FALSE" | "no" | "NO") => PackedGroupedExpertMode::ForceOff,
        Some(_) => PackedGroupedExpertMode::ForceOff,
    }
}

fn packed_grouped_expert_mode() -> PackedGroupedExpertMode {
    static MODE: OnceLock<PackedGroupedExpertMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let value = std::env::var("QWEN_DSV4_PACKED_GROUPED_EXPERTS").ok();
        parse_packed_grouped_expert_mode(value.as_deref())
    })
}

fn packed_grouped_iq2_mma16_qualified(n_tokens: usize) -> bool {
    matches!(
        n_tokens,
        PACKED_GROUPED_IQ2_MMA16_NARROW_TOKENS | PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS
    )
}

fn packed_grouped_expert_scope(
    mode: PackedGroupedExpertMode,
    n_tokens: usize,
) -> Result<bool, DeepSeekV4MetalError> {
    if n_tokens <= PACKED_GROUPED_EXPERT_MAX_TOKENS {
        return Ok(true);
    }
    if mode == PackedGroupedExpertMode::ForceOn {
        return invalid(format!(
            "packed grouped experts are qualified through {PACKED_GROUPED_EXPERT_MAX_TOKENS} tokens, got {n_tokens}"
        ));
    }
    Ok(false)
}

fn packed_grouped_expert_policy(
    ctx: &MetalContext,
    n_tokens: usize,
) -> Result<PackedExpertPolicy, DeepSeekV4MetalError> {
    let enabled = match packed_grouped_expert_mode() {
        PackedGroupedExpertMode::Auto => {
            ctx.device.name().to_string() == PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE
        }
        PackedGroupedExpertMode::ForceOn => true,
        PackedGroupedExpertMode::ForceOff => false,
    } && packed_grouped_expert_kernels_supported(ctx);
    if packed_grouped_iq2_mma16_enabled()
        && enabled
        && packed_grouped_iq2_mma16_qualified(n_tokens)
        && ctx.device.name().to_string() == PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE
        && packed_grouped_iq2_mma16_candidate_supported(ctx)
    {
        return Ok(PackedExpertPolicy::GroupedIq2XsIq3XxsMma16QualifiedChunk);
    }
    Ok(if enabled {
        PackedExpertPolicy::GroupedIq2XsIq3Xxs
    } else {
        PackedExpertPolicy::Current
    })
}

#[cfg(all(test, feature = "dsv4-diagnostics"))]
pub(super) fn packed_grouped_expert_enabled_for_test(ctx: &MetalContext) -> bool {
    packed_grouped_expert_policy(ctx, PACKED_GROUPED_IQ2_MMA16_NARROW_TOKENS)
        .expect("valid packed grouped expert policy")
        .uses_iq2_target()
}

#[cfg(all(test, feature = "dsv4-diagnostics"))]
pub(super) fn packed_grouped_iq3_candidate_supported_for_test(ctx: &MetalContext) -> bool {
    packed_grouped_iq3_candidate_supported(ctx)
}

struct PackedLayerTrace {
    layer: usize,
    pre_expert_seconds: f64,
    pre_expert_gpu_seconds: f64,
    pre_expert_encode_seconds: f64,
    pre_expert_wait_seconds: f64,
    pre_expert_wait_residual_seconds: f64,
    pre_expert_post_seconds: f64,
    route_seconds: f64,
    post_route_seconds: f64,
    post_route_gpu_seconds: f64,
    post_route_encode_seconds: f64,
    post_route_wait_seconds: f64,
    post_route_wait_residual_seconds: f64,
    bucket_count: usize,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackedPrefillStageKind {
    BeforeAttentionBody,
    SparseIndexerPrepare,
    SparseIndexerScore,
    SparseSelection,
    AttentionCore,
    InverseRope,
    AttentionOutputProjections,
    AfterAttentionOutput,
}

#[cfg(feature = "dsv4-diagnostics")]
pub(super) const PACKED_PREFILL_STAGE_KINDS: [PackedPrefillStageKind; 8] = [
    PackedPrefillStageKind::BeforeAttentionBody,
    PackedPrefillStageKind::SparseIndexerPrepare,
    PackedPrefillStageKind::SparseIndexerScore,
    PackedPrefillStageKind::SparseSelection,
    PackedPrefillStageKind::AttentionCore,
    PackedPrefillStageKind::InverseRope,
    PackedPrefillStageKind::AttentionOutputProjections,
    PackedPrefillStageKind::AfterAttentionOutput,
];

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct PackedPrefillStageTiming {
    pub kind: PackedPrefillStageKind,
    pub start_timestamp: Option<u64>,
    pub end_timestamp: Option<u64>,
    pub duration_ticks: u64,
    pub duration_ms_scaled: f64,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct PackedPrefillStageTransition {
    pub from: PackedPrefillStageKind,
    pub to: PackedPrefillStageKind,
    pub delta_ticks: i128,
    pub gap_ticks: u64,
    pub overlap_ticks: u64,
    pub gap_ms_scaled: f64,
    pub overlap_ms_scaled: f64,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct PackedPrefillSampledLayerProfile {
    pub layer: usize,
    pub command_gpu_ms: f64,
    pub sampled_span_ticks: u64,
    pub raw_span_ms_assuming_ns: f64,
    pub raw_coverage_assuming_ns: f64,
    pub encoder_gap_ms_scaled: f64,
    pub encoder_overlap_ms_scaled: f64,
    pub stages: Vec<PackedPrefillStageTiming>,
    pub transitions: Vec<PackedPrefillStageTransition>,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct PackedPrefillStageProfile {
    pub sampled: bool,
    pub command_gpu_ms: Vec<f64>,
    pub sampled_layers: Vec<PackedPrefillSampledLayerProfile>,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug)]
struct PackedPrefillPendingStageSample {
    layer: usize,
    kind: PackedPrefillStageKind,
    samples: Option<(usize, usize)>,
}

#[cfg(feature = "dsv4-diagnostics")]
fn resolve_packed_prefill_layer_stage_samples(
    layer: usize,
    records: &[PackedPrefillPendingStageSample],
    timestamps: &[u64],
    command_gpu_ms: f64,
    allowed_empty: &[PackedPrefillStageKind],
) -> Result<PackedPrefillSampledLayerProfile, DeepSeekV4MetalError> {
    if records.len() != PACKED_PREFILL_STAGE_KINDS.len() {
        return invalid(format!(
            "packed prefill sampled layer {layer} produced {} stages, expected {}",
            records.len(),
            PACKED_PREFILL_STAGE_KINDS.len()
        ));
    }
    if !command_gpu_ms.is_finite() || command_gpu_ms <= 0.0 {
        return invalid(format!(
            "packed prefill sampled layer {layer} has invalid command GPU duration {command_gpu_ms}"
        ));
    }
    for (record, expected) in records.iter().zip(PACKED_PREFILL_STAGE_KINDS) {
        if record.layer != layer || record.kind != expected {
            return invalid(format!(
                "packed prefill sampled layer {layer} recorded {:?} for layer {}, expected {expected:?}",
                record.kind, record.layer
            ));
        }
        match record.samples {
            Some((start_sample, end_sample)) => {
                if start_sample >= timestamps.len() || end_sample >= timestamps.len() {
                    return invalid(format!(
                        "packed prefill sampled layer {layer} stage {:?} indexes samples {start_sample}..{end_sample} from {} timestamps",
                        record.kind,
                        timestamps.len()
                    ));
                }
            }
            None => {
                if !allowed_empty.contains(&record.kind) {
                    return invalid(format!(
                        "packed prefill sampled layer {layer} has unexpected empty stage {:?}",
                        record.kind
                    ));
                }
            }
        }
    }

    let first_timestamp = records
        .iter()
        .find_map(|record| record.samples.map(|(start, _)| timestamps[start]))
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "packed prefill sampled layer {layer} has no physical stages"
            ))
        })?;
    let last_timestamp = records
        .iter()
        .rev()
        .find_map(|record| record.samples.map(|(_, end)| timestamps[end]))
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "packed prefill sampled layer {layer} has no physical stages"
            ))
        })?;
    let sampled_span_ticks = last_timestamp.checked_sub(first_timestamp).ok_or_else(|| {
        DeepSeekV4MetalError::Invalid(format!(
            "packed prefill sampled layer {layer} returned non-monotonic span timestamps"
        ))
    })?;
    if sampled_span_ticks == 0 {
        return invalid(format!(
            "packed prefill sampled layer {layer} returned a zero timestamp span"
        ));
    }

    let scale_ms_per_tick = command_gpu_ms / sampled_span_ticks as f64;
    let mut stage_ticks = 0u64;
    let mut gap_ticks = 0u64;
    let mut overlap_ticks = 0u64;
    let mut previous_start = None;
    let mut previous_end = None;
    let mut previous_kind = None;
    let mut stages = Vec::with_capacity(records.len());
    let mut transitions = Vec::with_capacity(records.len() - 1);
    for record in records {
        let Some((start_sample, end_sample)) = record.samples else {
            stages.push(PackedPrefillStageTiming {
                kind: record.kind,
                start_timestamp: None,
                end_timestamp: None,
                duration_ticks: 0,
                duration_ms_scaled: 0.0,
            });
            continue;
        };
        let start_timestamp = timestamps[start_sample];
        let end_timestamp = timestamps[end_sample];
        let duration_ticks = end_timestamp.checked_sub(start_timestamp).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "packed prefill sampled layer {layer} stage {:?} returned inverted timestamps",
                record.kind
            ))
        })?;
        if previous_start.is_some_and(|previous| start_timestamp < previous)
            || previous_end.is_some_and(|previous| end_timestamp < previous)
        {
            return invalid(format!(
                "packed prefill sampled layer {layer} stage {:?} reverses physical start/end order",
                record.kind
            ));
        }
        if let (Some(previous_end), Some(previous_kind)) = (previous_end, previous_kind) {
            let delta_ticks = start_timestamp as i128 - previous_end as i128;
            let (transition_gap, transition_overlap) = if delta_ticks >= 0 {
                (delta_ticks as u64, 0)
            } else {
                (0, (-delta_ticks) as u64)
            };
            gap_ticks = gap_ticks.checked_add(transition_gap).ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "packed prefill encoder-gap tick total overflow".into(),
                )
            })?;
            overlap_ticks = overlap_ticks
                .checked_add(transition_overlap)
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "packed prefill encoder-overlap tick total overflow".into(),
                    )
                })?;
            transitions.push(PackedPrefillStageTransition {
                from: previous_kind,
                to: record.kind,
                delta_ticks,
                gap_ticks: transition_gap,
                overlap_ticks: transition_overlap,
                gap_ms_scaled: transition_gap as f64 * scale_ms_per_tick,
                overlap_ms_scaled: transition_overlap as f64 * scale_ms_per_tick,
            });
        }
        previous_start = Some(start_timestamp);
        previous_end = Some(end_timestamp);
        previous_kind = Some(record.kind);
        stage_ticks = stage_ticks.checked_add(duration_ticks).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed prefill sampled stage tick total overflow".into())
        })?;
        stages.push(PackedPrefillStageTiming {
            kind: record.kind,
            start_timestamp: Some(start_timestamp),
            end_timestamp: Some(end_timestamp),
            duration_ticks,
            duration_ms_scaled: duration_ticks as f64 * scale_ms_per_tick,
        });
    }
    let accounted_ticks = stage_ticks as i128 + gap_ticks as i128 - overlap_ticks as i128;
    if accounted_ticks != sampled_span_ticks as i128 {
        return invalid(format!(
            "packed prefill sampled layer {layer} stage/gap/overlap ticks {accounted_ticks} do not close span {sampled_span_ticks}"
        ));
    }
    let raw_span_ms_assuming_ns = sampled_span_ticks as f64 * 1e-6;
    Ok(PackedPrefillSampledLayerProfile {
        layer,
        command_gpu_ms,
        sampled_span_ticks,
        raw_span_ms_assuming_ns,
        raw_coverage_assuming_ns: raw_span_ms_assuming_ns / command_gpu_ms,
        encoder_gap_ms_scaled: gap_ticks as f64 * scale_ms_per_tick,
        encoder_overlap_ms_scaled: overlap_ticks as f64 * scale_ms_per_tick,
        stages,
        transitions,
    })
}

#[cfg(feature = "dsv4-diagnostics")]
struct PackedPrefillStageRecorder {
    sampled: bool,
    samples: Option<MetalTimestampSampleBuffer>,
    next_sample: usize,
    records: Vec<PackedPrefillPendingStageSample>,
    command_gpu_ms: [Option<f64>; DEEPSEEK_V4_LAYER_COUNT],
}

#[cfg(feature = "dsv4-diagnostics")]
impl PackedPrefillStageRecorder {
    fn new(ctx: &MetalContext, sampled: bool) -> Result<Self, DeepSeekV4MetalError> {
        let record_count = DEEPSEEK_V4_LAYER_COUNT
            .checked_mul(PACKED_PREFILL_STAGE_KINDS.len())
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "packed prefill timestamp record count overflow".into(),
                )
            })?;
        let sample_count = record_count.checked_mul(2).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed prefill timestamp sample count overflow".into())
        })?;
        Ok(Self {
            sampled,
            samples: if sampled {
                Some(ctx.timestamp_sample_buffer(sample_count)?)
            } else {
                None
            },
            next_sample: 0,
            records: Vec::with_capacity(if sampled { record_count } else { 0 }),
            command_gpu_ms: [None; DEEPSEEK_V4_LAYER_COUNT],
        })
    }

    fn begin_encoder(
        &mut self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        layer: usize,
        kind: PackedPrefillStageKind,
    ) -> Result<KernelEncoder, DeepSeekV4MetalError> {
        if !self.sampled || layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid("packed prefill stage recorder received an invalid sampled layer");
        }
        let start_sample = self.next_sample;
        let end_sample = start_sample.checked_add(1).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed prefill sample index overflow".into())
        })?;
        let samples = self.samples.as_ref().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed prefill timestamp buffer is absent".into())
        })?;
        if end_sample >= samples.sample_count() {
            return invalid(format!(
                "packed prefill timestamp buffer exhausted at sample {end_sample}"
            ));
        }
        self.next_sample = end_sample + 1;
        self.records.push(PackedPrefillPendingStageSample {
            layer,
            kind,
            samples: Some((start_sample, end_sample)),
        });
        Ok(KernelEncoder::try_begin_sampled(
            command,
            samples,
            start_sample,
            end_sample,
            false,
        )?)
    }

    fn record_empty_stage(
        &mut self,
        layer: usize,
        kind: PackedPrefillStageKind,
    ) -> Result<(), DeepSeekV4MetalError> {
        if !self.sampled || layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid("packed prefill empty stage has invalid sampled layer");
        }
        let record_index = self.records.len();
        let expected_layer = record_index / PACKED_PREFILL_STAGE_KINDS.len();
        let expected_kind =
            PACKED_PREFILL_STAGE_KINDS[record_index % PACKED_PREFILL_STAGE_KINDS.len()];
        if layer != expected_layer || kind != expected_kind {
            return invalid(format!(
                "packed prefill empty stage {kind:?} for layer {layer} expected {expected_kind:?} for layer {expected_layer}"
            ));
        }
        self.records.push(PackedPrefillPendingStageSample {
            layer,
            kind,
            samples: None,
        });
        Ok(())
    }

    fn record_command_gpu_seconds(
        &mut self,
        layer: usize,
        command_gpu_seconds: f64,
    ) -> Result<(), DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT
            || !command_gpu_seconds.is_finite()
            || command_gpu_seconds <= 0.0
            || self.command_gpu_ms[layer].is_some()
        {
            return invalid(format!(
                "packed prefill layer {layer} has invalid or duplicate GPU duration {command_gpu_seconds}"
            ));
        }
        self.command_gpu_ms[layer] = Some(command_gpu_seconds * 1e3);
        Ok(())
    }

    fn resolve(
        self,
        ctx: &MetalContext,
    ) -> Result<PackedPrefillStageProfile, DeepSeekV4MetalError> {
        let command_gpu_ms = self
            .command_gpu_ms
            .into_iter()
            .enumerate()
            .map(|(layer, duration)| {
                duration.ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "packed prefill layer {layer} has no GPU duration"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !self.sampled {
            if self.samples.is_some() || self.next_sample != 0 || !self.records.is_empty() {
                return invalid("ordinary packed prefill profile retained sampled state");
            }
            return Ok(PackedPrefillStageProfile {
                sampled: false,
                command_gpu_ms,
                sampled_layers: Vec::new(),
            });
        }
        let expected_records = DEEPSEEK_V4_LAYER_COUNT * PACKED_PREFILL_STAGE_KINDS.len();
        let expected_samples = self
            .records
            .iter()
            .filter(|record| record.samples.is_some())
            .count()
            * 2;
        if self.records.len() != expected_records || self.next_sample != expected_samples {
            return invalid(format!(
                "packed prefill stage recorder produced {} records/{} samples, expected {expected_records}/{expected_samples}",
                self.records.len(),
                self.next_sample
            ));
        }
        let samples = self.samples.as_ref().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed prefill timestamp buffer is absent".into())
        })?;
        let timestamps = ctx.resolve_timestamp_samples(samples, self.next_sample)?;
        let mut sampled_layers = Vec::with_capacity(DEEPSEEK_V4_LAYER_COUNT);
        for (layer, &duration) in command_gpu_ms.iter().enumerate() {
            let records = &self.records[layer * PACKED_PREFILL_STAGE_KINDS.len()
                ..(layer + 1) * PACKED_PREFILL_STAGE_KINDS.len()];
            sampled_layers.push(resolve_packed_prefill_layer_stage_samples(
                layer,
                records,
                &timestamps,
                duration,
                &[
                    PackedPrefillStageKind::SparseIndexerPrepare,
                    PackedPrefillStageKind::SparseIndexerScore,
                    PackedPrefillStageKind::SparseSelection,
                ],
            )?);
        }
        Ok(PackedPrefillStageProfile {
            sampled: true,
            command_gpu_ms,
            sampled_layers,
        })
    }
}

#[cfg(feature = "dsv4-diagnostics")]
struct PackedPrefillLayerEncoder<'command, 'recorder> {
    command: &'command Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    layer: usize,
    sampled: bool,
    recorder: Option<&'recorder mut PackedPrefillStageRecorder>,
    encoder: Option<KernelEncoder>,
}

#[cfg(feature = "dsv4-diagnostics")]
impl<'command, 'recorder> PackedPrefillLayerEncoder<'command, 'recorder> {
    fn begin(
        command: &'command Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        layer: usize,
        mut recorder: Option<&'recorder mut PackedPrefillStageRecorder>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        let sampled = recorder.as_deref().is_some_and(|recorder| recorder.sampled);
        let encoder = if sampled {
            recorder.as_deref_mut().unwrap().begin_encoder(
                command,
                layer,
                PackedPrefillStageKind::BeforeAttentionBody,
            )?
        } else {
            KernelEncoder::begin(command)
        };
        Ok(Self {
            command,
            layer,
            sampled,
            recorder,
            encoder: Some(encoder),
        })
    }

    fn boundary(&mut self, next: PackedPrefillStageKind) -> Result<(), DeepSeekV4MetalError> {
        if !self.sampled {
            return Ok(());
        }
        if let Some(encoder) = self.encoder.take() {
            encoder.end();
        }
        self.encoder = Some(
            self.recorder
                .as_deref_mut()
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "sampled packed prefill encoder lost its recorder".into(),
                    )
                })?
                .begin_encoder(self.command, self.layer, next)?,
        );
        Ok(())
    }

    fn skip_stages(
        &mut self,
        skipped: &[PackedPrefillStageKind],
        next: PackedPrefillStageKind,
    ) -> Result<(), DeepSeekV4MetalError> {
        if !self.sampled {
            return Ok(());
        }
        if let Some(encoder) = self.encoder.take() {
            encoder.end();
        }
        let recorder = self.recorder.as_deref_mut().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("sampled packed prefill encoder lost its recorder".into())
        })?;
        for &kind in skipped {
            recorder.record_empty_stage(self.layer, kind)?;
        }
        self.encoder = Some(recorder.begin_encoder(self.command, self.layer, next)?);
        Ok(())
    }

    fn end(mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.end();
        }
    }
}

#[cfg(feature = "dsv4-diagnostics")]
impl std::ops::Deref for PackedPrefillLayerEncoder<'_, '_> {
    type Target = KernelEncoder;

    fn deref(&self) -> &Self::Target {
        self.encoder
            .as_ref()
            .expect("packed prefill layer encoder ended before stage completion")
    }
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackedPostRouteStageKind {
    RoutedExperts,
    RoutedGateUp,
    RoutedSwiGlu,
    RoutedDown,
    SharedExpert,
    ExpertCombine,
    HyperPostAndHead,
}

#[cfg(feature = "dsv4-diagnostics")]
pub(super) const PACKED_POST_ROUTE_STAGE_KINDS: [PackedPostRouteStageKind; 4] = [
    PackedPostRouteStageKind::RoutedExperts,
    PackedPostRouteStageKind::SharedExpert,
    PackedPostRouteStageKind::ExpertCombine,
    PackedPostRouteStageKind::HyperPostAndHead,
];

#[cfg(feature = "dsv4-diagnostics")]
const PACKED_BM16_POST_ROUTE_STAGE_KINDS: [PackedPostRouteStageKind; 6] = [
    PackedPostRouteStageKind::RoutedGateUp,
    PackedPostRouteStageKind::RoutedSwiGlu,
    PackedPostRouteStageKind::RoutedDown,
    PackedPostRouteStageKind::SharedExpert,
    PackedPostRouteStageKind::ExpertCombine,
    PackedPostRouteStageKind::HyperPostAndHead,
];

#[cfg(feature = "dsv4-diagnostics")]
fn packed_post_route_stage_kinds(bm16: bool) -> &'static [PackedPostRouteStageKind] {
    if bm16 {
        &PACKED_BM16_POST_ROUTE_STAGE_KINDS
    } else {
        &PACKED_POST_ROUTE_STAGE_KINDS
    }
}

#[cfg(feature = "dsv4-diagnostics")]
fn packed_post_route_expert_counts(
    n_tokens: usize,
    schedule: &[ExpertBucket],
) -> Result<[u16; MOE_EXPERT_COUNT], DeepSeekV4MetalError> {
    checked_token_count(n_tokens)?;
    let expected = checked_mul(n_tokens, MOE_TOP_K, "packed expert-count coverage")?;
    let mut counts = [0u16; MOE_EXPERT_COUNT];
    let mut covered = 0usize;
    let mut previous_expert = None;
    for bucket in schedule {
        if bucket.expert >= MOE_EXPERT_COUNT
            || bucket.len == 0
            || bucket.len > n_tokens
            || bucket.start != covered
            || previous_expert.is_some_and(|previous| bucket.expert <= previous)
        {
            return invalid("packed expert-count schedule is not canonical");
        }
        counts[bucket.expert] = u16::try_from(bucket.len)
            .map_err(|_| DeepSeekV4MetalError::Invalid("packed expert count exceeds u16".into()))?;
        covered = covered
            .checked_add(bucket.len)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("packed expert count overflow".into()))?;
        previous_expert = Some(bucket.expert);
    }
    if covered != expected {
        return invalid(format!(
            "packed expert-count coverage {covered} does not match {expected}"
        ));
    }
    Ok(counts)
}

#[cfg(feature = "dsv4-diagnostics")]
fn packed_post_route_expert_ids(
    n_tokens: usize,
    expert_ids: &[i32],
    bucket_rows: &[i32],
    bucket_slots: &[i32],
    schedule: &[ExpertBucket],
) -> Result<Vec<u16>, DeepSeekV4MetalError> {
    validate_packed_expert_schedule(n_tokens, expert_ids, bucket_rows, bucket_slots, schedule)?;
    let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed route-ID coverage")?;
    let mut reconstructed = vec![u16::MAX; route_count];
    for bucket in schedule {
        for &stored_slot in &bucket_slots[bucket.start..bucket.start + bucket.len] {
            let slot = usize::try_from(stored_slot).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed route-ID slot is negative".into())
            })?;
            if slot >= route_count || reconstructed[slot] != u16::MAX {
                return invalid("packed route-ID slot coverage is invalid");
            }
            reconstructed[slot] = u16::try_from(bucket.expert).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed route-ID expert exceeds u16".into())
            })?;
        }
    }
    if reconstructed.contains(&u16::MAX) {
        return invalid("packed route-ID capture omitted a slot");
    }
    for (&reconstructed, &stored) in reconstructed.iter().zip(expert_ids) {
        if usize::from(reconstructed) >= MOE_EXPERT_COUNT || i32::from(reconstructed) != stored {
            return invalid("packed route-ID capture differs from routed storage");
        }
    }
    for token_ids in reconstructed.chunks_exact(MOE_TOP_K) {
        let mut seen = [false; MOE_EXPERT_COUNT];
        for &expert in token_ids {
            if std::mem::replace(&mut seen[usize::from(expert)], true) {
                return invalid("packed route-ID capture duplicates an expert within a token");
            }
        }
    }

    let mut rebuilt_rows = Vec::with_capacity(route_count);
    let mut rebuilt_slots = Vec::with_capacity(route_count);
    let mut rebuilt_schedule = Vec::with_capacity(schedule.len());
    for expert in 0..MOE_EXPERT_COUNT {
        let start = rebuilt_slots.len();
        for (slot, &routed_expert) in reconstructed.iter().enumerate() {
            if usize::from(routed_expert) == expert {
                rebuilt_rows.push((slot / MOE_TOP_K) as i32);
                rebuilt_slots.push(slot as i32);
            }
        }
        if rebuilt_slots.len() > start {
            rebuilt_schedule.push(ExpertBucket {
                expert,
                start,
                len: rebuilt_slots.len() - start,
            });
        }
    }
    if rebuilt_rows != bucket_rows
        || rebuilt_slots != bucket_slots
        || rebuilt_schedule.len() != schedule.len()
    {
        return invalid("packed route-ID capture does not rebuild the routed schedule");
    }
    for (rebuilt, original) in rebuilt_schedule.iter().zip(schedule) {
        if rebuilt.expert != original.expert
            || rebuilt.start != original.start
            || rebuilt.len != original.len
        {
            return invalid("packed route-ID capture changes expert bucket geometry");
        }
    }
    Ok(reconstructed)
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedPostRouteLayerMetadata {
    pub layer: usize,
    pub gate_dtype: GgmlType,
    pub up_dtype: GgmlType,
    pub down_dtype: GgmlType,
    pub bucket_count: usize,
    pub expert_counts: [u16; MOE_EXPERT_COUNT],
    pub route_expert_ids: Vec<u16>,
    pub grouped_iq2: bool,
    pub grouped_iq3: bool,
    pub bm16: bool,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct PackedPostRouteStageTiming {
    pub kind: PackedPostRouteStageKind,
    pub start_timestamp: u64,
    pub end_timestamp: u64,
    pub duration_ticks: u64,
    pub duration_ms_scaled: f64,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct PackedPostRouteSampledLayerProfile {
    pub layer: usize,
    pub command_gpu_ms: f64,
    pub sampled_span_ticks: u64,
    pub raw_span_ms_assuming_ns: f64,
    pub raw_coverage_assuming_ns: f64,
    pub encoder_gap_ms_scaled: f64,
    pub encoder_overlap_ms_scaled: f64,
    pub stages: Vec<PackedPostRouteStageTiming>,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct PackedPostRouteStageProfile {
    pub sampled: bool,
    pub q8_compressor_matrix_invocations: u32,
    pub command_gpu_ms: Vec<f64>,
    pub metadata: Vec<PackedPostRouteLayerMetadata>,
    pub sampled_layers: Vec<PackedPostRouteSampledLayerProfile>,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct PackedChunkProfile {
    pub pre_expert: PackedPrefillStageProfile,
    pub post_route: PackedPostRouteStageProfile,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug)]
struct PackedPostRoutePendingStageSample {
    layer: usize,
    kind: PackedPostRouteStageKind,
    start_sample: usize,
    end_sample: usize,
}

#[cfg(feature = "dsv4-diagnostics")]
fn resolve_packed_post_route_layer_stage_samples(
    layer: usize,
    records: &[PackedPostRoutePendingStageSample],
    expected_kinds: &[PackedPostRouteStageKind],
    timestamps: &[u64],
    command_gpu_ms: f64,
) -> Result<PackedPostRouteSampledLayerProfile, DeepSeekV4MetalError> {
    if records.len() != expected_kinds.len() {
        return invalid(format!(
            "packed post-route sampled layer {layer} produced {} stages, expected {}",
            records.len(),
            expected_kinds.len()
        ));
    }
    if !command_gpu_ms.is_finite() || command_gpu_ms <= 0.0 {
        return invalid(format!(
            "packed post-route sampled layer {layer} has invalid command GPU duration {command_gpu_ms}"
        ));
    }
    for (record, &expected) in records.iter().zip(expected_kinds) {
        if record.layer != layer || record.kind != expected {
            return invalid(format!(
                "packed post-route sampled layer {layer} recorded {:?} for layer {}, expected {expected:?}",
                record.kind, record.layer
            ));
        }
        if record.start_sample >= timestamps.len() || record.end_sample >= timestamps.len() {
            return invalid(format!(
                "packed post-route sampled layer {layer} stage {:?} indexes samples {}..{} from {} timestamps",
                record.kind,
                record.start_sample,
                record.end_sample,
                timestamps.len()
            ));
        }
    }

    let first_timestamp = timestamps[records[0].start_sample];
    let last_timestamp = timestamps[records[records.len() - 1].end_sample];
    let sampled_span_ticks = last_timestamp.checked_sub(first_timestamp).ok_or_else(|| {
        DeepSeekV4MetalError::Invalid(format!(
            "packed post-route sampled layer {layer} returned non-monotonic span timestamps"
        ))
    })?;
    if sampled_span_ticks == 0 {
        return invalid(format!(
            "packed post-route sampled layer {layer} returned a zero timestamp span"
        ));
    }

    let scale_ms_per_tick = command_gpu_ms / sampled_span_ticks as f64;
    let mut stage_ticks = 0u64;
    let mut gap_ticks = 0u64;
    let mut overlap_ticks = 0u64;
    let mut previous_start = None;
    let mut previous_end = None;
    let mut stages = Vec::with_capacity(records.len());
    for record in records {
        let start_timestamp = timestamps[record.start_sample];
        let end_timestamp = timestamps[record.end_sample];
        let duration_ticks = end_timestamp.checked_sub(start_timestamp).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "packed post-route sampled layer {layer} stage {:?} returned inverted timestamps",
                record.kind
            ))
        })?;
        if previous_start.is_some_and(|previous| start_timestamp < previous)
            || previous_end.is_some_and(|previous| end_timestamp < previous)
        {
            return invalid(format!(
                "packed post-route sampled layer {layer} stage {:?} reverses physical start/end order",
                record.kind
            ));
        }
        if let Some(previous_end) = previous_end {
            if start_timestamp >= previous_end {
                gap_ticks = gap_ticks
                    .checked_add(start_timestamp - previous_end)
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(
                            "packed post-route encoder-gap tick total overflow".into(),
                        )
                    })?;
            } else {
                overlap_ticks = overlap_ticks
                    .checked_add(previous_end - start_timestamp)
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(
                            "packed post-route encoder-overlap tick total overflow".into(),
                        )
                    })?;
            }
        }
        previous_start = Some(start_timestamp);
        previous_end = Some(end_timestamp);
        stage_ticks = stage_ticks.checked_add(duration_ticks).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "packed post-route sampled stage tick total overflow".into(),
            )
        })?;
        stages.push(PackedPostRouteStageTiming {
            kind: record.kind,
            start_timestamp,
            end_timestamp,
            duration_ticks,
            duration_ms_scaled: duration_ticks as f64 * scale_ms_per_tick,
        });
    }
    let accounted_ticks = stage_ticks as i128 + gap_ticks as i128 - overlap_ticks as i128;
    if accounted_ticks != sampled_span_ticks as i128 {
        return invalid(format!(
            "packed post-route sampled layer {layer} stage/gap/overlap ticks {accounted_ticks} do not close span {sampled_span_ticks}"
        ));
    }
    let raw_span_ms_assuming_ns = sampled_span_ticks as f64 * 1e-6;
    Ok(PackedPostRouteSampledLayerProfile {
        layer,
        command_gpu_ms,
        sampled_span_ticks,
        raw_span_ms_assuming_ns,
        raw_coverage_assuming_ns: raw_span_ms_assuming_ns / command_gpu_ms,
        encoder_gap_ms_scaled: gap_ticks as f64 * scale_ms_per_tick,
        encoder_overlap_ms_scaled: overlap_ticks as f64 * scale_ms_per_tick,
        stages,
    })
}

#[cfg(feature = "dsv4-diagnostics")]
struct PackedPostRouteStageRecorder {
    sampled: bool,
    samples: Option<MetalTimestampSampleBuffer>,
    next_sample: usize,
    records: Vec<PackedPostRoutePendingStageSample>,
    command_gpu_ms: [Option<f64>; DEEPSEEK_V4_LAYER_COUNT],
    metadata: [Option<PackedPostRouteLayerMetadata>; DEEPSEEK_V4_LAYER_COUNT],
}

#[cfg(feature = "dsv4-diagnostics")]
impl PackedPostRouteStageRecorder {
    fn new(ctx: &MetalContext, sampled: bool) -> Result<Self, DeepSeekV4MetalError> {
        let record_count = DEEPSEEK_V4_LAYER_COUNT
            .checked_mul(PACKED_BM16_POST_ROUTE_STAGE_KINDS.len())
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "packed post-route timestamp record count overflow".into(),
                )
            })?;
        let sample_count = record_count.checked_mul(2).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "packed post-route timestamp sample count overflow".into(),
            )
        })?;
        Ok(Self {
            sampled,
            samples: if sampled {
                Some(ctx.timestamp_sample_buffer(sample_count)?)
            } else {
                None
            },
            next_sample: 0,
            records: Vec::with_capacity(if sampled { record_count } else { 0 }),
            command_gpu_ms: [None; DEEPSEEK_V4_LAYER_COUNT],
            metadata: std::array::from_fn(|_| None),
        })
    }

    fn begin_encoder(
        &mut self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        layer: usize,
        kind: PackedPostRouteStageKind,
    ) -> Result<KernelEncoder, DeepSeekV4MetalError> {
        if !self.sampled || layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid("packed post-route stage recorder received an invalid sampled layer");
        }
        let start_sample = self.next_sample;
        let end_sample = start_sample.checked_add(1).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed post-route sample index overflow".into())
        })?;
        let samples = self.samples.as_ref().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed post-route timestamp buffer is absent".into())
        })?;
        if end_sample >= samples.sample_count() {
            return invalid(format!(
                "packed post-route timestamp buffer exhausted at sample {end_sample}"
            ));
        }
        self.next_sample = end_sample + 1;
        self.records.push(PackedPostRoutePendingStageSample {
            layer,
            kind,
            start_sample,
            end_sample,
        });
        Ok(KernelEncoder::try_begin_sampled(
            command,
            samples,
            start_sample,
            end_sample,
            false,
        )?)
    }

    fn record_layer(
        &mut self,
        metadata: PackedPostRouteLayerMetadata,
    ) -> Result<(), DeepSeekV4MetalError> {
        let layer = metadata.layer;
        if layer >= DEEPSEEK_V4_LAYER_COUNT || self.metadata[layer].replace(metadata).is_some() {
            return invalid(format!(
                "packed post-route layer {layer} has invalid or duplicate metadata"
            ));
        }
        Ok(())
    }

    fn record_command_gpu_seconds(
        &mut self,
        layer: usize,
        command_gpu_seconds: f64,
    ) -> Result<(), DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT
            || !command_gpu_seconds.is_finite()
            || command_gpu_seconds <= 0.0
            || self.command_gpu_ms[layer].is_some()
        {
            return invalid(format!(
                "packed post-route layer {layer} has invalid or duplicate GPU duration {command_gpu_seconds}"
            ));
        }
        self.command_gpu_ms[layer] = Some(command_gpu_seconds * 1e3);
        Ok(())
    }

    fn resolve(
        self,
        ctx: &MetalContext,
    ) -> Result<PackedPostRouteStageProfile, DeepSeekV4MetalError> {
        let command_gpu_ms = self
            .command_gpu_ms
            .into_iter()
            .enumerate()
            .map(|(layer, duration)| {
                duration.ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "packed post-route layer {layer} has no GPU duration"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let metadata = self
            .metadata
            .into_iter()
            .enumerate()
            .map(|(layer, metadata)| {
                metadata.ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "packed post-route layer {layer} has no metadata"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !self.sampled {
            if self.samples.is_some() || self.next_sample != 0 || !self.records.is_empty() {
                return invalid("ordinary packed post-route profile retained sampled state");
            }
            return Ok(PackedPostRouteStageProfile {
                sampled: false,
                q8_compressor_matrix_invocations: 0,
                command_gpu_ms,
                metadata,
                sampled_layers: Vec::new(),
            });
        }
        let expected_records = metadata
            .iter()
            .map(|metadata| packed_post_route_stage_kinds(metadata.bm16).len())
            .sum::<usize>();
        let expected_samples = expected_records * 2;
        if self.records.len() != expected_records || self.next_sample != expected_samples {
            return invalid(format!(
                "packed post-route stage recorder produced {} records/{} samples, expected {expected_records}/{expected_samples}",
                self.records.len(),
                self.next_sample
            ));
        }
        let samples = self.samples.as_ref().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed post-route timestamp buffer is absent".into())
        })?;
        let timestamps = ctx.resolve_timestamp_samples(samples, self.next_sample)?;
        let mut sampled_layers = Vec::with_capacity(DEEPSEEK_V4_LAYER_COUNT);
        let mut record_cursor = 0usize;
        for (layer, &duration) in command_gpu_ms.iter().enumerate() {
            let expected_kinds = packed_post_route_stage_kinds(metadata[layer].bm16);
            let record_end = record_cursor + expected_kinds.len();
            let records = &self.records[record_cursor..record_end];
            sampled_layers.push(resolve_packed_post_route_layer_stage_samples(
                layer,
                records,
                expected_kinds,
                &timestamps,
                duration,
            )?);
            record_cursor = record_end;
        }
        Ok(PackedPostRouteStageProfile {
            sampled: true,
            q8_compressor_matrix_invocations: 0,
            command_gpu_ms,
            metadata,
            sampled_layers,
        })
    }
}

struct PackedPostRouteLayerEncoder<'a> {
    _command: &'a Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    #[cfg(feature = "dsv4-diagnostics")]
    layer: usize,
    #[cfg(feature = "dsv4-diagnostics")]
    sampled: bool,
    #[cfg(feature = "dsv4-diagnostics")]
    split_bm16: bool,
    #[cfg(feature = "dsv4-diagnostics")]
    recorder: Option<&'a mut PackedPostRouteStageRecorder>,
    encoder: Option<KernelEncoder>,
}

impl<'a> PackedPostRouteLayerEncoder<'a> {
    #[cfg(feature = "dsv4-diagnostics")]
    fn begin(
        command: &'a Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        layer: usize,
        split_bm16: bool,
        mut recorder: Option<&'a mut PackedPostRouteStageRecorder>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        let sampled = recorder.as_deref().is_some_and(|recorder| recorder.sampled);
        let encoder = if sampled {
            recorder.as_deref_mut().unwrap().begin_encoder(
                command,
                layer,
                if split_bm16 {
                    PackedPostRouteStageKind::RoutedGateUp
                } else {
                    PackedPostRouteStageKind::RoutedExperts
                },
            )?
        } else {
            KernelEncoder::begin(command)
        };
        Ok(Self {
            _command: command,
            layer,
            sampled,
            split_bm16,
            recorder,
            encoder: Some(encoder),
        })
    }

    #[cfg(not(feature = "dsv4-diagnostics"))]
    fn begin(
        command: &'a Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        Ok(Self {
            _command: command,
            encoder: Some(KernelEncoder::begin(command)),
        })
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn boundary(&mut self, next: PackedPostRouteStageKind) -> Result<(), DeepSeekV4MetalError> {
        if !self.sampled {
            return Ok(());
        }
        if let Some(encoder) = self.encoder.take() {
            encoder.end();
        }
        self.encoder = Some(
            self.recorder
                .as_deref_mut()
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "sampled packed post-route encoder lost its recorder".into(),
                    )
                })?
                .begin_encoder(self._command, self.layer, next)?,
        );
        Ok(())
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn splits_bm16_stages(&self) -> bool {
        self.sampled && self.split_bm16
    }

    fn end(mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.end();
        }
    }
}

impl std::ops::Deref for PackedPostRouteLayerEncoder<'_> {
    type Target = KernelEncoder;

    fn deref(&self) -> &Self::Target {
        self.encoder
            .as_ref()
            .expect("packed post-route layer encoder ended before stage completion")
    }
}

#[derive(Clone, Copy)]
enum PackedRouteSource<'a> {
    Hash,
    Learned(&'a MetalTensor),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PackedRoutePolicy {
    Cpu,
    GpuCompact,
    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    GpuExperimental,
    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    GpuExperimentalCpuWeights,
}

impl PackedRoutePolicy {
    fn uses_gpu(self) -> bool {
        match self {
            Self::Cpu => false,
            Self::GpuCompact => true,
            #[cfg(all(test, feature = "dsv4-diagnostics"))]
            Self::GpuExperimental => true,
            #[cfg(all(test, feature = "dsv4-diagnostics"))]
            Self::GpuExperimentalCpuWeights => true,
        }
    }
}

#[cfg(feature = "dsv4-diagnostics")]
fn validate_packed_route_policy_scope(
    policy: PackedRoutePolicy,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if policy.uses_gpu() && n_tokens > PACKED_GPU_ROUTE_MAX_TOKENS {
        return invalid(format!(
            "experimental packed GPU routing is qualified through {PACKED_GPU_ROUTE_MAX_TOKENS} tokens"
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PackedExpertPolicy {
    Current,
    GroupedIq2XsIq3Xxs,
    GroupedIq2XsIq3XxsMma16QualifiedChunk,
    GroupedIq2XsIq3XxsAndIq3Xxs,
    GroupedIq2XsIq3XxsMma16AndIq3XxsQualifiedChunk,
}

impl PackedExpertPolicy {
    fn uses_iq2_target(self) -> bool {
        match self {
            Self::Current => false,
            Self::GroupedIq2XsIq3Xxs => true,
            Self::GroupedIq2XsIq3XxsMma16QualifiedChunk => true,
            Self::GroupedIq2XsIq3XxsAndIq3Xxs => true,
            Self::GroupedIq2XsIq3XxsMma16AndIq3XxsQualifiedChunk => true,
        }
    }

    fn uses_iq2_mma16(self, n_tokens: usize) -> bool {
        matches!(
            self,
            Self::GroupedIq2XsIq3XxsMma16QualifiedChunk
                | Self::GroupedIq2XsIq3XxsMma16AndIq3XxsQualifiedChunk
        ) && packed_grouped_iq2_mma16_qualified(n_tokens)
    }

    fn uses_iq3_target(self) -> bool {
        matches!(
            self,
            Self::GroupedIq2XsIq3XxsAndIq3Xxs
                | Self::GroupedIq2XsIq3XxsMma16AndIq3XxsQualifiedChunk
        )
    }

    fn with_iq3_target(self) -> Self {
        match self {
            Self::GroupedIq2XsIq3Xxs => Self::GroupedIq2XsIq3XxsAndIq3Xxs,
            Self::GroupedIq2XsIq3XxsMma16QualifiedChunk => {
                Self::GroupedIq2XsIq3XxsMma16AndIq3XxsQualifiedChunk
            }
            other => other,
        }
    }
}

fn packed_gpu_compact_expert_layer_qualified(
    ctx: &MetalContext,
    policy: PackedExpertPolicy,
    n_tokens: usize,
    allow_iq3_route: bool,
    gate_dtype: GgmlType,
    up_dtype: GgmlType,
    down_dtype: GgmlType,
) -> bool {
    (policy.uses_iq2_mma16(n_tokens)
        && gate_dtype == GgmlType::IQ2_XS
        && up_dtype == GgmlType::IQ2_XS
        && down_dtype == GgmlType::IQ3_XXS
        && packed_grouped_expert_kernels_supported(ctx))
        || (allow_iq3_route
            && policy.uses_iq3_target()
            && gate_dtype == GgmlType::IQ3_XXS
            && up_dtype == GgmlType::IQ3_XXS
            && down_dtype == GgmlType::IQ3_XXS
            && packed_grouped_iq3_candidate_supported(ctx))
}

impl PrefillMoeScratch {
    fn gpu_route_buffers<'a>(
        &'a self,
        logits: &'a MetalTensor,
        token_ids: &'a MetalTensor,
    ) -> PackedGpuRouteBuffers<'a> {
        PackedGpuRouteBuffers {
            logits,
            token_ids,
            expert_ids: &self.expert_ids,
            weights: &self.weights,
            route_generations: &self.gpu_route.route_generations,
            route_status: &self.gpu_route.route_status,
            counts: &self.gpu_route.counts,
            #[cfg(any(test, feature = "dsv4-diagnostics"))]
            slot_ids: &self.gpu_route.slot_ids,
            #[cfg(any(test, feature = "dsv4-diagnostics"))]
            schedule_generations: &self.gpu_route.schedule_generations,
            #[cfg(any(test, feature = "dsv4-diagnostics"))]
            aggregate: &self.gpu_route.aggregate,
            #[cfg(any(test, feature = "dsv4-diagnostics"))]
            signature: &self.gpu_route.signature,
            compact_header: &self.gpu_route.compact_header,
        }
    }

    fn take_gpu_route_generation(&self) -> Result<NonZeroU32, DeepSeekV4MetalError> {
        let generation =
            NonZeroU32::new(self.gpu_route.next_generation.get()).ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("packed GPU route generation reached zero".into())
            })?;
        let next = generation.get().checked_add(1).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "packed GPU route generation exhausted before wrap".into(),
            )
        })?;
        self.gpu_route.next_generation.set(next);
        Ok(generation)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_gpu_route_compact(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        views: &PackedMoeViews,
        source: PackedRouteSource<'_>,
        token_ids: &MetalTensor,
        token_to_expert: Option<&MetalTensor>,
        n_tokens: usize,
        routed_scale: f32,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let buffers = self.gpu_route_buffers(&views.logits, token_ids);
        match source {
            PackedRouteSource::Hash => buffers.encode_hash(
                ctx,
                enc,
                token_to_expert.ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "packed compact hash route has no token-to-expert map".into(),
                    )
                })?,
                n_tokens,
                n_tokens,
                generation,
                routed_scale,
            )?,
            PackedRouteSource::Learned(bias) => buffers.encode_learned(
                ctx,
                enc,
                bias,
                n_tokens,
                n_tokens,
                generation,
                routed_scale,
            )?,
        }
        let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed compact route count")?;
        let rows = i32_prefix(
            &self.bucket_rows,
            vec![route_count as u64],
            "packed compact route rows",
        )?;
        let slots = i32_prefix(
            &self.bucket_slots,
            vec![route_count as u64],
            "packed compact route slots",
        )?;
        buffers.encode_compact(
            ctx,
            enc,
            &rows,
            &slots,
            &self.grouped_tiles,
            &self.grouped_iq2_mma16_tiles,
            n_tokens,
            generation,
        )
    }

    fn capture_gpu_compact_schedule(
        &self,
        n_tokens: usize,
        generation: NonZeroU32,
    ) -> Result<Vec<ExpertBucket>, DeepSeekV4MetalError> {
        let header = host_read_i32(
            &self.gpu_route.compact_header,
            "packed compact route header",
        )?;
        let expected_completion = packed_route_compact_completion(generation.get(), n_tokens);
        if header.len() != PACKED_COMPACT_ROUTE_HEADER_WIDTH
            || header[0] != generation.get() as i32
            || header[1] != PACKED_COMPACT_ROUTE_STATUS_READY
            || header[2] != (n_tokens * MOE_TOP_K) as i32
            || header[3] < 0
            || header[3] as usize > MOE_EXPERT_COUNT
            || header[4] < 0
            || header[4] as usize > PACKED_GROUPED_IQ2_MMA16_MAX_TILES
            || header[5] < 0
            || header[5] as usize > PACKED_GROUPED_EXPERT_MAX_TILES
            || header[6] as u32 != expected_completion
            || header[7] != n_tokens as i32
        {
            return invalid(format!(
                "packed compact route header {header:?} is invalid for generation {} and N={n_tokens}",
                generation.get(),
            ));
        }
        let counts = host_read_i32(&self.gpu_route.counts, "packed compact expert counts")?;
        if counts.len() != MOE_EXPERT_COUNT {
            return invalid("packed compact expert counts have invalid length");
        }
        let mut cursor = 0usize;
        let mut schedule = Vec::with_capacity(header[3].max(0) as usize);
        let mut tile16_count = 0usize;
        let mut tile32_count = 0usize;
        for (expert, count) in counts.into_iter().enumerate() {
            let count = usize::try_from(count).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!(
                    "packed compact expert {expert} has negative count"
                ))
            })?;
            if count > n_tokens {
                return invalid(format!(
                    "packed compact expert {expert} count {count} exceeds N={n_tokens}"
                ));
            }
            if count != 0 {
                schedule.push(ExpertBucket {
                    expert,
                    start: cursor,
                    len: count,
                });
                tile16_count += count.div_ceil(PACKED_GROUPED_IQ2_MMA16_TILE_ROWS);
                tile32_count += count.div_ceil(PACKED_GROUPED_EXPERT_TILE_ROWS);
            }
            cursor = cursor.checked_add(count).ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("packed compact route count overflow".into())
            })?;
        }
        if cursor != n_tokens * MOE_TOP_K
            || schedule.len() != header[3] as usize
            || tile16_count != header[4] as usize
            || tile32_count != header[5] as usize
        {
            return invalid(format!(
                "packed compact schedule has {cursor} routes, {} experts, {tile16_count} 16-row tiles, and {tile32_count} 32-row tiles",
                schedule.len(),
            ));
        }
        Ok(schedule)
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "dsv4-diagnostics")]
    fn encode_gpu_route_schedule(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        views: &PackedMoeViews,
        source: PackedRouteSource<'_>,
        token_ids: &MetalTensor,
        token_to_expert: Option<&MetalTensor>,
        n_tokens: usize,
        routed_scale: f32,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let buffers = self.gpu_route_buffers(&views.logits, token_ids);
        match source {
            PackedRouteSource::Hash => buffers.encode_hash(
                ctx,
                enc,
                token_to_expert.ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "packed GPU hash route has no token-to-expert map".into(),
                    )
                })?,
                n_tokens,
                n_tokens,
                generation,
                routed_scale,
            )?,
            PackedRouteSource::Learned(bias) => buffers.encode_learned(
                ctx,
                enc,
                bias,
                n_tokens,
                n_tokens,
                generation,
                routed_scale,
            )?,
        }
        buffers.encode_schedule(ctx, enc, n_tokens, MOE_EXPERT_COUNT, generation)?;
        buffers.encode_validate(ctx, enc, n_tokens, generation)?;
        buffers.encode_signature(ctx, enc, n_tokens, generation)
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn capture_gpu_route_schedule(
        &self,
        n_tokens: usize,
        generation: NonZeroU32,
    ) -> Result<Vec<ExpertBucket>, DeepSeekV4MetalError> {
        checked_token_count(n_tokens)?;
        let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed GPU route count")?;
        let schedule_count = checked_mul(n_tokens, MOE_EXPERT_COUNT, "packed GPU schedule count")?;
        let mut expert_ids = host_read_i32(&self.expert_ids, "packed GPU route IDs")?;
        let mut weights = host_read_f32(&self.weights, "packed GPU route weights")?;
        let mut route_generations = host_read_i32(
            &self.gpu_route.route_generations,
            "packed GPU route generations",
        )?;
        let mut route_status =
            host_read_i32(&self.gpu_route.route_status, "packed GPU route statuses")?;
        let counts = host_read_i32(&self.gpu_route.counts, "packed GPU route counts")?;
        let mut slot_ids = host_read_i32(&self.gpu_route.slot_ids, "packed GPU route slot IDs")?;
        let schedule_generations = host_read_i32(
            &self.gpu_route.schedule_generations,
            "packed GPU schedule generations",
        )?;
        let aggregate = host_read_i32(&self.gpu_route.aggregate, "packed GPU route aggregate")?;
        let signature = host_read_i32(&self.gpu_route.signature, "packed GPU route signature")?;
        expert_ids.truncate(route_count);
        weights.truncate(route_count);
        route_generations.truncate(n_tokens);
        route_status.truncate(n_tokens);
        slot_ids.truncate(schedule_count);

        let generation_i32 = generation.get() as i32;
        if route_generations
            .iter()
            .any(|&value| value != generation_i32)
        {
            return invalid("packed GPU route contains a stale token producer");
        }
        if let Some((token, &status)) = route_status
            .iter()
            .enumerate()
            .find(|(_, status)| **status != DEEPSEEK_V4_ROUTE_STATUS_READY)
        {
            return invalid(format!(
                "packed GPU route token {token} failed with status {status}"
            ));
        }
        if schedule_generations
            .iter()
            .any(|&value| value != generation_i32)
        {
            return invalid("packed GPU route contains a stale schedule producer");
        }
        let expected_aggregate = [
            generation_i32,
            DEEPSEEK_V4_ROUTE_STATUS_READY,
            route_count as i32,
            packed_route_completion(generation.get(), n_tokens) as i32,
        ];
        if aggregate != expected_aggregate {
            return invalid(format!(
                "packed GPU route aggregate {aggregate:?} differs from {expected_aggregate:?}"
            ));
        }
        let expected_signature = [
            generation_i32,
            DEEPSEEK_V4_ROUTE_STATUS_READY,
            packed_route_signature_hash(&expert_ids, &weights, &counts, &slot_ids, n_tokens)?
                as i32,
            packed_route_signature_completion(generation.get(), n_tokens) as i32,
        ];
        if signature != expected_signature {
            return invalid(format!(
                "packed GPU route signature {signature:?} differs from {expected_signature:?}"
            ));
        }

        for token in 0..n_tokens {
            let mut seen = [false; MOE_EXPERT_COUNT];
            for slot in 0..MOE_TOP_K {
                let index = token * MOE_TOP_K + slot;
                let expert = usize::try_from(expert_ids[index]).map_err(|_| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "packed GPU route token {token} slot {slot} has negative expert {}",
                        expert_ids[index]
                    ))
                })?;
                if expert >= MOE_EXPERT_COUNT || std::mem::replace(&mut seen[expert], true) {
                    return invalid(format!(
                        "packed GPU route token {token} slot {slot} has invalid expert {expert}"
                    ));
                }
                let weight = weights[index];
                if !weight.is_finite() || weight < 0.0 {
                    return invalid(format!(
                        "packed GPU route token {token} slot {slot} has invalid weight {weight}"
                    ));
                }
            }
        }

        let mut compact_rows = Vec::with_capacity(route_count);
        let mut compact_slots = Vec::with_capacity(route_count);
        let mut schedule = Vec::new();
        for (expert, &count_i32) in counts.iter().enumerate().take(MOE_EXPERT_COUNT) {
            let count = usize::try_from(count_i32).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!(
                    "packed GPU route expert {expert} has negative count {}",
                    count_i32
                ))
            })?;
            if count > n_tokens {
                return invalid(format!(
                    "packed GPU route expert {expert} count {count} exceeds {n_tokens}"
                ));
            }
            let base = expert * n_tokens;
            let start = compact_rows.len();
            let mut expected = Vec::with_capacity(count);
            for token in 0..n_tokens {
                for slot in 0..MOE_TOP_K {
                    let global_slot = token * MOE_TOP_K + slot;
                    if expert_ids[global_slot] == expert as i32 {
                        expected.push(global_slot as i32);
                    }
                }
            }
            if expected.len() != count || slot_ids[base..base + count] != expected {
                return invalid(format!(
                    "packed GPU route expert {expert} schedule differs from token/slot order"
                ));
            }
            if slot_ids[base + count..base + n_tokens]
                .iter()
                .any(|&slot| slot != -1)
            {
                return invalid(format!(
                    "packed GPU route expert {expert} has non-sentinel padding"
                ));
            }
            for &global_slot in &expected {
                compact_rows.push(global_slot / MOE_TOP_K as i32);
                compact_slots.push(global_slot);
            }
            if count > 0 {
                schedule.push(ExpertBucket {
                    expert,
                    start,
                    len: count,
                });
            }
        }
        if compact_rows.len() != route_count {
            return invalid(format!(
                "packed GPU route schedule has {} assignments, expected {route_count}",
                compact_rows.len()
            ));
        }
        validate_packed_expert_schedule(
            n_tokens,
            &expert_ids,
            &compact_rows,
            &compact_slots,
            &schedule,
        )?;
        let rows = i32_prefix(
            &self.bucket_rows,
            vec![route_count as u64],
            "packed GPU compact route rows",
        )?;
        let slots = i32_prefix(
            &self.bucket_slots,
            vec![route_count as u64],
            "packed GPU compact route slots",
        )?;
        host_write_i32(&rows, &compact_rows, "packed GPU compact route rows")?;
        host_write_i32(&slots, &compact_slots, "packed GPU compact route slots")?;
        Ok(schedule)
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    fn audit_gpu_route_against_cpu(
        &self,
        views: &PackedMoeViews,
        source: PackedRouteSource<'_>,
        n_tokens: usize,
        layer: usize,
        chunk_start: u32,
        generation: NonZeroU32,
    ) -> Result<(Vec<i32>, Vec<f32>), DeepSeekV4MetalError> {
        use sha2::{Digest, Sha256};

        let logits = host_read_f32(&views.logits, "packed route audit logits")?;
        let hash_ids = match source {
            PackedRouteSource::Hash => Some(host_read_i32(
                views.hash_ids.as_ref().ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid("packed route audit has no hash IDs".into())
                })?,
                "packed route audit hash IDs",
            )?),
            PackedRouteSource::Learned(_) => None,
        };
        let bias = match source {
            PackedRouteSource::Hash => None,
            PackedRouteSource::Learned(bias) => {
                Some(host_read_f32(bias, "packed route audit correction bias")?)
            }
        };
        let mut gpu_ids = host_read_i32(&self.expert_ids, "packed route audit GPU IDs")?;
        let mut gpu_weights = host_read_f32(&self.weights, "packed route audit GPU weights")?;
        gpu_ids.truncate(n_tokens * MOE_TOP_K);
        gpu_weights.truncate(n_tokens * MOE_TOP_K);
        let mut cpu_ids = Vec::with_capacity(n_tokens * MOE_TOP_K);
        let mut cpu_weights = Vec::with_capacity(n_tokens * MOE_TOP_K);
        let mut minimum_cutoff_margin = f32::INFINITY;
        let mut first_id_mismatch = None;
        let mut mismatch_tokens = 0usize;
        let mut symmetric_difference = 0usize;
        let mut weight_bit_mismatches = 0usize;
        let mut maximum_weight_delta = 0.0_f32;
        for token in 0..n_tokens {
            let start = token * MOE_EXPERT_COUNT;
            let scores = crate::deepseek_v4_oracle::sqrt_softplus_scores(
                &logits[start..start + MOE_EXPERT_COUNT],
            )
            .map_err(|error| {
                DeepSeekV4MetalError::Invalid(format!(
                    "packed route audit scores for token {token}: {error}"
                ))
            })?;
            let decision = if let Some(hash_ids) = hash_ids.as_ref() {
                let start = token * MOE_TOP_K;
                let selected = hash_ids[start..start + MOE_TOP_K]
                    .iter()
                    .map(|&expert| usize::try_from(expert).expect("validated hash expert"))
                    .collect::<Vec<_>>();
                crate::deepseek_v4_oracle::hash_route(&scores, &selected, 1.5)
            } else {
                let bias = bias.as_ref().expect("learned route bias");
                let mut ranked = (0..MOE_EXPERT_COUNT).collect::<Vec<_>>();
                ranked.sort_unstable_by(|&left, &right| {
                    (scores[right] + bias[right])
                        .total_cmp(&(scores[left] + bias[left]))
                        .then_with(|| left.cmp(&right))
                });
                let margin = (scores[ranked[MOE_TOP_K - 1]] + bias[ranked[MOE_TOP_K - 1]])
                    - (scores[ranked[MOE_TOP_K]] + bias[ranked[MOE_TOP_K]]);
                minimum_cutoff_margin = minimum_cutoff_margin.min(margin);
                crate::deepseek_v4_oracle::learned_route(&scores, bias, MOE_TOP_K, 1.5)
            }
            .map_err(|error| {
                DeepSeekV4MetalError::Invalid(format!(
                    "packed route audit decision for token {token}: {error}"
                ))
            })?;
            let expected_ids = decision
                .expert_ids
                .iter()
                .map(|&expert| expert as i32)
                .collect::<Vec<_>>();
            let gpu_start = token * MOE_TOP_K;
            let actual_ids = &gpu_ids[gpu_start..gpu_start + MOE_TOP_K];
            if actual_ids != expected_ids {
                mismatch_tokens += 1;
                symmetric_difference += actual_ids
                    .iter()
                    .filter(|expert| !expected_ids.contains(expert))
                    .count()
                    + expected_ids
                        .iter()
                        .filter(|expert| !actual_ids.contains(expert))
                        .count();
                if first_id_mismatch.is_none() {
                    first_id_mismatch = Some((token, expected_ids.clone(), actual_ids.to_vec()));
                }
            }
            for (slot, &expected) in decision.weights.iter().enumerate() {
                let actual = gpu_weights[gpu_start + slot];
                weight_bit_mismatches += usize::from(actual.to_bits() != expected.to_bits());
                maximum_weight_delta = maximum_weight_delta.max((actual - expected).abs());
            }
            cpu_ids.extend_from_slice(&expected_ids);
            cpu_weights.extend_from_slice(&decision.weights);
        }
        let source = if hash_ids.is_some() {
            "hash"
        } else {
            "learned"
        };
        let cutoff_margin = if minimum_cutoff_margin.is_finite() {
            format!("{minimum_cutoff_margin:.9}")
        } else {
            "n/a".into()
        };
        eprintln!(
            "deepseek_v4 packed_route_audit chunk_start={chunk_start} n={n_tokens} layer={layer} source={source} generation={} id_mismatch_tokens={mismatch_tokens} symmetric_difference={symmetric_difference} weight_bit_mismatches={weight_bit_mismatches} max_weight_delta={maximum_weight_delta:.9} min_rank6_rank7_margin={cutoff_margin} first_id_mismatch={first_id_mismatch:?} cpu_ids_sha256={:x} gpu_ids_sha256={:x} cpu_weights_sha256={:x} gpu_weights_sha256={:x}",
            generation.get(),
            Sha256::digest(bytemuck::cast_slice(&cpu_ids)),
            Sha256::digest(bytemuck::cast_slice(&gpu_ids)),
            Sha256::digest(bytemuck::cast_slice(&cpu_weights)),
            Sha256::digest(bytemuck::cast_slice(&gpu_weights)),
        );
        Ok((cpu_ids, cpu_weights))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_router(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        input: &MetalTensor,
        ffn_norm: &MetalTensor,
        gate_inp: &MetalTensor,
        token_ids: &MetalTensor,
        hash_map: Option<&MetalTensor>,
        n_tokens: usize,
        rms_eps: f32,
    ) -> Result<PackedMoeViews, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_moe_router_batch")?;
        checked_token_count(n_tokens)?;
        validate_eps(rms_eps, "packed MoE RMSNorm epsilon")?;
        validate_f32(
            input,
            &[DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            false,
            "packed MoE input",
        )?;
        validate_f32(
            ffn_norm,
            &[DEEPSEEK_V4_HIDDEN_SIZE as u64],
            false,
            "packed MoE norm weight",
        )?;
        let normalized_input = f32_prefix(
            &self.normalized_input,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            "packed MoE normalized input",
        )?;
        let logits = f32_prefix(
            &self.logits,
            vec![MOE_EXPERT_COUNT as u64, n_tokens as u64],
            "packed MoE logits",
        )?;
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            input,
            ffn_norm,
            &normalized_input,
            n_tokens,
            DEEPSEEK_V4_HIDDEN_SIZE,
            rms_eps,
        )?;
        encode_batch_projection(
            ctx,
            enc,
            gate_inp,
            &normalized_input,
            &logits,
            DEEPSEEK_V4_HIDDEN_SIZE,
            MOE_EXPERT_COUNT,
            n_tokens,
            "packed MoE router",
        )?;
        let hash_ids = if let Some(hash_map) = hash_map {
            let hash_ids = i32_prefix(
                &self.hash_ids,
                vec![MOE_TOP_K as u64, n_tokens as u64],
                "packed hash route IDs",
            )?;
            encode_hash_gather(ctx, enc, token_ids, hash_map, &hash_ids, n_tokens)?;
            Some(hash_ids)
        } else {
            None
        };
        Ok(PackedMoeViews {
            normalized_input,
            logits,
            hash_ids,
        })
    }

    fn route(
        &self,
        views: &PackedMoeViews,
        source: PackedRouteSource<'_>,
        n_tokens: usize,
        routed_scale: f32,
    ) -> Result<Vec<ExpertBucket>, DeepSeekV4MetalError> {
        checked_token_count(n_tokens)?;
        if !routed_scale.is_finite() || routed_scale <= 0.0 {
            return invalid("packed MoE routed scale must be finite and positive");
        }
        let logits = host_read_f32(&views.logits, "packed MoE logits")?;
        let hash_ids = match source {
            PackedRouteSource::Hash => Some(host_read_i32(
                views.hash_ids.as_ref().ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "packed hash route did not gather token IDs".into(),
                    )
                })?,
                "packed hash route IDs",
            )?),
            PackedRouteSource::Learned(_) => None,
        };
        let bias = match source {
            PackedRouteSource::Hash => None,
            PackedRouteSource::Learned(bias) => {
                validate_f32(
                    bias,
                    &[MOE_EXPERT_COUNT as u64],
                    false,
                    "packed router correction bias",
                )?;
                Some(host_read_f32(bias, "packed router correction bias")?)
            }
        };
        let mut expert_ids = Vec::with_capacity(n_tokens * MOE_TOP_K);
        let mut weights = Vec::with_capacity(n_tokens * MOE_TOP_K);
        for token in 0..n_tokens {
            let start = token * MOE_EXPERT_COUNT;
            let scores = crate::deepseek_v4_oracle::sqrt_softplus_scores(
                &logits[start..start + MOE_EXPERT_COUNT],
            )
            .map_err(|error| {
                DeepSeekV4MetalError::Invalid(format!(
                    "packed router scores for token {token}: {error}"
                ))
            })?;
            let decision = if let Some(hash_ids) = hash_ids.as_ref() {
                let start = token * MOE_TOP_K;
                let selected = hash_ids[start..start + MOE_TOP_K]
                    .iter()
                    .map(|&expert| {
                        let expert = usize::try_from(expert).map_err(|_| {
                            DeepSeekV4MetalError::Invalid(format!(
                                "packed hash route contains negative ID {expert}"
                            ))
                        })?;
                        if expert >= MOE_EXPERT_COUNT {
                            return invalid(format!(
                                "packed hash route expert {expert} exceeds {MOE_EXPERT_COUNT}"
                            ));
                        }
                        Ok(expert)
                    })
                    .collect::<Result<Vec<_>, DeepSeekV4MetalError>>()?;
                crate::deepseek_v4_oracle::hash_route(&scores, &selected, routed_scale)
            } else {
                crate::deepseek_v4_oracle::learned_route(
                    &scores,
                    bias.as_ref().expect("learned route bias"),
                    MOE_TOP_K,
                    routed_scale,
                )
            }
            .map_err(|error| {
                DeepSeekV4MetalError::Invalid(format!("packed route for token {token}: {error}"))
            })?;
            expert_ids.extend(decision.expert_ids.iter().map(|&expert| expert as i32));
            weights.extend_from_slice(&decision.weights);
        }
        let expert_ids_view = i32_prefix(
            &self.expert_ids,
            vec![MOE_TOP_K as u64, n_tokens as u64],
            "packed selected expert IDs",
        )?;
        let weights_view = f32_prefix(
            &self.weights,
            vec![MOE_TOP_K as u64, n_tokens as u64],
            "packed selected expert weights",
        )?;
        host_write_i32(&expert_ids_view, &expert_ids, "packed selected expert IDs")?;
        host_write_f32(&weights_view, &weights, "packed selected expert weights")?;

        let mut by_expert = (0..MOE_EXPERT_COUNT)
            .map(|_| Vec::<(usize, usize)>::new())
            .collect::<Vec<_>>();
        for token in 0..n_tokens {
            for slot in 0..MOE_TOP_K {
                let expert = expert_ids[token * MOE_TOP_K + slot] as usize;
                by_expert[expert].push((token, token * MOE_TOP_K + slot));
            }
        }
        let mut bucket_rows = Vec::with_capacity(n_tokens * MOE_TOP_K);
        let mut bucket_slots = Vec::with_capacity(n_tokens * MOE_TOP_K);
        let mut schedule = Vec::new();
        for (expert, assignments) in by_expert.into_iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }
            let start = bucket_rows.len();
            for (token, slot) in assignments {
                bucket_rows.push(token as i32);
                bucket_slots.push(slot as i32);
            }
            schedule.push(ExpertBucket {
                expert,
                start,
                len: bucket_rows.len() - start,
            });
        }
        if bucket_rows.len() != n_tokens * MOE_TOP_K {
            return invalid("packed expert bucket schedule lost route assignments");
        }
        validate_packed_expert_schedule(
            n_tokens,
            &expert_ids,
            &bucket_rows,
            &bucket_slots,
            &schedule,
        )?;
        let bucket_rows_view = i32_prefix(
            &self.bucket_rows,
            vec![(n_tokens * MOE_TOP_K) as u64],
            "packed expert bucket rows",
        )?;
        let bucket_slots_view = i32_prefix(
            &self.bucket_slots,
            vec![(n_tokens * MOE_TOP_K) as u64],
            "packed expert bucket slots",
        )?;
        host_write_i32(&bucket_rows_view, &bucket_rows, "packed expert bucket rows")?;
        host_write_i32(
            &bucket_slots_view,
            &bucket_slots,
            "packed expert bucket slots",
        )?;
        Ok(schedule)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_experts(
        &self,
        ctx: &MetalContext,
        enc: &mut PackedPostRouteLayerEncoder<'_>,
        normalized_input: &MetalTensor,
        schedule: &[ExpertBucket],
        gpu_compacted: bool,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        shared_gate: &MetalTensor,
        shared_up: &MetalTensor,
        shared_down: &MetalTensor,
        expert_policy: PackedExpertPolicy,
        expert_clamp: f32,
        shared_clamp: f32,
        n_tokens: usize,
    ) -> Result<MetalTensor, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_moe_experts_batch")?;
        checked_token_count(n_tokens)?;
        if !expert_clamp.is_finite() || expert_clamp <= 0.0 {
            return invalid("packed expert clamp must be finite and positive");
        }
        if !shared_clamp.is_finite() || shared_clamp <= 0.0 {
            return invalid("packed shared-expert clamp must be finite and positive");
        }
        validate_expert_bank(
            gate_bank,
            DEEPSEEK_V4_HIDDEN_SIZE,
            MOE_FFN_SIZE,
            MOE_EXPERT_COUNT,
            "packed routed gate bank",
        )?;
        validate_expert_bank(
            up_bank,
            DEEPSEEK_V4_HIDDEN_SIZE,
            MOE_FFN_SIZE,
            MOE_EXPERT_COUNT,
            "packed routed up bank",
        )?;
        validate_expert_bank(
            down_bank,
            MOE_FFN_SIZE,
            DEEPSEEK_V4_HIDDEN_SIZE,
            MOE_EXPERT_COUNT,
            "packed routed down bank",
        )?;
        if gpu_compacted
            && !packed_gpu_compact_expert_layer_qualified(
                ctx,
                expert_policy,
                n_tokens,
                packed_gpu_route_iq3_enabled(),
                gate_bank.dtype,
                up_bank.dtype,
                down_bank.dtype,
            )
        {
            return invalid("packed GPU compaction reached an unqualified expert layer");
        }
        let expert_outputs = f32_prefix(
            &self.expert_outputs,
            vec![
                DEEPSEEK_V4_HIDDEN_SIZE as u64,
                MOE_TOP_K as u64,
                n_tokens as u64,
            ],
            "packed expert outputs",
        )?;
        let used_grouped_iq2 = if expert_policy.uses_iq2_target()
            && gate_bank.dtype == GgmlType::IQ2_XS
            && up_bank.dtype == GgmlType::IQ2_XS
            && down_bank.dtype == GgmlType::IQ3_XXS
            && packed_grouped_expert_kernels_supported(ctx)
        {
            let slots = i32_prefix(
                &self.bucket_slots,
                vec![(n_tokens * MOE_TOP_K) as u64],
                "packed grouped expert slots",
            )?;
            let grouped_inner = f32_prefix(
                &self.grouped_inner,
                vec![MOE_FFN_SIZE as u64, MOE_TOP_K as u64, n_tokens as u64],
                "packed grouped expert inner",
            )?;
            let grouped_plan = if gpu_compacted {
                PackedGroupedExpertPlan::from_device(
                    &self.grouped_tiles,
                    PACKED_GROUPED_EXPERT_MAX_TILES,
                )?
            } else {
                PackedGroupedExpertPlan::new(n_tokens, schedule, Some(&self.grouped_tiles))?
            };
            let used_iq2_mma16 = if expert_policy.uses_iq2_mma16(n_tokens) {
                let mma16_plan = if gpu_compacted {
                    PackedGroupedExpertPlan::from_device(
                        &self.grouped_iq2_mma16_tiles,
                        PACKED_GROUPED_IQ2_MMA16_MAX_TILES,
                    )?
                } else {
                    PackedGroupedExpertPlan::new_iq2_mma16(
                        n_tokens,
                        schedule,
                        Some(&self.grouped_iq2_mma16_tiles),
                    )?
                };
                let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed BM16 IQ2 routes")?;
                let rows = i32_prefix(
                    &self.bucket_rows,
                    vec![route_count as u64],
                    "packed BM16 IQ2 source rows",
                )?;
                let expert_output_flat = expert_outputs
                    .view_subrange(0, vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, route_count as u64]);
                let (gate, up) = packed_grouped_gate_up_views(
                    &expert_output_flat,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    route_count,
                )?;
                #[cfg(feature = "dsv4-diagnostics")]
                if enc.splits_bm16_stages() {
                    for (bank, projection) in [(gate_bank, &gate), (up_bank, &up)] {
                        encode_packed_grouped_mapped_iq2_xs_f32_mma16(
                            ctx,
                            enc,
                            bank,
                            normalized_input,
                            &rows,
                            &slots,
                            &mma16_plan,
                            projection,
                            DEEPSEEK_V4_HIDDEN_SIZE,
                            MOE_FFN_SIZE,
                            MOE_EXPERT_COUNT,
                            MOE_TOP_K,
                            n_tokens,
                            n_tokens,
                            route_count,
                        )?;
                    }
                    enc.boundary(PackedPostRouteStageKind::RoutedSwiGlu)?;
                    let projected_elements = checked_mul(
                        MOE_FFN_SIZE,
                        route_count,
                        "packed BM16 IQ2 projected elements",
                    )?;
                    encode_ds4_clamped_swiglu(
                        ctx,
                        enc,
                        &gate.view_subrange(0, vec![projected_elements as u64]),
                        &up.view_subrange(0, vec![projected_elements as u64]),
                        &grouped_inner.view_subrange(0, vec![projected_elements as u64]),
                        expert_clamp,
                    )?;
                    enc.boundary(PackedPostRouteStageKind::RoutedDown)?;
                } else {
                    encode_packed_grouped_mapped_iq2_xs_swiglu_f32_mma16(
                        ctx,
                        enc,
                        gate_bank,
                        up_bank,
                        normalized_input,
                        &rows,
                        &slots,
                        &mma16_plan,
                        &gate,
                        &up,
                        &grouped_inner,
                        DEEPSEEK_V4_HIDDEN_SIZE,
                        MOE_FFN_SIZE,
                        MOE_EXPERT_COUNT,
                        MOE_TOP_K,
                        n_tokens,
                        n_tokens,
                        route_count,
                        expert_clamp,
                    )?;
                }
                #[cfg(not(feature = "dsv4-diagnostics"))]
                encode_packed_grouped_mapped_iq2_xs_swiglu_f32_mma16(
                    ctx,
                    enc,
                    gate_bank,
                    up_bank,
                    normalized_input,
                    &rows,
                    &slots,
                    &mma16_plan,
                    &gate,
                    &up,
                    &grouped_inner,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    MOE_EXPERT_COUNT,
                    MOE_TOP_K,
                    n_tokens,
                    n_tokens,
                    route_count,
                    expert_clamp,
                )?;
                true
            } else {
                false
            };
            if !used_iq2_mma16 {
                encode_packed_grouped_swiglu_iq2_xs_f32(
                    ctx,
                    enc,
                    gate_bank,
                    up_bank,
                    normalized_input,
                    &slots,
                    &grouped_plan,
                    &grouped_inner,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    MOE_EXPERT_COUNT,
                    MOE_TOP_K,
                    n_tokens,
                    expert_clamp,
                )?;
            }
            encode_packed_grouped_down_iq3_xxs_f32(
                ctx,
                enc,
                down_bank,
                &grouped_inner,
                &slots,
                &grouped_plan,
                &expert_outputs,
                MOE_FFN_SIZE,
                DEEPSEEK_V4_HIDDEN_SIZE,
                MOE_EXPERT_COUNT,
                MOE_TOP_K,
                n_tokens,
            )?;
            #[cfg(feature = "dsv4-diagnostics")]
            {
                self.grouped_iq2_invocations.set(
                    self.grouped_iq2_invocations
                        .get()
                        .checked_add(1)
                        .ok_or_else(|| {
                            DeepSeekV4MetalError::Invalid(
                                "packed grouped IQ2 invocation count overflow".into(),
                            )
                        })?,
                );
            }
            true
        } else {
            false
        };
        let used_grouped_iq3 = if expert_policy.uses_iq3_target()
            && gate_bank.dtype == GgmlType::IQ3_XXS
            && up_bank.dtype == GgmlType::IQ3_XXS
            && down_bank.dtype == GgmlType::IQ3_XXS
            && packed_grouped_iq3_candidate_supported(ctx)
        {
            let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed grouped IQ3 routes")?;
            let rows = i32_prefix(
                &self.bucket_rows,
                vec![route_count as u64],
                "packed grouped IQ3 source rows",
            )?;
            let slots = i32_prefix(
                &self.bucket_slots,
                vec![route_count as u64],
                "packed grouped IQ3 destination slots",
            )?;
            let expert_output_flat = expert_outputs
                .view_subrange(0, vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, route_count as u64]);
            let grouped_inner = f32_prefix(
                &self.grouped_inner,
                vec![MOE_FFN_SIZE as u64, route_count as u64],
                "packed grouped IQ3 inner",
            )?;
            let grouped_plan = if gpu_compacted {
                PackedGroupedExpertPlan::from_device(
                    &self.grouped_tiles,
                    PACKED_GROUPED_EXPERT_MAX_TILES,
                )?
            } else {
                PackedGroupedExpertPlan::new(n_tokens, schedule, Some(&self.grouped_tiles))?
            };
            encode_packed_grouped_all_iq3(
                ctx,
                enc,
                gate_bank,
                up_bank,
                down_bank,
                normalized_input,
                &rows,
                &slots,
                &grouped_plan,
                &expert_output_flat,
                &grouped_inner,
                DEEPSEEK_V4_HIDDEN_SIZE,
                MOE_FFN_SIZE,
                MOE_EXPERT_COUNT,
                MOE_TOP_K,
                n_tokens,
                expert_clamp,
            )?;
            #[cfg(all(test, feature = "dsv4-diagnostics"))]
            {
                self.grouped_iq3_invocations.set(
                    self.grouped_iq3_invocations
                        .get()
                        .checked_add(1)
                        .ok_or_else(|| {
                            DeepSeekV4MetalError::Invalid(
                                "packed grouped IQ3 invocation count overflow".into(),
                            )
                        })?,
                );
            }
            true
        } else {
            false
        };
        let used_grouped_target = used_grouped_iq2 || used_grouped_iq3;
        if !used_grouped_target {
            for bucket in schedule {
                let rows = i32_slice(
                    &self.bucket_rows,
                    bucket.start,
                    bucket.len,
                    "packed expert input rows",
                )?;
                let slots = i32_slice(
                    &self.bucket_slots,
                    bucket.start,
                    bucket.len,
                    "packed expert output slots",
                )?;
                let expert_input = f32_prefix(
                    &self.expert_input,
                    vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, bucket.len as u64],
                    "packed expert input",
                )?;
                encode_get_rows_f32(
                    ctx,
                    enc,
                    normalized_input,
                    &rows,
                    &expert_input,
                    bucket.len,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                )?;
                let gate = f32_prefix(
                    &self.gate,
                    vec![MOE_FFN_SIZE as u64, bucket.len as u64],
                    "packed routed gate",
                )?;
                let up = f32_prefix(
                    &self.up,
                    vec![MOE_FFN_SIZE as u64, bucket.len as u64],
                    "packed routed up",
                )?;
                let inner = f32_prefix(
                    &self.inner,
                    vec![MOE_FFN_SIZE as u64, bucket.len as u64],
                    "packed routed inner",
                )?;
                let bucket_output = f32_prefix(
                    &self.bucket_output,
                    vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, bucket.len as u64],
                    "packed routed bucket output",
                )?;
                let gate_weight = expert_weight_view(
                    gate_bank,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    bucket.expert,
                    "packed routed gate slice",
                )?;
                let up_weight = expert_weight_view(
                    up_bank,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    bucket.expert,
                    "packed routed up slice",
                )?;
                let down_weight = expert_weight_view(
                    down_bank,
                    MOE_FFN_SIZE,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    bucket.expert,
                    "packed routed down slice",
                )?;
                encode_batch_projection(
                    ctx,
                    enc,
                    &gate_weight,
                    &expert_input,
                    &gate,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    bucket.len,
                    "packed routed gate",
                )?;
                encode_batch_projection(
                    ctx,
                    enc,
                    &up_weight,
                    &expert_input,
                    &up,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    bucket.len,
                    "packed routed up",
                )?;
                let flat_len = checked_mul(bucket.len, MOE_FFN_SIZE, "packed SwiGLU")?;
                let gate_flat = gate.view_subrange(0, vec![flat_len as u64]);
                let up_flat = up.view_subrange(0, vec![flat_len as u64]);
                let inner_flat = inner.view_subrange(0, vec![flat_len as u64]);
                encode_ds4_clamped_swiglu(
                    ctx,
                    enc,
                    &gate_flat,
                    &up_flat,
                    &inner_flat,
                    expert_clamp,
                )?;
                if down_weight.dtype == GgmlType::MXFP4 {
                    for row in 0..bucket.len {
                        let inner_row = f32_row(
                            &inner,
                            row,
                            MOE_FFN_SIZE,
                            vec![MOE_FFN_SIZE as u64],
                            "packed MXFP4 routed inner row",
                        )?;
                        let output_row = f32_row(
                            &bucket_output,
                            row,
                            DEEPSEEK_V4_HIDDEN_SIZE,
                            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64],
                            "packed MXFP4 routed output row",
                        )?;
                        encode_projection(
                            ctx,
                            enc,
                            &down_weight,
                            &inner_row,
                            &output_row,
                            MOE_FFN_SIZE,
                            DEEPSEEK_V4_HIDDEN_SIZE,
                            "packed MXFP4 routed down",
                        )?;
                    }
                } else {
                    encode_batch_projection(
                        ctx,
                        enc,
                        &down_weight,
                        &inner,
                        &bucket_output,
                        MOE_FFN_SIZE,
                        DEEPSEEK_V4_HIDDEN_SIZE,
                        bucket.len,
                        "packed routed down",
                    )?;
                }
                crate::metal::encode_scatter_rows_f32_unique(
                    ctx,
                    enc,
                    &bucket_output,
                    &slots,
                    &expert_outputs,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    bucket.len,
                )?;
            }
        }

        #[cfg(feature = "dsv4-diagnostics")]
        enc.boundary(PackedPostRouteStageKind::SharedExpert)?;

        let gate = f32_prefix(
            &self.gate,
            vec![MOE_FFN_SIZE as u64, n_tokens as u64],
            "packed shared gate",
        )?;
        let up = f32_prefix(
            &self.up,
            vec![MOE_FFN_SIZE as u64, n_tokens as u64],
            "packed shared up",
        )?;
        let inner = f32_prefix(
            &self.inner,
            vec![MOE_FFN_SIZE as u64, n_tokens as u64],
            "packed shared inner",
        )?;
        let shared_output = f32_prefix(
            &self.shared_output,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            "packed shared output",
        )?;
        encode_batch_projection(
            ctx,
            enc,
            shared_gate,
            normalized_input,
            &gate,
            DEEPSEEK_V4_HIDDEN_SIZE,
            MOE_FFN_SIZE,
            n_tokens,
            "packed shared gate",
        )?;
        encode_batch_projection(
            ctx,
            enc,
            shared_up,
            normalized_input,
            &up,
            DEEPSEEK_V4_HIDDEN_SIZE,
            MOE_FFN_SIZE,
            n_tokens,
            "packed shared up",
        )?;
        let flat_len = checked_mul(n_tokens, MOE_FFN_SIZE, "packed shared SwiGLU")?;
        encode_ds4_clamped_swiglu(
            ctx,
            enc,
            &gate.view_subrange(0, vec![flat_len as u64]),
            &up.view_subrange(0, vec![flat_len as u64]),
            &inner.view_subrange(0, vec![flat_len as u64]),
            shared_clamp,
        )?;
        encode_batch_projection(
            ctx,
            enc,
            shared_down,
            &inner,
            &shared_output,
            MOE_FFN_SIZE,
            DEEPSEEK_V4_HIDDEN_SIZE,
            n_tokens,
            "packed shared down",
        )?;

        #[cfg(feature = "dsv4-diagnostics")]
        enc.boundary(PackedPostRouteStageKind::ExpertCombine)?;
        let weights = f32_prefix(
            &self.weights,
            vec![MOE_TOP_K as u64, n_tokens as u64],
            "packed selected expert weights",
        )?;
        let routed_output = f32_prefix(
            &self.routed_output,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            "packed routed output",
        )?;
        let final_output = f32_prefix(
            &self.final_output,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            "packed MoE output",
        )?;
        crate::metal::encode_moe_weighted_sum_packed_f32(
            ctx,
            enc,
            &expert_outputs,
            &weights,
            &routed_output,
            DEEPSEEK_V4_HIDDEN_SIZE,
            MOE_TOP_K,
            n_tokens,
        )?;
        crate::metal::encode_add_f32(ctx, enc, &routed_output, &shared_output, &final_output)?;

        #[cfg(feature = "dsv4-diagnostics")]
        enc.boundary(PackedPostRouteStageKind::HyperPostAndHead)?;
        Ok(final_output)
    }
}

fn encode_hash_gather(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_ids: &MetalTensor,
    token_to_expert: &MetalTensor,
    output: &MetalTensor,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    validate_i32(token_ids, &[n_tokens as u64], false, "packed token IDs")?;
    validate_i32_bank(token_to_expert, MOE_TOP_K, "packed token-to-expert map")?;
    validate_i32(
        output,
        &[MOE_TOP_K as u64, n_tokens as u64],
        true,
        "packed hash route IDs",
    )?;
    let vocab_size = usize::try_from(token_to_expert.shape[1])
        .map_err(|_| DeepSeekV4MetalError::Invalid("hash vocabulary exceeds usize".into()))?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_tokens: u32,
        top_k: u32,
        vocab_size: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_hash_gather")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_tokens: checked_token_count(n_tokens)?,
            top_k: MOE_TOP_K as u32,
            vocab_size: u32::try_from(vocab_size)
                .map_err(|_| DeepSeekV4MetalError::Invalid("hash vocabulary exceeds u32".into()))?,
        },
    );
    enc.set_tensor(1, token_ids);
    enc.set_tensor(2, token_to_expert);
    enc.set_tensor(3, output);
    let total = checked_mul(n_tokens, MOE_TOP_K, "packed hash route IDs")?;
    enc.dispatch(
        MTLSize {
            width: total.div_ceil(64),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_copy_raw_ring_f16_bits(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    source: &MetalTensor,
    destination: &MetalTensor,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "deepseek_v4_copy_raw_ring_f16_bits")?;
    let shape = [512, DEEPSEEK_V4_LOCAL_WINDOW as u64];
    validate_f16(source, &shape, false, "packed source raw ring")?;
    validate_f16(destination, &shape, true, "packed preserved raw ring")?;
    let elements = checked_mul(512, DEEPSEEK_V4_LOCAL_WINDOW, "packed raw-ring copy")?;
    let elements = u32::try_from(elements)
        .map_err(|_| DeepSeekV4MetalError::Invalid("packed raw-ring copy exceeds u32".into()))?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_copy_u16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &Args { n: elements });
    enc.note_read(source);
    enc.set_tensor(1, source);
    enc.note_write(destination);
    enc.set_tensor(2, destination);
    enc.dispatch(
        MTLSize {
            width: (elements as usize).div_ceil(256),
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

fn encode_publish_raw_chunk_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    source: &MetalTensor,
    chunk: &MetalTensor,
    ring: &MetalTensor,
    start_position: u32,
    n_tokens: usize,
    head_dim: usize,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "deepseek_v4_publish_raw_chunk_f16")?;
    checked_token_count(n_tokens)?;
    validate_f32(
        source,
        &[head_dim as u64, n_tokens as u64],
        false,
        "packed raw chunk source",
    )?;
    validate_f16(
        chunk,
        &[head_dim as u64, n_tokens as u64],
        true,
        "packed raw chunk",
    )?;
    validate_f16(
        ring,
        &[head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        true,
        "packed raw ring",
    )?;
    start_position
        .checked_add(u32::try_from(n_tokens - 1).map_err(|_| {
            DeepSeekV4MetalError::Invalid("packed raw chunk count exceeds u32".into())
        })?)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed raw chunk position overflow".into())
        })?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_dim: u32,
        row_count: u32,
        start_position: u32,
        window: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_publish_raw_chunk_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_dim: u32::try_from(head_dim).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed raw chunk width exceeds u32".into())
            })?,
            row_count: checked_token_count(n_tokens)?,
            start_position,
            window: DEEPSEEK_V4_LOCAL_WINDOW as u32,
        },
    );
    enc.set_tensor(1, source);
    enc.set_tensor(2, chunk);
    enc.set_tensor(3, ring);
    let count = checked_mul(n_tokens, head_dim, "packed raw chunk elements")?;
    enc.dispatch(
        MTLSize {
            width: count.div_ceil(256),
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
fn encode_packed_dense_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    compressed: Option<DeepSeekV4PublishedRows<'_>>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    kind: AttentionKind,
    start_position: u32,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    checked_token_count(n_tokens)?;
    let config = deepseek_v4_session_attention_config();
    let raw_chunk = f16_prefix(
        raw_cache,
        vec![config.head_dim as u64, n_tokens as u64],
        "packed dense raw chunk",
    )?;
    let Some(query_offset) = (kind == AttentionKind::HeavilyCompressed)
        .then(|| tiled_hca_query_offset(start_position, n_tokens))
        .flatten()
    else {
        return encode_packed_cooperative_dense_sink_attention_f16(
            ctx,
            enc,
            queries,
            &raw_chunk,
            raw_cache_before_chunk,
            compressed,
            sinks,
            output,
            kind,
            start_position,
            n_tokens,
        );
    };
    let rows = compressed.ok_or_else(|| {
        DeepSeekV4MetalError::Invalid(
            "packed tiled HCA requires a published compressed history".into(),
        )
    })?;
    let query_width = config.checked()?.query_width;
    if query_offset > 0 {
        let prefix_queries = f32_prefix(
            queries,
            vec![query_width as u64, query_offset as u64],
            "packed cooperative-HCA prefix queries",
        )?;
        let prefix_output = f32_prefix(
            output,
            vec![query_width as u64, query_offset as u64],
            "packed cooperative-HCA prefix output",
        )?;
        let prefix_raw_chunk = f16_prefix(
            &raw_chunk,
            vec![config.head_dim as u64, query_offset as u64],
            "packed cooperative-HCA prefix raw chunk",
        )?;
        let prefix_end = start_position
            .checked_add(u32::try_from(query_offset).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed HCA prefix exceeds u32".into())
            })?)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("packed HCA prefix overflow".into()))?;
        encode_packed_cooperative_dense_sink_attention_f16(
            ctx,
            enc,
            &prefix_queries,
            &prefix_raw_chunk,
            raw_cache_before_chunk,
            Some(DeepSeekV4PublishedRows {
                cache: rows.cache,
                count: prefix_end as usize / 128,
                capacity_rows: rows.capacity_rows,
            }),
            sinks,
            &prefix_output,
            kind,
            start_position,
            query_offset,
        )?;
    }
    encode_tiled_dense_sink_attention_f16(
        ctx,
        enc,
        queries,
        &raw_chunk,
        raw_cache_before_chunk,
        DeepSeekV4RawCacheLayout::Chunk,
        rows,
        sinks,
        output,
        start_position,
        query_offset,
        n_tokens - query_offset,
        128,
        config,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_packed_cooperative_dense_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    compressed: Option<DeepSeekV4PublishedRows<'_>>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    kind: AttentionKind,
    start_position: u32,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let encode = if packed_grouped_dense_attention_enabled() {
        encode_grouped_online_dense_sink_attention_f16
    } else {
        encode_cooperative_dense_sink_attention_f16
    };
    encode(
        ctx,
        enc,
        queries,
        raw_cache,
        raw_cache_before_chunk,
        DeepSeekV4RawCacheLayout::Chunk,
        compressed,
        sinks,
        output,
        kind,
        start_position,
        n_tokens,
        deepseek_v4_session_attention_config(),
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_packed_selected_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    rows: DeepSeekV4CsaRows<'_>,
    sparse: PackedCsaSelectionView<'_>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    start_position: u32,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "deepseek_v4_packed_selected_attention")?;
    let config = deepseek_v4_session_attention_config();
    let dims = config.checked()?;
    checked_token_count(n_tokens)?;
    let sparse_end = sparse
        .query_offset
        .checked_add(sparse.query_count)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed sparse query range overflow".into())
        })?;
    if sparse.query_count == 0
        || sparse.query_offset >= n_tokens
        || sparse_end != n_tokens
        || rows.count <= DEEPSEEK_V4_CSA_TOP_K
        || rows.count > rows.capacity_rows
    {
        return invalid(format!(
            "packed selected attention geometry is invalid: offset={} count={} tokens={n_tokens} rows={}/{}",
            sparse.query_offset, sparse.query_count, rows.count, rows.capacity_rows
        ));
    }
    validate_f32(
        queries,
        &[dims.query_width as u64, n_tokens as u64],
        false,
        "packed selected attention queries",
    )?;
    validate_f16(
        raw_cache,
        &[config.head_dim as u64, n_tokens as u64],
        false,
        "packed selected raw chunk",
    )?;
    validate_f16(
        raw_cache_before_chunk,
        &[config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        false,
        "packed selected preserved raw cache",
    )?;
    validate_f16(
        rows.attention_cache,
        &[config.head_dim as u64, rows.capacity_rows as u64],
        false,
        "packed selected compressed cache",
    )?;
    validate_i32(
        sparse.cache_order_ids,
        &[DEEPSEEK_V4_CSA_TOP_K as u64, sparse.query_count as u64],
        false,
        "packed selected cache-order IDs",
    )?;
    for (tensor, name) in [
        (sparse.selected_counts, "packed selected row counts"),
        (sparse.visible_counts, "packed selected visible counts"),
    ] {
        validate_i32(tensor, &[sparse.query_count as u64], false, name)?;
    }
    validate_f32(
        sinks,
        &[config.head_count as u64],
        false,
        "packed selected attention sinks",
    )?;
    validate_f32(
        output,
        &[dims.query_width as u64, n_tokens as u64],
        true,
        "packed selected attention output",
    )?;

    let online = packed_selected_online_enabled();
    static POLICY_LOGGED: std::sync::Once = std::sync::Once::new();
    POLICY_LOGGED.call_once(|| {
        eprintln!(
            "deepseek_v4: packed selected attention policy={}; rollback=QWEN_DSV4_PACKED_SELECTED_ONLINE=0",
            if online { "online" } else { "legacy" },
        );
    });
    encode_cooperative_selected_sink_attention_f16(
        ctx,
        enc,
        queries,
        raw_cache,
        raw_cache_before_chunk,
        DeepSeekV4RawCacheLayout::Chunk,
        rows.attention_cache,
        rows.capacity_rows,
        sparse.cache_order_ids,
        sparse.selected_counts,
        sparse.visible_counts,
        sinks,
        output,
        start_position,
        sparse.query_offset,
        sparse.query_count,
        n_tokens,
        DEEPSEEK_V4_CSA_TOP_K,
        online,
        config,
    )
}

impl DeepSeekV4Session {
    /// Execute one layer-major chunk and expose logits for its final token.
    /// Weight projections are batched; causal cache/compressor transitions
    /// remain position-ordered and preserve any retained prefix.
    pub fn prefill_tokens(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        self.prefill_tokens_with_progress(ctx, token_ids, |_| {})
    }

    pub fn prefill_tokens_with_progress(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        mut layer_completed: impl FnMut(usize),
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        self.execute_packed_tokens_with_progress(ctx, token_ids, true, &mut layer_completed)?;
        self.logits()
    }

    /// Advance one layer-major teacher-forced chunk without computing logits.
    /// Successful advancement revokes the session's current logits and final
    /// hidden observation; host values copied out earlier remain owned copies.
    pub fn advance_tokens(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
    ) -> Result<(), DeepSeekV4MetalError> {
        self.execute_packed_tokens_with_progress(ctx, token_ids, false, &mut |_| {})
    }

    #[cfg(feature = "dsv4-diagnostics")]
    pub fn profile_packed_chunk(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        sampled: bool,
    ) -> Result<PackedChunkProfile, DeepSeekV4MetalError> {
        let grouped_mode = packed_grouped_expert_mode();
        let expert_policy = if packed_grouped_expert_scope(grouped_mode, token_ids.len())? {
            packed_grouped_expert_policy(ctx, token_ids.len())?
        } else {
            PackedExpertPolicy::Current
        };
        let mut pre_expert = PackedPrefillStageRecorder::new(ctx, sampled)?;
        let mut post_route = PackedPostRouteStageRecorder::new(ctx, sampled)?;
        self.execute_packed_tokens_with_progress_policy(
            ctx,
            token_ids,
            emit_logits,
            PackedRoutePolicy::Cpu,
            expert_policy,
            Some(&mut pre_expert),
            Some(&mut post_route),
            &mut |_| {},
        )?;
        let pre_expert = pre_expert.resolve(ctx)?;
        let mut post_route = post_route.resolve(ctx)?;
        post_route.q8_compressor_matrix_invocations =
            self.prefill.compressor.q8_matrix_invocations();
        Ok(PackedChunkProfile {
            pre_expert,
            post_route,
        })
    }

    #[cfg(feature = "dsv4-diagnostics")]
    pub fn profile_packed_post_route(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        sampled: bool,
    ) -> Result<PackedPostRouteStageProfile, DeepSeekV4MetalError> {
        let grouped_mode = packed_grouped_expert_mode();
        let expert_policy = if packed_grouped_expert_scope(grouped_mode, token_ids.len())? {
            packed_grouped_expert_policy(ctx, token_ids.len())?
        } else {
            PackedExpertPolicy::Current
        };
        let mut recorder = PackedPostRouteStageRecorder::new(ctx, sampled)?;
        self.execute_packed_tokens_with_progress_policy(
            ctx,
            token_ids,
            emit_logits,
            PackedRoutePolicy::Cpu,
            expert_policy,
            None,
            Some(&mut recorder),
            &mut |_| {},
        )?;
        let mut profile = recorder.resolve(ctx)?;
        profile.q8_compressor_matrix_invocations = self.prefill.compressor.q8_matrix_invocations();
        Ok(profile)
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn execute_packed_tokens_with_route_policy_for_test(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        gpu_route: bool,
        preserve_cpu_weights: bool,
    ) -> Result<(), DeepSeekV4MetalError> {
        let route_policy = match (gpu_route, preserve_cpu_weights) {
            (false, false) => PackedRoutePolicy::Cpu,
            (true, false) => PackedRoutePolicy::GpuExperimental,
            (true, true) => PackedRoutePolicy::GpuExperimentalCpuWeights,
            (false, true) => {
                return invalid("packed route cannot preserve CPU weights without GPU scheduling");
            }
        };
        self.execute_packed_tokens_with_progress_policy(
            ctx,
            token_ids,
            emit_logits,
            route_policy,
            PackedExpertPolicy::Current,
            None,
            None,
            &mut |_| {},
        )
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn execute_packed_tokens_with_expert_policy_for_test(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        grouped_target: bool,
        grouped_iq3_target: bool,
    ) -> Result<(), DeepSeekV4MetalError> {
        let expert_policy = match (grouped_target, grouped_iq3_target) {
            (false, false) => PackedExpertPolicy::Current,
            (true, false) => PackedExpertPolicy::GroupedIq2XsIq3Xxs,
            (true, true) => PackedExpertPolicy::GroupedIq2XsIq3XxsAndIq3Xxs,
            (false, true) => {
                return invalid("packed grouped IQ3 test policy requires the IQ2 baseline");
            }
        };
        self.execute_packed_tokens_with_progress_policy(
            ctx,
            token_ids,
            emit_logits,
            PackedRoutePolicy::Cpu,
            expert_policy,
            None,
            None,
            &mut |_| {},
        )
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn execute_packed_tokens_with_stage_profile_for_test(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        sampled: bool,
    ) -> Result<PackedPrefillStageProfile, DeepSeekV4MetalError> {
        let mut recorder = PackedPrefillStageRecorder::new(ctx, sampled)?;
        self.execute_packed_tokens_with_progress_policy(
            ctx,
            token_ids,
            emit_logits,
            PackedRoutePolicy::Cpu,
            PackedExpertPolicy::GroupedIq2XsIq3Xxs,
            Some(&mut recorder),
            None,
            &mut |_| {},
        )?;
        recorder.resolve(ctx)
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn execute_packed_tokens_with_post_route_stage_profile_for_test(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        sampled: bool,
        grouped_iq3: bool,
    ) -> Result<PackedPostRouteStageProfile, DeepSeekV4MetalError> {
        let mut recorder = PackedPostRouteStageRecorder::new(ctx, sampled)?;
        let expert_policy = if grouped_iq3 {
            PackedExpertPolicy::GroupedIq2XsIq3XxsAndIq3Xxs
        } else {
            PackedExpertPolicy::GroupedIq2XsIq3Xxs
        };
        self.execute_packed_tokens_with_progress_policy(
            ctx,
            token_ids,
            emit_logits,
            PackedRoutePolicy::Cpu,
            expert_policy,
            None,
            Some(&mut recorder),
            &mut |_| {},
        )?;
        recorder.resolve(ctx)
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn packed_route_generation_for_test(&self) -> u32 {
        self.prefill.moe.gpu_route.next_generation.get()
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn packed_grouped_iq2_invocations_for_test(&self) -> u32 {
        self.prefill.moe.grouped_iq2_invocations.get()
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn packed_grouped_iq3_invocations_for_test(&self) -> u32 {
        self.prefill.moe.grouped_iq3_invocations.get()
    }

    fn execute_packed_tokens_with_progress(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        layer_completed: &mut impl FnMut(usize),
    ) -> Result<(), DeepSeekV4MetalError> {
        let grouped_mode = packed_grouped_expert_mode();
        let expert_policy = if packed_grouped_expert_scope(grouped_mode, token_ids.len())? {
            packed_grouped_expert_policy(ctx, token_ids.len())?
        } else {
            PackedExpertPolicy::Current
        };
        self.execute_packed_tokens_with_progress_policy(
            ctx,
            token_ids,
            emit_logits,
            PackedRoutePolicy::Cpu,
            expert_policy,
            #[cfg(feature = "dsv4-diagnostics")]
            None,
            #[cfg(feature = "dsv4-diagnostics")]
            None,
            layer_completed,
        )
    }

    fn execute_packed_tokens_with_progress_policy(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        route_policy: PackedRoutePolicy,
        expert_policy: PackedExpertPolicy,
        #[cfg(feature = "dsv4-diagnostics")] stage_recorder: Option<
            &mut PackedPrefillStageRecorder,
        >,
        #[cfg(feature = "dsv4-diagnostics")] post_route_stage_recorder: Option<
            &mut PackedPostRouteStageRecorder,
        >,
        layer_completed: &mut impl FnMut(usize),
    ) -> Result<(), DeepSeekV4MetalError> {
        #[cfg(feature = "dsv4-diagnostics")]
        self.decision_diagnostics
            .ensure_no_active_capture("execute packed tokens")?;
        if ctx.device.registryID() != self.device_registry_id {
            return invalid(format!(
                "DeepSeek V4 session belongs to Metal device registry {}, got {}",
                self.device_registry_id,
                ctx.device.registryID()
            ));
        }
        let n_tokens = checked_token_count(token_ids.len())?;
        let route_policy = if route_policy == PackedRoutePolicy::Cpu
            && packed_gpu_route_compact_enabled()
            && packed_q8_compressor_matrix_scope_qualified(
                &ctx.device.name().to_string(),
                self.residency.report().tensor_count,
                self.residency.report().source_bytes,
                token_ids.len(),
            ) {
            PackedRoutePolicy::GpuCompact
        } else {
            route_policy
        };
        let expert_policy =
            if packed_grouped_iq3_enabled() && packed_grouped_iq3_candidate_supported(ctx) {
                expert_policy.with_iq3_target()
            } else {
                expert_policy
            };
        let q_b_projection =
            packed_q8_qb_projection_for_chunk(ctx, &self.residency, token_ids.len())?;
        let output_projection =
            packed_q8_output_projection_for_chunk(ctx, &self.residency, token_ids.len())?;
        let compressor_matrix =
            packed_q8_compressor_matrix_for_chunk(ctx, &self.residency, token_ids.len());
        #[cfg(feature = "dsv4-diagnostics")]
        self.prefill.compressor.reset_q8_matrix_invocations();
        #[cfg(feature = "dsv4-diagnostics")]
        validate_packed_route_policy_scope(route_policy, token_ids.len())?;
        if expert_policy.uses_iq2_target() && token_ids.len() > PACKED_GROUPED_EXPERT_MAX_TOKENS {
            return invalid(format!(
                "packed grouped expert policy exceeds its {PACKED_GROUPED_EXPERT_MAX_TOKENS}-token qualification"
            ));
        }
        let start_position = self.phase.ready_position()?;
        let end_position = start_position
            .checked_add(n_tokens)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("packed position overflow".into()))?;
        for (index, &token) in token_ids.iter().enumerate() {
            if token as usize >= DEEPSEEK_V4_VOCAB_SIZE {
                return invalid(format!(
                    "packed token {index} id {token} is outside vocabulary {DEEPSEEK_V4_VOCAB_SIZE}"
                ));
            }
            let position = start_position
                .checked_add(u32::try_from(index).map_err(|_| {
                    DeepSeekV4MetalError::Invalid("packed token index exceeds u32".into())
                })?)
                .ok_or_else(|| DeepSeekV4MetalError::Invalid("packed position overflow".into()))?;
            self.capacity.validate_position(position)?;
        }
        self.validate_committed_token_append(start_position, token_ids.len())?;
        #[cfg(feature = "dsv4-diagnostics")]
        if self.fp4_selection_mode.is_counterfactual() {
            // Reject unsupported geometry while the session is still ready;
            // token staging and causal mutation both occur below this gate.
            validate_fp4_selection_counterfactual_packed(start_position, token_ids.len())?;
        }
        let token_values = token_ids
            .iter()
            .map(|&token| token as i32)
            .collect::<Vec<_>>();
        let token_view = i32_prefix(
            &self.prefill.token_ids,
            vec![token_ids.len() as u64],
            "packed token IDs",
        )?;
        host_write_i32(&token_view, &token_values, "packed token IDs")?;
        #[cfg(feature = "dsv4-diagnostics")]
        {
            let final_position = end_position - 1;
            let sparse_query_count = sparse_csa_query_offset(start_position, token_ids.len())
                .map(|offset| token_ids.len() - offset)
                .unwrap_or(0);
            self.fp4_shadow_diagnostics
                .begin_packed(final_position, sparse_query_count)?;
        }

        let begun_position = self.phase.begin_mutation()?;
        debug_assert_eq!(begun_position, start_position);
        let result = self.prefill_tokens_inner(
            ctx,
            token_ids,
            &token_view,
            start_position,
            emit_logits,
            route_policy,
            expert_policy,
            q_b_projection,
            output_projection,
            compressor_matrix,
            #[cfg(feature = "dsv4-diagnostics")]
            stage_recorder,
            #[cfg(feature = "dsv4-diagnostics")]
            post_route_stage_recorder,
            layer_completed,
        );
        match result {
            Ok(()) => {
                self.commit_tokens(token_ids);
                self.phase
                    .complete_mutation(start_position, end_position, emit_logits)?;
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    fn prefill_tokens_inner(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        token_view: &MetalTensor,
        start_position: u32,
        emit_logits: bool,
        route_policy: PackedRoutePolicy,
        expert_policy: PackedExpertPolicy,
        q_b_projection: Q8PrecisionProjection,
        output_projection: Q8PrecisionProjection,
        compressor_matrix: bool,
        #[cfg(feature = "dsv4-diagnostics")] mut stage_recorder: Option<
            &mut PackedPrefillStageRecorder,
        >,
        #[cfg(feature = "dsv4-diagnostics")] mut post_route_stage_recorder: Option<
            &mut PackedPostRouteStageRecorder,
        >,
        layer_completed: &mut impl FnMut(usize),
    ) -> Result<(), DeepSeekV4MetalError> {
        let n_tokens = token_ids.len();
        if packed_grouped_dense_attention_enabled() {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: grouped-head online dense attention active; rollback=QWEN_DSV4_PACKED_GROUP8_DENSE=0"
                );
            }
        }
        if packed_indexer_q_matrix_enabled() {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: F32 Q8 sparse-indexer Q matrix active; rollback=QWEN_DSV4_PACKED_INDEXER_Q_MATRIX=0"
                );
            }
        }
        if compressor_matrix {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: Q8 compressor matrices active for full N={n_tokens} chunks; rollback=QWEN_DSV4_PACKED_Q8_COMPRESSOR_MATRIX=0"
                );
            }
        }
        if q_b_projection.uses_full_chunk_f32(n_tokens) {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: F32 Q8 Q-B matrix active for full N={n_tokens} chunks; rollback=QWEN_DSV4_PACKED_Q8_QB=exact"
                );
            }
        }
        if output_projection.uses_full_chunk_f32(n_tokens) {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: F32 Q8 output A/B matrix active for full N={n_tokens} chunks; rollback=QWEN_DSV4_PACKED_Q8_OUTPUT=exact"
                );
            }
        }
        if expert_policy.uses_iq2_mma16(n_tokens) {
            let mut eligible_layers = 0usize;
            for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
                let gate = self.layer_tensor(layer, "ffn_gate_exps.weight")?;
                let up = self.layer_tensor(layer, "ffn_up_exps.weight")?;
                let down = self.layer_tensor(layer, "ffn_down_exps.weight")?;
                eligible_layers += usize::from(
                    gate.dtype == GgmlType::IQ2_XS
                        && up.dtype == GgmlType::IQ2_XS
                        && down.dtype == GgmlType::IQ3_XXS,
                );
            }
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if eligible_layers > 0 && !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: BM16 IQ2 packed prefill active for full N={n_tokens} chunks; eligible_layers={eligible_layers}; rollback=QWEN_DSV4_PACKED_BM16_IQ2=0"
                );
            }
        }
        if expert_policy.uses_iq3_target() {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: grouped all-IQ3 packed experts active for full N={n_tokens} chunks; rollback=QWEN_DSV4_PACKED_GROUPED_IQ3=0"
                );
            }
        }
        if route_policy == PackedRoutePolicy::GpuCompact {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let scope = if packed_gpu_route_iq3_enabled() {
                    "IQ2/all-IQ3"
                } else {
                    "IQ2"
                };
                eprintln!(
                    "deepseek_v4: compact GPU routing active for qualified N={n_tokens} {scope} layers; unprofiled execution merges router and experts; rollback=QWEN_DSV4_PACKED_GPU_ROUTE_COMPACT=0"
                );
            }
        }
        let last_position = start_position
            .checked_add(u32::try_from(n_tokens - 1).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed token count exceeds u32".into())
            })?)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("packed position overflow".into()))?;
        let rms_eps = self.residency.config().attention_rms_epsilon;
        let hc_eps = self.residency.config().hyper_connection_epsilon;
        let attention_config = deepseek_v4_session_attention_config();
        let attention_dims = attention_config.checked()?;
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_score_plan = self
            .fp4_selection_mode
            .score_plan(self.fp4_shadow_diagnostics.is_capturing());
        #[cfg(feature = "dsv4-diagnostics")]
        let mut fp4_score_dispatch_ledger = DeepSeekV4Fp4ScoreDispatchLedger::new(
            DeepSeekV4Fp4ShadowExecution::Packed,
            last_position,
            fp4_score_plan.kind(),
            fp4_score_plan.consumed_source(),
        );
        #[cfg(feature = "dsv4-diagnostics")]
        {
            self.fp4_score_dispatch_ledger = None;
        }
        let trace_layers = std::env::var_os("QWEN_DSV4_PREFILL_TRACE").is_some();
        let batched_rope = packed_batched_rope_enabled();
        let batched_compressor = packed_batched_compressor_enabled();
        if !batched_rope {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: packed batched RoPE disabled; rollback=QWEN_DSV4_BATCHED_ROPE=0"
                );
            }
        }
        if !batched_compressor {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: packed batched compressor disabled; rollback=QWEN_DSV4_BATCHED_COMPRESSOR=0"
                );
            }
        }
        let mut layer_traces = Vec::with_capacity(if trace_layers {
            DEEPSEEK_V4_LAYER_COUNT
        } else {
            0
        });
        let embedding = f32_prefix(
            &self.prefill.embedding,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            "packed embeddings",
        )?;
        let residual_primary = f32_prefix(
            &self.prefill.residual_primary,
            vec![
                DEEPSEEK_V4_HIDDEN_SIZE as u64,
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
                n_tokens as u64,
            ],
            "packed primary residual",
        )?;
        let residual_secondary = f32_prefix(
            &self.prefill.residual_secondary,
            vec![
                DEEPSEEK_V4_HIDDEN_SIZE as u64,
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
                n_tokens as u64,
            ],
            "packed secondary residual",
        )?;

        for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
            let pre_expert_started = trace_layers.then(std::time::Instant::now);
            let routed_gate_dtype = self.layer_tensor(layer, "ffn_gate_exps.weight")?.dtype;
            let routed_up_dtype = self.layer_tensor(layer, "ffn_up_exps.weight")?.dtype;
            let routed_down_dtype = self.layer_tensor(layer, "ffn_down_exps.weight")?.dtype;
            let compact_gpu_route = route_policy == PackedRoutePolicy::GpuCompact
                && packed_gpu_compact_expert_layer_qualified(
                    ctx,
                    expert_policy,
                    n_tokens,
                    packed_gpu_route_iq3_enabled(),
                    routed_gate_dtype,
                    routed_up_dtype,
                    routed_down_dtype,
                );
            #[cfg(feature = "dsv4-diagnostics")]
            let merge_gpu_route = compact_gpu_route
                && !trace_layers
                && stage_recorder.is_none()
                && post_route_stage_recorder.is_none()
                && !self.fp4_shadow_diagnostics.is_capturing()
                && !self.fp4_selection_mode.is_counterfactual();
            #[cfg(not(feature = "dsv4-diagnostics"))]
            let merge_gpu_route = compact_gpu_route;
            let gpu_route_generation = if compact_gpu_route
                || (route_policy.uses_gpu() && route_policy != PackedRoutePolicy::GpuCompact)
            {
                Some(self.prefill.moe.take_gpu_route_generation()?)
            } else {
                None
            };
            let raw_cache = self.raw_cache_layer(layer)?;
            let rope = deepseek_v4_layer_rope(self.residency.config(), layer)?;
            let attention_kind = self.residency.config().attention_kinds[layer];
            let sparse_query_offset = (attention_kind == AttentionKind::CompressedSparse)
                .then(|| sparse_csa_query_offset(start_position, n_tokens))
                .flatten();
            let command = ctx.queue.commandBuffer().ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "failed to allocate packed layer {layer} router command buffer"
                ))
            })?;
            #[cfg(feature = "dsv4-diagnostics")]
            let mut encoder =
                PackedPrefillLayerEncoder::begin(&command, layer, stage_recorder.as_deref_mut())?;
            #[cfg(not(feature = "dsv4-diagnostics"))]
            let encoder = KernelEncoder::begin(&command);
            let router_result = (|| {
                #[cfg(feature = "dsv4-diagnostics")]
                let mut captured_sparse: Option<PackedSparseCsaViews> = None;
                #[cfg(not(feature = "dsv4-diagnostics"))]
                let captured_sparse: Option<PackedSparseCsaViews> = None;
                encode_copy_raw_ring_f16_bits(
                    ctx,
                    &encoder,
                    &raw_cache,
                    &self.prefill.attention.raw_cache_before_chunk,
                )?;
                if layer == 0 {
                    encode_get_rows_f32(
                        ctx,
                        &encoder,
                        self.residency.require_tensor("token_embd.weight")?,
                        token_view,
                        &embedding,
                        n_tokens,
                        DEEPSEEK_V4_HIDDEN_SIZE,
                    )?;
                    self.prefill.hyper.encode_initial_repeat(
                        ctx,
                        &encoder,
                        &embedding,
                        &residual_primary,
                        n_tokens,
                    )?;
                }

                let attention_input = self.prefill.hyper.encode_pre(
                    ctx,
                    &encoder,
                    &residual_primary,
                    self.layer_tensor(layer, "hc_attn_fn.weight")?,
                    self.layer_tensor(layer, "hc_attn_scale.weight")?,
                    self.layer_tensor(layer, "hc_attn_base.weight")?,
                    n_tokens,
                    rms_eps,
                    hc_eps,
                )?;

                let attention = self.prefill.attention.encode_prepare(
                    ctx,
                    &encoder,
                    &attention_input,
                    self.layer_tensor(layer, "attn_norm.weight")?,
                    self.layer_tensor(layer, "attn_q_a.weight")?,
                    self.layer_tensor(layer, "attn_q_a_norm.weight")?,
                    self.layer_tensor(layer, "attn_q_b.weight")?,
                    self.layer_tensor(layer, "attn_kv.weight")?,
                    self.layer_tensor(layer, "attn_kv_a_norm.weight")?,
                    n_tokens,
                    rms_eps,
                    q_b_projection,
                )?;

                let compressor = self.prefill.compressor.encode_layer_projections(
                    ctx,
                    &encoder,
                    &self.residency,
                    layer,
                    &attention.normalized_input,
                    n_tokens,
                    compressor_matrix,
                )?;

                let query_heads = attention.queries.view_subrange(
                    0,
                    vec![
                        attention_config.head_dim as u64,
                        attention_config.head_count as u64,
                        n_tokens as u64,
                    ],
                );
                if batched_rope {
                    encode_ds4_rope_tail_adjacent_batch_in_place(
                        ctx,
                        &encoder,
                        &query_heads,
                        start_position,
                        n_tokens,
                        1,
                        rope,
                        false,
                    )?;
                    encode_ds4_rope_tail_adjacent_batch_in_place(
                        ctx,
                        &encoder,
                        &attention.kv,
                        start_position,
                        n_tokens,
                        1,
                        rope,
                        false,
                    )?;
                } else {
                    for row in 0..n_tokens {
                        let position = start_position
                            .checked_add(u32::try_from(row).map_err(|_| {
                                DeepSeekV4MetalError::Invalid("packed RoPE row exceeds u32".into())
                            })?)
                            .ok_or_else(|| {
                                DeepSeekV4MetalError::Invalid(
                                    "packed RoPE position overflow".into(),
                                )
                            })?;
                        let queries = f32_row(
                            &attention.queries,
                            row,
                            attention_dims.query_width,
                            vec![
                                attention_config.head_dim as u64,
                                attention_config.head_count as u64,
                            ],
                            "packed query row",
                        )?;
                        let kv = f32_row(
                            &attention.kv,
                            row,
                            attention_config.head_dim,
                            vec![attention_config.head_dim as u64],
                            "packed KV row",
                        )?;
                        encode_ds4_rope_tail_adjacent_in_place(
                            ctx, &encoder, &queries, position, rope, false,
                        )?;
                        encode_ds4_rope_tail_adjacent_in_place(
                            ctx, &encoder, &kv, position, rope, false,
                        )?;
                    }
                }
                let raw_chunk = f16_prefix(
                    &self.prefill.attention.raw_chunk,
                    vec![attention_config.head_dim as u64, n_tokens as u64],
                    "packed raw chunk",
                )?;
                encode_publish_raw_chunk_f16(
                    ctx,
                    &encoder,
                    &attention.kv,
                    &raw_chunk,
                    &raw_cache,
                    start_position,
                    n_tokens,
                    attention_config.head_dim,
                )?;

                if batched_compressor {
                    self.compressor_frontiers.encode_layer_projected_chunk(
                        ctx,
                        &encoder,
                        &self.residency,
                        layer,
                        start_position,
                        n_tokens,
                        &compressor,
                        &self.prefill.compressor,
                        rope,
                        rms_eps,
                    )?;
                } else {
                    for row in 0..n_tokens {
                        let position = start_position
                            .checked_add(u32::try_from(row).map_err(|_| {
                                DeepSeekV4MetalError::Invalid("packed row exceeds u32".into())
                            })?)
                            .ok_or_else(|| {
                                DeepSeekV4MetalError::Invalid("packed position overflow".into())
                            })?;
                        self.compressor_frontiers.encode_layer_projected_row(
                            ctx,
                            &encoder,
                            &self.residency,
                            layer,
                            row,
                            position,
                            &compressor,
                            rope,
                            rms_eps,
                        )?;
                    }
                }

                #[cfg(feature = "dsv4-diagnostics")]
                if sparse_query_offset.is_some() {
                    encoder.boundary(PackedPrefillStageKind::SparseIndexerPrepare)?;
                } else {
                    encoder.skip_stages(
                        &[
                            PackedPrefillStageKind::SparseIndexerPrepare,
                            PackedPrefillStageKind::SparseIndexerScore,
                            PackedPrefillStageKind::SparseSelection,
                        ],
                        PackedPrefillStageKind::AttentionCore,
                    )?;
                }

                let compressed = self
                    .compressor_frontiers
                    .attention_rows(layer, last_position)?;
                if let Some(query_offset) = sparse_query_offset {
                    let rows = self
                        .compressor_frontiers
                        .csa_rows(layer, last_position)?
                        .ok_or_else(|| {
                            DeepSeekV4MetalError::Invalid(format!(
                                "CSA layer {layer} has no rows at sparse position {last_position}"
                            ))
                        })?;
                    #[cfg(not(feature = "dsv4-diagnostics"))]
                    let sparse = self.prefill.attention.sparse_csa.encode(
                        ctx,
                        &encoder,
                        &attention.q_lora,
                        &attention.normalized_input,
                        self.layer_tensor(layer, "indexer.attn_q_b.weight")?,
                        self.layer_tensor(layer, "indexer.proj.weight")?,
                        rows,
                        start_position,
                        query_offset,
                        n_tokens,
                        rope,
                    )?;
                    #[cfg(feature = "dsv4-diagnostics")]
                    let sparse = self.prefill.attention.sparse_csa.encode_prepare(
                        ctx,
                        &encoder,
                        &attention.q_lora,
                        &attention.normalized_input,
                        self.layer_tensor(layer, "indexer.attn_q_b.weight")?,
                        self.layer_tensor(layer, "indexer.proj.weight")?,
                        rows,
                        start_position,
                        query_offset,
                        n_tokens,
                        rope,
                    )?;
                    #[cfg(feature = "dsv4-diagnostics")]
                    fp4_score_dispatch_ledger.record_common_prepare()?;
                    #[cfg(feature = "dsv4-diagnostics")]
                    encoder.boundary(PackedPrefillStageKind::SparseIndexerScore)?;
                    #[cfg(feature = "dsv4-diagnostics")]
                    if fp4_score_plan.runs_f16() {
                        self.prefill
                            .attention
                            .sparse_csa
                            .encode_f16_scores(ctx, &encoder, rows, &sparse)?;
                    }
                    #[cfg(feature = "dsv4-diagnostics")]
                    if fp4_score_plan.runs_fp4() {
                        if sparse.query_count != 1 {
                            return invalid(format!(
                                "FP4 packed shadow expected one sparse query, got {}",
                                sparse.query_count
                            ));
                        }
                        self.fp4_shadow.encode(
                            ctx,
                            &encoder,
                            &sparse.index_queries,
                            &sparse.head_weights,
                            rows,
                            &sparse.visible_counts,
                        )?;
                        fp4_score_dispatch_ledger.record_fp4_pipeline()?;
                    }
                    #[cfg(feature = "dsv4-diagnostics")]
                    if self.fp4_shadow_diagnostics.is_capturing() {
                        captured_sparse = Some(sparse.clone());
                    }
                    #[cfg(feature = "dsv4-diagnostics")]
                    encoder.boundary(PackedPrefillStageKind::SparseSelection)?;
                    #[cfg(feature = "dsv4-diagnostics")]
                    if fp4_score_plan.runs_f16() {
                        self.prefill
                            .attention
                            .sparse_csa
                            .encode_f16_selection(ctx, &encoder, rows, &sparse)?;
                        fp4_score_dispatch_ledger.record_f16_score_and_selector()?;
                    }
                    #[cfg(feature = "dsv4-diagnostics")]
                    let selected = if fp4_score_plan.consumes_fp4() {
                        PackedCsaSelectionView {
                            query_offset: sparse.query_offset,
                            query_count: sparse.query_count,
                            cache_order_ids: &self.fp4_shadow.cache_order_ids,
                            selected_counts: &self.fp4_shadow.selected_count,
                            visible_counts: &self.fp4_shadow.eligible_visible,
                        }
                    } else {
                        sparse.selection_view()
                    };
                    #[cfg(not(feature = "dsv4-diagnostics"))]
                    let selected = sparse.selection_view();
                    #[cfg(feature = "dsv4-diagnostics")]
                    encoder.boundary(PackedPrefillStageKind::AttentionCore)?;
                    if query_offset > 0 {
                        let dense_queries = f32_prefix(
                            &attention.queries,
                            vec![attention_dims.query_width as u64, query_offset as u64],
                            "packed dense-prefix queries",
                        )?;
                        let dense_output = f32_prefix(
                            &attention.attention,
                            vec![attention_dims.query_width as u64, query_offset as u64],
                            "packed dense-prefix attention",
                        )?;
                        let dense_last = start_position
                            .checked_add(u32::try_from(query_offset - 1).map_err(|_| {
                                DeepSeekV4MetalError::Invalid(
                                    "packed dense-prefix offset exceeds u32".into(),
                                )
                            })?)
                            .ok_or_else(|| {
                                DeepSeekV4MetalError::Invalid(
                                    "packed dense-prefix position overflow".into(),
                                )
                            })?;
                        let dense_count = csa_visible_rows(dense_last);
                        encode_packed_dense_sink_attention_f16(
                            ctx,
                            &encoder,
                            &dense_queries,
                            &raw_chunk,
                            &self.prefill.attention.raw_cache_before_chunk,
                            Some(DeepSeekV4PublishedRows {
                                cache: rows.attention_cache,
                                count: dense_count,
                                capacity_rows: rows.capacity_rows,
                            }),
                            self.layer_tensor(layer, "attn_sinks.weight")?,
                            &dense_output,
                            attention_kind,
                            start_position,
                            query_offset,
                        )?;
                    }
                    encode_packed_selected_sink_attention_f16(
                        ctx,
                        &encoder,
                        &attention.queries,
                        &raw_chunk,
                        &self.prefill.attention.raw_cache_before_chunk,
                        rows,
                        selected,
                        self.layer_tensor(layer, "attn_sinks.weight")?,
                        &attention.attention,
                        start_position,
                        n_tokens,
                    )?;
                } else {
                    encode_packed_dense_sink_attention_f16(
                        ctx,
                        &encoder,
                        &attention.queries,
                        &raw_chunk,
                        &self.prefill.attention.raw_cache_before_chunk,
                        compressed,
                        self.layer_tensor(layer, "attn_sinks.weight")?,
                        &attention.attention,
                        attention_kind,
                        start_position,
                        n_tokens,
                    )?;
                }
                #[cfg(feature = "dsv4-diagnostics")]
                encoder.boundary(PackedPrefillStageKind::InverseRope)?;
                let attention_heads = attention.attention.view_subrange(
                    0,
                    vec![
                        attention_config.head_dim as u64,
                        attention_config.head_count as u64,
                        n_tokens as u64,
                    ],
                );
                if batched_rope {
                    encode_ds4_rope_tail_adjacent_batch_in_place(
                        ctx,
                        &encoder,
                        &attention_heads,
                        start_position,
                        n_tokens,
                        1,
                        rope,
                        true,
                    )?;
                } else {
                    for row in 0..n_tokens {
                        let position = start_position
                            .checked_add(u32::try_from(row).map_err(|_| {
                                DeepSeekV4MetalError::Invalid(
                                    "packed inverse-RoPE row exceeds u32".into(),
                                )
                            })?)
                            .ok_or_else(|| {
                                DeepSeekV4MetalError::Invalid(
                                    "packed inverse-RoPE position overflow".into(),
                                )
                            })?;
                        let attention_row = f32_row(
                            &attention.attention,
                            row,
                            attention_dims.query_width,
                            vec![
                                attention_config.head_dim as u64,
                                attention_config.head_count as u64,
                            ],
                            "packed attention row",
                        )?;
                        encode_ds4_rope_tail_adjacent_in_place(
                            ctx,
                            &encoder,
                            &attention_row,
                            position,
                            rope,
                            true,
                        )?;
                    }
                }

                #[cfg(feature = "dsv4-diagnostics")]
                encoder.boundary(PackedPrefillStageKind::AttentionOutputProjections)?;

                let output_a = self.layer_tensor(layer, "attn_output_a.weight")?;
                let output_b = self.layer_tensor(layer, "attn_output_b.weight")?;
                let attention_output = if output_projection.uses_full_chunk_f32(n_tokens) {
                    self.prefill.attention.encode_output_q8_precision(
                        ctx,
                        &encoder,
                        &attention.attention,
                        output_a,
                        output_b,
                        n_tokens,
                        Q8PrecisionProjection::F32Matrix,
                        Q8PrecisionProjection::F32Matrix,
                    )?
                } else {
                    self.prefill.attention.encode_output(
                        ctx,
                        &encoder,
                        &attention.attention,
                        output_a,
                        output_b,
                        n_tokens,
                    )?
                };

                #[cfg(feature = "dsv4-diagnostics")]
                encoder.boundary(PackedPrefillStageKind::AfterAttentionOutput)?;

                self.prefill.hyper.encode_post(
                    ctx,
                    &encoder,
                    &attention_output,
                    &residual_primary,
                    &residual_secondary,
                    n_tokens,
                )?;
                let ffn_input = self.prefill.hyper.encode_pre(
                    ctx,
                    &encoder,
                    &residual_secondary,
                    self.layer_tensor(layer, "hc_ffn_fn.weight")?,
                    self.layer_tensor(layer, "hc_ffn_scale.weight")?,
                    self.layer_tensor(layer, "hc_ffn_base.weight")?,
                    n_tokens,
                    rms_eps,
                    hc_eps,
                )?;

                let hash_map = if layer < self.residency.config().hash_layer_count as usize {
                    Some(self.layer_tensor(layer, "ffn_gate_tid2eid.weight")?)
                } else {
                    None
                };
                let moe_views = self.prefill.moe.encode_router(
                    ctx,
                    &encoder,
                    &ffn_input,
                    self.layer_tensor(layer, "ffn_norm.weight")?,
                    self.layer_tensor(layer, "ffn_gate_inp.weight")?,
                    token_view,
                    hash_map,
                    n_tokens,
                    rms_eps,
                )?;
                if let Some(generation) = gpu_route_generation {
                    let source = if hash_map.is_some() {
                        PackedRouteSource::Hash
                    } else {
                        PackedRouteSource::Learned(self.layer_tensor(layer, "exp_probs_b.bias")?)
                    };
                    if compact_gpu_route {
                        self.prefill.moe.encode_gpu_route_compact(
                            ctx,
                            &encoder,
                            &moe_views,
                            source,
                            token_view,
                            hash_map,
                            n_tokens,
                            self.residency.config().expert_weights_scale,
                            generation,
                        )?;
                    } else {
                        #[cfg(feature = "dsv4-diagnostics")]
                        self.prefill.moe.encode_gpu_route_schedule(
                            ctx,
                            &encoder,
                            &moe_views,
                            source,
                            token_view,
                            hash_map,
                            n_tokens,
                            self.residency.config().expert_weights_scale,
                            generation,
                        )?;
                        #[cfg(not(feature = "dsv4-diagnostics"))]
                        return invalid("packed diagnostic GPU route is unavailable");
                    }
                }
                Ok::<_, DeepSeekV4MetalError>((moe_views, captured_sparse))
            })();
            encoder.end();
            let (moe_views, captured_sparse) = router_result?;
            #[cfg(not(feature = "dsv4-diagnostics"))]
            let _ = captured_sparse;
            let pre_expert_encode_seconds = pre_expert_started
                .as_ref()
                .map_or(0.0, |started| started.elapsed().as_secs_f64());
            #[cfg(feature = "dsv4-diagnostics")]
            let stage_profile_active = stage_recorder.is_some();
            #[cfg(not(feature = "dsv4-diagnostics"))]
            let stage_profile_active = false;
            let (
                pre_expert_wait_seconds,
                pre_expert_command_seconds,
                pre_expert_gpu_seconds,
                pre_expert_wait_residual_seconds,
            ) = if merge_gpu_route {
                (0.0, 0.0, 0.0, 0.0)
            } else {
                let wait_started = trace_layers.then(std::time::Instant::now);
                command.commit();
                command.waitUntilCompleted();
                let wait_seconds = wait_started
                    .as_ref()
                    .map_or(0.0, |started| started.elapsed().as_secs_f64());
                let command_seconds = pre_expert_started
                    .as_ref()
                    .map_or(0.0, |started| started.elapsed().as_secs_f64());
                let gpu_seconds = if trace_layers || stage_profile_active {
                    command.GPUEndTime() - command.GPUStartTime()
                } else {
                    0.0
                };
                if let Some(error) = command.error() {
                    return invalid(format!(
                        "packed layer {layer} router command failed: {error:?}"
                    ));
                }
                #[cfg(feature = "dsv4-diagnostics")]
                if let Some(recorder) = stage_recorder.as_deref_mut() {
                    recorder.record_command_gpu_seconds(layer, gpu_seconds)?;
                }
                (
                    wait_seconds,
                    command_seconds,
                    gpu_seconds,
                    wait_seconds - gpu_seconds,
                )
            };
            let pre_expert_seconds = if merge_gpu_route {
                0.0
            } else {
                pre_expert_started
                    .as_ref()
                    .map_or(0.0, |started| started.elapsed().as_secs_f64())
            };
            let pre_expert_post_seconds = pre_expert_seconds - pre_expert_command_seconds;

            let route_started = trace_layers.then(std::time::Instant::now);
            let mut schedule = if merge_gpu_route {
                Vec::new()
            } else if compact_gpu_route {
                self.prefill.moe.capture_gpu_compact_schedule(
                    n_tokens,
                    gpu_route_generation.expect("compact route generation"),
                )?
            } else {
                #[cfg(feature = "dsv4-diagnostics")]
                if let Some(generation) = gpu_route_generation {
                    #[cfg(test)]
                    let cpu_route = {
                        let source = if layer < self.residency.config().hash_layer_count as usize {
                            PackedRouteSource::Hash
                        } else {
                            PackedRouteSource::Learned(
                                self.layer_tensor(layer, "exp_probs_b.bias")?,
                            )
                        };
                        self.prefill.moe.audit_gpu_route_against_cpu(
                            &moe_views,
                            source,
                            n_tokens,
                            layer,
                            start_position,
                            generation,
                        )?
                    };
                    let schedule = self
                        .prefill
                        .moe
                        .capture_gpu_route_schedule(n_tokens, generation)?;
                    #[cfg(test)]
                    if route_policy == PackedRoutePolicy::GpuExperimentalCpuWeights {
                        let (cpu_ids, cpu_weights) = cpu_route;
                        let route_count = n_tokens * MOE_TOP_K;
                        let mut gpu_ids = host_read_i32(
                            &self.prefill.moe.expert_ids,
                            "packed GPU hybrid route IDs",
                        )?;
                        gpu_ids.truncate(route_count);
                        if gpu_ids != cpu_ids {
                            return invalid(format!(
                                "packed GPU hybrid layer {layer} route IDs differ from Rust"
                            ));
                        }
                        let weights = f32_prefix(
                            &self.prefill.moe.weights,
                            vec![route_count as u64],
                            "packed GPU hybrid route weights",
                        )?;
                        host_write_f32(&weights, &cpu_weights, "packed GPU hybrid route weights")?;
                    }
                    schedule
                } else {
                    let source = if layer < self.residency.config().hash_layer_count as usize {
                        PackedRouteSource::Hash
                    } else {
                        PackedRouteSource::Learned(self.layer_tensor(layer, "exp_probs_b.bias")?)
                    };
                    self.prefill.moe.route(
                        &moe_views,
                        source,
                        n_tokens,
                        self.residency.config().expert_weights_scale,
                    )?
                }
                #[cfg(not(feature = "dsv4-diagnostics"))]
                {
                    let source = if layer < self.residency.config().hash_layer_count as usize {
                        PackedRouteSource::Hash
                    } else {
                        PackedRouteSource::Learned(self.layer_tensor(layer, "exp_probs_b.bias")?)
                    };
                    self.prefill.moe.route(
                        &moe_views,
                        source,
                        n_tokens,
                        self.residency.config().expert_weights_scale,
                    )?
                }
            };
            let route_seconds = route_started
                .as_ref()
                .map_or(0.0, |started| started.elapsed().as_secs_f64());

            let post_route_started = trace_layers.then(std::time::Instant::now);
            let routed_gate = self.layer_tensor(layer, "ffn_gate_exps.weight")?;
            let routed_up = self.layer_tensor(layer, "ffn_up_exps.weight")?;
            let routed_down = self.layer_tensor(layer, "ffn_down_exps.weight")?;
            let shared_gate = self.layer_tensor(layer, "ffn_gate_shexp.weight")?;
            let shared_up = self.layer_tensor(layer, "ffn_up_shexp.weight")?;
            let shared_down = self.layer_tensor(layer, "ffn_down_shexp.weight")?;
            #[cfg(feature = "dsv4-diagnostics")]
            if let Some(recorder) = post_route_stage_recorder.as_deref_mut() {
                let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed metadata routes")?;
                let expert_ids = host_read_i32(
                    &i32_prefix(
                        &self.prefill.moe.expert_ids,
                        vec![route_count as u64],
                        "packed metadata expert IDs",
                    )?,
                    "packed metadata expert IDs",
                )?;
                let bucket_rows = host_read_i32(
                    &i32_prefix(
                        &self.prefill.moe.bucket_rows,
                        vec![route_count as u64],
                        "packed metadata bucket rows",
                    )?,
                    "packed metadata bucket rows",
                )?;
                let bucket_slots = host_read_i32(
                    &i32_prefix(
                        &self.prefill.moe.bucket_slots,
                        vec![route_count as u64],
                        "packed metadata bucket slots",
                    )?,
                    "packed metadata bucket slots",
                )?;
                let grouped_iq2 = expert_policy.uses_iq2_target()
                    && routed_gate.dtype == GgmlType::IQ2_XS
                    && routed_up.dtype == GgmlType::IQ2_XS
                    && routed_down.dtype == GgmlType::IQ3_XXS
                    && packed_grouped_expert_kernels_supported(ctx);
                let grouped_iq3 = expert_policy.uses_iq3_target()
                    && routed_gate.dtype == GgmlType::IQ3_XXS
                    && routed_up.dtype == GgmlType::IQ3_XXS
                    && routed_down.dtype == GgmlType::IQ3_XXS
                    && packed_grouped_iq3_candidate_supported(ctx);
                recorder.record_layer(PackedPostRouteLayerMetadata {
                    layer,
                    gate_dtype: routed_gate.dtype,
                    up_dtype: routed_up.dtype,
                    down_dtype: routed_down.dtype,
                    bucket_count: schedule.len(),
                    expert_counts: packed_post_route_expert_counts(n_tokens, &schedule)?,
                    route_expert_ids: packed_post_route_expert_ids(
                        n_tokens,
                        &expert_ids,
                        &bucket_rows,
                        &bucket_slots,
                        &schedule,
                    )?,
                    grouped_iq2,
                    grouped_iq3,
                    bm16: grouped_iq2 && expert_policy.uses_iq2_mma16(n_tokens),
                })?;
            }

            let separate_expert_command = if merge_gpu_route {
                None
            } else {
                Some(ctx.queue.commandBuffer().ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "failed to allocate packed layer {layer} expert command buffer"
                    ))
                })?)
            };
            let expert_command = separate_expert_command.as_ref().unwrap_or(&command);
            #[cfg(feature = "dsv4-diagnostics")]
            let mut encoder = PackedPostRouteLayerEncoder::begin(
                expert_command,
                layer,
                expert_policy.uses_iq2_mma16(n_tokens)
                    && routed_gate.dtype == GgmlType::IQ2_XS
                    && routed_up.dtype == GgmlType::IQ2_XS
                    && routed_down.dtype == GgmlType::IQ3_XXS
                    && packed_grouped_expert_kernels_supported(ctx),
                post_route_stage_recorder.as_deref_mut(),
            )?;
            #[cfg(not(feature = "dsv4-diagnostics"))]
            let mut encoder = PackedPostRouteLayerEncoder::begin(expert_command)?;
            let expert_result = (|| {
                let moe_output = self.prefill.moe.encode_experts(
                    ctx,
                    &mut encoder,
                    &moe_views.normalized_input,
                    &schedule,
                    compact_gpu_route,
                    routed_gate,
                    routed_up,
                    routed_down,
                    shared_gate,
                    shared_up,
                    shared_down,
                    expert_policy,
                    self.residency.config().swiglu_clamp_experts[layer],
                    self.residency.config().swiglu_clamp_shared[layer],
                    n_tokens,
                )?;
                self.prefill.hyper.encode_post(
                    ctx,
                    &encoder,
                    &moe_output,
                    &residual_secondary,
                    &residual_primary,
                    n_tokens,
                )?;
                if layer + 1 == DEEPSEEK_V4_LAYER_COUNT && emit_logits {
                    let final_residual = f32_row(
                        &residual_primary,
                        n_tokens - 1,
                        residual_len(DEEPSEEK_V4_HIDDEN_SIZE)?,
                        vec![
                            DEEPSEEK_V4_HIDDEN_SIZE as u64,
                            DEEPSEEK_V4_CONNECTION_COUNT as u64,
                        ],
                        "packed final residual",
                    )?;
                    self.hyper_connection.encode_head(
                        ctx,
                        &encoder,
                        &final_residual,
                        self.residency.require_tensor("output_hc_fn.weight")?,
                        self.residency.require_tensor("output_hc_scale.weight")?,
                        self.residency.require_tensor("output_hc_base.weight")?,
                        &self.final_hidden,
                        rms_eps,
                        hc_eps,
                    )?;
                    encode_rms_norm_mul_f32(
                        ctx,
                        &encoder,
                        &self.final_hidden,
                        self.residency.require_tensor("output_norm.weight")?,
                        &self.final_normalized_hidden,
                        rms_eps,
                    )?;
                    encode_projection(
                        ctx,
                        &encoder,
                        self.residency.require_tensor("output.weight")?,
                        &self.final_normalized_hidden,
                        &self.logits,
                        DEEPSEEK_V4_HIDDEN_SIZE,
                        DEEPSEEK_V4_VOCAB_SIZE,
                        "packed output logits",
                    )?;
                }
                Ok::<(), DeepSeekV4MetalError>(())
            })();
            encoder.end();
            expert_result?;
            let post_route_encode_seconds = post_route_started
                .as_ref()
                .map_or(0.0, |started| started.elapsed().as_secs_f64());
            let post_route_wait_started = trace_layers.then(std::time::Instant::now);
            expert_command.commit();
            expert_command.waitUntilCompleted();
            let post_route_wait_seconds = post_route_wait_started
                .as_ref()
                .map_or(0.0, |started| started.elapsed().as_secs_f64());
            if let Some(error) = expert_command.error() {
                return invalid(format!("packed layer {layer} command failed: {error:?}"));
            }
            if merge_gpu_route {
                schedule = self.prefill.moe.capture_gpu_compact_schedule(
                    n_tokens,
                    gpu_route_generation.expect("compact route generation"),
                )?;
            }
            if let Some(query_offset) = sparse_query_offset
                && {
                    #[cfg(feature = "dsv4-diagnostics")]
                    {
                        fp4_score_plan.runs_f16()
                    }
                    #[cfg(not(feature = "dsv4-diagnostics"))]
                    {
                        true
                    }
                }
            {
                self.prefill
                    .attention
                    .sparse_csa
                    .validate_completed(n_tokens - query_offset)?;
            }
            #[cfg(feature = "dsv4-diagnostics")]
            if self.fp4_shadow_diagnostics.is_capturing()
                && attention_kind == AttentionKind::CompressedSparse
            {
                let sparse = captured_sparse.as_ref().ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "FP4 packed shadow layer {layer} did not retain sparse views"
                    ))
                })?;
                let rows = self
                    .compressor_frontiers
                    .csa_rows(layer, last_position)?
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(format!(
                            "FP4 packed shadow layer {layer} has no published rows"
                        ))
                    })?;
                let report = self.fp4_shadow.capture_layer(
                    layer,
                    last_position,
                    rows,
                    &sparse.scores,
                    &sparse.selected_mask,
                    &sparse.cache_order_ids,
                    &sparse.selected_counts,
                    &sparse.status,
                )?;
                self.fp4_shadow_diagnostics.capture_layer(report)?;
            }
            #[cfg(feature = "dsv4-diagnostics")]
            if fp4_score_plan.consumes_fp4() && sparse_query_offset.is_some() {
                self.fp4_shadow.validate_completed()?;
                self.fp4_shadow.record_counterfactual_selection(
                    &mut self.fp4_counterfactual_trace,
                    DeepSeekV4Fp4ShadowExecution::Packed,
                    DeepSeekV4Fp4SelectionSource::Fp4,
                    last_position,
                    layer,
                )?;
            }
            let measure_post_route_gpu = trace_layers;
            #[cfg(feature = "dsv4-diagnostics")]
            let measure_post_route_gpu =
                measure_post_route_gpu || post_route_stage_recorder.is_some();
            let measured_post_route_gpu_seconds = measure_post_route_gpu
                .then(|| expert_command.GPUEndTime() - expert_command.GPUStartTime());
            let post_route_gpu_seconds = if trace_layers {
                measured_post_route_gpu_seconds.ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "packed post-route trace omitted command GPU duration".into(),
                    )
                })?
            } else {
                0.0
            };
            #[cfg(feature = "dsv4-diagnostics")]
            if let Some(recorder) = post_route_stage_recorder.as_deref_mut() {
                recorder.record_command_gpu_seconds(
                    layer,
                    measured_post_route_gpu_seconds.ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(
                            "packed post-route profile omitted command GPU duration".into(),
                        )
                    })?,
                )?;
            }
            let post_route_wait_residual_seconds = post_route_wait_seconds - post_route_gpu_seconds;
            let post_route_seconds = post_route_started
                .as_ref()
                .map_or(0.0, |started| started.elapsed().as_secs_f64());
            if trace_layers {
                layer_traces.push(PackedLayerTrace {
                    layer,
                    pre_expert_seconds,
                    pre_expert_gpu_seconds,
                    pre_expert_encode_seconds,
                    pre_expert_wait_seconds,
                    pre_expert_wait_residual_seconds,
                    pre_expert_post_seconds,
                    route_seconds,
                    post_route_seconds,
                    post_route_gpu_seconds,
                    post_route_encode_seconds,
                    post_route_wait_seconds,
                    post_route_wait_residual_seconds,
                    bucket_count: schedule.len(),
                });
            }
            layer_completed(layer);
        }
        if trace_layers {
            for trace in &layer_traces {
                eprintln!(
                    "deepseek_v4 packed layer={} pre_expert={:.4}s pre_expert_gpu={:.4}s pre_expert_encode={:.4}s pre_expert_wait={:.4}s pre_expert_wait_residual={:.4}s pre_expert_post={:.4}s route={:.4}s post_route={:.4}s post_route_gpu={:.4}s post_route_encode={:.4}s post_route_wait={:.4}s post_route_wait_residual={:.4}s buckets={}",
                    trace.layer,
                    trace.pre_expert_seconds,
                    trace.pre_expert_gpu_seconds,
                    trace.pre_expert_encode_seconds,
                    trace.pre_expert_wait_seconds,
                    trace.pre_expert_wait_residual_seconds,
                    trace.pre_expert_post_seconds,
                    trace.route_seconds,
                    trace.post_route_seconds,
                    trace.post_route_gpu_seconds,
                    trace.post_route_encode_seconds,
                    trace.post_route_wait_seconds,
                    trace.post_route_wait_residual_seconds,
                    trace.bucket_count,
                );
            }
            let pre_expert_total = layer_traces
                .iter()
                .map(|trace| trace.pre_expert_seconds)
                .sum::<f64>();
            let pre_expert_gpu_total = layer_traces
                .iter()
                .map(|trace| trace.pre_expert_gpu_seconds)
                .sum::<f64>();
            let pre_expert_encode_total = layer_traces
                .iter()
                .map(|trace| trace.pre_expert_encode_seconds)
                .sum::<f64>();
            let pre_expert_wait_total = layer_traces
                .iter()
                .map(|trace| trace.pre_expert_wait_seconds)
                .sum::<f64>();
            let pre_expert_wait_residual_total = layer_traces
                .iter()
                .map(|trace| trace.pre_expert_wait_residual_seconds)
                .sum::<f64>();
            let pre_expert_post_total = layer_traces
                .iter()
                .map(|trace| trace.pre_expert_post_seconds)
                .sum::<f64>();
            let route_total = layer_traces
                .iter()
                .map(|trace| trace.route_seconds)
                .sum::<f64>();
            let post_route_total = layer_traces
                .iter()
                .map(|trace| trace.post_route_seconds)
                .sum::<f64>();
            let post_route_gpu_total = layer_traces
                .iter()
                .map(|trace| trace.post_route_gpu_seconds)
                .sum::<f64>();
            let post_route_encode_total = layer_traces
                .iter()
                .map(|trace| trace.post_route_encode_seconds)
                .sum::<f64>();
            let post_route_wait_total = layer_traces
                .iter()
                .map(|trace| trace.post_route_wait_seconds)
                .sum::<f64>();
            let post_route_wait_residual_total = layer_traces
                .iter()
                .map(|trace| trace.post_route_wait_residual_seconds)
                .sum::<f64>();
            let pre_expert_negative_residuals = layer_traces
                .iter()
                .filter(|trace| trace.pre_expert_wait_residual_seconds < 0.0)
                .count();
            let post_route_negative_residuals = layer_traces
                .iter()
                .filter(|trace| trace.post_route_wait_residual_seconds < 0.0)
                .count();
            eprintln!(
                "deepseek_v4 packed totals route_policy={route_policy:?} expert_policy={expert_policy:?} pre_expert={pre_expert_total:.3}s pre_expert_gpu={pre_expert_gpu_total:.3}s pre_expert_encode={pre_expert_encode_total:.3}s pre_expert_wait={pre_expert_wait_total:.3}s pre_expert_wait_residual={pre_expert_wait_residual_total:.3}s pre_expert_post={pre_expert_post_total:.3}s pre_expert_negative_residuals={pre_expert_negative_residuals} route={route_total:.3}s post_route={post_route_total:.3}s post_route_gpu={post_route_gpu_total:.3}s post_route_encode={post_route_encode_total:.3}s post_route_wait={post_route_wait_total:.3}s post_route_wait_residual={post_route_wait_residual_total:.3}s post_route_negative_residuals={post_route_negative_residuals}"
            );
        }
        #[cfg(feature = "dsv4-diagnostics")]
        self.fp4_shadow_diagnostics.finish()?;
        #[cfg(feature = "dsv4-diagnostics")]
        {
            fp4_score_dispatch_ledger.validate_completed()?;
            self.fp4_score_dispatch_ledger = Some(fp4_score_dispatch_ledger);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "dsv4-diagnostics")]
    use sha2::{Digest, Sha256};

    const PACKED_ROUTE_AGGREGATE_WIDTH: usize = 4;
    const PACKED_ROUTE_STALE_ROUTE: i32 = -101;
    const PACKED_ROUTE_FAILED_ROUTE: i32 = -102;
    const PACKED_ROUTE_INVALID_ID: i32 = -103;
    const PACKED_ROUTE_DUPLICATE_ID: i32 = -104;
    const PACKED_ROUTE_INVALID_WEIGHT: i32 = -105;
    const PACKED_ROUTE_STALE_SCHEDULE: i32 = -106;
    const PACKED_ROUTE_INVALID_COUNT: i32 = -107;
    const PACKED_ROUTE_INVALID_SCHEDULE: i32 = -108;
    const PACKED_ROUTE_INVALID_PADDING: i32 = -109;
    const PACKED_ROUTE_INVALID_AGGREGATE: i32 = -200;
    const PACKED_ROUTE_SLOT_GUARD_BYTES: usize = 64;
    const PACKED_ROUTE_SLOT_PREFIX: u8 = 0xa5;
    const PACKED_ROUTE_SLOT_SUFFIX: u8 = 0x5a;
    #[cfg(feature = "dsv4-diagnostics")]
    const PACKED_GROUPED_IQ2_COUNT_CENSUS_FIXTURE: &str =
        include_str!("../../tests/fixtures/deepseek_v4_packed_grouped_iq2_count_census_v1.json");
    #[cfg(feature = "dsv4-diagnostics")]
    const PACKED_GROUPED_IQ2_ROUTE_CENSUS_FIXTURE: &str =
        include_str!("../../tests/fixtures/deepseek_v4_packed_grouped_iq2_route_census_v1.json");
    #[cfg(feature = "dsv4-diagnostics")]
    const PACKED_GROUPED_IQ2_REPRESENTATIVE_ROUTE_CENSUS_FIXTURE: &str = include_str!(
        "../../tests/fixtures/deepseek_v4_packed_grouped_iq2_route_census_representative_v1.json"
    );
    #[cfg(feature = "dsv4-diagnostics")]
    const PACKED_ALL_IQ3_ROUTE_CENSUS_FIXTURE: &str =
        include_str!("../../tests/fixtures/deepseek_v4_packed_all_iq3_route_census_v1.json");

    #[cfg(feature = "dsv4-diagnostics")]
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PackedGroupedIq2CountCensusFixture {
        schema_version: usize,
        count_payload_domain_hex: String,
        count_payload_sha256: String,
        model_content_id: String,
        prompt_token_ids_sha256: String,
        n_tokens: usize,
        top_k: usize,
        expert_count: usize,
        grouped_layer_ids: Vec<usize>,
        total_active_experts: usize,
        inactive_experts: usize,
        total_t32: usize,
        total_padding: usize,
        active_count_histogram_1_to_128: Vec<usize>,
        tile_histogram_1_to_4: Vec<usize>,
        layers: Vec<PackedGroupedIq2CountLayerFixture>,
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PackedGroupedIq2CountLayerFixture {
        layer: usize,
        active_experts: usize,
        t32: usize,
        padding: usize,
        counts: Vec<u16>,
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PackedGroupedIq2RouteCensusFixture {
        schema_version: usize,
        route_payload_domain_hex: String,
        route_payload_sha256: String,
        model_content_id: String,
        prompt_token_ids_sha256: String,
        prompt_token_ids: Option<Vec<i32>>,
        count_payload_sha256: String,
        n_tokens: usize,
        top_k: usize,
        expert_count: usize,
        layer_count: usize,
        route_count: usize,
        grouped_layer_ids: Vec<usize>,
        layers: Vec<PackedGroupedIq2RouteLayerFixture>,
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PackedGroupedIq2RouteLayerFixture {
        layer: usize,
        route_expert_ids: Vec<u16>,
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PackedAllIq3RouteCensusFixture {
        schema_version: usize,
        count_payload_domain_hex: String,
        count_payload_sha256: String,
        route_payload_domain_hex: String,
        route_payload_sha256: String,
        model_content_id: String,
        prompt_token_ids_sha256: String,
        n_tokens: usize,
        top_k: usize,
        expert_count: usize,
        layer_count: usize,
        route_count: usize,
        total_active_experts: usize,
        total_t32: usize,
        total_padding: usize,
        all_iq3_layer_ids: Vec<usize>,
        layers: Vec<PackedAllIq3RouteLayerFixture>,
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PackedAllIq3RouteLayerFixture {
        layer: usize,
        active_experts: usize,
        t32: usize,
        padding: usize,
        expert_counts: Vec<u16>,
        route_expert_ids: Vec<u16>,
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn packed_grouped_iq2_count_census_fixture() -> PackedGroupedIq2CountCensusFixture {
        serde_json::from_str(PACKED_GROUPED_IQ2_COUNT_CENSUS_FIXTURE)
            .expect("valid grouped-IQ2 count census fixture")
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn packed_grouped_iq2_route_census_fixture() -> PackedGroupedIq2RouteCensusFixture {
        serde_json::from_str(PACKED_GROUPED_IQ2_ROUTE_CENSUS_FIXTURE)
            .expect("valid grouped-IQ2 route census fixture")
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn packed_grouped_iq2_representative_route_census_fixture() -> PackedGroupedIq2RouteCensusFixture
    {
        serde_json::from_str(PACKED_GROUPED_IQ2_REPRESENTATIVE_ROUTE_CENSUS_FIXTURE)
            .expect("valid representative grouped-IQ2 route census fixture")
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn packed_all_iq3_route_census_fixture() -> PackedAllIq3RouteCensusFixture {
        serde_json::from_str(PACKED_ALL_IQ3_ROUTE_CENSUS_FIXTURE)
            .expect("valid packed all-IQ3 route census fixture")
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn packed_grouped_schedule_from_route_ids(
        n_tokens: usize,
        route_ids: &[u16],
    ) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<ExpertBucket>) {
        assert_eq!(route_ids.len(), n_tokens * MOE_TOP_K);
        let expert_ids = route_ids
            .iter()
            .map(|&expert| {
                assert!(usize::from(expert) < MOE_EXPERT_COUNT);
                i32::from(expert)
            })
            .collect::<Vec<_>>();
        for token_ids in route_ids.chunks_exact(MOE_TOP_K) {
            let mut seen = [false; MOE_EXPERT_COUNT];
            for &expert in token_ids {
                assert!(!std::mem::replace(&mut seen[usize::from(expert)], true));
            }
        }
        let mut rows = Vec::with_capacity(route_ids.len());
        let mut slots = Vec::with_capacity(route_ids.len());
        let mut schedule = Vec::new();
        for expert in 0..MOE_EXPERT_COUNT {
            let start = slots.len();
            for (slot, &routed_expert) in route_ids.iter().enumerate() {
                if usize::from(routed_expert) == expert {
                    rows.push((slot / MOE_TOP_K) as i32);
                    slots.push(slot as i32);
                }
            }
            if slots.len() > start {
                schedule.push(ExpertBucket {
                    expert,
                    start,
                    len: slots.len() - start,
                });
            }
        }
        validate_packed_expert_schedule(n_tokens, &expert_ids, &rows, &slots, &schedule).unwrap();
        (expert_ids, rows, slots, schedule)
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[test]
    fn packed_prefill_stage_resolver_closes_signed_overlaps() {
        let mut records = Vec::new();
        let mut timestamps = Vec::new();
        let mut cursor = 100u64;
        for (index, kind) in PACKED_PREFILL_STAGE_KINDS.into_iter().enumerate() {
            let start_sample = timestamps.len();
            let start_timestamp = if index == 1 { cursor - 5 } else { cursor };
            timestamps.push(start_timestamp);
            cursor = start_timestamp + 10 + index as u64;
            let end_sample = timestamps.len();
            timestamps.push(cursor);
            records.push(PackedPrefillPendingStageSample {
                layer: 0,
                kind,
                samples: Some((start_sample, end_sample)),
            });
            cursor += 3;
        }
        let profile =
            resolve_packed_prefill_layer_stage_samples(0, &records, &timestamps, 2.0, &[]).unwrap();
        assert_eq!(profile.layer, 0);
        assert_eq!(profile.command_gpu_ms, 2.0);
        assert_eq!(profile.stages.len(), PACKED_PREFILL_STAGE_KINDS.len());
        assert!(profile.sampled_span_ticks > 0);
        assert!(profile.raw_span_ms_assuming_ns > 0.0);
        assert!(profile.raw_coverage_assuming_ns > 0.0);
        assert!(profile.encoder_gap_ms_scaled > 0.0);
        assert!(profile.encoder_overlap_ms_scaled > 0.0);
        let stage_ms = profile
            .stages
            .iter()
            .zip(PACKED_PREFILL_STAGE_KINDS)
            .map(|(stage, expected)| {
                assert_eq!(stage.kind, expected);
                if let (Some(start), Some(end)) = (stage.start_timestamp, stage.end_timestamp) {
                    assert_eq!(stage.duration_ticks, end - start);
                } else {
                    panic!("physical test stage {expected:?} has no samples");
                }
                stage.duration_ms_scaled
            })
            .sum::<f64>();
        assert!(
            (stage_ms + profile.encoder_gap_ms_scaled - profile.encoder_overlap_ms_scaled - 2.0)
                .abs()
                < 1e-12
        );
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[test]
    fn packed_post_route_stage_resolver_closes_signed_overlaps() {
        let mut records = Vec::new();
        let mut timestamps = Vec::new();
        let mut cursor = 100u64;
        for (index, kind) in PACKED_POST_ROUTE_STAGE_KINDS.into_iter().enumerate() {
            let start_sample = timestamps.len();
            let start_timestamp = if index == 2 { cursor - 4 } else { cursor };
            timestamps.push(start_timestamp);
            cursor = start_timestamp + 20 + index as u64;
            let end_sample = timestamps.len();
            timestamps.push(cursor);
            records.push(PackedPostRoutePendingStageSample {
                layer: 0,
                kind,
                start_sample,
                end_sample,
            });
            cursor += 3;
        }
        let profile = resolve_packed_post_route_layer_stage_samples(
            0,
            &records,
            &PACKED_POST_ROUTE_STAGE_KINDS,
            &timestamps,
            2.0,
        )
        .unwrap();
        assert_eq!(profile.layer, 0);
        assert_eq!(profile.command_gpu_ms, 2.0);
        assert_eq!(profile.stages.len(), PACKED_POST_ROUTE_STAGE_KINDS.len());
        assert!(profile.sampled_span_ticks > 0);
        assert!(profile.raw_span_ms_assuming_ns > 0.0);
        assert!(profile.raw_coverage_assuming_ns > 0.0);
        assert!(profile.encoder_gap_ms_scaled > 0.0);
        assert!(profile.encoder_overlap_ms_scaled > 0.0);
        let stage_ms = profile
            .stages
            .iter()
            .zip(PACKED_POST_ROUTE_STAGE_KINDS)
            .map(|(stage, expected)| {
                assert_eq!(stage.kind, expected);
                assert_eq!(
                    stage.duration_ticks,
                    stage.end_timestamp - stage.start_timestamp
                );
                stage.duration_ms_scaled
            })
            .sum::<f64>();
        assert!(
            (stage_ms + profile.encoder_gap_ms_scaled
                - profile.encoder_overlap_ms_scaled
                - profile.command_gpu_ms)
                .abs()
                < 1e-12
        );
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[test]
    fn packed_post_route_stage_resolver_accepts_bm16_split() {
        let mut records = Vec::new();
        let mut timestamps = Vec::new();
        let mut cursor = 500u64;
        for &kind in &PACKED_BM16_POST_ROUTE_STAGE_KINDS {
            let start_sample = timestamps.len();
            timestamps.push(cursor);
            cursor += 17;
            let end_sample = timestamps.len();
            timestamps.push(cursor);
            records.push(PackedPostRoutePendingStageSample {
                layer: 3,
                kind,
                start_sample,
                end_sample,
            });
            cursor += 2;
        }
        let profile = resolve_packed_post_route_layer_stage_samples(
            3,
            &records,
            &PACKED_BM16_POST_ROUTE_STAGE_KINDS,
            &timestamps,
            4.0,
        )
        .unwrap();
        assert_eq!(profile.layer, 3);
        assert_eq!(profile.stages.len(), 6);
        assert_eq!(
            profile.stages[0].kind,
            PackedPostRouteStageKind::RoutedGateUp
        );
        assert_eq!(
            profile.stages[1].kind,
            PackedPostRouteStageKind::RoutedSwiGlu
        );
        assert_eq!(profile.stages[2].kind, PackedPostRouteStageKind::RoutedDown);
        let accounted = profile
            .stages
            .iter()
            .map(|stage| stage.duration_ms_scaled)
            .sum::<f64>()
            + profile.encoder_gap_ms_scaled
            - profile.encoder_overlap_ms_scaled;
        assert!((accounted - profile.command_gpu_ms).abs() < 1e-12);
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[test]
    fn packed_prefill_stage_resolver_represents_empty_stage_explicitly() {
        let records = [
            PackedPrefillPendingStageSample {
                layer: 0,
                kind: PackedPrefillStageKind::BeforeAttentionBody,
                samples: Some((0, 1)),
            },
            PackedPrefillPendingStageSample {
                layer: 0,
                kind: PackedPrefillStageKind::SparseIndexerPrepare,
                samples: None,
            },
            PackedPrefillPendingStageSample {
                layer: 0,
                kind: PackedPrefillStageKind::SparseIndexerScore,
                samples: None,
            },
            PackedPrefillPendingStageSample {
                layer: 0,
                kind: PackedPrefillStageKind::SparseSelection,
                samples: None,
            },
            PackedPrefillPendingStageSample {
                layer: 0,
                kind: PackedPrefillStageKind::AttentionCore,
                samples: Some((2, 3)),
            },
            PackedPrefillPendingStageSample {
                layer: 0,
                kind: PackedPrefillStageKind::InverseRope,
                samples: Some((4, 5)),
            },
            PackedPrefillPendingStageSample {
                layer: 0,
                kind: PackedPrefillStageKind::AttentionOutputProjections,
                samples: Some((6, 7)),
            },
            PackedPrefillPendingStageSample {
                layer: 0,
                kind: PackedPrefillStageKind::AfterAttentionOutput,
                samples: Some((8, 9)),
            },
        ];
        let profile = resolve_packed_prefill_layer_stage_samples(
            0,
            &records,
            &[100, 110, 113, 120, 123, 130, 133, 140, 143, 150],
            1.0,
            &[
                PackedPrefillStageKind::SparseIndexerPrepare,
                PackedPrefillStageKind::SparseIndexerScore,
                PackedPrefillStageKind::SparseSelection,
            ],
        )
        .unwrap();
        for stage in &profile.stages[1..=3] {
            assert_eq!(stage.start_timestamp, None);
            assert_eq!(stage.end_timestamp, None);
            assert_eq!(stage.duration_ticks, 0);
            assert_eq!(stage.duration_ms_scaled, 0.0);
        }
        assert_eq!(profile.transitions.len(), 4);
        assert_eq!(
            profile.transitions[0].from,
            PackedPrefillStageKind::BeforeAttentionBody
        );
        assert_eq!(
            profile.transitions[0].to,
            PackedPrefillStageKind::AttentionCore
        );
    }

    fn grouped_test_bank(
        ctx: &MetalContext,
        dtype: GgmlType,
        n_in: usize,
        n_out: usize,
        expert_count: usize,
        seed: usize,
    ) -> MetalTensor {
        let (block_elements, block_bytes) = ggml_type_layout(dtype).unwrap();
        let block_elements = block_elements as usize;
        let block_bytes = block_bytes as usize;
        let blocks = n_in * n_out * expert_count / block_elements;
        let mut payload = vec![0u8; blocks * block_bytes];
        for block in 0..blocks {
            let start = block * block_bytes;
            payload[start..start + 2].copy_from_slice(
                &half::f16::from_f32(0.00390625 * (1 + (block + seed) % 7) as f32)
                    .to_bits()
                    .to_le_bytes(),
            );
            for byte in 2..block_bytes {
                payload[start + byte] = (block * 31 + byte * 13 + seed * 19 + 5) as u8;
            }
        }
        MetalTensor {
            buffer: ctx.buffer_from(&payload).unwrap(),
            offset: 0,
            shape: vec![n_in as u64, n_out as u64, expert_count as u64],
            dtype,
            provenance: MetalTensorProvenance::OwnedWritable,
        }
    }

    fn grouped_guarded_f32(ctx: &MetalContext, shape: Vec<u64>, poison: f32) -> MetalTensor {
        const GUARD: usize = 64;
        let elements = shape.iter().product::<u64>() as usize;
        let mut bytes = vec![0xa5u8; GUARD];
        let poison_values = vec![poison; elements];
        bytes.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&poison_values));
        bytes.extend_from_slice(&[0x5au8; GUARD]);
        MetalTensor {
            buffer: ctx.buffer_from(&bytes).unwrap(),
            offset: GUARD as u64,
            shape,
            dtype: GgmlType::F32,
            provenance: MetalTensorProvenance::OwnedWritable,
        }
    }

    fn assert_grouped_guards(label: &str, tensor: &MetalTensor) {
        const GUARD: usize = 64;
        let base = tensor.buffer.contents().as_ptr().cast::<u8>();
        let prefix =
            unsafe { std::slice::from_raw_parts(base.add(tensor.offset as usize - GUARD), GUARD) };
        let suffix = unsafe {
            std::slice::from_raw_parts(
                base.add(tensor.offset as usize + tensor.n_bytes() as usize),
                GUARD,
            )
        };
        assert!(
            prefix.iter().all(|&byte| byte == 0xa5),
            "{label} prefix guard changed: {prefix:?}"
        );
        assert!(
            suffix.iter().all(|&byte| byte == 0x5a),
            "{label} suffix guard changed: {suffix:?}"
        );
    }

    fn q8_precision_test_weight(ctx: &MetalContext, n_in: usize, n_out: usize) -> MetalTensor {
        const SCALE_BITS: [u16; 6] = [0x0001, 0x03ff, 0x0400, 0x1a24, 0x2e66, 0x3800];
        const QUANTS: [i8; 12] = [-128, -127, -63, -1, 0, 1, 17, 63, 126, 127, -31, 47];
        let blocks = n_in * n_out / 32;
        let mut payload = vec![0u8; blocks * 34];
        for block in 0..blocks {
            let start = block * 34;
            payload[start..start + 2]
                .copy_from_slice(&SCALE_BITS[block % SCALE_BITS.len()].to_le_bytes());
            for index in 0..32 {
                payload[start + 2 + index] = QUANTS[(block * 7 + index * 5) % QUANTS.len()] as u8;
            }
        }
        MetalTensor::from_bytes(
            ctx,
            &payload,
            vec![n_in as u64, n_out as u64],
            GgmlType::Q8_0,
        )
        .unwrap()
    }

    fn q8_precision_test_input(elements: usize) -> Vec<f32> {
        const SPECIAL: [f32; 12] = [
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            0.000_061_005_354,
            -0.000_061_005_354,
            0.333_251_95,
            -0.333_251_95,
            1.000_488_3,
            -1.000_488_3,
            0.125_030_52,
            -0.125_030_52,
        ];
        (0..elements)
            .map(|index| {
                if index % 5 == 0 {
                    SPECIAL[(index / 5) % SPECIAL.len()]
                } else {
                    ((index * 37 + index / 11 + 3) % 509) as f32 * 0.001 - 0.254
                }
            })
            .collect()
    }

    fn q8_output_test_weight(
        ctx: &MetalContext,
        n_in: usize,
        n_out: usize,
        seed: usize,
    ) -> MetalTensor {
        let blocks = n_in * n_out / 32;
        let mut payload = vec![0u8; blocks * 34];
        for block in 0..blocks {
            let start = block * 34;
            let scale = half::f16::from_f32(0.003 + ((block + seed) % 19) as f32 * 0.0002);
            payload[start..start + 2].copy_from_slice(&scale.to_bits().to_le_bytes());
            for index in 0..32 {
                payload[start + 2 + index] =
                    (((block * 13 + index * 17 + seed * 7) % 127) as i8 - 63) as u8;
            }
        }
        MetalTensor::from_bytes(
            ctx,
            &payload,
            vec![n_in as u64, n_out as u64],
            GgmlType::Q8_0,
        )
        .unwrap()
    }

    fn q8_differential(actual: &[f32], expected: &[f32]) -> (f64, f64, f32) {
        let mut dot = 0.0f64;
        let mut actual_norm = 0.0f64;
        let mut expected_norm = 0.0f64;
        let mut error = 0.0f64;
        let mut max_abs = 0.0f32;
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(actual.is_finite() && expected.is_finite());
            let actual_f64 = actual as f64;
            let expected_f64 = expected as f64;
            let delta = actual - expected;
            dot += actual_f64 * expected_f64;
            actual_norm += actual_f64 * actual_f64;
            expected_norm += expected_f64 * expected_f64;
            error += (delta as f64) * (delta as f64);
            max_abs = max_abs.max(delta.abs());
        }
        (
            dot / (actual_norm * expected_norm).sqrt(),
            (error / expected_norm).sqrt(),
            max_abs,
        )
    }

    fn grouped_test_schedule(n_tokens: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<ExpertBucket>) {
        let mut expert_ids = Vec::with_capacity(n_tokens * MOE_TOP_K);
        for token in 0..n_tokens {
            expert_ids.push((MOE_EXPERT_COUNT - 1) as i32);
            for slot in 1..MOE_TOP_K {
                expert_ids
                    .push(((token * (MOE_TOP_K - 1) + slot - 1) % (MOE_EXPERT_COUNT - 1)) as i32);
            }
        }
        let mut assignments = (0..MOE_EXPERT_COUNT)
            .map(|_| Vec::<(usize, usize)>::new())
            .collect::<Vec<_>>();
        for token in 0..n_tokens {
            for slot in 0..MOE_TOP_K {
                let route_slot = token * MOE_TOP_K + slot;
                assignments[expert_ids[route_slot] as usize].push((token, route_slot));
            }
        }
        let mut rows = Vec::with_capacity(n_tokens * MOE_TOP_K);
        let mut slots = Vec::with_capacity(n_tokens * MOE_TOP_K);
        let mut schedule = Vec::new();
        for (expert, assignments) in assignments.into_iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }
            let start = rows.len();
            for (row, slot) in assignments {
                rows.push(row as i32);
                slots.push(slot as i32);
            }
            schedule.push(ExpertBucket {
                expert,
                start,
                len: rows.len() - start,
            });
        }
        validate_packed_expert_schedule(n_tokens, &expert_ids, &rows, &slots, &schedule).unwrap();
        (expert_ids, rows, slots, schedule)
    }

    #[test]
    fn packed_grouped_expert_mode_has_an_isolated_fail_closed_rollback() {
        assert_eq!(
            DEEPSEEK_V4_PREFILL_MAX_TOKENS,
            PACKED_GROUPED_EXPERT_MAX_TOKENS
        );
        let mma16 = PackedExpertPolicy::GroupedIq2XsIq3XxsMma16QualifiedChunk;
        assert!(!mma16.uses_iq2_mma16(127));
        assert!(mma16.uses_iq2_mma16(PACKED_GROUPED_IQ2_MMA16_NARROW_TOKENS));
        assert!(!mma16.uses_iq2_mma16(129));
        assert!(!mma16.uses_iq2_mma16(PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS - 1));
        assert!(mma16.uses_iq2_mma16(PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS));
        assert!(!mma16.uses_iq2_mma16(PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS + 1));
        assert!(!mma16.uses_iq3_target());
        let gpu_compact = mma16.with_iq3_target();
        assert_eq!(
            gpu_compact,
            PackedExpertPolicy::GroupedIq2XsIq3XxsMma16AndIq3XxsQualifiedChunk
        );
        assert!(gpu_compact.uses_iq2_mma16(PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS));
        assert!(gpu_compact.uses_iq3_target());
        assert!(
            packed_grouped_expert_scope(
                PackedGroupedExpertMode::Auto,
                PACKED_GROUPED_EXPERT_MAX_TOKENS,
            )
            .unwrap()
        );
        assert!(
            packed_grouped_expert_scope(
                PackedGroupedExpertMode::Auto,
                DEEPSEEK_V4_PREFILL_MAX_TOKENS,
            )
            .unwrap()
        );
        assert!(
            packed_grouped_expert_scope(
                PackedGroupedExpertMode::ForceOn,
                DEEPSEEK_V4_PREFILL_MAX_TOKENS + 1,
            )
            .unwrap_err()
            .to_string()
            .contains("qualified through 2048 tokens")
        );
        assert_eq!(
            parse_packed_grouped_expert_mode(None),
            PackedGroupedExpertMode::Auto
        );
        for enabled in ["1", "true", "TRUE", "yes", "YES"] {
            assert_eq!(
                parse_packed_grouped_expert_mode(Some(enabled)),
                PackedGroupedExpertMode::ForceOn
            );
        }
        for disabled in ["0", "false", "FALSE", "no", "NO", "invalid"] {
            assert_eq!(
                parse_packed_grouped_expert_mode(Some(disabled)),
                PackedGroupedExpertMode::ForceOff
            );
        }
        for automatic in ["auto", "AUTO"] {
            assert_eq!(
                parse_packed_grouped_expert_mode(Some(automatic)),
                PackedGroupedExpertMode::Auto
            );
        }
    }

    #[test]
    fn packed_q8_qb_matrix_policy_and_scope_are_explicit() {
        assert_eq!(
            parse_packed_q8_qb_policy(None).unwrap(),
            PackedQ8MatrixPolicy::Auto
        );
        assert_eq!(
            parse_packed_q8_qb_policy(Some("auto")).unwrap(),
            PackedQ8MatrixPolicy::Auto
        );
        assert_eq!(
            parse_packed_q8_qb_policy(Some("exact")).unwrap(),
            PackedQ8MatrixPolicy::Exact
        );
        assert_eq!(
            parse_packed_q8_qb_policy(Some("f32_matrix")).unwrap(),
            PackedQ8MatrixPolicy::F32Matrix
        );
        assert_eq!(
            resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::Auto, true),
            Q8PrecisionProjection::F32Matrix
        );
        assert_eq!(
            resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::Auto, false),
            Q8PrecisionProjection::Exact
        );
        assert_eq!(
            resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::Exact, true),
            Q8PrecisionProjection::Exact
        );
        assert_eq!(
            resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::F32Matrix, false),
            Q8PrecisionProjection::F32Matrix
        );
        let matrix = Q8PrecisionProjection::F32Matrix;
        assert!(!matrix.uses_full_chunk_f32(512));
        assert!(matrix.uses_full_chunk_f32(DEEPSEEK_V4_PREFILL_MAX_TOKENS));
        assert!(parse_packed_q8_qb_policy(Some("half_matrix")).is_err());
    }

    #[test]
    fn packed_q8_output_matrix_policy_is_explicit_and_full_chunk_only() {
        assert_eq!(
            parse_packed_q8_output_policy(None).unwrap(),
            PackedQ8MatrixPolicy::Auto
        );
        assert_eq!(
            parse_packed_q8_output_policy(Some("auto")).unwrap(),
            PackedQ8MatrixPolicy::Auto
        );
        assert_eq!(
            parse_packed_q8_output_policy(Some("exact")).unwrap(),
            PackedQ8MatrixPolicy::Exact
        );
        assert_eq!(
            parse_packed_q8_output_policy(Some("f32_matrix")).unwrap(),
            PackedQ8MatrixPolicy::F32Matrix
        );
        assert!(parse_packed_q8_output_policy(Some("half_matrix")).is_err());
        assert_eq!(
            resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::Auto, true),
            Q8PrecisionProjection::F32Matrix
        );
        assert_eq!(
            resolve_packed_q8_matrix_policy(PackedQ8MatrixPolicy::Auto, false),
            Q8PrecisionProjection::Exact
        );
        let matrix = Q8PrecisionProjection::F32Matrix;
        assert!(!matrix.uses_full_chunk_f32(512));
        assert!(matrix.uses_full_chunk_f32(DEEPSEEK_V4_PREFILL_MAX_TOKENS));
    }

    #[test]
    fn packed_q8_compressor_matrix_scope_is_exact() {
        let qualified = |device, tensors, bytes, tokens| {
            packed_q8_compressor_matrix_scope_qualified(device, tensors, bytes, tokens)
        };
        assert!(qualified(
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
            DEEPSEEK_V4_PREFILL_MAX_TOKENS,
        ));
        assert!(!qualified(
            "Apple M3 Max",
            1_328,
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
            DEEPSEEK_V4_PREFILL_MAX_TOKENS,
        ));
        assert!(!qualified(
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_327,
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
            DEEPSEEK_V4_PREFILL_MAX_TOKENS,
        ));
        assert!(!qualified(
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES - 1,
            DEEPSEEK_V4_PREFILL_MAX_TOKENS,
        ));
        assert!(!qualified(
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE,
            1_328,
            PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
            DEEPSEEK_V4_PREFILL_MAX_TOKENS - 1,
        ));
    }

    #[test]
    fn packed_grouped_schedule_preflight_and_tile_bound_are_exact() {
        let (expert_ids, rows, slots, schedule) = grouped_test_schedule(128);
        let tiles = packed_grouped_expert_tiles(128, &schedule).unwrap();
        assert_eq!(tiles.len(), 259);
        assert_eq!(
            tiles.iter().map(|tile| tile.count as usize).sum::<usize>(),
            768
        );
        assert!(tiles.iter().all(|tile| (1..=32).contains(&tile.count)));
        #[cfg(feature = "dsv4-diagnostics")]
        assert_eq!(
            packed_post_route_expert_ids(128, &expert_ids, &rows, &slots, &schedule).unwrap(),
            expert_ids
                .iter()
                .map(|&expert| expert as u16)
                .collect::<Vec<_>>()
        );

        let mut bad_rows = rows.clone();
        bad_rows[0] ^= 1;
        assert!(
            validate_packed_expert_schedule(128, &expert_ids, &bad_rows, &slots, &schedule)
                .is_err()
        );
        let mut bad_slots = slots.clone();
        bad_slots[1] = bad_slots[0];
        assert!(
            validate_packed_expert_schedule(128, &expert_ids, &rows, &bad_slots, &schedule)
                .is_err()
        );
        let mut bad_ids = expert_ids.clone();
        bad_ids[slots[0] as usize] ^= 1;
        assert!(validate_packed_expert_schedule(128, &bad_ids, &rows, &slots, &schedule).is_err());

        let mut cursor = 0usize;
        let maximum = (0..MOE_EXPERT_COUNT)
            .map(|expert| {
                let len = if expert < 16 { 33 } else { 1 };
                let bucket = ExpertBucket {
                    expert,
                    start: cursor,
                    len,
                };
                cursor += len;
                bucket
            })
            .collect::<Vec<_>>();
        assert_eq!(cursor, 128 * MOE_TOP_K);
        assert_eq!(
            packed_grouped_expert_tiles(128, &maximum).unwrap().len(),
            272
        );
        assert_eq!(
            packed_grouped_iq2_mma16_tiles(128, &maximum).unwrap().len(),
            288
        );

        let mut cursor = 0usize;
        let maximum = (0..MOE_EXPERT_COUNT)
            .map(|expert| {
                let len = match expert {
                    0..5 => 2_017,
                    5 => 1_953,
                    _ => 1,
                };
                let bucket = ExpertBucket {
                    expert,
                    start: cursor,
                    len,
                };
                cursor += len;
                bucket
            })
            .collect::<Vec<_>>();
        assert_eq!(cursor, DEEPSEEK_V4_PREFILL_MAX_TOKENS * MOE_TOP_K);
        assert_eq!(
            packed_grouped_expert_tiles(DEEPSEEK_V4_PREFILL_MAX_TOKENS, &maximum)
                .unwrap()
                .len(),
            PACKED_GROUPED_EXPERT_MAX_TILES
        );

        let mut cursor = 0usize;
        let mma16_maximum = (0..MOE_EXPERT_COUNT)
            .map(|expert| {
                let len = match expert {
                    0..5 => 2_033,
                    5 => 1_873,
                    _ => 1,
                };
                let bucket = ExpertBucket {
                    expert,
                    start: cursor,
                    len,
                };
                cursor += len;
                bucket
            })
            .collect::<Vec<_>>();
        assert_eq!(cursor, PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS * MOE_TOP_K);
        assert_eq!(
            packed_grouped_iq2_mma16_tiles(PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS, &mma16_maximum,)
                .unwrap()
                .len(),
            PACKED_GROUPED_IQ2_MMA16_MAX_TILES
        );
        assert_eq!(PACKED_GROUPED_IQ2_MMA16_MAX_TILES, 1_008);
        assert_eq!(PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS, 3_024);
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[test]
    fn packed_post_route_expert_counts_pin_panel_boundaries() {
        let lengths = [31usize, 32, 33, 128, 128, 128, 128, 128, 32];
        let mut cursor = 0usize;
        let schedule = lengths
            .into_iter()
            .enumerate()
            .map(|(expert, len)| {
                let bucket = ExpertBucket {
                    expert,
                    start: cursor,
                    len,
                };
                cursor += len;
                bucket
            })
            .collect::<Vec<_>>();
        assert_eq!(cursor, 128 * MOE_TOP_K);

        let counts = packed_post_route_expert_counts(128, &schedule).unwrap();
        assert_eq!(&counts[..lengths.len()], &lengths.map(|len| len as u16));
        assert!(counts[lengths.len()..].iter().all(|&count| count == 0));
        assert_eq!(
            counts
                .iter()
                .map(|&count| usize::from(count).div_ceil(32))
                .sum::<usize>(),
            25
        );

        let copy_schedule = || {
            schedule
                .iter()
                .map(|bucket| ExpertBucket {
                    expert: bucket.expert,
                    start: bucket.start,
                    len: bucket.len,
                })
                .collect::<Vec<_>>()
        };
        let mut duplicate = copy_schedule();
        duplicate[1].expert = duplicate[0].expert;
        assert!(packed_post_route_expert_counts(128, &duplicate).is_err());

        let mut oversized = copy_schedule();
        oversized[0].len = 129;
        assert!(packed_post_route_expert_counts(128, &oversized).is_err());
        assert!(packed_post_route_expert_counts(128, &schedule[..8]).is_err());
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[test]
    fn packed_grouped_iq2_count_census_fixture_is_canonical() {
        const DOMAIN: &[u8] = b"qwen-llm:dsv4:packed-grouped-iq2-counts:v1\0";
        const GROUPED_LAYERS: [usize; 25] = [
            0, 2, 3, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 35, 36, 37, 38,
            39, 41,
        ];

        fn hex(digest: impl AsRef<[u8]>) -> String {
            digest
                .as_ref()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect()
        }

        let fixture = packed_grouped_iq2_count_census_fixture();
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(
            fixture.count_payload_domain_hex,
            "7177656e2d6c6c6d3a647376343a7061636b65642d67726f757065642d6971322d636f756e74733a763100"
        );
        assert_eq!(
            fixture.count_payload_sha256,
            "0ab9925350288116288794f3d7f5081595dfad4ffdb9671a9a146f27568358a6"
        );
        assert_eq!(
            fixture.model_content_id,
            "ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2"
        );
        assert_eq!(
            fixture.prompt_token_ids_sha256,
            "b57816bcb0d5fdf5a8e2ddc7a0afe9e57fb0ca6ffc2b849285e1635d04772843"
        );
        assert_eq!(fixture.n_tokens, 128);
        assert_eq!(fixture.top_k, MOE_TOP_K);
        assert_eq!(fixture.expert_count, MOE_EXPERT_COUNT);
        assert_eq!(fixture.grouped_layer_ids, GROUPED_LAYERS);
        assert_eq!(fixture.layers.len(), GROUPED_LAYERS.len());

        let mut payload = Sha256::new();
        payload.update(DOMAIN);
        let mut active_count_histogram = vec![0usize; fixture.n_tokens];
        let mut tile_histogram = vec![0usize; 4];
        let mut total_active_experts = 0usize;
        let mut total_t32 = 0usize;
        let mut total_padding = 0usize;
        for (&expected_layer, layer) in GROUPED_LAYERS.iter().zip(&fixture.layers) {
            assert_eq!(layer.layer, expected_layer);
            assert_eq!(layer.counts.len(), MOE_EXPERT_COUNT);
            assert!(
                layer
                    .counts
                    .iter()
                    .all(|&count| usize::from(count) <= fixture.n_tokens)
            );
            assert_eq!(
                layer
                    .counts
                    .iter()
                    .map(|&count| usize::from(count))
                    .sum::<usize>(),
                fixture.n_tokens * fixture.top_k
            );
            let active_experts = layer.counts.iter().filter(|&&count| count > 0).count();
            let t32 = layer
                .counts
                .iter()
                .map(|&count| usize::from(count).div_ceil(32))
                .sum::<usize>();
            let padding = 32 * t32 - fixture.n_tokens * fixture.top_k;
            assert_eq!(layer.active_experts, active_experts);
            assert_eq!(layer.t32, t32);
            assert_eq!(layer.padding, padding);
            total_active_experts += active_experts;
            total_t32 += t32;
            total_padding += padding;
            payload.update((layer.layer as u32).to_le_bytes());
            for &count in &layer.counts {
                payload.update(count.to_le_bytes());
                if count > 0 {
                    active_count_histogram[usize::from(count) - 1] += 1;
                    tile_histogram[usize::from(count).div_ceil(32) - 1] += 1;
                }
            }
        }
        assert_eq!(fixture.total_active_experts, total_active_experts);
        assert_eq!(fixture.inactive_experts, 25 * 256 - total_active_experts);
        assert_eq!(fixture.total_t32, total_t32);
        assert_eq!(fixture.total_padding, total_padding);
        assert_eq!(
            fixture.active_count_histogram_1_to_128,
            active_count_histogram
        );
        assert_eq!(fixture.tile_histogram_1_to_4, tile_histogram);
        assert_eq!(total_active_experts, 1_542);
        assert_eq!(total_t32, 1_748);
        assert_eq!(total_padding, 36_736);
        assert_eq!(fixture.tile_histogram_1_to_4, [1_395, 100, 35, 12]);
        assert_eq!(hex(payload.finalize()), fixture.count_payload_sha256);
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[test]
    fn packed_grouped_iq2_route_census_fixture_is_canonical() {
        const COUNT_DOMAIN: &[u8] = b"qwen-llm:dsv4:packed-grouped-iq2-counts:v1\0";
        const ROUTE_DOMAIN: &[u8] = b"qwen-llm:dsv4:packed-grouped-iq2-route-ids:v1\0";

        fn hex(digest: impl AsRef<[u8]>) -> String {
            digest
                .as_ref()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect()
        }

        fn decode_hex_32(value: &str) -> [u8; 32] {
            assert_eq!(value.len(), 64);
            std::array::from_fn(|index| {
                u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).unwrap()
            })
        }

        let counts = packed_grouped_iq2_count_census_fixture();
        let routes = packed_grouped_iq2_route_census_fixture();
        assert_eq!(routes.schema_version, 1);
        assert_eq!(
            routes.route_payload_domain_hex,
            "7177656e2d6c6c6d3a647376343a7061636b65642d67726f757065642d6971322d726f7574652d6964733a763100"
        );
        assert_eq!(
            routes.route_payload_sha256,
            "505cb93ff9c3e1557bbad8f27a773e08b0e8c3475fbb4d7096069667c1fbafdd"
        );
        assert_eq!(routes.model_content_id, counts.model_content_id);
        assert_eq!(
            routes.prompt_token_ids_sha256,
            counts.prompt_token_ids_sha256
        );
        assert!(routes.prompt_token_ids.is_none());
        assert_eq!(routes.count_payload_sha256, counts.count_payload_sha256);
        assert_eq!(routes.n_tokens, counts.n_tokens);
        assert_eq!(routes.top_k, counts.top_k);
        assert_eq!(routes.expert_count, counts.expert_count);
        assert_eq!(routes.layer_count, counts.layers.len());
        assert_eq!(routes.route_count, routes.n_tokens * routes.top_k);
        assert_eq!(routes.grouped_layer_ids, counts.grouped_layer_ids);
        assert_eq!(routes.layers.len(), routes.layer_count);

        let mut count_payload = Sha256::new();
        count_payload.update(COUNT_DOMAIN);
        for layer in &counts.layers {
            count_payload.update((layer.layer as u32).to_le_bytes());
            for &count in &layer.counts {
                count_payload.update(count.to_le_bytes());
            }
        }
        let count_payload_digest = count_payload.finalize();
        assert_eq!(hex(count_payload_digest), routes.count_payload_sha256);

        let mut route_payload = Sha256::new();
        route_payload.update(ROUTE_DOMAIN);
        route_payload.update(decode_hex_32(&routes.model_content_id));
        route_payload.update(decode_hex_32(&routes.prompt_token_ids_sha256));
        route_payload.update(count_payload_digest);
        for value in [
            routes.n_tokens,
            routes.top_k,
            routes.expert_count,
            routes.layer_count,
            routes.route_count,
        ] {
            route_payload.update((value as u32).to_le_bytes());
        }
        for (count_layer, route_layer) in counts.layers.iter().zip(&routes.layers) {
            assert_eq!(route_layer.layer, count_layer.layer);
            assert_eq!(route_layer.route_expert_ids.len(), routes.route_count);
            let (expert_ids, rows, slots, schedule) = packed_grouped_schedule_from_route_ids(
                routes.n_tokens,
                &route_layer.route_expert_ids,
            );
            assert_eq!(
                packed_post_route_expert_ids(
                    routes.n_tokens,
                    &expert_ids,
                    &rows,
                    &slots,
                    &schedule,
                )
                .unwrap(),
                route_layer.route_expert_ids
            );
            assert_eq!(
                packed_post_route_expert_counts(routes.n_tokens, &schedule)
                    .unwrap()
                    .as_slice(),
                count_layer.counts.as_slice()
            );
            assert_eq!(
                packed_grouped_expert_tiles(routes.n_tokens, &schedule)
                    .unwrap()
                    .len(),
                count_layer.t32
            );
            route_payload.update((route_layer.layer as u32).to_le_bytes());
            for &expert in &route_layer.route_expert_ids {
                route_payload.update(expert.to_le_bytes());
            }
        }
        assert_eq!(hex(route_payload.finalize()), routes.route_payload_sha256);
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[test]
    fn packed_grouped_iq2_representative_route_census_fixture_is_canonical() {
        const COUNT_DOMAIN: &[u8] = b"qwen-llm:dsv4:packed-grouped-iq2-counts:v1\0";
        const ROUTE_DOMAIN: &[u8] = b"qwen-llm:dsv4:packed-grouped-iq2-route-ids:v1\0";

        fn hex(digest: impl AsRef<[u8]>) -> String {
            digest
                .as_ref()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect()
        }

        fn decode_hex_32(value: &str) -> [u8; 32] {
            assert_eq!(value.len(), 64);
            std::array::from_fn(|index| {
                u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).unwrap()
            })
        }

        let routes = packed_grouped_iq2_representative_route_census_fixture();
        assert_eq!(routes.schema_version, 1);
        assert_eq!(routes.route_payload_domain_hex, hex(ROUTE_DOMAIN));
        assert_eq!(
            routes.route_payload_sha256,
            "7454b2692359464c0d932e1c2fffe53d345e0ea969db9de67458ed992ddd539c"
        );
        assert_eq!(
            routes.model_content_id,
            "ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2"
        );
        assert_eq!(
            routes.prompt_token_ids_sha256,
            "ee09a95c18d0231d195a88cc4d96c34ae27df5b3fd2fecb0c5e909807e4be8da"
        );
        let prompt_token_ids = routes
            .prompt_token_ids
            .as_ref()
            .expect("representative route fixture retains exact prompt IDs");
        assert_eq!(prompt_token_ids.len(), routes.n_tokens);
        assert_eq!(
            hex(Sha256::digest(bytemuck::cast_slice(prompt_token_ids))),
            routes.prompt_token_ids_sha256
        );
        assert_eq!(
            routes.count_payload_sha256,
            "0486f37c39a37cab0cbb1cfe41d8d4fca401059b7abbe872901e841fdd4d394a"
        );
        assert_eq!(routes.n_tokens, 128);
        assert_eq!(routes.top_k, MOE_TOP_K);
        assert_eq!(routes.expert_count, MOE_EXPERT_COUNT);
        assert_eq!(routes.layer_count, 25);
        assert_eq!(routes.route_count, routes.n_tokens * routes.top_k);
        assert_eq!(routes.layers.len(), routes.layer_count);

        let mut count_payload = Sha256::new();
        count_payload.update(COUNT_DOMAIN);
        let mut route_payload = Sha256::new();
        route_payload.update(ROUTE_DOMAIN);
        route_payload.update(decode_hex_32(&routes.model_content_id));
        route_payload.update(decode_hex_32(&routes.prompt_token_ids_sha256));
        let mut total_active_experts = 0usize;
        let mut total_tiles_16 = 0usize;
        let mut total_padding_16 = 0usize;
        let mut counts_by_layer = Vec::with_capacity(routes.layer_count);
        for layer in &routes.layers {
            assert_eq!(layer.route_expert_ids.len(), routes.route_count);
            let (_, rows, slots, schedule) =
                packed_grouped_schedule_from_route_ids(routes.n_tokens, &layer.route_expert_ids);
            let counts = packed_post_route_expert_counts(routes.n_tokens, &schedule).unwrap();
            assert_eq!(counts.len(), MOE_EXPERT_COUNT);
            let tiles_16 = packed_grouped_iq2_mma16_tiles(routes.n_tokens, &schedule)
                .unwrap()
                .len();
            total_active_experts += schedule.len();
            total_tiles_16 += tiles_16;
            total_padding_16 += tiles_16 * 16 - routes.route_count;
            count_payload.update((layer.layer as u32).to_le_bytes());
            for &count in &counts {
                count_payload.update(count.to_le_bytes());
            }
            assert_eq!(rows.len(), routes.route_count);
            assert_eq!(slots.len(), routes.route_count);
            counts_by_layer.push(counts);
        }
        let count_payload_digest = count_payload.finalize();
        assert_eq!(hex(count_payload_digest), routes.count_payload_sha256);
        route_payload.update(count_payload_digest);
        for value in [
            routes.n_tokens,
            routes.top_k,
            routes.expert_count,
            routes.layer_count,
            routes.route_count,
        ] {
            route_payload.update((value as u32).to_le_bytes());
        }
        for layer in &routes.layers {
            route_payload.update((layer.layer as u32).to_le_bytes());
            for &expert in &layer.route_expert_ids {
                route_payload.update(expert.to_le_bytes());
            }
        }
        assert_eq!(hex(route_payload.finalize()), routes.route_payload_sha256);
        assert_eq!(total_active_experts, 3_154);
        assert_eq!(total_tiles_16, 3_558);
        assert_eq!(total_padding_16, 37_728);
        assert_eq!(counts_by_layer.len(), routes.layer_count);
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[test]
    fn packed_all_iq3_route_census_fixture_is_canonical() {
        const COUNT_DOMAIN: &[u8] = b"qwen-llm:dsv4:packed-all-iq3-counts:v1\0";
        const ROUTE_DOMAIN: &[u8] = b"qwen-llm:dsv4:packed-all-iq3-route-ids:v1\0";
        const LAYERS: [usize; 16] = [1, 4, 5, 8, 9, 20, 24, 27, 28, 29, 30, 31, 32, 33, 34, 40];
        const BUCKETS: [usize; 16] = [
            23, 38, 46, 68, 73, 72, 71, 79, 75, 79, 84, 77, 65, 62, 59, 65,
        ];

        fn hex(digest: impl AsRef<[u8]>) -> String {
            digest
                .as_ref()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect()
        }

        fn decode_hex_32(value: &str) -> [u8; 32] {
            assert_eq!(value.len(), 64);
            std::array::from_fn(|index| {
                u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).unwrap()
            })
        }

        let fixture = packed_all_iq3_route_census_fixture();
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.count_payload_domain_hex, hex(COUNT_DOMAIN));
        assert_eq!(fixture.route_payload_domain_hex, hex(ROUTE_DOMAIN));
        assert_eq!(
            fixture.count_payload_sha256,
            "72b5d2dba179d5d65334ec413bd82df784b9001d545b8c6e7b74aa29f309ff7c"
        );
        assert_eq!(
            fixture.route_payload_sha256,
            "ee106712f42aed80cc559414140327537dd9f256110b6acffee6a1b893319582"
        );
        assert_eq!(
            fixture.model_content_id,
            "ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2"
        );
        assert_eq!(
            fixture.prompt_token_ids_sha256,
            "b57816bcb0d5fdf5a8e2ddc7a0afe9e57fb0ca6ffc2b849285e1635d04772843"
        );
        assert_eq!(fixture.n_tokens, 128);
        assert_eq!(fixture.top_k, MOE_TOP_K);
        assert_eq!(fixture.expert_count, MOE_EXPERT_COUNT);
        assert_eq!(fixture.layer_count, LAYERS.len());
        assert_eq!(fixture.route_count, fixture.n_tokens * fixture.top_k);
        assert_eq!(fixture.all_iq3_layer_ids, LAYERS);
        assert_eq!(fixture.layers.len(), fixture.layer_count);

        let mut count_payload = Sha256::new();
        count_payload.update(COUNT_DOMAIN);
        count_payload.update(decode_hex_32(&fixture.model_content_id));
        count_payload.update(decode_hex_32(&fixture.prompt_token_ids_sha256));
        for value in [
            fixture.n_tokens,
            fixture.top_k,
            fixture.expert_count,
            fixture.layer_count,
            fixture.route_count,
        ] {
            count_payload.update((value as u32).to_le_bytes());
        }

        let mut route_payload = Sha256::new();
        route_payload.update(ROUTE_DOMAIN);
        route_payload.update(decode_hex_32(&fixture.model_content_id));
        route_payload.update(decode_hex_32(&fixture.prompt_token_ids_sha256));
        route_payload.update(decode_hex_32(&fixture.count_payload_sha256));
        for value in [
            fixture.n_tokens,
            fixture.top_k,
            fixture.expert_count,
            fixture.layer_count,
            fixture.route_count,
        ] {
            route_payload.update((value as u32).to_le_bytes());
        }

        let mut total_active_experts = 0usize;
        let mut total_t32 = 0usize;
        let mut total_padding = 0usize;
        for (index, layer) in fixture.layers.iter().enumerate() {
            assert_eq!(layer.layer, LAYERS[index]);
            assert_eq!(layer.active_experts, BUCKETS[index]);
            assert_eq!(layer.expert_counts.len(), MOE_EXPERT_COUNT);
            assert_eq!(layer.route_expert_ids.len(), fixture.route_count);
            assert_eq!(
                layer
                    .expert_counts
                    .iter()
                    .map(|&count| usize::from(count))
                    .sum::<usize>(),
                fixture.route_count
            );
            assert_eq!(
                layer
                    .expert_counts
                    .iter()
                    .filter(|&&count| count > 0)
                    .count(),
                layer.active_experts
            );
            let t32 = layer
                .expert_counts
                .iter()
                .map(|&count| usize::from(count).div_ceil(32))
                .sum::<usize>();
            assert_eq!(layer.t32, t32);
            assert_eq!(layer.padding, t32 * 32 - fixture.route_count);

            let (expert_ids, rows, slots, schedule) =
                packed_grouped_schedule_from_route_ids(fixture.n_tokens, &layer.route_expert_ids);
            assert_eq!(schedule.len(), layer.active_experts);
            assert_eq!(
                packed_post_route_expert_ids(
                    fixture.n_tokens,
                    &expert_ids,
                    &rows,
                    &slots,
                    &schedule,
                )
                .unwrap(),
                layer.route_expert_ids
            );
            assert_eq!(
                packed_post_route_expert_counts(fixture.n_tokens, &schedule)
                    .unwrap()
                    .as_slice(),
                layer.expert_counts.as_slice()
            );
            assert_eq!(
                packed_grouped_expert_tiles(fixture.n_tokens, &schedule)
                    .unwrap()
                    .len(),
                layer.t32
            );

            count_payload.update((layer.layer as u32).to_le_bytes());
            for &count in &layer.expert_counts {
                count_payload.update(count.to_le_bytes());
            }
            route_payload.update((layer.layer as u32).to_le_bytes());
            for &expert in &layer.route_expert_ids {
                route_payload.update(expert.to_le_bytes());
            }
            total_active_experts += layer.active_experts;
            total_t32 += layer.t32;
            total_padding += layer.padding;
        }
        assert_eq!(total_active_experts, 1_036);
        assert_eq!(total_t32, 1_187);
        assert_eq!(total_padding, 25_696);
        assert_eq!(fixture.total_active_experts, total_active_experts);
        assert_eq!(fixture.total_t32, total_t32);
        assert_eq!(fixture.total_padding, total_padding);
        assert_eq!(hex(count_payload.finalize()), fixture.count_payload_sha256);
        assert_eq!(hex(route_payload.finalize()), fixture.route_payload_sha256);
    }

    #[test]
    fn packed_grouped_mapped_iq3_consumes_explicit_source_rows() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const H: usize = 256;
        const O: usize = 64;
        const N: usize = 2;
        const E: usize = MOE_EXPERT_COUNT;
        const K: usize = MOE_TOP_K;

        let bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, H, O, E, 23);
        let (_expert_ids, rows, slots, schedule) = grouped_test_schedule(N);
        let mapped_rows = rows
            .iter()
            .map(|&row| i32::try_from(N - 1).unwrap() - row)
            .collect::<Vec<_>>();
        assert!(
            mapped_rows
                .iter()
                .zip(&slots)
                .any(|(&row, &slot)| row as usize != slot as usize / K)
        );
        let input_values = (0..N * H)
            .map(|index| ((index * 41 + 7) % 257) as f32 * 0.002 - 0.25)
            .collect::<Vec<_>>();
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&input_values),
            vec![H as u64, N as u64],
            GgmlType::F32,
        )
        .unwrap();
        let source_rows = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&mapped_rows),
            vec![(N * K) as u64],
            GgmlType::I32,
        )
        .unwrap();
        let destination_slots = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&slots),
            vec![(N * K) as u64],
            GgmlType::I32,
        )
        .unwrap();
        let gathered = MetalTensor::zeros_f32(&ctx, vec![H as u64, N as u64]).unwrap();
        let projected = MetalTensor::zeros_f32(&ctx, vec![O as u64, N as u64]).unwrap();
        let control = grouped_guarded_f32(&ctx, vec![O as u64, (N * K) as u64], 7.0);
        let candidate = grouped_guarded_f32(&ctx, vec![O as u64, (N * K) as u64], 11.0);

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for bucket in &schedule {
            let row_view = i32_slice(
                &source_rows,
                bucket.start,
                bucket.len,
                "mapped IQ3 control rows",
            )
            .unwrap();
            let slot_view = i32_slice(
                &destination_slots,
                bucket.start,
                bucket.len,
                "mapped IQ3 control slots",
            )
            .unwrap();
            let input_view = f32_prefix(
                &gathered,
                vec![H as u64, bucket.len as u64],
                "mapped IQ3 control input",
            )
            .unwrap();
            let output_view = f32_prefix(
                &projected,
                vec![O as u64, bucket.len as u64],
                "mapped IQ3 control output",
            )
            .unwrap();
            encode_get_rows_f32(
                &ctx,
                &encoder,
                &input,
                &row_view,
                &input_view,
                bucket.len,
                H,
            )
            .unwrap();
            let weight =
                expert_weight_view(&bank, H, O, bucket.expert, "mapped IQ3 control weight")
                    .unwrap();
            encode_batch_projection(
                &ctx,
                &encoder,
                &weight,
                &input_view,
                &output_view,
                H,
                O,
                bucket.len,
                "mapped IQ3 control projection",
            )
            .unwrap();
            crate::metal::encode_scatter_rows_f32_unique(
                &ctx,
                &encoder,
                &output_view,
                &slot_view,
                &control,
                O,
                bucket.len,
            )
            .unwrap();
        }
        encode_packed_grouped_mapped_iq3_xxs_f32(
            &ctx,
            &encoder,
            &bank,
            &input,
            &source_rows,
            &destination_slots,
            &schedule,
            &candidate,
            H,
            O,
            E,
            K,
            N,
            N,
            N * K,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "{:?}", command.error());

        assert_eq!(
            host_read_f32(&candidate, "mapped IQ3 candidate").unwrap(),
            host_read_f32(&control, "mapped IQ3 control").unwrap()
        );
        assert_grouped_guards("mapped IQ3 control", &control);
        assert_grouped_guards("mapped IQ3 candidate", &candidate);
    }

    #[test]
    fn q8_f32_mma_r2c4k64_reduces_operand_rounding_and_preserves_guards() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };

        fn submit(
            ctx: &MetalContext,
            encode: impl FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
        ) {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let result = encode(&encoder);
            encoder.end();
            result.unwrap();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none(), "{:?}", command.error());
        }

        for (n_in, n_out) in [(64usize, 16usize), (128, 32)] {
            let weight = q8_precision_test_weight(&ctx, n_in, n_out);
            for n_tokens in [1usize, 31, 32, 33, 128] {
                let padded_tokens = n_tokens.div_ceil(32) * 32;
                let input_storage = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&q8_precision_test_input(padded_tokens * n_in)),
                    vec![n_in as u64, padded_tokens as u64],
                    GgmlType::F32,
                )
                .unwrap();
                let input = input_storage.view_subrange(0, vec![n_in as u64, n_tokens as u64]);
                let exact = grouped_guarded_f32(&ctx, vec![n_out as u64, n_tokens as u64], 3.0);
                let half = grouped_guarded_f32(&ctx, vec![n_out as u64, n_tokens as u64], 5.0);
                let candidate = grouped_guarded_f32(&ctx, vec![n_out as u64, n_tokens as u64], 7.0);
                let repeat = grouped_guarded_f32(&ctx, vec![n_out as u64, n_tokens as u64], 11.0);
                submit(&ctx, |encoder| {
                    crate::metal::encode_mat_vec_q8_0_batch_f32(
                        &ctx, encoder, &weight, &input, &exact, n_in, n_out, n_tokens,
                    )
                    .map_err(DeepSeekV4MetalError::Metal)
                });
                submit(&ctx, |encoder| {
                    crate::metal::encode_mat_mat_q8_0_f32(
                        &ctx, encoder, &weight, &input, &half, n_in, n_out, n_tokens,
                    )
                    .map_err(DeepSeekV4MetalError::Metal)
                });
                for output in [&candidate, &repeat] {
                    submit(&ctx, |encoder| {
                        encode_q8_f32_mma_r2c4k64(
                            &ctx, encoder, &weight, &input, output, n_in, n_out, n_tokens,
                        )
                    });
                }

                let exact_values = host_read_f32(&exact, "Q8 F32 exact").unwrap();
                let half_values = host_read_f32(&half, "Q8 F32 half").unwrap();
                let candidate_values = host_read_f32(&candidate, "Q8 F32 candidate").unwrap();
                let repeat_values = host_read_f32(&repeat, "Q8 F32 repeat").unwrap();
                let half_diff = q8_differential(&half_values, &exact_values);
                let candidate_diff = q8_differential(&candidate_values, &exact_values);
                assert!(
                    1.0 - candidate_diff.0 <= 1.0 - half_diff.0 + 1e-15
                        && candidate_diff.1 < half_diff.1
                        && candidate_diff.2 < half_diff.2,
                    "K={n_in} M={n_out} N={n_tokens} half={half_diff:?} candidate={candidate_diff:?}"
                );
                assert_eq!(
                    candidate_values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    repeat_values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "K={n_in} M={n_out} N={n_tokens} repeat"
                );
                for (label, tensor) in [
                    ("Q8 F32 exact", &exact),
                    ("Q8 F32 half", &half),
                    ("Q8 F32 candidate", &candidate),
                    ("Q8 F32 repeat", &repeat),
                ] {
                    assert_grouped_guards(label, tensor);
                }
            }
        }
    }

    #[test]
    fn q8_f32_mma_r2c4k64_rejects_unqualified_storage_before_dispatch() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const K: usize = 64;
        const M: usize = 16;
        const N: usize = 33;

        fn reject(
            ctx: &MetalContext,
            encode: impl FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
        ) -> String {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let result = encode(&encoder);
            encoder.end();
            result.unwrap_err().to_string()
        }

        let weight = q8_precision_test_weight(&ctx, K, M);
        let input = MetalTensor::zeros_f32(&ctx, vec![K as u64, N as u64]).unwrap();
        let output = MetalTensor::zeros_f32(&ctx, vec![M as u64, N as u64]).unwrap();
        let error = reject(&ctx, |encoder| {
            encode_q8_f32_mma_r2c4k64(&ctx, encoder, &weight, &input, &output, K, M, N)
        });
        assert!(error.contains("padded input backing"), "{error}");

        let padded = MetalTensor::zeros_f32(&ctx, vec![K as u64, 64]).unwrap();
        let padded_input = padded.view_subrange(0, vec![K as u64, N as u64]);
        let overlapping_output = padded.view_subrange(0, vec![M as u64, N as u64]);
        let error = reject(&ctx, |encoder| {
            encode_q8_f32_mma_r2c4k64(
                &ctx,
                encoder,
                &weight,
                &padded_input,
                &overlapping_output,
                K,
                M,
                N,
            )
        });
        assert!(error.contains("overlaps an input"), "{error}");

        let padding_output = padded.view_subrange((K * N) as u64, vec![M as u64, N as u64]);
        let error = reject(&ctx, |encoder| {
            encode_q8_f32_mma_r2c4k64(
                &ctx,
                encoder,
                &weight,
                &padded_input,
                &padding_output,
                K,
                M,
                N,
            )
        });
        assert!(error.contains("overlaps an input"), "{error}");

        let malformed_input = padded.view_subrange(0, vec![(K * N) as u64]);
        let error = reject(&ctx, |encoder| {
            encode_q8_f32_mma_r2c4k64(&ctx, encoder, &weight, &malformed_input, &output, K, M, N)
        });
        assert!(error.contains("input"), "{error}");

        let wrong_weight = MetalTensor::zeros_f32(&ctx, vec![K as u64, M as u64]).unwrap();
        let error = reject(&ctx, |encoder| {
            encode_q8_f32_mma_r2c4k64(
                &ctx,
                encoder,
                &wrong_weight,
                &padded_input,
                &output,
                K,
                M,
                N,
            )
        });
        assert!(error.contains("invalid geometry or storage"), "{error}");
    }

    #[test]
    #[ignore = "sealed GO; do not rerun without material implementation or device drift"]
    fn profile_q8_f32_mma_r2c4k64_attention_output_packet() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        assert_eq!(
            ctx.device.name().to_string(),
            PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE
        );
        const N: usize = 128;
        const SAMPLES: usize = 24;

        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        enum Variant {
            Exact,
            HalfA,
            HalfB,
            HalfBoth,
            F32A,
            F32B,
            F32Both,
        }

        fn projections(variant: Variant) -> (Q8PrecisionProjection, Q8PrecisionProjection) {
            match variant {
                Variant::Exact => (Q8PrecisionProjection::Exact, Q8PrecisionProjection::Exact),
                Variant::HalfA => (
                    Q8PrecisionProjection::HalfMatrix,
                    Q8PrecisionProjection::Exact,
                ),
                Variant::HalfB => (
                    Q8PrecisionProjection::Exact,
                    Q8PrecisionProjection::HalfMatrix,
                ),
                Variant::HalfBoth => (
                    Q8PrecisionProjection::HalfMatrix,
                    Q8PrecisionProjection::HalfMatrix,
                ),
                Variant::F32A => (
                    Q8PrecisionProjection::F32Matrix,
                    Q8PrecisionProjection::Exact,
                ),
                Variant::F32B => (
                    Q8PrecisionProjection::Exact,
                    Q8PrecisionProjection::F32Matrix,
                ),
                Variant::F32Both => (
                    Q8PrecisionProjection::F32Matrix,
                    Q8PrecisionProjection::F32Matrix,
                ),
            }
        }

        let output_a = q8_output_test_weight(&ctx, GROUP_WIDTH, LOW_RANK_WIDTH, 3);
        let output_b = q8_output_test_weight(&ctx, LOW_RANK_WIDTH, DEEPSEEK_V4_HIDDEN_SIZE, 11);
        let attention_values = (0..QUERY_WIDTH * N)
            .map(|index| ((index * 31 + index / 11 + 5) % 257) as f32 * 0.004 - 0.51)
            .collect::<Vec<_>>();
        let attention = grouped_guarded_f32(&ctx, vec![QUERY_WIDTH as u64, N as u64], 13.0);
        write_raw_f32(&attention, &attention_values);
        let mut scratch = DeepSeekV4PrefillScratch::new(&ctx, DEEPSEEK_V4_CSA_TOP_K)
            .unwrap()
            .attention;
        scratch.low_rank = grouped_guarded_f32(&ctx, vec![LOW_RANK_WIDTH as u64, N as u64], 17.0);
        scratch.output =
            grouped_guarded_f32(&ctx, vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, N as u64], 19.0);
        scratch.group_input = grouped_guarded_f32(&ctx, vec![GROUP_WIDTH as u64, N as u64], 23.0);
        scratch.group_output = grouped_guarded_f32(&ctx, vec![1_024, N as u64], 29.0);
        let low_rank = f32_prefix(
            &scratch.low_rank,
            vec![LOW_RANK_WIDTH as u64, N as u64],
            "Q8 F32 packet low rank",
        )
        .unwrap();
        let output = f32_prefix(
            &scratch.output,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, N as u64],
            "Q8 F32 packet output",
        )
        .unwrap();

        let execute = |variant: Variant| -> (f64, f64) {
            let started = std::time::Instant::now();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let result = if variant == Variant::Exact {
                scratch.encode_output(&ctx, &encoder, &attention, &output_a, &output_b, N)
            } else {
                let (output_a_projection, output_b_projection) = projections(variant);
                scratch.encode_output_q8_precision(
                    &ctx,
                    &encoder,
                    &attention,
                    &output_a,
                    &output_b,
                    N,
                    output_a_projection,
                    output_b_projection,
                )
            };
            encoder.end();
            result.unwrap();
            command.commit();
            command.waitUntilCompleted();
            let wall_ms = started.elapsed().as_secs_f64() * 1e3;
            assert!(
                command.error().is_none(),
                "{variant:?}: {:?}",
                command.error()
            );
            (
                (command.GPUEndTime() - command.GPUStartTime()) * 1e3,
                wall_ms,
            )
        };
        let capture = || {
            (
                host_read_f32(&low_rank, "Q8 F32 packet low rank").unwrap(),
                host_read_f32(&output, "Q8 F32 packet output").unwrap(),
            )
        };
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };

        execute(Variant::Exact);
        let (reference_low, reference_output) = capture();
        let evaluate = |variant| {
            execute(variant);
            let (low, output) = capture();
            let low_diff = q8_differential(&low, &reference_low);
            let output_diff = q8_differential(&output, &reference_output);
            (low, output, low_diff, output_diff)
        };
        let half_a = evaluate(Variant::HalfA);
        let half_b = evaluate(Variant::HalfB);
        let half_both = evaluate(Variant::HalfBoth);
        let f32_a = evaluate(Variant::F32A);
        let f32_b = evaluate(Variant::F32B);
        let f32_both = evaluate(Variant::F32Both);

        for (variant, expected_low, expected_output) in [
            (Variant::F32A, &f32_a.0, &f32_a.1),
            (Variant::F32B, &f32_b.0, &f32_b.1),
            (Variant::F32Both, &f32_both.0, &f32_both.1),
        ] {
            execute(variant);
            let (actual_low, actual_output) = capture();
            assert_eq!(
                bits(&actual_low),
                bits(expected_low),
                "{variant:?} low repeat"
            );
            assert_eq!(
                bits(&actual_output),
                bits(expected_output),
                "{variant:?} output repeat"
            );
        }

        eprintln!(
            "deepseek_v4 q8_f32_precision half_a_low={:?} half_a_output={:?} half_b_output={:?} half_both_low={:?} half_both_output={:?} f32_a_low={:?} f32_a_output={:?} f32_b_output={:?} f32_both_low={:?} f32_both_output={:?}",
            half_a.2,
            half_a.3,
            half_b.3,
            half_both.2,
            half_both.3,
            f32_a.2,
            f32_a.3,
            f32_b.3,
            f32_both.2,
            f32_both.3,
        );

        for index in 0usize..5 {
            if index.is_multiple_of(2) {
                execute(Variant::Exact);
                execute(Variant::F32Both);
            } else {
                execute(Variant::F32Both);
                execute(Variant::Exact);
            }
        }
        let collect = |variant| (0..SAMPLES).map(|_| execute(variant)).collect::<Vec<_>>();
        let control_before = collect(Variant::Exact);
        let candidate = collect(Variant::F32Both);
        let control_after = collect(Variant::Exact);
        let split = |samples: &[(f64, f64)]| {
            (
                samples.iter().map(|sample| sample.0).collect::<Vec<_>>(),
                samples.iter().map(|sample| sample.1).collect::<Vec<_>>(),
            )
        };
        let (control_before_gpu, control_before_wall) = split(&control_before);
        let (candidate_gpu, candidate_wall) = split(&candidate);
        let (control_after_gpu, control_after_wall) = split(&control_after);
        let gpu_control_drift = relative_drift(
            median_ms(&control_before_gpu),
            median_ms(&control_after_gpu),
        );
        let wall_control_drift = relative_drift(
            median_ms(&control_before_wall),
            median_ms(&control_after_wall),
        );
        let gpu_candidate_drift = relative_drift(
            median_ms(&candidate_gpu[..SAMPLES / 2]),
            median_ms(&candidate_gpu[SAMPLES / 2..]),
        );
        let wall_candidate_drift = relative_drift(
            median_ms(&candidate_wall[..SAMPLES / 2]),
            median_ms(&candidate_wall[SAMPLES / 2..]),
        );
        let gpu_control_median = median_ms(&control_before_gpu).min(median_ms(&control_after_gpu));
        let wall_control_median =
            median_ms(&control_before_wall).min(median_ms(&control_after_wall));
        let gpu_control_p95 =
            percentile_ms(&control_before_gpu, 0.95).min(percentile_ms(&control_after_gpu, 0.95));
        let wall_control_p95 =
            percentile_ms(&control_before_wall, 0.95).min(percentile_ms(&control_after_wall, 0.95));
        let candidate_gpu_median = median_ms(&candidate_gpu);
        let gpu_median_saving = 1.0 - candidate_gpu_median / gpu_control_median;
        let wall_median_saving = 1.0 - median_ms(&candidate_wall) / wall_control_median;
        let gpu_p95_saving = 1.0 - percentile_ms(&candidate_gpu, 0.95) / gpu_control_p95;
        let wall_p95_saving = 1.0 - percentile_ms(&candidate_wall, 0.95) / wall_control_p95;

        execute(Variant::Exact);
        let exact_after = capture();
        execute(Variant::F32Both);
        let candidate_after = capture();
        assert_eq!(
            bits(&exact_after.0),
            bits(&reference_low),
            "post-timing exact low"
        );
        assert_eq!(
            bits(&exact_after.1),
            bits(&reference_output),
            "post-timing exact output"
        );
        assert_eq!(
            bits(&candidate_after.0),
            bits(&f32_both.0),
            "post-timing candidate low"
        );
        assert_eq!(
            bits(&candidate_after.1),
            bits(&f32_both.1),
            "post-timing candidate output"
        );

        let _trace = crate::metal::kernel_trace_begin();
        execute(Variant::Exact);
        let control_trace = crate::metal::kernel_trace_take_delta();
        execute(Variant::F32Both);
        let candidate_trace = crate::metal::kernel_trace_take_delta();
        assert_eq!(control_trace.encoders, 1);
        assert_eq!(control_trace.concurrent_encoders, 0);
        assert_eq!(control_trace.dispatches, 25);
        assert_eq!(candidate_trace.encoders, 1);
        assert_eq!(candidate_trace.concurrent_encoders, 0);
        assert_eq!(candidate_trace.dispatches, 25);

        eprintln!(
            "deepseek_v4 q8_f32_packet control_before_gpu_ms={control_before_gpu:?} control_before_wall_ms={control_before_wall:?} candidate_gpu_ms={candidate_gpu:?} candidate_wall_ms={candidate_wall:?} control_after_gpu_ms={control_after_gpu:?} control_after_wall_ms={control_after_wall:?} gpu_control_drift={gpu_control_drift:.6} wall_control_drift={wall_control_drift:.6} gpu_candidate_drift={gpu_candidate_drift:.6} wall_candidate_drift={wall_candidate_drift:.6} candidate_gpu_median_ms={candidate_gpu_median:.6} gpu_median_saving={gpu_median_saving:.6} wall_median_saving={wall_median_saving:.6} gpu_p95_saving={gpu_p95_saving:.6} wall_p95_saving={wall_p95_saving:.6}"
        );

        for (label, tensor) in [
            ("Q8 F32 packet attention", &attention),
            ("Q8 F32 packet low rank", &scratch.low_rank),
            ("Q8 F32 packet output", &scratch.output),
            ("Q8 F32 packet group input", &scratch.group_input),
            ("Q8 F32 packet group output", &scratch.group_output),
        ] {
            assert_grouped_guards(label, tensor);
        }

        for (label, half, candidate) in [
            ("A-only output", half_a.3, f32_a.3),
            ("B-only output", half_b.3, f32_b.3),
            ("A+B output", half_both.3, f32_both.3),
            ("A-only low rank", half_a.2, f32_a.2),
            ("A+B low rank", half_both.2, f32_both.2),
        ] {
            let half_deficit = (1.0 - half.0).max(0.0);
            let candidate_deficit = (1.0 - candidate.0).max(0.0);
            assert!(
                candidate_deficit <= half_deficit * 0.25 + 1e-15
                    && candidate.1 <= half.1 * 0.25
                    && candidate.2 <= half.2 * 0.25,
                "{label} did not improve fourfold: half={half:?} candidate={candidate:?}"
            );
        }
        assert!(
            f32_a.2.1 <= 0.00020 && f32_a.2.2 <= 0.001,
            "F32 A low-rank gate failed: {:?}",
            f32_a.2
        );
        assert!(
            f32_both.3.0 >= 0.999_999_8 && f32_both.3.1 <= 0.00030 && f32_both.3.2 <= 0.008,
            "F32 A+B output gate failed: {:?}",
            f32_both.3
        );
        assert!(
            gpu_control_drift <= 0.05,
            "GPU control drift {gpu_control_drift}"
        );
        assert!(
            wall_control_drift <= 0.05,
            "wall control drift {wall_control_drift}"
        );
        assert!(
            gpu_candidate_drift <= 0.05,
            "GPU candidate drift {gpu_candidate_drift}"
        );
        assert!(
            wall_candidate_drift <= 0.05,
            "wall candidate drift {wall_candidate_drift}"
        );
        assert!(
            gpu_median_saving >= 0.58,
            "GPU median saving {gpu_median_saving}"
        );
        assert!(
            wall_median_saving >= 0.55,
            "wall median saving {wall_median_saving}"
        );
        assert!(gpu_p95_saving >= 0.55, "GPU p95 saving {gpu_p95_saving}");
        assert!(wall_p95_saving >= 0.50, "wall p95 saving {wall_p95_saving}");
        assert!(
            candidate_gpu_median <= 3.75,
            "candidate GPU {candidate_gpu_median} ms"
        );
    }

    #[test]
    fn packed_grouped_iq2_xs_iq3_xxs_matches_bucket_path() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const H: usize = 256;
        const F: usize = 256;
        const E: usize = MOE_EXPERT_COUNT;
        const K: usize = MOE_TOP_K;
        const CLAMP: f32 = 0.25;

        let gate_bank = grouped_test_bank(&ctx, GgmlType::IQ2_XS, H, F, E, 1);
        let up_bank = grouped_test_bank(&ctx, GgmlType::IQ2_XS, H, F, E, 3);
        let down_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, F, H, E, 5);
        let tile_buffer =
            MetalTensor::zeros_i32(&ctx, vec![PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS as u64])
                .unwrap();

        for n_tokens in [1, 12, 31, 32, 33, 64, 128, 2_048] {
            let (_expert_ids, rows, slots, schedule) = grouped_test_schedule(n_tokens);
            let grouped_plan =
                PackedGroupedExpertPlan::new(n_tokens, &schedule, Some(&tile_buffer)).unwrap();
            let input_values = (0..n_tokens * H)
                .map(|index| ((index * 37 + 5) % 251) as f32 * 0.001 - 0.125)
                .collect::<Vec<_>>();
            let input = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&input_values),
                vec![H as u64, n_tokens as u64],
                GgmlType::F32,
            )
            .unwrap();
            let bucket_rows = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&rows),
                vec![(n_tokens * K) as u64],
                GgmlType::I32,
            )
            .unwrap();
            let bucket_slots = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&slots),
                vec![(n_tokens * K) as u64],
                GgmlType::I32,
            )
            .unwrap();

            let expert_input =
                MetalTensor::zeros_f32(&ctx, vec![H as u64, n_tokens as u64]).unwrap();
            let gate = MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
            let up = MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
            let bucket_inner =
                MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
            let bucket_output =
                MetalTensor::zeros_f32(&ctx, vec![H as u64, n_tokens as u64]).unwrap();
            let control_inner =
                grouped_guarded_f32(&ctx, vec![F as u64, K as u64, n_tokens as u64], 7.0);
            let control_output =
                grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 9.0);
            let candidate_inner =
                grouped_guarded_f32(&ctx, vec![F as u64, K as u64, n_tokens as u64], 11.0);
            let candidate_output =
                grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 13.0);
            let repeat_inner =
                grouped_guarded_f32(&ctx, vec![F as u64, K as u64, n_tokens as u64], 15.0);
            let repeat_output =
                grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 17.0);

            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            for bucket in &schedule {
                let row_view = i32_slice(
                    &bucket_rows,
                    bucket.start,
                    bucket.len,
                    "grouped control rows",
                )
                .unwrap();
                let slot_view = i32_slice(
                    &bucket_slots,
                    bucket.start,
                    bucket.len,
                    "grouped control slots",
                )
                .unwrap();
                let input_view = f32_prefix(
                    &expert_input,
                    vec![H as u64, bucket.len as u64],
                    "grouped control input",
                )
                .unwrap();
                let gate_view = f32_prefix(
                    &gate,
                    vec![F as u64, bucket.len as u64],
                    "grouped control gate",
                )
                .unwrap();
                let up_view =
                    f32_prefix(&up, vec![F as u64, bucket.len as u64], "grouped control up")
                        .unwrap();
                let inner_view = f32_prefix(
                    &bucket_inner,
                    vec![F as u64, bucket.len as u64],
                    "grouped control inner",
                )
                .unwrap();
                let output_view = f32_prefix(
                    &bucket_output,
                    vec![H as u64, bucket.len as u64],
                    "grouped control output",
                )
                .unwrap();
                encode_get_rows_f32(
                    &ctx,
                    &encoder,
                    &input,
                    &row_view,
                    &input_view,
                    bucket.len,
                    H,
                )
                .unwrap();
                let gate_weight =
                    expert_weight_view(&gate_bank, H, F, bucket.expert, "grouped control gate")
                        .unwrap();
                let up_weight =
                    expert_weight_view(&up_bank, H, F, bucket.expert, "grouped control up")
                        .unwrap();
                let down_weight =
                    expert_weight_view(&down_bank, F, H, bucket.expert, "grouped control down")
                        .unwrap();
                encode_batch_projection(
                    &ctx,
                    &encoder,
                    &gate_weight,
                    &input_view,
                    &gate_view,
                    H,
                    F,
                    bucket.len,
                    "grouped control gate",
                )
                .unwrap();
                encode_batch_projection(
                    &ctx,
                    &encoder,
                    &up_weight,
                    &input_view,
                    &up_view,
                    H,
                    F,
                    bucket.len,
                    "grouped control up",
                )
                .unwrap();
                encode_ds4_clamped_swiglu(
                    &ctx,
                    &encoder,
                    &gate_view.view_subrange(0, vec![(F * bucket.len) as u64]),
                    &up_view.view_subrange(0, vec![(F * bucket.len) as u64]),
                    &inner_view.view_subrange(0, vec![(F * bucket.len) as u64]),
                    CLAMP,
                )
                .unwrap();
                encode_batch_projection(
                    &ctx,
                    &encoder,
                    &down_weight,
                    &inner_view,
                    &output_view,
                    F,
                    H,
                    bucket.len,
                    "grouped control down",
                )
                .unwrap();
                crate::metal::encode_scatter_rows_f32_unique(
                    &ctx,
                    &encoder,
                    &inner_view,
                    &slot_view,
                    &control_inner,
                    F,
                    bucket.len,
                )
                .unwrap();
                crate::metal::encode_scatter_rows_f32_unique(
                    &ctx,
                    &encoder,
                    &output_view,
                    &slot_view,
                    &control_output,
                    H,
                    bucket.len,
                )
                .unwrap();
            }
            for (inner, output) in [
                (&candidate_inner, &candidate_output),
                (&repeat_inner, &repeat_output),
            ] {
                encode_packed_grouped_swiglu_iq2_xs_f32(
                    &ctx,
                    &encoder,
                    &gate_bank,
                    &up_bank,
                    &input,
                    &bucket_slots,
                    &grouped_plan,
                    inner,
                    H,
                    F,
                    E,
                    K,
                    n_tokens,
                    CLAMP,
                )
                .unwrap();
                encode_packed_grouped_down_iq3_xxs_f32(
                    &ctx,
                    &encoder,
                    &down_bank,
                    inner,
                    &bucket_slots,
                    &grouped_plan,
                    output,
                    F,
                    H,
                    E,
                    K,
                    n_tokens,
                )
                .unwrap();
            }
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(
                command.error().is_none(),
                "N={n_tokens}: {:?}",
                command.error()
            );

            let bits = |tensor: &MetalTensor| {
                host_read_f32(tensor, "packed grouped differential")
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                bits(&candidate_inner),
                bits(&control_inner),
                "N={n_tokens} inner"
            );
            assert_eq!(
                bits(&candidate_output),
                bits(&control_output),
                "N={n_tokens} output"
            );
            assert_eq!(
                bits(&repeat_inner),
                bits(&candidate_inner),
                "N={n_tokens} repeat inner"
            );
            assert_eq!(
                bits(&repeat_output),
                bits(&candidate_output),
                "N={n_tokens} repeat output"
            );
            for (label, tensor) in [
                ("control inner", &control_inner),
                ("control output", &control_output),
                ("candidate inner", &candidate_inner),
                ("candidate output", &candidate_output),
                ("repeat inner", &repeat_inner),
                ("repeat output", &repeat_output),
            ] {
                assert_grouped_guards(label, tensor);
            }
        }
    }

    #[test]
    fn packed_grouped_iq2_xs_f32_mma16_matches_reduced_k_scalar() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const H: usize = 256;
        const F: usize = 256;
        const E: usize = MOE_EXPERT_COUNT;
        const K: usize = MOE_TOP_K;
        const CLAMP: f32 = 0.25;

        fn metrics(reference: &[f32], candidate: &[f32]) -> (f64, f64, f64) {
            assert_eq!(reference.len(), candidate.len());
            let mut dot = 0.0f64;
            let mut reference_sq = 0.0f64;
            let mut candidate_sq = 0.0f64;
            let mut error_sq = 0.0f64;
            let mut max_abs = 0.0f64;
            for (&reference, &candidate) in reference.iter().zip(candidate) {
                assert!(reference.is_finite() && candidate.is_finite());
                let reference = f64::from(reference);
                let candidate = f64::from(candidate);
                let error = candidate - reference;
                dot += reference * candidate;
                reference_sq += reference * reference;
                candidate_sq += candidate * candidate;
                error_sq += error * error;
                max_abs = max_abs.max(error.abs());
            }
            (
                dot / (reference_sq.sqrt() * candidate_sq.sqrt()),
                (error_sq / reference_sq).sqrt(),
                max_abs,
            )
        }

        let gate_bank = grouped_test_bank(&ctx, GgmlType::IQ2_XS, H, F, E, 71);
        let up_bank = grouped_test_bank(&ctx, GgmlType::IQ2_XS, H, F, E, 73);
        let grouped_tile_buffer =
            MetalTensor::zeros_i32(&ctx, vec![PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS as u64])
                .unwrap();
        let mma16_tile_buffer =
            MetalTensor::zeros_i32(&ctx, vec![PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS as u64])
                .unwrap();
        for n_tokens in [1, 12, 15, 16, 17, 31, 32, 33, 64, 128, 2_048] {
            let route_count = n_tokens * K;
            let (_expert_ids, rows, slots, schedule) = grouped_test_schedule(n_tokens);
            let grouped_plan =
                PackedGroupedExpertPlan::new(n_tokens, &schedule, Some(&grouped_tile_buffer))
                    .unwrap();
            let mma16_plan = PackedGroupedExpertPlan::new_iq2_mma16(
                n_tokens,
                &schedule,
                Some(&mma16_tile_buffer),
            )
            .unwrap();
            let input_values = (0..n_tokens * H)
                .map(|index| ((index * 37 + index / 11 + 5) % 251) as f32 * 0.001 - 0.125)
                .collect::<Vec<_>>();
            let input = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&input_values),
                vec![H as u64, n_tokens as u64],
                GgmlType::F32,
            )
            .unwrap();
            let source_rows = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&rows),
                vec![route_count as u64],
                GgmlType::I32,
            )
            .unwrap();
            let destination_slots = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&slots),
                vec![route_count as u64],
                GgmlType::I32,
            )
            .unwrap();

            let expert_input =
                MetalTensor::zeros_f32(&ctx, vec![H as u64, n_tokens as u64]).unwrap();
            let bucket_gate =
                MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
            let bucket_up = MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
            let control_gate = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 74.0);
            let control_up = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 76.0);
            let control_inner =
                grouped_guarded_f32(&ctx, vec![F as u64, K as u64, n_tokens as u64], 79.0);
            let candidate_gate =
                grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 83.0);
            let candidate_up = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 89.0);
            let candidate_inner =
                grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 97.0);
            let repeat_gate = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 101.0);
            let repeat_up = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 103.0);
            let repeat_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 107.0);

            let _trace = crate::metal::kernel_trace_begin();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            for bucket in &schedule {
                let row_view = i32_slice(
                    &source_rows,
                    bucket.start,
                    bucket.len,
                    "IQ2 MMA control rows",
                )
                .unwrap();
                let slot_view = i32_slice(
                    &destination_slots,
                    bucket.start,
                    bucket.len,
                    "IQ2 MMA control slots",
                )
                .unwrap();
                let input_view = f32_prefix(
                    &expert_input,
                    vec![H as u64, bucket.len as u64],
                    "IQ2 MMA control input",
                )
                .unwrap();
                let gate_view = f32_prefix(
                    &bucket_gate,
                    vec![F as u64, bucket.len as u64],
                    "IQ2 MMA control gate",
                )
                .unwrap();
                let up_view = f32_prefix(
                    &bucket_up,
                    vec![F as u64, bucket.len as u64],
                    "IQ2 MMA control up",
                )
                .unwrap();
                encode_get_rows_f32(
                    &ctx,
                    &encoder,
                    &input,
                    &row_view,
                    &input_view,
                    bucket.len,
                    H,
                )
                .unwrap();
                for (bank, temporary, destination, name) in [
                    (
                        &gate_bank,
                        &gate_view,
                        &control_gate,
                        "IQ2 MMA control gate",
                    ),
                    (&up_bank, &up_view, &control_up, "IQ2 MMA control up"),
                ] {
                    let weight = expert_weight_view(bank, H, F, bucket.expert, name).unwrap();
                    encode_batch_projection(
                        &ctx,
                        &encoder,
                        &weight,
                        &input_view,
                        temporary,
                        H,
                        F,
                        bucket.len,
                        name,
                    )
                    .unwrap();
                    crate::metal::encode_scatter_rows_f32_unique(
                        &ctx,
                        &encoder,
                        temporary,
                        &slot_view,
                        destination,
                        F,
                        bucket.len,
                    )
                    .unwrap();
                }
            }
            encode_packed_grouped_swiglu_iq2_xs_f32(
                &ctx,
                &encoder,
                &gate_bank,
                &up_bank,
                &input,
                &destination_slots,
                &grouped_plan,
                &control_inner,
                H,
                F,
                E,
                K,
                n_tokens,
                CLAMP,
            )
            .unwrap();
            for (gate, up, inner) in [
                (&candidate_gate, &candidate_up, &candidate_inner),
                (&repeat_gate, &repeat_up, &repeat_inner),
            ] {
                encode_packed_grouped_mapped_iq2_xs_swiglu_f32_mma16(
                    &ctx,
                    &encoder,
                    &gate_bank,
                    &up_bank,
                    &input,
                    &source_rows,
                    &destination_slots,
                    &mma16_plan,
                    gate,
                    up,
                    inner,
                    H,
                    F,
                    E,
                    K,
                    n_tokens,
                    n_tokens,
                    route_count,
                    CLAMP,
                )
                .unwrap();
            }
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(
                command.error().is_none(),
                "N={n_tokens}: {:?}",
                command.error()
            );
            let trace = crate::metal::kernel_trace_take_delta();
            assert_eq!(trace.encoders, 1);
            assert_eq!(trace.concurrent_encoders, 0);
            assert_eq!(trace.dispatches, (schedule.len() * 5 + 7) as u64);

            let read = |tensor: &MetalTensor, label| host_read_f32(tensor, label).unwrap();
            let control = read(&control_inner, "IQ2 MMA control inner");
            let control_gate_values = read(&control_gate, "IQ2 MMA control gate");
            let control_up_values = read(&control_up, "IQ2 MMA control up");
            let candidate_gate_values = read(&candidate_gate, "IQ2 MMA candidate gate");
            let candidate_up_values = read(&candidate_up, "IQ2 MMA candidate up");
            let candidate = read(&candidate_inner, "IQ2 MMA candidate inner");
            let repeat = read(&repeat_inner, "IQ2 MMA repeat inner");
            let gate_result = metrics(&control_gate_values, &candidate_gate_values);
            let up_result = metrics(&control_up_values, &candidate_up_values);
            let result = metrics(&control, &candidate);
            eprintln!(
                "deepseek_v4 iq2_mma16_differential n={n_tokens} gate={gate_result:?} up={up_result:?} inner={result:?}",
            );
            assert!(
                result.0 >= 0.999 && result.1 <= 0.05,
                "N={n_tokens} numerical gate failed: {result:?}"
            );
            assert_eq!(
                control_gate_values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                candidate_gate_values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "N={n_tokens} reduced-K gate scalar lineage"
            );
            assert_eq!(
                control_up_values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                candidate_up_values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "N={n_tokens} reduced-K up scalar lineage"
            );
            assert_eq!(
                control
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                candidate
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "N={n_tokens} reduced-K SwiGLU scalar lineage"
            );
            assert_eq!(
                candidate
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                repeat
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "N={n_tokens} repeat inner"
            );
            assert_eq!(
                candidate_gate_values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                read(&repeat_gate, "IQ2 MMA repeat gate")
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "N={n_tokens} repeat gate"
            );
            assert_eq!(
                candidate_up_values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                read(&repeat_up, "IQ2 MMA repeat up")
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "N={n_tokens} repeat up"
            );
            for (label, tensor) in [
                ("IQ2 MMA control gate", &control_gate),
                ("IQ2 MMA control up", &control_up),
                ("IQ2 MMA control inner", &control_inner),
                ("IQ2 MMA candidate gate", &candidate_gate),
                ("IQ2 MMA candidate up", &candidate_up),
                ("IQ2 MMA candidate inner", &candidate_inner),
                ("IQ2 MMA repeat gate", &repeat_gate),
                ("IQ2 MMA repeat up", &repeat_up),
                ("IQ2 MMA repeat inner", &repeat_inner),
            ] {
                assert_grouped_guards(label, tensor);
            }
        }
    }

    #[test]
    fn packed_grouped_iq2_xs_f32_mma16_traverses_production_k() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const H: usize = 4_096;
        const F: usize = 32;
        const E: usize = MOE_EXPERT_COUNT;
        const K: usize = MOE_TOP_K;
        const CLAMP: f32 = 0.25;

        fn metrics(reference: &[f32], candidate: &[f32]) -> (f64, f64, f64) {
            assert_eq!(reference.len(), candidate.len());
            let mut dot = 0.0f64;
            let mut reference_sq = 0.0f64;
            let mut candidate_sq = 0.0f64;
            let mut error_sq = 0.0f64;
            let mut max_abs = 0.0f64;
            for (&reference, &candidate) in reference.iter().zip(candidate) {
                assert!(reference.is_finite() && candidate.is_finite());
                let reference = f64::from(reference);
                let candidate = f64::from(candidate);
                let error = candidate - reference;
                dot += reference * candidate;
                reference_sq += reference * reference;
                candidate_sq += candidate * candidate;
                error_sq += error * error;
                max_abs = max_abs.max(error.abs());
            }
            (
                dot / (reference_sq.sqrt() * candidate_sq.sqrt()),
                (error_sq / reference_sq).sqrt(),
                max_abs,
            )
        }

        let gate_bank = grouped_test_bank(&ctx, GgmlType::IQ2_XS, H, F, E, 109);
        let up_bank = grouped_test_bank(&ctx, GgmlType::IQ2_XS, H, F, E, 113);
        let grouped_tile_buffer =
            MetalTensor::zeros_i32(&ctx, vec![PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS as u64])
                .unwrap();
        let mma16_tile_buffer =
            MetalTensor::zeros_i32(&ctx, vec![PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS as u64])
                .unwrap();
        for n_tokens in [1, 15, 16, 17, 128, 2_048] {
            let route_count = n_tokens * K;
            let (_expert_ids, rows, slots, schedule) = grouped_test_schedule(n_tokens);
            let grouped_plan =
                PackedGroupedExpertPlan::new(n_tokens, &schedule, Some(&grouped_tile_buffer))
                    .unwrap();
            let mma16_plan = PackedGroupedExpertPlan::new_iq2_mma16(
                n_tokens,
                &schedule,
                Some(&mma16_tile_buffer),
            )
            .unwrap();
            let input_values = (0..n_tokens * H)
                .map(|index| {
                    ((index * 41 + index / 17 + index / H * 13 + 7) % 509) as f32 * 0.0005 - 0.127
                })
                .collect::<Vec<_>>();
            let input = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&input_values),
                vec![H as u64, n_tokens as u64],
                GgmlType::F32,
            )
            .unwrap();
            let source_rows = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&rows),
                vec![route_count as u64],
                GgmlType::I32,
            )
            .unwrap();
            let destination_slots = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&slots),
                vec![route_count as u64],
                GgmlType::I32,
            )
            .unwrap();
            let expert_input =
                MetalTensor::zeros_f32(&ctx, vec![H as u64, n_tokens as u64]).unwrap();
            let bucket_gate =
                MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
            let bucket_up = MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
            let control_gate = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 117.0);
            let control_up = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 121.0);
            let control_inner =
                grouped_guarded_f32(&ctx, vec![F as u64, K as u64, n_tokens as u64], 127.0);
            let candidate_gate =
                grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 131.0);
            let candidate_up = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 137.0);
            let candidate_inner =
                grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 139.0);
            let repeat_gate = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 149.0);
            let repeat_up = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 151.0);
            let repeat_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 157.0);

            let _trace = crate::metal::kernel_trace_begin();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            for bucket in &schedule {
                let row_view = i32_slice(
                    &source_rows,
                    bucket.start,
                    bucket.len,
                    "production-K control rows",
                )
                .unwrap();
                let slot_view = i32_slice(
                    &destination_slots,
                    bucket.start,
                    bucket.len,
                    "production-K control slots",
                )
                .unwrap();
                let input_view = f32_prefix(
                    &expert_input,
                    vec![H as u64, bucket.len as u64],
                    "production-K control input",
                )
                .unwrap();
                let gate_view = f32_prefix(
                    &bucket_gate,
                    vec![F as u64, bucket.len as u64],
                    "production-K control gate",
                )
                .unwrap();
                let up_view = f32_prefix(
                    &bucket_up,
                    vec![F as u64, bucket.len as u64],
                    "production-K control up",
                )
                .unwrap();
                encode_get_rows_f32(
                    &ctx,
                    &encoder,
                    &input,
                    &row_view,
                    &input_view,
                    bucket.len,
                    H,
                )
                .unwrap();
                for (bank, temporary, destination, name) in [
                    (
                        &gate_bank,
                        &gate_view,
                        &control_gate,
                        "production-K control gate",
                    ),
                    (&up_bank, &up_view, &control_up, "production-K control up"),
                ] {
                    let weight = expert_weight_view(bank, H, F, bucket.expert, name).unwrap();
                    encode_batch_projection(
                        &ctx,
                        &encoder,
                        &weight,
                        &input_view,
                        temporary,
                        H,
                        F,
                        bucket.len,
                        name,
                    )
                    .unwrap();
                    crate::metal::encode_scatter_rows_f32_unique(
                        &ctx,
                        &encoder,
                        temporary,
                        &slot_view,
                        destination,
                        F,
                        bucket.len,
                    )
                    .unwrap();
                }
            }
            encode_packed_grouped_swiglu_iq2_xs_f32(
                &ctx,
                &encoder,
                &gate_bank,
                &up_bank,
                &input,
                &destination_slots,
                &grouped_plan,
                &control_inner,
                H,
                F,
                E,
                K,
                n_tokens,
                CLAMP,
            )
            .unwrap();
            for (gate, up, inner) in [
                (&candidate_gate, &candidate_up, &candidate_inner),
                (&repeat_gate, &repeat_up, &repeat_inner),
            ] {
                encode_packed_grouped_mapped_iq2_xs_swiglu_f32_mma16(
                    &ctx,
                    &encoder,
                    &gate_bank,
                    &up_bank,
                    &input,
                    &source_rows,
                    &destination_slots,
                    &mma16_plan,
                    gate,
                    up,
                    inner,
                    H,
                    F,
                    E,
                    K,
                    n_tokens,
                    n_tokens,
                    route_count,
                    CLAMP,
                )
                .unwrap();
            }
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(
                command.error().is_none(),
                "N={n_tokens}: {:?}",
                command.error()
            );
            let trace = crate::metal::kernel_trace_take_delta();
            assert_eq!(trace.encoders, 1);
            assert_eq!(trace.concurrent_encoders, 0);
            assert_eq!(trace.dispatches, (schedule.len() * 5 + 7) as u64);

            let control_gate_values =
                host_read_f32(&control_gate, "production-K scalar gate").unwrap();
            let control_up_values = host_read_f32(&control_up, "production-K scalar up").unwrap();
            let candidate_gate_values =
                host_read_f32(&candidate_gate, "production-K BM16 gate").unwrap();
            let candidate_up_values = host_read_f32(&candidate_up, "production-K BM16 up").unwrap();
            let control = host_read_f32(&control_inner, "production-K scalar inner").unwrap();
            let candidate = host_read_f32(&candidate_inner, "production-K BM16 inner").unwrap();
            let repeat = host_read_f32(&repeat_inner, "production-K repeat inner").unwrap();
            let result = metrics(&control, &candidate);
            eprintln!(
                "deepseek_v4 iq2_mma16_production_k n={n_tokens} cosine={:.9} rel_rms={:.9} max_abs={:.9}",
                result.0, result.1, result.2,
            );
            assert!(
                result.0 >= 0.999_999 && result.1 <= 0.001,
                "N={n_tokens} production-K numerical gate failed: {result:?}"
            );
            assert_eq!(
                control_gate_values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                candidate_gate_values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "N={n_tokens} production-K gate scalar lineage"
            );
            assert_eq!(
                control_up_values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                candidate_up_values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "N={n_tokens} production-K up scalar lineage"
            );
            assert_eq!(
                control
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                candidate
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "N={n_tokens} production-K SwiGLU scalar lineage"
            );
            assert_eq!(
                candidate
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                repeat
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "N={n_tokens} production-K repeat inner"
            );
            assert_eq!(
                candidate_gate_values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                host_read_f32(&repeat_gate, "production-K repeat gate")
                    .unwrap()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "N={n_tokens} production-K repeat gate"
            );
            assert_eq!(
                candidate_up_values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                host_read_f32(&repeat_up, "production-K repeat up")
                    .unwrap()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "N={n_tokens} production-K repeat up"
            );
            for (label, tensor) in [
                ("production-K control gate", &control_gate),
                ("production-K control up", &control_up),
                ("production-K control inner", &control_inner),
                ("production-K candidate gate", &candidate_gate),
                ("production-K candidate up", &candidate_up),
                ("production-K candidate inner", &candidate_inner),
                ("production-K repeat gate", &repeat_gate),
                ("production-K repeat up", &repeat_up),
                ("production-K repeat inner", &repeat_inner),
            ] {
                assert_grouped_guards(label, tensor);
            }
        }
    }

    #[test]
    fn packed_gpu_route_compaction_feeds_all_iq3_at_n2048() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        assert!(packed_grouped_iq3_candidate_supported(&ctx));
        const H: usize = 512;
        const F: usize = 256;
        const E: usize = MOE_EXPERT_COUNT;
        const K: usize = MOE_TOP_K;
        const N: usize = DEEPSEEK_V4_PREFILL_MAX_TOKENS;
        const CLAMP: f32 = 0.25;

        let fixture = PackedRouteFixture::new(&ctx);
        let policy = PackedExpertPolicy::GroupedIq2XsIq3XxsMma16QualifiedChunk.with_iq3_target();
        assert!(packed_gpu_compact_expert_layer_qualified(
            &ctx,
            policy,
            N,
            true,
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_XXS,
        ));
        assert!(!packed_gpu_compact_expert_layer_qualified(
            &ctx,
            policy,
            N,
            false,
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_XXS,
        ));
        assert!(packed_gpu_compact_expert_layer_qualified(
            &ctx,
            policy,
            N,
            false,
            GgmlType::IQ2_XS,
            GgmlType::IQ2_XS,
            GgmlType::IQ3_XXS,
        ));
        assert!(!packed_gpu_compact_expert_layer_qualified(
            &ctx,
            policy,
            N,
            true,
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_XXS,
            GgmlType::MXFP4,
        ));

        let gate_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, H, F, E, 163);
        let up_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, H, F, E, 167);
        let down_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, F, H, E, 173);
        let route_count = N * K;
        let input_values = (0..N * H)
            .map(|index| ((index * 43 + index / H * 17 + 11) % 509) as f32 * 0.0005 - 0.127)
            .collect::<Vec<_>>();
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&input_values),
            vec![H as u64, N as u64],
            GgmlType::F32,
        )
        .unwrap();
        let device_plan = PackedGroupedExpertPlan::from_device(
            &fixture.scratch.compact_tiles32,
            PACKED_GROUPED_EXPERT_MAX_TILES,
        )
        .unwrap();
        let device_output = grouped_guarded_f32(&ctx, vec![H as u64, K as u64, N as u64], 181.0);
        let device_output_flat = device_output.view_subrange(0, vec![H as u64, route_count as u64]);
        let device_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 183.0);
        let generation = fixture.generations.take().unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        fixture
            .scratch
            .encode_learned(&ctx, &encoder, &fixture.bias, N, N, generation)
            .unwrap();
        fixture
            .scratch
            .encode_compact(&ctx, &encoder, N, generation)
            .unwrap();
        encode_packed_grouped_all_iq3(
            &ctx,
            &encoder,
            &gate_bank,
            &up_bank,
            &down_bank,
            &input,
            &fixture.scratch.compact_rows,
            &fixture.scratch.compact_slots,
            &device_plan,
            &device_output_flat,
            &device_inner,
            H,
            F,
            E,
            K,
            N,
            CLAMP,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "{:?}", command.error());

        let capture = fixture.scratch.capture_compact(N);
        assert_packed_compact_capture(
            &fixture,
            PackedRouteMicroproofSource::Learned,
            N,
            generation.get(),
            &capture,
        );
        let (_, _, _, schedule) = expected_compact_schedule(&capture.expert_ids, N);
        let host_descriptors =
            MetalTensor::zeros_i32(&ctx, vec![PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS as u64])
                .unwrap();
        let host_plan =
            PackedGroupedExpertPlan::new(N, &schedule, Some(&host_descriptors)).unwrap();
        let host_output = grouped_guarded_f32(&ctx, vec![H as u64, K as u64, N as u64], 191.0);
        let host_output_flat = host_output.view_subrange(0, vec![H as u64, route_count as u64]);
        let host_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 193.0);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_packed_grouped_all_iq3(
            &ctx,
            &encoder,
            &gate_bank,
            &up_bank,
            &down_bank,
            &input,
            &fixture.scratch.compact_rows,
            &fixture.scratch.compact_slots,
            &host_plan,
            &host_output_flat,
            &host_inner,
            H,
            F,
            E,
            K,
            N,
            CLAMP,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "{:?}", command.error());

        let bits = |tensor: &MetalTensor, label| {
            host_read_f32(tensor, label)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bits(&device_inner, "all-IQ3 device-plan inner"),
            bits(&host_inner, "all-IQ3 host-plan inner")
        );
        assert_eq!(
            bits(&device_output, "all-IQ3 device-plan output"),
            bits(&host_output, "all-IQ3 host-plan output")
        );
        for (label, tensor) in [
            ("all-IQ3 host-plan inner", &host_inner),
            ("all-IQ3 host-plan output", &host_output),
            ("all-IQ3 device-plan inner", &device_inner),
            ("all-IQ3 device-plan output", &device_output),
        ] {
            assert_grouped_guards(label, tensor);
        }
    }

    #[test]
    fn packed_grouped_all_iq3_matches_bucket_path_and_arena_contract() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        assert!(
            packed_grouped_iq3_fused_candidate_supported(&ctx),
            "fused all-IQ3 pipeline is not qualified"
        );
        const H: usize = 512;
        const F: usize = 256;
        const E: usize = MOE_EXPERT_COUNT;
        const K: usize = MOE_TOP_K;
        const CLAMP: f32 = 0.25;

        fn submit(
            ctx: &MetalContext,
            label: &str,
            encode: impl FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
        ) {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let result = encode(&encoder);
            encoder.end();
            result.unwrap();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none(), "{label}: {:?}", command.error());
        }

        let gate_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, H, F, E, 29);
        let up_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, H, F, E, 31);
        let down_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, F, H, E, 37);

        for n_tokens in [1, 12, 31, 32, 33, 64, 128] {
            let route_count = n_tokens * K;
            let (_expert_ids, rows, slots, schedule) = grouped_test_schedule(n_tokens);
            let input_values = (0..n_tokens * H)
                .map(|index| ((index * 43 + 11) % 263) as f32 * 0.001 - 0.125)
                .collect::<Vec<_>>();
            let input = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&input_values),
                vec![H as u64, n_tokens as u64],
                GgmlType::F32,
            )
            .unwrap();
            let bucket_rows = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&rows),
                vec![route_count as u64],
                GgmlType::I32,
            )
            .unwrap();
            let bucket_slots = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&slots),
                vec![route_count as u64],
                GgmlType::I32,
            )
            .unwrap();

            let expert_input =
                MetalTensor::zeros_f32(&ctx, vec![H as u64, n_tokens as u64]).unwrap();
            let gate = MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
            let up = MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
            let bucket_inner =
                MetalTensor::zeros_f32(&ctx, vec![F as u64, n_tokens as u64]).unwrap();
            let bucket_output =
                MetalTensor::zeros_f32(&ctx, vec![H as u64, n_tokens as u64]).unwrap();
            let control_gate =
                grouped_guarded_f32(&ctx, vec![F as u64, K as u64, n_tokens as u64], 3.0);
            let control_up =
                grouped_guarded_f32(&ctx, vec![F as u64, K as u64, n_tokens as u64], 5.0);
            let control_inner =
                grouped_guarded_f32(&ctx, vec![F as u64, K as u64, n_tokens as u64], 7.0);
            let control_output =
                grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 9.0);

            submit(&ctx, "all-IQ3 bucket control", |encoder| {
                for bucket in &schedule {
                    let row_view = i32_slice(
                        &bucket_rows,
                        bucket.start,
                        bucket.len,
                        "all-IQ3 control rows",
                    )?;
                    let slot_view = i32_slice(
                        &bucket_slots,
                        bucket.start,
                        bucket.len,
                        "all-IQ3 control slots",
                    )?;
                    let input_view = f32_prefix(
                        &expert_input,
                        vec![H as u64, bucket.len as u64],
                        "all-IQ3 control input",
                    )?;
                    let gate_view = f32_prefix(
                        &gate,
                        vec![F as u64, bucket.len as u64],
                        "all-IQ3 control gate",
                    )?;
                    let up_view =
                        f32_prefix(&up, vec![F as u64, bucket.len as u64], "all-IQ3 control up")?;
                    let inner_view = f32_prefix(
                        &bucket_inner,
                        vec![F as u64, bucket.len as u64],
                        "all-IQ3 control inner",
                    )?;
                    let output_view = f32_prefix(
                        &bucket_output,
                        vec![H as u64, bucket.len as u64],
                        "all-IQ3 control output",
                    )?;
                    encode_get_rows_f32(
                        &ctx,
                        encoder,
                        &input,
                        &row_view,
                        &input_view,
                        bucket.len,
                        H,
                    )?;
                    let gate_weight = expert_weight_view(
                        &gate_bank,
                        H,
                        F,
                        bucket.expert,
                        "all-IQ3 control gate",
                    )?;
                    let up_weight =
                        expert_weight_view(&up_bank, H, F, bucket.expert, "all-IQ3 control up")?;
                    let down_weight = expert_weight_view(
                        &down_bank,
                        F,
                        H,
                        bucket.expert,
                        "all-IQ3 control down",
                    )?;
                    encode_batch_projection(
                        &ctx,
                        encoder,
                        &gate_weight,
                        &input_view,
                        &gate_view,
                        H,
                        F,
                        bucket.len,
                        "all-IQ3 control gate",
                    )?;
                    encode_batch_projection(
                        &ctx,
                        encoder,
                        &up_weight,
                        &input_view,
                        &up_view,
                        H,
                        F,
                        bucket.len,
                        "all-IQ3 control up",
                    )?;
                    let inner_len = F * bucket.len;
                    encode_ds4_clamped_swiglu(
                        &ctx,
                        encoder,
                        &gate_view.view_subrange(0, vec![inner_len as u64]),
                        &up_view.view_subrange(0, vec![inner_len as u64]),
                        &inner_view.view_subrange(0, vec![inner_len as u64]),
                        CLAMP,
                    )?;
                    encode_batch_projection(
                        &ctx,
                        encoder,
                        &down_weight,
                        &inner_view,
                        &output_view,
                        F,
                        H,
                        bucket.len,
                        "all-IQ3 control down",
                    )?;
                    for (source, destination, width) in [
                        (&gate_view, &control_gate, F),
                        (&up_view, &control_up, F),
                        (&inner_view, &control_inner, F),
                        (&output_view, &control_output, H),
                    ] {
                        crate::metal::encode_scatter_rows_f32_unique(
                            &ctx,
                            encoder,
                            source,
                            &slot_view,
                            destination,
                            width,
                            bucket.len,
                        )?;
                    }
                }
                Ok(())
            });

            if n_tokens == 128 {
                let gate_values = host_read_f32(&control_gate, "all-IQ3 clamp gate").unwrap();
                let up_values = host_read_f32(&control_up, "all-IQ3 clamp up").unwrap();
                let gate_max = gate_values
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let up_min = up_values.iter().copied().fold(f32::INFINITY, f32::min);
                let up_max = up_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                assert!(
                    gate_max > CLAMP,
                    "synthetic gate never crosses upper clamp: {gate_max}"
                );
                assert!(
                    up_min < -CLAMP && up_max > CLAMP,
                    "synthetic up misses clamp sides: [{up_min}, {up_max}]"
                );
            }

            let bits = |tensor: &MetalTensor| {
                host_read_f32(tensor, "all-IQ3 grouped differential")
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
            };
            let candidate_arena =
                grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 11.0);
            let candidate_output =
                candidate_arena.view_subrange(0, vec![H as u64, route_count as u64]);
            let (candidate_gate, candidate_up) =
                packed_grouped_gate_up_views(&candidate_output, H, F, route_count).unwrap();
            let candidate_inner =
                grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 13.0);
            let up_poison = bits(&candidate_up);

            submit(&ctx, "all-IQ3 mapped gate", |encoder| {
                encode_packed_grouped_mapped_iq3_xxs_f32(
                    &ctx,
                    encoder,
                    &gate_bank,
                    &input,
                    &bucket_rows,
                    &bucket_slots,
                    &schedule,
                    &candidate_gate,
                    H,
                    F,
                    E,
                    K,
                    n_tokens,
                    n_tokens,
                    route_count,
                )
            });
            assert_eq!(
                bits(&candidate_gate),
                bits(&control_gate),
                "N={n_tokens} gate"
            );
            assert_eq!(bits(&candidate_up), up_poison, "N={n_tokens} up poison");
            let gate_after_gate = bits(&candidate_gate);

            submit(&ctx, "all-IQ3 mapped up", |encoder| {
                encode_packed_grouped_mapped_iq3_xxs_f32(
                    &ctx,
                    encoder,
                    &up_bank,
                    &input,
                    &bucket_rows,
                    &bucket_slots,
                    &schedule,
                    &candidate_up,
                    H,
                    F,
                    E,
                    K,
                    n_tokens,
                    n_tokens,
                    route_count,
                )
            });
            assert_eq!(
                bits(&candidate_gate),
                gate_after_gate,
                "N={n_tokens} gate stable"
            );
            assert_eq!(bits(&candidate_up), bits(&control_up), "N={n_tokens} up");
            let up_after_up = bits(&candidate_up);

            submit(&ctx, "all-IQ3 mapped SwiGLU", |encoder| {
                encode_ds4_clamped_swiglu(
                    &ctx,
                    encoder,
                    &candidate_gate.view_subrange(0, vec![(F * route_count) as u64]),
                    &candidate_up.view_subrange(0, vec![(F * route_count) as u64]),
                    &candidate_inner.view_subrange(0, vec![(F * route_count) as u64]),
                    CLAMP,
                )
            });
            assert_eq!(
                bits(&candidate_gate),
                gate_after_gate,
                "N={n_tokens} gate after SwiGLU"
            );
            assert_eq!(
                bits(&candidate_up),
                up_after_up,
                "N={n_tokens} up after SwiGLU"
            );
            assert_eq!(
                bits(&candidate_inner),
                bits(&control_inner),
                "N={n_tokens} inner"
            );
            let inner_after_swiglu = bits(&candidate_inner);

            submit(&ctx, "all-IQ3 mapped down", |encoder| {
                encode_packed_grouped_mapped_iq3_xxs_f32(
                    &ctx,
                    encoder,
                    &down_bank,
                    &candidate_inner,
                    &bucket_slots,
                    &bucket_slots,
                    &schedule,
                    &candidate_output,
                    F,
                    H,
                    E,
                    K,
                    n_tokens,
                    route_count,
                    route_count,
                )
            });
            assert_eq!(
                bits(&candidate_arena),
                bits(&control_output),
                "N={n_tokens} output"
            );
            assert_eq!(
                bits(&candidate_inner),
                inner_after_swiglu,
                "N={n_tokens} inner after down"
            );

            let fused_arena =
                grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 23.0);
            let fused_output = fused_arena.view_subrange(0, vec![H as u64, route_count as u64]);
            let (fused_gate, fused_up) =
                packed_grouped_gate_up_views(&fused_output, H, F, route_count).unwrap();
            let fused_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 29.0);
            submit(&ctx, "all-IQ3 fused gate/up/SwiGLU", |encoder| {
                encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
                    &ctx,
                    encoder,
                    &gate_bank,
                    &up_bank,
                    &input,
                    &bucket_rows,
                    &bucket_slots,
                    &schedule,
                    &fused_output,
                    &fused_inner,
                    H,
                    F,
                    E,
                    K,
                    n_tokens,
                    n_tokens,
                    route_count,
                    CLAMP,
                )
            });
            assert_eq!(
                bits(&fused_gate),
                bits(&control_gate),
                "N={n_tokens} fused gate"
            );
            assert_eq!(bits(&fused_up), bits(&control_up), "N={n_tokens} fused up");
            assert_eq!(
                bits(&fused_inner),
                bits(&control_inner),
                "N={n_tokens} fused inner"
            );
            let fused_gate_after = bits(&fused_gate);
            let fused_up_after = bits(&fused_up);
            let fused_inner_after = bits(&fused_inner);
            submit(&ctx, "all-IQ3 fused mapped down", |encoder| {
                encode_packed_grouped_mapped_iq3_xxs_f32(
                    &ctx,
                    encoder,
                    &down_bank,
                    &fused_inner,
                    &bucket_slots,
                    &bucket_slots,
                    &schedule,
                    &fused_output,
                    F,
                    H,
                    E,
                    K,
                    n_tokens,
                    route_count,
                    route_count,
                )
            });
            assert_eq!(
                bits(&fused_arena),
                bits(&control_output),
                "N={n_tokens} fused output"
            );
            assert_eq!(
                bits(&fused_inner),
                fused_inner_after,
                "N={n_tokens} fused inner stable"
            );

            let fused_repeat_arena =
                grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 31.0);
            let fused_repeat_output =
                fused_repeat_arena.view_subrange(0, vec![H as u64, route_count as u64]);
            let (fused_repeat_gate, fused_repeat_up) =
                packed_grouped_gate_up_views(&fused_repeat_output, H, F, route_count).unwrap();
            let fused_repeat_inner =
                grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 37.0);
            submit(&ctx, "all-IQ3 fused repeat gate/up/SwiGLU", |encoder| {
                encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
                    &ctx,
                    encoder,
                    &gate_bank,
                    &up_bank,
                    &input,
                    &bucket_rows,
                    &bucket_slots,
                    &schedule,
                    &fused_repeat_output,
                    &fused_repeat_inner,
                    H,
                    F,
                    E,
                    K,
                    n_tokens,
                    n_tokens,
                    route_count,
                    CLAMP,
                )
            });
            assert_eq!(
                bits(&fused_repeat_gate),
                fused_gate_after,
                "N={n_tokens} fused repeat gate"
            );
            assert_eq!(
                bits(&fused_repeat_up),
                fused_up_after,
                "N={n_tokens} fused repeat up"
            );
            assert_eq!(
                bits(&fused_repeat_inner),
                fused_inner_after,
                "N={n_tokens} fused repeat inner"
            );
            submit(&ctx, "all-IQ3 fused repeat down", |encoder| {
                encode_packed_grouped_mapped_iq3_xxs_f32(
                    &ctx,
                    encoder,
                    &down_bank,
                    &fused_repeat_inner,
                    &bucket_slots,
                    &bucket_slots,
                    &schedule,
                    &fused_repeat_output,
                    F,
                    H,
                    E,
                    K,
                    n_tokens,
                    route_count,
                    route_count,
                )
            });
            assert_eq!(
                bits(&fused_repeat_inner),
                fused_inner_after,
                "N={n_tokens} fused repeat inner"
            );
            assert_eq!(
                bits(&fused_repeat_arena),
                bits(&control_output),
                "N={n_tokens} fused repeat output"
            );

            let fused_chain_arena =
                grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 41.0);
            let fused_chain_output =
                fused_chain_arena.view_subrange(0, vec![H as u64, route_count as u64]);
            let fused_chain_inner =
                grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 43.0);
            submit(&ctx, "all-IQ3 fused one-command chain", |encoder| {
                encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
                    &ctx,
                    encoder,
                    &gate_bank,
                    &up_bank,
                    &input,
                    &bucket_rows,
                    &bucket_slots,
                    &schedule,
                    &fused_chain_output,
                    &fused_chain_inner,
                    H,
                    F,
                    E,
                    K,
                    n_tokens,
                    n_tokens,
                    route_count,
                    CLAMP,
                )?;
                encode_packed_grouped_mapped_iq3_xxs_f32(
                    &ctx,
                    encoder,
                    &down_bank,
                    &fused_chain_inner,
                    &bucket_slots,
                    &bucket_slots,
                    &schedule,
                    &fused_chain_output,
                    F,
                    H,
                    E,
                    K,
                    n_tokens,
                    route_count,
                    route_count,
                )
            });
            assert_eq!(
                bits(&fused_chain_arena),
                bits(&control_output),
                "N={n_tokens} fused one-command output"
            );
            assert_eq!(
                bits(&fused_chain_inner),
                bits(&control_inner),
                "N={n_tokens} fused one-command inner"
            );

            let repeat_arena =
                grouped_guarded_f32(&ctx, vec![H as u64, K as u64, n_tokens as u64], 17.0);
            let repeat_output = repeat_arena.view_subrange(0, vec![H as u64, route_count as u64]);
            let (repeat_gate, repeat_up) =
                packed_grouped_gate_up_views(&repeat_output, H, F, route_count).unwrap();
            let repeat_inner = grouped_guarded_f32(&ctx, vec![F as u64, route_count as u64], 19.0);
            submit(&ctx, "all-IQ3 one-command chain", |encoder| {
                encode_packed_grouped_mapped_iq3_xxs_f32(
                    &ctx,
                    encoder,
                    &gate_bank,
                    &input,
                    &bucket_rows,
                    &bucket_slots,
                    &schedule,
                    &repeat_gate,
                    H,
                    F,
                    E,
                    K,
                    n_tokens,
                    n_tokens,
                    route_count,
                )?;
                encode_packed_grouped_mapped_iq3_xxs_f32(
                    &ctx,
                    encoder,
                    &up_bank,
                    &input,
                    &bucket_rows,
                    &bucket_slots,
                    &schedule,
                    &repeat_up,
                    H,
                    F,
                    E,
                    K,
                    n_tokens,
                    n_tokens,
                    route_count,
                )?;
                encode_ds4_clamped_swiglu(
                    &ctx,
                    encoder,
                    &repeat_gate.view_subrange(0, vec![(F * route_count) as u64]),
                    &repeat_up.view_subrange(0, vec![(F * route_count) as u64]),
                    &repeat_inner.view_subrange(0, vec![(F * route_count) as u64]),
                    CLAMP,
                )?;
                encode_packed_grouped_mapped_iq3_xxs_f32(
                    &ctx,
                    encoder,
                    &down_bank,
                    &repeat_inner,
                    &bucket_slots,
                    &bucket_slots,
                    &schedule,
                    &repeat_output,
                    F,
                    H,
                    E,
                    K,
                    n_tokens,
                    route_count,
                    route_count,
                )
            });
            assert_eq!(
                bits(&repeat_arena),
                bits(&control_output),
                "N={n_tokens} repeat output"
            );
            assert_eq!(
                bits(&repeat_inner),
                bits(&control_inner),
                "N={n_tokens} repeat inner"
            );

            for (label, tensor) in [
                ("all-IQ3 control gate", &control_gate),
                ("all-IQ3 control up", &control_up),
                ("all-IQ3 control inner", &control_inner),
                ("all-IQ3 control output", &control_output),
                ("all-IQ3 candidate arena", &candidate_arena),
                ("all-IQ3 candidate inner", &candidate_inner),
                ("all-IQ3 fused arena", &fused_arena),
                ("all-IQ3 fused inner", &fused_inner),
                ("all-IQ3 fused repeat arena", &fused_repeat_arena),
                ("all-IQ3 fused repeat inner", &fused_repeat_inner),
                ("all-IQ3 fused chain arena", &fused_chain_arena),
                ("all-IQ3 fused chain inner", &fused_chain_inner),
                ("all-IQ3 repeat arena", &repeat_arena),
                ("all-IQ3 repeat inner", &repeat_inner),
            ] {
                assert_grouped_guards(label, tensor);
            }
        }
    }

    #[test]
    fn packed_grouped_fused_all_iq3_rejects_invalid_contracts_before_dispatch() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const H: usize = 512;
        const F: usize = 256;
        const E: usize = MOE_EXPERT_COUNT;
        const K: usize = MOE_TOP_K;
        const N: usize = 1;

        fn reject(
            ctx: &MetalContext,
            encode: impl FnOnce(&KernelEncoder) -> Result<(), DeepSeekV4MetalError>,
        ) -> String {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let result = encode(&encoder);
            encoder.end();
            result.unwrap_err().to_string()
        }

        let gate_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, H, F, E, 47);
        let up_bank = grouped_test_bank(&ctx, GgmlType::IQ3_XXS, H, F, E, 53);
        let wrong_dtype_bank = grouped_test_bank(&ctx, GgmlType::IQ3_S, H, F, E, 59);
        let input = MetalTensor::zeros_f32(&ctx, vec![H as u64, N as u64]).unwrap();
        let (_expert_ids, rows, slots, schedule) = grouped_test_schedule(N);
        let source_rows = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&rows),
            vec![K as u64],
            GgmlType::I32,
        )
        .unwrap();
        let destination_slots = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&slots),
            vec![K as u64],
            GgmlType::I32,
        )
        .unwrap();
        let malformed_rows = MetalTensor::zeros_f32(&ctx, vec![K as u64]).unwrap();
        let arena = MetalTensor::zeros_f32(&ctx, vec![H as u64, K as u64]).unwrap();
        let inner = MetalTensor::zeros_f32(&ctx, vec![F as u64, K as u64]).unwrap();

        let error = reject(&ctx, |encoder| {
            encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
                &ctx,
                encoder,
                &gate_bank,
                &wrong_dtype_bank,
                &input,
                &source_rows,
                &destination_slots,
                &schedule,
                &arena,
                &inner,
                H,
                F,
                E,
                K,
                N,
                N,
                K,
                0.25,
            )
        });
        assert!(error.contains("invalid geometry or storage"), "{error}");

        let error = reject(&ctx, |encoder| {
            encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
                &ctx,
                encoder,
                &gate_bank,
                &up_bank,
                &input,
                &malformed_rows,
                &destination_slots,
                &schedule,
                &arena,
                &inner,
                H,
                F,
                E,
                K,
                N,
                N,
                K,
                0.25,
            )
        });
        assert!(error.contains("source rows"), "{error}");

        let mut malformed_schedule = schedule
            .iter()
            .map(|bucket| ExpertBucket {
                expert: bucket.expert,
                start: bucket.start,
                len: bucket.len,
            })
            .collect::<Vec<_>>();
        malformed_schedule[0].start = 1;
        let error = reject(&ctx, |encoder| {
            encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
                &ctx,
                encoder,
                &gate_bank,
                &up_bank,
                &input,
                &source_rows,
                &destination_slots,
                &malformed_schedule,
                &arena,
                &inner,
                H,
                F,
                E,
                K,
                N,
                N,
                K,
                0.25,
            )
        });
        assert!(error.contains("invalid bucket geometry"), "{error}");

        let overlapping_inner = arena.view_subrange(0, vec![F as u64, K as u64]);
        let error = reject(&ctx, |encoder| {
            encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
                &ctx,
                encoder,
                &gate_bank,
                &up_bank,
                &input,
                &source_rows,
                &destination_slots,
                &schedule,
                &arena,
                &overlapping_inner,
                H,
                F,
                E,
                K,
                N,
                N,
                K,
                0.25,
            )
        });
        assert!(error.contains("inner overlaps gate/up arena"), "{error}");

        let error = reject(&ctx, |encoder| {
            encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
                &ctx,
                encoder,
                &gate_bank,
                &up_bank,
                &input,
                &source_rows,
                &destination_slots,
                &schedule,
                &arena,
                &inner,
                H,
                F,
                E,
                K,
                N,
                N,
                K,
                0.0,
            )
        });
        assert!(error.contains("invalid geometry or storage"), "{error}");
    }

    #[test]
    #[ignore = "sealed KILL; do not rerun without material implementation or device drift"]
    fn profile_packed_grouped_fused_all_iq3_production_shape() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        assert_eq!(
            ctx.device.name().to_string(),
            PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE
        );
        assert!(
            packed_grouped_iq3_fused_candidate_supported(&ctx),
            "fused all-IQ3 pipeline is not qualified"
        );

        const H: usize = 4_096;
        const F: usize = 2_048;
        const E: usize = MOE_EXPERT_COUNT;
        const K: usize = MOE_TOP_K;
        const N: usize = 128;
        const ROUTES: usize = N * K;
        const CLAMP: f32 = 7.0;
        const SAMPLES: usize = 24;

        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        enum Arm {
            Control,
            Candidate,
        }

        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        enum ScheduleShape {
            Hot,
            Sparse,
        }

        #[derive(Debug)]
        struct GateResult {
            schedule: &'static str,
            gpu_control_drift: f64,
            wall_control_drift: f64,
            gpu_candidate_drift: f64,
            wall_candidate_drift: f64,
            gpu_median_saving: f64,
            wall_median_saving: f64,
            gpu_p95_saving: f64,
            wall_p95_saving: f64,
        }

        fn timing_bank(ctx: &MetalContext, n_in: usize, n_out: usize, fill: u8) -> MetalTensor {
            let (block_elements, block_bytes) = ggml_type_layout(GgmlType::IQ3_XXS).unwrap();
            let elements = n_in * n_out * E;
            assert!(elements.is_multiple_of(block_elements as usize));
            let bytes = elements / block_elements as usize * block_bytes as usize;
            let buffer = ctx.buffer_uninit(bytes).unwrap();
            unsafe {
                std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), fill, bytes);
            }
            MetalTensor {
                buffer,
                offset: 0,
                shape: vec![n_in as u64, n_out as u64, E as u64],
                dtype: GgmlType::IQ3_XXS,
                provenance: MetalTensorProvenance::OwnedWritable,
            }
        }

        fn timing_schedule(shape: ScheduleShape) -> (Vec<i32>, Vec<i32>, Vec<ExpertBucket>) {
            let mut assignments = (0..E)
                .map(|_| Vec::<(usize, usize)>::new())
                .collect::<Vec<_>>();
            for token in 0..N {
                for slot in 0..K {
                    let route_slot = token * K + slot;
                    let expert = match shape {
                        ScheduleShape::Hot => slot,
                        ScheduleShape::Sparse => route_slot % E,
                    };
                    assignments[expert].push((token, route_slot));
                }
            }
            let mut rows = Vec::with_capacity(ROUTES);
            let mut slots = Vec::with_capacity(ROUTES);
            let mut schedule = Vec::new();
            for (expert, assignments) in assignments.into_iter().enumerate() {
                if assignments.is_empty() {
                    continue;
                }
                let start = rows.len();
                for (row, slot) in assignments {
                    rows.push(row as i32);
                    slots.push(slot as i32);
                }
                schedule.push(ExpertBucket {
                    expert,
                    start,
                    len: rows.len() - start,
                });
            }
            assert_eq!(rows.len(), ROUTES);
            assert_eq!(slots.len(), ROUTES);
            assert!(packed_grouped_expert_tiles(N, &schedule).is_ok());
            (rows, slots, schedule)
        }

        let gate_bank = timing_bank(&ctx, H, F, 0x20);
        let up_bank = timing_bank(&ctx, H, F, 0x24);
        let down_bank = timing_bank(&ctx, F, H, 0x28);
        let input_values = (0..H * N)
            .map(|index| ((index * 43 + index / 17 + 11) % 521) as f32 * 0.0005 - 0.13)
            .collect::<Vec<_>>();
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&input_values),
            vec![H as u64, N as u64],
            GgmlType::F32,
        )
        .unwrap();
        eprintln!(
            "deepseek_v4 fused_all_iq3_fixture allocated_bytes={} bank_bytes={}",
            ctx.current_allocated_size(),
            gate_bank.n_bytes() + up_bank.n_bytes() + down_bank.n_bytes(),
        );

        let mut results = Vec::new();
        for schedule_shape in [ScheduleShape::Hot, ScheduleShape::Sparse] {
            let schedule_name = match schedule_shape {
                ScheduleShape::Hot => "hot",
                ScheduleShape::Sparse => "sparse",
            };
            let (rows, slots, schedule) = timing_schedule(schedule_shape);
            let source_rows = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&rows),
                vec![ROUTES as u64],
                GgmlType::I32,
            )
            .unwrap();
            let destination_slots = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&slots),
                vec![ROUTES as u64],
                GgmlType::I32,
            )
            .unwrap();
            let control_arena = grouped_guarded_f32(&ctx, vec![H as u64, ROUTES as u64], 47.0);
            let control_output = control_arena.view_subrange(0, vec![H as u64, ROUTES as u64]);
            let (control_gate, control_up) =
                packed_grouped_gate_up_views(&control_output, H, F, ROUTES).unwrap();
            let control_inner = grouped_guarded_f32(&ctx, vec![F as u64, ROUTES as u64], 53.0);
            let candidate_arena = grouped_guarded_f32(&ctx, vec![H as u64, ROUTES as u64], 59.0);
            let candidate_output = candidate_arena.view_subrange(0, vec![H as u64, ROUTES as u64]);
            let candidate_inner = grouped_guarded_f32(&ctx, vec![F as u64, ROUTES as u64], 61.0);

            let sample = |arm: Arm| -> (f64, f64) {
                let started = std::time::Instant::now();
                let command = ctx.queue.commandBuffer().unwrap();
                let encoder = KernelEncoder::begin(&command);
                let result = match arm {
                    Arm::Control => (|| {
                        encode_packed_grouped_mapped_iq3_xxs_f32(
                            &ctx,
                            &encoder,
                            &gate_bank,
                            &input,
                            &source_rows,
                            &destination_slots,
                            &schedule,
                            &control_gate,
                            H,
                            F,
                            E,
                            K,
                            N,
                            N,
                            ROUTES,
                        )?;
                        encode_packed_grouped_mapped_iq3_xxs_f32(
                            &ctx,
                            &encoder,
                            &up_bank,
                            &input,
                            &source_rows,
                            &destination_slots,
                            &schedule,
                            &control_up,
                            H,
                            F,
                            E,
                            K,
                            N,
                            N,
                            ROUTES,
                        )?;
                        encode_ds4_clamped_swiglu(
                            &ctx,
                            &encoder,
                            &control_gate.view_subrange(0, vec![(F * ROUTES) as u64]),
                            &control_up.view_subrange(0, vec![(F * ROUTES) as u64]),
                            &control_inner.view_subrange(0, vec![(F * ROUTES) as u64]),
                            CLAMP,
                        )?;
                        encode_packed_grouped_mapped_iq3_xxs_f32(
                            &ctx,
                            &encoder,
                            &down_bank,
                            &control_inner,
                            &destination_slots,
                            &destination_slots,
                            &schedule,
                            &control_output,
                            F,
                            H,
                            E,
                            K,
                            N,
                            ROUTES,
                            ROUTES,
                        )
                    })(),
                    Arm::Candidate => (|| {
                        encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
                            &ctx,
                            &encoder,
                            &gate_bank,
                            &up_bank,
                            &input,
                            &source_rows,
                            &destination_slots,
                            &schedule,
                            &candidate_output,
                            &candidate_inner,
                            H,
                            F,
                            E,
                            K,
                            N,
                            N,
                            ROUTES,
                            CLAMP,
                        )?;
                        encode_packed_grouped_mapped_iq3_xxs_f32(
                            &ctx,
                            &encoder,
                            &down_bank,
                            &candidate_inner,
                            &destination_slots,
                            &destination_slots,
                            &schedule,
                            &candidate_output,
                            F,
                            H,
                            E,
                            K,
                            N,
                            ROUTES,
                            ROUTES,
                        )
                    })(),
                };
                encoder.end();
                result.unwrap();
                command.commit();
                command.waitUntilCompleted();
                let wall_ms = started.elapsed().as_secs_f64() * 1e3;
                assert!(
                    command.error().is_none(),
                    "{schedule_name} {arm:?}: {:?}",
                    command.error()
                );
                let gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
                (gpu_ms, wall_ms)
            };

            let bits = |tensor: &MetalTensor, label: &str| {
                host_read_f32(tensor, label)
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
            };
            let assert_exact = || {
                assert_eq!(
                    bits(&candidate_arena, "fused all-IQ3 candidate output"),
                    bits(&control_arena, "fused all-IQ3 control output"),
                    "{schedule_name} output"
                );
                assert_eq!(
                    bits(&candidate_inner, "fused all-IQ3 candidate inner"),
                    bits(&control_inner, "fused all-IQ3 control inner"),
                    "{schedule_name} inner"
                );
            };

            sample(Arm::Control);
            sample(Arm::Candidate);
            assert_exact();
            for index in 0usize..5 {
                if index.is_multiple_of(2) {
                    sample(Arm::Control);
                    sample(Arm::Candidate);
                } else {
                    sample(Arm::Candidate);
                    sample(Arm::Control);
                }
            }

            let collect = |arm| (0..SAMPLES).map(|_| sample(arm)).collect::<Vec<_>>();
            let control_before = collect(Arm::Control);
            let candidate = collect(Arm::Candidate);
            let control_after = collect(Arm::Control);
            assert_exact();

            let split = |samples: &[(f64, f64)]| {
                (
                    samples.iter().map(|sample| sample.0).collect::<Vec<_>>(),
                    samples.iter().map(|sample| sample.1).collect::<Vec<_>>(),
                )
            };
            let (control_before_gpu, control_before_wall) = split(&control_before);
            let (candidate_gpu, candidate_wall) = split(&candidate);
            let (control_after_gpu, control_after_wall) = split(&control_after);
            let gpu_control_drift = relative_drift(
                median_ms(&control_before_gpu),
                median_ms(&control_after_gpu),
            );
            let wall_control_drift = relative_drift(
                median_ms(&control_before_wall),
                median_ms(&control_after_wall),
            );
            let gpu_candidate_drift = relative_drift(
                median_ms(&candidate_gpu[..SAMPLES / 2]),
                median_ms(&candidate_gpu[SAMPLES / 2..]),
            );
            let wall_candidate_drift = relative_drift(
                median_ms(&candidate_wall[..SAMPLES / 2]),
                median_ms(&candidate_wall[SAMPLES / 2..]),
            );
            let gpu_control_median =
                median_ms(&control_before_gpu).min(median_ms(&control_after_gpu));
            let wall_control_median =
                median_ms(&control_before_wall).min(median_ms(&control_after_wall));
            let gpu_control_p95 = percentile_ms(&control_before_gpu, 0.95)
                .min(percentile_ms(&control_after_gpu, 0.95));
            let wall_control_p95 = percentile_ms(&control_before_wall, 0.95)
                .min(percentile_ms(&control_after_wall, 0.95));
            let gpu_median_saving = 1.0 - median_ms(&candidate_gpu) / gpu_control_median;
            let wall_median_saving = 1.0 - median_ms(&candidate_wall) / wall_control_median;
            let gpu_p95_saving = 1.0 - percentile_ms(&candidate_gpu, 0.95) / gpu_control_p95;
            let wall_p95_saving = 1.0 - percentile_ms(&candidate_wall, 0.95) / wall_control_p95;

            eprintln!(
                "deepseek_v4 fused_all_iq3_packet schedule={schedule_name} control_before_gpu_ms={control_before_gpu:?} control_before_wall_ms={control_before_wall:?} candidate_gpu_ms={candidate_gpu:?} candidate_wall_ms={candidate_wall:?} control_after_gpu_ms={control_after_gpu:?} control_after_wall_ms={control_after_wall:?} gpu_control_drift={gpu_control_drift:.6} wall_control_drift={wall_control_drift:.6} gpu_candidate_drift={gpu_candidate_drift:.6} wall_candidate_drift={wall_candidate_drift:.6} gpu_median_saving={gpu_median_saving:.6} wall_median_saving={wall_median_saving:.6} gpu_p95_saving={gpu_p95_saving:.6} wall_p95_saving={wall_p95_saving:.6}"
            );

            if schedule_shape == ScheduleShape::Sparse {
                let _trace = crate::metal::kernel_trace_begin();
                sample(Arm::Control);
                let control_trace = crate::metal::kernel_trace_take_delta();
                sample(Arm::Candidate);
                let candidate_trace = crate::metal::kernel_trace_take_delta();
                assert_eq!(control_trace.encoders, 1);
                assert_eq!(control_trace.concurrent_encoders, 0);
                assert_eq!(control_trace.dispatches, 4);
                assert_eq!(candidate_trace.encoders, 1);
                assert_eq!(candidate_trace.concurrent_encoders, 0);
                assert_eq!(candidate_trace.dispatches, 2);
            }
            for (label, tensor) in [
                ("fused all-IQ3 control arena", &control_arena),
                ("fused all-IQ3 control inner", &control_inner),
                ("fused all-IQ3 candidate arena", &candidate_arena),
                ("fused all-IQ3 candidate inner", &candidate_inner),
            ] {
                assert_grouped_guards(label, tensor);
            }
            results.push(GateResult {
                schedule: schedule_name,
                gpu_control_drift,
                wall_control_drift,
                gpu_candidate_drift,
                wall_candidate_drift,
                gpu_median_saving,
                wall_median_saving,
                gpu_p95_saving,
                wall_p95_saving,
            });
        }

        for result in results {
            assert!(
                result.gpu_control_drift <= 0.05,
                "{} GPU control drift {:.3} exceeded 5%",
                result.schedule,
                result.gpu_control_drift
            );
            assert!(
                result.wall_control_drift <= 0.05,
                "{} wall control drift {:.3} exceeded 5%",
                result.schedule,
                result.wall_control_drift
            );
            assert!(
                result.gpu_candidate_drift <= 0.05,
                "{} GPU candidate drift {:.3} exceeded 5%",
                result.schedule,
                result.gpu_candidate_drift
            );
            assert!(
                result.wall_candidate_drift <= 0.05,
                "{} wall candidate drift {:.3} exceeded 5%",
                result.schedule,
                result.wall_candidate_drift
            );
            assert!(
                result.gpu_median_saving >= 0.15,
                "{} GPU median saving {:.3} missed 15%",
                result.schedule,
                result.gpu_median_saving
            );
            assert!(
                result.wall_median_saving >= 0.10,
                "{} wall median saving {:.3} missed 10%",
                result.schedule,
                result.wall_median_saving
            );
            assert!(
                result.gpu_p95_saving >= 0.10,
                "{} GPU p95 saving {:.3} missed 10%",
                result.schedule,
                result.gpu_p95_saving
            );
            assert!(
                result.wall_p95_saving >= 0.10,
                "{} wall p95 saving {:.3} missed 10%",
                result.schedule,
                result.wall_p95_saving
            );
        }
    }

    struct PackedRouteGenerationOwner {
        next: Cell<u32>,
    }

    impl PackedRouteGenerationOwner {
        fn new() -> Self {
            Self { next: Cell::new(1) }
        }

        fn with_next(next: u32) -> Self {
            Self {
                next: Cell::new(next),
            }
        }

        fn take(&self) -> Result<NonZeroU32, DeepSeekV4MetalError> {
            let generation = NonZeroU32::new(self.next.get()).ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("packed route generation owner reached zero".into())
            })?;
            let next = generation.get().checked_add(1).ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "packed route generation owner exhausted before wrap".into(),
                )
            })?;
            self.next.set(next);
            Ok(generation)
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum PackedRouteMicroproofSource {
        Learned,
        Hash,
    }

    struct PackedRouteMicroproofScratch {
        logits: MetalTensor,
        token_ids: MetalTensor,
        expert_ids: MetalTensor,
        weights: MetalTensor,
        route_generations: MetalTensor,
        route_status: MetalTensor,
        counts: MetalTensor,
        slot_ids: MetalTensor,
        schedule_generations: MetalTensor,
        aggregate: MetalTensor,
        signature: MetalTensor,
        compact_header: MetalTensor,
        compact_rows: MetalTensor,
        compact_slots: MetalTensor,
        compact_tiles32: MetalTensor,
        compact_tiles16: MetalTensor,
    }

    struct PackedRouteCapture {
        generation: u32,
        expert_ids: Vec<i32>,
        weights: Vec<f32>,
        route_generations: Vec<i32>,
        route_status: Vec<i32>,
        counts: Vec<i32>,
        slot_ids: Vec<i32>,
        schedule_generations: Vec<i32>,
        aggregate: Vec<i32>,
        signature: Vec<i32>,
    }

    struct PackedRouteCompactCapture {
        header: Vec<i32>,
        expert_ids: Vec<i32>,
        counts: Vec<i32>,
        rows: Vec<i32>,
        slots: Vec<i32>,
        tiles32: Vec<i32>,
        tiles16: Vec<i32>,
    }

    impl PackedRouteMicroproofScratch {
        fn new(ctx: &MetalContext) -> Result<Self, DeepSeekV4MetalError> {
            let n = PACKED_GPU_ROUTE_MAX_TOKENS as u64;
            let slot_elements = PACKED_GPU_ROUTE_MAX_TOKENS * MOE_EXPERT_COUNT;
            let mut guarded_slots = vec![
                PACKED_ROUTE_SLOT_PREFIX;
                PACKED_ROUTE_SLOT_GUARD_BYTES
                    + slot_elements * std::mem::size_of::<i32>()
                    + PACKED_ROUTE_SLOT_GUARD_BYTES
            ];
            guarded_slots
                [PACKED_ROUTE_SLOT_GUARD_BYTES + slot_elements * std::mem::size_of::<i32>()..]
                .fill(PACKED_ROUTE_SLOT_SUFFIX);
            let slot_ids = MetalTensor {
                buffer: ctx.buffer_from(&guarded_slots)?,
                offset: PACKED_ROUTE_SLOT_GUARD_BYTES as u64,
                shape: vec![n, MOE_EXPERT_COUNT as u64],
                dtype: GgmlType::I32,
                provenance: MetalTensorProvenance::OwnedWritable,
            };
            Ok(Self {
                logits: MetalTensor::zeros_f32(ctx, vec![MOE_EXPERT_COUNT as u64, n])?,
                token_ids: MetalTensor::zeros_i32(ctx, vec![n])?,
                expert_ids: MetalTensor::zeros_i32(ctx, vec![MOE_TOP_K as u64, n])?,
                weights: MetalTensor::zeros_f32(ctx, vec![MOE_TOP_K as u64, n])?,
                route_generations: MetalTensor::zeros_i32(ctx, vec![n])?,
                route_status: MetalTensor::zeros_i32(ctx, vec![n])?,
                counts: MetalTensor::zeros_i32(ctx, vec![MOE_EXPERT_COUNT as u64])?,
                slot_ids,
                schedule_generations: MetalTensor::zeros_i32(ctx, vec![MOE_EXPERT_COUNT as u64])?,
                aggregate: MetalTensor::zeros_i32(ctx, vec![PACKED_ROUTE_AGGREGATE_WIDTH as u64])?,
                signature: MetalTensor::zeros_i32(ctx, vec![PACKED_ROUTE_AGGREGATE_WIDTH as u64])?,
                compact_header: MetalTensor::zeros_i32(
                    ctx,
                    vec![PACKED_COMPACT_ROUTE_HEADER_WIDTH as u64],
                )?,
                compact_rows: MetalTensor::zeros_i32(ctx, vec![n * MOE_TOP_K as u64])?,
                compact_slots: MetalTensor::zeros_i32(ctx, vec![n * MOE_TOP_K as u64])?,
                compact_tiles32: MetalTensor::zeros_i32(
                    ctx,
                    vec![PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS as u64],
                )?,
                compact_tiles16: MetalTensor::zeros_i32(
                    ctx,
                    vec![PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS as u64],
                )?,
            })
        }

        fn assert_slot_guards(&self) {
            let payload_bytes = self.slot_ids.n_bytes() as usize;
            let base = self.slot_ids.buffer.contents().as_ptr().cast::<u8>();
            let prefix = unsafe {
                std::slice::from_raw_parts(
                    base.add(self.slot_ids.offset as usize - PACKED_ROUTE_SLOT_GUARD_BYTES),
                    PACKED_ROUTE_SLOT_GUARD_BYTES,
                )
            };
            let suffix = unsafe {
                std::slice::from_raw_parts(
                    base.add(self.slot_ids.offset as usize + payload_bytes),
                    PACKED_ROUTE_SLOT_GUARD_BYTES,
                )
            };
            assert!(prefix.iter().all(|&byte| byte == PACKED_ROUTE_SLOT_PREFIX));
            assert!(suffix.iter().all(|&byte| byte == PACKED_ROUTE_SLOT_SUFFIX));
        }

        fn buffers(&self) -> PackedGpuRouteBuffers<'_> {
            PackedGpuRouteBuffers {
                logits: &self.logits,
                token_ids: &self.token_ids,
                expert_ids: &self.expert_ids,
                weights: &self.weights,
                route_generations: &self.route_generations,
                route_status: &self.route_status,
                counts: &self.counts,
                slot_ids: &self.slot_ids,
                schedule_generations: &self.schedule_generations,
                aggregate: &self.aggregate,
                signature: &self.signature,
                compact_header: &self.compact_header,
            }
        }

        fn encode_learned(
            &self,
            ctx: &MetalContext,
            enc: &KernelEncoder,
            bias: &MetalTensor,
            n_tokens: usize,
            produced_tokens: usize,
            generation: NonZeroU32,
        ) -> Result<(), DeepSeekV4MetalError> {
            self.buffers().encode_learned(
                ctx,
                enc,
                bias,
                n_tokens,
                produced_tokens,
                generation,
                1.5,
            )
        }

        fn encode_hash(
            &self,
            ctx: &MetalContext,
            enc: &KernelEncoder,
            token_to_expert: &MetalTensor,
            n_tokens: usize,
            produced_tokens: usize,
            generation: NonZeroU32,
        ) -> Result<(), DeepSeekV4MetalError> {
            self.buffers().encode_hash(
                ctx,
                enc,
                token_to_expert,
                n_tokens,
                produced_tokens,
                generation,
                1.5,
            )
        }

        fn encode_schedule(
            &self,
            ctx: &MetalContext,
            enc: &KernelEncoder,
            n_tokens: usize,
            produced_experts: usize,
            generation: NonZeroU32,
        ) -> Result<(), DeepSeekV4MetalError> {
            self.buffers()
                .encode_schedule(ctx, enc, n_tokens, produced_experts, generation)
        }

        fn encode_validate(
            &self,
            ctx: &MetalContext,
            enc: &KernelEncoder,
            n_tokens: usize,
            generation: NonZeroU32,
        ) -> Result<(), DeepSeekV4MetalError> {
            self.buffers()
                .encode_validate(ctx, enc, n_tokens, generation)
        }

        fn encode_signature(
            &self,
            ctx: &MetalContext,
            enc: &KernelEncoder,
            n_tokens: usize,
            generation: NonZeroU32,
        ) -> Result<(), DeepSeekV4MetalError> {
            self.buffers()
                .encode_signature(ctx, enc, n_tokens, generation)
        }

        fn encode_compact(
            &self,
            ctx: &MetalContext,
            enc: &KernelEncoder,
            n_tokens: usize,
            generation: NonZeroU32,
        ) -> Result<(), DeepSeekV4MetalError> {
            let route_count = n_tokens * MOE_TOP_K;
            self.buffers().encode_compact(
                ctx,
                enc,
                &i32_prefix(
                    &self.compact_rows,
                    vec![route_count as u64],
                    "packed compact proof rows",
                )?,
                &i32_prefix(
                    &self.compact_slots,
                    vec![route_count as u64],
                    "packed compact proof slots",
                )?,
                &self.compact_tiles32,
                &self.compact_tiles16,
                n_tokens,
                generation,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_pipeline(
            &self,
            ctx: &MetalContext,
            enc: &KernelEncoder,
            source: PackedRouteMicroproofSource,
            bias: &MetalTensor,
            token_to_expert: &MetalTensor,
            n_tokens: usize,
            produced_tokens: usize,
            produced_experts: usize,
            generation: NonZeroU32,
        ) -> Result<(), DeepSeekV4MetalError> {
            match source {
                PackedRouteMicroproofSource::Learned => {
                    self.encode_learned(ctx, enc, bias, n_tokens, produced_tokens, generation)?
                }
                PackedRouteMicroproofSource::Hash => self.encode_hash(
                    ctx,
                    enc,
                    token_to_expert,
                    n_tokens,
                    produced_tokens,
                    generation,
                )?,
            }
            self.encode_schedule(ctx, enc, n_tokens, produced_experts, generation)?;
            self.encode_validate(ctx, enc, n_tokens, generation)?;
            self.encode_signature(ctx, enc, n_tokens, generation)
        }

        fn capture(&self, n_tokens: usize, generation: u32) -> PackedRouteCapture {
            let routes = n_tokens * MOE_TOP_K;
            let schedule = n_tokens * MOE_EXPERT_COUNT;
            let mut expert_ids = host_read_i32(&self.expert_ids, "packed route IDs").unwrap();
            let mut weights = host_read_f32(&self.weights, "packed route weights").unwrap();
            let mut route_generations =
                host_read_i32(&self.route_generations, "packed route generations").unwrap();
            let mut route_status =
                host_read_i32(&self.route_status, "packed route statuses").unwrap();
            let mut slot_ids = host_read_i32(&self.slot_ids, "packed route slot IDs").unwrap();
            expert_ids.truncate(routes);
            weights.truncate(routes);
            route_generations.truncate(n_tokens);
            route_status.truncate(n_tokens);
            slot_ids.truncate(schedule);
            PackedRouteCapture {
                generation,
                expert_ids,
                weights,
                route_generations,
                route_status,
                counts: host_read_i32(&self.counts, "packed route counts").unwrap(),
                slot_ids,
                schedule_generations: host_read_i32(
                    &self.schedule_generations,
                    "packed schedule generations",
                )
                .unwrap(),
                aggregate: host_read_i32(&self.aggregate, "packed route aggregate").unwrap(),
                signature: host_read_i32(&self.signature, "packed route signature").unwrap(),
            }
        }

        fn capture_compact(&self, n_tokens: usize) -> PackedRouteCompactCapture {
            let route_count = n_tokens * MOE_TOP_K;
            let mut expert_ids =
                host_read_i32(&self.expert_ids, "packed compact proof expert IDs").unwrap();
            let mut rows = host_read_i32(&self.compact_rows, "packed compact proof rows").unwrap();
            let mut slots =
                host_read_i32(&self.compact_slots, "packed compact proof slots").unwrap();
            rows.truncate(route_count);
            slots.truncate(route_count);
            expert_ids.truncate(route_count);
            PackedRouteCompactCapture {
                header: host_read_i32(&self.compact_header, "packed compact proof header").unwrap(),
                expert_ids,
                counts: host_read_i32(&self.counts, "packed compact proof counts").unwrap(),
                rows,
                slots,
                tiles32: host_read_i32(&self.compact_tiles32, "packed compact proof 32-row tiles")
                    .unwrap(),
                tiles16: host_read_i32(&self.compact_tiles16, "packed compact proof 16-row tiles")
                    .unwrap(),
            }
        }
    }

    struct PackedRouteFixture {
        scratch: PackedRouteMicroproofScratch,
        bias: MetalTensor,
        token_to_expert: MetalTensor,
        logits: Vec<f32>,
        bias_values: Vec<f32>,
        token_ids: Vec<i32>,
        hash_map: Vec<i32>,
        generations: PackedRouteGenerationOwner,
    }

    impl PackedRouteFixture {
        fn new(ctx: &MetalContext) -> Self {
            const VOCAB_SIZE: usize = PACKED_GPU_ROUTE_MAX_TOKENS + 1;
            let scratch = PackedRouteMicroproofScratch::new(ctx).unwrap();
            let logits = (0..PACKED_GPU_ROUTE_MAX_TOKENS)
                .flat_map(|token| {
                    (0..MOE_EXPERT_COUNT).map(move |expert| {
                        if token.is_multiple_of(29) {
                            (token % 5) as f32 * 0.125 - 0.25
                        } else {
                            let mixed =
                                expert * 1_103 + token * 7_919 + (expert ^ token) * 53 + token / 3;
                            (mixed % 8_191) as f32 * 0.0025 - 10.0
                        }
                    })
                })
                .collect::<Vec<_>>();
            let bias_values = (0..MOE_EXPERT_COUNT)
                .map(|expert| ((expert * 193 + 7) % 257) as f32 * 0.0002 - 0.0256)
                .collect::<Vec<_>>();
            let token_ids = (0..PACKED_GPU_ROUTE_MAX_TOKENS as i32).collect::<Vec<_>>();
            let hash_map = (0..VOCAB_SIZE)
                .flat_map(|token| {
                    (0..MOE_TOP_K).map(move |slot| ((token * 17 + slot * 37) % 256) as i32)
                })
                .collect::<Vec<_>>();
            for row in hash_map.chunks_exact(MOE_TOP_K) {
                let mut sorted = row.to_vec();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(sorted.len(), MOE_TOP_K);
            }
            host_write_f32(&scratch.logits, &logits, "packed route fixture logits").unwrap();
            host_write_i32(
                &scratch.token_ids,
                &token_ids,
                "packed route fixture token IDs",
            )
            .unwrap();
            let bias = MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&bias_values),
                vec![MOE_EXPERT_COUNT as u64],
                GgmlType::F32,
            )
            .unwrap();
            let token_to_expert = MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&hash_map),
                vec![MOE_TOP_K as u64, VOCAB_SIZE as u64],
                GgmlType::I32,
            )
            .unwrap();
            Self {
                scratch,
                bias,
                token_to_expert,
                logits,
                bias_values,
                token_ids,
                hash_map,
                generations: PackedRouteGenerationOwner::new(),
            }
        }

        fn run(
            &self,
            ctx: &MetalContext,
            source: PackedRouteMicroproofSource,
            n_tokens: usize,
            produced_tokens: usize,
            produced_experts: usize,
        ) -> PackedRouteCapture {
            let generation = self.generations.take().unwrap();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            self.scratch
                .encode_pipeline(
                    ctx,
                    &encoder,
                    source,
                    &self.bias,
                    &self.token_to_expert,
                    n_tokens,
                    produced_tokens,
                    produced_experts,
                    generation,
                )
                .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(
                command.error().is_none(),
                "packed route command failed: {:?}",
                command.error()
            );
            self.scratch.capture(n_tokens, generation.get())
        }

        fn run_compact(
            &self,
            ctx: &MetalContext,
            source: PackedRouteMicroproofSource,
            n_tokens: usize,
        ) -> (u32, PackedRouteCompactCapture) {
            let generation = self.generations.take().unwrap();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            match source {
                PackedRouteMicroproofSource::Learned => self
                    .scratch
                    .encode_learned(ctx, &encoder, &self.bias, n_tokens, n_tokens, generation)
                    .unwrap(),
                PackedRouteMicroproofSource::Hash => self
                    .scratch
                    .encode_hash(
                        ctx,
                        &encoder,
                        &self.token_to_expert,
                        n_tokens,
                        n_tokens,
                        generation,
                    )
                    .unwrap(),
            }
            self.scratch
                .encode_compact(ctx, &encoder, n_tokens, generation)
                .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(
                command.error().is_none(),
                "packed compact route command failed: {:?}",
                command.error()
            );
            (generation.get(), self.scratch.capture_compact(n_tokens))
        }

        fn expected_routes(
            &self,
            source: PackedRouteMicroproofSource,
            n_tokens: usize,
        ) -> (Vec<i32>, Vec<f32>) {
            let mut ids = Vec::with_capacity(n_tokens * MOE_TOP_K);
            let mut weights = Vec::with_capacity(n_tokens * MOE_TOP_K);
            for token in 0..n_tokens {
                let logits = &self.logits[token * MOE_EXPERT_COUNT..(token + 1) * MOE_EXPERT_COUNT];
                let scores = crate::deepseek_v4_oracle::sqrt_softplus_scores(logits).unwrap();
                let decision = match source {
                    PackedRouteMicroproofSource::Learned => {
                        crate::deepseek_v4_oracle::learned_route(
                            &scores,
                            &self.bias_values,
                            MOE_TOP_K,
                            1.5,
                        )
                    }
                    PackedRouteMicroproofSource::Hash => {
                        let token_id = self.token_ids[token] as usize;
                        let selected = self.hash_map
                            [token_id * MOE_TOP_K..(token_id + 1) * MOE_TOP_K]
                            .iter()
                            .map(|&expert| expert as usize)
                            .collect::<Vec<_>>();
                        crate::deepseek_v4_oracle::hash_route(&scores, &selected, 1.5)
                    }
                }
                .unwrap();
                ids.extend(decision.expert_ids.iter().map(|&expert| expert as i32));
                weights.extend_from_slice(&decision.weights);
            }
            (ids, weights)
        }
    }

    fn expected_packed_schedule(expert_ids: &[i32], n_tokens: usize) -> (Vec<i32>, Vec<i32>) {
        let mut counts = vec![0; MOE_EXPERT_COUNT];
        let mut slots = vec![-1; MOE_EXPERT_COUNT * n_tokens];
        for expert in 0..MOE_EXPERT_COUNT {
            let mut count = 0;
            for token in 0..n_tokens {
                for slot in 0..MOE_TOP_K {
                    let global_slot = token * MOE_TOP_K + slot;
                    if expert_ids[global_slot] == expert as i32 {
                        slots[expert * n_tokens + count] = global_slot as i32;
                        count += 1;
                    }
                }
            }
            counts[expert] = count as i32;
        }
        (counts, slots)
    }

    fn expected_compact_schedule(
        expert_ids: &[i32],
        n_tokens: usize,
    ) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<ExpertBucket>) {
        let (counts, dense_slots) = expected_packed_schedule(expert_ids, n_tokens);
        let mut rows = Vec::with_capacity(n_tokens * MOE_TOP_K);
        let mut slots = Vec::with_capacity(n_tokens * MOE_TOP_K);
        let mut buckets = Vec::new();
        for (expert, &count) in counts.iter().enumerate() {
            let count = count as usize;
            if count == 0 {
                continue;
            }
            let start = slots.len();
            for &slot in &dense_slots[expert * n_tokens..expert * n_tokens + count] {
                rows.push(slot / MOE_TOP_K as i32);
                slots.push(slot);
            }
            buckets.push(ExpertBucket {
                expert,
                start,
                len: count,
            });
        }
        (counts, rows, slots, buckets)
    }

    fn padded_tile_words(tiles: &[PackedGroupedExpertTile], descriptor_words: usize) -> Vec<i32> {
        let mut words = bytemuck::cast_slice::<PackedGroupedExpertTile, i32>(tiles).to_vec();
        words.resize(descriptor_words, 0);
        words
    }

    fn assert_packed_compact_capture(
        fixture: &PackedRouteFixture,
        source: PackedRouteMicroproofSource,
        n_tokens: usize,
        generation: u32,
        capture: &PackedRouteCompactCapture,
    ) {
        let (expected_ids, _) = fixture.expected_routes(source, n_tokens);
        for token in 0..n_tokens {
            let start = token * MOE_TOP_K;
            let mut actual = capture.expert_ids[start..start + MOE_TOP_K].to_vec();
            let mut expected = expected_ids[start..start + MOE_TOP_K].to_vec();
            actual.sort_unstable();
            expected.sort_unstable();
            assert_eq!(
                actual, expected,
                "packed {source:?} route set at token {token}"
            );
        }
        let (counts, rows, slots, buckets) =
            expected_compact_schedule(&capture.expert_ids, n_tokens);
        let tiles32 = packed_grouped_expert_tiles(n_tokens, &buckets).unwrap();
        let tiles16 = packed_grouped_iq2_mma16_tiles(n_tokens, &buckets).unwrap();
        assert_eq!(capture.counts, counts);
        assert_eq!(capture.rows, rows);
        assert_eq!(capture.slots, slots);
        assert_eq!(
            capture.tiles32,
            padded_tile_words(&tiles32, PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS)
        );
        assert_eq!(
            capture.tiles16,
            padded_tile_words(&tiles16, PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS)
        );
        assert_eq!(
            capture.header,
            vec![
                generation as i32,
                PACKED_COMPACT_ROUTE_STATUS_READY,
                (n_tokens * MOE_TOP_K) as i32,
                buckets.len() as i32,
                tiles16.len() as i32,
                tiles32.len() as i32,
                packed_route_compact_completion(generation, n_tokens) as i32,
                n_tokens as i32,
            ]
        );
    }

    fn write_raw_f32(tensor: &MetalTensor, values: &[f32]) {
        assert_eq!(tensor.dtype, GgmlType::F32);
        assert!(tensor.is_writable());
        assert_eq!(tensor.n_elements() as usize, values.len());
        let destination = unsafe {
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>()
        };
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), destination, values.len());
        }
    }

    fn assert_packed_route_capture(
        fixture: &PackedRouteFixture,
        source: PackedRouteMicroproofSource,
        n_tokens: usize,
        capture: &PackedRouteCapture,
    ) {
        let (expected_ids, expected_weights) = fixture.expected_routes(source, n_tokens);
        if let Some(index) = capture
            .expert_ids
            .iter()
            .zip(&expected_ids)
            .position(|(actual, expected)| actual != expected)
        {
            let token = index / MOE_TOP_K;
            let start = token * MOE_TOP_K;
            panic!(
                "packed {source:?} route ID differs at N={n_tokens} token {token}: actual={:?} expected={:?}",
                &capture.expert_ids[start..start + MOE_TOP_K],
                &expected_ids[start..start + MOE_TOP_K],
            );
        }
        for (index, (&actual, &expected)) in
            capture.weights.iter().zip(&expected_weights).enumerate()
        {
            let allowed = 1e-4 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= allowed,
                "packed route weight {index}: {actual} vs {expected}, allowed {allowed}"
            );
        }
        let (expected_counts, expected_slots) = expected_packed_schedule(&expected_ids, n_tokens);
        assert_eq!(capture.counts, expected_counts);
        assert_eq!(capture.slot_ids, expected_slots);
        assert_eq!(
            capture.route_generations,
            vec![capture.generation as i32; n_tokens]
        );
        assert_eq!(
            capture.route_status,
            vec![DEEPSEEK_V4_ROUTE_STATUS_READY; n_tokens]
        );
        assert_eq!(
            capture.schedule_generations,
            vec![capture.generation as i32; MOE_EXPERT_COUNT]
        );
        assert_eq!(
            capture.aggregate,
            vec![
                capture.generation as i32,
                DEEPSEEK_V4_ROUTE_STATUS_READY,
                (n_tokens * MOE_TOP_K) as i32,
                packed_route_completion(capture.generation, n_tokens) as i32,
            ]
        );
        assert_eq!(capture.signature[0], capture.generation as i32);
        assert_eq!(capture.signature[1], DEEPSEEK_V4_ROUTE_STATUS_READY);
        assert_eq!(
            capture.signature[2] as u32,
            packed_route_signature_hash(
                &capture.expert_ids,
                &capture.weights,
                &capture.counts,
                &capture.slot_ids,
                n_tokens,
            )
            .unwrap()
        );
        assert_eq!(
            capture.signature[3],
            packed_route_signature_completion(capture.generation, n_tokens) as i32
        );
    }

    fn assert_packed_route_set_capture(
        fixture: &PackedRouteFixture,
        source: PackedRouteMicroproofSource,
        n_tokens: usize,
        capture: &PackedRouteCapture,
    ) {
        let (expected_ids, expected_weights) = fixture.expected_routes(source, n_tokens);
        for token in 0..n_tokens {
            let start = token * MOE_TOP_K;
            let actual_ids = &capture.expert_ids[start..start + MOE_TOP_K];
            let expected_ids = &expected_ids[start..start + MOE_TOP_K];
            let mut actual_set = actual_ids.to_vec();
            let mut expected_set = expected_ids.to_vec();
            actual_set.sort_unstable();
            expected_set.sort_unstable();
            assert_eq!(
                actual_set, expected_set,
                "packed {source:?} route set at N={n_tokens} token {token}"
            );
            for (slot, &expert) in actual_ids.iter().enumerate() {
                let expected_slot = expected_ids
                    .iter()
                    .position(|&expected| expected == expert)
                    .unwrap();
                let actual = capture.weights[start + slot];
                let expected = expected_weights[start + expected_slot];
                let allowed = 1e-4 * expected.abs().max(1.0);
                assert!(
                    (actual - expected).abs() <= allowed,
                    "packed route weight for expert {expert}: {actual} vs {expected}, allowed {allowed}"
                );
            }
        }
        let (expected_counts, expected_slots) =
            expected_packed_schedule(&capture.expert_ids, n_tokens);
        assert_eq!(capture.counts, expected_counts);
        assert_eq!(capture.slot_ids, expected_slots);
        assert_eq!(
            capture.route_generations,
            vec![capture.generation as i32; n_tokens]
        );
        assert_eq!(
            capture.route_status,
            vec![DEEPSEEK_V4_ROUTE_STATUS_READY; n_tokens]
        );
        assert_eq!(
            capture.schedule_generations,
            vec![capture.generation as i32; MOE_EXPERT_COUNT]
        );
        assert_eq!(
            capture.aggregate,
            vec![
                capture.generation as i32,
                DEEPSEEK_V4_ROUTE_STATUS_READY,
                (n_tokens * MOE_TOP_K) as i32,
                packed_route_completion(capture.generation, n_tokens) as i32,
            ]
        );
        assert_eq!(
            capture.signature,
            vec![
                capture.generation as i32,
                DEEPSEEK_V4_ROUTE_STATUS_READY,
                packed_route_signature_hash(
                    &capture.expert_ids,
                    &capture.weights,
                    &capture.counts,
                    &capture.slot_ids,
                    n_tokens,
                )
                .unwrap() as i32,
                packed_route_signature_completion(capture.generation, n_tokens) as i32,
            ]
        );
    }

    fn assert_packed_route_failure(
        capture: &PackedRouteCapture,
        n_tokens: usize,
        status: i32,
        total: i32,
    ) {
        assert_eq!(
            capture.aggregate,
            vec![
                capture.generation as i32,
                status,
                total,
                packed_route_completion(capture.generation, n_tokens) as i32,
            ]
        );
        assert_eq!(
            capture.signature,
            vec![
                capture.generation as i32,
                PACKED_ROUTE_INVALID_AGGREGATE,
                0,
                packed_route_signature_completion(capture.generation, n_tokens) as i32,
            ]
        );
    }

    #[test]
    fn packed_gpu_routes_and_schedules_match_cpu_at_representative_widths() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let fixture = PackedRouteFixture::new(&ctx);
        for n_tokens in [1, 2, 7, 16, 31, 32, 127, 128, 129, 511, 512, 2_047, 2_048] {
            for source in [
                PackedRouteMicroproofSource::Learned,
                PackedRouteMicroproofSource::Hash,
            ] {
                let capture = fixture.run(&ctx, source, n_tokens, n_tokens, MOE_EXPERT_COUNT);
                if n_tokens <= 128 || matches!(source, PackedRouteMicroproofSource::Hash) {
                    assert_packed_route_capture(&fixture, source, n_tokens, &capture);
                } else {
                    assert_packed_route_set_capture(&fixture, source, n_tokens, &capture);
                }
            }
        }
    }

    #[test]
    fn packed_gpu_route_compaction_matches_cpu_at_capacity_boundaries() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let fixture = PackedRouteFixture::new(&ctx);
        for n_tokens in [1, 127, 128, 129, 2_047, 2_048] {
            for source in [
                PackedRouteMicroproofSource::Learned,
                PackedRouteMicroproofSource::Hash,
            ] {
                let (generation, capture) = fixture.run_compact(&ctx, source, n_tokens);
                assert_packed_compact_capture(&fixture, source, n_tokens, generation, &capture);
            }
        }
    }

    #[test]
    fn packed_gpu_route_compaction_rejects_invalid_late_tokens() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let fixture = PackedRouteFixture::new(&ctx);
        let n_tokens = 512;
        let (valid_generation, valid_capture) =
            fixture.run_compact(&ctx, PackedRouteMicroproofSource::Learned, n_tokens);
        assert_packed_compact_capture(
            &fixture,
            PackedRouteMicroproofSource::Learned,
            n_tokens,
            valid_generation,
            &valid_capture,
        );
        let (valid_ids, valid_weights) =
            fixture.expected_routes(PackedRouteMicroproofSource::Learned, n_tokens);
        let run = |ids: &[i32], weights: &[f32], generations: &[i32], statuses: &[i32]| {
            let generation = fixture.generations.take().unwrap();
            let mut full_ids = vec![-1; PACKED_GPU_ROUTE_MAX_TOKENS * MOE_TOP_K];
            full_ids[..ids.len()].copy_from_slice(ids);
            host_write_i32(
                &fixture.scratch.expert_ids,
                &full_ids,
                "compact invalid IDs",
            )
            .unwrap();
            let mut full_weights = vec![0.0; PACKED_GPU_ROUTE_MAX_TOKENS * MOE_TOP_K];
            full_weights[..weights.len()].copy_from_slice(weights);
            write_raw_f32(&fixture.scratch.weights, &full_weights);
            let mut full_generations = vec![0; PACKED_GPU_ROUTE_MAX_TOKENS];
            full_generations[..generations.len()].copy_from_slice(generations);
            host_write_i32(
                &fixture.scratch.route_generations,
                &full_generations,
                "compact invalid generations",
            )
            .unwrap();
            let mut full_statuses = vec![0; PACKED_GPU_ROUTE_MAX_TOKENS];
            full_statuses[..statuses.len()].copy_from_slice(statuses);
            host_write_i32(
                &fixture.scratch.route_status,
                &full_statuses,
                "compact invalid statuses",
            )
            .unwrap();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            fixture
                .scratch
                .encode_compact(&ctx, &encoder, n_tokens, generation)
                .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none());
            let header =
                host_read_i32(&fixture.scratch.compact_header, "compact invalid header").unwrap();
            assert_eq!(header[0], generation.get() as i32);
            assert_eq!(header[1], PACKED_COMPACT_ROUTE_STATUS_INVALID_ROUTE);
            assert_eq!(
                header[6],
                packed_route_compact_completion(generation.get(), n_tokens) as i32
            );
            assert_eq!(header[7], n_tokens as i32);
        };

        let ready = vec![DEEPSEEK_V4_ROUTE_STATUS_READY; n_tokens];
        let next_generation = fixture.generations.next.get() as i32;
        let valid_generations = vec![next_generation; n_tokens];

        let mut stale_generations = valid_generations.clone();
        stale_generations[256] = 0;
        run(&valid_ids, &valid_weights, &stale_generations, &ready);

        let next_generation = fixture.generations.next.get() as i32;
        let valid_generations = vec![next_generation; n_tokens];
        let mut failed_status = ready.clone();
        failed_status[300] = DEEPSEEK_V4_ROUTE_STATUS_INVALID_TOKEN;
        run(
            &valid_ids,
            &valid_weights,
            &valid_generations,
            &failed_status,
        );

        let next_generation = fixture.generations.next.get() as i32;
        let valid_generations = vec![next_generation; n_tokens];
        let mut duplicate_ids = valid_ids.clone();
        duplicate_ids[400 * MOE_TOP_K + 1] = duplicate_ids[400 * MOE_TOP_K];
        run(&duplicate_ids, &valid_weights, &valid_generations, &ready);

        let next_generation = fixture.generations.next.get() as i32;
        let valid_generations = vec![next_generation; n_tokens];
        let mut nonfinite_weights = valid_weights;
        nonfinite_weights[511 * MOE_TOP_K + 5] = f32::NAN;
        run(&valid_ids, &nonfinite_weights, &valid_generations, &ready);
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[test]
    fn packed_gpu_route_policy_rejects_unqualified_widths() {
        validate_packed_route_policy_scope(
            PackedRoutePolicy::GpuExperimental,
            PACKED_GPU_ROUTE_MAX_TOKENS,
        )
        .unwrap();
        assert!(
            validate_packed_route_policy_scope(
                PackedRoutePolicy::GpuExperimental,
                PACKED_GPU_ROUTE_MAX_TOKENS + 1,
            )
            .is_err()
        );
        validate_packed_route_policy_scope(PackedRoutePolicy::Cpu, DEEPSEEK_V4_PREFILL_MAX_TOKENS)
            .unwrap();
    }

    #[test]
    fn packed_gpu_routes_are_bitwise_singleton_equivalent_on_adversarial_scores() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let packed = PackedRouteMicroproofScratch::new(&ctx).unwrap();
        let singleton = DeepSeekV4MoeScratch::new(
            &ctx,
            DeepSeekV4MoeConfig {
                hidden_size: 1,
                ffn_size: 1,
                expert_count: MOE_EXPERT_COUNT,
                top_k: MOE_TOP_K,
                routed_scale: 1.5,
            },
        )
        .unwrap();
        let generations = PackedRouteGenerationOwner::new();
        let branch_values = [
            f32::from_bits((-20.0f32).to_bits() + 1),
            -20.0,
            f32::from_bits((-20.0f32).to_bits() - 1),
            f32::from_bits(20.0f32.to_bits() - 1),
            20.0,
            f32::from_bits(20.0f32.to_bits() + 1),
            -f32::MAX,
            f32::MAX,
        ];
        let mut cutoff_bias = vec![-1.0; MOE_EXPERT_COUNT];
        cutoff_bias[..7].fill(0.25);
        cutoff_bias[6] = f32::from_bits(0.25f32.to_bits() - 1);
        let cases = [
            (vec![0.0; MOE_EXPERT_COUNT], vec![0.0; MOE_EXPERT_COUNT]),
            (vec![0.0; MOE_EXPERT_COUNT], cutoff_bias),
            (
                (0..MOE_EXPERT_COUNT)
                    .map(|expert| branch_values[expert % branch_values.len()])
                    .collect(),
                (0..MOE_EXPERT_COUNT)
                    .map(|expert| (expert % 11) as f32 * 0.0001 - 0.0005)
                    .collect(),
            ),
        ];
        let hash_values = (0..MOE_TOP_K as i32).collect::<Vec<_>>();
        let hash_map = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&hash_values),
            vec![MOE_TOP_K as u64, 1],
            GgmlType::I32,
        )
        .unwrap();
        host_write_i32(
            &packed.token_ids,
            &vec![0; PACKED_GPU_ROUTE_MAX_TOKENS],
            "singleton-equivalent packed token IDs",
        )
        .unwrap();

        for (case, (logits, bias_values)) in cases.into_iter().enumerate() {
            host_write_f32(
                &singleton.logits,
                &logits,
                "singleton-equivalent singleton logits",
            )
            .unwrap();
            let mut packed_logits = vec![0.0; MOE_EXPERT_COUNT * PACKED_GPU_ROUTE_MAX_TOKENS];
            packed_logits[..MOE_EXPERT_COUNT].copy_from_slice(&logits);
            host_write_f32(
                &packed.logits,
                &packed_logits,
                "singleton-equivalent packed logits",
            )
            .unwrap();
            let bias = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&bias_values),
                vec![MOE_EXPERT_COUNT as u64],
                GgmlType::F32,
            )
            .unwrap();

            for source in [
                PackedRouteMicroproofSource::Learned,
                PackedRouteMicroproofSource::Hash,
            ] {
                let generation = generations.take().unwrap();
                let command = ctx.queue.commandBuffer().unwrap();
                let encoder = KernelEncoder::begin(&command);
                match source {
                    PackedRouteMicroproofSource::Learned => {
                        singleton
                            .encode_route_learned_gpu(&ctx, &encoder, &bias)
                            .unwrap();
                        packed
                            .encode_learned(&ctx, &encoder, &bias, 1, 1, generation)
                            .unwrap();
                    }
                    PackedRouteMicroproofSource::Hash => {
                        singleton
                            .encode_route_hash_gpu(&ctx, &encoder, 0, &hash_map)
                            .unwrap();
                        packed
                            .encode_hash(&ctx, &encoder, &hash_map, 1, 1, generation)
                            .unwrap();
                    }
                }
                encoder.end();
                command.commit();
                command.waitUntilCompleted();
                assert!(command.error().is_none(), "route case {case} failed");
                let expected = singleton.capture_gpu_route_record().unwrap();
                let mut actual_ids =
                    host_read_i32(&packed.expert_ids, "singleton-equivalent packed IDs").unwrap();
                let mut actual_weights =
                    host_read_f32(&packed.weights, "singleton-equivalent packed weights").unwrap();
                let actual_status =
                    host_read_i32(&packed.route_status, "singleton-equivalent packed status")
                        .unwrap();
                actual_ids.truncate(MOE_TOP_K);
                actual_weights.truncate(MOE_TOP_K);
                assert_eq!(actual_status[0], expected.status, "case {case}");
                assert_eq!(actual_ids, expected.expert_ids, "case {case}");
                assert_eq!(
                    actual_weights
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .weights
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "case {case}"
                );
            }
        }
    }

    #[test]
    fn packed_route_signature_binds_each_payload_class() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let fixture = PackedRouteFixture::new(&ctx);
        let n_tokens = 12;
        let capture = fixture.run(
            &ctx,
            PackedRouteMicroproofSource::Learned,
            n_tokens,
            n_tokens,
            MOE_EXPERT_COUNT,
        );
        assert_packed_route_capture(
            &fixture,
            PackedRouteMicroproofSource::Learned,
            n_tokens,
            &capture,
        );
        let hash = |ids: &[i32], weights: &[f32], counts: &[i32], slots: &[i32]| {
            packed_route_signature_hash(ids, weights, counts, slots, n_tokens).unwrap()
        };
        let baseline = hash(
            &capture.expert_ids,
            &capture.weights,
            &capture.counts,
            &capture.slot_ids,
        );

        let mut ids = capture.expert_ids.clone();
        ids[0] ^= 1;
        assert_ne!(
            hash(&ids, &capture.weights, &capture.counts, &capture.slot_ids),
            baseline
        );
        let mut weights = capture.weights.clone();
        weights[0] = f32::from_bits(weights[0].to_bits() ^ 1);
        assert_ne!(
            hash(
                &capture.expert_ids,
                &weights,
                &capture.counts,
                &capture.slot_ids,
            ),
            baseline
        );
        let mut counts = capture.counts.clone();
        counts[0] ^= 1;
        assert_ne!(
            hash(
                &capture.expert_ids,
                &capture.weights,
                &counts,
                &capture.slot_ids,
            ),
            baseline
        );
        let mut slots = capture.slot_ids.clone();
        let occupied = slots.iter().position(|&slot| slot >= 0).unwrap();
        slots[occupied] ^= 1;
        assert_ne!(
            hash(
                &capture.expert_ids,
                &capture.weights,
                &capture.counts,
                &slots,
            ),
            baseline
        );
        let mut padding = capture.slot_ids.clone();
        let sentinel = padding.iter().position(|&slot| slot == -1).unwrap();
        padding[sentinel] = -2;
        assert_ne!(
            hash(
                &capture.expert_ids,
                &capture.weights,
                &capture.counts,
                &padding,
            ),
            baseline
        );
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[test]
    fn production_packed_gpu_route_owns_and_compacts_the_qualified_schedule() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let fixture = PackedRouteFixture::new(&ctx);
        let production =
            DeepSeekV4PrefillScratch::new(&ctx, DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS).unwrap();
        host_write_f32(
            &production.moe.logits,
            &fixture.logits,
            "production packed GPU route logits",
        )
        .unwrap();
        host_write_i32(
            &production.token_ids,
            &fixture.token_ids,
            "production packed GPU route token IDs",
        )
        .unwrap();

        for (source_kind, n_tokens) in [
            (PackedRouteMicroproofSource::Learned, 12),
            (PackedRouteMicroproofSource::Hash, 128),
        ] {
            let logits = f32_prefix(
                &production.moe.logits,
                vec![MOE_EXPERT_COUNT as u64, n_tokens as u64],
                "production packed GPU route logits view",
            )
            .unwrap();
            let normalized_input = f32_prefix(
                &production.moe.normalized_input,
                vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
                "production packed GPU route normalized view",
            )
            .unwrap();
            let token_ids = i32_prefix(
                &production.token_ids,
                vec![n_tokens as u64],
                "production packed GPU route token view",
            )
            .unwrap();
            let views = PackedMoeViews {
                normalized_input,
                logits,
                hash_ids: None,
            };
            let generation = production.moe.take_gpu_route_generation().unwrap();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let source = match source_kind {
                PackedRouteMicroproofSource::Learned => PackedRouteSource::Learned(&fixture.bias),
                PackedRouteMicroproofSource::Hash => PackedRouteSource::Hash,
            };
            production
                .moe
                .encode_gpu_route_compact(
                    &ctx,
                    &encoder,
                    &views,
                    source,
                    &token_ids,
                    (matches!(source_kind, PackedRouteMicroproofSource::Hash))
                        .then_some(&fixture.token_to_expert),
                    n_tokens,
                    1.5,
                    generation,
                )
                .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none());
            let schedule = production
                .moe
                .capture_gpu_compact_schedule(n_tokens, generation)
                .unwrap();

            let (expected_ids, expected_weights) = fixture.expected_routes(source_kind, n_tokens);
            let (_, compact_rows, compact_slots, expected_buckets) =
                expected_compact_schedule(&expected_ids, n_tokens);
            let mut actual_ids =
                host_read_i32(&production.moe.expert_ids, "production packed route IDs").unwrap();
            let mut actual_weights =
                host_read_f32(&production.moe.weights, "production packed route weights").unwrap();
            actual_ids.truncate(n_tokens * MOE_TOP_K);
            actual_weights.truncate(n_tokens * MOE_TOP_K);
            assert_eq!(actual_ids, expected_ids);
            for (&actual, &expected) in actual_weights.iter().zip(&expected_weights) {
                assert!((actual - expected).abs() <= 1e-4 * expected.abs().max(1.0));
            }

            assert_eq!(schedule.len(), expected_buckets.len());
            for (actual, expected) in schedule.iter().zip(&expected_buckets) {
                assert_eq!(
                    (actual.expert, actual.start, actual.len),
                    (expected.expert, expected.start, expected.len)
                );
            }
            let mut actual_rows = host_read_i32(
                &production.moe.bucket_rows,
                "production packed compact rows",
            )
            .unwrap();
            let mut actual_slots = host_read_i32(
                &production.moe.bucket_slots,
                "production packed compact slots",
            )
            .unwrap();
            actual_rows.truncate(n_tokens * MOE_TOP_K);
            actual_slots.truncate(n_tokens * MOE_TOP_K);
            assert_eq!(actual_rows, compact_rows);
            assert_eq!(actual_slots, compact_slots);
            let expected_tiles32 =
                packed_grouped_expert_tiles(n_tokens, &expected_buckets).unwrap();
            let expected_tiles16 =
                packed_grouped_iq2_mma16_tiles(n_tokens, &expected_buckets).unwrap();
            assert_eq!(
                host_read_i32(
                    &production.moe.grouped_tiles,
                    "production packed compact 32-row tiles",
                )
                .unwrap(),
                padded_tile_words(&expected_tiles32, PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS,)
            );
            assert_eq!(
                host_read_i32(
                    &production.moe.grouped_iq2_mma16_tiles,
                    "production packed compact 16-row tiles",
                )
                .unwrap(),
                padded_tile_words(&expected_tiles16, PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS,)
            );
        }

        production.moe.gpu_route.next_generation.set(u32::MAX);
        assert!(
            production
                .moe
                .take_gpu_route_generation()
                .unwrap_err()
                .to_string()
                .contains("exhausted before wrap")
        );
    }

    #[test]
    fn packed_gpu_route_records_repeat_and_reject_missing_producers() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let fixture = PackedRouteFixture::new(&ctx);
        for (source, n_tokens) in [
            (PackedRouteMicroproofSource::Learned, 12),
            (PackedRouteMicroproofSource::Hash, 128),
        ] {
            let first = fixture.run(&ctx, source, n_tokens, n_tokens, MOE_EXPERT_COUNT);
            let second = fixture.run(&ctx, source, n_tokens, n_tokens, MOE_EXPERT_COUNT);
            assert_packed_route_capture(&fixture, source, n_tokens, &first);
            assert_packed_route_capture(&fixture, source, n_tokens, &second);
            assert_eq!(second.expert_ids, first.expert_ids);
            assert_eq!(
                second
                    .weights
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                first
                    .weights
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(second.counts, first.counts);
            assert_eq!(second.slot_ids, first.slot_ids);
            assert_eq!(second.signature[2], first.signature[2]);
        }

        let n_tokens = 12;
        fixture.run(
            &ctx,
            PackedRouteMicroproofSource::Learned,
            n_tokens,
            n_tokens,
            MOE_EXPERT_COUNT,
        );
        let missing_route = fixture.run(
            &ctx,
            PackedRouteMicroproofSource::Learned,
            n_tokens,
            n_tokens - 1,
            MOE_EXPERT_COUNT,
        );
        assert_packed_route_failure(
            &missing_route,
            n_tokens,
            PACKED_ROUTE_STALE_ROUTE,
            (n_tokens * MOE_TOP_K) as i32,
        );

        let wide_tokens = 512;
        let valid_hash_route = fixture.run(
            &ctx,
            PackedRouteMicroproofSource::Hash,
            wide_tokens,
            wide_tokens,
            MOE_EXPERT_COUNT,
        );
        assert_packed_route_capture(
            &fixture,
            PackedRouteMicroproofSource::Hash,
            wide_tokens,
            &valid_hash_route,
        );
        let missing_hash_route = fixture.run(
            &ctx,
            PackedRouteMicroproofSource::Hash,
            wide_tokens,
            257,
            MOE_EXPERT_COUNT,
        );
        assert_packed_route_failure(
            &missing_hash_route,
            wide_tokens,
            PACKED_ROUTE_STALE_ROUTE,
            (wide_tokens * MOE_TOP_K) as i32,
        );

        let missing_schedule = fixture.run(
            &ctx,
            PackedRouteMicroproofSource::Learned,
            n_tokens,
            n_tokens,
            MOE_EXPERT_COUNT - 1,
        );
        assert_packed_route_failure(
            &missing_schedule,
            n_tokens,
            PACKED_ROUTE_STALE_SCHEDULE,
            missing_schedule.counts[..MOE_EXPERT_COUNT - 1].iter().sum(),
        );

        let valid_before_missing_validator = fixture.run(
            &ctx,
            PackedRouteMicroproofSource::Learned,
            n_tokens,
            n_tokens,
            MOE_EXPERT_COUNT,
        );
        let generation = fixture.generations.take().unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        fixture
            .scratch
            .encode_learned(
                &ctx,
                &encoder,
                &fixture.bias,
                n_tokens,
                n_tokens,
                generation,
            )
            .unwrap();
        fixture
            .scratch
            .encode_schedule(&ctx, &encoder, n_tokens, MOE_EXPERT_COUNT, generation)
            .unwrap();
        fixture
            .scratch
            .encode_signature(&ctx, &encoder, n_tokens, generation)
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());
        let missing_validator = fixture.scratch.capture(n_tokens, generation.get());
        assert_eq!(
            missing_validator.aggregate,
            valid_before_missing_validator.aggregate
        );
        assert_eq!(
            missing_validator.signature,
            vec![
                generation.get() as i32,
                PACKED_ROUTE_INVALID_AGGREGATE,
                0,
                packed_route_signature_completion(generation.get(), n_tokens) as i32,
            ]
        );

        fixture.scratch.assert_slot_guards();

        let concurrent_command = ctx.queue.commandBuffer().unwrap();
        let concurrent = KernelEncoder::begin_concurrent(&concurrent_command);
        let concurrent_generation = fixture.generations.take().unwrap();
        for error in [
            fixture.scratch.encode_learned(
                &ctx,
                &concurrent,
                &fixture.bias,
                n_tokens,
                n_tokens,
                concurrent_generation,
            ),
            fixture.scratch.encode_hash(
                &ctx,
                &concurrent,
                &fixture.token_to_expert,
                n_tokens,
                n_tokens,
                concurrent_generation,
            ),
            fixture.scratch.encode_schedule(
                &ctx,
                &concurrent,
                n_tokens,
                MOE_EXPERT_COUNT,
                concurrent_generation,
            ),
            fixture
                .scratch
                .encode_validate(&ctx, &concurrent, n_tokens, concurrent_generation),
            fixture
                .scratch
                .encode_signature(&ctx, &concurrent, n_tokens, concurrent_generation),
        ] {
            assert!(error.unwrap_err().to_string().contains("ordered serial"));
        }
        concurrent.end();

        let exhausted = PackedRouteGenerationOwner::with_next(u32::MAX);
        let error = exhausted.take().unwrap_err();
        assert!(error.to_string().contains("exhausted before wrap"));
        assert!(
            exhausted
                .take()
                .unwrap_err()
                .to_string()
                .contains("exhausted")
        );
    }

    #[test]
    fn packed_route_authority_rejects_invalid_producers_and_private_state() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let fixture = PackedRouteFixture::new(&ctx);
        let run_pipeline = |source: PackedRouteMicroproofSource,
                            bias: &MetalTensor,
                            token_to_expert: &MetalTensor,
                            n_tokens: usize| {
            let generation = fixture.generations.take().unwrap();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            fixture
                .scratch
                .encode_pipeline(
                    &ctx,
                    &encoder,
                    source,
                    bias,
                    token_to_expert,
                    n_tokens,
                    n_tokens,
                    MOE_EXPERT_COUNT,
                    generation,
                )
                .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none());
            fixture.scratch.capture(n_tokens, generation.get())
        };

        let mut nonfinite_logits = fixture.logits.clone();
        nonfinite_logits[0] = f32::NAN;
        write_raw_f32(&fixture.scratch.logits, &nonfinite_logits);
        let failed = run_pipeline(
            PackedRouteMicroproofSource::Learned,
            &fixture.bias,
            &fixture.token_to_expert,
            1,
        );
        assert_eq!(
            failed.route_status[0],
            DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_LOGIT
        );
        assert_packed_route_failure(&failed, 1, PACKED_ROUTE_FAILED_ROUTE, 0);

        host_write_f32(
            &fixture.scratch.logits,
            &fixture.logits,
            "restore packed route logits",
        )
        .unwrap();
        let mut nonfinite_bias = fixture.bias_values.clone();
        nonfinite_bias[0] = f32::NAN;
        let nonfinite_bias = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&nonfinite_bias),
            vec![MOE_EXPERT_COUNT as u64],
            GgmlType::F32,
        )
        .unwrap();
        let failed = run_pipeline(
            PackedRouteMicroproofSource::Learned,
            &nonfinite_bias,
            &fixture.token_to_expert,
            1,
        );
        assert_eq!(
            failed.route_status[0],
            DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_BIAS
        );
        assert_packed_route_failure(&failed, 1, PACKED_ROUTE_FAILED_ROUTE, 0);

        let mut invalid_tokens = fixture.token_ids.clone();
        invalid_tokens[0] = -1;
        host_write_i32(
            &fixture.scratch.token_ids,
            &invalid_tokens,
            "invalid packed hash token",
        )
        .unwrap();
        let failed = run_pipeline(
            PackedRouteMicroproofSource::Hash,
            &fixture.bias,
            &fixture.token_to_expert,
            1,
        );
        assert_eq!(
            failed.route_status[0],
            DEEPSEEK_V4_ROUTE_STATUS_INVALID_TOKEN
        );
        assert_packed_route_failure(&failed, 1, PACKED_ROUTE_FAILED_ROUTE, 0);
        host_write_i32(
            &fixture.scratch.token_ids,
            &fixture.token_ids,
            "restore packed hash tokens",
        )
        .unwrap();

        for (map, expected_status) in [
            (
                vec![-1, 1, 2, 3, 4, 5],
                DEEPSEEK_V4_ROUTE_STATUS_INVALID_EXPERT,
            ),
            (
                vec![0, 0, 2, 3, 4, 5],
                DEEPSEEK_V4_ROUTE_STATUS_DUPLICATE_EXPERT,
            ),
        ] {
            let map = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&map),
                vec![MOE_TOP_K as u64, 1],
                GgmlType::I32,
            )
            .unwrap();
            let failed = run_pipeline(PackedRouteMicroproofSource::Hash, &fixture.bias, &map, 1);
            assert_eq!(failed.route_status[0], expected_status);
            assert_packed_route_failure(&failed, 1, PACKED_ROUTE_FAILED_ROUTE, 0);
        }

        write_raw_f32(&fixture.scratch.logits, &nonfinite_logits);
        let failed = run_pipeline(
            PackedRouteMicroproofSource::Hash,
            &fixture.bias,
            &fixture.token_to_expert,
            1,
        );
        assert_eq!(
            failed.route_status[0],
            DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_LOGIT
        );
        assert_packed_route_failure(&failed, 1, PACKED_ROUTE_FAILED_ROUTE, 0);
        host_write_f32(
            &fixture.scratch.logits,
            &fixture.logits,
            "restore packed route logits after hash fault",
        )
        .unwrap();

        let write_route_state =
            |generation: NonZeroU32, n_tokens: usize, expert_ids: &[i32], weights: &[f32]| {
                let mut full_ids = vec![-1; PACKED_GPU_ROUTE_MAX_TOKENS * MOE_TOP_K];
                full_ids[..expert_ids.len()].copy_from_slice(expert_ids);
                host_write_i32(&fixture.scratch.expert_ids, &full_ids, "prepared route IDs")
                    .unwrap();
                let mut full_weights = vec![0.0; PACKED_GPU_ROUTE_MAX_TOKENS * MOE_TOP_K];
                full_weights[..weights.len()].copy_from_slice(weights);
                write_raw_f32(&fixture.scratch.weights, &full_weights);
                let mut generations = vec![0; PACKED_GPU_ROUTE_MAX_TOKENS];
                generations[..n_tokens].fill(generation.get() as i32);
                host_write_i32(
                    &fixture.scratch.route_generations,
                    &generations,
                    "prepared route generations",
                )
                .unwrap();
                let mut statuses = vec![0; PACKED_GPU_ROUTE_MAX_TOKENS];
                statuses[..n_tokens].fill(DEEPSEEK_V4_ROUTE_STATUS_READY);
                host_write_i32(
                    &fixture.scratch.route_status,
                    &statuses,
                    "prepared route statuses",
                )
                .unwrap();
            };
        let run_route_state = |n_tokens: usize, expert_ids: &[i32], weights: &[f32]| {
            let generation = fixture.generations.take().unwrap();
            write_route_state(generation, n_tokens, expert_ids, weights);
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            fixture
                .scratch
                .encode_schedule(&ctx, &encoder, n_tokens, MOE_EXPERT_COUNT, generation)
                .unwrap();
            fixture
                .scratch
                .encode_validate(&ctx, &encoder, n_tokens, generation)
                .unwrap();
            fixture
                .scratch
                .encode_signature(&ctx, &encoder, n_tokens, generation)
                .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none());
            fixture.scratch.capture(n_tokens, generation.get())
        };

        let n_tokens = 12;
        let valid = fixture.run(
            &ctx,
            PackedRouteMicroproofSource::Learned,
            n_tokens,
            n_tokens,
            MOE_EXPERT_COUNT,
        );
        let mut invalid_ids = valid.expert_ids.clone();
        invalid_ids[0] = -1;
        let failed = run_route_state(n_tokens, &invalid_ids, &valid.weights);
        assert_packed_route_failure(
            &failed,
            n_tokens,
            PACKED_ROUTE_INVALID_ID,
            (n_tokens * MOE_TOP_K - 1) as i32,
        );

        let mut duplicate_ids = valid.expert_ids.clone();
        duplicate_ids[1] = duplicate_ids[0];
        let failed = run_route_state(n_tokens, &duplicate_ids, &valid.weights);
        assert_packed_route_failure(
            &failed,
            n_tokens,
            PACKED_ROUTE_DUPLICATE_ID,
            (n_tokens * MOE_TOP_K) as i32,
        );

        let mut invalid_weights = valid.weights.clone();
        invalid_weights[0] = f32::NAN;
        let failed = run_route_state(n_tokens, &valid.expert_ids, &invalid_weights);
        assert_packed_route_failure(
            &failed,
            n_tokens,
            PACKED_ROUTE_INVALID_WEIGHT,
            (n_tokens * MOE_TOP_K) as i32,
        );

        let run_schedule_state =
            |counts: &[i32], slot_ids: &[i32], expected_status: i32, expected_total: i32| {
                let generation = fixture.generations.take().unwrap();
                write_route_state(generation, n_tokens, &valid.expert_ids, &valid.weights);
                host_write_i32(&fixture.scratch.counts, counts, "corrupt schedule counts").unwrap();
                let mut full_slots = vec![-1; PACKED_GPU_ROUTE_MAX_TOKENS * MOE_EXPERT_COUNT];
                full_slots[..slot_ids.len()].copy_from_slice(slot_ids);
                host_write_i32(
                    &fixture.scratch.slot_ids,
                    &full_slots,
                    "corrupt schedule slots",
                )
                .unwrap();
                host_write_i32(
                    &fixture.scratch.schedule_generations,
                    &vec![generation.get() as i32; MOE_EXPERT_COUNT],
                    "corrupt schedule generations",
                )
                .unwrap();
                let command = ctx.queue.commandBuffer().unwrap();
                let encoder = KernelEncoder::begin(&command);
                fixture
                    .scratch
                    .encode_validate(&ctx, &encoder, n_tokens, generation)
                    .unwrap();
                fixture
                    .scratch
                    .encode_signature(&ctx, &encoder, n_tokens, generation)
                    .unwrap();
                encoder.end();
                command.commit();
                command.waitUntilCompleted();
                assert!(command.error().is_none());
                let capture = fixture.scratch.capture(n_tokens, generation.get());
                assert_packed_route_failure(&capture, n_tokens, expected_status, expected_total);
            };

        let mut invalid_counts = valid.counts.clone();
        let counted_expert = valid.expert_ids[0] as usize;
        let omitted_count = invalid_counts[counted_expert];
        invalid_counts[counted_expert] = n_tokens as i32 + 1;
        run_schedule_state(
            &invalid_counts,
            &valid.slot_ids,
            PACKED_ROUTE_INVALID_COUNT,
            (n_tokens * MOE_TOP_K) as i32 - omitted_count,
        );
        let mut invalid_slots = valid.slot_ids.clone();
        let occupied = invalid_slots.iter().position(|&slot| slot >= 0).unwrap();
        invalid_slots[occupied] ^= 1;
        run_schedule_state(
            &valid.counts,
            &invalid_slots,
            PACKED_ROUTE_INVALID_SCHEDULE,
            (n_tokens * MOE_TOP_K) as i32,
        );
        let mut invalid_padding = valid.slot_ids.clone();
        let padding = invalid_padding.iter().position(|&slot| slot == -1).unwrap();
        invalid_padding[padding] = -2;
        run_schedule_state(
            &valid.counts,
            &invalid_padding,
            PACKED_ROUTE_INVALID_PADDING,
            (n_tokens * MOE_TOP_K) as i32,
        );

        let terminal = fixture.run(
            &ctx,
            PackedRouteMicroproofSource::Learned,
            PACKED_GPU_ROUTE_MAX_TOKENS,
            PACKED_GPU_ROUTE_MAX_TOKENS,
            MOE_EXPERT_COUNT,
        );
        let terminal_ids = vec![255; PACKED_GPU_ROUTE_MAX_TOKENS * MOE_TOP_K];
        let failed = run_route_state(
            PACKED_GPU_ROUTE_MAX_TOKENS,
            &terminal_ids,
            &terminal.weights,
        );
        assert_packed_route_failure(
            &failed,
            PACKED_GPU_ROUTE_MAX_TOKENS,
            PACKED_ROUTE_INVALID_COUNT,
            0,
        );
        assert_eq!(
            failed.counts[255],
            (PACKED_GPU_ROUTE_MAX_TOKENS * MOE_TOP_K) as i32
        );
        fixture.scratch.assert_slot_guards();
    }

    fn percentile_ms(values: &[f64], percentile: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let rank = (percentile * sorted.len() as f64).ceil() as usize;
        sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
    }

    fn median_ms(values: &[f64]) -> f64 {
        percentile_ms(values, 0.5)
    }

    fn relative_drift(left: f64, right: f64) -> f64 {
        (left - right).abs() / left.min(right)
    }

    #[test]
    #[ignore = "focused exact packed-route profiler; run release with --nocapture"]
    fn profile_exact_packed_gpu_route_and_schedule_packet() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let fixture = PackedRouteFixture::new(&ctx);
        let sample = |n_tokens: usize, stages: usize| {
            assert!((1..=4).contains(&stages));
            let started = std::time::Instant::now();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let mut last_generation = None;
            for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
                let generation = fixture.generations.take().unwrap();
                let source = if layer < 3 {
                    PackedRouteMicroproofSource::Hash
                } else {
                    PackedRouteMicroproofSource::Learned
                };
                match source {
                    PackedRouteMicroproofSource::Learned => fixture
                        .scratch
                        .encode_learned(
                            &ctx,
                            &encoder,
                            &fixture.bias,
                            n_tokens,
                            n_tokens,
                            generation,
                        )
                        .unwrap(),
                    PackedRouteMicroproofSource::Hash => fixture
                        .scratch
                        .encode_hash(
                            &ctx,
                            &encoder,
                            &fixture.token_to_expert,
                            n_tokens,
                            n_tokens,
                            generation,
                        )
                        .unwrap(),
                }
                if stages >= 2 {
                    fixture
                        .scratch
                        .encode_schedule(&ctx, &encoder, n_tokens, MOE_EXPERT_COUNT, generation)
                        .unwrap();
                }
                if stages >= 3 {
                    fixture
                        .scratch
                        .encode_validate(&ctx, &encoder, n_tokens, generation)
                        .unwrap();
                }
                if stages >= 4 {
                    fixture
                        .scratch
                        .encode_signature(&ctx, &encoder, n_tokens, generation)
                        .unwrap();
                }
                last_generation = Some(generation);
            }
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            let wall_ms = started.elapsed().as_secs_f64() * 1e3;
            assert!(command.error().is_none());
            let gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
            let generation = last_generation.unwrap();
            if stages == 4 {
                let signature =
                    host_read_i32(&fixture.scratch.signature, "profile signature").unwrap();
                assert_eq!(signature[0], generation.get() as i32);
                assert_eq!(signature[1], DEEPSEEK_V4_ROUTE_STATUS_READY);
                assert_eq!(
                    signature[3],
                    packed_route_signature_completion(generation.get(), n_tokens) as i32
                );
            }
            (gpu_ms, wall_ms)
        };

        for n_tokens in [12, 128] {
            let warm_started = std::time::Instant::now();
            let mut warm_samples = 0;
            while warm_samples < 96 || warm_started.elapsed() < std::time::Duration::from_secs(1) {
                sample(n_tokens, 4);
                warm_samples += 1;
            }
            for stages in 1..=4 {
                for _ in 0..2 {
                    sample(n_tokens, stages);
                }
                let phase_samples = (0..9).map(|_| sample(n_tokens, stages)).collect::<Vec<_>>();
                let phase_gpu = phase_samples
                    .iter()
                    .map(|sample| sample.0)
                    .collect::<Vec<_>>();
                let phase_wall = phase_samples
                    .iter()
                    .map(|sample| sample.1)
                    .collect::<Vec<_>>();
                eprintln!(
                    "deepseek_v4 packed_route_stage n={n_tokens} stages={stages} gpu_median_ms={:.6} gpu_p95_ms={:.6} wall_median_ms={:.6} wall_p95_ms={:.6}",
                    median_ms(&phase_gpu),
                    percentile_ms(&phase_gpu, 0.95),
                    median_ms(&phase_wall),
                    percentile_ms(&phase_wall, 0.95),
                );
            }
            for _ in 0..32 {
                sample(n_tokens, 4);
            }
            let collect =
                |count: usize| (0..count).map(|_| sample(n_tokens, 4)).collect::<Vec<_>>();
            let control_a = collect(12);
            let candidate = collect(40);
            let control_b = collect(12);
            let split = |samples: &[(f64, f64)]| {
                (
                    samples.iter().map(|sample| sample.0).collect::<Vec<_>>(),
                    samples.iter().map(|sample| sample.1).collect::<Vec<_>>(),
                )
            };
            let (control_a_gpu, control_a_wall) = split(&control_a);
            let (candidate_gpu, candidate_wall) = split(&candidate);
            let (control_b_gpu, control_b_wall) = split(&control_b);
            let gpu_drift = relative_drift(median_ms(&control_a_gpu), median_ms(&control_b_gpu));
            let wall_drift = relative_drift(median_ms(&control_a_wall), median_ms(&control_b_wall));
            let gpu_p95 = percentile_ms(&candidate_gpu, 0.95);
            let wall_p95 = percentile_ms(&candidate_wall, 0.95);
            eprintln!(
                "deepseek_v4 packed_route_packet n={n_tokens} candidate_gpu_ms={candidate_gpu:?} candidate_wall_ms={candidate_wall:?} control_a_gpu_ms={control_a_gpu:?} control_a_wall_ms={control_a_wall:?} control_b_gpu_ms={control_b_gpu:?} control_b_wall_ms={control_b_wall:?} gpu_p95_ms={gpu_p95:.6} wall_p95_ms={wall_p95:.6} gpu_control_drift={gpu_drift:.6} wall_control_drift={wall_drift:.6}"
            );
            let ceiling_ms = if n_tokens == 12 { 4.3 } else { 8.6 };
            assert!(gpu_p95 <= ceiling_ms, "GPU p95 {gpu_p95} > {ceiling_ms}");
            assert!(wall_p95 <= ceiling_ms, "wall p95 {wall_p95} > {ceiling_ms}");
            assert!(gpu_drift <= 0.05, "GPU control drift {gpu_drift}");
            assert!(wall_drift <= 0.05, "wall control drift {wall_drift}");
        }
    }

    #[test]
    fn packed_sparse_visibility_tracks_publication_cadence() {
        assert_eq!(
            packed_sparse_visible_counts(2_051, 0, 5, 514).unwrap(),
            [513, 513, 513, 513, 514]
        );
        assert_eq!(
            packed_sparse_visible_counts(2_047, 4, 9, 514).unwrap(),
            [513, 513, 513, 513, 514]
        );
        assert!(
            packed_sparse_visible_counts(2_051, 0, 5, 515)
                .unwrap_err()
                .to_string()
                .contains("differs from published row count 515")
        );
        assert!(
            packed_sparse_visible_counts(2_051, 5, 5, 514)
                .unwrap_err()
                .to_string()
                .contains("visibility geometry is invalid")
        );
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[test]
    fn fp4_selection_counterfactual_rejects_multi_query_chunks_before_encoding() {
        validate_fp4_selection_counterfactual_packed(0, 128).unwrap();
        validate_fp4_selection_counterfactual_packed(2_048, 4).unwrap();
        let error = validate_fp4_selection_counterfactual_packed(2_048, 5).unwrap_err();
        assert!(error.to_string().contains("got 2"), "{error}");
        let error = validate_fp4_selection_counterfactual_packed(2_052, 2).unwrap_err();
        assert!(error.to_string().contains("got 2"), "{error}");
    }

    #[test]
    fn tiled_hca_offset_starts_at_the_513th_visible_row() {
        assert_eq!(tiled_hca_query_offset(65_535, 1), None);
        assert_eq!(tiled_hca_query_offset(65_536, 127), None);
        assert_eq!(tiled_hca_query_offset(65_536, 128), Some(127));
        assert_eq!(tiled_hca_query_offset(65_662, 2), Some(1));
        assert_eq!(tiled_hca_query_offset(65_663, 1), Some(0));
        assert_eq!(tiled_hca_query_offset(1_048_448, 128), Some(0));
    }

    #[test]
    fn packed_raw_chunk_publication_matches_ordered_ring_updates() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const HEAD_DIM: usize = 16;

        fn read_f16_bits(tensor: &MetalTensor) -> Vec<u16> {
            assert_eq!(tensor.dtype, GgmlType::F16);
            unsafe {
                let pointer = tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(tensor.offset as usize)
                    .cast::<u16>();
                std::slice::from_raw_parts(pointer, tensor.n_elements() as usize).to_vec()
            }
        }

        for (start_position, n_tokens) in [
            (0_u32, 1_usize),
            (123, 12),
            (257, 127),
            (511, 128),
            (511, DEEPSEEK_V4_PREFILL_MAX_TOKENS),
        ] {
            let source_values = (0..n_tokens * HEAD_DIM)
                .map(|index| ((index * 29 + index / 7 + 3) % 197) as f32 * 0.0031 - 0.29)
                .collect::<Vec<_>>();
            let initial_ring = (0..DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM)
                .map(|index| half::f16::from_f32((index % 113) as f32 * 0.001 - 0.04).to_bits())
                .collect::<Vec<_>>();
            let source = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&source_values),
                vec![HEAD_DIM as u64, n_tokens as u64],
                GgmlType::F32,
            )
            .unwrap();
            let chunk =
                MetalTensor::zeros_f16(&ctx, vec![HEAD_DIM as u64, n_tokens as u64]).unwrap();
            let ring = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&initial_ring),
                vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
                GgmlType::F16,
            )
            .unwrap();
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            encode_publish_raw_chunk_f16(
                &ctx,
                &encoder,
                &source,
                &chunk,
                &ring,
                start_position,
                n_tokens,
                HEAD_DIM,
            )
            .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none());

            let expected_chunk = source_values
                .iter()
                .map(|&value| half::f16::from_f32(value).to_bits())
                .collect::<Vec<_>>();
            let mut expected_ring = initial_ring;
            for row in 0..n_tokens {
                let slot = (start_position as usize + row) % DEEPSEEK_V4_LOCAL_WINDOW;
                let source = &expected_chunk[row * HEAD_DIM..(row + 1) * HEAD_DIM];
                expected_ring[slot * HEAD_DIM..(slot + 1) * HEAD_DIM].copy_from_slice(source);
            }
            assert_eq!(read_f16_bits(&chunk), expected_chunk);
            assert_eq!(read_f16_bits(&ring), expected_ring);
        }
    }

    #[test]
    fn packed_dense_attention_matches_ordered_singleton_rows_within_roundoff() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let n_tokens = 8;
        let config = deepseek_v4_session_attention_config();
        let dims = config.checked().unwrap();
        let queries = (0..n_tokens * dims.query_width)
            .map(|index| ((index * 17 + index / 11) % 257) as f32 * 0.0007 - 0.08)
            .collect::<Vec<_>>();
        let raw = (0..DEEPSEEK_V4_LOCAL_WINDOW * config.head_dim)
            .map(|index| ((index * 13 + 5) % 193) as f32 * 0.0011 - 0.09)
            .collect::<Vec<_>>();
        let compressed = (0..DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS * config.head_dim)
            .map(|index| ((index * 19 + 3) % 211) as f32 * 0.0009 - 0.085)
            .collect::<Vec<_>>();
        let sinks = (0..config.head_count)
            .map(|head| head as f32 * 0.013 - 0.31)
            .collect::<Vec<_>>();
        let queries = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&queries),
            vec![dims.query_width as u64, n_tokens as u64],
            GgmlType::F32,
        )
        .unwrap();
        let raw_bits = raw
            .iter()
            .map(|&value| half::f16::from_f32(value).to_bits())
            .collect::<Vec<_>>();
        let raw = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&raw_bits),
            vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            GgmlType::F16,
        )
        .unwrap();
        let raw_chunk = raw.view_subrange(0, vec![config.head_dim as u64, n_tokens as u64]);
        let raw_before = MetalTensor::zeros_f16(
            &ctx,
            vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        )
        .unwrap();
        let compressed_bits = compressed
            .iter()
            .map(|&value| half::f16::from_f32(value).to_bits())
            .collect::<Vec<_>>();
        let compressed = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&compressed_bits),
            vec![
                config.head_dim as u64,
                DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64,
            ],
            GgmlType::F16,
        )
        .unwrap();
        let sinks = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&sinks),
            vec![config.head_count as u64],
            GgmlType::F32,
        )
        .unwrap();
        let packed =
            MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, n_tokens as u64]).unwrap();
        let ordered =
            MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, n_tokens as u64]).unwrap();
        let overlapping_raw_storage = MetalTensor::zeros_f16(
            &ctx,
            vec![
                config.head_dim as u64,
                (DEEPSEEK_V4_LOCAL_WINDOW + 1) as u64,
            ],
        )
        .unwrap();
        let overlapping_raw =
            overlapping_raw_storage.view_subrange(0, vec![config.head_dim as u64, n_tokens as u64]);
        let overlapping_raw_before = overlapping_raw_storage.view_subrange(
            config.head_dim as u64,
            vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        );
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let empty = encode_cooperative_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &queries,
            &raw_chunk,
            &raw_before,
            DeepSeekV4RawCacheLayout::Chunk,
            None,
            &sinks,
            &packed,
            AttentionKind::SlidingWindow,
            0,
            0,
            config,
        )
        .unwrap_err();
        assert!(empty.to_string().contains("requires at least one token"));
        let oversized = encode_cooperative_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &queries,
            &raw_chunk,
            &raw_before,
            DeepSeekV4RawCacheLayout::Chunk,
            None,
            &sinks,
            &packed,
            AttentionKind::SlidingWindow,
            0,
            DEEPSEEK_V4_PREFILL_MAX_TOKENS + 1,
            config,
        )
        .unwrap_err();
        assert!(
            oversized
                .to_string()
                .contains("exceeds retained chunk limit")
        );
        let aliased = encode_cooperative_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &queries,
            &raw_chunk,
            &raw,
            DeepSeekV4RawCacheLayout::Chunk,
            Some(DeepSeekV4PublishedRows {
                cache: &compressed,
                count: n_tokens / 4,
                capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
            }),
            &sinks,
            &packed,
            AttentionKind::CompressedSparse,
            0,
            n_tokens,
            config,
        )
        .unwrap_err();
        assert!(
            aliased
                .to_string()
                .contains("requires disjoint current and preserved raw caches")
        );
        let partially_aliased = encode_cooperative_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &queries,
            &overlapping_raw,
            &overlapping_raw_before,
            DeepSeekV4RawCacheLayout::Chunk,
            Some(DeepSeekV4PublishedRows {
                cache: &compressed,
                count: n_tokens / 4,
                capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
            }),
            &sinks,
            &packed,
            AttentionKind::CompressedSparse,
            0,
            n_tokens,
            config,
        )
        .unwrap_err();
        assert!(
            partially_aliased
                .to_string()
                .contains("requires disjoint current and preserved raw caches")
        );
        encode_packed_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &queries,
            &raw_chunk,
            &raw_before,
            Some(DeepSeekV4PublishedRows {
                cache: &compressed,
                count: n_tokens / 4,
                capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
            }),
            &sinks,
            &packed,
            AttentionKind::CompressedSparse,
            0,
            n_tokens,
        )
        .unwrap();
        for row in 0..n_tokens {
            let query = f32_row(
                &queries,
                row,
                dims.query_width,
                vec![config.head_dim as u64, config.head_count as u64],
                "ordered query",
            )
            .unwrap();
            let output = f32_row(
                &ordered,
                row,
                dims.query_width,
                vec![config.head_dim as u64, config.head_count as u64],
                "ordered output",
            )
            .unwrap();
            let count = (row + 1) / 4;
            encode_dense_sink_attention_f16(
                &ctx,
                &encoder,
                &query,
                &raw,
                (count > 0).then_some(DeepSeekV4PublishedRows {
                    cache: &compressed,
                    count,
                    capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
                }),
                &sinks,
                &output,
                row as u32,
                config,
            )
            .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());
        let packed = host_read_f32(&packed, "packed attention").unwrap();
        let ordered = host_read_f32(&ordered, "ordered attention").unwrap();
        assert_eq!(packed.len(), ordered.len());
        for (index, (&packed, &ordered)) in packed.iter().zip(&ordered).enumerate() {
            let allowed = 2.0 * f32::EPSILON * ordered.abs().max(1.0);
            assert!(
                (packed - ordered).abs() <= allowed,
                "packed attention differs at {index}: {packed} vs {ordered}, allowed {allowed}"
            );
        }
    }

    #[test]
    fn packed_hca_splits_the_first_tiled_query_without_future_raw_leakage() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const START_POSITION: usize = 65_662;
        const N_TOKENS: usize = 2;
        const CAPACITY: usize = 768;
        let config = deepseek_v4_session_attention_config();
        let dims = config.checked().unwrap();
        let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
        let query_values = (0..N_TOKENS * dims.query_width)
            .map(|index| {
                let token = index / dims.query_width;
                let within = index % dims.query_width;
                let head = within / config.head_dim;
                let dimension = within % config.head_dim;
                let tag = (token * 31 + head * 17 + dimension * 7) % 149;
                (tag as f32 - 74.0) * 0.0011
            })
            .collect::<Vec<_>>();
        let future_row = query_values[..config.head_dim]
            .iter()
            .map(|value| round_f16(value * 512.0))
            .collect::<Vec<_>>();
        let raw_value = |position: usize, dimension: usize| {
            let tag = (position * 23 + dimension * 11 + position / 13) % 137;
            round_f16((tag as f32 - 68.0) * 0.0013)
        };
        let mut raw_before = vec![0.0f32; DEEPSEEK_V4_LOCAL_WINDOW * config.head_dim];
        for position in START_POSITION - DEEPSEEK_V4_LOCAL_WINDOW..START_POSITION {
            let slot = position % DEEPSEEK_V4_LOCAL_WINDOW;
            for dimension in 0..config.head_dim {
                raw_before[slot * config.head_dim + dimension] = raw_value(position, dimension);
            }
        }
        let future_row_values = &future_row;
        let raw_current = (0..N_TOKENS)
            .flat_map(|token| {
                (0..config.head_dim).map(move |dimension| {
                    if token == 1 {
                        future_row_values[dimension]
                    } else {
                        raw_value(START_POSITION + token, dimension)
                    }
                })
            })
            .collect::<Vec<_>>();
        let mut compressed_values = (0..CAPACITY * config.head_dim)
            .map(|index| {
                let row = index / config.head_dim;
                let dimension = index % config.head_dim;
                let tag = (row * 43 + dimension * 5 + row / 7) % 151;
                round_f16((tag as f32 - 75.0) * 0.0012)
            })
            .collect::<Vec<_>>();
        for dimension in 0..config.head_dim {
            compressed_values[512 * config.head_dim + dimension] =
                round_f16(query_values[dimension] * 640.0);
        }
        let sinks_values = (0..config.head_count)
            .map(|head| head as f32 * 0.003 - 0.27)
            .collect::<Vec<_>>();
        let mut expected = Vec::with_capacity(N_TOKENS * dims.query_width);
        for token in 0..N_TOKENS {
            let position = START_POSITION + token;
            let raw_start = position + 1 - DEEPSEEK_V4_LOCAL_WINDOW;
            let mut raw_rows = Vec::with_capacity(DEEPSEEK_V4_LOCAL_WINDOW * config.head_dim);
            for logical_position in raw_start..=position {
                if logical_position == START_POSITION + 1 {
                    raw_rows.extend_from_slice(&future_row);
                } else {
                    raw_rows.extend(
                        (0..config.head_dim)
                            .map(|dimension| raw_value(logical_position, dimension)),
                    );
                }
            }
            let count = (position + 1) / 128;
            expected.extend(
                crate::deepseek_v4_oracle::shared_kv_attention(
                    &query_values[token * dims.query_width..(token + 1) * dims.query_width],
                    config.head_count,
                    config.head_dim,
                    &raw_rows,
                    &compressed_values[..count * config.head_dim],
                    None,
                    &sinks_values,
                )
                .unwrap(),
            );
        }
        let first_raw_start = START_POSITION + 1 - DEEPSEEK_V4_LOCAL_WINDOW;
        let mut leaked_raw = Vec::with_capacity(DEEPSEEK_V4_LOCAL_WINDOW * config.head_dim);
        for logical_position in first_raw_start..=START_POSITION {
            if logical_position == first_raw_start {
                leaked_raw.extend_from_slice(&future_row);
            } else {
                leaked_raw.extend(
                    (0..config.head_dim).map(|dimension| raw_value(logical_position, dimension)),
                );
            }
        }
        let leaked = crate::deepseek_v4_oracle::shared_kv_attention(
            &query_values[..dims.query_width],
            config.head_count,
            config.head_dim,
            &leaked_raw,
            &compressed_values[..DEEPSEEK_V4_CSA_TOP_K * config.head_dim],
            None,
            &sinks_values,
        )
        .unwrap();
        assert!(
            expected[..dims.query_width]
                .iter()
                .zip(leaked)
                .any(|(correct, leaked)| (correct - leaked).abs() > 1e-3)
        );

        let queries = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&query_values),
            vec![dims.query_width as u64, N_TOKENS as u64],
            GgmlType::F32,
        )
        .unwrap();
        let make_raw = |values: &[f32], rows: usize| {
            let bits = values
                .iter()
                .map(|value| half::f16::from_f32(*value).to_bits())
                .collect::<Vec<_>>();
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&bits),
                vec![config.head_dim as u64, rows as u64],
                GgmlType::F16,
            )
            .unwrap()
        };
        let raw_before = make_raw(&raw_before, DEEPSEEK_V4_LOCAL_WINDOW);
        let raw_current = make_raw(&raw_current, N_TOKENS);
        let compressed_bits = compressed_values
            .iter()
            .map(|value| half::f16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        let compressed = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&compressed_bits),
            vec![config.head_dim as u64, CAPACITY as u64],
            GgmlType::F16,
        )
        .unwrap();
        let sinks = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&sinks_values),
            vec![config.head_count as u64],
            GgmlType::F32,
        )
        .unwrap();
        let output =
            MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, N_TOKENS as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_packed_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &queries,
            &raw_current,
            &raw_before,
            Some(DeepSeekV4PublishedRows {
                cache: &compressed,
                count: 513,
                capacity_rows: CAPACITY,
            }),
            &sinks,
            &output,
            AttentionKind::HeavilyCompressed,
            START_POSITION as u32,
            N_TOKENS,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "packed HCA split command failed: {:?}",
            command.error()
        );
        let actual = host_read_f32(&output, "packed HCA split output").unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            let allowed = 8e-5 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= allowed,
                "packed HCA split output[{index}]={actual}, expected {expected}, allowed {allowed}"
            );
        }
    }

    #[test]
    fn packed_sparse_suffix_matches_cpu_with_original_chunk_ring_visibility() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let config = deepseek_v4_session_attention_config();
        let dims = config.checked().unwrap();
        let start_position = 1_540_u32;
        let n_tokens = 512;
        let query_offset = n_tokens - 1;
        let query_count = n_tokens - query_offset;
        let raw_value = |position: usize, dimension: usize| {
            let tag = (position * 31 + dimension * 17 + position / 11) % 181;
            (tag as f32 - 90.0) * 0.0017
        };
        let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
        let mut prior_ring = vec![0.0; DEEPSEEK_V4_LOCAL_WINDOW * config.head_dim];
        for position in 0..start_position as usize {
            let slot = position % DEEPSEEK_V4_LOCAL_WINDOW;
            for dimension in 0..config.head_dim {
                prior_ring[slot * config.head_dim + dimension] =
                    round_f16(raw_value(position, dimension));
            }
        }
        let prior_bits = prior_ring
            .iter()
            .map(|&value| half::f16::from_f32(value).to_bits())
            .collect::<Vec<_>>();
        let raw_cache = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&prior_bits),
            vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            GgmlType::F16,
        )
        .unwrap();
        let raw_cache_before_chunk = MetalTensor::zeros_f16(
            &ctx,
            vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        )
        .unwrap();
        let raw_chunk =
            MetalTensor::zeros_f16(&ctx, vec![config.head_dim as u64, n_tokens as u64]).unwrap();
        let new_raw = (start_position as usize..start_position as usize + n_tokens)
            .flat_map(|position| {
                (0..config.head_dim).map(move |dimension| raw_value(position, dimension))
            })
            .collect::<Vec<_>>();
        let new_raw_tensor = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&new_raw),
            vec![config.head_dim as u64, n_tokens as u64],
            GgmlType::F32,
        )
        .unwrap();
        let query_values = (0..n_tokens * dims.query_width)
            .map(|index| {
                let token = index / dims.query_width;
                let within = index % dims.query_width;
                let head = within / config.head_dim;
                let dimension = within % config.head_dim;
                let tag = (token * 23 + head * 13 + dimension * 7) % 173;
                (tag as f32 - 86.0) * 0.0013
            })
            .collect::<Vec<_>>();
        let queries = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&query_values),
            vec![dims.query_width as u64, n_tokens as u64],
            GgmlType::F32,
        )
        .unwrap();
        let compressed_values = (0..DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS * config.head_dim)
            .map(|index| {
                let row = index / config.head_dim;
                let dimension = index % config.head_dim;
                let tag = (row * 43 + dimension * 5 + row / 7) % 191;
                round_f16((tag as f32 - 95.0) * 0.0015)
            })
            .collect::<Vec<_>>();
        let compressed_bits = compressed_values
            .iter()
            .map(|&value| half::f16::from_f32(value).to_bits())
            .collect::<Vec<_>>();
        let compressed = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&compressed_bits),
            vec![
                config.head_dim as u64,
                DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64,
            ],
            GgmlType::F16,
        )
        .unwrap();
        let indexer = MetalTensor::zeros_f16(
            &ctx,
            vec![
                INDEXER_HEAD_DIM as u64,
                DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64,
            ],
        )
        .unwrap();
        let selected_ids = (1..=DEEPSEEK_V4_CSA_TOP_K as i32).collect::<Vec<_>>();
        let selected_ids = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&selected_ids),
            vec![DEEPSEEK_V4_CSA_TOP_K as u64, query_count as u64],
            GgmlType::I32,
        )
        .unwrap();
        let selected_counts = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&vec![DEEPSEEK_V4_CSA_TOP_K as i32; query_count]),
            vec![query_count as u64],
            GgmlType::I32,
        )
        .unwrap();
        let visible_counts = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&vec![513_i32; query_count]),
            vec![query_count as u64],
            GgmlType::I32,
        )
        .unwrap();
        let sinks_values = (0..config.head_count)
            .map(|head| head as f32 * 0.007 - 0.23)
            .collect::<Vec<_>>();
        let sinks = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&sinks_values),
            vec![config.head_count as u64],
            GgmlType::F32,
        )
        .unwrap();
        let output =
            MetalTensor::zeros_f32(&ctx, vec![dims.query_width as u64, n_tokens as u64]).unwrap();
        let rows = DeepSeekV4CsaRows {
            attention_cache: &compressed,
            indexer_cache: &indexer,
            #[cfg(feature = "dsv4-diagnostics")]
            indexer_fp4_sidecar: None,
            count: 513,
            capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
        };
        let sparse = PackedSparseCsaViews {
            query_offset,
            query_count,
            cache_order_ids: selected_ids,
            selected_counts,
            visible_counts,
            index_queries: MetalTensor::zeros_f32(&ctx, vec![128, 64, query_count as u64]).unwrap(),
            head_weights: MetalTensor::zeros_f32(&ctx, vec![64, query_count as u64]).unwrap(),
            scores: MetalTensor::zeros_f32(
                &ctx,
                vec![
                    DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64,
                    query_count as u64,
                ],
            )
            .unwrap(),
            selected_mask: MetalTensor::zeros_i32(
                &ctx,
                vec![
                    DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64,
                    query_count as u64,
                ],
            )
            .unwrap(),
            status: MetalTensor::zeros_i32(&ctx, vec![query_count as u64]).unwrap(),
        };

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_copy_raw_ring_f16_bits(&ctx, &encoder, &raw_cache, &raw_cache_before_chunk).unwrap();
        encode_publish_raw_chunk_f16(
            &ctx,
            &encoder,
            &new_raw_tensor,
            &raw_chunk,
            &raw_cache,
            start_position,
            n_tokens,
            config.head_dim,
        )
        .unwrap();
        let dense_queries = f32_prefix(
            &queries,
            vec![dims.query_width as u64, query_offset as u64],
            "packed sparse dense-prefix queries",
        )
        .unwrap();
        let dense_output = f32_prefix(
            &output,
            vec![dims.query_width as u64, query_offset as u64],
            "packed sparse dense-prefix output",
        )
        .unwrap();
        encode_packed_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &dense_queries,
            &raw_chunk,
            &raw_cache_before_chunk,
            Some(DeepSeekV4PublishedRows {
                cache: &compressed,
                count: DEEPSEEK_V4_CSA_TOP_K,
                capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
            }),
            &sinks,
            &dense_output,
            AttentionKind::CompressedSparse,
            start_position,
            query_offset,
        )
        .unwrap();
        encode_packed_selected_sink_attention_f16(
            &ctx,
            &encoder,
            &queries,
            &raw_chunk,
            &raw_cache_before_chunk,
            rows,
            sparse.selection_view(),
            &sinks,
            &output,
            start_position,
            n_tokens,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "packed sparse attention failed: {:?}",
            command.error()
        );

        let actual = host_read_f32(&output, "packed sparse attention").unwrap();
        for token in [0, 127, 128, 255, 384, query_offset] {
            let position = start_position as usize + token;
            let raw_start = position + 1 - DEEPSEEK_V4_LOCAL_WINDOW;
            let raw_rows = (raw_start..=position)
                .flat_map(|logical_position| {
                    (0..config.head_dim)
                        .map(move |dimension| round_f16(raw_value(logical_position, dimension)))
                })
                .collect::<Vec<_>>();
            let compressed_count = (position + 1) / 4;
            let mask = (token >= query_offset).then(|| {
                let local = token - query_offset;
                let mut mask = vec![false; compressed_count];
                if local == 0 {
                    mask[1..=DEEPSEEK_V4_CSA_TOP_K].fill(true);
                } else {
                    mask[..DEEPSEEK_V4_CSA_TOP_K].fill(true);
                }
                mask
            });
            let expected = crate::deepseek_v4_oracle::shared_kv_attention(
                &query_values[token * dims.query_width..(token + 1) * dims.query_width],
                config.head_count,
                config.head_dim,
                &raw_rows,
                &compressed_values[..compressed_count * config.head_dim],
                mask.as_deref(),
                &sinks_values,
            )
            .unwrap();
            let actual = &actual[token * dims.query_width..(token + 1) * dims.query_width];
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                let allowed = 8e-5 * expected.abs().max(1.0);
                assert!(
                    (actual - expected).abs() <= allowed,
                    "packed sparse token {token} differs at {index}: {actual} vs {expected}, allowed {allowed}"
                );
            }
        }
    }

    #[test]
    fn retained_packed_attention_preserves_ring_and_absolute_visibility() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };

        fn run_case(
            ctx: &MetalContext,
            kind: AttentionKind,
            start_position: u32,
            n_tokens: usize,
            checked_tokens: Option<&[usize]>,
        ) {
            let config = deepseek_v4_session_attention_config();
            let dims = config.checked().unwrap();
            let ratio = match kind {
                AttentionKind::SlidingWindow => 0,
                AttentionKind::CompressedSparse => 4,
                AttentionKind::HeavilyCompressed => 128,
            };
            let end_position = start_position as usize + n_tokens;
            let raw_value = |position: usize, dimension: usize| {
                let tag = (position * 29 + dimension * 11 + position / 7) % 137;
                (tag as f32 - 68.0) * 0.0027
                    + if (position + dimension).is_multiple_of(31) {
                        0.043
                    } else {
                        -0.009
                    }
            };
            let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
            let mut prior_ring = vec![0.0; DEEPSEEK_V4_LOCAL_WINDOW * config.head_dim];
            for position in 0..start_position as usize {
                let slot = position % DEEPSEEK_V4_LOCAL_WINDOW;
                for dimension in 0..config.head_dim {
                    prior_ring[slot * config.head_dim + dimension] =
                        round_f16(raw_value(position, dimension));
                }
            }
            let prior_bits = prior_ring
                .iter()
                .map(|&value| half::f16::from_f32(value).to_bits())
                .collect::<Vec<_>>();
            let raw_cache = MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&prior_bits),
                vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
                GgmlType::F16,
            )
            .unwrap();
            let raw_cache_before_chunk = MetalTensor::zeros_f16(
                ctx,
                vec![config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            )
            .unwrap();
            let raw_chunk =
                MetalTensor::zeros_f16(ctx, vec![config.head_dim as u64, n_tokens as u64]).unwrap();
            let new_raw = (start_position as usize..end_position)
                .flat_map(|position| {
                    (0..config.head_dim).map(move |dimension| raw_value(position, dimension))
                })
                .collect::<Vec<_>>();
            let new_raw = MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&new_raw),
                vec![config.head_dim as u64, n_tokens as u64],
                GgmlType::F32,
            )
            .unwrap();
            let queries_values = (0..n_tokens * dims.query_width)
                .map(|index| {
                    let token = index / dims.query_width;
                    let within = index % dims.query_width;
                    let head = within / config.head_dim;
                    let dimension = within % config.head_dim;
                    let tag =
                        (start_position as usize * 13 + token * 17 + head * 19 + dimension * 5)
                            % 149;
                    (tag as f32 - 74.0) * 0.0019
                })
                .collect::<Vec<_>>();
            let queries = MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&queries_values),
                vec![dims.query_width as u64, n_tokens as u64],
                GgmlType::F32,
            )
            .unwrap();
            let compressed_values = (0..DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS * config.head_dim)
                .map(|index| {
                    let row = index / config.head_dim;
                    let dimension = index % config.head_dim;
                    let tag = (row * 37 + dimension * 7 + row / 3) % 139;
                    round_f16((tag as f32 - 69.0) * 0.0023 - 0.007)
                })
                .collect::<Vec<_>>();
            let compressed_bits = compressed_values
                .iter()
                .map(|&value| half::f16::from_f32(value).to_bits())
                .collect::<Vec<_>>();
            let compressed = MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&compressed_bits),
                vec![
                    config.head_dim as u64,
                    DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64,
                ],
                GgmlType::F16,
            )
            .unwrap();
            let sinks_values = (0..config.head_count)
                .map(|head| head as f32 * 0.009 - 0.27)
                .collect::<Vec<_>>();
            let sinks = MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&sinks_values),
                vec![config.head_count as u64],
                GgmlType::F32,
            )
            .unwrap();
            let output =
                MetalTensor::zeros_f32(ctx, vec![dims.query_width as u64, n_tokens as u64])
                    .unwrap();
            let final_compressed_count = end_position.checked_div(ratio).unwrap_or(0);
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            encode_copy_raw_ring_f16_bits(ctx, &encoder, &raw_cache, &raw_cache_before_chunk)
                .unwrap();
            encode_publish_raw_chunk_f16(
                ctx,
                &encoder,
                &new_raw,
                &raw_chunk,
                &raw_cache,
                start_position,
                n_tokens,
                config.head_dim,
            )
            .unwrap();
            encode_packed_dense_sink_attention_f16(
                ctx,
                &encoder,
                &queries,
                &raw_chunk,
                &raw_cache_before_chunk,
                (final_compressed_count > 0).then_some(DeepSeekV4PublishedRows {
                    cache: &compressed,
                    count: final_compressed_count,
                    capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
                }),
                &sinks,
                &output,
                kind,
                start_position,
                n_tokens,
            )
            .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(
                command.error().is_none(),
                "retained {kind:?} command failed: {:?}",
                command.error()
            );

            let actual = host_read_f32(&output, "retained packed attention").unwrap();
            let all_tokens = (0..n_tokens).collect::<Vec<_>>();
            for &token in checked_tokens.unwrap_or(&all_tokens) {
                assert!(token < n_tokens);
                let position = start_position as usize + token;
                let raw_start = (position + 1).saturating_sub(DEEPSEEK_V4_LOCAL_WINDOW);
                let raw_rows = (raw_start..=position)
                    .flat_map(|logical_position| {
                        (0..config.head_dim)
                            .map(move |dimension| round_f16(raw_value(logical_position, dimension)))
                    })
                    .collect::<Vec<_>>();
                let compressed_count = (position + 1).checked_div(ratio).unwrap_or(0);
                let expected = crate::deepseek_v4_oracle::shared_kv_attention(
                    &queries_values[token * dims.query_width..(token + 1) * dims.query_width],
                    config.head_count,
                    config.head_dim,
                    &raw_rows,
                    &compressed_values[..compressed_count * config.head_dim],
                    None,
                    &sinks_values,
                )
                .unwrap();
                let actual = &actual[token * dims.query_width..(token + 1) * dims.query_width];
                for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                    let allowed = 8e-5 * expected.abs().max(1.0);
                    assert!(
                        (actual - expected).abs() <= allowed,
                        "retained {kind:?} token {token} attention differs at {index}: {actual} vs {expected}, allowed {allowed}"
                    );
                }
            }
        }

        run_case(&ctx, AttentionKind::SlidingWindow, 127, 4, None);
        run_case(&ctx, AttentionKind::CompressedSparse, 125, 8, None);
        run_case(&ctx, AttentionKind::HeavilyCompressed, 125, 8, None);
        const WIDE_CHUNK_TOKENS: usize = 512;

        run_case(
            &ctx,
            AttentionKind::CompressedSparse,
            128,
            128,
            Some(&[0, 1, 63, 127]),
        );
        run_case(&ctx, AttentionKind::CompressedSparse, 1_020, 4, None);
        run_case(
            &ctx,
            AttentionKind::CompressedSparse,
            2_044,
            4,
            Some(&[0, 3]),
        );
        run_case(
            &ctx,
            AttentionKind::SlidingWindow,
            129,
            WIDE_CHUNK_TOKENS,
            Some(&[0, 127, 128, 255, 384, 511]),
        );
        run_case(
            &ctx,
            AttentionKind::HeavilyCompressed,
            65_152,
            WIDE_CHUNK_TOKENS,
            Some(&[0, 383, 384, 510, 511]),
        );
    }

    #[test]
    fn q8_token_axis_gemv_is_bitwise_singleton_equivalent() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        const N_IN: usize = 64;
        const N_OUT: usize = 7;
        const N_TOKENS: usize = 4;
        let mut weight_bytes = Vec::with_capacity(N_OUT * (N_IN / 32) * 34);
        for row in 0..N_OUT {
            for block in 0..N_IN / 32 {
                let scale = half::f16::from_f32(0.0075 + row as f32 * 0.0003);
                weight_bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
                for index in 0..32 {
                    let quant = ((row * 19 + block * 11 + index * 7) % 101) as i8 - 50;
                    weight_bytes.push(quant as u8);
                }
            }
        }
        let inputs = (0..N_TOKENS * N_IN)
            .map(|index| ((index * 13 + index / 9) % 89) as f32 * 0.013 - 0.51)
            .collect::<Vec<_>>();
        let weight = MetalTensor::from_bytes(
            &ctx,
            &weight_bytes,
            vec![N_IN as u64, N_OUT as u64],
            GgmlType::Q8_0,
        )
        .unwrap();
        let inputs = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&inputs),
            vec![N_IN as u64, N_TOKENS as u64],
            GgmlType::F32,
        )
        .unwrap();
        let packed = MetalTensor::zeros_f32(&ctx, vec![N_OUT as u64, N_TOKENS as u64]).unwrap();
        let singleton = MetalTensor::zeros_f32(&ctx, vec![N_OUT as u64, N_TOKENS as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::encode_mat_vec_q8_0_batch_f32(
            &ctx, &encoder, &weight, &inputs, &packed, N_IN, N_OUT, N_TOKENS,
        )
        .unwrap();
        for token in 0..N_TOKENS {
            let input = f32_row(
                &inputs,
                token,
                N_IN,
                vec![N_IN as u64],
                "singleton Q8 input",
            )
            .unwrap();
            let output = f32_row(
                &singleton,
                token,
                N_OUT,
                vec![N_OUT as u64],
                "singleton Q8 output",
            )
            .unwrap();
            crate::metal::encode_mat_vec_q8_0_f32(
                &ctx, &encoder, &weight, &input, &output, N_IN, N_OUT,
            )
            .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());
        let packed = host_read_f32(&packed, "packed Q8 output").unwrap();
        let singleton = host_read_f32(&singleton, "singleton Q8 output").unwrap();
        assert_eq!(
            packed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            singleton
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}
