use super::*;
use objc2_metal::MTLComputePipelineState;

pub const DEEPSEEK_V4_PREFILL_MAX_TOKENS: usize = 128;

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
}

impl DeepSeekV4PrefillScratch {
    pub(super) fn new(ctx: &MetalContext) -> Result<Self, DeepSeekV4MetalError> {
        let n = DEEPSEEK_V4_PREFILL_MAX_TOKENS as u64;
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
                    index_queries: MetalTensor::zeros_f32(
                        ctx,
                        vec![INDEXER_HEAD_DIM as u64, INDEXER_HEAD_COUNT as u64, n],
                    )?,
                    head_weights: MetalTensor::zeros_f32(ctx, vec![INDEXER_HEAD_COUNT as u64, n])?,
                    visible_counts: MetalTensor::zeros_i32(ctx, vec![n])?,
                    scores: MetalTensor::zeros_f32(
                        ctx,
                        vec![DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64, n],
                    )?,
                    selected_mask: MetalTensor::zeros_i32(
                        ctx,
                        vec![DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS as u64, n],
                    )?,
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
            },
        })
    }
}

pub(super) fn append_session_allocation_requests(
    requests: &mut Vec<DeepSeekV4SessionAllocationRequest>,
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
            checked_mul(
                n,
                DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
                "packed sparse row scratch",
            )?,
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

struct PackedSparseCsaViews {
    query_offset: usize,
    query_count: usize,
    cache_order_ids: MetalTensor,
    selected_counts: MetalTensor,
    visible_counts: MetalTensor,
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

impl PrefillSparseCsaScratch {
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
        require_serial(enc, "deepseek_v4_packed_sparse_csa_indexer")?;
        checked_token_count(n_tokens)?;
        if query_offset >= n_tokens
            || rows.count <= DEEPSEEK_V4_CSA_TOP_K
            || rows.count > rows.capacity_rows
            || rows.capacity_rows != DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS
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
        let q_lora_suffix = q_lora.view_subrange(
            checked_mul(query_offset, 1_024, "packed sparse Q-LoRA offset")? as u64,
            vec![1_024, query_count as u64],
        );
        let normalized_suffix = normalized_input.view_subrange(
            checked_mul(
                query_offset,
                DEEPSEEK_V4_HIDDEN_SIZE,
                "packed sparse normalized-input offset",
            )? as u64,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, query_count as u64],
        );
        let index_queries = f32_prefix(
            &self.index_queries,
            vec![
                INDEXER_HEAD_DIM as u64,
                INDEXER_HEAD_COUNT as u64,
                query_count as u64,
            ],
            "packed sparse index queries",
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

        let visible = (0..query_count)
            .map(|local| {
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
                let count = csa_visible_rows(position);
                if count <= DEEPSEEK_V4_CSA_TOP_K || count > rows.count {
                    return invalid(format!(
                        "packed sparse token {token} sees {count} rows outside 513..={} final rows",
                        rows.count
                    ));
                }
                i32::try_from(count).map_err(|_| {
                    DeepSeekV4MetalError::Invalid("packed sparse visible count exceeds i32".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if visible.last().copied() != Some(rows.count as i32) {
            return invalid(format!(
                "packed sparse final visibility {:?} differs from published row count {}",
                visible.last(),
                rows.count
            ));
        }
        host_write_i32(
            &visible_counts,
            &visible,
            "packed sparse CSA visible counts",
        )?;

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
                    DeepSeekV4MetalError::Invalid("packed sparse token offset exceeds u32".into())
                })?)
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid("packed sparse position overflow".into())
                })?;
            encode_ds4_rope_tail_adjacent_in_place(ctx, enc, &query, position, rope, false)?;
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
        encode_lightning_indexer_scores_f16(
            ctx,
            enc,
            &index_queries,
            &head_weights,
            rows.indexer_cache,
            &visible_counts,
            &scores,
            INDEXER_HEAD_COUNT,
            INDEXER_HEAD_DIM,
            rows.capacity_rows,
            query_count,
        )?;
        encode_select_top_k_f32(
            ctx,
            enc,
            &scores,
            &visible_counts,
            &selected_mask,
            None,
            &cache_order_ids,
            &selected_counts,
            &status,
            rows.capacity_rows,
            DEEPSEEK_V4_CSA_TOP_K,
            query_count,
        )?;
        Ok(PackedSparseCsaViews {
            query_offset,
            query_count,
            cache_order_ids,
            selected_counts,
            visible_counts,
        })
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
    fn encode_layer_projections(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residency: &DeepSeekV4MetalResidency,
        layer: usize,
        normalized_input: &MetalTensor,
        n_tokens: usize,
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
                    encode_state_batch_projection(
                        ctx,
                        enc,
                        weight,
                        normalized_input,
                        output,
                        DEEPSEEK_V4_HIDDEN_SIZE,
                        width,
                        n_tokens,
                        name,
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
                encode_state_batch_projection(
                    ctx,
                    enc,
                    tensor("attn_compressor_kv.weight")?,
                    normalized_input,
                    &attention_kv,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    512,
                    n_tokens,
                    "packed HCA compressor KV",
                )?;
                encode_state_batch_projection(
                    ctx,
                    enc,
                    tensor("attn_compressor_gate.weight")?,
                    normalized_input,
                    &attention_score,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    512,
                    n_tokens,
                    "packed HCA compressor score",
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

enum PackedRouteSource<'a> {
    Hash,
    Learned(&'a MetalTensor),
}

impl PrefillMoeScratch {
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
        enc: &KernelEncoder,
        normalized_input: &MetalTensor,
        schedule: &[ExpertBucket],
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        shared_gate: &MetalTensor,
        shared_up: &MetalTensor,
        shared_down: &MetalTensor,
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
        let expert_outputs = f32_prefix(
            &self.expert_outputs,
            vec![
                DEEPSEEK_V4_HIDDEN_SIZE as u64,
                MOE_TOP_K as u64,
                n_tokens as u64,
            ],
            "packed expert outputs",
        )?;
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
            encode_ds4_clamped_swiglu(ctx, enc, &gate_flat, &up_flat, &inner_flat, expert_clamp)?;
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
    let config = deepseek_v4_session_attention_config();
    let dims = config.checked()?;
    checked_token_count(n_tokens)?;
    validate_f32(
        queries,
        &[dims.query_width as u64, n_tokens as u64],
        false,
        "packed attention queries",
    )?;
    validate_f16(
        raw_cache,
        &[config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        false,
        "packed raw cache",
    )?;
    validate_f16(
        raw_cache_before_chunk,
        &[config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        false,
        "packed preserved raw cache",
    )?;
    validate_f32(
        sinks,
        &[config.head_count as u64],
        false,
        "packed attention sinks",
    )?;
    validate_f32(
        output,
        &[dims.query_width as u64, n_tokens as u64],
        true,
        "packed attention output",
    )?;
    let ratio = match kind {
        AttentionKind::SlidingWindow => 0,
        AttentionKind::CompressedSparse => 4,
        AttentionKind::HeavilyCompressed => 128,
    };
    let end_position = start_position
        .checked_add(checked_token_count(n_tokens)?)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid("packed position overflow".into()))?;
    let expected_rows = if ratio == 0 {
        0
    } else {
        end_position as usize / ratio
    };
    let compressed_cache = match (expected_rows, compressed) {
        (0, None) => raw_cache,
        (0, Some(rows)) if rows.count == 0 => rows.cache,
        (expected, Some(rows)) if rows.count == expected => {
            validate_f16(
                rows.cache,
                &[config.head_dim as u64, rows.capacity_rows as u64],
                false,
                "packed compressed cache",
            )?;
            if expected > DEEPSEEK_V4_CSA_TOP_K || expected > rows.capacity_rows {
                return invalid(format!(
                    "packed dense attention cannot consume {expected} rows from capacity {}",
                    rows.capacity_rows
                ));
            }
            rows.cache
        }
        (expected, rows) => {
            return invalid(format!(
                "packed attention expected {expected} compressed rows, got {}",
                rows.map_or(0, |rows| rows.count)
            ));
        }
    };
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        n_tokens: u32,
        compression_ratio: u32,
        start_position: u32,
        window: u32,
        scale: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_packed_dense_sink_attention_f16")?;
    let maximum_rows = DEEPSEEK_V4_LOCAL_WINDOW + DEEPSEEK_V4_CSA_TOP_K;
    let threadgroup_width = config.head_dim.max(maximum_rows);
    if pso.maxTotalThreadsPerThreadgroup() < threadgroup_width {
        return invalid(format!(
            "packed attention pipeline supports {} threads, requires {}",
            pso.maxTotalThreadsPerThreadgroup(),
            threadgroup_width
        ));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: config.head_count as u32,
            head_dim: config.head_dim as u32,
            n_tokens: n_tokens as u32,
            compression_ratio: ratio as u32,
            start_position,
            window: DEEPSEEK_V4_LOCAL_WINDOW as u32,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, raw_cache);
    enc.set_tensor(3, raw_cache_before_chunk);
    enc.set_tensor(4, compressed_cache);
    enc.set_tensor(5, sinks);
    enc.set_tensor(6, output);
    enc.set_threadgroup_memory(0, (maximum_rows + 1) * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_tokens,
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
fn encode_packed_selected_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    rows: DeepSeekV4CsaRows<'_>,
    sparse: &PackedSparseCsaViews,
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
        || rows.capacity_rows != DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS
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
    for (tensor, name) in [
        (raw_cache, "packed selected raw cache"),
        (
            raw_cache_before_chunk,
            "packed selected preserved raw cache",
        ),
    ] {
        validate_f16(
            tensor,
            &[config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            false,
            name,
        )?;
    }
    validate_f16(
        rows.attention_cache,
        &[config.head_dim as u64, rows.capacity_rows as u64],
        false,
        "packed selected compressed cache",
    )?;
    validate_i32(
        &sparse.cache_order_ids,
        &[DEEPSEEK_V4_CSA_TOP_K as u64, sparse.query_count as u64],
        false,
        "packed selected cache-order IDs",
    )?;
    for (tensor, name) in [
        (&sparse.selected_counts, "packed selected row counts"),
        (&sparse.visible_counts, "packed selected visible counts"),
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
        scale: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_packed_selected_sink_attention_f16")?;
    let maximum_rows = DEEPSEEK_V4_LOCAL_WINDOW + DEEPSEEK_V4_CSA_TOP_K;
    let threadgroup_width = config.head_dim.max(maximum_rows);
    if pso.maxTotalThreadsPerThreadgroup() < threadgroup_width {
        return invalid(format!(
            "packed selected attention pipeline supports {} threads, requires {}",
            pso.maxTotalThreadsPerThreadgroup(),
            threadgroup_width
        ));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: u32::try_from(config.head_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed selected head count exceeds u32".into())
            })?,
            head_dim: u32::try_from(config.head_dim).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed selected head dimension exceeds u32".into())
            })?,
            query_count: u32::try_from(sparse.query_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed selected query count exceeds u32".into())
            })?,
            query_token_offset: u32::try_from(sparse.query_offset).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed selected query offset exceeds u32".into())
            })?,
            chunk_start_position: start_position,
            window: u32::try_from(DEEPSEEK_V4_LOCAL_WINDOW).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed selected window exceeds u32".into())
            })?,
            selected_slots: u32::try_from(DEEPSEEK_V4_CSA_TOP_K).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed selected slot count exceeds u32".into())
            })?,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, raw_cache);
    enc.set_tensor(3, raw_cache_before_chunk);
    enc.set_tensor(4, rows.attention_cache);
    enc.set_tensor(5, &sparse.cache_order_ids);
    enc.set_tensor(6, &sparse.selected_counts);
    enc.set_tensor(7, &sparse.visible_counts);
    enc.set_tensor(8, sinks);
    enc.set_tensor(9, output);
    enc.set_threadgroup_memory(0, (maximum_rows + 1) * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: sparse.query_count,
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

    fn execute_packed_tokens_with_progress(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        layer_completed: &mut impl FnMut(usize),
    ) -> Result<(), DeepSeekV4MetalError> {
        if ctx.device.registryID() != self.device_registry_id {
            return invalid(format!(
                "DeepSeek V4 session belongs to Metal device registry {}, got {}",
                self.device_registry_id,
                ctx.device.registryID()
            ));
        }
        let n_tokens = checked_token_count(token_ids.len())?;
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
            validate_promoted_session_position(position)?;
        }
        self.validate_committed_token_append(start_position, token_ids.len())?;
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

        let begun_position = self.phase.begin_mutation()?;
        debug_assert_eq!(begun_position, start_position);
        let result = self.prefill_tokens_inner(
            ctx,
            token_ids,
            &token_view,
            start_position,
            emit_logits,
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
        &self,
        ctx: &MetalContext,
        token_ids: &[u32],
        token_view: &MetalTensor,
        start_position: u32,
        emit_logits: bool,
        layer_completed: &mut impl FnMut(usize),
    ) -> Result<(), DeepSeekV4MetalError> {
        let n_tokens = token_ids.len();
        let last_position = start_position
            .checked_add(u32::try_from(n_tokens - 1).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed token count exceeds u32".into())
            })?)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("packed position overflow".into()))?;
        let rms_eps = self.residency.config().attention_rms_epsilon;
        let hc_eps = self.residency.config().hyper_connection_epsilon;
        let attention_config = deepseek_v4_session_attention_config();
        let attention_dims = attention_config.checked()?;
        let trace_layers = std::env::var_os("QWEN_DSV4_PREFILL_TRACE").is_some();
        let mut router_total = 0.0_f64;
        let mut route_total = 0.0_f64;
        let mut expert_total = 0.0_f64;
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
            let router_started = std::time::Instant::now();
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
            let encoder = KernelEncoder::begin(&command);
            let router_result = (|| {
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
                )?;
                let compressor = self.prefill.compressor.encode_layer_projections(
                    ctx,
                    &encoder,
                    &self.residency,
                    layer,
                    &attention.normalized_input,
                    n_tokens,
                )?;
                for row in 0..n_tokens {
                    let position = start_position
                        .checked_add(u32::try_from(row).map_err(|_| {
                            DeepSeekV4MetalError::Invalid("packed row exceeds u32".into())
                        })?)
                        .ok_or_else(|| {
                            DeepSeekV4MetalError::Invalid("packed position overflow".into())
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
                    encode_scatter_offset_f32_to_f16(
                        ctx,
                        &encoder,
                        &kv,
                        &raw_cache,
                        (position as usize % DEEPSEEK_V4_LOCAL_WINDOW) * attention_config.head_dim,
                        attention_config.head_dim,
                    )?;
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
                            &raw_cache,
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
                    encode_packed_selected_sink_attention_f16(
                        ctx,
                        &encoder,
                        &attention.queries,
                        &raw_cache,
                        &self.prefill.attention.raw_cache_before_chunk,
                        rows,
                        &sparse,
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
                        &raw_cache,
                        &self.prefill.attention.raw_cache_before_chunk,
                        compressed,
                        self.layer_tensor(layer, "attn_sinks.weight")?,
                        &attention.attention,
                        attention_kind,
                        start_position,
                        n_tokens,
                    )?;
                }
                for row in 0..n_tokens {
                    let position = start_position
                        .checked_add(u32::try_from(row).map_err(|_| {
                            DeepSeekV4MetalError::Invalid("packed row exceeds u32".into())
                        })?)
                        .ok_or_else(|| {
                            DeepSeekV4MetalError::Invalid("packed position overflow".into())
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
                let attention_output = self.prefill.attention.encode_output(
                    ctx,
                    &encoder,
                    &attention.attention,
                    self.layer_tensor(layer, "attn_output_a.weight")?,
                    self.layer_tensor(layer, "attn_output_b.weight")?,
                    n_tokens,
                )?;
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
                self.prefill.moe.encode_router(
                    ctx,
                    &encoder,
                    &ffn_input,
                    self.layer_tensor(layer, "ffn_norm.weight")?,
                    self.layer_tensor(layer, "ffn_gate_inp.weight")?,
                    token_view,
                    hash_map,
                    n_tokens,
                    rms_eps,
                )
            })();
            encoder.end();
            let moe_views = router_result?;
            command.commit();
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                return invalid(format!(
                    "packed layer {layer} router command failed: {error:?}"
                ));
            }
            if let Some(query_offset) = sparse_query_offset {
                self.prefill
                    .attention
                    .sparse_csa
                    .validate_completed(n_tokens - query_offset)?;
            }
            let router_seconds = router_started.elapsed().as_secs_f64();
            router_total += router_seconds;

            let route_started = std::time::Instant::now();
            let source = if layer < self.residency.config().hash_layer_count as usize {
                PackedRouteSource::Hash
            } else {
                PackedRouteSource::Learned(self.layer_tensor(layer, "exp_probs_b.bias")?)
            };
            let schedule = self.prefill.moe.route(
                &moe_views,
                source,
                n_tokens,
                self.residency.config().expert_weights_scale,
            )?;
            let route_seconds = route_started.elapsed().as_secs_f64();
            route_total += route_seconds;

            let expert_started = std::time::Instant::now();
            let command = ctx.queue.commandBuffer().ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "failed to allocate packed layer {layer} expert command buffer"
                ))
            })?;
            let encoder = KernelEncoder::begin(&command);
            let expert_result = (|| {
                let moe_output = self.prefill.moe.encode_experts(
                    ctx,
                    &encoder,
                    &moe_views.normalized_input,
                    &schedule,
                    self.layer_tensor(layer, "ffn_gate_exps.weight")?,
                    self.layer_tensor(layer, "ffn_up_exps.weight")?,
                    self.layer_tensor(layer, "ffn_down_exps.weight")?,
                    self.layer_tensor(layer, "ffn_gate_shexp.weight")?,
                    self.layer_tensor(layer, "ffn_up_shexp.weight")?,
                    self.layer_tensor(layer, "ffn_down_shexp.weight")?,
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
            command.commit();
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                return invalid(format!(
                    "packed layer {layer} expert command failed: {error:?}"
                ));
            }
            let expert_seconds = expert_started.elapsed().as_secs_f64();
            expert_total += expert_seconds;
            if trace_layers {
                eprintln!(
                    "deepseek_v4 packed layer={layer} router={router_seconds:.4}s route={route_seconds:.4}s experts={expert_seconds:.4}s buckets={}",
                    schedule.len()
                );
            }
            layer_completed(layer);
        }
        if trace_layers {
            eprintln!(
                "deepseek_v4 packed totals router={router_total:.3}s route={route_total:.3}s experts={expert_total:.3}s"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_packed_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &queries,
            &raw,
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
    fn packed_sparse_suffix_matches_cpu_with_original_chunk_ring_visibility() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let config = deepseek_v4_session_attention_config();
        let dims = config.checked().unwrap();
        let start_position = 2_048_u32;
        let n_tokens = 5;
        let query_offset = 3;
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
        let selected_ids = (1..=DEEPSEEK_V4_CSA_TOP_K as i32)
            .chain(0..DEEPSEEK_V4_CSA_TOP_K as i32)
            .collect::<Vec<_>>();
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
            count: 513,
            capacity_rows: DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
        };
        let sparse = PackedSparseCsaViews {
            query_offset,
            query_count,
            cache_order_ids: selected_ids,
            selected_counts,
            visible_counts,
        };

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_copy_raw_ring_f16_bits(&ctx, &encoder, &raw_cache, &raw_cache_before_chunk).unwrap();
        for token in 0..n_tokens {
            let source = f32_row(
                &new_raw_tensor,
                token,
                config.head_dim,
                vec![config.head_dim as u64],
                "packed sparse raw row",
            )
            .unwrap();
            let position = start_position as usize + token;
            encode_scatter_offset_f32_to_f16(
                &ctx,
                &encoder,
                &source,
                &raw_cache,
                (position % DEEPSEEK_V4_LOCAL_WINDOW) * config.head_dim,
                config.head_dim,
            )
            .unwrap();
        }
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
            &raw_cache,
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
            &raw_cache,
            &raw_cache_before_chunk,
            rows,
            &sparse,
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
        for token in 0..n_tokens {
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
                    for row in 1..=DEEPSEEK_V4_CSA_TOP_K {
                        mask[row] = true;
                    }
                } else {
                    for row in 0..DEEPSEEK_V4_CSA_TOP_K {
                        mask[row] = true;
                    }
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
            let final_compressed_count = if ratio == 0 { 0 } else { end_position / ratio };
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            encode_copy_raw_ring_f16_bits(ctx, &encoder, &raw_cache, &raw_cache_before_chunk)
                .unwrap();
            for token in 0..n_tokens {
                let position = start_position as usize + token;
                let source = f32_row(
                    &new_raw,
                    token,
                    config.head_dim,
                    vec![config.head_dim as u64],
                    "retained packed raw row",
                )
                .unwrap();
                encode_scatter_offset_f32_to_f16(
                    ctx,
                    &encoder,
                    &source,
                    &raw_cache,
                    (position % DEEPSEEK_V4_LOCAL_WINDOW) * config.head_dim,
                    config.head_dim,
                )
                .unwrap();
            }
            encode_packed_dense_sink_attention_f16(
                ctx,
                &encoder,
                &queries,
                &raw_cache,
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
                let compressed_count = if ratio == 0 {
                    0
                } else {
                    (position + 1) / ratio
                };
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
