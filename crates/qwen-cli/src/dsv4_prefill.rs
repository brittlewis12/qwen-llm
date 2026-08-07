use anyhow::{Context, Result, ensure};
use clap::Args;
use objc2_metal::MTLDevice;
use qwen_llm::deepseek_v4::AttentionKind;
use qwen_llm::deepseek_v4_metal::{
    DEEPSEEK_V4_PREFILL_DEFAULT_TOKENS, DEEPSEEK_V4_PREFILL_MAX_TOKENS, DeepSeekV4MetalResidency,
    DeepSeekV4Session, PackedChunkProfile, PackedPostRouteStageKind, PackedPrefillStageKind,
};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use qwen_llm::tokenizer::Tokenizer;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Args, Debug)]
pub struct Dsv4PrefillArgs {
    /// Path to the first shard of a DeepSeek V4 Flash-0731 GGUF.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Total prompt tokens in the profiled request.
    #[arg(short = 'p', long, default_value_t = 2_048)]
    tokens: usize,
    /// Production packed-chunk size used to traverse the prompt.
    #[arg(long, default_value_t = DEEPSEEK_V4_PREFILL_DEFAULT_TOKENS)]
    chunk_tokens: usize,
    /// Larger chunk size whose optimistic boundary ceiling is reported.
    #[arg(long, default_value_t = DEEPSEEK_V4_PREFILL_DEFAULT_TOKENS * 2)]
    candidate_chunk_tokens: usize,
    /// Optional raw prompt text. The first --tokens independently encoded IDs are used.
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Unsampled production-policy passes before acquisition.
    #[arg(long, default_value_t = 1)]
    warmups: usize,
    /// Sampled production-policy passes to report.
    #[arg(long, default_value_t = 1)]
    samples: usize,
    /// Optional path for the complete per-chunk and per-layer JSON report.
    #[arg(long)]
    json_out: Option<PathBuf>,
    /// Emit the complete report to stdout instead of the compact summary.
    #[arg(long)]
    full_json: bool,
}

#[derive(Serialize)]
struct Dsv4PrefillReport {
    schema_version: u32,
    build: Value,
    model: String,
    device: String,
    token_source: String,
    n_tokens: usize,
    chunk_tokens: usize,
    candidate_chunk_tokens: usize,
    load_ms: f64,
    ordinary_reference_wall_ms: f64,
    logits_bit_exact: bool,
    logits_sha256: String,
    warmups: Vec<Dsv4PrefillOrdinaryRun>,
    ordinary: Dsv4PrefillOrdinaryRun,
    samples: Vec<Dsv4PrefillRun>,
}

#[derive(Serialize)]
struct Dsv4PrefillOrdinaryRun {
    wall_ms: f64,
    chunk_wall_ms: Vec<f64>,
}

#[derive(Serialize)]
struct Dsv4PrefillRun {
    sampled: bool,
    wall_ms: f64,
    chunk_count: usize,
    pre_expert_gpu_ms: f64,
    post_route_gpu_ms: f64,
    non_gpu_residual_ms: f64,
    q8_compressor_matrix_invocations: u32,
    bm16_gpu_ms: f64,
    optimistic_boundary_ceiling_ms: f64,
    optimistic_boundary_ceiling_percent: f64,
    chunks: Vec<Dsv4PrefillChunk>,
}

#[derive(Serialize)]
struct Dsv4PrefillSummary<'a> {
    schema_version: u32,
    report_kind: &'static str,
    full_report_schema_version: u32,
    build: &'a Value,
    model: &'a str,
    device: &'a str,
    token_source: &'a str,
    n_tokens: usize,
    chunk_tokens: usize,
    candidate_chunk_tokens: usize,
    load_ms: f64,
    ordinary_wall_ms: f64,
    ordinary_prefill_tps: f64,
    ordinary_chunk_wall_ms: &'a [f64],
    logits_bit_exact: bool,
    logits_sha256: &'a str,
    warmup_wall_ms: Vec<f64>,
    samples: Vec<Dsv4PrefillSampleSummary>,
}

#[derive(Serialize)]
struct Dsv4PrefillSampleSummary {
    wall_ms: f64,
    prefill_tps: f64,
    chunk_count: usize,
    pre_expert_gpu_ms: f64,
    post_route_gpu_ms: f64,
    non_gpu_residual_ms: f64,
    q8_compressor_matrix_invocations: u32,
    bm16_gpu_ms: f64,
    optimistic_boundary_ceiling_ms: f64,
    optimistic_boundary_ceiling_percent: f64,
    pre_expert_stage_ms: BTreeMap<&'static str, f64>,
    attention_body_stage_ms: BTreeMap<&'static str, f64>,
    sparse_indexer_stage_ms: BTreeMap<&'static str, f64>,
    post_route_stage_ms: BTreeMap<&'static str, f64>,
    bm16_stage_ms: BTreeMap<&'static str, f64>,
}

#[derive(Serialize)]
struct Dsv4PrefillChunk {
    chunk_index: usize,
    token_start: usize,
    token_count: usize,
    emit_logits: bool,
    sampled: bool,
    wall_ms: f64,
    pre_expert_gpu_ms: f64,
    post_route_gpu_ms: f64,
    non_gpu_residual_ms: f64,
    q8_compressor_matrix_invocations: u32,
    bm16_gpu_ms: f64,
    pre_expert_encoder_gap_ms: f64,
    pre_expert_encoder_overlap_ms: f64,
    post_route_encoder_gap_ms: f64,
    post_route_encoder_overlap_ms: f64,
    pre_expert_stage_ms: BTreeMap<&'static str, f64>,
    attention_kind_stage_ms: BTreeMap<&'static str, BTreeMap<&'static str, f64>>,
    attention_body_stage_ms: BTreeMap<&'static str, f64>,
    attention_kind_body_stage_ms: BTreeMap<&'static str, BTreeMap<&'static str, f64>>,
    sparse_indexer_stage_ms: BTreeMap<&'static str, f64>,
    attention_kind_sparse_indexer_stage_ms: BTreeMap<&'static str, BTreeMap<&'static str, f64>>,
    post_route_stage_ms: BTreeMap<&'static str, f64>,
    bm16_stage_ms: BTreeMap<&'static str, f64>,
    pre_expert_layers: Vec<Dsv4PrefillPreExpertLayer>,
    layers: Vec<Dsv4PrefillLayer>,
}

#[derive(Serialize)]
struct Dsv4PrefillPreExpertLayer {
    layer: usize,
    attention_kind: &'static str,
    command_gpu_ms: f64,
    encoder_gap_ms: f64,
    encoder_overlap_ms: f64,
    raw_coverage: f64,
    stage_ms: BTreeMap<&'static str, f64>,
    attention_body_stage_ms: BTreeMap<&'static str, f64>,
    sparse_indexer_stage_ms: BTreeMap<&'static str, f64>,
}

#[derive(Serialize)]
struct Dsv4PrefillLayer {
    layer: usize,
    gate_dtype: String,
    up_dtype: String,
    down_dtype: String,
    grouped_iq2: bool,
    grouped_iq3: bool,
    bm16: bool,
    bucket_count: usize,
    active_experts: usize,
    max_routes_per_expert: u16,
    route_count: usize,
    route_tiles16: usize,
    route_tiles32: usize,
    route_tile16_occupancy: f64,
    route_tile32_occupancy: f64,
    command_gpu_ms: f64,
    encoder_gap_ms: f64,
    encoder_overlap_ms: f64,
    raw_coverage: f64,
    stage_ms: BTreeMap<&'static str, f64>,
}

fn pre_expert_stage_label(kind: PackedPrefillStageKind) -> &'static str {
    match kind {
        PackedPrefillStageKind::BeforeAttentionBody => "before_attention_body",
        PackedPrefillStageKind::SparseIndexerPrepare
        | PackedPrefillStageKind::SparseIndexerScore
        | PackedPrefillStageKind::SparseSelection
        | PackedPrefillStageKind::AttentionCore
        | PackedPrefillStageKind::InverseRope => "attention_body",
        PackedPrefillStageKind::AttentionOutputProjections => "attention_output_projections",
        PackedPrefillStageKind::AfterAttentionOutput => "after_attention_output",
    }
}

fn attention_body_stage_label(kind: PackedPrefillStageKind) -> Option<&'static str> {
    match kind {
        PackedPrefillStageKind::SparseIndexerPrepare
        | PackedPrefillStageKind::SparseIndexerScore
        | PackedPrefillStageKind::SparseSelection => Some("sparse_indexer_and_selection"),
        PackedPrefillStageKind::AttentionCore => Some("attention_core"),
        PackedPrefillStageKind::InverseRope => Some("inverse_rope"),
        PackedPrefillStageKind::BeforeAttentionBody
        | PackedPrefillStageKind::AttentionOutputProjections
        | PackedPrefillStageKind::AfterAttentionOutput => None,
    }
}

fn sparse_indexer_stage_label(kind: PackedPrefillStageKind) -> Option<&'static str> {
    match kind {
        PackedPrefillStageKind::SparseIndexerPrepare => Some("prepare"),
        PackedPrefillStageKind::SparseIndexerScore => Some("score"),
        PackedPrefillStageKind::SparseSelection => Some("selection"),
        PackedPrefillStageKind::BeforeAttentionBody
        | PackedPrefillStageKind::AttentionCore
        | PackedPrefillStageKind::InverseRope
        | PackedPrefillStageKind::AttentionOutputProjections
        | PackedPrefillStageKind::AfterAttentionOutput => None,
    }
}

fn attention_kind_label(kind: AttentionKind) -> &'static str {
    match kind {
        AttentionKind::SlidingWindow => "sliding_window",
        AttentionKind::CompressedSparse => "compressed_sparse",
        AttentionKind::HeavilyCompressed => "heavily_compressed",
    }
}

fn post_route_stage_label(kind: PackedPostRouteStageKind) -> &'static str {
    match kind {
        PackedPostRouteStageKind::RoutedExperts => "routed_experts",
        PackedPostRouteStageKind::RoutedGateUp => "routed_gate_up",
        PackedPostRouteStageKind::RoutedSwiGlu => "routed_swiglu",
        PackedPostRouteStageKind::RoutedDown => "routed_down",
        PackedPostRouteStageKind::SharedExpert => "shared_expert",
        PackedPostRouteStageKind::ExpertCombine => "expert_combine",
        PackedPostRouteStageKind::HyperPostAndHead => "hyper_post_and_head",
    }
}

fn add_stage(map: &mut BTreeMap<&'static str, f64>, label: &'static str, value: f64) {
    *map.entry(label).or_default() += value;
}

fn summarize_chunk(
    profile: PackedChunkProfile,
    chunk_index: usize,
    token_start: usize,
    token_count: usize,
    emit_logits: bool,
    wall_ms: f64,
    attention_kinds: &[AttentionKind],
) -> Result<Dsv4PrefillChunk> {
    let PackedChunkProfile {
        pre_expert,
        post_route,
    } = profile;
    ensure!(
        pre_expert.sampled == post_route.sampled,
        "pre-expert and post-route profiler modes differ"
    );
    ensure!(
        pre_expert.command_gpu_ms.len() == post_route.command_gpu_ms.len()
            && post_route.command_gpu_ms.len() == post_route.metadata.len(),
        "packed chunk profiler layer vectors differ"
    );
    if post_route.sampled {
        ensure!(
            pre_expert.sampled_layers.len() == pre_expert.command_gpu_ms.len()
                && post_route.sampled_layers.len() == post_route.command_gpu_ms.len(),
            "sampled packed chunk profiler omitted layers"
        );
    }
    ensure!(
        attention_kinds.len() == pre_expert.command_gpu_ms.len(),
        "packed chunk attention-kind and profiler layer counts differ"
    );

    let pre_expert_gpu_ms = pre_expert.command_gpu_ms.iter().sum();
    let post_route_gpu_ms = post_route.command_gpu_ms.iter().sum();
    let bm16_gpu_ms = post_route
        .metadata
        .iter()
        .zip(&post_route.command_gpu_ms)
        .filter_map(|(metadata, &duration)| metadata.bm16.then_some(duration))
        .sum();
    let mut pre_expert_stage_ms = BTreeMap::new();
    let mut attention_kind_stage_ms = BTreeMap::new();
    let mut attention_body_stage_ms = BTreeMap::new();
    let mut attention_kind_body_stage_ms = BTreeMap::new();
    let mut sparse_indexer_stage_ms = BTreeMap::new();
    let mut attention_kind_sparse_indexer_stage_ms = BTreeMap::new();
    let mut post_route_stage_ms = BTreeMap::new();
    let mut bm16_stage_ms = BTreeMap::new();
    let mut pre_expert_encoder_gap_ms = 0.0;
    let mut pre_expert_encoder_overlap_ms = 0.0;
    let mut pre_expert_layers = Vec::with_capacity(pre_expert.sampled_layers.len());
    for sampled in &pre_expert.sampled_layers {
        let attention_kind = *attention_kinds
            .get(sampled.layer)
            .context("sampled pre-expert layer exceeds attention schedule")?;
        let attention_kind = attention_kind_label(attention_kind);
        let mut layer_stages = BTreeMap::new();
        let mut layer_attention_body_stages = BTreeMap::new();
        let mut layer_sparse_indexer_stages = BTreeMap::new();
        pre_expert_encoder_gap_ms += sampled.encoder_gap_ms_scaled;
        pre_expert_encoder_overlap_ms += sampled.encoder_overlap_ms_scaled;
        for stage in &sampled.stages {
            let label = pre_expert_stage_label(stage.kind);
            add_stage(&mut layer_stages, label, stage.duration_ms_scaled);
            add_stage(&mut pre_expert_stage_ms, label, stage.duration_ms_scaled);
            add_stage(
                attention_kind_stage_ms
                    .entry(attention_kind)
                    .or_insert_with(BTreeMap::new),
                label,
                stage.duration_ms_scaled,
            );
            if let Some(body_label) = attention_body_stage_label(stage.kind) {
                add_stage(
                    &mut layer_attention_body_stages,
                    body_label,
                    stage.duration_ms_scaled,
                );
                add_stage(
                    &mut attention_body_stage_ms,
                    body_label,
                    stage.duration_ms_scaled,
                );
                add_stage(
                    attention_kind_body_stage_ms
                        .entry(attention_kind)
                        .or_insert_with(BTreeMap::new),
                    body_label,
                    stage.duration_ms_scaled,
                );
            }
            if let Some(sparse_label) = sparse_indexer_stage_label(stage.kind) {
                add_stage(
                    &mut layer_sparse_indexer_stages,
                    sparse_label,
                    stage.duration_ms_scaled,
                );
                add_stage(
                    &mut sparse_indexer_stage_ms,
                    sparse_label,
                    stage.duration_ms_scaled,
                );
                add_stage(
                    attention_kind_sparse_indexer_stage_ms
                        .entry(attention_kind)
                        .or_insert_with(BTreeMap::new),
                    sparse_label,
                    stage.duration_ms_scaled,
                );
            }
        }
        pre_expert_layers.push(Dsv4PrefillPreExpertLayer {
            layer: sampled.layer,
            attention_kind,
            command_gpu_ms: sampled.command_gpu_ms,
            encoder_gap_ms: sampled.encoder_gap_ms_scaled,
            encoder_overlap_ms: sampled.encoder_overlap_ms_scaled,
            raw_coverage: sampled.raw_coverage_assuming_ns,
            stage_ms: layer_stages,
            attention_body_stage_ms: layer_attention_body_stages,
            sparse_indexer_stage_ms: layer_sparse_indexer_stages,
        });
    }

    let mut post_route_encoder_gap_ms = 0.0;
    let mut post_route_encoder_overlap_ms = 0.0;
    let mut layers = Vec::with_capacity(post_route.metadata.len());
    for (index, metadata) in post_route.metadata.iter().enumerate() {
        let sampled = post_route.sampled_layers.get(index);
        let mut layer_stages = BTreeMap::new();
        if let Some(sampled) = sampled {
            ensure!(
                sampled.layer == metadata.layer,
                "sampled post-route layer {} differs from metadata layer {}",
                sampled.layer,
                metadata.layer
            );
            for stage in &sampled.stages {
                let label = post_route_stage_label(stage.kind);
                add_stage(&mut layer_stages, label, stage.duration_ms_scaled);
                add_stage(&mut post_route_stage_ms, label, stage.duration_ms_scaled);
                if metadata.bm16 {
                    add_stage(&mut bm16_stage_ms, label, stage.duration_ms_scaled);
                }
            }
            post_route_encoder_gap_ms += sampled.encoder_gap_ms_scaled;
            post_route_encoder_overlap_ms += sampled.encoder_overlap_ms_scaled;
        }
        let route_count = metadata
            .expert_counts
            .iter()
            .map(|&count| usize::from(count))
            .sum::<usize>();
        let route_tiles16 = metadata
            .expert_counts
            .iter()
            .map(|&count| usize::from(count).div_ceil(16))
            .sum::<usize>();
        let route_tiles32 = metadata
            .expert_counts
            .iter()
            .map(|&count| usize::from(count).div_ceil(32))
            .sum::<usize>();
        layers.push(Dsv4PrefillLayer {
            layer: metadata.layer,
            gate_dtype: format!("{:?}", metadata.gate_dtype),
            up_dtype: format!("{:?}", metadata.up_dtype),
            down_dtype: format!("{:?}", metadata.down_dtype),
            grouped_iq2: metadata.grouped_iq2,
            grouped_iq3: metadata.grouped_iq3,
            bm16: metadata.bm16,
            bucket_count: metadata.bucket_count,
            active_experts: metadata
                .expert_counts
                .iter()
                .filter(|&&count| count > 0)
                .count(),
            max_routes_per_expert: metadata.expert_counts.iter().copied().max().unwrap_or(0),
            route_count,
            route_tiles16,
            route_tiles32,
            route_tile16_occupancy: route_count as f64 / (route_tiles16 * 16) as f64,
            route_tile32_occupancy: route_count as f64 / (route_tiles32 * 32) as f64,
            command_gpu_ms: post_route.command_gpu_ms[index],
            encoder_gap_ms: sampled.map_or(0.0, |profile| profile.encoder_gap_ms_scaled),
            encoder_overlap_ms: sampled.map_or(0.0, |profile| profile.encoder_overlap_ms_scaled),
            raw_coverage: sampled.map_or(0.0, |profile| profile.raw_coverage_assuming_ns),
            stage_ms: layer_stages,
        });
    }

    Ok(Dsv4PrefillChunk {
        chunk_index,
        token_start,
        token_count,
        emit_logits,
        sampled: post_route.sampled,
        wall_ms,
        pre_expert_gpu_ms,
        post_route_gpu_ms,
        non_gpu_residual_ms: wall_ms - pre_expert_gpu_ms - post_route_gpu_ms,
        q8_compressor_matrix_invocations: post_route.q8_compressor_matrix_invocations,
        bm16_gpu_ms,
        pre_expert_encoder_gap_ms,
        pre_expert_encoder_overlap_ms,
        post_route_encoder_gap_ms,
        post_route_encoder_overlap_ms,
        pre_expert_stage_ms,
        attention_kind_stage_ms,
        attention_body_stage_ms,
        attention_kind_body_stage_ms,
        sparse_indexer_stage_ms,
        attention_kind_sparse_indexer_stage_ms,
        post_route_stage_ms,
        bm16_stage_ms,
        pre_expert_layers,
        layers,
    })
}

fn summarize_run(
    wall_ms: f64,
    chunks: Vec<Dsv4PrefillChunk>,
    ordinary_reference_wall_ms: f64,
) -> Dsv4PrefillRun {
    let mut optimistic_boundary_ceiling_ms = chunks
        .chunks(2)
        .filter(|pair| pair.len() == 2)
        .map(|pair| {
            pair.iter()
                .map(|chunk| chunk.non_gpu_residual_ms.max(0.0))
                .fold(0.0f64, f64::max)
        })
        .sum::<f64>();
    if optimistic_boundary_ceiling_ms == 0.0 {
        optimistic_boundary_ceiling_ms = 0.0;
    }
    Dsv4PrefillRun {
        sampled: true,
        wall_ms,
        chunk_count: chunks.len(),
        pre_expert_gpu_ms: chunks.iter().map(|chunk| chunk.pre_expert_gpu_ms).sum(),
        post_route_gpu_ms: chunks.iter().map(|chunk| chunk.post_route_gpu_ms).sum(),
        non_gpu_residual_ms: chunks.iter().map(|chunk| chunk.non_gpu_residual_ms).sum(),
        q8_compressor_matrix_invocations: chunks
            .iter()
            .map(|chunk| chunk.q8_compressor_matrix_invocations)
            .sum(),
        bm16_gpu_ms: chunks.iter().map(|chunk| chunk.bm16_gpu_ms).sum(),
        optimistic_boundary_ceiling_ms,
        optimistic_boundary_ceiling_percent: optimistic_boundary_ceiling_ms
            / ordinary_reference_wall_ms
            * 100.0,
        chunks,
    }
}

fn add_stage_totals(
    totals: &mut BTreeMap<&'static str, f64>,
    stages: &BTreeMap<&'static str, f64>,
) {
    for (&stage, &milliseconds) in stages {
        *totals.entry(stage).or_default() += milliseconds;
    }
}

fn sample_summary(run: &Dsv4PrefillRun, n_tokens: usize) -> Dsv4PrefillSampleSummary {
    let mut pre_expert_stage_ms = BTreeMap::new();
    let mut attention_body_stage_ms = BTreeMap::new();
    let mut sparse_indexer_stage_ms = BTreeMap::new();
    let mut post_route_stage_ms = BTreeMap::new();
    let mut bm16_stage_ms = BTreeMap::new();
    for chunk in &run.chunks {
        add_stage_totals(&mut pre_expert_stage_ms, &chunk.pre_expert_stage_ms);
        add_stage_totals(&mut attention_body_stage_ms, &chunk.attention_body_stage_ms);
        add_stage_totals(&mut sparse_indexer_stage_ms, &chunk.sparse_indexer_stage_ms);
        add_stage_totals(&mut post_route_stage_ms, &chunk.post_route_stage_ms);
        add_stage_totals(&mut bm16_stage_ms, &chunk.bm16_stage_ms);
    }
    Dsv4PrefillSampleSummary {
        wall_ms: run.wall_ms,
        prefill_tps: n_tokens as f64 * 1e3 / run.wall_ms,
        chunk_count: run.chunk_count,
        pre_expert_gpu_ms: run.pre_expert_gpu_ms,
        post_route_gpu_ms: run.post_route_gpu_ms,
        non_gpu_residual_ms: run.non_gpu_residual_ms,
        q8_compressor_matrix_invocations: run.q8_compressor_matrix_invocations,
        bm16_gpu_ms: run.bm16_gpu_ms,
        optimistic_boundary_ceiling_ms: run.optimistic_boundary_ceiling_ms,
        optimistic_boundary_ceiling_percent: run.optimistic_boundary_ceiling_percent,
        pre_expert_stage_ms,
        attention_body_stage_ms,
        sparse_indexer_stage_ms,
        post_route_stage_ms,
        bm16_stage_ms,
    }
}

fn compact_summary(report: &Dsv4PrefillReport) -> Dsv4PrefillSummary<'_> {
    Dsv4PrefillSummary {
        schema_version: 1,
        report_kind: "dsv4_prefill_summary",
        full_report_schema_version: report.schema_version,
        build: &report.build,
        model: &report.model,
        device: &report.device,
        token_source: &report.token_source,
        n_tokens: report.n_tokens,
        chunk_tokens: report.chunk_tokens,
        candidate_chunk_tokens: report.candidate_chunk_tokens,
        load_ms: report.load_ms,
        ordinary_wall_ms: report.ordinary.wall_ms,
        ordinary_prefill_tps: report.n_tokens as f64 * 1e3 / report.ordinary.wall_ms,
        ordinary_chunk_wall_ms: &report.ordinary.chunk_wall_ms,
        logits_bit_exact: report.logits_bit_exact,
        logits_sha256: &report.logits_sha256,
        warmup_wall_ms: report.warmups.iter().map(|run| run.wall_ms).collect(),
        samples: report
            .samples
            .iter()
            .map(|run| sample_summary(run, report.n_tokens))
            .collect(),
    }
}

fn logits_sha256(logit_bits: &[u32]) -> String {
    format!(
        "{:x}",
        Sha256::digest(bytemuck::cast_slice::<u32, u8>(logit_bits))
    )
}

fn prompt_tokens(args: &Dsv4PrefillArgs, gguf: &GgufFile) -> Result<(Vec<u32>, String)> {
    let tokenizer = Tokenizer::from_gguf(gguf).context("load DeepSeek V4 tokenizer")?;
    if let Some(path) = &args.prompt_file {
        let prompt = std::fs::read_to_string(path)
            .with_context(|| format!("read prompt {}", path.display()))?;
        let encoded = tokenizer
            .encode(&prompt, false)
            .with_context(|| format!("tokenize prompt {}", path.display()))?;
        ensure!(
            encoded.len() >= args.tokens,
            "prompt {} produced {} tokens, fewer than requested {}",
            path.display(),
            encoded.len(),
            args.tokens
        );
        let tokens = encoded
            .into_iter()
            .take(args.tokens)
            .map(|token| u32::try_from(token).context("prompt token is negative"))
            .collect::<Result<Vec<_>>>()?;
        return Ok((tokens, format!("prompt_file:{}", path.display())));
    }

    let vocab = usize::try_from(tokenizer.n_vocab()).context("vocabulary exceeds usize")?;
    let tokens = (0..args.tokens)
        .map(|index| ((35 + index * 7_919) % vocab) as u32)
        .collect();
    Ok((tokens, "synthetic_ramp_7919".into()))
}

fn copy_logit_bits(session: &DeepSeekV4Session) -> Result<Vec<u32>> {
    Ok(session
        .copy_logits_f32()
        .context("copy DeepSeek V4 profile logits")?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

fn execute_ordinary_request(
    ctx: &MetalContext,
    residency: &mut Option<DeepSeekV4MetalResidency>,
    tokens: &[u32],
    chunk_tokens: usize,
) -> Result<(Dsv4PrefillOrdinaryRun, Vec<u32>)> {
    let current = residency
        .take()
        .context("DeepSeek V4 profiler lost model residency")?;
    let mut session = DeepSeekV4Session::new(ctx, current)
        .context("create ordinary DeepSeek V4 prefill profile session")?;
    let chunk_count = tokens.len().div_ceil(chunk_tokens);
    let request_started = Instant::now();
    let mut chunk_wall_ms = Vec::with_capacity(chunk_count);
    for (chunk_index, chunk) in tokens.chunks(chunk_tokens).enumerate() {
        let chunk_started = Instant::now();
        if chunk_index + 1 == chunk_count {
            session
                .prefill_tokens(ctx, chunk)
                .context("execute final ordinary DeepSeek V4 profile chunk")?;
        } else {
            session
                .advance_tokens(ctx, chunk)
                .context("advance ordinary DeepSeek V4 profile chunk")?;
        }
        chunk_wall_ms.push(chunk_started.elapsed().as_secs_f64() * 1e3);
    }
    let wall_ms = request_started.elapsed().as_secs_f64() * 1e3;
    let logits = copy_logit_bits(&session)?;
    *residency = Some(session.into_residency());
    Ok((
        Dsv4PrefillOrdinaryRun {
            wall_ms,
            chunk_wall_ms,
        },
        logits,
    ))
}

fn execute_profiled_request(
    ctx: &MetalContext,
    residency: &mut Option<DeepSeekV4MetalResidency>,
    tokens: &[u32],
    chunk_tokens: usize,
    ordinary_reference_wall_ms: f64,
    attention_kinds: &[AttentionKind],
) -> Result<(Dsv4PrefillRun, Vec<u32>)> {
    let current = residency
        .take()
        .context("DeepSeek V4 profiler lost model residency")?;
    let mut session = DeepSeekV4Session::new(ctx, current)
        .context("create sampled DeepSeek V4 prefill profile session")?;
    let chunk_count = tokens.len().div_ceil(chunk_tokens);
    let request_started = Instant::now();
    let mut chunks = Vec::with_capacity(chunk_count);
    for (chunk_index, chunk) in tokens.chunks(chunk_tokens).enumerate() {
        let emit_logits = chunk_index + 1 == chunk_count;
        let token_start = chunk_index * chunk_tokens;
        let chunk_started = Instant::now();
        let profile = session
            .profile_packed_chunk(ctx, chunk, emit_logits, true)
            .context("profile sequential DeepSeek V4 packed chunk")?;
        chunks.push(summarize_chunk(
            profile,
            chunk_index,
            token_start,
            chunk.len(),
            emit_logits,
            chunk_started.elapsed().as_secs_f64() * 1e3,
            attention_kinds,
        )?);
    }
    let wall_ms = request_started.elapsed().as_secs_f64() * 1e3;
    let logits = copy_logit_bits(&session)?;
    let run = summarize_run(wall_ms, chunks, ordinary_reference_wall_ms);
    *residency = Some(session.into_residency());
    Ok((run, logits))
}

pub fn run(args: Dsv4PrefillArgs, build: Value) -> Result<()> {
    ensure!(
        (1..=32_768).contains(&args.tokens),
        "--tokens must be 1..=32768"
    );
    ensure!(
        (1..=DEEPSEEK_V4_PREFILL_MAX_TOKENS).contains(&args.chunk_tokens),
        "--chunk-tokens must be 1..={DEEPSEEK_V4_PREFILL_MAX_TOKENS}"
    );
    ensure!(
        args.chunk_tokens.checked_mul(2) == Some(args.candidate_chunk_tokens),
        "--candidate-chunk-tokens must be exactly twice --chunk-tokens"
    );
    ensure!(args.warmups > 0, "--warmups must be nonzero");
    ensure!(args.samples > 0, "--samples must be nonzero");
    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open model {}", args.model.display()))?;
    let (tokens, token_source) = prompt_tokens(&args, &gguf)?;
    let ctx = MetalContext::new().context("create Metal context")?;
    let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, tokens.len())
        .context("plan DeepSeek V4 prefill profile")?;
    let attention_kinds = plan.config().attention_kinds.clone();
    let admitted = plan
        .admit(ctx.memory_signals())
        .context("admit DeepSeek V4 prefill profile")?;
    let load_started = Instant::now();
    let realized = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted)
        .context("load DeepSeek V4 residency")?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1e3;
    let mut residency = Some(realized.into_residency());

    let mut warmups = Vec::with_capacity(args.warmups);
    let mut reference_logits = None;
    for _ in 0..args.warmups {
        let (run, logits) =
            execute_ordinary_request(&ctx, &mut residency, &tokens, args.chunk_tokens)?;
        if let Some(reference) = &reference_logits {
            ensure!(&logits == reference, "ordinary profiler logits changed");
        } else {
            reference_logits = Some(logits);
        }
        warmups.push(run);
    }
    let (ordinary, ordinary_logits) =
        execute_ordinary_request(&ctx, &mut residency, &tokens, args.chunk_tokens)?;
    ensure!(
        reference_logits.as_ref() == Some(&ordinary_logits),
        "ordinary reference logits differ from warmup execution"
    );
    let ordinary_reference_wall_ms = ordinary.wall_ms;
    ensure!(
        ordinary_reference_wall_ms.is_finite() && ordinary_reference_wall_ms > 0.0,
        "ordinary DeepSeek V4 reference wall is invalid"
    );
    let mut samples = Vec::with_capacity(args.samples);
    for _ in 0..args.samples {
        let (run, logits) = execute_profiled_request(
            &ctx,
            &mut residency,
            &tokens,
            args.chunk_tokens,
            ordinary_reference_wall_ms,
            &attention_kinds,
        )?;
        ensure!(
            reference_logits.as_ref() == Some(&logits),
            "sampled profiler logits differ from ordinary execution"
        );
        samples.push(run);
    }
    let report = Dsv4PrefillReport {
        schema_version: 5,
        build,
        model: args.model.display().to_string(),
        device: ctx.device.name().to_string(),
        token_source,
        n_tokens: tokens.len(),
        chunk_tokens: args.chunk_tokens,
        candidate_chunk_tokens: args.candidate_chunk_tokens,
        load_ms,
        ordinary_reference_wall_ms,
        logits_bit_exact: true,
        logits_sha256: logits_sha256(&ordinary_logits),
        warmups,
        ordinary,
        samples,
    };
    let full_json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &args.json_out {
        std::fs::write(path, format!("{full_json}\n"))
            .with_context(|| format!("write profile {}", path.display()))?;
    }
    if args.full_json {
        println!("{full_json}");
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(&compact_summary(&report))?
        );
    }
    Ok(())
}
