use anyhow::{Context, Result, ensure};
use clap::Args;
use objc2_metal::MTLDevice;
use qwen_llm::deepseek_v4_metal::{
    DeepSeekV4MetalResidency, DeepSeekV4Session, PackedPostRouteStageKind,
    PackedPostRouteStageProfile,
};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use qwen_llm::tokenizer::Tokenizer;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Args, Debug)]
pub struct Dsv4PrefillArgs {
    /// Path to the first shard of a DeepSeek V4 Flash-0731 GGUF.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Number of tokens in the profiled packed chunk.
    #[arg(short = 'p', long, default_value_t = 2_048)]
    tokens: usize,
    /// Optional raw prompt text. The first --tokens independently encoded IDs are used.
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Unsampled production-policy passes before acquisition.
    #[arg(long, default_value_t = 1)]
    warmups: usize,
    /// Sampled production-policy passes to report.
    #[arg(long, default_value_t = 1)]
    samples: usize,
    /// Optional path for the same JSON emitted to stdout.
    #[arg(long)]
    json_out: Option<PathBuf>,
}

#[derive(Serialize)]
struct Dsv4PrefillReport {
    schema_version: u32,
    build: Value,
    model: String,
    device: String,
    token_source: String,
    n_tokens: usize,
    load_ms: f64,
    logits_bit_exact: bool,
    warmups: Vec<Dsv4PrefillRun>,
    samples: Vec<Dsv4PrefillRun>,
}

#[derive(Serialize)]
struct Dsv4PrefillRun {
    sampled: bool,
    wall_ms: f64,
    post_route_gpu_ms: f64,
    bm16_gpu_ms: f64,
    encoder_gap_ms: f64,
    encoder_overlap_ms: f64,
    stage_ms: BTreeMap<&'static str, f64>,
    bm16_stage_ms: BTreeMap<&'static str, f64>,
    layers: Vec<Dsv4PrefillLayer>,
}

#[derive(Serialize)]
struct Dsv4PrefillLayer {
    layer: usize,
    gate_dtype: String,
    up_dtype: String,
    down_dtype: String,
    grouped_iq2: bool,
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

fn stage_label(kind: PackedPostRouteStageKind) -> &'static str {
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

fn summarize(profile: PackedPostRouteStageProfile, wall_ms: f64) -> Dsv4PrefillRun {
    let post_route_gpu_ms = profile.command_gpu_ms.iter().sum();
    let bm16_gpu_ms = profile
        .metadata
        .iter()
        .zip(&profile.command_gpu_ms)
        .filter_map(|(metadata, &duration)| metadata.bm16.then_some(duration))
        .sum();
    if !profile.sampled {
        return Dsv4PrefillRun {
            sampled: false,
            wall_ms,
            post_route_gpu_ms,
            bm16_gpu_ms,
            encoder_gap_ms: 0.0,
            encoder_overlap_ms: 0.0,
            stage_ms: BTreeMap::new(),
            bm16_stage_ms: BTreeMap::new(),
            layers: Vec::new(),
        };
    }

    let mut stage_ms = BTreeMap::new();
    let mut bm16_stage_ms = BTreeMap::new();
    let mut encoder_gap_ms = 0.0;
    let mut encoder_overlap_ms = 0.0;
    let mut layers = Vec::with_capacity(profile.sampled_layers.len());
    for (sampled, metadata) in profile.sampled_layers.iter().zip(&profile.metadata) {
        let mut layer_stages = BTreeMap::new();
        for stage in &sampled.stages {
            let label = stage_label(stage.kind);
            add_stage(&mut layer_stages, label, stage.duration_ms_scaled);
            add_stage(&mut stage_ms, label, stage.duration_ms_scaled);
            if metadata.bm16 {
                add_stage(&mut bm16_stage_ms, label, stage.duration_ms_scaled);
            }
        }
        encoder_gap_ms += sampled.encoder_gap_ms_scaled;
        encoder_overlap_ms += sampled.encoder_overlap_ms_scaled;
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
            command_gpu_ms: sampled.command_gpu_ms,
            encoder_gap_ms: sampled.encoder_gap_ms_scaled,
            encoder_overlap_ms: sampled.encoder_overlap_ms_scaled,
            raw_coverage: sampled.raw_coverage_assuming_ns,
            stage_ms: layer_stages,
        });
    }
    Dsv4PrefillRun {
        sampled: true,
        wall_ms,
        post_route_gpu_ms,
        bm16_gpu_ms,
        encoder_gap_ms,
        encoder_overlap_ms,
        stage_ms,
        bm16_stage_ms,
        layers,
    }
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

pub fn run(args: Dsv4PrefillArgs, build: Value) -> Result<()> {
    ensure!(
        (1..=2_048).contains(&args.tokens),
        "--tokens must be 1..=2048"
    );
    ensure!(args.warmups > 0, "--warmups must be nonzero");
    ensure!(args.samples > 0, "--samples must be nonzero");
    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open model {}", args.model.display()))?;
    let (tokens, token_source) = prompt_tokens(&args, &gguf)?;
    let ctx = MetalContext::new().context("create Metal context")?;
    let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, tokens.len())
        .context("plan DeepSeek V4 prefill profile")?;
    let admitted = plan
        .admit(ctx.memory_signals())
        .context("admit DeepSeek V4 prefill profile")?;
    let load_started = Instant::now();
    let realized = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted)
        .context("load DeepSeek V4 residency")?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1e3;
    let mut residency = Some(realized.into_residency());

    let mut execute = |sampled: bool| -> Result<(Dsv4PrefillRun, Vec<u32>)> {
        let current = residency
            .take()
            .context("DeepSeek V4 profiler lost model residency")?;
        let mut session = DeepSeekV4Session::new(&ctx, current)
            .context("create DeepSeek V4 prefill profile session")?;
        let started = Instant::now();
        let profile = session
            .profile_packed_post_route(&ctx, &tokens, true, sampled)
            .context("profile DeepSeek V4 packed prefill")?;
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        let logits = session
            .copy_logits_f32()
            .context("copy DeepSeek V4 profile logits")?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        residency = Some(session.into_residency());
        Ok((summarize(profile, wall_ms), logits))
    };

    let mut warmups = Vec::with_capacity(args.warmups);
    let mut reference_logits = None;
    for _ in 0..args.warmups {
        let (run, logits) = execute(false)?;
        if let Some(reference) = &reference_logits {
            ensure!(&logits == reference, "ordinary profiler logits changed");
        } else {
            reference_logits = Some(logits);
        }
        warmups.push(run);
    }
    let mut samples = Vec::with_capacity(args.samples);
    for _ in 0..args.samples {
        let (run, logits) = execute(true)?;
        ensure!(
            reference_logits.as_ref() == Some(&logits),
            "sampled profiler logits differ from ordinary execution"
        );
        samples.push(run);
    }
    let report = Dsv4PrefillReport {
        schema_version: 1,
        build,
        model: args.model.display().to_string(),
        device: ctx.device.name().to_string(),
        token_source,
        n_tokens: tokens.len(),
        load_ms,
        logits_bit_exact: true,
        warmups,
        samples,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.json_out {
        std::fs::write(&path, format!("{json}\n"))
            .with_context(|| format!("write profile {}", path.display()))?;
    }
    println!("{json}");
    Ok(())
}
