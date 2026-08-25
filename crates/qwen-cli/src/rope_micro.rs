use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLDevice};
use qwen_llm::{
    metal::{
        KernelEncoder, MetalContext, MetalTensor, encode_qk_rms_norm_rope_f32_packed_consecutive,
        encode_rms_norm_batched_f32, encode_rms_norm_batched_src_strided_f32,
        encode_rope_neox_f32_packed_consecutive,
        encode_rope_neox_pair_adaptive_f32_packed_consecutive, encode_rope_neox_pair_f32,
        encode_rope_neox_pair_f32_packed_consecutive, encode_rope_neox_pair_shared_f32,
        encode_rope_neox_pair_shared_f32_packed_consecutive,
        encode_rope_neox_pair_shared_minimax_f32,
        encode_rope_neox_pair_shared_minimax_f32_packed_consecutive,
        encode_rope_neox_pair_sincos_f32,
    },
    tensor::GgmlType,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Parser, Debug)]
pub struct RopeMicroArgs {
    /// Packed token counts to measure.
    #[arg(long, value_delimiter = ',', default_value = "1,8,128,512")]
    tokens: Vec<usize>,
    /// Q heads in each row.
    #[arg(long, default_value = "24")]
    q_heads: usize,
    /// KV heads in each row.
    #[arg(long, default_value = "4")]
    kv_heads: usize,
    /// Dimensions per attention head.
    #[arg(long, default_value = "256")]
    head_dim: usize,
    /// Rotated dimensions per head.
    #[arg(long, default_value = "64")]
    rotary_dim: usize,
    /// RoPE frequency base.
    #[arg(long, default_value = "10000000")]
    theta: f32,
    /// First packed position and singleton timing position.
    #[arg(long, default_value = "65536")]
    start_position: u32,
    /// Positions used by the singleton numerical sweep.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "0,1,127,65535,65536,262143,1048575"
    )]
    positions: Vec<u32>,
    /// RoPE layer dispatches encoded into each measured command.
    #[arg(long, default_value = "16")]
    layers: usize,
    /// Interleaved timed rounds after one warmup per arm.
    #[arg(long, default_value = "12")]
    runs: usize,
    /// `text` or `json`.
    #[arg(short = 'o', long, value_enum, default_value = "text")]
    output: OutputFormat,
}

#[derive(Clone, Copy, Debug)]
enum DecodeArm {
    Baseline,
    Sincos,
    Shared,
    Minimax,
}

impl DecodeArm {
    const ALL: [Self; 4] = [Self::Baseline, Self::Sincos, Self::Shared, Self::Minimax];

    fn name(self) -> &'static str {
        match self {
            Self::Baseline => "head_parallel_sin_cos",
            Self::Sincos => "head_parallel_sincos",
            Self::Shared => "shared_head_sincos",
            Self::Minimax => "shared_head_minimax",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug)]
enum PackedArm {
    Split,
    Paired,
    Shared,
    Minimax,
}

impl PackedArm {
    const ALL: [Self; 4] = [Self::Split, Self::Paired, Self::Shared, Self::Minimax];

    fn name(self) -> &'static str {
        match self {
            Self::Split => "split_q_k",
            Self::Paired => "paired_head_parallel",
            Self::Shared => "paired_shared_head",
            Self::Minimax => "paired_shared_minimax",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug)]
enum NormRopeArm {
    ComposedSplit,
    ComposedAdaptive,
    Fused,
}

impl NormRopeArm {
    const ALL: [Self; 3] = [Self::ComposedSplit, Self::ComposedAdaptive, Self::Fused];

    fn name(self) -> &'static str {
        match self {
            Self::ComposedSplit => "norms_plus_split_rope",
            Self::ComposedAdaptive => "norms_plus_adaptive_rope",
            Self::Fused => "fused_qk_norm_rope",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

struct ArmBuffers {
    q: MetalTensor,
    k: MetalTensor,
}

struct NormRopeBuffers {
    q_src: MetalTensor,
    q_weight: MetalTensor,
    q_out: MetalTensor,
    k_src: MetalTensor,
    k_weight: MetalTensor,
    k_out: MetalTensor,
}

#[derive(Default)]
struct DiffAccumulator {
    max_abs: f32,
    sum_sq: f64,
    count: usize,
}

impl DiffAccumulator {
    fn observe(&mut self, baseline: &[f32], candidate: &[f32]) {
        for (&left, &right) in baseline.iter().zip(candidate) {
            let diff = (left - right).abs();
            self.max_abs = self.max_abs.max(diff);
            self.sum_sq += f64::from(diff) * f64::from(diff);
            self.count += 1;
        }
    }

    fn rms_abs(&self) -> f64 {
        (self.sum_sq / self.count.max(1) as f64).sqrt()
    }
}

#[derive(serde::Serialize)]
struct AccuracyRow {
    workload: &'static str,
    arm: &'static str,
    cases: usize,
    values_compared: usize,
    max_abs: f32,
    rms_abs: f64,
}

#[derive(serde::Serialize)]
struct TimingArmRow {
    arm: &'static str,
    mean_command_ms: f64,
    median_command_ms: f64,
    stdev_command_ms: f64,
    mean_us_per_layer: f64,
    speedup_vs_baseline: f64,
    samples_ms: Vec<f64>,
}

#[derive(serde::Serialize)]
struct TimingRow {
    workload: &'static str,
    tokens: usize,
    layers_per_command: usize,
    arms: Vec<TimingArmRow>,
}

#[derive(serde::Serialize)]
struct RopeMicroReport {
    schema_version: u32,
    test: &'static str,
    device: String,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    start_position: u32,
    accuracy_positions: Vec<u32>,
    runs: usize,
    accuracy: Vec<AccuracyRow>,
    timings: Vec<TimingRow>,
    build_identity: serde_json::Value,
}

fn input_values(len: usize, modulus: usize, scale: f32) -> Vec<f32> {
    let midpoint = (modulus / 2) as f32;
    (0..len)
        .map(|index| ((index % modulus) as f32 - midpoint) * scale)
        .collect()
}

fn make_buffers(ctx: &MetalContext, q_values: &[f32], k_values: &[f32]) -> Result<ArmBuffers> {
    Ok(ArmBuffers {
        q: MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(q_values),
            vec![q_values.len() as u64],
            GgmlType::F32,
        )?,
        k: MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(k_values),
            vec![k_values.len() as u64],
            GgmlType::F32,
        )?,
    })
}

fn make_norm_rope_buffers(
    ctx: &MetalContext,
    args: &RopeMicroArgs,
    tokens: usize,
) -> Result<NormRopeBuffers> {
    let q_rows = tokens
        .checked_mul(args.q_heads)
        .context("norm-RoPE Q rows overflow")?;
    let k_rows = tokens
        .checked_mul(args.kv_heads)
        .context("norm-RoPE K rows overflow")?;
    let q_source = input_values(q_rows * 2 * args.head_dim, 41, 0.03125);
    let k_source = input_values(k_rows * args.head_dim, 37, 0.046875);
    let q_weight: Vec<f32> = (0..args.head_dim)
        .map(|index| 0.5 + (index % 11) as f32 * 0.0625)
        .collect();
    let k_weight: Vec<f32> = (0..args.head_dim)
        .map(|index| 0.625 + (index % 7) as f32 * 0.078125)
        .collect();
    Ok(NormRopeBuffers {
        q_src: MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&q_source),
            vec![q_source.len() as u64],
            GgmlType::F32,
        )?,
        q_weight: MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&q_weight),
            vec![q_weight.len() as u64],
            GgmlType::F32,
        )?,
        q_out: MetalTensor::zeros_f32(ctx, vec![(q_rows * args.head_dim) as u64])?,
        k_src: MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&k_source),
            vec![k_source.len() as u64],
            GgmlType::F32,
        )?,
        k_weight: MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&k_weight),
            vec![k_weight.len() as u64],
            GgmlType::F32,
        )?,
        k_out: MetalTensor::zeros_f32(ctx, vec![(k_rows * args.head_dim) as u64])?,
    })
}

fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
    let len = tensor.n_elements() as usize;
    let mut values = vec![0.0; len];
    unsafe {
        let source = (tensor.buffer.contents().as_ptr() as *const u8).add(tensor.offset as usize)
            as *const f32;
        std::ptr::copy_nonoverlapping(source, values.as_mut_ptr(), len);
    }
    values
}

fn run_command(
    ctx: &MetalContext,
    encode: impl FnOnce(&KernelEncoder) -> Result<()>,
) -> Result<f64> {
    let command = ctx.queue.commandBuffer().context("RoPE micro command")?;
    let encoder = KernelEncoder::begin(&command);
    encode(&encoder)?;
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    Ok((command.GPUEndTime() - command.GPUStartTime()) * 1e3)
}

fn encode_decode(
    ctx: &MetalContext,
    encoder: &KernelEncoder,
    buffers: &ArmBuffers,
    arm: DecodeArm,
    args: &RopeMicroArgs,
    position: u32,
) -> Result<()> {
    let result = match arm {
        DecodeArm::Baseline => encode_rope_neox_pair_f32(
            ctx,
            encoder,
            &buffers.q,
            &buffers.k,
            args.q_heads,
            args.kv_heads,
            args.head_dim,
            args.rotary_dim,
            position,
            args.theta,
        ),
        DecodeArm::Sincos => encode_rope_neox_pair_sincos_f32(
            ctx,
            encoder,
            &buffers.q,
            &buffers.k,
            args.q_heads,
            args.kv_heads,
            args.head_dim,
            args.rotary_dim,
            position,
            args.theta,
        ),
        DecodeArm::Shared => encode_rope_neox_pair_shared_f32(
            ctx,
            encoder,
            &buffers.q,
            &buffers.k,
            args.q_heads,
            args.kv_heads,
            args.head_dim,
            args.rotary_dim,
            position,
            args.theta,
        ),
        DecodeArm::Minimax => encode_rope_neox_pair_shared_minimax_f32(
            ctx,
            encoder,
            &buffers.q,
            &buffers.k,
            args.q_heads,
            args.kv_heads,
            args.head_dim,
            args.rotary_dim,
            position,
            args.theta,
        ),
    };
    result.map_err(Into::into)
}

fn encode_packed(
    ctx: &MetalContext,
    encoder: &KernelEncoder,
    buffers: &ArmBuffers,
    arm: PackedArm,
    args: &RopeMicroArgs,
    tokens: usize,
) -> Result<()> {
    let result = match arm {
        PackedArm::Split => {
            encode_rope_neox_f32_packed_consecutive(
                ctx,
                encoder,
                &buffers.q,
                tokens,
                args.q_heads,
                args.head_dim,
                args.rotary_dim,
                args.start_position,
                args.theta,
            )?;
            encode_rope_neox_f32_packed_consecutive(
                ctx,
                encoder,
                &buffers.k,
                tokens,
                args.kv_heads,
                args.head_dim,
                args.rotary_dim,
                args.start_position,
                args.theta,
            )
        }
        PackedArm::Paired => encode_rope_neox_pair_f32_packed_consecutive(
            ctx,
            encoder,
            &buffers.q,
            &buffers.k,
            tokens,
            args.q_heads,
            args.kv_heads,
            args.head_dim,
            args.rotary_dim,
            args.start_position,
            args.theta,
        ),
        PackedArm::Shared => encode_rope_neox_pair_shared_f32_packed_consecutive(
            ctx,
            encoder,
            &buffers.q,
            &buffers.k,
            tokens,
            args.q_heads,
            args.kv_heads,
            args.head_dim,
            args.rotary_dim,
            args.start_position,
            args.theta,
        ),
        PackedArm::Minimax => encode_rope_neox_pair_shared_minimax_f32_packed_consecutive(
            ctx,
            encoder,
            &buffers.q,
            &buffers.k,
            tokens,
            args.q_heads,
            args.kv_heads,
            args.head_dim,
            args.rotary_dim,
            args.start_position,
            args.theta,
        ),
    };
    result.map_err(Into::into)
}

fn encode_norm_rope(
    ctx: &MetalContext,
    encoder: &KernelEncoder,
    buffers: &NormRopeBuffers,
    arm: NormRopeArm,
    args: &RopeMicroArgs,
    tokens: usize,
) -> Result<()> {
    const EPS: f32 = 1e-6;
    if matches!(arm, NormRopeArm::Fused) {
        return encode_qk_rms_norm_rope_f32_packed_consecutive(
            ctx,
            encoder,
            &buffers.q_src,
            &buffers.q_weight,
            &buffers.q_out,
            &buffers.k_src,
            &buffers.k_weight,
            &buffers.k_out,
            tokens,
            args.q_heads,
            args.kv_heads,
            args.head_dim,
            args.rotary_dim,
            args.start_position,
            EPS,
            args.theta,
        )
        .map_err(Into::into);
    }

    encode_rms_norm_batched_src_strided_f32(
        ctx,
        encoder,
        &buffers.q_src,
        &buffers.q_weight,
        &buffers.q_out,
        tokens * args.q_heads,
        args.head_dim,
        2 * args.head_dim,
        0,
        EPS,
    )?;
    encode_rms_norm_batched_f32(
        ctx,
        encoder,
        &buffers.k_src,
        &buffers.k_weight,
        &buffers.k_out,
        tokens * args.kv_heads,
        args.head_dim,
        EPS,
    )?;
    match arm {
        NormRopeArm::ComposedSplit => {
            encode_rope_neox_f32_packed_consecutive(
                ctx,
                encoder,
                &buffers.q_out,
                tokens,
                args.q_heads,
                args.head_dim,
                args.rotary_dim,
                args.start_position,
                args.theta,
            )?;
            encode_rope_neox_f32_packed_consecutive(
                ctx,
                encoder,
                &buffers.k_out,
                tokens,
                args.kv_heads,
                args.head_dim,
                args.rotary_dim,
                args.start_position,
                args.theta,
            )?;
        }
        NormRopeArm::ComposedAdaptive => {
            encode_rope_neox_pair_adaptive_f32_packed_consecutive(
                ctx,
                encoder,
                &buffers.q_out,
                &buffers.k_out,
                tokens,
                args.q_heads,
                args.kv_heads,
                args.head_dim,
                args.rotary_dim,
                args.start_position,
                args.theta,
            )?;
        }
        NormRopeArm::Fused => unreachable!(),
    }
    Ok(())
}

fn run_decode_once(
    ctx: &MetalContext,
    args: &RopeMicroArgs,
    q_values: &[f32],
    k_values: &[f32],
    arm: DecodeArm,
    position: u32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let buffers = make_buffers(ctx, q_values, k_values)?;
    run_command(ctx, |encoder| {
        encode_decode(ctx, encoder, &buffers, arm, args, position)
    })?;
    Ok((read_f32(&buffers.q), read_f32(&buffers.k)))
}

fn run_packed_once(
    ctx: &MetalContext,
    args: &RopeMicroArgs,
    q_values: &[f32],
    k_values: &[f32],
    arm: PackedArm,
    tokens: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let buffers = make_buffers(ctx, q_values, k_values)?;
    run_command(ctx, |encoder| {
        encode_packed(ctx, encoder, &buffers, arm, args, tokens)
    })?;
    Ok((read_f32(&buffers.q), read_f32(&buffers.k)))
}

fn measure_interleaved<A: Copy>(
    ctx: &MetalContext,
    arms: &[A],
    runs: usize,
    mut encode: impl FnMut(A, &KernelEncoder) -> Result<()>,
) -> Result<Vec<Vec<f64>>> {
    for &arm in arms {
        run_command(ctx, |encoder| encode(arm, encoder))?;
    }

    let mut samples = vec![Vec::with_capacity(runs); arms.len()];
    for round in 0..runs {
        let mut order: Vec<usize> = (0..arms.len()).collect();
        order.rotate_left((round / 2) % arms.len());
        if round % 2 == 1 {
            order.reverse();
        }
        for index in order {
            let elapsed = run_command(ctx, |encoder| encode(arms[index], encoder))?;
            samples[index].push(elapsed);
        }
    }
    Ok(samples)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.is_empty() {
        0.0
    } else if sorted.len().is_multiple_of(2) {
        let upper = sorted.len() / 2;
        (sorted[upper - 1] + sorted[upper]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    }
}

fn stdev(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let average = mean(values);
    (values
        .iter()
        .map(|value| (value - average) * (value - average))
        .sum::<f64>()
        / (values.len() - 1) as f64)
        .sqrt()
}

fn timing_arms(names: &[&'static str], samples: Vec<Vec<f64>>, layers: usize) -> Vec<TimingArmRow> {
    let baseline = mean(&samples[0]);
    names
        .iter()
        .zip(samples)
        .map(|(&arm, samples_ms)| {
            let average = mean(&samples_ms);
            TimingArmRow {
                arm,
                mean_command_ms: average,
                median_command_ms: median(&samples_ms),
                stdev_command_ms: stdev(&samples_ms),
                mean_us_per_layer: average * 1_000.0 / layers as f64,
                speedup_vs_baseline: baseline / average,
                samples_ms,
            }
        })
        .collect()
}

fn validate(args: &RopeMicroArgs) -> Result<()> {
    ensure!(
        args.q_heads > 0 && args.kv_heads > 0,
        "head counts must be nonzero"
    );
    ensure!(args.head_dim > 0, "head_dim must be nonzero");
    ensure!(
        args.rotary_dim > 0
            && args.rotary_dim <= args.head_dim
            && args.rotary_dim.is_multiple_of(2),
        "rotary_dim must be positive, even, and at most head_dim"
    );
    ensure!(
        args.theta.is_finite() && args.theta > 1.0,
        "theta must be finite and > 1"
    );
    ensure!(
        args.layers > 0 && args.runs > 0,
        "layers and runs must be nonzero"
    );
    ensure!(
        !args.tokens.is_empty() && args.tokens.iter().all(|&n| n > 0),
        "tokens must be nonempty and positive"
    );
    ensure!(!args.positions.is_empty(), "positions must be nonempty");
    for &tokens in &args.tokens {
        let span = u32::try_from(tokens - 1).context("packed token count exceeds u32")?;
        args.start_position
            .checked_add(span)
            .context("packed position span exceeds u32")?;
    }
    Ok(())
}

pub fn run(args: RopeMicroArgs, build_identity: serde_json::Value) -> Result<()> {
    validate(&args)?;
    let ctx = MetalContext::new()?;
    let q_row = args
        .q_heads
        .checked_mul(args.head_dim)
        .context("Q row size overflow")?;
    let k_row = args
        .kv_heads
        .checked_mul(args.head_dim)
        .context("K row size overflow")?;
    let q_single = input_values(q_row, 37, 0.03125);
    let k_single = input_values(k_row, 29, 0.046875);

    let mut accuracy = Vec::new();
    let mut decode_diffs = [
        DiffAccumulator::default(),
        DiffAccumulator::default(),
        DiffAccumulator::default(),
    ];
    for &position in &args.positions {
        let baseline = run_decode_once(
            &ctx,
            &args,
            &q_single,
            &k_single,
            DecodeArm::Baseline,
            position,
        )?;
        for (index, arm) in [DecodeArm::Sincos, DecodeArm::Shared, DecodeArm::Minimax]
            .into_iter()
            .enumerate()
        {
            let candidate = run_decode_once(&ctx, &args, &q_single, &k_single, arm, position)?;
            decode_diffs[index].observe(&baseline.0, &candidate.0);
            decode_diffs[index].observe(&baseline.1, &candidate.1);
        }
    }
    for (arm, diff) in [DecodeArm::Sincos, DecodeArm::Shared, DecodeArm::Minimax]
        .into_iter()
        .zip(decode_diffs)
    {
        accuracy.push(AccuracyRow {
            workload: "decode",
            arm: arm.name(),
            cases: args.positions.len(),
            values_compared: diff.count,
            max_abs: diff.max_abs,
            rms_abs: diff.rms_abs(),
        });
    }

    let mut packed_diffs = [
        DiffAccumulator::default(),
        DiffAccumulator::default(),
        DiffAccumulator::default(),
    ];
    for &tokens in &args.tokens {
        let q_values = input_values(
            tokens
                .checked_mul(q_row)
                .context("packed Q size overflow")?,
            37,
            0.03125,
        );
        let k_values = input_values(
            tokens
                .checked_mul(k_row)
                .context("packed K size overflow")?,
            29,
            0.046875,
        );
        let baseline =
            run_packed_once(&ctx, &args, &q_values, &k_values, PackedArm::Split, tokens)?;
        for (index, arm) in [PackedArm::Paired, PackedArm::Shared, PackedArm::Minimax]
            .into_iter()
            .enumerate()
        {
            let candidate = run_packed_once(&ctx, &args, &q_values, &k_values, arm, tokens)?;
            packed_diffs[index].observe(&baseline.0, &candidate.0);
            packed_diffs[index].observe(&baseline.1, &candidate.1);
        }
    }
    for (arm, diff) in [PackedArm::Paired, PackedArm::Shared, PackedArm::Minimax]
        .into_iter()
        .zip(packed_diffs)
    {
        accuracy.push(AccuracyRow {
            workload: "prefill",
            arm: arm.name(),
            cases: args.tokens.len(),
            values_compared: diff.count,
            max_abs: diff.max_abs,
            rms_abs: diff.rms_abs(),
        });
    }

    let decode_buffers: Vec<ArmBuffers> = DecodeArm::ALL
        .iter()
        .map(|_| make_buffers(&ctx, &q_single, &k_single))
        .collect::<Result<_>>()?;
    let decode_samples = measure_interleaved(&ctx, &DecodeArm::ALL, args.runs, |arm, encoder| {
        for _ in 0..args.layers {
            encode_decode(
                &ctx,
                encoder,
                &decode_buffers[arm.index()],
                arm,
                &args,
                args.start_position,
            )?;
        }
        Ok(())
    })?;
    let mut timings = vec![TimingRow {
        workload: "decode",
        tokens: 1,
        layers_per_command: args.layers,
        arms: timing_arms(
            &DecodeArm::ALL.map(DecodeArm::name),
            decode_samples,
            args.layers,
        ),
    }];

    for &tokens in &args.tokens {
        let q_values = input_values(tokens * q_row, 37, 0.03125);
        let k_values = input_values(tokens * k_row, 29, 0.046875);
        let packed_buffers: Vec<ArmBuffers> = PackedArm::ALL
            .iter()
            .map(|_| make_buffers(&ctx, &q_values, &k_values))
            .collect::<Result<_>>()?;
        let samples = measure_interleaved(&ctx, &PackedArm::ALL, args.runs, |arm, encoder| {
            for _ in 0..args.layers {
                encode_packed(
                    &ctx,
                    encoder,
                    &packed_buffers[arm.index()],
                    arm,
                    &args,
                    tokens,
                )?;
            }
            Ok(())
        })?;
        timings.push(TimingRow {
            workload: "prefill",
            tokens,
            layers_per_command: args.layers,
            arms: timing_arms(&PackedArm::ALL.map(PackedArm::name), samples, args.layers),
        });

        let norm_rope_buffers: Vec<NormRopeBuffers> = NormRopeArm::ALL
            .iter()
            .map(|_| make_norm_rope_buffers(&ctx, &args, tokens))
            .collect::<Result<_>>()?;
        let norm_rope_samples =
            measure_interleaved(&ctx, &NormRopeArm::ALL, args.runs, |arm, encoder| {
                for _ in 0..args.layers {
                    encode_norm_rope(
                        &ctx,
                        encoder,
                        &norm_rope_buffers[arm.index()],
                        arm,
                        &args,
                        tokens,
                    )?;
                }
                Ok(())
            })?;
        timings.push(TimingRow {
            workload: "prefill_norm_rope",
            tokens,
            layers_per_command: args.layers,
            arms: timing_arms(
                &NormRopeArm::ALL.map(NormRopeArm::name),
                norm_rope_samples,
                args.layers,
            ),
        });
    }

    let report = RopeMicroReport {
        schema_version: 1,
        test: "rope-micro",
        device: ctx.device.name().to_string(),
        q_heads: args.q_heads,
        kv_heads: args.kv_heads,
        head_dim: args.head_dim,
        rotary_dim: args.rotary_dim,
        theta: args.theta,
        start_position: args.start_position,
        accuracy_positions: args.positions.clone(),
        runs: args.runs,
        accuracy,
        timings,
        build_identity,
    };

    match args.output {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
        OutputFormat::Text => {
            println!(
                "device={} q_heads={} kv_heads={} head_dim={} rotary_dim={} theta={} start={}",
                report.device,
                report.q_heads,
                report.kv_heads,
                report.head_dim,
                report.rotary_dim,
                report.theta,
                report.start_position
            );
            println!("accuracy\tworkload\tarm\tcases\tmax_abs\trms_abs");
            for row in &report.accuracy {
                println!(
                    "accuracy\t{}\t{}\t{}\t{:.9e}\t{:.9e}",
                    row.workload, row.arm, row.cases, row.max_abs, row.rms_abs
                );
            }
            println!("timing\tworkload\ttokens\tarm\tmean_ms\tstdev_ms\tus_per_layer\tspeedup");
            for row in &report.timings {
                for arm in &row.arms {
                    println!(
                        "timing\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.3}\t{:.4}x",
                        row.workload,
                        row.tokens,
                        arm.arm,
                        arm.mean_command_ms,
                        arm.stdev_command_ms,
                        arm.mean_us_per_layer,
                        arm.speedup_vs_baseline
                    );
                }
            }
        }
    }
    Ok(())
}
