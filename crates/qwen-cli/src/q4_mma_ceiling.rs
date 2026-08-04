use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};
use qwen_llm::{
    gguf::GgufFile,
    loader::Model,
    metal::{
        KernelEncoder, MetalContext, MetalTensor, Q4MatMatMmaCeilingArm, Q4MatMatNoDequantArm,
        encode_fill_f32, encode_mat_mat_q4_k_f32, encode_mat_mat_q4_k_mma_ceiling,
        encode_mat_mat_q4_k_no_dequant,
    },
    model::ArchKind,
    tensor::GgmlType,
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;

const N_IN: usize = 5120;
const N_OUT: usize = 17408;
const N_QUERY: usize = 1024;
const TENSOR_NAME: &str = "blk.0.ffn_gate.weight";
const GUARD_ELEMENTS: usize = 4096;
const GUARD_VALUE: f32 = -1234.5;
const NOMINAL_FLOPS: f64 = 2.0 * N_IN as f64 * N_OUT as f64 * N_QUERY as f64;

#[derive(Parser, Debug)]
pub struct Q4MmaCeilingArgs {
    /// Path to the exact dense-27B Q4_K_M GGUF fixture.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Untimed warmup dispatches per arm.
    #[arg(long, default_value = "12")]
    warmups: usize,
    /// Repetitions of the ten-sequence Williams design (ten samples/arm each).
    #[arg(long, default_value = "6")]
    sequence_repeats: usize,
    /// Runtime operand nonce for the synthetic attribution arms.
    #[arg(long, default_value = "1")]
    nonce: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Production,
    SourceSegmentsLiveNoDequant,
    NoSourceNoDequant,
    MmaOnly,
    MmaOnlyTgm8,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Self::Production => "a_production",
            Self::SourceSegmentsLiveNoDequant => "b_source_segments_live_no_dequant",
            Self::NoSourceNoDequant => "c_no_source_no_dequant",
            Self::MmaOnly => "e0_mma_only",
            Self::MmaOnlyTgm8 => "e8_mma_only_tgm8_cap_matched",
        }
    }
}

const SEQUENCES: [[Arm; 5]; 10] = [
    [
        Arm::Production,
        Arm::SourceSegmentsLiveNoDequant,
        Arm::MmaOnlyTgm8,
        Arm::NoSourceNoDequant,
        Arm::MmaOnly,
    ],
    [
        Arm::SourceSegmentsLiveNoDequant,
        Arm::NoSourceNoDequant,
        Arm::Production,
        Arm::MmaOnly,
        Arm::MmaOnlyTgm8,
    ],
    [
        Arm::NoSourceNoDequant,
        Arm::MmaOnly,
        Arm::SourceSegmentsLiveNoDequant,
        Arm::MmaOnlyTgm8,
        Arm::Production,
    ],
    [
        Arm::MmaOnly,
        Arm::MmaOnlyTgm8,
        Arm::NoSourceNoDequant,
        Arm::Production,
        Arm::SourceSegmentsLiveNoDequant,
    ],
    [
        Arm::MmaOnlyTgm8,
        Arm::Production,
        Arm::MmaOnly,
        Arm::SourceSegmentsLiveNoDequant,
        Arm::NoSourceNoDequant,
    ],
    [
        Arm::MmaOnly,
        Arm::NoSourceNoDequant,
        Arm::MmaOnlyTgm8,
        Arm::SourceSegmentsLiveNoDequant,
        Arm::Production,
    ],
    [
        Arm::MmaOnlyTgm8,
        Arm::MmaOnly,
        Arm::Production,
        Arm::NoSourceNoDequant,
        Arm::SourceSegmentsLiveNoDequant,
    ],
    [
        Arm::Production,
        Arm::MmaOnlyTgm8,
        Arm::SourceSegmentsLiveNoDequant,
        Arm::MmaOnly,
        Arm::NoSourceNoDequant,
    ],
    [
        Arm::SourceSegmentsLiveNoDequant,
        Arm::Production,
        Arm::NoSourceNoDequant,
        Arm::MmaOnlyTgm8,
        Arm::MmaOnly,
    ],
    [
        Arm::NoSourceNoDequant,
        Arm::SourceSegmentsLiveNoDequant,
        Arm::MmaOnly,
        Arm::Production,
        Arm::MmaOnlyTgm8,
    ],
];

#[derive(Debug, Serialize)]
struct Sample {
    ordinal: usize,
    repeat: usize,
    sequence: usize,
    position: usize,
    dispatch_predecessor: &'static str,
    sequence_wash_in: bool,
    arm: &'static str,
    gpu_ms: f64,
    wall_ms: f64,
    nominal_tflops: f64,
}

fn encode_arm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    arm: Arm,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    nonce: u32,
) -> Result<()> {
    match arm {
        Arm::Production => encode_mat_mat_q4_k_f32(ctx, enc, weight, x, y, N_IN, N_OUT, N_QUERY)?,
        Arm::SourceSegmentsLiveNoDequant => encode_mat_mat_q4_k_no_dequant(
            ctx,
            enc,
            weight,
            x,
            y,
            Q4MatMatNoDequantArm::SourceSegmentsLive,
            nonce,
        )?,
        Arm::NoSourceNoDequant => encode_mat_mat_q4_k_no_dequant(
            ctx,
            enc,
            weight,
            x,
            y,
            Q4MatMatNoDequantArm::NoSource,
            nonce,
        )?,
        Arm::MmaOnly => {
            encode_mat_mat_q4_k_mma_ceiling(ctx, enc, y, Q4MatMatMmaCeilingArm::Pure, nonce)?
        }
        Arm::MmaOnlyTgm8 => encode_mat_mat_q4_k_mma_ceiling(
            ctx,
            enc,
            y,
            Q4MatMatMmaCeilingArm::Tgm8CapMatched,
            nonce,
        )?,
    }
    Ok(())
}

fn run_dispatch(
    ctx: &MetalContext,
    arm: Arm,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    nonce: u32,
    poison: bool,
) -> Result<(f64, f64)> {
    let cmd = ctx
        .queue
        .commandBuffer()
        .context("q4 MMA ceiling command buffer")?;
    let enc = KernelEncoder::begin(&cmd);
    if poison {
        encode_fill_f32(ctx, &enc, y, f32::NAN)?;
    }
    encode_arm(ctx, &enc, arm, weight, x, y, nonce)?;
    enc.end();

    let wall_start = Instant::now();
    cmd.commit();
    cmd.waitUntilCompleted();
    let wall_ms = wall_start.elapsed().as_secs_f64() * 1e3;
    let status = cmd.status();
    let error = cmd.error();
    if status != MTLCommandBufferStatus::Completed || error.is_some() {
        bail!(
            "{} command failed: status={status:?} error={error:?}",
            arm.label()
        );
    }
    let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
    if !gpu_ms.is_finite() || gpu_ms <= 0.0 {
        bail!("{} returned invalid GPU time {gpu_ms}", arm.label());
    }
    Ok((gpu_ms, wall_ms))
}

fn fill_tensor(ctx: &MetalContext, tensor: &MetalTensor, value: f32) -> Result<()> {
    let cmd = ctx
        .queue
        .commandBuffer()
        .context("q4 attribution fill command buffer")?;
    let enc = KernelEncoder::begin(&cmd);
    encode_fill_f32(ctx, &enc, tensor, value)?;
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    let status = cmd.status();
    let error = cmd.error();
    if status != MTLCommandBufferStatus::Completed || error.is_some() {
        bail!("fill command failed: status={status:?} error={error:?}");
    }
    Ok(())
}

fn read_output(y: &MetalTensor) -> Vec<f32> {
    let n = N_QUERY * N_OUT;
    unsafe {
        let ptr = (y.buffer.contents().as_ptr() as *const u8).add(y.offset as usize) as *const f32;
        std::slice::from_raw_parts(ptr, n).to_vec()
    }
}

fn sha256_f32(values: &[f32]) -> String {
    let digest = Sha256::digest(bytemuck::cast_slice(values));
    format!("{digest:x}")
}

fn validate_guards(storage: &MetalTensor) -> Result<()> {
    let output_elements = N_QUERY * N_OUT;
    let expected_bits = GUARD_VALUE.to_bits();
    unsafe {
        let ptr = (storage.buffer.contents().as_ptr() as *const u8).add(storage.offset as usize)
            as *const f32;
        for index in 0..GUARD_ELEMENTS {
            if ptr.add(index).read().to_bits() != expected_bits {
                bail!("output prefix guard changed at index {index}");
            }
        }
        let suffix = GUARD_ELEMENTS + output_elements;
        for index in 0..GUARD_ELEMENTS {
            if ptr.add(suffix + index).read().to_bits() != expected_bits {
                bail!("output suffix guard changed at index {index}");
            }
        }
    }
    Ok(())
}

fn expected_mma_value(nonce: u32, output_row: usize, query_row: usize) -> f32 {
    let a_base = (1 + (nonce & 1)) as f32 / 256.0;
    let b_base = (1 + ((nonce >> 1) & 1)) as f32 / 256.0;
    let a_index = (output_row % 32) / 8 + 1;
    let b_index = (query_row % 16) / 8 + 1;
    N_IN as f32 * (a_base * a_index as f32) * (b_base * b_index as f32)
}

fn expected_no_dequant_value(nonce: u32, activation: f32) -> f32 {
    let a_base = (1 + (nonce & 1)) as f32 / 256.0;
    84_480.0 * a_base * activation
}

fn validate_constant_output(values: &[f32], expected: f32, arm: Arm) -> Result<()> {
    if values.len() != N_QUERY * N_OUT {
        bail!("{} returned {} values", arm.label(), values.len());
    }
    if let Some(index) = values
        .iter()
        .position(|value| value.to_bits() != expected.to_bits())
    {
        bail!(
            "{} mismatch at output index {index}: {:?} != {:?}",
            arm.label(),
            values[index],
            expected
        );
    }
    Ok(())
}

fn validate_mma_output(values: &[f32], nonce: u32, arm: Arm) -> Result<()> {
    if values.len() != N_QUERY * N_OUT {
        bail!("{} returned {} values", arm.label(), values.len());
    }
    for query_row in 0..N_QUERY {
        for output_row in 0..N_OUT {
            let index = query_row * N_OUT + output_row;
            let actual = values[index];
            let expected = expected_mma_value(nonce, output_row, query_row);
            if actual.to_bits() != expected.to_bits() {
                bail!(
                    "{} mismatch at q={query_row} out={output_row}: {actual:?} != {expected:?}",
                    arm.label()
                );
            }
        }
    }
    Ok(())
}

fn validate_mma_pair(
    ctx: &MetalContext,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    nonce: u32,
) -> Result<Vec<f32>> {
    run_dispatch(ctx, Arm::MmaOnly, weight, x, y, nonce, true)?;
    let e0 = read_output(y);
    validate_mma_output(&e0, nonce, Arm::MmaOnly)?;
    run_dispatch(ctx, Arm::MmaOnly, weight, x, y, nonce, true)?;
    let e0_repeat = read_output(y);
    if let Some(index) = e0
        .iter()
        .zip(&e0_repeat)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        bail!("E0 is not repeatable at output index {index}");
    }

    run_dispatch(ctx, Arm::MmaOnlyTgm8, weight, x, y, nonce, true)?;
    let e8 = read_output(y);
    validate_mma_output(&e8, nonce, Arm::MmaOnlyTgm8)?;
    run_dispatch(ctx, Arm::MmaOnlyTgm8, weight, x, y, nonce, true)?;
    let e8_repeat = read_output(y);
    if let Some(index) = e8
        .iter()
        .zip(&e8_repeat)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        bail!("E8 is not repeatable at output index {index}");
    }
    if let Some(index) = e0
        .iter()
        .zip(&e8)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        bail!("E0/E8 differ at output index {index} for nonce {nonce}");
    }
    Ok(e0)
}

fn validate_no_dequant_pair(
    ctx: &MetalContext,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    nonce: u32,
    activation: f32,
) -> Result<Vec<f32>> {
    let expected = expected_no_dequant_value(nonce, activation);
    run_dispatch(
        ctx,
        Arm::SourceSegmentsLiveNoDequant,
        weight,
        x,
        y,
        nonce,
        true,
    )?;
    let b = read_output(y);
    validate_constant_output(&b, expected, Arm::SourceSegmentsLiveNoDequant)?;
    run_dispatch(
        ctx,
        Arm::SourceSegmentsLiveNoDequant,
        weight,
        x,
        y,
        nonce,
        true,
    )?;
    let b_repeat = read_output(y);
    if let Some(index) = b
        .iter()
        .zip(&b_repeat)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        bail!("B is not repeatable at output index {index}");
    }

    run_dispatch(ctx, Arm::NoSourceNoDequant, weight, x, y, nonce, true)?;
    let c = read_output(y);
    validate_constant_output(&c, expected, Arm::NoSourceNoDequant)?;
    if let Some(index) = b
        .iter()
        .zip(&c)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        bail!("B/C differ at output index {index}");
    }
    run_dispatch(ctx, Arm::NoSourceNoDequant, weight, x, y, nonce, true)?;
    let c_repeat = read_output(y);
    if let Some(index) = c
        .iter()
        .zip(&c_repeat)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        bail!("C is not repeatable at output index {index}");
    }
    Ok(b)
}

fn validate_outputs(
    ctx: &MetalContext,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    nonce: u32,
) -> Result<Value> {
    run_dispatch(ctx, Arm::Production, weight, x, y, nonce, true)?;
    let production = read_output(y);
    if production.iter().any(|value| !value.is_finite()) {
        bail!("production output retained poison or non-finite values");
    }
    if production.iter().all(|value| *value == 0.0) {
        bail!("production output is identically zero");
    }
    run_dispatch(ctx, Arm::Production, weight, x, y, nonce, true)?;
    let production_repeat = read_output(y);
    if let Some(index) = production
        .iter()
        .zip(&production_repeat)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        bail!("production output is not repeatable at index {index}");
    }

    let bc = validate_no_dequant_pair(ctx, weight, x, y, nonce, 1.0)?;
    let bc_alternate_nonce = nonce ^ 1;
    let bc_alternate = validate_no_dequant_pair(ctx, weight, x, y, bc_alternate_nonce, 1.0)?;
    if bc
        .iter()
        .zip(&bc_alternate)
        .all(|(a, b)| a.to_bits() == b.to_bits())
    {
        bail!("nonce bit 0 did not change B/C output");
    }
    fill_tensor(ctx, x, 0.5)?;
    validate_no_dequant_pair(ctx, weight, x, y, nonce, 0.5)?;
    fill_tensor(ctx, x, 1.0)?;

    let e0 = validate_mma_pair(ctx, weight, x, y, nonce)?;
    let alternate_a_nonce = nonce ^ 1;
    let alternate_a = validate_mma_pair(ctx, weight, x, y, alternate_a_nonce)?;
    if e0
        .iter()
        .zip(&alternate_a)
        .all(|(a, b)| a.to_bits() == b.to_bits())
    {
        bail!("nonce bit 0 did not change MMA output");
    }
    let alternate_b_nonce = nonce ^ 2;
    let alternate_b = validate_mma_pair(ctx, weight, x, y, alternate_b_nonce)?;
    if e0
        .iter()
        .zip(&alternate_b)
        .all(|(a, b)| a.to_bits() == b.to_bits())
    {
        bail!("nonce bit 1 did not change MMA output");
    }

    Ok(json!({
        "production_all_finite": true,
        "production_nonzero": true,
        "production_repeat_bit_exact": true,
        "production_sha256": sha256_f32(&production),
        "b_exact_analytic": true,
        "c_exact_analytic": true,
        "b_c_bit_exact": true,
        "b_c_repeat_bit_exact": true,
        "b_c_alternate_nonce": bc_alternate_nonce,
        "b_c_nonce_changes_output": true,
        "b_c_half_activation_exact": true,
        "timed_activation_restored_to_one": true,
        "e0_exact_analytic": true,
        "e8_exact_analytic": true,
        "e0_e8_bit_exact": true,
        "e_repeat_bit_exact": true,
        "nonce": nonce,
        "meaningful_nonce_bits": [0, 1],
        "alternate_a_nonce": alternate_a_nonce,
        "alternate_b_nonce": alternate_b_nonce,
        "both_nonce_bits_change_output": true,
        "elements_checked_per_arm": N_QUERY * N_OUT,
    }))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_stdev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let avg = mean(values);
    let variance = values
        .iter()
        .map(|value| (value - avg) * (value - avg))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    }
}

fn summarize(samples: &[Sample], arm: Arm) -> Value {
    let selected: Vec<&Sample> = samples
        .iter()
        .filter(|sample| sample.arm == arm.label())
        .collect();
    let gpu_ms: Vec<f64> = selected.iter().map(|sample| sample.gpu_ms).collect();
    let tflops: Vec<f64> = selected
        .iter()
        .map(|sample| sample.nominal_tflops)
        .collect();
    json!({
        "arm": arm.label(),
        "samples": selected.len(),
        "mean_gpu_ms": mean(&gpu_ms),
        "median_gpu_ms": median(&gpu_ms),
        "sample_stdev_gpu_ms": sample_stdev(&gpu_ms),
        "mean_nominal_tflops": mean(&tflops),
        "median_nominal_tflops": median(&tflops),
        "sample_stdev_nominal_tflops": sample_stdev(&tflops),
    })
}

fn pipeline_row(ctx: &MetalContext, kernel: &str, dynamic_tgm_bytes: usize) -> Result<Value> {
    let info = ctx.pipeline_info(kernel)?;
    Ok(json!({
        "kernel": kernel,
        "dynamic_tgm_bytes": dynamic_tgm_bytes,
        "thread_execution_width": info.thread_execution_width,
        "max_total_threads_per_threadgroup": info.max_total_threads_per_threadgroup,
        "static_threadgroup_memory_length": info.static_threadgroup_memory_length,
        "supports_indirect_command_buffers": info.supports_indirect_command_buffers,
    }))
}

pub fn run(args: Q4MmaCeilingArgs, build_identity: Value) -> Result<()> {
    if args.warmups != 12 || args.sequence_repeats != 6 {
        bail!("canonical floor requires warmups=12 and sequence_repeats=6");
    }
    let n64_env = std::env::var("QWEN_MATMAT_Q4_K_N64").ok();
    if matches!(
        n64_env.as_deref(),
        Some("0" | "false" | "FALSE" | "no" | "NO")
    ) {
        bail!("QWEN_MATMAT_Q4_K_N64 disables the production N64 control");
    }

    let gguf = GgufFile::open(&args.model)?;
    let model = Model::from_gguf(&gguf)?;
    if model.arch.kind != ArchKind::Dense
        || model.arch.n_layer != 64
        || model.arch.hidden_size as usize != N_IN
        || model.arch.intermediate_size as usize != N_OUT
        || model.arch.vocab_size != 248_320
    {
        bail!(
            "fixture architecture does not match the dense-27B anchor: {:?}",
            model.arch
        );
    }
    let descriptor = gguf
        .tensors
        .iter()
        .find(|tensor| tensor.name == TENSOR_NAME)
        .ok_or_else(|| anyhow!("missing tensor {TENSOR_NAME}"))?;
    if descriptor.dtype != GgmlType::Q4_K
        || descriptor.shape.as_slice() != [N_IN as u64, N_OUT as u64]
    {
        bail!(
            "{} must be Q4_K [{N_IN},{N_OUT}], got {:?} {:?}",
            descriptor.name,
            descriptor.dtype,
            descriptor.shape
        );
    }

    let tensor_bytes = gguf.try_slice(descriptor)?;
    let tensor_sha256 = format!("{:x}", Sha256::digest(tensor_bytes));
    let ctx = MetalContext::new()?;
    let weight = MetalTensor::from_bytes(
        &ctx,
        tensor_bytes,
        vec![N_IN as u64, N_OUT as u64],
        GgmlType::Q4_K,
    )?;
    let x = MetalTensor::zeros_f32(&ctx, vec![N_QUERY as u64, N_IN as u64])?;
    let output_elements = N_QUERY * N_OUT;
    let y_storage =
        MetalTensor::zeros_f32(&ctx, vec![(output_elements + 2 * GUARD_ELEMENTS) as u64])?;
    let y = y_storage.view_subrange(GUARD_ELEMENTS as u64, vec![N_QUERY as u64, N_OUT as u64]);

    let init_cmd = ctx
        .queue
        .commandBuffer()
        .context("q4 MMA ceiling init command")?;
    let init_enc = KernelEncoder::begin(&init_cmd);
    encode_fill_f32(&ctx, &init_enc, &x, 1.0)?;
    encode_fill_f32(&ctx, &init_enc, &y_storage, GUARD_VALUE)?;
    init_enc.end();
    init_cmd.commit();
    init_cmd.waitUntilCompleted();
    let init_status = init_cmd.status();
    let init_error = init_cmd.error();
    if init_status != MTLCommandBufferStatus::Completed || init_error.is_some() {
        bail!("initialization failed: status={init_status:?} error={init_error:?}");
    }
    validate_guards(&y_storage)?;

    let validation = validate_outputs(&ctx, &weight, &x, &y, args.nonce)?;
    validate_guards(&y_storage)?;

    let warmup_order = [
        Arm::Production,
        Arm::SourceSegmentsLiveNoDequant,
        Arm::NoSourceNoDequant,
        Arm::MmaOnly,
        Arm::MmaOnlyTgm8,
    ];
    for index in 0..args.warmups {
        for offset in 0..warmup_order.len() {
            let arm = warmup_order[(index + offset) % warmup_order.len()];
            run_dispatch(&ctx, arm, &weight, &x, &y, args.nonce, false)?;
        }
    }

    let mut samples = Vec::with_capacity(args.sequence_repeats * SEQUENCES.len() * 5);
    for repeat in 0..args.sequence_repeats {
        for block in 0..SEQUENCES.len() {
            let sequence = (block + repeat) % SEQUENCES.len();
            let order = &SEQUENCES[sequence];
            run_dispatch(&ctx, order[0], &weight, &x, &y, args.nonce, false)?;
            let mut previous_arm = order[0];
            for (position, &arm) in order.iter().enumerate() {
                let (gpu_ms, wall_ms) =
                    run_dispatch(&ctx, arm, &weight, &x, &y, args.nonce, false)?;
                samples.push(Sample {
                    ordinal: samples.len() + 1,
                    repeat: repeat + 1,
                    sequence: sequence + 1,
                    position: position + 1,
                    dispatch_predecessor: previous_arm.label(),
                    sequence_wash_in: position == 0,
                    arm: arm.label(),
                    gpu_ms,
                    wall_ms,
                    nominal_tflops: NOMINAL_FLOPS / (gpu_ms * 1e9),
                });
                previous_arm = arm;
            }
        }
    }
    validate_guards(&y_storage)?;

    let metadata =
        std::fs::metadata(&args.model).with_context(|| format!("stat {}", args.model.display()))?;
    let arms = [
        Arm::Production,
        Arm::SourceSegmentsLiveNoDequant,
        Arm::NoSourceNoDequant,
        Arm::MmaOnly,
        Arm::MmaOnlyTgm8,
    ];
    let summaries: Vec<Value> = arms
        .iter()
        .copied()
        .map(|arm| summarize(&samples, arm))
        .collect();
    let sequences: Vec<Vec<&str>> = SEQUENCES
        .iter()
        .map(|order| order.iter().map(|arm| arm.label()).collect())
        .collect();
    let pipeline_rows = vec![
        pipeline_row(&ctx, "kernel_mat_mat_q4_K_f32_n64", 8192)?,
        pipeline_row(
            &ctx,
            "kernel_mat_mat_q4_K_f32_n64_source_segments_live_no_dequant",
            8192,
        )?,
        pipeline_row(
            &ctx,
            "kernel_mat_mat_q4_K_f32_n64_no_source_no_dequant",
            8192,
        )?,
        pipeline_row(&ctx, "kernel_mat_mat_q4_K_f32_n64_mma_ceiling", 0)?,
        pipeline_row(&ctx, "kernel_mat_mat_q4_K_f32_n64_mma_ceiling_tgm8", 8192)?,
    ];

    let row = json!({
        "schema_version": 2,
        "test": "q4_matmat_attribution",
        "claim_scope": "production-grid synthetic attribution bounds; no production authority",
        "device": ctx.describe(),
        "build_identity": build_identity,
        "model": {
            "path": args.model,
            "primary_file_bytes": metadata.len(),
            "total_mapped_bytes": gguf.total_mapped_len(),
            "shards": gguf.shard_count(),
            "shard_mapped_bytes": gguf.shard_mapped_lengths(),
            "architecture_matches_qwen3_27b": true,
        },
        "tensor": {
            "name": descriptor.name,
            "dtype": "Q4_K",
            "shape": [N_IN, N_OUT],
            "bytes": tensor_bytes.len(),
            "sha256": tensor_sha256,
        },
        "geometry": {
            "n_in": N_IN,
            "n_out": N_OUT,
            "n_query": N_QUERY,
            "grid": [N_QUERY / 64, N_OUT / 64, 1],
            "threads_per_threadgroup": 256,
            "simdgroups_per_threadgroup": 8,
            "k_step": 32,
            "k_loops": N_IN / 32,
            "inner_substeps": 4,
            "mma_calls_per_substep_per_simdgroup": 8,
            "nominal_flops": NOMINAL_FLOPS as u64,
        },
        "arms": [
            {
                "label": Arm::Production.label(),
                "kernel": "kernel_mat_mat_q4_K_f32_n64",
                "dynamic_tgm_bytes": 8192,
            },
            {
                "label": Arm::SourceSegmentsLiveNoDequant.label(),
                "kernel": "kernel_mat_mat_q4_K_f32_n64_source_segments_live_no_dequant",
                "dynamic_tgm_bytes": 8192,
                "qualification": "two aligned volatile source-segment proxies; not exact production load timing",
            },
            {
                "label": Arm::NoSourceNoDequant.label(),
                "kernel": "kernel_mat_mat_q4_K_f32_n64_no_source_no_dequant",
                "dynamic_tgm_bytes": 8192,
            },
            {
                "label": Arm::MmaOnly.label(),
                "kernel": "kernel_mat_mat_q4_K_f32_n64_mma_ceiling",
                "dynamic_tgm_bytes": 0,
            },
            {
                "label": Arm::MmaOnlyTgm8.label(),
                "kernel": "kernel_mat_mat_q4_K_f32_n64_mma_ceiling_tgm8",
                "dynamic_tgm_bytes": 8192,
                "qualification": "8-KiB-cap-matched, not occupancy-equivalent",
            },
        ],
        "pipeline_reflection": pipeline_rows,
        "measurement": {
            "warmups_per_arm": args.warmups,
            "sequence_repeats": args.sequence_repeats,
            "samples_per_arm": args.sequence_repeats * SEQUENCES.len(),
            "sequences": sequences,
            "sequence_order": "ten-sequence Williams design, rotated by repeat",
            "unscored_wash_in": "one first-arm dispatch before every sequence",
            "nonce": args.nonce,
            "qwen_matmat_q4_k_n64_env": n64_env,
            "one_dispatch_per_command_buffer": true,
            "primary_clock": "MTLCommandBuffer GPUStartTime/GPUEndTime",
            "wall_clock": "commit through waitUntilCompleted; encoding excluded",
            "command_status_required": "Completed with no error",
        },
        "validation": validation,
        "guards": {
            "elements_each_side": GUARD_ELEMENTS,
            "value_bits": GUARD_VALUE.to_bits(),
            "intact_after_validation_and_measurement": true,
        },
        "summaries": summaries,
        "samples": samples,
    });
    println!("{}", serde_json::to_string(&row)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn williams_sequences_balance_position_and_predecessor() {
        let arms = [
            Arm::Production,
            Arm::SourceSegmentsLiveNoDequant,
            Arm::NoSourceNoDequant,
            Arm::MmaOnly,
            Arm::MmaOnlyTgm8,
        ];
        let mut positions = BTreeMap::new();
        let mut predecessors = BTreeMap::new();
        let mut first = BTreeMap::new();
        for order in SEQUENCES {
            *first.entry(order[0].label()).or_insert(0usize) += 1;
            for (position, arm) in order.into_iter().enumerate() {
                *positions.entry((arm.label(), position)).or_insert(0usize) += 1;
                if position > 0 {
                    *predecessors
                        .entry((order[position - 1].label(), arm.label()))
                        .or_insert(0usize) += 1;
                }
            }
        }
        for arm in arms {
            assert_eq!(first[arm.label()], 2);
            for position in 0..5 {
                assert_eq!(positions[&(arm.label(), position)], 2);
            }
        }
        for predecessor in arms {
            for arm in arms {
                if predecessor != arm {
                    assert_eq!(predecessors[&(predecessor.label(), arm.label())], 2);
                }
            }
        }
    }

    #[test]
    fn analytic_values_are_finite_exact_powers_of_two() {
        for nonce in [0, 1, 2, 3, u32::MAX] {
            for output_row in [0, 7, 8, 31, 32, N_OUT - 1] {
                for query_row in [0, 7, 8, 15, 16, N_QUERY - 1] {
                    let value = expected_mma_value(nonce, output_row, query_row);
                    assert!(value.is_finite());
                    assert!(value > 0.0);
                    assert_eq!(value * 1024.0, (value * 1024.0).round());
                }
            }
        }
        assert_ne!(
            expected_mma_value(1, 0, 0).to_bits(),
            expected_mma_value(1 ^ 1, 0, 0).to_bits()
        );
        assert_eq!(expected_no_dequant_value(0, 1.0), 330.0);
        assert_eq!(expected_no_dequant_value(1, 1.0), 660.0);
        assert_eq!(expected_no_dequant_value(1, 0.5), 330.0);
    }
}
