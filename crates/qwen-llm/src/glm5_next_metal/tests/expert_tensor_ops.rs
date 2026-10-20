use super::*;
use crate::glm5_next_metal::tests::perf_lease;
use crate::metal::{KernelEncoder, encode_moe_swiglu_clamped_f32_grouped_slots_split};
use objc2_metal::MTLSize;
use serde_json::json;
use std::{io::Write, time::Instant};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TensorArgs {
    hidden: u32,
    ffn: u32,
    rows: u32,
    padded_rows: u32,
    padded_ffn: u32,
    expert: u32,
    expert_stride_bytes: u32,
    clamp_value: f32,
}

fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
    let count = tensor.n_elements() as usize;
    unsafe {
        std::slice::from_raw_parts(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>(),
            count,
        )
        .to_vec()
    }
}

fn production_call(
    ctx: &MetalContext,
    gate: &MetalTensor,
    up: &MetalTensor,
    input: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    output: &MetalTensor,
    hidden: usize,
    ffn: usize,
    experts: usize,
    rows: usize,
    clamp_value: f32,
) -> f64 {
    let command = ctx
        .queue
        .commandBuffer()
        .expect("production command buffer");
    let encoder = KernelEncoder::begin(&command);
    encode_moe_swiglu_clamped_f32_grouped_slots_split(
        ctx,
        &encoder,
        gate,
        up,
        input,
        counts,
        ids,
        output,
        hidden,
        ffn,
        experts,
        1,
        rows,
        clamp_value,
        17,
    )
    .expect("production grouped gate/up");
    encoder.end();
    let start = Instant::now();
    command.commit();
    crate::metal::wait_completed(&command).expect("production command completion");
    start.elapsed().as_secs_f64() * 1e3
}

fn tensor_call(
    ctx: &MetalContext,
    gate: &MetalTensor,
    up: &MetalTensor,
    input: &MetalTensor,
    partial: &MetalTensor,
    output: &MetalTensor,
    args: &TensorArgs,
) -> f64 {
    let command = ctx.queue.commandBuffer().expect("tensor command buffer");
    let encoder = KernelEncoder::begin(&command);
    let partial_pipeline = ctx
        .tensor_pipeline("kernel_tensor_glm_iq2s_expert_partial")
        .expect("IQ2_S TensorOps pipeline");
    encoder.set_pipeline(&partial_pipeline);
    encoder.set_bytes(0, args);
    encoder.set_tensor(1, gate);
    encoder.set_tensor(2, up);
    encoder.set_tensor(3, input);
    encoder.set_tensor(4, partial);
    encoder.set_threadgroup_memory(0, 2 * 512 * std::mem::size_of::<half::f16>());
    encoder.dispatch(
        MTLSize {
            width: (args.ffn as usize).div_ceil(16),
            height: (args.rows as usize).div_ceil(16),
            depth: args.hidden as usize / 32,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    let finish_pipeline = ctx
        .tensor_pipeline("kernel_tensor_glm_iq2s_expert_finish")
        .expect("IQ2_S TensorOps finish pipeline");
    encoder.set_pipeline(&finish_pipeline);
    encoder.set_bytes(0, args);
    encoder.set_tensor(1, partial);
    encoder.set_tensor(2, output);
    encoder.dispatch(
        MTLSize {
            width: (args.rows as usize * args.ffn as usize).div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    encoder.end();
    let start = Instant::now();
    command.commit();
    crate::metal::wait_completed(&command).expect("tensor command completion");
    start.elapsed().as_secs_f64() * 1e3
}

#[test]
#[ignore = "release, leased GPU; Maya IQ2_S and new GLM53_STAGE_OUT JSON path"]
fn maya_iq2_s_glm_shaped_expert_tensor_ab() {
    assert!(!cfg!(debug_assertions), "requires release test binary");
    let _lease = perf_lease();
    let model_path = std::env::var_os("GLM53_GGUF").expect("GLM53_GGUF must name Maya");
    let output_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(std::env::var_os("GLM53_STAGE_OUT").expect("GLM53_STAGE_OUT"));
    let mut output_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_path)
        .unwrap();
    let ctx = MetalContext::new().expect("Metal context");
    assert!(
        ctx.tensor_ops_status().is_available(),
        "Metal 4 TensorOps required"
    );
    let gguf = GgufFile::open(std::path::PathBuf::from(model_path)).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load Maya IQ2_S");
    let (layer, moe) = weights
        .blocks
        .iter()
        .enumerate()
        .find_map(|(layer, block)| match &block.ffn {
            FfnTensors::Moe(moe) => Some((layer, moe)),
            _ => None,
        })
        .expect("Maya routed expert layer");
    let gate = &moe.gate_experts;
    let up = &moe.up_experts;
    assert_eq!(gate.dtype, GgmlType::IQ2_S);
    assert_eq!(up.dtype, GgmlType::IQ2_S);
    assert_eq!(gate.shape, up.shape);
    let [hidden, ffn, experts] = <[u64; 3]>::try_from(gate.shape.as_slice()).unwrap();
    let (hidden, ffn, experts) = (hidden as usize, ffn as usize, experts as usize);
    let clamp_value = weights.config.swiglu_clamp;
    let chosen_expert = 0usize;
    let mut cases = Vec::new();
    for rows in [10usize, 39, 256] {
        let input_values = (0..hidden * rows)
            .map(|index| ((index.wrapping_mul(37) % 251) as f32 - 125.0) / 256.0)
            .collect::<Vec<_>>();
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&input_values),
            vec![hidden as u64, rows as u64],
            GgmlType::F32,
        )
        .unwrap();
        let mut count_values = vec![0i32; experts];
        count_values[chosen_expert] = rows as i32;
        let mut id_values = vec![0i32; experts * rows];
        for row in 0..rows {
            id_values[chosen_expert * rows + row] = row as i32;
        }
        let counts = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&count_values),
            vec![experts as u64],
            GgmlType::I32,
        )
        .unwrap();
        let ids = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&id_values),
            vec![experts as u64, rows as u64],
            GgmlType::I32,
        )
        .unwrap();
        let production_output =
            MetalTensor::zeros_f32(&ctx, vec![ffn as u64, rows as u64]).unwrap();
        let tensor_output = MetalTensor::zeros_f32(&ctx, vec![ffn as u64, rows as u64]).unwrap();
        let padded_rows = rows.div_ceil(16) * 16;
        let padded_ffn = ffn.div_ceil(16) * 16;
        let tiles_k = hidden / 32;
        let partial =
            MetalTensor::zeros_f32(&ctx, vec![(2 * tiles_k * padded_rows * padded_ffn) as u64])
                .unwrap();
        let args = TensorArgs {
            hidden: hidden as u32,
            ffn: ffn as u32,
            rows: rows as u32,
            padded_rows: padded_rows as u32,
            padded_ffn: padded_ffn as u32,
            expert: chosen_expert as u32,
            expert_stride_bytes: (gate.n_bytes() as usize / experts) as u32,
            clamp_value,
        };
        let tensor_run = || tensor_call(&ctx, gate, up, &input, &partial, &tensor_output, &args);
        let production_run = || {
            production_call(
                &ctx,
                gate,
                up,
                &input,
                &counts,
                &ids,
                &production_output,
                hidden,
                ffn,
                experts,
                rows,
                clamp_value,
            )
        };
        production_run();
        tensor_run();
        let mut production_ms = Vec::new();
        let mut tensor_ms = Vec::new();
        for repetition in 0..5 {
            if repetition % 2 == 0 {
                production_ms.push(production_run());
                tensor_ms.push(tensor_run());
            } else {
                tensor_ms.push(tensor_run());
                production_ms.push(production_run());
            }
        }
        production_run();
        tensor_run();
        let production_values = read_f32(&production_output);
        let tensor_values = read_f32(&tensor_output);
        if rows == 10 {
            eprintln!(
                "production_head={:?} tensor_ops_head={:?}",
                &production_values[..8],
                &tensor_values[..8],
            );
        }
        let mut max_abs = 0.0f32;
        let mut error_square = 0.0f64;
        let mut reference_square = 0.0f64;
        for (&reference, &candidate) in production_values.iter().zip(&tensor_values) {
            max_abs = max_abs.max((candidate - reference).abs());
            error_square += f64::from(candidate - reference).powi(2);
            reference_square += f64::from(reference).powi(2);
        }
        let relative_rms = (error_square / reference_square).sqrt();
        let production_mean = production_ms.iter().sum::<f64>() / production_ms.len() as f64;
        let tensor_mean = tensor_ms.iter().sum::<f64>() / tensor_ms.len() as f64;
        let flops = 4.0 * rows as f64 * hidden as f64 * ffn as f64;
        let case = json!({
            "M": rows,
            "layer": layer,
            "expert": chosen_expert,
            "input": "same deterministic F32 rows, cast to F16 by each kernel",
            "production_kernel": "kernel_moe_swiglu_clamped_iq2_s_f32_grouped_slots_generic plus n32 split at M>=17",
            "production_ms": production_ms,
            "production_mean_ms": production_mean,
            "production_tflops": flops / (production_mean * 1e9),
            "tensor_ops_ms": tensor_ms,
            "tensor_ops_mean_ms": tensor_mean,
            "tensor_ops_tflops": flops / (tensor_mean * 1e9),
            "max_abs": max_abs,
            "relative_rms": relative_rms,
        });
        eprintln!(
            "M={rows} production={production_mean:.3}ms/{:.3}TFLOP/s tensor={tensor_mean:.3}ms/{:.3}TFLOP/s max_abs={max_abs:.6} relative_rms={relative_rms:.6}",
            flops / (production_mean * 1e9),
            flops / (tensor_mean * 1e9),
        );
        cases.push(case);
    }
    serde_json::to_writer_pretty(
        &mut output_file,
        &json!({
            "schema":"glm53.maya_iq2_s_expert_tensor_ab.v1",
            "model":"GLM-5.3-Flash-Maya-M-IQ2_S",
            "layer":layer,
            "gate_shape":gate.shape,
            "up_shape":up.shape,
            "weight_dtype":format!("{:?}",gate.dtype),
            "input_dtype":"F32, rounded to F16 inside both paths",
            "accumulation":"F32",
            "repetitions_per_arm":5,
            "order":"warm production,tensor; then five interleaved AB/BA pairs; then untimed comparison calls",
            "cases":cases,
        }),
    )
    .unwrap();
    writeln!(output_file).unwrap();
}
