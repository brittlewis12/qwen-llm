use objc2_metal::MTLBuffer;
use qwen_llm::codec::dequant_to_f32;
use qwen_llm::metal::{MetalContext, MetalTensor};
use qwen_llm::tensor::{GgmlType, TensorDesc};
use serde_json::json;
use std::ffi::c_void;
use std::hint::black_box;
use std::time::Instant;

const ROW_ELEMENTS: usize = 4096;
const BLOCK_ELEMENTS: usize = 256;
const BLOCK_BYTES: usize = 144;

#[derive(Clone, Copy, Debug)]
enum Arm {
    Zeroed,
    Uninitialized,
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    allocation_ms: f64,
    producer_ms: f64,
    total_ms: f64,
    checksum: u64,
    digest: [u8; 32],
}

#[derive(Clone, Copy, Debug)]
struct MetalSample {
    host_dequant_ms: f64,
    metal_allocation_or_copy_ms: f64,
    direct_producer_ms: f64,
    total_ms: f64,
    checksum: u64,
    digest: [u8; 32],
}

fn parse_usize(flag: &str, default: usize) -> usize {
    let mut args = std::env::args();
    while let Some(argument) = args.next() {
        if argument == flag {
            return args
                .next()
                .unwrap_or_else(|| panic!("{flag} requires a value"))
                .parse()
                .unwrap_or_else(|_| panic!("{flag} requires a positive integer"));
        }
    }
    default
}

fn parse_string(flag: &str, default: &str) -> String {
    let mut args = std::env::args();
    while let Some(argument) = args.next() {
        if argument == flag {
            return args
                .next()
                .unwrap_or_else(|| panic!("{flag} requires a value"));
        }
    }
    default.to_string()
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

fn checksum(values: &[f32]) -> u64 {
    let stride = (values.len() / 4096).max(1);
    values.iter().step_by(stride).enumerate().fold(
        0xcbf2_9ce4_8422_2325u64,
        |state, (index, value)| {
            state.wrapping_mul(0x0000_0100_0000_01b3) ^ u64::from(value.to_bits()) ^ index as u64
        },
    )
}

fn run_arm(
    arm: Arm,
    source: &[u8],
    elements: usize,
    to_float: unsafe extern "C" fn(*const c_void, *mut f32, i64),
) -> Sample {
    let total_started = Instant::now();
    let allocation_started = Instant::now();
    let mut output = match arm {
        Arm::Zeroed => vec![0.0f32; elements],
        Arm::Uninitialized => {
            let mut output = Vec::new();
            output
                .try_reserve_exact(elements)
                .expect("reserve dequant output");
            output
        }
    };
    let allocation_ms = allocation_started.elapsed().as_secs_f64() * 1e3;

    let producer_started = Instant::now();
    unsafe {
        to_float(
            source.as_ptr().cast::<c_void>(),
            output.as_mut_ptr(),
            i64::try_from(elements).expect("element count fits i64"),
        );
        if matches!(arm, Arm::Uninitialized) {
            output.set_len(elements);
        }
    }
    let producer_ms = producer_started.elapsed().as_secs_f64() * 1e3;
    let total_ms = total_started.elapsed().as_secs_f64() * 1e3;
    let checksum = checksum(black_box(&output));
    let digest = *blake3::hash(bytemuck::cast_slice(&output)).as_bytes();
    black_box(output);
    Sample {
        allocation_ms,
        producer_ms,
        total_ms,
        checksum,
        digest,
    }
}

fn run_metal_arm(
    arm_name: &str,
    ctx: &MetalContext,
    desc: &TensorDesc,
    source: &[u8],
    elements: usize,
    to_float: unsafe extern "C" fn(*const c_void, *mut f32, i64),
) -> MetalSample {
    let total_started = Instant::now();
    let (tensor, host_dequant_ms, metal_allocation_or_copy_ms, direct_producer_ms) = match arm_name
    {
        "staged-metal" => {
            let host_started = Instant::now();
            let output = dequant_to_f32(desc, source).expect("host dequant");
            let host_dequant_ms = host_started.elapsed().as_secs_f64() * 1e3;
            let metal_started = Instant::now();
            let tensor = MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&output),
                desc.shape.clone(),
                GgmlType::F32,
            )
            .expect("copy dequant output into Metal");
            let metal_allocation_or_copy_ms = metal_started.elapsed().as_secs_f64() * 1e3;
            (tensor, host_dequant_ms, metal_allocation_or_copy_ms, 0.0)
        }
        "direct-metal" => {
            let metal_started = Instant::now();
            let tensor = MetalTensor::zeros_f32(ctx, desc.shape.clone())
                .expect("allocate direct Metal output");
            let metal_allocation_or_copy_ms = metal_started.elapsed().as_secs_f64() * 1e3;
            let producer_started = Instant::now();
            unsafe {
                let output = (tensor.buffer.contents().as_ptr() as *mut u8)
                    .add(tensor.offset as usize)
                    .cast::<f32>();
                to_float(
                    source.as_ptr().cast::<c_void>(),
                    output,
                    i64::try_from(elements).expect("element count fits i64"),
                );
            }
            let direct_producer_ms = producer_started.elapsed().as_secs_f64() * 1e3;
            (tensor, 0.0, metal_allocation_or_copy_ms, direct_producer_ms)
        }
        _ => panic!("unknown Metal arm {arm_name}"),
    };
    let total_ms = total_started.elapsed().as_secs_f64() * 1e3;
    let values = unsafe {
        let start = (tensor.buffer.contents().as_ptr() as *const u8)
            .add(tensor.offset as usize)
            .cast::<f32>();
        std::slice::from_raw_parts(start, elements)
    };
    let checksum = checksum(black_box(values));
    let digest = *blake3::hash(bytemuck::cast_slice(values)).as_bytes();
    black_box(tensor);
    MetalSample {
        host_dequant_ms,
        metal_allocation_or_copy_ms,
        direct_producer_ms,
        total_ms,
        checksum,
        digest,
    }
}

fn main() {
    let output_mib = parse_usize("--mib", 256);
    let repetitions = parse_usize("--reps", 9);
    let arm_name = parse_string("--arm", "paired");
    assert!(output_mib > 0, "--mib must be positive");
    assert!(repetitions > 0, "--reps must be positive");

    let output_bytes = output_mib
        .checked_mul(1024 * 1024)
        .expect("output byte count overflow");
    assert!(output_bytes.is_multiple_of(std::mem::size_of::<f32>()));
    let elements = output_bytes / std::mem::size_of::<f32>();
    assert!(elements.is_multiple_of(ROW_ELEMENTS));
    let rows = elements / ROW_ELEMENTS;
    let quantized_bytes = elements / BLOCK_ELEMENTS * BLOCK_BYTES;

    let mut quantized = vec![0u8; quantized_bytes];
    for (block_index, block) in quantized.chunks_exact_mut(BLOCK_BYTES).enumerate() {
        block[..2].copy_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.015625).to_bits().to_le_bytes());
        for (index, byte) in block[4..].iter_mut().enumerate() {
            *byte = ((block_index * 29 + index * 17 + 11) & 0xff) as u8;
        }
    }
    let dtype = GgmlType::Q4_K as u32;

    let desc = TensorDesc {
        name: "codec-init-floor-q4-k".to_string(),
        shape: vec![ROW_ELEMENTS as u64, rows as u64],
        dtype: GgmlType::Q4_K,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: quantized_bytes as u64,
    };
    let traits = unsafe { llama_cpp_sys_2::ggml_get_type_traits(dtype) };
    assert!(!traits.is_null(), "missing Q4_K type traits");
    let to_float = unsafe { (*traits).to_float }.expect("missing Q4_K to_float");

    if matches!(arm_name.as_str(), "staged-metal" | "direct-metal") {
        let ctx = MetalContext::new().expect("Metal context");
        let samples = (0..repetitions)
            .map(|_| run_metal_arm(&arm_name, &ctx, &desc, &quantized, elements, to_float))
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "schema": "qwen-codec-metal-destination-floor/v1",
                "arm": arm_name,
                "dtype": "Q4_K",
                "output_mib": output_mib,
                "output_bytes": output_bytes,
                "source_bytes": quantized_bytes,
                "rows": rows,
                "row_elements": ROW_ELEMENTS,
                "repetitions": repetitions,
                "checksum": samples[0].checksum,
                "digest": blake3::Hash::from_bytes(samples[0].digest).to_hex().to_string(),
                "host_dequant_ms": samples.iter().map(|sample| sample.host_dequant_ms).collect::<Vec<_>>(),
                "metal_allocation_or_copy_ms": samples.iter().map(|sample| sample.metal_allocation_or_copy_ms).collect::<Vec<_>>(),
                "direct_producer_ms": samples.iter().map(|sample| sample.direct_producer_ms).collect::<Vec<_>>(),
                "total_ms": samples.iter().map(|sample| sample.total_ms).collect::<Vec<_>>(),
                "median_total_ms": median(&samples.iter().map(|sample| sample.total_ms).collect::<Vec<_>>()),
            }))
            .expect("serialize result")
        );
        return;
    }

    if arm_name != "paired" {
        let arm = match arm_name.as_str() {
            "zeroed" => Arm::Zeroed,
            "uninitialized" => Arm::Uninitialized,
            _ => {
                panic!("--arm must be zeroed, uninitialized, staged-metal, direct-metal, or paired")
            }
        };
        let samples = (0..repetitions)
            .map(|_| run_arm(arm, &quantized, elements, to_float))
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "schema": "qwen-codec-init-floor/v1",
                "arm": arm_name,
                "dtype": "Q4_K",
                "output_mib": output_mib,
                "output_bytes": output_bytes,
                "source_bytes": quantized_bytes,
                "rows": rows,
                "row_elements": ROW_ELEMENTS,
                "repetitions": repetitions,
                "checksum": samples[0].checksum,
                "digest": blake3::Hash::from_bytes(samples[0].digest).to_hex().to_string(),
                "allocation_ms": samples.iter().map(|sample| sample.allocation_ms).collect::<Vec<_>>(),
                "producer_ms": samples.iter().map(|sample| sample.producer_ms).collect::<Vec<_>>(),
                "total_ms": samples.iter().map(|sample| sample.total_ms).collect::<Vec<_>>(),
                "median_total_ms": median(&samples.iter().map(|sample| sample.total_ms).collect::<Vec<_>>()),
            }))
            .expect("serialize result")
        );
        return;
    }

    let old = run_arm(Arm::Zeroed, &quantized, elements, to_float);
    let actual = dequant_to_f32(&desc, &quantized).expect("candidate dequant");
    assert_eq!(old.checksum, checksum(&actual));
    let actual_digest = blake3::hash(bytemuck::cast_slice(&actual));
    assert_eq!(old.digest, *actual_digest.as_bytes());
    drop(actual);

    black_box(run_arm(Arm::Uninitialized, &quantized, elements, to_float));

    let mut zeroed = Vec::with_capacity(repetitions);
    let mut uninitialized = Vec::with_capacity(repetitions);
    for repetition in 0..repetitions {
        if repetition.is_multiple_of(2) {
            zeroed.push(run_arm(Arm::Zeroed, &quantized, elements, to_float));
            uninitialized.push(run_arm(Arm::Uninitialized, &quantized, elements, to_float));
        } else {
            uninitialized.push(run_arm(Arm::Uninitialized, &quantized, elements, to_float));
            zeroed.push(run_arm(Arm::Zeroed, &quantized, elements, to_float));
        }
    }
    assert!(
        zeroed
            .iter()
            .chain(&uninitialized)
            .all(|sample| sample.checksum == old.checksum)
    );

    let zeroed_total = zeroed
        .iter()
        .map(|sample| sample.total_ms)
        .collect::<Vec<_>>();
    let uninitialized_total = uninitialized
        .iter()
        .map(|sample| sample.total_ms)
        .collect::<Vec<_>>();
    let zeroed_median = median(&zeroed_total);
    let uninitialized_median = median(&uninitialized_total);
    let paired_savings = zeroed
        .iter()
        .zip(&uninitialized)
        .map(|(control, candidate)| control.total_ms - candidate.total_ms)
        .collect::<Vec<_>>();

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "qwen-codec-init-floor/v1",
            "dtype": "Q4_K",
            "output_mib": output_mib,
            "output_bytes": output_bytes,
            "source_bytes": quantized_bytes,
            "rows": rows,
            "row_elements": ROW_ELEMENTS,
            "repetitions": repetitions,
            "digest": actual_digest.to_hex().to_string(),
            "zeroed": {
                "allocation_ms": zeroed.iter().map(|sample| sample.allocation_ms).collect::<Vec<_>>(),
                "producer_ms": zeroed.iter().map(|sample| sample.producer_ms).collect::<Vec<_>>(),
                "total_ms": zeroed_total,
                "median_total_ms": zeroed_median,
            },
            "uninitialized": {
                "allocation_ms": uninitialized.iter().map(|sample| sample.allocation_ms).collect::<Vec<_>>(),
                "producer_ms": uninitialized.iter().map(|sample| sample.producer_ms).collect::<Vec<_>>(),
                "total_ms": uninitialized_total,
                "median_total_ms": uninitialized_median,
            },
            "paired_savings_ms": paired_savings,
            "median_paired_saving_ms": median(&paired_savings),
            "median_speedup": zeroed_median / uninitialized_median,
        }))
        .expect("serialize result")
    );
}
