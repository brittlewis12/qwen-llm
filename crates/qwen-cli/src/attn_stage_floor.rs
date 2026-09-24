use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
use qwen_llm::metal::{
    KernelEncoder, MetalContext, MetalTensor, encode_attn_stage_floor_g16, encode_copy_offset_f32,
};
use qwen_llm::tensor::GgmlType;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

const CONTEXT: usize = 32768;
const N_KV: usize = 2;
const HD: usize = 256;
const NWG: usize = 256;
const ROW_BYTES: usize = 288;
const RAMP_REPS: usize = 512;
const TARGET_MS: f64 = 0.15587;

#[derive(Parser, Debug)]
pub struct AttnStageFloorArgs {
    /// Canonical attention capture directory.
    #[arg(long)]
    capture: PathBuf,
    /// Raw timed samples after warmup.
    #[arg(long, default_value = "9")]
    runs: usize,
    /// Idle seconds after quantization and validation.
    #[arg(long, default_value = "120")]
    cooldown_secs: u64,
    /// Permit noncanonical runs/cooldowns without a decision.
    #[arg(long)]
    allow_noncanonical_smoke: bool,
}

fn read_prefix(path: &Path, bytes: usize) -> Result<Vec<u8>> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let length = usize::try_from(file.metadata()?.len())?;
    if length < bytes {
        bail!(
            "{} has {length} bytes, need at least {bytes}",
            path.display()
        );
    }
    let mut result = vec![0u8; bytes];
    file.read_exact(&mut result)?;
    Ok(result)
}

fn sha256_file(path: &Path) -> Result<(usize, String)> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut bytes = 0usize;
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
        bytes += count;
    }
    Ok((bytes, format!("{:x}", digest.finalize())))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_f16_source(name: &str, bytes: &[u8]) -> Result<()> {
    let mut nonzero = false;
    for pair in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        if bits & 0x7c00 == 0x7c00 {
            bail!("{name} prefix contains a non-finite F16 value");
        }
        nonzero |= bits & 0x7fff != 0;
    }
    if !nonzero {
        bail!("{name} prefix is all zero");
    }
    Ok(())
}

fn source_entry<'a>(manifest: &'a Value, name: &str) -> Result<&'a Value> {
    manifest["tensors"]
        .as_array()
        .and_then(|tensors| tensors.iter().find(|entry| entry["name"] == name))
        .ok_or_else(|| anyhow!("manifest tensor entry missing: {name}"))
}

fn verify_source(root: &Path, manifest: &Value, name: &str) -> Result<(Vec<u8>, Value)> {
    let entry = source_entry(manifest, name)?;
    if entry["dtype"] != "f16le"
        || entry["shape"] != json!([131072, N_KV, HD])
        || entry["strides_bytes"] != json!([1024, 512, 2])
    {
        bail!("manifest tensor metadata violates the canonical contract: {name}");
    }
    let path = root.join(name);
    let (full_bytes, full_hash) = sha256_file(&path)?;
    if entry["hashes"]["byte_length"] != full_bytes || entry["hashes"]["sha256"] != full_hash {
        bail!("full-file identity mismatch: {name}");
    }
    let prefix_bytes = CONTEXT * N_KV * HD * 2;
    let prefix = read_prefix(&path, prefix_bytes)?;
    let prefix_hash = sha256_bytes(&prefix);
    if entry["hashes"]["prefix_32768_sha256"] != prefix_hash {
        bail!("32K-prefix identity mismatch: {name}");
    }
    validate_f16_source(name, &prefix)?;
    Ok((
        prefix,
        json!({
            "name": name,
            "full_bytes": full_bytes,
            "full_sha256": full_hash,
            "prefix_bytes": prefix_bytes,
            "prefix_sha256": prefix_hash,
        }),
    ))
}

fn quantize_g16(source: &[u8]) -> Result<Vec<u8>> {
    let source_row_bytes = HD * 2;
    let rows = CONTEXT * N_KV;
    if source.len() != rows * source_row_bytes {
        bail!("F16 source byte length violates the fixed capture shape");
    }
    let mut output = vec![0u8; rows * ROW_BYTES];
    for row in 0..rows {
        for group in 0..16 {
            let source_base = row * source_row_bytes + group * 16 * 2;
            let output_base = row * ROW_BYTES;
            let mut values = [0.0f32; 16];
            let mut max_abs = 0.0f32;
            for (index, value) in values.iter_mut().enumerate() {
                let at = source_base + index * 2;
                let bits = u16::from_le_bytes([source[at], source[at + 1]]);
                *value = half::f16::from_bits(bits).to_f32();
                max_abs = max_abs.max(value.abs());
            }
            let raw_scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            let scale_f16 = half::f16::from_f32(raw_scale);
            let scale = scale_f16.to_f32();
            let scale_at = output_base + group * 2;
            output[scale_at..scale_at + 2].copy_from_slice(&scale_f16.to_bits().to_le_bytes());
            let payload = output_base + 32 + group * 16;
            for (index, value) in values.iter().enumerate() {
                let code = (value / scale).round_ties_even().clamp(-127.0, 127.0) as i8;
                output[payload + index] = code as u8;
            }
        }
    }
    Ok(output)
}

fn decode_value(cache: &[u8], row: usize, dim: usize) -> f32 {
    let group = dim / 16;
    let base = row * ROW_BYTES;
    let scale_at = base + group * 2;
    let scale_bits = u16::from_le_bytes([cache[scale_at], cache[scale_at + 1]]);
    let scale = half::f16::from_bits(scale_bits).to_f32();
    let code = cache[base + 32 + dim] as i8;
    half::f16::from_f32(scale * f32::from(code)).to_f32()
}

fn expected_checksum(k: &[u8], v: &[u8]) -> Vec<f32> {
    let rows_per_partition = CONTEXT.div_ceil(NWG);
    let mut output = vec![0.0f32; N_KV * NWG * HD];
    for kvh in 0..N_KV {
        for partition in 0..NWG {
            let begin = partition * rows_per_partition;
            let end = (begin + rows_per_partition).min(CONTEXT);
            for dim in 0..HD {
                let mut sum = 0.0f32;
                for tile in (begin..end).step_by(32) {
                    let tile_end = (tile + 32).min(end);
                    for position in tile..tile_end {
                        sum += decode_value(k, position * N_KV + kvh, dim);
                    }
                    for position in tile..tile_end {
                        sum += decode_value(v, position * N_KV + kvh, dim);
                    }
                }
                output[(kvh * NWG + partition) * HD + dim] = sum;
            }
        }
    }
    output
}

fn read_f32(tensor: &MetalTensor) -> Result<Vec<f32>> {
    if tensor.dtype != GgmlType::F32 {
        bail!("checksum tensor is not F32");
    }
    let count = usize::try_from(tensor.n_elements())?;
    let offset = usize::try_from(tensor.offset)?;
    unsafe {
        let base = (tensor.buffer.contents().as_ptr() as *const u8).add(offset);
        Ok(std::slice::from_raw_parts(base.cast::<f32>(), count).to_vec())
    }
}

fn validate_checksum(reference: &[f32], actual: &[f32]) -> Result<Value> {
    if reference.len() != actual.len() {
        bail!("checksum lengths differ");
    }
    let mut max_abs = 0.0f32;
    let mut diff2 = 0.0f64;
    let mut ref2 = 0.0f64;
    for (&expected, &observed) in reference.iter().zip(actual) {
        if !expected.is_finite() || !observed.is_finite() {
            bail!("checksum contains a non-finite value");
        }
        max_abs = max_abs.max((expected - observed).abs());
        let delta = f64::from(expected) - f64::from(observed);
        diff2 += delta * delta;
        ref2 += f64::from(expected) * f64::from(expected);
    }
    if ref2 == 0.0 {
        bail!("checksum reference is all zero");
    }
    let relative_l2 = (diff2 / ref2).sqrt();
    if relative_l2 > 1e-5 || max_abs > 1e-3 {
        bail!("checksum mismatch: relative_l2={relative_l2} max_abs={max_abs}");
    }
    Ok(json!({"relative_l2": relative_l2, "max_abs": max_abs}))
}

fn median(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    }
}

pub fn run(args: AttnStageFloorArgs, build_identity: Value) -> Result<()> {
    let canonical_protocol = args.runs == 9 && args.cooldown_secs == 120;
    if !canonical_protocol && !args.allow_noncanonical_smoke {
        bail!("canonical packet requires --runs 9 and --cooldown-secs 120");
    }
    if canonical_protocol && args.allow_noncanonical_smoke {
        bail!("--allow-noncanonical-smoke is valid only for a noncanonical run");
    }
    if args.runs < 3 {
        bail!("smoke runs must be at least 3");
    }
    let manifest_path = args.capture.join("manifest.json");
    let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    let expected_architecture = json!({
        "name": "Qwen3.6-A3B", "q_heads": 16, "kv_heads": 2,
        "group": 8, "head_dim": 256, "kv_dtype": "f16",
    });
    let expected_blocks = json!([
        {"block": 3, "kv_slot": 0},
        {"block": 19, "kv_slot": 4},
        {"block": 39, "kv_slot": 9},
    ]);
    if manifest["schema_version"] != 1
        || manifest["claim_scope"] != "prefill_causal_query"
        || manifest["architecture"] != expected_architecture
        || manifest["blocks"] != expected_blocks
        || manifest["selector_environment"] != json!({})
        || manifest["identity"]["build"]["status"] != "match"
        || manifest["identity"]["build"]["build_dirty"] != false
        || manifest["capture_tier"] != "canonical"
        || manifest["context"] != 131072
    {
        bail!("capture manifest does not satisfy the canonical contract");
    }
    let (k_source, k_source_identity) = verify_source(&args.capture, &manifest, "block-3-k.f16le")?;
    let (v_source, v_source_identity) = verify_source(&args.capture, &manifest, "block-3-v.f16le")?;
    let k_quantized = quantize_g16(&k_source)?;
    let v_quantized = quantize_g16(&v_source)?;
    let k_quantized_hash = sha256_bytes(&k_quantized);
    let v_quantized_hash = sha256_bytes(&v_quantized);
    let reference = expected_checksum(&k_quantized, &v_quantized);

    let ctx = MetalContext::new().context("initialize Metal")?;
    let k = MetalTensor::from_bytes(
        &ctx,
        &k_quantized,
        vec![(k_quantized.len() / 2) as u64],
        GgmlType::F16,
    )?;
    let v = MetalTensor::from_bytes(
        &ctx,
        &v_quantized,
        vec![(v_quantized.len() / 2) as u64],
        GgmlType::F16,
    )?;
    let checksum = MetalTensor::zeros_f32(&ctx, vec![(N_KV * NWG * HD) as u64])?;
    let scrub_elements = 32usize * 1024 * 1024;
    let scrub_source = MetalTensor::zeros_f32(&ctx, vec![scrub_elements as u64])?;
    let scrub_target = MetalTensor::zeros_f32(&ctx, vec![scrub_elements as u64])?;

    let encode = |enc: &KernelEncoder| -> Result<()> {
        Ok(encode_attn_stage_floor_g16(
            &ctx, enc, &k, &v, &checksum, CONTEXT, N_KV, NWG,
        )?)
    };
    let dispatch = || -> Result<f64> {
        let command = ctx.queue.commandBuffer().context("stage-floor command")?;
        let enc = KernelEncoder::begin(&command);
        encode(&enc)?;
        enc.end();
        command.commit();
        command.waitUntilCompleted();
        if let Err(error) = qwen_llm::metal::command_buffer_completed(&command) {
            return Err(anyhow!("stage-floor command failed: {error}"));
        }
        let start = command.GPUStartTime();
        let end = command.GPUEndTime();
        if !start.is_finite() || !end.is_finite() || start <= 0.0 || end <= start {
            bail!("invalid GPU timestamps: start={start} end={end}");
        }
        Ok((end - start) * 1e3)
    };

    dispatch()?;
    let checksum_metrics = validate_checksum(&reference, &read_f32(&checksum)?)?;
    let scrub = || -> Result<()> {
        let command = ctx.queue.commandBuffer().context("stage-floor scrub")?;
        let enc = KernelEncoder::begin(&command);
        encode_copy_offset_f32(&ctx, &enc, &scrub_source, 0, &scrub_target, scrub_elements)?;
        enc.end();
        command.commit();
        command.waitUntilCompleted();
        if let Err(error) = qwen_llm::metal::command_buffer_completed(&command) {
            bail!("stage-floor scrub failed: {error}");
        }
        Ok(())
    };
    scrub()?;
    if args.cooldown_secs != 0 {
        std::thread::sleep(Duration::from_secs(args.cooldown_secs));
    }
    let ramp = ctx.queue.commandBuffer().context("stage-floor ramp")?;
    let ramp_enc = KernelEncoder::begin(&ramp);
    for _ in 0..RAMP_REPS {
        encode(&ramp_enc)?;
    }
    ramp_enc.end();
    ramp.commit();
    ramp.waitUntilCompleted();
    if let Err(error) = qwen_llm::metal::command_buffer_completed(&ramp) {
        bail!("stage-floor ramp failed: {error}");
    }

    let mut samples = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        scrub()?;
        samples.push(dispatch()?);
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let measured = median(&sorted);
    let final_checksum_metrics = validate_checksum(&reference, &read_f32(&checksum)?)?;
    let packet = json!({
        "schema_version": 1,
        "build": build_identity,
        "capture": args.capture,
        "capture_model_sha256": manifest["identity"]["model_sha256"],
        "source_identity": [k_source_identity, v_source_identity],
        "shape": {"n_pos": CONTEXT, "n_kv_heads": N_KV, "head_dim": HD,
            "nwg": NWG, "c": 32, "grid": [N_KV, 1, NWG], "threads_tg": 256,
            "threadgroup_bytes": 32 * HD * 2},
        "format": {"name": "split-plane-g16-q8", "row_bytes": ROW_BYTES,
            "scale_bytes": 32, "payload_bytes": 256,
            "quantizer": "symmetric-maxabs-rne-decoded-f16-scale",
            "k_sha256": k_quantized_hash, "v_sha256": v_quantized_hash},
        "protocol": {"runs": args.runs, "cooldown_secs": args.cooldown_secs,
            "ramp_reps": RAMP_REPS, "scrub_working_set_bytes": scrub_elements * 4,
            "scrub_traffic_bytes": scrub_elements * 8,
            "canonical": canonical_protocol},
        "checksum_before": checksum_metrics,
        "checksum_after": final_checksum_metrics,
        "raw_ms": samples,
        "sorted_ms": sorted,
        "median_ms": measured,
        "gate_ms": TARGET_MS,
        "disposition": if !canonical_protocol { "noncanonical_smoke" }
            else if measured < TARGET_MS { "pass_floor" } else { "kill" },
    });
    println!("{}", serde_json::to_string_pretty(&packet)?);
    Ok(())
}
