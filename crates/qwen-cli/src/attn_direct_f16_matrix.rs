use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
use qwen_llm::metal::{
    KernelEncoder, MetalContext, MetalTensor, attn_v4_choose_group_tile, attn_v4_choose_nwg,
    attn_v4_choose_tile_c, encode_attn_decode_v4_main_only_f32,
    encode_attn_decode_v4_reduce_only_f32, encode_attn_direct_f16_matrix_g8_c32,
    encode_copy_offset_f32,
};
use qwen_llm::tensor::GgmlType;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

const CONTEXT: usize = 32_768;
const N_Q: usize = 16;
const N_KV: usize = 2;
const GROUP: usize = 8;
const HD: usize = 256;
const NWG: usize = 256;
const RAMP_REPS: usize = 512;

#[derive(Parser, Debug)]
pub struct AttnDirectF16MatrixArgs {
    /// Canonical v0.575 attention capture directory.
    #[arg(long)]
    capture: PathBuf,
    /// Raw main and main-plus-reduce samples per row.
    #[arg(long, default_value = "9")]
    runs: usize,
    /// Idle seconds before each A1/P/A2 row.
    #[arg(long, default_value = "120")]
    cooldown_secs: u64,
    /// Permit noncanonical runs/cooldowns without a decision.
    #[arg(long)]
    allow_noncanonical_smoke: bool,
}

#[derive(Clone, Copy)]
enum Variant {
    V4,
    DirectF16Matrix,
}

#[derive(Clone, Copy)]
struct Metrics {
    cosine: f64,
    max_abs: f64,
    relative_l2: f64,
}

impl Metrics {
    fn json(self) -> Value {
        json!({
            "cosine": self.cosine,
            "max_abs": self.max_abs,
            "relative_l2": self.relative_l2,
        })
    }
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

fn source_entry<'a>(manifest: &'a Value, name: &str) -> Result<&'a Value> {
    manifest["tensors"]
        .as_array()
        .and_then(|entries| entries.iter().find(|entry| entry["name"] == name))
        .ok_or_else(|| anyhow!("manifest tensor entry missing: {name}"))
}

fn verify_kv_source(root: &Path, manifest: &Value, name: &str) -> Result<(Vec<u8>, Value)> {
    let entry = source_entry(manifest, name)?;
    if entry["dtype"] != "f16le"
        || entry["shape"] != json!([131072, N_KV, HD])
        || entry["strides_bytes"] != json!([1024, 512, 2])
    {
        bail!("manifest tensor metadata violates the KV contract: {name}");
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
    let mut nonzero = false;
    for pair in prefix.chunks_exact(2) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        if bits & 0x7c00 == 0x7c00 {
            bail!("{name} prefix contains a non-finite F16 value");
        }
        nonzero |= bits & 0x7fff != 0;
    }
    if !nonzero {
        bail!("{name} prefix is all zero");
    }
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

fn verify_f32_source(root: &Path, manifest: &Value, name: &str) -> Result<(Vec<u8>, Value)> {
    let entry = source_entry(manifest, name)?;
    if entry["dtype"] != "f32le"
        || entry["shape"] != json!([128, N_Q, HD])
        || entry["strides_bytes"] != json!([16384, 1024, 4])
    {
        bail!("manifest tensor metadata violates the F32 contract: {name}");
    }
    let path = root.join(name);
    let bytes = fs::read(&path)?;
    let hash = sha256_bytes(&bytes);
    if entry["hashes"]["byte_length"] != bytes.len() || entry["hashes"]["sha256"] != hash {
        bail!("full-file identity mismatch: {name}");
    }
    Ok((
        bytes,
        json!({
            "name": name,
            "full_bytes": entry["hashes"]["byte_length"],
            "full_sha256": hash,
        }),
    ))
}

fn selected_row(manifest: &Value) -> Result<usize> {
    let view = manifest["views"]
        .as_array()
        .and_then(|views| views.iter().find(|view| view["context"] == CONTEXT))
        .ok_or_else(|| anyhow!("capture is missing the 32K view"))?;
    let positions = view["positions"]
        .as_array()
        .ok_or_else(|| anyhow!("32K positions are missing"))?;
    let rows = view["deduplicated_rows"]
        .as_array()
        .ok_or_else(|| anyhow!("32K deduplicated rows are missing"))?;
    if positions.len() != rows.len() {
        bail!("32K position and row counts differ");
    }
    let index = positions
        .iter()
        .position(|position| position == CONTEXT - 1)
        .ok_or_else(|| anyhow!("32K tail query is missing"))?;
    let row = rows[index]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| anyhow!("32K tail row is invalid"))?;
    if row >= 128 {
        bail!("32K tail row {row} is outside the captured tensor");
    }
    Ok(row)
}

fn captured_row(source: &[u8], row: usize) -> Result<Vec<u8>> {
    let stride = N_Q * HD * 4;
    let begin = row
        .checked_mul(stride)
        .ok_or_else(|| anyhow!("captured row offset overflow"))?;
    source
        .get(begin..begin + stride)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| anyhow!("captured row is truncated"))
}

fn parse_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!("F32 byte length is not divisible by four");
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    if !values.iter().all(|value| value.is_finite()) {
        bail!("captured F32 row contains a non-finite value");
    }
    Ok(values)
}

fn read_f32(tensor: &MetalTensor) -> Result<Vec<f32>> {
    if tensor.dtype != GgmlType::F32 {
        bail!("readback tensor is not F32");
    }
    let count = usize::try_from(tensor.n_elements())?;
    let offset = usize::try_from(tensor.offset)?;
    unsafe {
        let base = (tensor.buffer.contents().as_ptr() as *const u8).add(offset);
        Ok(std::slice::from_raw_parts(base.cast::<f32>(), count).to_vec())
    }
}

fn poison(tensor: &MetalTensor) -> Result<()> {
    if tensor.dtype != GgmlType::F32 {
        bail!("poison tensor is not F32");
    }
    let count = usize::try_from(tensor.n_elements())?;
    let offset = usize::try_from(tensor.offset)?;
    unsafe {
        let base = (tensor.buffer.contents().as_ptr() as *mut u8).add(offset);
        std::slice::from_raw_parts_mut(base.cast::<f32>(), count).fill(f32::NAN);
    }
    Ok(())
}

fn metrics(reference: &[f32], actual: &[f32]) -> Result<Metrics> {
    if reference.len() != actual.len() || reference.is_empty() {
        bail!("metric vectors have incompatible lengths");
    }
    let mut dot = 0.0f64;
    let mut reference2 = 0.0f64;
    let mut actual2 = 0.0f64;
    let mut diff2 = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&expected, &observed) in reference.iter().zip(actual) {
        if !expected.is_finite() || !observed.is_finite() {
            bail!("metric vector contains a non-finite value");
        }
        let expected = f64::from(expected);
        let observed = f64::from(observed);
        let delta = expected - observed;
        dot += expected * observed;
        reference2 += expected * expected;
        actual2 += observed * observed;
        diff2 += delta * delta;
        max_abs = max_abs.max(delta.abs());
    }
    if reference2 == 0.0 || actual2 == 0.0 {
        bail!("metric vector has zero norm");
    }
    Ok(Metrics {
        cosine: dot / (reference2 * actual2).sqrt(),
        max_abs,
        relative_l2: (diff2 / reference2).sqrt(),
    })
}

fn validate_partials(o: &[f32], ml: &[f32]) -> Result<Value> {
    if o.len() != N_KV * NWG * GROUP * HD || ml.len() != N_KV * NWG * GROUP * 2 {
        bail!("partial tensor lengths violate the frozen ABI");
    }
    if !o.iter().all(|value| value.is_finite()) {
        bail!("candidate left a non-finite output partial");
    }
    let mut min_l = f32::INFINITY;
    let mut max_m = f32::NEG_INFINITY;
    for state in ml.chunks_exact(2) {
        if !state[0].is_finite() || !state[1].is_finite() || state[1] <= 0.0 {
            bail!("candidate left invalid online-softmax state");
        }
        max_m = max_m.max(state[0]);
        min_l = min_l.min(state[1]);
    }
    Ok(json!({"all_finite": true, "min_l": min_l, "max_m": max_m}))
}

fn tensor_digest(tensors: &[&MetalTensor]) -> Result<String> {
    let mut digest = Sha256::new();
    for tensor in tensors {
        let bytes = usize::try_from(tensor.n_bytes())?;
        let offset = usize::try_from(tensor.offset)?;
        unsafe {
            let base = (tensor.buffer.contents().as_ptr() as *const u8).add(offset);
            digest.update(std::slice::from_raw_parts(base, bytes));
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn median(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    }
}

pub fn run(args: AttnDirectF16MatrixArgs, build_identity: Value) -> Result<()> {
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
    if canonical_protocol
        && (build_identity["status"] != "match"
            || build_identity["build_dirty"] != false
            || build_identity["runtime_dirty"] != false
            || build_identity["problems"] != json!([])
            || build_identity["overrides"] != json!([]))
    {
        bail!("canonical packet requires a clean matching build identity");
    }
    let qwen_environment = std::env::vars()
        .filter(|(name, _)| name.starts_with("QWEN_"))
        .collect::<Vec<_>>();
    if !qwen_environment.is_empty() {
        bail!("direct-F16 matrix packets forbid QWEN_* overrides");
    }

    let manifest: Value = serde_json::from_slice(&fs::read(args.capture.join("manifest.json"))?)?;
    let expected_architecture = json!({
        "name": "Qwen3.6-A3B",
        "q_heads": N_Q,
        "kv_heads": N_KV,
        "group": GROUP,
        "head_dim": HD,
        "kv_dtype": "f16",
    });
    if manifest["schema_version"] != 1
        || manifest["claim_scope"] != "prefill_causal_query"
        || manifest["architecture"] != expected_architecture
        || manifest["capture_tier"] != "canonical"
        || manifest["context"] != 131072
        || manifest["selector_environment"] != json!({})
        || manifest["identity"]["build"]["status"] != "match"
        || manifest["identity"]["build"]["build_dirty"] != false
    {
        bail!("capture manifest does not satisfy the canonical contract");
    }

    let (k_source, k_identity) = verify_kv_source(&args.capture, &manifest, "block-3-k.f16le")?;
    let (v_source, v_identity) = verify_kv_source(&args.capture, &manifest, "block-3-v.f16le")?;
    let (q_source, q_identity) = verify_f32_source(&args.capture, &manifest, "block-3-q.f32le")?;
    let (o_source, o_identity) = verify_f32_source(&args.capture, &manifest, "block-3-o.f32le")?;
    let query_row = selected_row(&manifest)?;
    let q_bytes = captured_row(&q_source, query_row)?;
    let captured_o_bytes = captured_row(&o_source, query_row)?;
    let captured_o = parse_f32(&captured_o_bytes)?;

    if attn_v4_choose_nwg(CONTEXT, GROUP) != NWG
        || attn_v4_choose_group_tile(CONTEXT, GROUP) != 4
        || attn_v4_choose_tile_c(CONTEXT, GROUP) != 64
    {
        bail!("current V4 selector no longer matches the frozen 32K anchor");
    }

    let ctx = MetalContext::new().context("initialize Metal")?;
    let q = MetalTensor::from_bytes(&ctx, &q_bytes, vec![(N_Q * HD) as u64], GgmlType::F32)?;
    let k = MetalTensor::from_bytes(
        &ctx,
        &k_source,
        vec![(CONTEXT * N_KV * HD) as u64],
        GgmlType::F16,
    )?;
    let v = MetalTensor::from_bytes(
        &ctx,
        &v_source,
        vec![(CONTEXT * N_KV * HD) as u64],
        GgmlType::F16,
    )?;
    let partial_elements = N_KV * NWG * GROUP * HD;
    let ml_elements = N_KV * NWG * GROUP * 2;
    let v4_o = MetalTensor::zeros_f32(&ctx, vec![partial_elements as u64])?;
    let v4_ml = MetalTensor::zeros_f32(&ctx, vec![ml_elements as u64])?;
    let v4_out = MetalTensor::zeros_f32(&ctx, vec![(N_Q * HD) as u64])?;
    let candidate_o = MetalTensor::zeros_f32(&ctx, vec![partial_elements as u64])?;
    let candidate_ml = MetalTensor::zeros_f32(&ctx, vec![ml_elements as u64])?;
    let candidate_out = MetalTensor::zeros_f32(&ctx, vec![(N_Q * HD) as u64])?;
    let scrub_elements = 32usize * 1024 * 1024;
    let scrub_source = MetalTensor::zeros_f32(&ctx, vec![scrub_elements as u64])?;
    let scrub_target = MetalTensor::zeros_f32(&ctx, vec![scrub_elements as u64])?;

    let encode_main =
        |variant: Variant, enc: &KernelEncoder, o: &MetalTensor, ml: &MetalTensor| -> Result<()> {
            match variant {
                Variant::V4 => encode_attn_decode_v4_main_only_f32(
                    &ctx, enc, &q, &k, &v, o, ml, N_Q, N_KV, HD, CONTEXT, NWG, 64,
                )?,
                Variant::DirectF16Matrix => {
                    encode_attn_direct_f16_matrix_g8_c32(&ctx, enc, &q, &k, &v, o, ml)?
                }
            }
            Ok(())
        };
    let buffers = |variant: Variant| match variant {
        Variant::V4 => (&v4_o, &v4_ml, &v4_out),
        Variant::DirectF16Matrix => (&candidate_o, &candidate_ml, &candidate_out),
    };
    let dispatch = |variant: Variant, paired: bool| -> Result<f64> {
        let (o, ml, out) = buffers(variant);
        let command = ctx.queue.commandBuffer().context("attention command")?;
        let enc = KernelEncoder::begin(&command);
        encode_main(variant, &enc, o, ml)?;
        if paired {
            encode_attn_decode_v4_reduce_only_f32(&ctx, &enc, o, ml, out, N_Q, N_KV, HD, NWG)?;
        }
        enc.end();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            return Err(anyhow!("attention command failed: {error:?}"));
        }
        let start = command.GPUStartTime();
        let end = command.GPUEndTime();
        if !start.is_finite() || !end.is_finite() || start <= 0.0 || end <= start {
            bail!("invalid GPU timestamps: start={start} end={end}");
        }
        Ok((end - start) * 1e3)
    };

    poison(&v4_o)?;
    poison(&v4_ml)?;
    poison(&v4_out)?;
    dispatch(Variant::V4, true)?;
    let v4_output = read_f32(&v4_out)?;
    let v4_capture_metrics = metrics(&captured_o, &v4_output)?;
    if v4_capture_metrics.cosine < 0.99999 || v4_capture_metrics.max_abs > 0.005 {
        bail!("V4 does not reproduce the captured 32K output");
    }

    poison(&candidate_o)?;
    poison(&candidate_ml)?;
    poison(&candidate_out)?;
    dispatch(Variant::DirectF16Matrix, true)?;
    let candidate_partials =
        validate_partials(&read_f32(&candidate_o)?, &read_f32(&candidate_ml)?)?;
    let candidate_output = read_f32(&candidate_out)?;
    let candidate_v4_metrics = metrics(&v4_output, &candidate_output)?;
    let candidate_capture_metrics = metrics(&captured_o, &candidate_output)?;
    if candidate_v4_metrics.cosine < 0.99999
        || candidate_v4_metrics.max_abs > 0.005
        || candidate_capture_metrics.cosine < 0.9999
        || candidate_capture_metrics.max_abs >= 0.01
    {
        bail!("direct-F16 matrix body failed its correctness gates");
    }

    let candidate_pipeline = ctx.pipeline_info("kernel_attn_direct_f16_matrix_g8_c32")?;
    let v4_pipeline = ctx.pipeline_info("kernel_attn_decode_v4_g8_t4_c64_f32")?;
    if candidate_pipeline.thread_execution_width != 32
        || candidate_pipeline.max_total_threads_per_threadgroup < 256
    {
        bail!("candidate pipeline cannot execute the frozen 256-thread topology");
    }
    let input_digest_before = tensor_digest(&[&q, &k, &v])?;

    let scrub = || -> Result<()> {
        let command = ctx.queue.commandBuffer().context("attention scrub")?;
        let enc = KernelEncoder::begin(&command);
        encode_copy_offset_f32(&ctx, &enc, &scrub_source, 0, &scrub_target, scrub_elements)?;
        enc.end();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("attention scrub failed: {error:?}");
        }
        Ok(())
    };
    let ramp = || -> Result<()> {
        let command = ctx.queue.commandBuffer().context("attention ramp")?;
        let enc = KernelEncoder::begin(&command);
        for _ in 0..RAMP_REPS {
            encode_main(Variant::V4, &enc, &v4_o, &v4_ml)?;
            encode_attn_decode_v4_reduce_only_f32(
                &ctx, &enc, &v4_o, &v4_ml, &v4_out, N_Q, N_KV, HD, NWG,
            )?;
        }
        enc.end();
        command.commit();
        command.waitUntilCompleted();
        if let Some(error) = command.error() {
            bail!("attention ramp failed: {error:?}");
        }
        Ok(())
    };
    let measure_row = |label: &str, variant: Variant| -> Result<Value> {
        dispatch(variant, true)?;
        let (o, ml, out) = buffers(variant);
        let before = tensor_digest(&[o, ml, out])?;
        scrub()?;
        if args.cooldown_secs != 0 {
            std::thread::sleep(Duration::from_secs(args.cooldown_secs));
        }
        ramp()?;
        let mut main_raw = Vec::with_capacity(args.runs);
        let mut pair_raw = Vec::with_capacity(args.runs);
        for sample in 0..args.runs {
            if sample % 2 == 0 {
                scrub()?;
                main_raw.push(dispatch(variant, false)?);
                scrub()?;
                pair_raw.push(dispatch(variant, true)?);
            } else {
                scrub()?;
                pair_raw.push(dispatch(variant, true)?);
                scrub()?;
                main_raw.push(dispatch(variant, false)?);
            }
        }
        let after = tensor_digest(&[o, ml, out])?;
        if before != after {
            bail!("{label} output checksum changed across timing");
        }
        let mut main_sorted = main_raw.clone();
        let mut pair_sorted = pair_raw.clone();
        main_sorted.sort_by(f64::total_cmp);
        pair_sorted.sort_by(f64::total_cmp);
        Ok(json!({
            "label": label,
            "main_raw_ms": main_raw,
            "main_sorted_ms": main_sorted,
            "main_median_ms": median(&main_sorted),
            "pair_raw_ms": pair_raw,
            "pair_sorted_ms": pair_sorted,
            "pair_median_ms": median(&pair_sorted),
            "checksum_before": before,
            "checksum_after": after,
        }))
    };

    let a1 = measure_row("A1_v4", Variant::V4)?;
    let candidate = measure_row("P_direct_f16_matrix", Variant::DirectF16Matrix)?;
    let a2 = measure_row("A2_v4", Variant::V4)?;
    let a1_main = a1["main_median_ms"].as_f64().context("A1 main median")?;
    let a2_main = a2["main_median_ms"].as_f64().context("A2 main median")?;
    let a1_pair = a1["pair_median_ms"].as_f64().context("A1 pair median")?;
    let a2_pair = a2["pair_median_ms"].as_f64().context("A2 pair median")?;
    let candidate_main = candidate["main_median_ms"]
        .as_f64()
        .context("candidate main median")?;
    let candidate_pair = candidate["pair_median_ms"]
        .as_f64()
        .context("candidate pair median")?;
    let main_anchor = a1_main.min(a2_main);
    let pair_anchor = a1_pair.min(a2_pair);
    let main_anchor_spread = a1_main.max(a2_main) / main_anchor;
    let pair_anchor_spread = a1_pair.max(a2_pair) / pair_anchor;
    let anchors_valid = main_anchor_spread <= 1.01 && pair_anchor_spread <= 1.01;
    let main_pass = candidate_main <= 0.90 * main_anchor;
    let pair_pass = candidate_pair <= 0.90 * pair_anchor;
    let disposition = if !canonical_protocol {
        "noncanonical_smoke"
    } else if !anchors_valid {
        "invalid_anchor"
    } else if main_pass && pair_pass {
        "authorize_131k"
    } else {
        "kill"
    };
    let input_digest_after = tensor_digest(&[&q, &k, &v])?;
    if input_digest_before != input_digest_after {
        bail!("candidate mutated an input tensor");
    }

    let packet = json!({
        "schema_version": 1,
        "build": build_identity,
        "capture": args.capture,
        "capture_model_sha256": manifest["identity"]["model_sha256"],
        "source_identity": [k_identity, v_identity, q_identity, o_identity],
        "query": {
            "position": CONTEXT - 1,
            "deduplicated_row": query_row,
            "q_row_sha256": sha256_bytes(&q_bytes),
            "o_row_sha256": sha256_bytes(&captured_o_bytes),
        },
        "shape": {
            "n_pos": CONTEXT,
            "n_q_heads": N_Q,
            "n_kv_heads": N_KV,
            "group": GROUP,
            "head_dim": HD,
            "nwg": NWG,
            "c": 32,
            "grid": [N_KV, 1, NWG],
            "threads_tg": 256,
            "threadgroup_bytes": 5152,
            "partial_abi": "v4-g8-h2",
        },
        "pipelines": {
            "candidate": {
                "name": candidate_pipeline.name,
                "thread_execution_width": candidate_pipeline.thread_execution_width,
                "max_threads_tg": candidate_pipeline.max_total_threads_per_threadgroup,
                "static_threadgroup_bytes": candidate_pipeline.static_threadgroup_memory_length,
                "dynamic_threadgroup_bytes": 5152,
            },
            "anchor": {
                "name": v4_pipeline.name,
                "thread_execution_width": v4_pipeline.thread_execution_width,
                "max_threads_tg": v4_pipeline.max_total_threads_per_threadgroup,
                "static_threadgroup_bytes": v4_pipeline.static_threadgroup_memory_length,
            },
        },
        "correctness": {
            "v4_vs_captured_f16": v4_capture_metrics.json(),
            "candidate_vs_v4": candidate_v4_metrics.json(),
            "candidate_vs_captured_f16": candidate_capture_metrics.json(),
            "candidate_partials": candidate_partials,
            "input_sha256_before": input_digest_before,
            "input_sha256_after": input_digest_after,
        },
        "protocol": {
            "runs_per_shape": args.runs,
            "cooldown_secs_per_row": args.cooldown_secs,
            "ramp_reps_per_row": RAMP_REPS,
            "scrub_working_set_bytes": scrub_elements * 4,
            "scrub_traffic_bytes": scrub_elements * 8,
            "row_order": ["A1_v4", "P_direct_f16_matrix", "A2_v4"],
            "canonical": canonical_protocol,
            "qwen_environment": qwen_environment,
        },
        "rows": [a1, candidate, a2],
        "gates": {
            "main_10pct_ms": 0.90 * main_anchor,
            "pair_10pct_ms": 0.90 * pair_anchor,
            "main_anchor_spread": main_anchor_spread,
            "pair_anchor_spread": pair_anchor_spread,
            "anchors_valid": anchors_valid,
            "main_pass": main_pass,
            "pair_pass": pair_pass,
        },
        "disposition": disposition,
    });
    println!("{}", serde_json::to_string_pretty(&packet)?);
    Ok(())
}
