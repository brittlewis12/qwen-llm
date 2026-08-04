use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use objc2_metal::MTLBuffer;
use qwen_llm::gguf::GgufFile;
use qwen_llm::loader::{Block, Model};
use qwen_llm::metal::{MetalContext, MetalTensor, evaluate_metal_memory_admission};
use qwen_llm::metal_dflash::{
    AttentionCapture, AttentionCaptureProvenance, MetalDFlashLayerMajorScratch,
    PrefillScratchConfig, plan_prefill_scratch_with_matrix_max_pos_configured,
    prefill_tokens_attention_capture,
};
use qwen_llm::metal_forward::{MetalForward, MetalModel, MetalSession};
use qwen_llm::model::ArchKind;
use qwen_llm::tensor::GgmlType;
use qwen_llm::tokenizer::Tokenizer;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
pub struct AttnCaptureArgs {
    /// Path to the Qwen3.6 A3B GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Raw prompt file. Tokenization uses add_special=false.
    #[arg(long)]
    file: PathBuf,
    /// Number of prompt tokens to prefill and capture.
    #[arg(long, default_value = "131072")]
    context: usize,
    /// New output directory. Existing paths are never overwritten.
    #[arg(long)]
    output: PathBuf,
    /// Comma-separated full-attention block indices.
    #[arg(long, default_value = "3,19,39", value_delimiter = ',')]
    blocks: Vec<usize>,
    /// Permit a context below 32768 for a non-representative smoke capture.
    #[arg(long)]
    allow_small_smoke: bool,
}

fn view_positions(context: usize) -> Result<Vec<usize>> {
    if context < 96 {
        bail!("smoke context must be at least 96 tokens for unique uniform queries");
    }
    let prefix = context - 32;
    let mut positions = (0..32)
        .map(|i| (2 * i + 1) * prefix / 64)
        .collect::<Vec<_>>();
    positions.extend(prefix..context);
    Ok(positions)
}

type CaptureLayout = (Vec<usize>, Vec<(usize, Vec<usize>)>);

fn capture_layout(context: usize) -> Result<CaptureLayout> {
    let contexts = if context < 32768 {
        vec![context]
    } else if context == 32768 {
        vec![32768]
    } else {
        vec![32768, context]
    };
    let views = contexts
        .into_iter()
        .map(|length| Ok((length, view_positions(length)?)))
        .collect::<Result<Vec<_>>>()?;
    let mut unique = views
        .iter()
        .flat_map(|(_, positions)| positions.iter().copied())
        .collect::<Vec<_>>();
    unique.sort_unstable();
    unique.dedup();
    Ok((unique, views))
}

fn validate_arch(
    gguf: &GgufFile,
    model: &Model<'_>,
    blocks: &[usize],
    allow_single_block: bool,
) -> Result<Vec<usize>> {
    let arch = &model.arch;
    if arch.kind != ArchKind::Moe
        || arch.n_layer != 40
        || arch.hidden_size != 2048
        || arch.n_q_heads != 16
        || arch.n_kv_heads != 2
        || arch.attn_head_dim != 256
        || arch.partial_rotary_factor != 0.25
        || arch.expert_count != 256
        || arch.expert_used_count != 8
        || arch.expert_feed_forward_length != 512
        || arch.expert_shared_feed_forward_length != 512
        || arch.full_attention_interval != 4
        || arch.gdn_n_k_heads != 16
        || arch.gdn_n_v_heads != 32
        || arch.gdn_head_dim != 128
        || arch.gdn_conv_kernel != 4
        || arch.mtp_n_hidden_layers != 0
        || gguf.get_u64("general.file_type") != Some(15)
        || gguf.get_str("general.base_model.0.name") != Some("Qwen3.6 35B A3B")
    {
        bail!("attn-capture requires exact Qwen3.6 A3B group8/head_dim256/F16-KV architecture");
    }
    let expected = block_mapping(blocks, allow_single_block)?;
    let actual = actual_block_mapping(model, blocks)?;
    if actual != expected {
        bail!("dynamic block-to-KV mapping {actual:?} does not match contract {expected:?}");
    }
    for &block in blocks {
        if !matches!(model.blocks.get(block), Some(Block::Attn(_))) {
            bail!("block {block} is not a full-attention block");
        }
    }
    Ok(expected)
}

fn actual_block_mapping(model: &Model<'_>, blocks: &[usize]) -> Result<Vec<usize>> {
    let mut attention_slot = 0usize;
    let mut result = Vec::with_capacity(blocks.len());
    for (block, layer) in model.blocks.iter().enumerate() {
        if matches!(layer, Block::Attn(_)) {
            if blocks.contains(&block) {
                result.push(attention_slot);
            }
            attention_slot += 1;
        }
    }
    if result.len() != blocks.len() {
        bail!("could not map every requested full-attention block");
    }
    Ok(result)
}

fn block_mapping(blocks: &[usize], allow_single_block: bool) -> Result<Vec<usize>> {
    if allow_single_block && blocks == [3] {
        Ok(vec![0])
    } else if blocks == [3, 19, 39] {
        Ok(vec![0, 4, 9])
    } else {
        bail!("canonical capture requires blocks 3,19,39; smoke/32K guard may use block 3");
    }
}

fn sha256_file(path: &Path) -> Result<(u64, String)> {
    let mut input = File::open(path)?;
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        bytes += n as u64;
    }
    Ok((bytes, format!("{:x}", hash.finalize())))
}

fn write_bytes(path: &Path, bytes: &[u8], prefix_len: Option<usize>) -> Result<Value> {
    let mut writer = BufWriter::new(File::create(path)?);
    let mut full = Sha256::new();
    let mut prefix = Sha256::new();
    let split = prefix_len.unwrap_or(0).min(bytes.len());
    writer.write_all(bytes)?;
    writer.flush()?;
    full.update(bytes);
    prefix.update(&bytes[..split]);
    Ok(json!({
        "byte_length": bytes.len(),
        "sha256": format!("{:x}", full.finalize()),
        "prefix_32768_sha256": prefix_len.map(|_| format!("{:x}", prefix.finalize())),
    }))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_f16_tensor(name: &str, bytes: &[u8]) -> Result<()> {
    if !bytes.len().is_multiple_of(2) {
        bail!("{name} has an odd F16 byte length");
    }
    let mut nonzero = false;
    for pair in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        if bits & 0x7c00 == 0x7c00 {
            bail!("{name} contains a non-finite F16 value");
        }
        nonzero |= bits & 0x7fff != 0;
    }
    if !nonzero {
        bail!("{name} is all zero");
    }
    Ok(())
}

fn validate_provenance(
    records: &[AttentionCaptureProvenance],
    blocks: &[usize],
    kv_slots: &[usize],
    positions: &[usize],
) -> Result<()> {
    let expected_count = blocks
        .len()
        .checked_mul(positions.len())
        .ok_or_else(|| anyhow!("provenance count overflow"))?;
    if records.len() != expected_count {
        bail!(
            "captured {} provenance rows, expected {expected_count}",
            records.len()
        );
    }
    let block_slots = blocks
        .iter()
        .copied()
        .zip(kv_slots.iter().copied())
        .collect::<std::collections::BTreeMap<_, _>>();
    let expected_positions = positions.iter().copied().collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    for record in records {
        if block_slots.get(&record.block) != Some(&record.kv_slot)
            || !expected_positions.contains(&record.position)
            || record.causal_length != record.position + 1
            || !seen.insert((record.block, record.position))
        {
            bail!("invalid or duplicate capture provenance row: {record:?}");
        }
        let valid_path = match record.path {
            "matrix" => {
                record.online_matrix
                    && record.query_rows.is_some()
                    && record.packed_rows.is_none()
                    && record.packed_qt.is_none()
                    && record.nwg.is_none()
                    && record.tile_c.is_none()
                    && record.group_tile.is_none()
            }
            "packed" => {
                !record.online_matrix
                    && record.query_rows.is_none()
                    && record.packed_rows.is_some()
                    && record.packed_qt.is_some()
                    && record.nwg.is_some()
                    && record.tile_c.is_none()
                    && record.group_tile.is_none()
            }
            "decode_fallback" => {
                !record.online_matrix
                    && record.query_rows.is_none()
                    && record.packed_rows.is_none()
                    && record.packed_qt.is_none()
                    && record.nwg.is_some()
                    && record.tile_c.is_some()
                    && record.group_tile.is_some()
            }
            _ => false,
        };
        if !valid_path {
            bail!("inconsistent capture path provenance: {record:?}");
        }
    }
    Ok(())
}

fn tensor_bytes(tensor: &MetalTensor) -> Result<&[u8]> {
    let offset = usize::try_from(tensor.offset)?;
    let length = usize::try_from(tensor.n_bytes())?;
    let buffer_len = tensor.buffer.length();
    let end = offset
        .checked_add(length)
        .ok_or_else(|| anyhow!("tensor byte overflow"))?;
    if end > buffer_len {
        bail!("tensor range {offset}..{end} exceeds Metal buffer length {buffer_len}");
    }
    unsafe {
        let base = (tensor.buffer.contents().as_ptr() as *const u8).add(offset);
        Ok(std::slice::from_raw_parts(base, length))
    }
}

pub fn run(args: AttnCaptureArgs, build_identity: Value) -> Result<()> {
    if !cfg!(target_endian = "little") {
        bail!("attn-capture raw tensor format requires a little-endian host");
    }
    let smoke = args.context < 32768;
    if smoke && !args.allow_small_smoke {
        bail!("context below 32768 requires --allow-small-smoke");
    }
    if !smoke && args.allow_small_smoke {
        bail!("--allow-small-smoke is valid only for a context below 32768");
    }
    if !smoke && !matches!(args.context, 32768 | 131072) {
        bail!("final capture context must be exactly 32768 or 131072 tokens");
    }
    let guard = args.context == 32768 && args.blocks == [3];
    let canonical = args.context == 131072 && args.blocks == [3, 19, 39];
    let capture_tier = match (smoke, guard, canonical) {
        (true, false, false) => "smoke",
        (false, true, false) => "guard_32k",
        (false, false, true) => "canonical",
        _ => {
            bail!("capture contract requires smoke/block3, 32K-guard/block3, or 131K/blocks3,19,39")
        }
    };
    let selector_environment = std::env::vars()
        .filter(|(key, _)| key.starts_with("QWEN_"))
        .collect::<std::collections::BTreeMap<_, _>>();
    if selector_environment.contains_key("QWEN_PREFILL_ATTN_MATRIX_MAX_POS") {
        bail!("QWEN_PREFILL_ATTN_MATRIX_MAX_POS conflicts with the frozen capture plan");
    }
    if !smoke && !selector_environment.is_empty() {
        bail!("final capture forbids attention selector environment overrides");
    }
    if args.output.exists() {
        bail!("output path already exists: {}", args.output.display());
    }
    let raw_prompt =
        fs::read(&args.file).with_context(|| format!("read raw prompt {}", args.file.display()))?;
    let prompt = std::str::from_utf8(&raw_prompt).context("raw prompt is not UTF-8")?;
    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open model {}", args.model.display()))?;
    let model = Model::from_gguf(&gguf).context("parse model architecture")?;
    let kv_slots = validate_arch(&gguf, &model, &args.blocks, smoke || guard)?;
    let tokenizer = Tokenizer::from_gguf(&gguf).context("open tokenizer")?;
    let mut tokens = tokenizer
        .encode(prompt, false)
        .context("tokenize raw prompt")?;
    if tokens.len() < args.context {
        bail!(
            "prompt has {} tokens, need at least {}",
            tokens.len(),
            args.context
        );
    }
    tokens.truncate(args.context);
    let (positions, views) = capture_layout(args.context)?;

    fs::create_dir(&args.output).context("create incomplete output directory")?;
    let ctx = MetalContext::new().context("initialize Metal")?;
    let metal_model = MetalModel::load(&ctx, &gguf, &model).context("load Metal model")?;
    let mut session = MetalSession::fresh(&ctx, &metal_model, args.context)
        .context("allocate capture session")?;
    for &slot in &kv_slots {
        if session.kv_k[slot].dtype != GgmlType::F16 || session.kv_v[slot].dtype != GgmlType::F16 {
            bail!("attn-capture requires production F16 K/V cache storage");
        }
    }
    let matrix_forced_off = selector_environment
        .get("QWEN_PREFILL_ATTN_MATRIX_G8")
        .is_some_and(|value| matches!(value.to_ascii_lowercase().as_str(), "0" | "false" | "no"));
    let matrix_query_cap = (!matrix_forced_off).then_some(64);
    let scratch_config = PrefillScratchConfig { matrix_query_cap };
    let scratch_plan = plan_prefill_scratch_with_matrix_max_pos_configured(
        &metal_model,
        1024,
        args.context,
        scratch_config,
    )
    .context("plan packed-prefill scratch")?;
    let expected_query_rows = matrix_query_cap.unwrap_or(1024) as u32;
    let expected_matrix_max_pos = if matrix_forced_off {
        0
    } else {
        args.context.max(1024) as u64
    };
    if scratch_plan.matrix_max_pos() != expected_matrix_max_pos
        || scratch_plan.matrix_query_rows() != expected_query_rows
    {
        bail!(
            "resolved scratch topology max_pos={} query_rows={} violates capture contract",
            scratch_plan.matrix_max_pos(),
            scratch_plan.matrix_query_rows()
        );
    }
    let scratch_upper_bytes = scratch_plan
        .priced_upper_bound(|bytes| Ok(ctx.shared_buffer_size_and_align(bytes)?.size))?;
    let capture_reserve_bytes =
        u64::try_from(args.blocks.len() * positions.len() * 2 * 4096 * std::mem::size_of::<f32>())?;
    let admission = evaluate_metal_memory_admission(
        scratch_upper_bytes,
        capture_reserve_bytes,
        ctx.memory_signals(),
        true,
    );
    if !admission.admitted {
        bail!(
            "capture memory admission denied: {}",
            admission.reason.as_str()
        );
    }
    let scratch_plan_json = json!({
        "block_size": scratch_plan.block_size(),
        "matrix_max_pos": scratch_plan.matrix_max_pos(),
        "matrix_query_rows": scratch_plan.matrix_query_rows(),
        "logical_bytes": scratch_plan.logical_bytes(),
        "maximum_logical_bytes": scratch_plan.maximum_logical_bytes()?,
        "priced_upper_bytes": scratch_upper_bytes,
        "allocations": scratch_plan.allocations().iter().map(|allocation| json!({
            "name": allocation.name(), "logical_bytes": allocation.logical_bytes(),
        })).collect::<Vec<_>>(),
        "deferred_allocations": scratch_plan.deferred_allocations().iter().map(|allocation| {
            json!({"name": allocation.name(), "logical_bytes": allocation.logical_bytes()})
        }).collect::<Vec<_>>(),
        "overlay": scratch_plan.overlay().map(|overlay| json!({
            "backing_bytes": overlay.backing_bytes,
            "attention_bytes": overlay.attention_bytes,
            "gdn_bytes": overlay.gdn_bytes,
            "saved_bytes": overlay.saved_bytes,
        })),
    });
    let admission_json = json!({
        "admitted": admission.admitted, "reason": admission.reason.as_str(),
        "scratch_upper_bytes": admission.scratch_upper_bytes,
        "capture_reserve_bytes": admission.reserve_bytes,
        "required_bytes": admission.required_bytes,
        "working_set_headroom_bytes": admission.working_set_headroom_bytes,
        "signals": {
            "recommended_max_bytes": admission.signals.recommended_max_bytes,
            "current_allocated_bytes": admission.signals.current_allocated_bytes,
            "process_limit_remaining_bytes": admission.signals.process_limit_remaining_bytes,
        },
    });
    let mut scratch =
        MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(&ctx, &metal_model, scratch_plan)
            .context("allocate packed-prefill scratch")?;
    let mut capture = AttentionCapture::new(&ctx, args.blocks.clone(), positions.clone(), 4096)
        .context("allocate sparse attention capture")?;
    let forward = MetalForward::new(&ctx, &metal_model);
    prefill_tokens_attention_capture(
        &forward,
        &tokens,
        0,
        &mut session,
        &mut scratch,
        &mut capture,
    )
    .context("capture packed prefill")?;
    validate_provenance(&capture.provenance, &args.blocks, &kv_slots, &positions)?;
    if !smoke
        && capture.provenance.iter().any(|record| {
            record.path != "matrix"
                || !record.online_matrix
                || !record.matrix_causal_skip
                || !record.query_tiled
        })
    {
        bail!("non-smoke capture did not use the frozen production matrix topology");
    }

    let token_bytes = bytemuck::cast_slice(&tokens);
    let tokens_meta = write_bytes(&args.output.join("tokens.i32le"), token_bytes, None)?;
    let kv_prefix_bytes = 32768usize * 2 * 256 * 2;
    let mut tensors = Vec::new();
    for (block_index, (&block, &slot)) in args.blocks.iter().zip(&kv_slots).enumerate() {
        for (kind, tensor) in [("k", &session.kv_k[slot]), ("v", &session.kv_v[slot])] {
            let filename = format!("block-{block}-{kind}.f16le");
            let bytes = tensor_bytes(tensor)?;
            validate_f16_tensor(&filename, bytes)?;
            let hash = write_bytes(
                &args.output.join(&filename),
                bytes,
                (args.context >= 32768).then_some(kv_prefix_bytes),
            )?;
            tensors.push(json!({
                "name": filename, "dtype": "f16le",
                "shape": [args.context, 2, 256], "strides_bytes": [1024, 512, 2],
                "hashes": hash,
            }));
        }
        for (kind, tensor) in [
            ("q", &capture.q[block_index]),
            ("o", &capture.o[block_index]),
        ] {
            let filename = format!("block-{block}-{kind}.f32le");
            let hash = write_bytes(&args.output.join(&filename), tensor_bytes(tensor)?, None)?;
            tensors.push(json!({
                "name": filename, "dtype": "f32le",
                "shape": [positions.len(), 16, 256],
                "strides_bytes": [16384, 1024, 4], "hashes": hash,
            }));
        }
    }
    let view_json = views
        .iter()
        .map(|(context, ordered)| {
            let rows = ordered
                .iter()
                .map(|position| {
                    positions
                        .binary_search(position)
                        .expect("deduplicated position")
                })
                .collect::<Vec<_>>();
            json!({"context": context, "positions": ordered, "deduplicated_rows": rows})
        })
        .collect::<Vec<_>>();
    let provenance = capture
        .provenance
        .iter()
        .map(|p| {
            json!({
                "semantics": "prefill_causal_query", "position": p.position,
                "causal_length": p.causal_length, "block": p.block, "kv_slot": p.kv_slot,
                "path": p.path, "online_matrix": p.online_matrix,
                "query_tiled": p.query_tiled, "query_rows": p.query_rows,
                "packed_rows": p.packed_rows, "packed_qt": p.packed_qt, "nwg": p.nwg,
                "tile_c": p.tile_c, "group_tile": p.group_tile,
                "matrix_causal_skip": p.matrix_causal_skip,
            })
        })
        .collect::<Vec<_>>();
    let resolved_paths = capture
        .provenance
        .iter()
        .map(|record| record.path)
        .collect::<BTreeSet<_>>();
    let (_, model_hash) = sha256_file(&args.model)?;
    let prompt_hash = sha256_bytes(&raw_prompt);
    let executable = std::env::current_exe().context("locate qwen-bench executable")?;
    let (_, executable_hash) = sha256_file(&executable)?;
    let manifest = json!({
        "schema_version": 1, "claim_scope": "prefill_causal_query",
        "capture_tier": capture_tier,
        "architecture": {"name": "Qwen3.6-A3B", "q_heads": 16, "kv_heads": 2,
            "group": 8, "head_dim": 256, "kv_dtype": "f16"},
        "context": args.context, "views": view_json, "unique_positions": positions,
        "prefill_chunk": 1024, "matrix_query_cap": matrix_query_cap,
        "resolved_attention_paths": resolved_paths,
        "scratch_plan": scratch_plan_json, "memory_admission": admission_json,
        "blocks": args.blocks.iter().zip(kv_slots).map(|(block, slot)| {
            json!({"block": block, "kv_slot": slot})
        }).collect::<Vec<_>>(),
        "tokens": {"name": "tokens.i32le", "shape": [tokens.len()],
            "dtype": "i32le", "hashes": tokens_meta},
        "tensors": tensors, "producer_provenance": provenance,
        "selector_environment": selector_environment,
        "identity": {"model_path": args.model, "model_sha256": model_hash,
            "raw_prompt_path": args.file, "raw_prompt_bytes": raw_prompt.len(),
            "raw_prompt_sha256": prompt_hash, "add_special_tokens": false,
            "qwen_bench_path": executable, "qwen_bench_sha256": executable_hash,
            "build": build_identity},
    });
    let manifest_tmp = args.output.join(".manifest.json.tmp");
    let mut manifest_file = BufWriter::new(File::create(&manifest_tmp)?);
    serde_json::to_writer_pretty(&mut manifest_file, &manifest)?;
    manifest_file.flush()?;
    drop(manifest_file);
    fs::rename(manifest_tmp, args.output.join("manifest.json"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_selection_is_unique_disjoint_and_in_range() {
        for context in [96, 32768, 131072] {
            let positions = view_positions(context).unwrap();
            assert_eq!(positions.len(), 64);
            let mut sorted = positions.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 64);
            assert!(positions[..32].iter().all(|&p| p < context - 32));
            assert!(
                positions[32..]
                    .iter()
                    .all(|&p| p >= context - 32 && p < context)
            );
        }
    }

    #[test]
    fn context_views_map_to_deduplicated_rows() {
        let (unique, views) = capture_layout(131072).unwrap();
        assert_eq!(views.len(), 2);
        for (_, positions) in views {
            for position in positions {
                assert!(unique.binary_search(&position).is_ok());
            }
        }
        assert!(unique.len() <= 128);
        assert_eq!(capture_layout(1024).unwrap().1.len(), 1);
    }

    #[test]
    fn tensor_byte_arithmetic_matches_contract() {
        assert_eq!(32768usize * 2 * 256 * 2, 33_554_432);
        assert_eq!(131072usize * 1024, 134_217_728);
        assert_eq!(64usize * 16 * 256 * 4, 1_048_576);
    }

    #[test]
    fn block_to_kv_mapping_is_frozen() {
        assert_eq!(block_mapping(&[3, 19, 39], false).unwrap(), [0, 4, 9]);
        assert_eq!(block_mapping(&[3], true).unwrap(), [0]);
        assert!(block_mapping(&[3], false).is_err());
        assert!(block_mapping(&[7, 19, 39], false).is_err());
    }

    #[test]
    fn raw_writer_hashes_exact_bytes() {
        let dir =
            std::env::temp_dir().join(format!("qwen-attn-capture-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir(&dir).unwrap();
        let meta = write_bytes(&dir.join("raw.bin"), b"abc", Some(2)).unwrap();
        assert_eq!(meta["byte_length"], 3);
        assert_eq!(
            meta["sha256"],
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
