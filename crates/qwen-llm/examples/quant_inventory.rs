//! CPU-only GGUF storage inventory and small Q2_0 row checks. No Metal context,
//! prefetch, or whole-tensor dequantization. This does not qualify execution.
use anyhow::{Context, Result, ensure};
use qwen_llm::{codec, gguf::GgufFile, tensor::GgmlType};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: quant_inventory FIRST_SHARD.gguf")?;
    let gguf = GgufFile::open(&path)?;
    let before = gguf.revalidate_retained_shard_stamps()?;
    let mut counts = BTreeMap::<String, usize>::new();
    let mut roles = BTreeMap::<String, BTreeMap<String, usize>>::new();
    let mut checks = Vec::new();
    for tensor in &gguf.tensors {
        let dtype = tensor.dtype.wire_name().to_owned();
        *counts.entry(dtype.clone()).or_default() += 1;
        let role = tensor
            .name
            .strip_prefix("blk.")
            .and_then(|name| name.split_once('.').map(|(_, suffix)| suffix))
            .unwrap_or(&tensor.name);
        *roles
            .entry(role.to_owned())
            .or_default()
            .entry(dtype)
            .or_default() += 1;
        if tensor.dtype != GgmlType::Q2_0 {
            continue;
        }
        let width = tensor.shape[0];
        if width > 65_536 {
            checks.push(json!({"name": tensor.name, "shape": tensor.shape,
                "status": "skipped", "reason": "row exceeds diagnostic sample limit of 65536 elements"}));
            continue;
        }
        let rows = tensor
            .checked_n_elements()
            .context("tensor element count overflow")?
            / width;
        let row_bytes = usize::try_from(width / 64 * 18)?;
        let payload = gguf.try_slice(tensor)?;
        let mut samples = Vec::new();
        let mut selected = vec![0, rows / 2, rows - 1];
        selected.sort_unstable();
        selected.dedup();
        for row in selected {
            let offset = usize::try_from(row)?
                .checked_mul(row_bytes)
                .context("row offset overflow")?;
            let bytes = &payload[offset..offset + row_bytes];
            let mut desc = tensor.clone();
            desc.shape = vec![width];
            desc.n_bytes = row_bytes as u64;
            desc.data_offset += offset as u64;
            let actual = codec::dequant_to_f32(&desc, bytes)?;
            let expected: Vec<f32> = bytes
                .chunks_exact(18)
                .flat_map(|block| {
                    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                    block[2..].iter().flat_map(move |&byte| {
                        [1u16, 4, 16, 64].into_iter().map(move |divisor| {
                            ((u16::from(byte) / divisor % 4) as f32 - 1.0) * scale
                        })
                    })
                })
                .collect();
            ensure!(
                actual.len() == expected.len(),
                "{} row {row}: length mismatch",
                tensor.name
            );
            ensure!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, e)| a.is_finite() && a.to_bits() == e.to_bits()),
                "{} row {row}: nonfinite value or codec mismatch",
                tensor.name
            );
            samples.push(json!({"row": row, "file_offset": desc.data_offset,
                "bytes": row_bytes, "sha256": format!("{:x}", Sha256::digest(bytes)),
                "comparison": "independent_decode_bitwise_equal"}));
        }
        checks.push(json!({"name": tensor.name, "shape": tensor.shape,
            "shard": tensor.shard_idx, "samples": samples}));
    }
    let after = gguf.revalidate_retained_shard_stamps()?;
    ensure!(
        before == after,
        "retained shard stamps changed during sampling"
    );
    let sources: Vec<_> = before.iter().map(|s| json!({
        "shard": s.shard_idx, "path": s.path, "device": s.device, "inode": s.inode,
        "bytes": s.size, "mtime": [s.mtime_sec, s.mtime_nsec], "ctime": [s.ctime_sec, s.ctime_nsec]
    })).collect();
    let hash = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
    let report = json!({"schema": "gguf_quant_inventory_v1", "path": path,
        "producer": {
            "example_sha256": hash(include_bytes!("quant_inventory.rs")),
            "codec_sha256": hash(include_bytes!("../src/codec.rs")),
            "tensor_sha256": hash(include_bytes!("../src/tensor.rs")),
            "loader_sha256": hash(include_bytes!("../src/gguf.rs")),
            "cargo_lock_sha256": hash(include_bytes!("../../../Cargo.lock"))
        }, "sources": sources, "source_stamps_unchanged": true,
        "shards": gguf.shard_count(), "tensor_count": gguf.tensors.len(),
        "types": counts, "roles": roles, "q2_0_checks": checks,
        "execution_qualification": "not_evaluated", "metal_initialized": false});
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    println!();
    Ok(())
}
