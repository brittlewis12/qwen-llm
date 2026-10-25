//! Metadata-only audit after opening GGUF: no payload reads, extra mappings,
//! prefetch or Metal context.
use anyhow::{Context, bail};
use qwen_llm::{
    deepseek_v41::gguf::{ResidentMappingKind, ResidentStoragePlan, resident_storage_audit},
    gguf::GgufFile,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn layout(p: &ResidentStoragePlan) -> anyhow::Result<Value> {
    let mut digest = Sha256::new();
    digest.update(b"v41-resident-window-layout-v1\0");
    let mut union = 0u64;
    let mut previous = None::<(usize, u64)>;
    let mut original = 0;
    let mut maximum = 0;
    let windows: Vec<_> = p
        .windows
        .iter()
        .map(|w| {
            let start = w.window.mmap_offset;
            let length = w.window.length as u64;
            let end = start + length;
            let covered_end = previous
                .filter(|(shard, _)| *shard == w.window.shard_idx)
                .map_or(start, |(_, end)| end.max(start));
            union += end.saturating_sub(covered_end);
            previous = Some((w.window.shard_idx, end.max(covered_end)));
            maximum = maximum.max(length);
            let (kind, tag) = match w.mapping_kind {
                ResidentMappingKind::OriginalShared => {
                    original += 1;
                    ("original_shared", 0u8)
                }
                ResidentMappingKind::SharedEof => ("shared_eof", 1),
                ResidentMappingKind::PrivateBoundary { .. } => ("private_boundary", 2),
            };
            digest.update((w.window.shard_idx as u64).to_le_bytes());
            digest.update(start.to_le_bytes());
            digest.update(length.to_le_bytes());
            digest.update([tag]);
            digest.update(w.mapping_kind.private_bytes().to_le_bytes());
            json!({"shard_idx": w.window.shard_idx, "mmap_offset": start, "length": length,
            "mapping_kind": kind, "private_shadow_bytes": w.mapping_kind.private_bytes()})
        })
        .collect();
    let view_plus_shadow = p
        .metal_view_bytes
        .checked_add(p.private_shadow_bytes)
        .context("view plus private shadow byte count overflow")?;
    Ok(json!({
        "soft_coalescing_target": p.soft_coalescing_target,
        "effective_coalescing_target": p.soft_coalescing_target.map(|target|
            (target / p.page_size * p.page_size).max(p.page_size)
                .min(p.max_buffer_length / p.page_size * p.page_size)),
        "window_count": p.windows.len(), "max_window_length": maximum,
        "metal_view_bytes": p.metal_view_bytes, "window_union_span_bytes": union,
        "repeated_page_bytes": p.metal_view_bytes - union,
        "original_shared_window_count": original,
        "shared_eof_window_count": p.shared_eof_window_count,
        "private_window_count": p.private_window_count,
        "private_shadow_bytes": p.private_shadow_bytes,
        "metal_view_plus_private_shadow_bytes": view_plus_shadow,
        "layout_sha256": format!("{:x}", digest.finalize()), "windows": windows,
    }))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    const USAGE: &str = "usage: deepseek_v41_storage_plan MODEL_PATH PAGE_SIZE MAX_BUFFER_LENGTH\nArchived M5 example: deepseek_v41_storage_plan MODEL_PATH 16384 179357827072\nMetadata geometry only, not physical residency or runtime fit.";
    if args.len() == 1 && args[0] == "--help" {
        println!("{USAGE}");
        return Ok(());
    }
    if args.len() != 3 {
        bail!("{USAGE}");
    }
    let page_size = args[1]
        .to_string_lossy()
        .parse::<usize>()
        .context("PAGE_SIZE must be an integer")?;
    let max_buffer_length = args[2]
        .to_string_lossy()
        .parse::<usize>()
        .context("MAX_BUFFER_LENGTH must be an integer")?;
    let g = GgufFile::open(&args[0])?;
    let audit = resident_storage_audit(&g, page_size, max_buffer_length)?;
    let t = &audit.tensors;
    let top: Vec<_> = t
        .top
        .iter()
        .map(|e| {
            json!({
                "name": e.tensor.name, "dtype": format!("{:?}", e.tensor.dtype),
                "shape": e.tensor.shape, "shard_idx": e.tensor.shard_idx,
                "data_offset": e.tensor.data_offset, "payload_bytes": e.tensor.n_bytes,
                "page_rounded_singleton_bytes": e.page_rounded_singleton_bytes,
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "page_size": page_size, "hardware_max_buffer_length": max_buffer_length,
            "selected_soft_coalescing_target": audit.selected_soft_coalescing_target,
            "comparison_soft_target": audit.candidate.soft_coalescing_target,
            "packed_expert_bytes": audit.control.bytes.packed_experts,
            "dense_bytes": audit.control.bytes.dense_bytes,
            "disk_engram_bytes": audit.control.bytes.disk_engram,
            "resident_tensor_count": t.count, "resident_payload_bytes": t.payload_bytes,
            "largest_resident_payload_bytes": t.largest_payload_bytes,
            "largest_resident_tensor": top.first(),
            "largest_resident_page_rounded_singleton_bytes": t.largest_page_rounded_singleton_bytes,
            "above_effective_soft_target_singleton_count": t.above_soft_target_count,
            "above_effective_soft_target_payload_bytes": t.above_soft_target_payload_bytes,
            "top_resident_tensors_by_payload": top,
            "control": layout(&audit.control)?, "candidate": layout(&audit.candidate)?,
            "runtime_fit_guaranteed": false,
            "causal_limit": "partition changes mapping provenance as well as MTL resource extents; oversized singletons stay direct views; no physical traffic/residency claim",
        }))?
    );
    Ok(())
}
