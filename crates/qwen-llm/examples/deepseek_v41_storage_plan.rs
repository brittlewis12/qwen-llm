//! CPU pre-import audit. Opening GGUF retains shared shard mappings; planning
//! reads bounded tensor-serialized controls, but no weight or table payloads.
//! It creates no Metal context or GPU mapping.
use anyhow::{Context, bail};
use qwen_llm::{
    deepseek_v41::gguf::{ResidentMappingKind, resident_storage_plan},
    gguf::GgufFile,
};
use serde_json::json;

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    const USAGE: &str = "usage: deepseek_v41_storage_plan MODEL_PATH PAGE_SIZE MAX_BUFFER_LENGTH\nArchived M5 geometry: PAGE_SIZE=16384 MAX_BUFFER_LENGTH=179357827072\nOutput is bytes, not a runtime fit/admission guarantee.";
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
    let p = resident_storage_plan(&g, page_size, max_buffer_length)?;
    let view_plus_shadow = p
        .metal_view_bytes
        .checked_add(p.private_shadow_bytes)
        .context("view plus private shadow byte count overflow")?;
    let windows: Vec<_> = p
        .windows
        .iter()
        .map(|w| {
            json!({
                "shard_idx": w.window.shard_idx,
                "mmap_offset": w.window.mmap_offset,
                "length": w.window.length,
                "mapping_kind": match w.mapping_kind {
                    ResidentMappingKind::OriginalShared => "original_shared",
                    ResidentMappingKind::SharedEof => "shared_eof",
                    ResidentMappingKind::PrivateBoundary { .. } => "private_boundary",
                },
                "private_shadow_bytes": w.mapping_kind.private_bytes(),
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "model_path": args[0].to_string_lossy(),
            "page_size": p.page_size,
            "max_buffer_length": p.max_buffer_length,
            "packed_expert_bytes": p.bytes.packed_experts,
            "dense_bytes": p.bytes.dense_bytes,
            "disk_engram_bytes": p.bytes.disk_engram,
            "metal_view_bytes": p.metal_view_bytes,
            "private_shadow_bytes": p.private_shadow_bytes,
            "metal_view_plus_private_shadow_bytes": view_plus_shadow,
            "shared_eof_window_count": p.shared_eof_window_count,
            "private_window_count": p.private_window_count,
            "window_count": p.windows.len(),
            "windows": windows,
            "runtime_fit_guaranteed": false,
        }))?
    );
    Ok(())
}
