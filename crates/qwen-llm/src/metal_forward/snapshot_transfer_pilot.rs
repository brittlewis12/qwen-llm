//! Model-free publication + one restore, with per-layer source/destination buffers.

use super::session::allocate_snapshot_arena;
use super::support::{read_tensor_into, write_tensor_bytes};
use super::*;
use crate::metal::BlitEncoder;
use objc2_metal::MTLCommandBufferStatus;
use serde_json::{Value, json};
use std::time::Instant;

pub(super) struct Section {
    pub(super) name: &'static str,
    pub(super) layer_bytes: usize,
    pub(super) layers: Vec<MetalTensor>,
}

enum Snapshot {
    Cpu(Vec<Vec<u8>>),
    Gpu(Vec<MetalTensor>),
}

pub(super) fn allocate(
    ctx: &MetalContext,
    shapes: &[(&'static str, usize, usize)],
) -> Vec<Section> {
    shapes
        .iter()
        .map(|&(name, count, layer_bytes)| {
            assert!(layer_bytes > 0 && layer_bytes.is_multiple_of(4));
            Section {
                name,
                layer_bytes,
                layers: (0..count)
                    .map(|_| MetalTensor::zeros_f32(ctx, vec![(layer_bytes / 4) as u64]).unwrap())
                    .collect(),
            }
        })
        .collect()
}

pub(super) fn payload(t: &MetalTensor) -> &[u8] {
    assert_eq!(t.buffer.storageMode(), objc2_metal::MTLStorageMode::Shared);
    assert!(t.offset + t.n_bytes() <= t.buffer.length() as u64);
    unsafe {
        std::slice::from_raw_parts(
            (t.buffer.contents().as_ptr() as *const u8).add(t.offset as usize),
            t.n_bytes() as usize,
        )
    }
}

pub(super) fn set_first(t: &MetalTensor, byte: u8) {
    assert!(!payload(t).is_empty());
    // Only used in the synchronous immutability oracle, never during GPU work.
    unsafe {
        *((t.buffer.contents().as_ptr() as *mut u8).add(t.offset as usize)) = byte;
    }
}

fn blit(ctx: &MetalContext, sections: &[Section], arenas: &[MetalTensor], capture: bool) {
    assert_eq!(sections.len(), arenas.len());
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = BlitEncoder::begin(&cmd);
    for (section, arena) in sections.iter().zip(arenas) {
        assert_eq!(
            arena.n_bytes() as usize,
            section.layer_bytes * section.layers.len()
        );
        for (index, layer) in section.layers.iter().enumerate() {
            let offset = (index * section.layer_bytes) as u64;
            if capture {
                enc.copy_buffer(
                    &layer.buffer,
                    layer.offset,
                    &arena.buffer,
                    arena.offset + offset,
                    section.layer_bytes as u64,
                );
            } else {
                enc.copy_buffer(
                    &arena.buffer,
                    arena.offset + offset,
                    &layer.buffer,
                    layer.offset,
                    section.layer_bytes as u64,
                );
            }
        }
    }
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
    assert!(cmd.error().is_none());
}

fn restore(ctx: &MetalContext, snapshot: &Snapshot, dst: &[Section]) {
    match snapshot {
        Snapshot::Cpu(arenas) => {
            for (section, arena) in dst.iter().zip(arenas) {
                assert_eq!(arena.len(), section.layer_bytes * section.layers.len());
                for (index, layer) in section.layers.iter().enumerate() {
                    let offset = index * section.layer_bytes;
                    write_tensor_bytes(layer, &arena[offset..offset + section.layer_bytes]);
                }
            }
        }
        Snapshot::Gpu(arenas) => blit(ctx, dst, arenas, false),
    }
}

fn roundtrip(ctx: &MetalContext, src: &[Section], gpu: bool, oracle: bool) -> Value {
    let before = ctx.current_allocated_size();
    let start = Instant::now();
    let snapshot = if gpu {
        let arenas = src
            .iter()
            .map(|s| {
                MetalTensor::zeros_f32(ctx, vec![(s.layer_bytes * s.layers.len() / 4) as u64])
                    .unwrap()
            })
            .collect::<Vec<_>>();
        blit(ctx, src, &arenas, true);
        Snapshot::Gpu(arenas)
    } else {
        let arenas = src
            .iter()
            .map(|section| {
                let mut arena = allocate_snapshot_arena(
                    section.name,
                    section.layer_bytes * section.layers.len(),
                )
                .unwrap();
                for (index, layer) in section.layers.iter().enumerate() {
                    let offset = index * section.layer_bytes;
                    read_tensor_into(&mut arena[offset..offset + section.layer_bytes], layer);
                }
                arena
            })
            .collect();
        Snapshot::Cpu(arenas)
    };
    let publication_ms = start.elapsed().as_secs_f64() * 1e3;
    let after_publication = ctx.current_allocated_size();
    let first = if oracle {
        payload(&src[0].layers[0])[0]
    } else {
        0
    };
    if oracle {
        set_first(&src[0].layers[0], first ^ 0xff);
    }
    let restore_start = Instant::now();
    let shapes: Vec<_> = src
        .iter()
        .map(|s| (s.name, s.layers.len(), s.layer_bytes))
        .collect();
    let dst = allocate(ctx, &shapes);
    let destination_allocation_ms = restore_start.elapsed().as_secs_f64() * 1e3;
    restore(ctx, &snapshot, &dst);
    let restore_with_allocation_ms = restore_start.elapsed().as_secs_f64() * 1e3;
    let total_ms = start.elapsed().as_secs_f64() * 1e3;
    let metal_increment = ctx.current_allocated_size() - before;
    let live_metal_bytes = ctx.current_allocated_size();
    let cpu_snapshot_capacity = match &snapshot {
        Snapshot::Cpu(arenas) => arenas.iter().map(|a| a.capacity()).sum::<usize>(),
        Snapshot::Gpu(_) => 0,
    };
    let recommended_bytes = ctx.recommended_max_working_set_size();
    assert!(live_metal_bytes + cpu_snapshot_capacity as u64 + 64 * 1024 * 1024 < recommended_bytes);
    if oracle {
        assert_eq!(
            payload(&dst[0].layers[0])[0],
            first,
            "producer mutation leaked into snapshot"
        );
        set_first(&src[0].layers[0], first);
        set_first(&dst[0].layers[0], first ^ 0x55);
        restore(ctx, &snapshot, &dst);
        for (a, b) in src.iter().zip(&dst) {
            for (index, (a, b)) in a.layers.iter().zip(&b.layers).enumerate() {
                assert!(
                    payload(a) == payload(b),
                    "restored payload differs at layer {index}"
                );
            }
        }
    }
    json!({"arm":if gpu {"B"} else {"A"}, "publication_ms":publication_ms,
        "restore_with_allocation_ms":restore_with_allocation_ms,
        "destination_allocation_ms":destination_allocation_ms, "total_ms":total_ms,
        "metal_increment_bytes":metal_increment, "cpu_snapshot_capacity_bytes":cpu_snapshot_capacity,
        "snapshot_metal_increment_bytes":after_publication-before,
        "destination_metal_increment_bytes":live_metal_bytes-after_publication,
        "live_metal_bytes":live_metal_bytes, "recommended_working_set_bytes":recommended_bytes,
        "driver_headroom_bytes":recommended_bytes-live_metal_bytes,
        "copy_regions_per_direction":src.iter().map(|s| s.layers.len()).sum::<usize>(),
        "gpu_waits":if gpu {2} else {0}, "oracle":oracle})
}

#[test]
#[ignore = "serial Metal, model-free immutable snapshot publication/restore"]
fn snapshot_cpu_vs_gpu_publication_restore() {
    assert!(
        std::env::var_os("QWEN_KV_Q8").is_none(),
        "this models F16 KV, not weight quantization"
    );
    let ctx = MetalContext::new().unwrap();
    for prefix in [8840usize, 32752] {
        // Dense27 with F16 KV. Weight quantization is not KV-cache quantization.
        let shapes = [
            ("K", 16, prefix * 4 * 256 * 2),
            ("V", 16, prefix * 4 * 256 * 2),
            ("GDN-conv", 48, 3 * 80 * 128 * 4),
            ("GDN-state", 48, 48 * 128 * 128 * 4),
        ];
        let logical: usize = shapes.iter().map(|&(_, n, size)| n * size).sum();
        assert!((3 * logical) as u64 + 64 * 1024 * 1024 < ctx.recommended_max_working_set_size());
        let before = ctx.current_allocated_size();
        let source = allocate(&ctx, &shapes);
        for (section, s) in source.iter().enumerate() {
            for (layer, t) in s.layers.iter().enumerate() {
                // Synthetic completed producer; exact raw bytes, no model forward.
                unsafe {
                    let dst = std::slice::from_raw_parts_mut(
                        t.buffer.contents().as_ptr() as *mut u8,
                        t.n_bytes() as usize,
                    );
                    for (i, byte) in dst.iter_mut().enumerate() {
                        *byte = (i ^ (layer * 17) ^ (section * 53)) as u8;
                    }
                }
            }
        }
        println!(
            "SNAPSHOT_TRANSFER_JSON {}",
            json!({"kind":"payload", "prefix":prefix,
            "logical_bytes":logical, "source_metal_bytes":ctx.current_allocated_size()-before,
            "three_live_payload_slots_bytes":3*logical, "model_loaded":false, "kv_dtype":"F16"})
        );
        for gpu in [false, true] {
            let mut result = objc2::rc::autoreleasepool(|_| roundtrip(&ctx, &source, gpu, true));
            result["kind"] = json!("oracle");
            result["prefix"] = json!(prefix);
            println!("SNAPSHOT_TRANSFER_JSON {result}");
        }
        // No payload readbacks in either conditioning or measured ABBA blocks.
        for (index, gpu) in [false, true, true, false, false, true, true, false]
            .into_iter()
            .enumerate()
        {
            let mut result = objc2::rc::autoreleasepool(|_| roundtrip(&ctx, &source, gpu, false));
            result["kind"] = json!(if index < 4 { "warmup" } else { "sample" });
            result["prefix"] = json!(prefix);
            result["index"] = json!(index);
            println!("SNAPSHOT_TRANSFER_JSON {result}");
        }
    }
}
