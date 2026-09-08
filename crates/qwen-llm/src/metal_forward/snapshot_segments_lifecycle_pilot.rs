//! Model-free, existing-root publication + full restore. No production anchor API.

use super::super::session::allocate_snapshot_arena;
use super::super::snapshot_transfer_pilot::{Section, allocate, payload, set_first};
use super::super::support::{read_tensor_into, write_tensor_bytes};
use super::Snapshot;
use crate::metal::{MetalContext, MetalTensor};
use objc2_metal::{MTLBuffer, MTLResource};
use serde_json::{Value, json};
use std::time::Instant;

const ROW_BYTES: usize = 2048;

struct Published {
    kv: [Snapshot; 2],
    recurrent: [Vec<u8>; 2],
}

struct Producer {
    sections: Vec<Section>,
    rows: usize,
    anchor: Option<[Snapshot; 2]>,
}

fn capture_flat(section: &Section, used: usize) -> Vec<u8> {
    assert!(used <= section.layer_bytes);
    let mut arena = allocate_snapshot_arena(section.name, section.layers.len() * used).unwrap();
    for (index, layer) in section.layers.iter().enumerate() {
        read_tensor_into(&mut arena[index * used..(index + 1) * used], layer);
    }
    arena
}

fn write_span(t: &MetalTensor, offset: usize, bytes: &[u8]) {
    assert!(offset.checked_add(bytes.len()).unwrap() <= t.n_bytes() as usize);
    assert_eq!(t.buffer.storageMode(), objc2_metal::MTLStorageMode::Shared);
    if offset == 0 {
        write_tensor_bytes(t, bytes);
        return;
    }
    // Dedicated test-owned buffers, synchronously accessed; no GPU commands run.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            (t.buffer.contents().as_ptr() as *mut u8).add(t.offset as usize + offset),
            bytes.len(),
        );
    }
}

impl Published {
    fn restore(&self, dst: &[Section]) {
        assert_eq!(dst.len(), 4);
        for (snapshot, section) in self.kv.iter().zip(dst) {
            snapshot.validate().unwrap();
            assert_eq!(snapshot.layers, section.layers.len());
            assert!(snapshot.rows * snapshot.row_bytes <= section.layer_bytes);
            let mut start = 0;
            for segment in &snapshot.segments {
                let span = segment.rows * snapshot.row_bytes;
                for (index, layer) in section.layers.iter().enumerate() {
                    write_span(
                        layer,
                        start,
                        &segment.bytes[index * span..(index + 1) * span],
                    );
                }
                start += span;
            }
        }
        for (arena, section) in self.recurrent.iter().zip(&dst[2..]) {
            assert_eq!(arena.len(), section.layer_bytes * section.layers.len());
            for (index, layer) in section.layers.iter().enumerate() {
                let start = index * section.layer_bytes;
                write_tensor_bytes(layer, &arena[start..start + section.layer_bytes]);
            }
        }
    }

    fn logical_bytes(&self) -> usize {
        self.kv.iter().map(Snapshot::logical_bytes).sum::<usize>()
            + self.recurrent.iter().map(Vec::len).sum::<usize>()
    }
}

impl Producer {
    fn restore(&mut self, root: &Published) {
        self.anchor = None;
        root.restore(&self.sections);
        assert_eq!(root.kv[0].rows, root.kv[1].rows);
        self.rows = root.kv[0].rows;
        self.anchor = Some(root.kv.clone());
    }

    fn append_synthetic(&mut self, rows: usize) {
        let end = self.rows.checked_add(rows).unwrap();
        for section in &self.sections[..2] {
            assert!(end * ROW_BYTES <= section.layer_bytes);
        }
        for (section_index, section) in self.sections.iter().enumerate() {
            for (layer_index, layer) in section.layers.iter().enumerate() {
                let (offset, count) = if section_index < 2 {
                    (self.rows * ROW_BYTES, rows * ROW_BYTES)
                } else {
                    (0, section.layer_bytes)
                };
                let bytes =
                    super::pattern(1, count, 1, section_index * 67 + layer_index + 91 + end);
                write_span(layer, offset, &bytes);
            }
        }
        self.rows = end;
    }

    fn raw_mut(&mut self) -> &mut [Section] {
        self.anchor = None;
        &mut self.sections
    }

    fn capture(&self, incremental: bool) -> (Published, usize) {
        let mut copied = 0;
        let kv = std::array::from_fn(|index| {
            let section = &self.sections[index];
            if incremental {
                let layers: Vec<_> = section.layers.iter().map(payload).collect();
                let (snapshot, bytes) = Snapshot::capture_from(
                    &layers,
                    ROW_BYTES,
                    self.rows,
                    self.anchor.as_ref().map(|a| &a[index]),
                )
                .unwrap();
                copied += bytes;
                snapshot
            } else {
                let bytes = capture_flat(section, self.rows * ROW_BYTES);
                copied += bytes.len();
                Snapshot::from_flat(section.layers.len(), self.rows, ROW_BYTES, bytes).unwrap()
            }
        });
        let recurrent = std::array::from_fn(|index| {
            let section = &self.sections[index + 2];
            let bytes = capture_flat(section, section.layer_bytes);
            copied += bytes.len();
            bytes
        });
        (Published { kv, recurrent }, copied)
    }
}

fn cycle(
    ctx: &MetalContext,
    producer: &Producer,
    root: &Published,
    incremental: bool,
    oracle: bool,
) -> Value {
    let before = ctx.current_allocated_size();
    let start = Instant::now();
    let (snapshot, copied) = producer.capture(incremental);
    let publication_ms = start.elapsed().as_secs_f64() * 1e3;
    let restore_start = Instant::now();
    let shapes: Vec<_> = producer
        .sections
        .iter()
        .map(|s| (s.name, s.layers.len(), s.layer_bytes))
        .collect();
    let dst = allocate(ctx, &shapes);
    let allocation_ms = restore_start.elapsed().as_secs_f64() * 1e3;
    snapshot.restore(&dst);
    let restore_ms = restore_start.elapsed().as_secs_f64() * 1e3;
    let total_ms = start.elapsed().as_secs_f64() * 1e3;
    let logical = snapshot.logical_bytes();
    let live_metal = ctx.current_allocated_size();
    let unique_cpu = root.logical_bytes() + copied;
    assert!(
        live_metal + unique_cpu as u64 + 64 * 1024 * 1024 < ctx.recommended_max_working_set_size()
    );
    if incremental && producer.anchor.is_some() {
        for (a, b) in snapshot.kv.iter().zip(&root.kv) {
            assert!(std::sync::Arc::ptr_eq(
                &a.segments[0].bytes,
                &b.segments[0].bytes
            ));
        }
        assert_eq!(
            copied,
            (producer.rows - root.kv[0].rows) * ROW_BYTES * 32 + 156_893_184
        );
    } else {
        assert_eq!(copied, logical);
    }
    if oracle {
        for (old, new) in root.recurrent.iter().zip(&snapshot.recurrent) {
            assert_ne!(old.as_ptr(), new.as_ptr());
            assert!(
                old != new,
                "synthetic recurrent transition did not change state"
            );
        }
        for (source, destination) in producer.sections.iter().zip(&dst) {
            for (a, b) in source.layers.iter().zip(&destination.layers) {
                assert!(payload(a) == payload(b), "full restored payload mismatch");
            }
        }
        // Change every producer/destination region after publication, then restore again.
        for (source, destination) in producer.sections.iter().zip(&dst) {
            for (a, b) in source.layers.iter().zip(&destination.layers) {
                set_first(a, payload(a)[0] ^ 0xff);
                set_first(b, payload(b)[0] ^ 0x55);
            }
        }
        snapshot.restore(&dst);
        for (source, destination) in producer.sections.iter().zip(&dst) {
            for (a, b) in source.layers.iter().zip(&destination.layers) {
                assert_eq!(payload(a)[0] ^ 0xff, payload(b)[0]);
                set_first(a, payload(a)[0] ^ 0xff);
                assert!(payload(a) == payload(b));
            }
        }
        for (k, section) in snapshot.kv.iter().zip(&producer.sections) {
            let flat = k.materialize().unwrap();
            for (index, layer) in section.layers.iter().enumerate() {
                assert!(
                    flat[index * section.layer_bytes..(index + 1) * section.layer_bytes]
                        == *payload(layer)
                );
            }
        }
    }
    json!({"arm":if incremental {"B"} else {"A"}, "publication_ms":publication_ms,
        "restore_with_allocation_ms":restore_ms, "destination_allocation_ms":allocation_ms,
        "total_ms":total_ms, "publication_copied_bytes":copied, "full_restore_bytes":logical,
        "root_logical_bytes":root.logical_bytes(), "snapshot_logical_bytes":logical,
        "two_entries_logical_bytes":root.logical_bytes()+logical, "unique_live_cpu_payload_bytes":unique_cpu,
        "live_metal_bytes":live_metal, "destination_metal_increment_bytes":live_metal-before,
        "kv_segments":snapshot.kv.iter().map(|k| k.segments.len()).collect::<Vec<_>>(),
        "restore_copy_regions":snapshot.kv.iter().map(|k| k.layers*k.segments.len()).sum::<usize>()+96,
        "oracle":oracle})
}

#[test]
#[ignore = "serial Metal, model-free incremental CPU snapshot lifecycle"]
fn snapshot_segmented_publication_full_restore() {
    assert!(std::env::var_os("QWEN_KV_Q8").is_none());
    let ctx = MetalContext::new().unwrap();
    for prefix in [8840usize, 32752] {
        let tail = 16;
        let shapes = [
            ("K", 16, (prefix + tail) * ROW_BYTES),
            ("V", 16, (prefix + tail) * ROW_BYTES),
            ("GDN-conv", 48, 3 * 80 * 128 * 4),
            ("GDN-state", 48, 48 * 128 * 128 * 4),
        ];
        let logical: usize = shapes.iter().map(|&(_, n, bytes)| n * bytes).sum();
        assert!(((5 * logical + 64 * 1024 * 1024) as u64) < ctx.recommended_max_working_set_size());
        let mut producer = Producer {
            sections: allocate(&ctx, &shapes),
            rows: 0,
            anchor: None,
        };
        producer.append_synthetic(prefix);
        let start = Instant::now();
        let (root, root_copied) = producer.capture(false);
        let root_publication_ms = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        producer.restore(&root);
        let initial_restore_ms = start.elapsed().as_secs_f64() * 1e3;
        producer.append_synthetic(tail);
        println!(
            "SNAPSHOT_SEGMENTS_JSON {}",
            json!({"kind":"setup", "prefix":prefix, "tail":tail,
            "root_publication_ms":root_publication_ms, "root_copied_bytes":root_copied,
            "initial_restore_ms":initial_restore_ms, "model_loaded":false, "kv_dtype":"F16"})
        );
        for incremental in [false, true] {
            let mut row =
                objc2::rc::autoreleasepool(|_| cycle(&ctx, &producer, &root, incremental, true));
            row["kind"] = json!("oracle");
            row["prefix"] = json!(prefix);
            println!("SNAPSHOT_SEGMENTS_JSON {row}");
        }
        // Negative gate: exposing unchanged raw state still requires full publication.
        let _ = producer.raw_mut();
        let (unanchored, copied) = producer.capture(true);
        assert_eq!(copied, logical);
        drop(unanchored);
        producer.restore(&root);
        producer.append_synthetic(tail);
        // Oracles/export finish before separate warm and measured ABBA, with no readbacks.
        for (index, incremental) in [false, true, true, false, false, true, true, false]
            .into_iter()
            .enumerate()
        {
            let mut row =
                objc2::rc::autoreleasepool(|_| cycle(&ctx, &producer, &root, incremental, false));
            row["kind"] = json!(if index < 4 { "warmup" } else { "sample" });
            row["prefix"] = json!(prefix);
            row["index"] = json!(index);
            println!("SNAPSHOT_SEGMENTS_JSON {row}");
        }
    }
}
