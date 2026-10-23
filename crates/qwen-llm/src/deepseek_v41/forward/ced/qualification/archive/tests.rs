use super::*;
use std::fs;
use std::os::unix::fs::DirBuilderExt;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture {
    dir: PathBuf,
    g: Geometry,
    input: Vec<u8>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.dir).unwrap();
    }
}

impl Fixture {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "v41-ced-qualification-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::DirBuilder::new().mode(0o700).create(&dir).unwrap();
        let g = Geometry {
            n: 7,
            layers: 3,
            cut: 1,
            window: 2,
            hidden: 2,
            head_dim: 2,
            index_dim: 2,
            vocab: 3,
            topk: 2,
            block: 2,
            owners: vec![(1, 1)],
            producers: vec![1, 2],
        };
        let input = serde_json::to_vec(
            &json!({"prompt": [0,1,2,1,0,2,1], "continuation": vec![1u32; STEPS]}),
        )
        .unwrap();
        let fixture = Self {
            dir,
            g: g.clone(),
            input,
        };
        let ids: Value = serde_json::from_slice(&fixture.input).unwrap();
        fixture.json("input.json", &ids);
        let shards = json!([{"shard_idx": 0, "size": 64, "alignment": 32, "tensor_data_start": 32, "sha256": sha(&[0; 64])}]);
        let layout = json!({"metadata": {}, "tensors": [], "shards": []});
        fixture.json("artifact.json", &json!({"layout": layout, "layout_sha256": json_sha(&layout).unwrap(), "shards": shards}));
        fixture.json("provenance.json", &json!({"runtime_source_commit": HARNESS, "original_runtime_pin": RUNTIME, "binary_sha256": sha(b"fixture")}));
        let ranges: Vec<_> = (g.cut..g.layers)
            .map(|l| {
                json!({"layer": l, "range": [g.output_start(l), g.n],
            "seed_range": [g.output_start(l).saturating_sub(g.window - 1), g.output_start(l)]})
            })
            .collect();
        fixture.json("header.json", &json!({"schema": SCHEMA, "arm": "native-token-major-f32", "original_runtime_pin": RUNTIME,
            "N": g.n, "capacity": g.n + STEPS, "continuation_steps": STEPS, "decoder_start": g.cut,
            "boundary_layer": g.cut - 1, "window": g.window, "prefill_chunk_rows": 2048,
            "B": 1 + (g.layers - g.cut) * (g.window - 1), "boundary_range": [g.boundary_start(), g.n],
            "decoder_output_ranges": ranges, "input_file_sha256": sha(&fixture.input), "input_ids_sha256": json_sha(&ids).unwrap(),
            "artifact_layout_sha256": json_sha(&layout).unwrap(), "expected_counts": {
                "boundary_rows_each_E_P": g.n - g.boundary_start(),
                "query_control_rows_each_selected_candidates": (g.cut..g.layers).map(|l| g.n - g.output_start(l)).sum::<usize>(),
                "full_logit_vectors": STEPS + 1, "causal_hash_positions": STEPS + 1,
                "causal_regions_each_position": g.layers + 4 * g.owners.len(), "full_binary_state_positions": [g.n, g.n + STEPS]}}));
        let mut data = Vec::new();
        let mut records = Vec::new();
        let mut add = |mut meta: Value, bytes: &[u8]| -> Value {
            meta["filename"] = json!("data.bin");
            meta["byte_offset"] = json!(data.len());
            meta["byte_length"] = json!(bytes.len());
            meta["sha256"] = json!(sha(bytes));
            meta["record"] = json!(records.len());
            data.extend_from_slice(bytes);
            records.push(meta.clone());
            meta
        };
        for row in g.boundary_start()..g.n {
            for role in ["E", "P"] {
                let shape = if role == "E" {
                    vec![4, g.hidden]
                } else {
                    vec![4]
                };
                add(
                    json!({"role": role, "dtype": "f32-le", "layer": g.cut - 1, "producer": g.cut - 1,
                    "absroworigin": row, "shape": shape}),
                    &vec![0; shape.iter().product::<usize>() * 4],
                );
            }
        }
        for layer in g.cut..g.layers {
            for row in g.output_start(layer)..g.n {
                let selected = vec![0i32, 1];
                let candidates = vec![1i32; (row + 1).div_ceil(g.block)];
                for (role, values, producer) in [
                    ("selected", &selected, g.producer(layer)),
                    ("candidates", &candidates, g.cut),
                ] {
                    add(
                        json!({"role": role, "dtype": "i32-le", "shape": [values.len()], "layer": layer,
                        "producer": producer, "absroworigin": row}),
                        bytemuck::cast_slice(values),
                    );
                }
            }
        }
        let mut states = Vec::new();
        for position in g.n..=g.n + STEPS {
            add(
                json!({"role": "logits", "dtype": "f32-le", "shape": [g.vocab], "layer": null,
                "producer": "head", "absroworigin": position - 1, "state_position": position}),
                &vec![0; g.vocab * 4],
            );
            let full = position == g.n || position == g.n + STEPS;
            let mut regions = Vec::new();
            for (role, layer) in g.state_keys() {
                let shape = g.state_shape(&role, layer, position).unwrap();
                let bytes = vec![0; shape.iter().product::<usize>() * 4];
                let meta = json!({"role": role, "dtype": "f32-le", "shape": shape, "layer": layer,
                    "producer": layer, "absroworigin": if role.starts_with("owner_completed_") { Some(0) } else { None },
                    "state_position": position, "sha256": sha(&bytes)});
                regions.push(if full { add(meta, &bytes) } else { meta });
            }
            let address = json!({"position": position, "compressed_pad": 2, "compressed_history_oldest_first": [0, 1]});
            states.push(json!({"position": position, "full_binary_state": full, "regions": regions,
                "region_count": g.layers + 4 * g.owners.len(), "address": address, "address_sha256": json_sha(&address).unwrap(),
                "physical_ring_next_slot": position % g.window, "pending_ratio2_active": position % 2 == 1,
                "attention_cursor": {"next_position": position, "next_layer": 0, "batch": null, "validated_by": "validate_snapshot_position"}}));
        }
        fs::write(fixture.dir.join("data.bin"), data).unwrap();
        fixture.jsonl("regions.jsonl", &records);
        fixture.jsonl("states.jsonl", &states);
        fixture.json("dispatch-first-scalar.json", &json!({}));
        fixture.json("dispatch-last-scalar.json", &json!({}));
        fixture.manifest();
        fixture
    }
    fn json(&self, name: &str, v: &Value) {
        fs::write(self.dir.join(name), serde_json::to_vec(v).unwrap()).unwrap();
    }
    fn read(&self, name: &str) -> Value {
        serde_json::from_slice(&fs::read(self.dir.join(name)).unwrap()).unwrap()
    }
    fn jsonl(&self, name: &str, values: &[Value]) {
        fs::write(
            self.dir.join(name),
            values
                .iter()
                .map(|v| format!("{}\n", serde_json::to_string(v).unwrap()))
                .collect::<String>(),
        )
        .unwrap();
    }
    fn lines(&self, name: &str) -> Vec<Value> {
        fs::read_to_string(self.dir.join(name))
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect()
    }
    fn manifest(&self) {
        let files: Vec<_> = FILES
            .iter()
            .map(|name| {
                let bytes = fs::read(self.dir.join(name)).unwrap();
                json!({"filename": name, "bytes": bytes.len(), "sha256": sha(&bytes)})
            })
            .collect();
        self.json("complete.json", &json!({"schema": SCHEMA, "complete": true, "initial_position": self.g.n,
            "final_position": self.g.n + STEPS, "state_positions": STEPS + 1, "regions": self.lines("regions.jsonl").len(), "files": files}));
    }
    fn load(&self) -> Result<Reference> {
        Reference::load(
            &self.dir,
            self.g.clone(),
            &self.input,
            "native-token-major-f32",
        )
    }
}

#[test]
fn complete_fixture_maps_absolute_rows_without_copying_payloads() {
    let f = Fixture::new();
    assert!(cpu_reserve(&f.dir).unwrap() > fs::metadata(f.dir.join("data.bin")).unwrap().len());
    let r = f.load().unwrap();
    for (start, rows) in [(0, 5), (5, 2), (4, 1), (5, 1), (6, 1)] {
        for row in start.max(f.g.boundary_start())..start + rows {
            assert_eq!(
                r.floats(r.get("E", Some(0), row).unwrap()).unwrap().len(),
                8
            );
        }
    }
    assert!(r.get("E", Some(0), 3).is_err());
    assert!(r.get("selected", Some(2), 5).is_err());
    assert_eq!(r.states.len(), 257);
    r.revalidate().unwrap();
    fs::write(f.dir.join("data.bin"), [1u8; 4]).unwrap();
    assert!(r.revalidate().is_err());
}

#[test]
fn missing_and_corrupted_complete_are_rejected() {
    let f = Fixture::new();
    fs::remove_file(f.dir.join("complete.json")).unwrap();
    assert!(f.load().is_err());
    f.json("complete.json", &json!({"complete": false}));
    assert!(f.load().is_err());
    fs::write(f.dir.join("complete.json"), b"{broken").unwrap();
    assert!(f.load().is_err());
    f.manifest();
    let mut c = f.read("complete.json");
    c["regions"] = json!(1);
    f.json("complete.json", &c);
    assert!(f.load().is_err());
}

#[test]
fn truncated_wronghash_missing_and_misaligned_records_are_rejected() {
    for kind in [
        "truncate",
        "manifest_hash",
        "record_hash",
        "shape",
        "row",
        "missing",
        "alignment",
        "producer",
        "states",
    ] {
        let f = Fixture::new();
        match kind {
            "truncate" => {
                let file = OpenOptions::new()
                    .write(true)
                    .open(f.dir.join("data.bin"))
                    .unwrap();
                file.set_len(4).unwrap();
            }
            "manifest_hash" => {
                let mut c = f.read("complete.json");
                c["files"][0]["sha256"] = json!(sha(b"wrong"));
                f.json("complete.json", &c);
            }
            "states" => {
                let mut s = f.lines("states.jsonl");
                s[1]["regions"][0]["shape"] = json!([999]);
                f.jsonl("states.jsonl", &s);
                f.manifest();
            }
            _ => {
                let mut r = f.lines("regions.jsonl");
                match kind {
                    "record_hash" => r[0]["sha256"] = json!(sha(b"wrong")),
                    "shape" => r[0]["shape"] = json!([8]),
                    "row" => r[0]["absroworigin"] = json!(3),
                    "missing" => {
                        r.remove(0);
                    }
                    "alignment" => r[0]["byte_offset"] = json!(1),
                    "producer" => r[0]["producer"] = json!(2),
                    _ => unreachable!(),
                }
                f.jsonl("regions.jsonl", &r);
                f.manifest();
            }
        }
        assert!(f.load().is_err(), "accepted {kind}");
    }
}

#[test]
fn raw_input_sha_ids_shape_and_arm_cannot_be_substituted() {
    let f = Fixture::new();
    assert!(Reference::load(&f.dir, f.g.clone(), b"{}", "native-token-major-f32").is_err());
    assert!(
        Reference::load(
            &f.dir,
            f.g.clone(),
            &f.input,
            "production-full-prefill-2048"
        )
        .is_err()
    );
    let mut g = f.g.clone();
    g.hidden += 1;
    assert!(Reference::load(&f.dir, g, &f.input, "native-token-major-f32").is_err());
    let mut h = f.read("header.json");
    h["input_ids_sha256"] = json!(sha(b"wrong"));
    f.json("header.json", &h);
    f.manifest();
    assert!(f.load().is_err());
}
