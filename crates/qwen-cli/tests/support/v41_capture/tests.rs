use super::*;
use qwen_llm::deepseek_v41::forward::capture::Checkpoint;

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "qwen-v41-capture-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn manifest(p: &Path) -> Value {
    serde_json::from_slice(&fs::read(p.join("manifest.json")).unwrap()).unwrap()
}

#[test]
fn capture_npy_histories_sparse_checkpoints_and_collision() {
    let temp = Temp::new();
    let root = temp.0.join("capture");
    let mut writer = Writer::create(&root, &[1, 2], json!({"native_revision":"test"})).unwrap();
    assert!(Writer::create(&root, &[1], json!({})).is_err());
    let input = vec![1.25; 5120];
    let output = vec![-0.5; 5120];
    for p in 0..3 {
        writer.begin_step(p, p as u32).unwrap();
        for layer in LAYERS {
            let ids = (0..(p + 1) / if layer < 20 { 2 } else { 1 })
                .map(|id| id as i32)
                .collect::<Vec<_>>();
            writer
                .layer(AttentionCapture {
                    position: p,
                    layer,
                    input: &input,
                    checkpoint: (p > 0).then_some(Checkpoint {
                        output: &output,
                        kv_owner: if layer < 20 { layer } else { 20 },
                        index_source: layer,
                        compressed_ids: &ids,
                        candidate_mask: (layer >= 20).then_some(&[1][..]),
                    }),
                })
                .unwrap();
        }
    }
    writer.finish().unwrap();
    let m = manifest(&root);
    assert_eq!(m["status"], "success");
    assert_eq!(m["token_ids"], json!([0, 1, 2]));
    assert_eq!(m["inputs"].as_object().unwrap().len(), 8);
    assert_eq!(m["checkpoints"].as_array().unwrap().len(), 2);
    assert_eq!(fs::read_dir(&root).unwrap().count(), 1 + 8 + 16);
    for layer in LAYERS {
        let data = fs::read(root.join(format!("input-{layer:02}.npy"))).unwrap();
        assert_eq!(&data[..8], b"\x93NUMPY\x01\x00");
        assert_eq!(u16::from_le_bytes([data[8], data[9]]) as usize, HEADER - 10);
        assert!(
            std::str::from_utf8(&data[10..HEADER])
                .unwrap()
                .contains("'shape': (3, 5120,)")
        );
        assert_eq!(data.len(), HEADER + 3 * 5120 * 4);
        assert!(
            data[HEADER..]
                .chunks_exact(4)
                .all(|b| b == 1.25f32.to_le_bytes())
        );
        let data = fs::read(root.join(format!("output-00002-{layer:02}.npy"))).unwrap();
        assert!(
            std::str::from_utf8(&data[10..HEADER])
                .unwrap()
                .contains("'shape': (5120,)")
        );
        assert!(
            data[HEADER..]
                .chunks_exact(4)
                .all(|b| b == (-0.5f32).to_le_bytes())
        );
        assert_eq!(
            m["checkpoints"][1]["layers"][layer.to_string()]["compressed_ids"],
            if layer < 20 {
                json!([0])
            } else {
                json!([0, 1, 2])
            }
        );
    }
}

#[test]
fn capture_partial_failure_and_cancellation_remain_incomplete() {
    let temp = Temp::new();
    let root = temp.0.join("capture");
    {
        let mut writer = Writer::create_inner(&root, &[1], 2, json!({})).unwrap();
        assert!(writer.begin_step(1, 0).is_err());
        writer.begin_step(0, 0).unwrap();
        writer
            .layer(AttentionCapture {
                position: 0,
                layer: 2,
                input: &[1., 2.],
                checkpoint: None,
            })
            .unwrap();
        assert!(
            writer
                .layer(AttentionCapture {
                    position: 0,
                    layer: 8,
                    input: &[f32::NAN, 2.],
                    checkpoint: None
                })
                .is_err()
        );
        writer.failure("cancelled", true);
    }
    let m = manifest(&root);
    assert_eq!(m["status"], "incomplete");
    assert_eq!(m["cancelled"], true);
    assert_eq!(m["reason"], "cancelled");
    assert_eq!(m["inputs"]["2"]["rows"], 1);
    assert_eq!(m["inputs"]["8"]["rows"], 0);
    let root = temp.0.join("early-stop");
    assert!(
        Writer::create_inner(&root, &[1], 2, json!({}))
            .unwrap()
            .finish()
            .is_err()
    );
    assert_eq!(manifest(&root)["status"], "incomplete");
    assert!(Writer::create_inner(&temp.0.join("bad"), &[2, 1], 2, json!({})).is_err());
    assert!(Writer::create_inner(&temp.0.join("zero"), &[0], 2, json!({})).is_err());
    let link = temp.0.join("link");
    std::os::unix::fs::symlink(&root, &link).unwrap();
    assert!(Writer::create_inner(&link, &[1], 2, json!({})).is_err());
}

#[test]
fn capture_library_records_explicit_checkpoints() {
    let temp = Temp::new();
    let root = temp.0.join("explicit-checkpoints");
    let writer =
        Writer::create_inner(&root, &[1, 2], 2, json!({"artifact_format":"gguf"})).unwrap();
    assert!(writer.is_checkpoint(1));
    assert!(writer.is_checkpoint(2));
    assert!(!writer.is_checkpoint(0));
    assert!(!writer.wants_position(3));
    assert_eq!(manifest(&root)["requested_checkpoints"], json!([1, 2]));
    drop(writer);
    assert_eq!(manifest(&root)["status"], "incomplete");
}
