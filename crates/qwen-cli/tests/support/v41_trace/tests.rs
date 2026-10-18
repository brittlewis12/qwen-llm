use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "qwen-v41-trace-{}-{nonce}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn create(path: &Path) -> Result<Trace> {
    Trace::create_inner(
        path,
        Shapes {
            hidden: 2,
            vocab: 3,
            layers: 40,
        },
        json!({"prompt_token_ids":[0]}),
    )
}

#[test]
fn converted_capture_has_one_production_mode_and_characterization_scope() {
    let temp = Temp::new();
    let root = temp.0.join("production-mode");
    let config = Config {
        hidden: 2,
        vocab: 3,
        ..Config::default()
    };
    let trace = Trace::create(&root, &config, 1, &[0]).unwrap();
    let manifest = status(&root);
    assert_eq!(manifest["execution"], "gpu_step");
    assert_eq!(
        manifest["comparison_scope"],
        "hf_conversion_and_execution_characterization"
    );
    assert_eq!(manifest["status"], "incomplete");
    drop(trace);
}

#[test]
fn execution_capture_is_distinct_from_hf_characterization() {
    let temp = Temp::new();
    let root = temp.0.join("execution");
    let config = Config {
        hidden: 2,
        vocab: 3,
        ..Config::default()
    };
    let trace = Trace::create_execution(&root, &config, 1, &[0]).unwrap();
    let manifest = status(&root);
    assert_eq!(
        manifest["comparison_scope"],
        "independent_gguf_execution_only"
    );
    assert_eq!(manifest["execution"], "gpu_step");
    assert_eq!(manifest["context"], 1);
    assert_eq!(manifest["status"], "incomplete");
    drop(trace);
}

fn status(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path.join("run.json")).unwrap()).unwrap()
}

#[test]
fn v41_trace_portable_bytes_all_layers_positions_and_success_manifest() {
    let temp = Temp::new();
    let root = temp.0.join("fresh");
    let mut trace = create(&root).unwrap();
    assert_eq!(status(&root)["status"], "incomplete");
    let hidden = [1., -0., -2.5, 0., 4., 5., 6., 7.];
    let pre = [1., 0., 0., 0.];
    for (position, token) in [0, 2].into_iter().enumerate() {
        trace.begin_step(position, token).unwrap();
        for layer in 0..40 {
            trace
                .layer(LayerTrace {
                    position,
                    layer,
                    hidden: &hidden,
                    next_pre_mix: &pre,
                })
                .unwrap();
        }
        trace.finish_step(&[0., -1., 2.]).unwrap();
        assert_eq!(status(&root)["status"], "incomplete");
        let bytes = fs::read(root.join(format!("pos-{position:05}-layer-00.hidden.f32"))).unwrap();
        assert_eq!(&bytes[..12], &[0, 0, 128, 63, 0, 0, 0, 128, 0, 0, 32, 192]);
        for layer in 0..40 {
            assert_eq!(
                fs::metadata(root.join(format!("pos-{position:05}-layer-{layer:02}.hidden.f32")))
                    .unwrap()
                    .len(),
                32
            );
            assert_eq!(
                fs::metadata(root.join(format!("pos-{position:05}-layer-{layer:02}.pre.f32")))
                    .unwrap()
                    .len(),
                16
            );
        }
        assert_eq!(
            fs::read(root.join(format!("pos-{position:05}-logits.f32"))).unwrap(),
            [0, 0, 0, 0, 0, 0, 128, 191, 0, 0, 0, 64]
        );
    }
    trace.finish(&[2, 1], Some(1)).unwrap();
    let manifest = status(&root);
    assert_eq!(manifest["status"], "success");
    assert_eq!(manifest["mode"], "NativeF32");
    assert_eq!(
        manifest["router_tie_policy"],
        "lower_expert_id_then_id_sorted"
    );
    assert_eq!(manifest["byte_order"], "little");
    assert_eq!(
        manifest["shapes"],
        json!({"hidden":[4,2],"next_pre_mix":[4],"logits":[3]})
    );
    assert_eq!(
        manifest["bytes_per_file"],
        json!({"hidden":32,"pre":16,"logits":12})
    );
    assert_eq!(
        manifest["positions"],
        json!([{"position":0,"token_id":0},{"position":1,"token_id":2}])
    );
    assert_eq!(manifest["generated_token_ids"], json!([2, 1]));
    assert_eq!(manifest["artifact_format"], "gguf");
    assert_eq!(manifest["provenance_verified"], false);
    assert_eq!(fs::read_dir(&root).unwrap().count(), 163);
    assert!(!root.join("run.json.tmp").exists());
}

#[test]
fn v41_trace_rejects_nonempty_files_symlinks_and_missing_parents() {
    use std::os::unix::fs::symlink;
    let temp = Temp::new();
    let file = temp.0.join("existing-file");
    fs::write(&file, "preserve").unwrap();
    assert!(create(&file).is_err());
    assert_eq!(fs::read_to_string(&file).unwrap(), "preserve");
    assert!(create(&temp.0).is_err());
    assert!(!temp.0.join("run.json").exists());
    let empty = temp.0.join("empty");
    fs::create_dir(&empty).unwrap();
    let link = temp.0.join("link");
    symlink(&empty, &link).unwrap();
    assert!(create(&link).is_err());
    assert_eq!(fs::read_dir(&empty).unwrap().count(), 0);
    let dangling = temp.0.join("dangling");
    symlink(temp.0.join("absent"), &dangling).unwrap();
    assert!(create(&dangling).is_err());
    assert!(create(&temp.0.join("absent/trace")).is_err());
    assert!(create(Path::new("")).is_err());
    let trace = create(&empty).unwrap();
    drop(trace);
    assert_eq!(status(&empty)["status"], "incomplete");
    assert!(create(&empty).is_err());
}

#[test]
fn v41_trace_rejects_wrong_shapes_missing_layers_and_file_collisions() {
    let temp = Temp::new();
    let root = temp.0.join("trace");
    let mut trace = create(&root).unwrap();
    assert!(trace.begin_step(1, 0).is_err());
    assert!(trace.begin_step(0, 3).is_err());
    trace.begin_step(0, 0).unwrap();
    assert!(trace.begin_step(0, 0).is_err());
    assert!(
        trace
            .layer(LayerTrace {
                position: 0,
                layer: 1,
                hidden: &[0.; 8],
                next_pre_mix: &[0.; 4]
            })
            .is_err()
    );
    assert!(
        trace
            .layer(LayerTrace {
                position: 1,
                layer: 0,
                hidden: &[0.; 8],
                next_pre_mix: &[0.; 4]
            })
            .is_err()
    );
    assert!(
        trace
            .layer(LayerTrace {
                position: 0,
                layer: 0,
                hidden: &[0.; 7],
                next_pre_mix: &[0.; 4]
            })
            .is_err()
    );
    assert!(
        trace
            .layer(LayerTrace {
                position: 0,
                layer: 0,
                hidden: &[0.; 8],
                next_pre_mix: &[0.; 3]
            })
            .is_err()
    );
    assert!(trace.finish_step(&[0.; 3]).is_err());
    fs::write(root.join("pos-00000-layer-00.pre.f32"), b"preserve").unwrap();
    assert!(
        trace
            .layer(LayerTrace {
                position: 0,
                layer: 0,
                hidden: &[0.; 8],
                next_pre_mix: &[0.; 4]
            })
            .is_err()
    );
    assert_eq!(
        fs::read(root.join("pos-00000-layer-00.pre.f32")).unwrap(),
        b"preserve"
    );
    assert!(trace.finish(&[1], Some(1)).is_err());
    assert_eq!(status(&root)["status"], "incomplete");
    assert!(!root.join("pos-00000-logits.f32").exists());
}

#[test]
fn v41_trace_checks_chunked_byte_counts_and_does_not_overwrite() {
    let temp = Temp::new();
    let path = temp.0.join("chunked.f32");
    let values: Vec<_> = (0..2049).map(|n| -(n as f32)).collect();
    write_f32(&path, &values).unwrap();
    let bytes = fs::read(&path).unwrap();
    assert_eq!(bytes.len(), 2049 * 4);
    for (chunk, expected) in bytes.chunks_exact(4).zip(values) {
        assert_eq!(
            f32::from_le_bytes(chunk.try_into().unwrap()).to_bits(),
            expected.to_bits()
        );
    }
    assert!(write_f32(&path, &[1.]).is_err());
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert!(byte_count(usize::MAX).is_err());
}

#[test]
fn v41_trace_wrong_logits_and_failed_manifest_leave_incomplete_status() {
    let temp = Temp::new();
    let root = temp.0.join("trace");
    let mut trace = create(&root).unwrap();
    trace.begin_step(0, 0).unwrap();
    for layer in 0..40 {
        trace
            .layer(LayerTrace {
                position: 0,
                layer,
                hidden: &[0.; 8],
                next_pre_mix: &[0.; 4],
            })
            .unwrap();
    }
    assert!(trace.finish_step(&[0.; 2]).is_err());
    assert!(!root.join("pos-00000-logits.f32").exists());
    assert_eq!(status(&root)["status"], "incomplete");
    trace.finish_step(&[0.; 3]).unwrap();
    fs::write(root.join("run.json.tmp"), "preserve").unwrap();
    assert!(trace.finish(&[2], None).is_err());
    assert_eq!(
        fs::read_to_string(root.join("run.json.tmp")).unwrap(),
        "preserve"
    );
    assert_eq!(status(&root)["status"], "incomplete");
}
