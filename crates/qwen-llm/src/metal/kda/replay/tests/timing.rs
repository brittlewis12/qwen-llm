//! Bounded synthetic leaf timing. No model, admission gate or runtime knobs.
//! Run the ignored test in a release test binary with --test-threads=1.
//! Optional GLM53_STAGE_OUT is a new JSON file, relative to workspace root.
use super::*;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::io::Write;

const HEADS: usize = 64;
const BLOCKS: usize = 34;
const STEPS: usize = 16; // two complete L8 windows, including every residue/flush

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn tensor_hash(t: &MetalTensor) -> String {
    hash(bytemuck::cast_slice(&tensor_f32_at_offset(t)))
}

#[test]
#[ignore = "research-only bounded KDA FP32 replay ABBA; release, idle GPU, no MTL_DEBUG_LAYER; optional new GLM53_STAGE_OUT JSON; no performance threshold"]
fn replay_f32_timing_abba_64_heads_34_states() {
    assert!(!cfg!(debug_assertions), "release timing only");
    assert!(std::env::var_os("MTL_DEBUG_LAYER").is_none());
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("GPU benchmark lease");
    let ctx = MetalContext::new().expect("Metal context");
    let mut output = std::env::var_os("GLM53_STAGE_OUT").map(|p| {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(p);
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("new output file")
    });
    let initial: Vec<_> = (0..BLOCKS)
        .map(|i| State::new(&ctx, HEADS, i, false))
        .collect();
    let work: Vec<_> = (0..BLOCKS)
        .map(|i| State::new(&ctx, HEADS, i, false))
        .collect();
    let records: Vec<_> = (0..BLOCKS).map(|_| Records::new(&ctx, HEADS, 8)).collect();
    let inputs: Vec<_> = (0..STEPS)
        .map(|i| Inputs::new(&ctx, HEADS, i, false))
        .collect();
    let width = HEADS * 128;
    // The recurrent out binding writes each actual block/token output directly;
    // no diagnostic copy dispatch is inserted in either timed arm.
    let capture = tensor(
        &ctx,
        &vec![f32::NAN; STEPS * BLOCKS * width],
        vec![width as u64, BLOCKS as u64, STEPS as u64],
    );
    let views: Vec<_> = (0..STEPS * BLOCKS)
        .map(|i| capture.view_subrange((i * width) as u64, vec![width as u64]))
        .collect();
    let reset = || {
        run(&ctx, |enc| {
            for (a, b) in initial.iter().zip(&work) {
                for (src, dst) in [(&a.h, &b.h), (&a.conv, &b.conv)] {
                    encode_copy_offset_f32(&ctx, enc, src, 0, dst, src.n_elements() as usize)
                        .unwrap();
                }
            }
            for r in &records {
                for t in &r.t {
                    encode_fill_f32(&ctx, enc, t, f32::NAN).unwrap();
                }
            }
            encode_fill_f32(&ctx, enc, &capture, f32::NAN).unwrap();
        })
    };
    // Precompile outside timing; ordinary and replay share the same context.
    for name in ["kernel_glm53_kda_decode", REPLAY_KERNEL, MATERIALIZE_KERNEL] {
        ctx.pipeline(name).unwrap();
    }
    let sample = |label: &str, replay: bool, warm: bool| {
        reset(); // identical reset command/commit/wait for both arms; untimed
        let start = std::time::Instant::now();
        let mut gpu_ms = Vec::new();
        let mut wall_ms = Vec::new();
        for step in 0..STEPS {
            let wall = std::time::Instant::now();
            let command = ctx.queue.commandBuffer().expect("command buffer");
            let enc = KernelEncoder::begin(&command);
            for block in 0..BLOCKS {
                let mut b = inputs[step].bind(&work[block]);
                b.out = &views[step * BLOCKS + block];
                // Chain real data dependencies through 34 distinct matrices,
                // rotating to the same layer only on the next token.
                if block > 0 {
                    b.q = &views[step * BLOCKS + block - 1];
                }
                if replay {
                    encode_kda_replay_f32(
                        &ctx,
                        &enc,
                        HEADS,
                        ReplayPosition {
                            window: 8,
                            residue: step % 8,
                        },
                        &b,
                        &records[block].bind(),
                        LOWER,
                        EPS,
                    )
                    .unwrap();
                } else {
                    encode_kda_decode(&ctx, &enc, HEADS, &b, LOWER, EPS).unwrap();
                }
            }
            enc.end();
            command.commit();
            wait_completed(&command).expect("timed command");
            wall_ms.push(wall.elapsed().as_secs_f64() * 1e3);
            gpu_ms.push((command.GPUEndTime() - command.GPUStartTime()) * 1e3);
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        // All readback/hashing is outside the timed region. Two full windows
        // end at residue zero, so checkpoints are also logical final states.
        let payload = tensor_f32_at_offset(&capture);
        let mut state_digest = Sha256::new();
        let mut conv_digest = Sha256::new();
        for s in &work {
            state_digest.update(bytemuck::cast_slice(&tensor_f32_at_offset(&s.h)));
            conv_digest.update(bytemuck::cast_slice(&tensor_f32_at_offset(&s.conv)));
        }
        let row = json!({"label":label,"warmup":warm,"arm":if replay {"replay_l8"} else {"ordinary"},
            "timed_commits":STEPS,"timed_waits":STEPS,"reset_commits":1,"reset_waits":1,
            "dispatches":STEPS*BLOCKS,"gpu_ms_by_residue":gpu_ms,"wall_ms_by_residue":wall_ms,
            "residues":(0..STEPS).map(|s|s%8).collect::<Vec<_>>(),"flush_steps":[7,15],
            "wall_total_ms":elapsed_ms,"gpu_total_ms":gpu_ms.iter().sum::<f64>(),
            "output_sha256_f32_le":hash(bytemuck::cast_slice(&payload)),
            "output_nonfinite":payload.iter().filter(|v|!v.is_finite()).count(),
            "final_state_sha256_f32_le":format!("{:x}",state_digest.finalize()),
            "final_conv_sha256_f32_le":format!("{:x}",conv_digest.finalize())});
        eprintln!(
            "KDA replay {label}: wall {elapsed_ms:.3} ms GPU {:.3} ms",
            gpu_ms.iter().sum::<f64>()
        );
        row
    };
    let samples = vec![
        sample("warm_A", false, true),
        sample("warm_B", true, true),
        sample("A1", false, false),
        sample("B1", true, false),
        sample("B2", true, false),
        sample("A2", false, false),
    ];
    let equality = json!({
        "output_hashes_match_A1": samples.iter().all(|s| s["output_sha256_f32_le"] == samples[2]["output_sha256_f32_le"]),
        "state_hashes_match_A1": samples.iter().all(|s| s["final_state_sha256_f32_le"] == samples[2]["final_state_sha256_f32_le"]),
        "conv_hashes_match_A1": samples.iter().all(|s| s["final_conv_sha256_f32_le"] == samples[2]["final_conv_sha256_f32_le"]),
    });
    let revision = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned());
    let mut input_digest = Sha256::new();
    for i in &inputs {
        for t in &i.t {
            input_digest.update(bytemuck::cast_slice(&tensor_f32_at_offset(t)));
        }
    }
    let initial_hashes: Vec<_> = initial
        .iter()
        .map(|s| json!({"checkpoint":tensor_hash(&s.h),"conv":tensor_hash(&s.conv)}))
        .collect();
    let state_bytes = BLOCKS * HEADS * 128 * 128 * 4;
    let record_bytes = BLOCKS * HEADS * 8 * 3 * 128 * 4;
    let ordinary_per_step = 2 * state_bytes;
    let replay_traffic: Vec<_> = (0..8)
        .map(|r| {
            json!({"residue":r,"flush":r==7,
        "checkpoint_read_bytes":state_bytes,"checkpoint_write_bytes":if r==7 {state_bytes} else {0},
        "record_read_bytes":BLOCKS*HEADS*r*3*128*4,
        "record_write_bytes":if r==7 {0} else {BLOCKS*HEADS*3*128*4}})
        })
        .collect();
    let report = json!({"experiment":"bounded_fp32_kda_replay_leaf_v1","qualification_gate":false,
        "precision":"FP32 checkpoint/records/operands; bitwise agreement is an owner-executed hypothesis",
        "heads":HEADS,"head_dim":128,"states":BLOCKS,"steps_per_sample":STEPS,"window":8,
        "schedule":"warm A/B then ABBA, reset from identical initial buffers before each sample",
        "chain":"q of block n comes from actual output of block n-1; distinct recurrent/conv state per block",
        "checkpoint_bytes_per_arm":state_bytes,"checkpoint_mib_per_arm":state_bytes/(1<<20),
        "replay_added_record_bytes":record_bytes,"conv_bytes_per_arm":BLOCKS*HEADS*128*9*4,
        "capture_bytes":capture.n_bytes(),"test_resident_checkpoint_copies":2,
        "memory_note":"FP32 replay adds records; it does not reduce the 136 MiB checkpoint for 34 blocks",
        "ordinary_state_read_write_bytes_per_token":ordinary_per_step,
        "replay_recurrent_logical_traffic_by_residue":replay_traffic,
        "traffic_note":"logical recurrent bytes, excluding conv/weights/activations/output and hardware caches",
        "threadgroup_bytes_decode":14912,"threadgroup_bytes_materialize":12288,
        "shader_sha256":hash(SOURCE.as_bytes()),"ordinary_shader_sha256":hash(ORDINARY.as_bytes()),
        "host_sha256":hash(include_str!("../../replay.rs").as_bytes()),
        "timing_source_sha256":hash(include_str!("timing.rs").as_bytes()),
        "input_sha256_f32_le":format!("{:x}",input_digest.finalize()),"initial_hashes":initial_hashes,
        "device":ctx.device_facts(),"revision":revision,"hash_observations":equality,"samples":samples});
    if let Some(f) = output.as_mut() {
        serde_json::to_writer_pretty(&mut *f, &report).unwrap();
        f.write_all(b"\n").unwrap();
        f.sync_all().unwrap();
    } else {
        eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
    }
}
