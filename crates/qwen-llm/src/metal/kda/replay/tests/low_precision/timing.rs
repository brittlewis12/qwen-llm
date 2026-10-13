//! Matched bounded leaf timing; all eight residues including repair/flush.
//! No product gate, tile/window sweep, or output-bit equality claim for LP.
use super::*;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
const HEADS: usize = 64;
const BLOCKS: usize = 34;
const STEPS: usize = 16;
const RELEASE_LOWER: f32 = -5.0;
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn binary_hash() -> String {
    let mut f = std::fs::File::open(std::env::current_exe().unwrap()).unwrap();
    let mut digest = Sha256::new();
    let mut buf = [0; 65536];
    loop {
        let n = f.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
    }
    format!("{:x}", digest.finalize())
}
#[derive(Clone, Copy, Debug)]
enum Arm {
    Ordinary,
    Fp32,
    Separate,
    Shared,
}
impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Ordinary => "ordinary_f32",
            Self::Fp32 => "replay_f32",
            Self::Separate => "separate_f32_anchor_f16_read",
            Self::Shared => "shared_f16_anchor_read",
        }
    }
    fn lp(self) -> Option<usize> {
        match self {
            Self::Separate => Some(0),
            Self::Shared => Some(1),
            _ => None,
        }
    }
}
#[test]
#[ignore = "release-only physical F16 KDA leaf: warm then ABBA for FP32 control and each LP policy, 64 heads/34 states/full L8 windows; optional new GLM53_STAGE_OUT; no threshold"]
fn replay_f16_timing_abba_64_heads_34_states() {
    assert!(!cfg!(debug_assertions), "release timing only");
    assert!(std::env::var_os("MTL_DEBUG_LAYER").is_none());
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let mut output = std::env::var_os("GLM53_STAGE_OUT").map(|p| {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join(p),
            )
            .expect("new output file")
    });
    let initial: Vec<_> = (0..BLOCKS)
        .map(|i| State::new(&ctx, HEADS, i, false))
        .collect();
    let work: Vec<_> = (0..BLOCKS)
        .map(|i| State::new(&ctx, HEADS, i, false))
        .collect();
    let records: Vec<_> = (0..BLOCKS).map(|_| Records::new(&ctx, HEADS, 8)).collect();
    let lp: Vec<Vec<_>> = MODES
        .into_iter()
        .map(|m| {
            (0..BLOCKS)
                .map(|i| LpState::new(&ctx, HEADS, i, false, m))
                .collect()
        })
        .collect();
    let initial_half: Vec<_> = lp[0].iter().map(|s| payload(&s.read)).collect();
    let initial_h: Vec<_> = initial.iter().map(|s| payload(&s.h)).collect();
    let initial_conv: Vec<_> = initial.iter().map(|s| payload(&s.conv)).collect();
    let inputs: Vec<_> = (0..STEPS)
        .map(|s| Inputs::new(&ctx, HEADS, s, false))
        .collect();
    let width = HEADS * 128;
    let capture = tensor(
        &ctx,
        &vec![f32::NAN; STEPS * BLOCKS * width],
        vec![width as u64, BLOCKS as u64, STEPS as u64],
    );
    let views: Vec<_> = (0..STEPS * BLOCKS)
        .map(|j| capture.view_subrange((j * width) as u64, vec![width as u64]))
        .collect();
    for name in [
        "kernel_glm53_kda_decode",
        REPLAY_KERNEL,
        MODES[0].kernel(),
        MODES[1].kernel(),
    ] {
        ctx.pipeline(name).unwrap();
    }
    let sample = |label: &str, arm: Arm, warm: bool| {
        // Identical CPU reset protocol after completed commands; no reset GPU
        // dispatch/commit in any arm. Reset bytes are outside timing.
        for block in 0..BLOCKS {
            if let Some(a) = arm.lp() {
                let s = &lp[a][block];
                write_payload(
                    &s.s.h,
                    if a == 0 {
                        &initial_h[block]
                    } else {
                        &initial_half[block]
                    },
                );
                if a == 0 {
                    write_payload(&s.read, &initial_half[block]);
                }
                write_payload(&s.s.conv, &initial_conv[block]);
                for t in &s.r.t {
                    write_payload(
                        t,
                        bytemuck::cast_slice(&vec![f32::NAN; t.n_elements() as usize]),
                    );
                }
            } else {
                write_payload(&work[block].h, &initial_h[block]);
                write_payload(&work[block].conv, &initial_conv[block]);
                for t in &records[block].t {
                    write_payload(
                        t,
                        bytemuck::cast_slice(&vec![f32::NAN; t.n_elements() as usize]),
                    );
                }
            }
        }
        write_payload(
            &capture,
            bytemuck::cast_slice(&vec![f32::NAN; capture.n_elements() as usize]),
        );
        let start = std::time::Instant::now();
        let mut gpu_ms = Vec::new();
        let mut wall_ms = Vec::new();
        for step in 0..STEPS {
            let wall = std::time::Instant::now();
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            for block in 0..BLOCKS {
                let state = if let Some(a) = arm.lp() {
                    &lp[a][block].s
                } else {
                    &work[block]
                };
                let mut b = inputs[step].bind(state);
                b.out = &views[step * BLOCKS + block];
                if block > 0 {
                    b.q = &views[step * BLOCKS + block - 1];
                }
                match arm {
                    Arm::Ordinary => {
                        encode_kda_decode(&ctx, &enc, HEADS, &b, RELEASE_LOWER, EPS).unwrap()
                    }
                    Arm::Fp32 => encode_kda_replay_f32(
                        &ctx,
                        &enc,
                        HEADS,
                        ReplayPosition {
                            window: 8,
                            residue: step % 8,
                        },
                        &b,
                        &records[block].bind(),
                        RELEASE_LOWER,
                        EPS,
                    )
                    .unwrap(),
                    _ => lp[arm.lp().unwrap()][block].encode(
                        &ctx,
                        &enc,
                        HEADS,
                        step,
                        &b,
                        RELEASE_LOWER,
                    ),
                }
            }
            enc.end();
            command.commit();
            wait_completed(&command).unwrap();
            let wall_time = wall.elapsed().as_secs_f64() * 1e3;
            let begin = command.GPUStartTime();
            let end = command.GPUEndTime();
            assert!(
                begin.is_finite() && end.is_finite() && begin > 0. && end > begin,
                "invalid GPU timestamps: {begin}/{end}"
            );
            assert!(wall_time.is_finite() && wall_time > 0.);
            wall_ms.push(wall_time);
            gpu_ms.push((end - begin) * 1e3);
        }
        let wall_total = start.elapsed().as_secs_f64() * 1e3;
        assert!(wall_total.is_finite() && wall_total > 0.);
        let mut anchor = Sha256::new();
        let mut read = Sha256::new();
        let mut conv = Sha256::new();
        let mut kv_conv = Sha256::new();
        let mut record = Sha256::new();
        let mut anchor_nonfinite = 0;
        let mut read_nonfinite = 0;
        for block in 0..BLOCKS {
            let (s, physical) = if let Some(a) = arm.lp() {
                (&lp[a][block].s, &lp[a][block].read)
            } else {
                (&work[block], &work[block].h)
            };
            anchor.update(payload(&s.h));
            read.update(payload(physical));
            conv.update(payload(&s.conv));
            // q convolution can differ because actual outputs feed the next
            // layer. k/v convolution and recurrent state do not depend on q.
            let c = payload(&s.conv);
            kv_conv.update(&c[3 * width * 4..]);
            let av = if s.h.dtype == GgmlType::F16 {
                read_half(&s.h)
            } else {
                tensor_f32_at_offset(&s.h)
            };
            let rv = if physical.dtype == GgmlType::F16 {
                read_half(physical)
            } else {
                tensor_f32_at_offset(physical)
            };
            anchor_nonfinite += av.iter().filter(|x| !x.is_finite()).count();
            read_nonfinite += rv.iter().filter(|x| !x.is_finite()).count();
            if let Some(a) = arm.lp() {
                for t in &lp[a][block].r.t {
                    record.update(payload(t));
                }
            } else if matches!(arm, Arm::Fp32) {
                for t in &records[block].t {
                    record.update(payload(t));
                }
            }
        }
        let actual = tensor_f32_at_offset(&capture);
        let row = json!({"label":label,"arm":arm.name(),"warmup":warm,"timed_commits":STEPS,"timed_waits":STEPS,"dispatches":STEPS*BLOCKS,"reset_commits":0,"reset_waits":0,
            "gpu_ms_by_step":gpu_ms,"wall_ms_by_step":wall_ms,"gpu_total_ms":gpu_ms.iter().sum::<f64>(),"wall_total_ms":wall_total,
            "residues":(0..STEPS).map(|s|s%8).collect::<Vec<_>>(),"flush_steps":[7,15],
            "anchor_sha256":format!("{:x}",anchor.finalize()),"read_sha256":format!("{:x}",read.finalize()),"conv_sha256":format!("{:x}",conv.finalize()),"kv_conv_sha256":format!("{:x}",kv_conv.finalize()),"records_sha256":format!("{:x}",record.finalize()),
            "output_sha256":hash(&payload(&capture)),"output_nonfinite":actual.iter().filter(|x|!x.is_finite()).count(),"anchor_nonfinite":anchor_nonfinite,"read_nonfinite":read_nonfinite});
        eprintln!(
            "{label} {}: GPU {:.3} ms wall {wall_total:.3} ms",
            arm.name(),
            gpu_ms.iter().sum::<f64>()
        );
        row
    };
    let mut pairs = Vec::new();
    for candidate in [Arm::Fp32, Arm::Separate, Arm::Shared] {
        let samples = vec![
            sample("warm_A", Arm::Ordinary, true),
            sample("warm_B", candidate, true),
            sample("A1", Arm::Ordinary, false),
            sample("B1", candidate, false),
            sample("B2", candidate, false),
            sample("A2", Arm::Ordinary, false),
        ];
        let observations = json!({"A_repeat_anchor_exact":samples[2]["anchor_sha256"]==samples[5]["anchor_sha256"],"B_repeat_anchor_exact":samples[3]["anchor_sha256"]==samples[4]["anchor_sha256"],
            "B_repeat_output_exact":samples[3]["output_sha256"]==samples[4]["output_sha256"],"B_anchor_equals_A":samples[3]["anchor_sha256"]==samples[2]["anchor_sha256"],"B_kv_conv_equals_A":samples[3]["kv_conv_sha256"]==samples[2]["kv_conv_sha256"],
            "B_output_equals_A":samples[3]["output_sha256"]==samples[2]["output_sha256"],"anchor_equality_expected":!matches!(candidate,Arm::Shared),"output_equality_expected":matches!(candidate,Arm::Fp32)});
        let ratio = |field: &str| {
            (samples[3][field].as_f64().unwrap() + samples[4][field].as_f64().unwrap())
                / (samples[2][field].as_f64().unwrap() + samples[5][field].as_f64().unwrap())
        };
        pairs.push(json!({"candidate":candidate.name(),"observations":observations,
            "gpu_B_over_A":ratio("gpu_total_ms"),"wall_B_over_A":ratio("wall_total_ms"),"samples":samples}));
    }
    let matrices = BLOCKS * HEADS * 128 * 128;
    let f32_bytes = matrices * 4;
    let f16_bytes = matrices * 2;
    let mut input = Sha256::new();
    for i in &inputs {
        for t in &i.t {
            input.update(payload(t));
        }
    }
    let mut init = Sha256::new();
    for b in initial_h.iter().chain(initial_conv.iter()) {
        init.update(b);
    }
    let storage:Vec<_>=[Arm::Ordinary,Arm::Fp32,Arm::Separate,Arm::Shared].into_iter().map(|a| {
        let anchor=if matches!(a,Arm::Shared) {f16_bytes} else {f32_bytes};
        let extra_read=if matches!(a,Arm::Separate) {f16_bytes} else {0};
        let records=match a {Arm::Ordinary=>0,Arm::Fp32=>BLOCKS*HEADS*8*3*128*4,_=>BLOCKS*HEADS*8*(4*128+1)*4};
        let traffic:Vec<_>=(0..8).map(|r| {
            let flush=r==7;let lp=a.lp().is_some();
            json!({"residue":r,"matrix_read_bytes":if lp && !(flush && matches!(a,Arm::Separate)) {f16_bytes} else {f32_bytes},
                "matrix_write_bytes":if matches!(a,Arm::Ordinary) || flush {anchor+extra_read} else {0},
                "record_read_bytes":if matches!(a,Arm::Ordinary) {0} else {BLOCKS*HEADS*r*(3*128+usize::from(lp && flush))*4},
                "record_write_bytes":if matches!(a,Arm::Ordinary) || flush {0} else {BLOCKS*HEADS*(if lp {4*128+1} else {3*128})*4}})
        }).collect();
        json!({"arm":a.name(),"anchor_bytes":anchor,"additional_read_bytes":extra_read,"record_bytes":records,"total_recurrent_storage_bytes":anchor+extra_read+records,"logical_traffic_by_residue":traffic})
    }).collect();
    let report = json!({"experiment":"physical_f16_kda_leaf_l8","qualification_gate":false,"heads":HEADS,"states":BLOCKS,"steps":STEPS,"window":8,"lower_bound":RELEASE_LOWER,
        "schedule":"warm A/B then A/B/B/A separately for FP32 replay and each physical-F16 policy; identical reset, commit and wait protocol",
        "chain":"34 distinct matrices; actual output of block n-1 supplies q of n; k/v independent of q so separate F32 anchors should equal ordinary even when outputs/q conv differ",
        "comparison_note":"LP outputs are not expected bitexact; hashes and errors are observations, no performance/quality threshold",
        "storage":storage,"traffic_note":"logical recurrent matrix/record bytes only, excluding caches, conv and activations; shared read aliases anchor, separate retention adds memory",
        "conv_bytes_per_arm":BLOCKS*HEADS*128*9*4,"capture_bytes":capture.n_bytes(),"lp_threadgroup_bytes":LP_TGM_BYTES,
        "binary_sha256":binary_hash(),"shader_sha256":hash(SOURCE.as_bytes()),"ordinary_shader_sha256":hash(ORDINARY.as_bytes()),
        "host_sha256":hash(include_str!("../../lp.rs").as_bytes()),"fp32_host_sha256":hash(include_str!("../../../replay.rs").as_bytes()),
        "ordinary_host_sha256":hash(include_str!("../../../../kda.rs").as_bytes()),
        "base_fixture_sha256":hash(include_str!("../../tests.rs").as_bytes()),
        "fixture_sha256":hash(include_str!("../low_precision.rs").as_bytes()),"timing_source_sha256":hash(include_str!("timing.rs").as_bytes()),
        "input_sha256":format!("{:x}",input.finalize()),"initial_f32_state_conv_sha256":format!("{:x}",init.finalize()),"device":ctx.device_facts(),"pairs":pairs});
    if let Some(f) = output.as_mut() {
        serde_json::to_writer_pretty(&mut *f, &report).unwrap();
        f.write_all(b"\n").unwrap();
        f.sync_all().unwrap();
    } else {
        eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
    }
}
