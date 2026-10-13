//! Independent persistent-f64 recurrence: state and convolution history never
//! round through F32 between steps. This is not the old F32 contract oracle.
use super::*;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

struct Oracle {
    h: Vec<f64>,
    conv: Vec<f64>,
    heads: usize,
}
impl Oracle {
    fn new(s: &State, heads: usize) -> Self {
        Self {
            h: tensor_f32_at_offset(&s.h)
                .into_iter()
                .map(f64::from)
                .collect(),
            conv: tensor_f32_at_offset(&s.conv)
                .into_iter()
                .map(f64::from)
                .collect(),
            heads,
        }
    }
    fn step(&mut self, input: &Inputs, lower: f32) -> Vec<f64> {
        let t: Vec<Vec<f64>> = input
            .t
            .iter()
            .map(|t| tensor_f32_at_offset(t).into_iter().map(f64::from).collect())
            .collect();
        let width = self.heads * 128;
        let mut qkv = vec![vec![0.; width]; 3];
        for which in 0..3 {
            for c in 0..width {
                let mut a = 0.;
                for tap in 0..3 {
                    a = self.conv[(which * 3 + tap) * width + c]
                        .mul_add(t[6 + which][c * 4 + tap], a);
                }
                a = t[which][c].mul_add(t[6 + which][c * 4 + 3], a);
                for tap in 0..2 {
                    self.conv[(which * 3 + tap) * width + c] =
                        self.conv[(which * 3 + tap + 1) * width + c];
                }
                self.conv[(which * 3 + 2) * width + c] = t[which][c];
                qkv[which][c] = a / (1. + (-a).exp());
            }
        }
        let mut output = vec![0.; width];
        for head in 0..self.heads {
            let base = head * 128;
            let qscale = 1.
                / ((qkv[0][base..base + 128].iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt()
                    * 128f64.sqrt());
            let kscale =
                1. / (qkv[1][base..base + 128].iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
            let beta = 1. / (1. + (-t[4][head]).exp());
            let mut decay = [0.; 128];
            for key in 0..128 {
                qkv[0][base + key] *= qscale;
                qkv[1][base + key] *= kscale;
                decay[key] = (f64::from(lower)
                    / (1. + (t[9][head] * (t[3][base + key] + t[10][base + key])).exp()))
                .exp();
            }
            for value in 0..128 {
                let row = (head * 128 + value) * 128;
                let mut hk = 0.;
                for key in 0..128 {
                    self.h[row + key] *= decay[key];
                    hk += self.h[row + key] * qkv[1][base + key];
                }
                let u = (qkv[2][base + value] - hk) * beta;
                let mut hq = 0.;
                for key in 0..128 {
                    self.h[row + key] = qkv[1][base + key].mul_add(u, self.h[row + key]);
                    hq += self.h[row + key] * qkv[0][base + key];
                }
                output[base + value] = hq;
            }
            let scale = 1.
                / (output[base..base + 128].iter().map(|x| x * x).sum::<f64>() / 128.
                    + f64::from(EPS))
                .sqrt();
            for value in 0..128 {
                let c = base + value;
                output[c] = output[c] * scale * t[11][value] / (1. + (-t[5][c]).exp());
            }
        }
        output
    }
}
#[derive(Default)]
struct ErrorStats {
    total: usize,
    finite_pairs: usize,
    square: f64,
    max: f64,
    nonfinite: usize,
}
impl ErrorStats {
    fn add(&mut self, actual: &[f32], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected) {
            self.total += 1;
            if !a.is_finite() || !b.is_finite() {
                self.nonfinite += 1;
                continue;
            }
            self.finite_pairs += 1;
            let e = (f64::from(a) - b).abs();
            self.square += e * e;
            self.max = self.max.max(e);
        }
    }
    fn json(&self) -> serde_json::Value {
        // Nonfinite pairs must not dilute the finite-pair error denominator.
        let rms = (self.finite_pairs > 0).then(|| (self.square / self.finite_pairs as f64).sqrt());
        let max_abs = (self.finite_pairs > 0).then_some(self.max);
        json!({"total":self.total,"finite_pairs":self.finite_pairs,
            "nonfinite":self.nonfinite,"rms":rms,"max_abs":max_abs})
    }
}
#[test]
fn replay_f16_error_stats_nonfinite_pairs() {
    let mut stats = ErrorStats::default();
    let empty = stats.json();
    assert_eq!(empty["total"], 0);
    assert_eq!(empty["finite_pairs"], 0);
    assert!(empty["rms"].is_null());
    assert!(empty["max_abs"].is_null());
    stats.add(
        &[f32::NAN, f32::INFINITY, 1., 2.],
        &[0., 0., f64::NAN, f64::INFINITY],
    );
    let invalid = stats.json();
    assert_eq!(invalid["total"], 4);
    assert_eq!(invalid["finite_pairs"], 0);
    assert_eq!(invalid["nonfinite"], 4);
    assert!(invalid["rms"].is_null());
    assert!(invalid["max_abs"].is_null());
    stats.add(&[4., 6.], &[1., 2.]); // finite errors 3 and 4
    let mixed = stats.json();
    assert_eq!(mixed["total"], 6);
    assert_eq!(mixed["finite_pairs"], 2);
    assert_eq!(mixed["nonfinite"], 4);
    assert_eq!(mixed["rms"].as_f64().unwrap(), (25.0f64 / 2.).sqrt());
    assert_eq!(mixed["max_abs"], 4.);
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn binary_hash() -> String {
    let mut file = std::fs::File::open(std::env::current_exe().unwrap()).unwrap();
    let mut digest = Sha256::new();
    let mut bytes = [0; 65536];
    loop {
        let n = file.read(&mut bytes).unwrap();
        if n == 0 {
            break;
        }
        digest.update(&bytes[..n]);
    }
    format!("{:x}", digest.finalize())
}

#[test]
#[ignore = "bounded synthetic 1024-step two-head slow-decay GPU/f64 study; optional new GLM53_STAGE_OUT JSON; descriptive errors, no quality threshold"]
fn replay_f16_slow_stream_persistent_f64_1024() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    const HEADS: usize = 2;
    const STEPS: usize = 1024;
    const BATCH: usize = 32;
    const SLOW: f32 = -0.001;
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
    let ordinary = State::new(&ctx, HEADS, 0, false);
    let fp32 = State::new(&ctx, HEADS, 0, false);
    let records = Records::new(&ctx, HEADS, 8);
    let lp: Vec<_> = MODES
        .into_iter()
        .map(|m| LpState::new(&ctx, HEADS, 0, false, m))
        .collect();
    let mut oracle = Oracle::new(&ordinary, HEADS);
    let mut state_error: [ErrorStats; 4] = std::array::from_fn(|_| ErrorStats::default());
    let mut output_error: [ErrorStats; 4] = std::array::from_fn(|_| ErrorStats::default());
    let mut lp_read_error: [ErrorStats; 2] = std::array::from_fn(|_| ErrorStats::default());
    let width = HEADS * 128;
    let captures: Vec<_> = (0..4)
        .map(|_| {
            tensor(
                &ctx,
                &vec![f32::NAN; BATCH * width],
                vec![width as u64, BATCH as u64],
            )
        })
        .collect();
    let views: Vec<Vec<_>> = captures
        .iter()
        .map(|t| {
            (0..BATCH)
                .map(|j| t.view_subrange((j * width) as u64, vec![width as u64]))
                .collect()
        })
        .collect();
    let matrix = HEADS * 128 * 128;
    let state_captures: Vec<_> = (0..4)
        .map(|_| {
            tensor(
                &ctx,
                &vec![f32::NAN; BATCH * matrix],
                vec![128, 128, HEADS as u64, BATCH as u64],
            )
        })
        .collect();
    let state_views: Vec<Vec<_>> = state_captures
        .iter()
        .map(|t| {
            (0..BATCH)
                .map(|j| t.view_subrange((j * matrix) as u64, vec![128, 128, HEADS as u64]))
                .collect()
        })
        .collect();
    let mut logical_state_error: [ErrorStats; 4] = std::array::from_fn(|_| ErrorStats::default());
    for start in (0..STEPS).step_by(BATCH) {
        let inputs: Vec<_> = (start..start + BATCH)
            .map(|s| Inputs::new(&ctx, HEADS, s, false))
            .collect();
        run(&ctx, |enc| {
            for (j, input) in inputs.iter().enumerate() {
                let mut b = input.bind(&ordinary);
                b.out = &views[0][j];
                encode_kda_decode(&ctx, enc, HEADS, &b, SLOW, EPS).unwrap();
                let mut b = input.bind(&fp32);
                b.out = &views[1][j];
                encode_kda_replay_f32(
                    &ctx,
                    enc,
                    HEADS,
                    ReplayPosition {
                        window: 8,
                        residue: (start + j) % 8,
                    },
                    &b,
                    &records.bind(),
                    SLOW,
                    EPS,
                )
                .unwrap();
                for (a, s) in lp.iter().enumerate() {
                    let mut b = input.bind(&s.s);
                    b.out = &views[a + 2][j];
                    s.encode(&ctx, enc, HEADS, start + j, &b, SLOW);
                }
                // Untimed precision experiment: capture the actual logical
                // state at every residue without mutating live checkpoints.
                let after = ReplayPosition {
                    window: 8,
                    residue: (start + j + 1) % 8,
                };
                encode_copy_offset_f32(&ctx, enc, &ordinary.h, 0, &state_views[0][j], matrix)
                    .unwrap();
                encode_kda_replay_materialize_f32(
                    &ctx,
                    enc,
                    HEADS,
                    after,
                    &fp32.h,
                    &records.bind(),
                    &state_views[1][j],
                )
                .unwrap();
                for (a, s) in lp.iter().enumerate() {
                    encode_kda_replay_materialize_f16(
                        &ctx,
                        enc,
                        HEADS,
                        after,
                        &s.read,
                        &s.r.bind().cached(),
                        &state_views[a + 2][j],
                    )
                    .unwrap();
                }
            }
        });
        let actual: Vec<_> = captures.iter().map(tensor_f32_at_offset).collect();
        let actual_state: Vec<_> = state_captures.iter().map(tensor_f32_at_offset).collect();
        for (j, input) in inputs.iter().enumerate() {
            let expected = oracle.step(input, SLOW);
            for a in 0..4 {
                output_error[a].add(&actual[a][j * width..(j + 1) * width], &expected);
                logical_state_error[a]
                    .add(&actual_state[a][j * matrix..(j + 1) * matrix], &oracle.h);
            }
        }
        for (a, h) in [
            tensor_f32_at_offset(&ordinary.h),
            tensor_f32_at_offset(&fp32.h),
            tensor_f32_at_offset(&lp[0].s.h),
            read_half(&lp[1].s.h),
        ]
        .iter()
        .enumerate()
        {
            state_error[a].add(h, &oracle.h);
        }
        for a in 0..2 {
            lp_read_error[a].add(&read_half(&lp[a].read), &oracle.h);
        }
        assert_bits(
            "separate anchor at completed windows",
            &ordinary.h,
            &lp[0].s.h,
        );
        assert_bits("FP32 control", &ordinary.h, &fp32.h);
    }
    let report = json!({"experiment":"physical_f16_persistent_f64_slow_stream","steps":STEPS,"heads":HEADS,"window":8,"lower_bound":SLOW,
        "commands":STEPS/BATCH,"steps_per_command":BATCH,"quality_gate":false,
        "scope":"Synthetic slow-decay precision experiment with lower_bound=-0.001 versus GLM release -5; not a natural GLM trace",
        "release_lower_bound":-5.0,"fixture_lower_bound":LOWER,
        "error_metrics":"RMS and max_abs cover finite pairs only (null with none); total and nonfinite counts are reported separately",
        "device":ctx.device_facts(),"binary_sha256":binary_hash(),
        "shader_sha256":hash(SOURCE.as_bytes()),"ordinary_shader_sha256":hash(ORDINARY.as_bytes()),
        "host_sha256":hash(include_str!("../../lp.rs").as_bytes()),
        "fp32_host_sha256":hash(include_str!("../../../replay.rs").as_bytes()),
        "ordinary_host_sha256":hash(include_str!("../../../../kda.rs").as_bytes()),
        "base_fixture_sha256":hash(include_str!("../../tests.rs").as_bytes()),
        "fixture_sha256":hash(include_str!("../low_precision.rs").as_bytes()),
        "precision_source_sha256":hash(include_str!("precision.rs").as_bytes()),
        "oracle":"persistent f64 conv/state and recurrence, F32 inputs widened once per step, no F32 feedback; independent scalar reductions",
        "anchor_sampling":"authoritative anchors after each 32-step batch (four complete windows)",
        "logical_state_sampling":"all residues; LP materialization at residue zero is rounded physical read, output still came from F32 work before rounding",
        "logical_state_error":logical_state_error.iter().map(ErrorStats::json).collect::<Vec<_>>(),
        "output_sampling":"every actual recurrent output at all residues including flush",
        "arms":["ordinary_f32","replay_f32","separate_f32_anchor_f16_read","shared_f16_anchor_read"],
        "state_error":state_error.iter().map(ErrorStats::json).collect::<Vec<_>>(),
        "output_error":output_error.iter().map(ErrorStats::json).collect::<Vec<_>>(),
        "physical_read_error":lp_read_error.iter().map(ErrorStats::json).collect::<Vec<_>>()});
    if let Some(f) = output.as_mut() {
        serde_json::to_writer_pretty(&mut *f, &report).unwrap();
        f.write_all(b"\n").unwrap();
        f.sync_all().unwrap();
    } else {
        eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
    }
}
