use super::*;
use crate::metal::test_support::{
    assert_offset_guards, offset_tensor, tensor_backing_bytes, tensor_f32_at_offset,
};

const LOWER: f32 = -1.25;
const EPS: f32 = 1e-5;
const SOURCE: &str = include_str!("../../../../../../kernels/research/glm53_kda_replay.metal");
const ORDINARY: &str = include_str!("../../../../../../kernels/glm53_kda.metal");

fn values(n: usize, salt: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7919 + salt * 104729) % 2003) as f32 / 2003.0 - 0.5) * scale)
        .collect()
}

fn tensor(ctx: &MetalContext, v: &[f32], shape: Vec<u64>) -> MetalTensor {
    offset_tensor(ctx, 32, bytemuck::cast_slice(v), 28, shape, GgmlType::F32)
}

struct Inputs {
    // Ordinary binding order: q/k/v/gate/beta/z/taps q/k/v/a/bias/norm.
    t: [MetalTensor; 12],
}

impl Inputs {
    fn new(ctx: &MetalContext, heads: usize, step: usize, silent: bool) -> Self {
        let width = heads * 128;
        let row = |salt, scale, zero: bool| {
            let mut v = values(width, salt + step * 19, scale);
            if silent && zero {
                v[(heads - 1) * 128..].fill(0.0);
            }
            tensor(ctx, &v, vec![width as u64])
        };
        Self {
            t: [
                row(1, 2.0, true),
                row(2, 2.0, true),
                row(3, 2.0, true),
                row(4, 4.0, false),
                tensor(ctx, &values(heads, 5 + step * 19, 3.0), vec![heads as u64]),
                row(6, 2.0, false),
                tensor(ctx, &values(width * 4, 7, 1.0), vec![4, 1, width as u64]),
                tensor(ctx, &values(width * 4, 8, 1.0), vec![4, 1, width as u64]),
                tensor(ctx, &values(width * 4, 9, 1.0), vec![4, 1, width as u64]),
                tensor(
                    ctx,
                    &(0..heads)
                        .map(|h| -0.4 - h as f32 * 0.003)
                        .collect::<Vec<_>>(),
                    vec![heads as u64],
                ),
                tensor(ctx, &values(width, 11, 1.0), vec![width as u64]),
                tensor(
                    ctx,
                    &(0..128).map(|i| 0.8 + i as f32 * 0.002).collect::<Vec<_>>(),
                    vec![128],
                ),
            ],
        }
    }

    fn bind<'a>(&'a self, s: &'a State) -> KdaDecode<'a> {
        KdaDecode {
            q: &self.t[0],
            k: &self.t[1],
            v: &self.t[2],
            raw_gate: &self.t[3],
            raw_beta: &self.t[4],
            output_gate: &self.t[5],
            q_conv: &self.t[6],
            k_conv: &self.t[7],
            v_conv: &self.t[8],
            neg_exp_a_log: &self.t[9],
            dt_bias: &self.t[10],
            output_norm: &self.t[11],
            conv_state: &s.conv,
            state: &s.h,
            out: &s.out,
        }
    }
}

struct State {
    h: MetalTensor,
    conv: MetalTensor,
    out: MetalTensor,
}
impl State {
    fn new(ctx: &MetalContext, heads: usize, seed: usize, silent: bool) -> Self {
        let width = heads * 128;
        let mut h = values(width * 128, seed + 21, 0.2);
        let mut conv = values(width * 9, seed + 22, 0.4);
        if silent {
            h[(heads - 1) * 128 * 128..].fill(0.0);
            for row in conv.chunks_exact_mut(width) {
                row[(heads - 1) * 128..].fill(0.0);
            }
        }
        Self {
            h: tensor(ctx, &h, vec![128, 128, heads as u64]),
            conv: tensor(ctx, &conv, vec![width as u64, 3, 3]),
            out: tensor(ctx, &vec![f32::NAN; width], vec![width as u64]),
        }
    }
}

struct Records {
    t: [MetalTensor; 3],
}
impl Records {
    fn new(ctx: &MetalContext, heads: usize, window: usize) -> Self {
        Self {
            t: std::array::from_fn(|_| {
                tensor(
                    ctx,
                    &vec![f32::NAN; 128 * heads * window],
                    vec![128, window as u64, heads as u64],
                )
            }),
        }
    }
    fn bind(&self) -> ReplayRecords<'_> {
        ReplayRecords {
            k: &self.t[0],
            d: &self.t[1],
            u: &self.t[2],
        }
    }
}

fn run(ctx: &MetalContext, f: impl FnOnce(&KernelEncoder)) {
    let c = ctx.queue.commandBuffer().expect("command buffer");
    let e = KernelEncoder::begin(&c);
    f(&e);
    e.end();
    c.commit();
    wait_completed(&c).unwrap();
}

fn bits(t: &MetalTensor) -> Vec<u32> {
    tensor_f32_at_offset(t)
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn assert_bits(label: &str, a: &MetalTensor, b: &MetalTensor) {
    let a = tensor_f32_at_offset(a);
    let b = tensor_f32_at_offset(b);
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(&b).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "{label}[{i}] nonfinite {a}/{b}"
        );
        assert_eq!(a.to_bits(), b.to_bits(), "{label}[{i}] {a}/{b}");
    }
}

#[test]
fn replay_f32_source_and_host_handshake() {
    assert_eq!(std::mem::size_of::<Args>(), 24);
    assert_eq!(SCRATCH_FLOATS * 4, 2624);
    assert_eq!((SCRATCH_FLOATS + 3 * 8 * 128) * 4, 14912);
    assert!(SOURCE.contains("scratch + 656u"));
    for name in [REPLAY_KERNEL, MATERIALIZE_KERNEL] {
        assert!(SOURCE.contains(name));
    }
    // Conv, SiLU, per-key decay, q/k normalization and output normalization
    // are copied verbatim; keep source drift visible before GPU validation.
    for (start, end) in [
        (
            "    if (tid < D) {\n        float q_acc",
            "    if (tid == 0u)",
        ),
        ("    float q_sumsq", "    const uint k0"),
        ("    float o_sumsq", "\n}\n"),
    ] {
        let a = ORDINARY.find(start).unwrap();
        let end_a = ORDINARY[a..].find(end).unwrap() + a;
        let b = SOURCE.find(start).unwrap();
        let end_b = SOURCE[b..].find(end).unwrap() + b;
        let candidate=SOURCE[b..end_b].replace("    if (!flush) {\n        record_k[append + tid] = sk[tid];\n        record_d[append + tid] = sd[tid];\n    }\n","");
        assert_eq!(&ORDINARY[a..end_a], candidate);
    }
    assert!(
        SOURCE
            .contains("h *= old_d;\n            h = fma(old_k, float4(hist_u[j * D + value]), h);")
    );
    assert!(!SOURCE.contains("dot(h, old_k)"));
    for window in [0, 2, 7, 9, usize::MAX] {
        assert!(geometry(REPLAY_KERNEL, 1, ReplayPosition { window, residue: 0 }).is_err());
    }
    assert!(
        geometry(
            REPLAY_KERNEL,
            usize::MAX,
            ReplayPosition {
                window: 8,
                residue: 0
            }
        )
        .is_err()
    );
}

#[test]
fn replay_f32_l1_l8_outputs_state_conv_bitexact_multiple_windows() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    const HEADS: usize = 3;
    for window in [1, 8] {
        let ordinary = State::new(&ctx, HEADS, 0, true);
        let replay = State::new(&ctx, HEADS, 0, true);
        let records = Records::new(&ctx, HEADS, window);
        let materialized = tensor(
            &ctx,
            &vec![f32::NAN; HEADS * 128 * 128],
            vec![128, 128, HEADS as u64],
        );
        let initial = tensor_f32_at_offset(&replay.h);
        assert_ne!(
            initial[1].to_bits(),
            initial[128].to_bits(),
            "nonsymmetric fixture"
        );
        let mut readonly_checkpoint = replay.h.clone();
        readonly_checkpoint.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        let readonly_records = records.t.clone().map(|mut t| {
            t.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
            t
        });
        run(&ctx, |enc| {
            encode_kda_replay_materialize_f32(
                &ctx,
                enc,
                HEADS,
                ReplayPosition { window, residue: 0 },
                &readonly_checkpoint,
                &ReplayRecords {
                    k: &readonly_records[0],
                    d: &readonly_records[1],
                    u: &readonly_records[2],
                },
                &materialized,
            )
            .unwrap()
        });
        assert_bits("initial residue zero", &ordinary.h, &materialized);
        let mut residue = 0;
        for step in 0..25 {
            let input = Inputs::new(&ctx, HEADS, step, true);
            let input_before: Vec<_> = input.t.iter().map(tensor_backing_bytes).collect();
            let cp_before = tensor_backing_bytes(&replay.h);
            let rec_before: Vec<_> = records.t.iter().map(bits).collect();
            dispatch_census_begin();
            run(&ctx, |enc| {
                encode_kda_decode(&ctx, enc, HEADS, &input.bind(&ordinary), LOWER, EPS).unwrap();
                encode_kda_replay_f32(
                    &ctx,
                    enc,
                    HEADS,
                    ReplayPosition { window, residue },
                    &input.bind(&replay),
                    &records.bind(),
                    LOWER,
                    EPS,
                )
                .unwrap();
            });
            let census = dispatch_census_take();
            assert_eq!(census.len(), 2);
            assert_eq!(census[1].kernel, REPLAY_KERNEL);
            let flush = residue + 1 == window;
            if !flush {
                assert_eq!(
                    tensor_backing_bytes(&replay.h),
                    cp_before,
                    "nonflush checkpoint"
                );
            } else {
                assert_bits("flush checkpoint", &replay.h, &ordinary.h);
            }
            assert_bits("output", &replay.out, &ordinary.out);
            assert_bits("conv", &replay.conv, &ordinary.conv);
            for (which, t) in records.t.iter().enumerate() {
                let after = bits(t);
                for head in 0..HEADS {
                    for slot in 0..window {
                        let off = (head * window + slot) * 128;
                        if flush || slot != residue {
                            assert_eq!(
                                &after[off..off + 128],
                                &rec_before[which][off..off + 128],
                                "inactive record"
                            );
                        } else {
                            assert!(
                                tensor_f32_at_offset(t)[off..off + 128]
                                    .iter()
                                    .all(|v| v.is_finite())
                            );
                            if head == 0 && residue > 0 && which < 2 {
                                assert_ne!(
                                    &after[off..off + 128],
                                    &after[off - 128..off],
                                    "distinct historical keys/decays"
                                );
                            }
                        }
                    }
                }
            }
            residue = (residue + 1) % window;
            let cp = tensor_backing_bytes(&replay.h);
            let rec: Vec<_> = records.t.iter().map(tensor_backing_bytes).collect();
            let conv = tensor_backing_bytes(&replay.conv);
            run(&ctx, |enc| {
                encode_kda_replay_materialize_f32(
                    &ctx,
                    enc,
                    HEADS,
                    ReplayPosition { window, residue },
                    &replay.h,
                    &records.bind(),
                    &materialized,
                )
                .unwrap()
            });
            assert_bits("materialized logical state", &materialized, &ordinary.h);
            assert_eq!(
                tensor_backing_bytes(&replay.h),
                cp,
                "materialization checkpoint mutation"
            );
            assert_eq!(
                records
                    .t
                    .iter()
                    .map(tensor_backing_bytes)
                    .collect::<Vec<_>>(),
                rec
            );
            assert_eq!(tensor_backing_bytes(&replay.conv), conv);
            assert!(
                tensor_f32_at_offset(&replay.out)[(HEADS - 1) * 128..]
                    .iter()
                    .all(|&v| v == 0.0)
            );
            assert!(
                tensor_f32_at_offset(&materialized)[(HEADS - 1) * 128 * 128..]
                    .iter()
                    .all(|&v| v == 0.0)
            );
            for (t, old) in input.t.iter().zip(input_before) {
                assert_eq!(tensor_backing_bytes(t), old);
            }
            for t in [
                &ordinary.h,
                &ordinary.conv,
                &ordinary.out,
                &replay.h,
                &replay.conv,
                &replay.out,
                &materialized,
                &records.t[0],
                &records.t[1],
                &records.t[2],
            ] {
                assert_offset_guards(t, 32, 28);
            }
        }
    }
}

#[test]
fn replay_f32_rejects_malformed_bindings_before_dispatch() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let i = Inputs::new(&ctx, 2, 0, false);
    let s = State::new(&ctx, 2, 0, false);
    let r = Records::new(&ctx, 2, 8);
    let output = tensor(&ctx, &vec![0.0; 2 * 128 * 128], vec![128, 128, 2]);
    let c = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&c);
    let p = ReplayPosition {
        window: 8,
        residue: 0,
    };
    let rejected = |result: Result<(), MetalError>| {
        let census = dispatch_census_take();
        assert!(result.is_err());
        assert!(census.is_empty());
    };
    for (heads, window, residue, lower, eps) in [
        (0, 8, 0, LOWER, EPS),
        (usize::MAX, 8, 0, LOWER, EPS),
        (2, 2, 0, LOWER, EPS),
        (2, 8, 8, LOWER, EPS),
        (2, 8, 0, f32::NAN, EPS),
        (2, 8, 0, LOWER, 0.0),
        (2, 8, 0, 0.0, EPS),
    ] {
        dispatch_census_begin();
        rejected(encode_kda_replay_f32(
            &ctx,
            &enc,
            heads,
            ReplayPosition { window, residue },
            &i.bind(&s),
            &r.bind(),
            lower,
            eps,
        ));
    }
    let mut malformed = Vec::new();
    let mut t = r.t[0].clone();
    t.dtype = GgmlType::F16;
    malformed.push(t);
    let mut t = r.t[0].clone();
    t.shape = vec![128, 7, 2];
    malformed.push(t);
    let mut t = r.t[0].clone();
    t.offset += 4;
    malformed.push(t);
    let mut t = r.t[0].clone();
    t.offset = u64::MAX - 15;
    malformed.push(t);
    let mut t = r.t[0].clone();
    t.offset = t.buffer.length() as u64;
    malformed.push(t);
    for t in &malformed {
        dispatch_census_begin();
        rejected(encode_kda_replay_f32(
            &ctx,
            &enc,
            2,
            p,
            &i.bind(&s),
            &ReplayRecords {
                k: t,
                d: &r.t[1],
                u: &r.t[2],
            },
            LOWER,
            EPS,
        ));
        dispatch_census_begin();
        rejected(encode_kda_replay_materialize_f32(
            &ctx,
            &enc,
            2,
            p,
            &s.h,
            &ReplayRecords {
                k: t,
                d: &r.t[1],
                u: &r.t[2],
            },
            &output,
        ));
    }
    let mut readonly = r.t[0].clone();
    readonly.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    dispatch_census_begin();
    rejected(encode_kda_replay_f32(
        &ctx,
        &enc,
        2,
        p,
        &i.bind(&s),
        &ReplayRecords {
            k: &readonly,
            d: &r.t[1],
            u: &r.t[2],
        },
        LOWER,
        EPS,
    ));
    let mut bad_checkpoint = s.h.clone();
    bad_checkpoint.dtype = GgmlType::F16;
    let mut b = i.bind(&s);
    b.state = &bad_checkpoint;
    dispatch_census_begin();
    rejected(encode_kda_replay_f32(
        &ctx,
        &enc,
        2,
        p,
        &b,
        &r.bind(),
        LOWER,
        EPS,
    ));
    dispatch_census_begin();
    rejected(encode_kda_replay_materialize_f32(
        &ctx,
        &enc,
        2,
        p,
        &bad_checkpoint,
        &r.bind(),
        &output,
    ));
    let mut readonly_out = output.clone();
    readonly_out.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    dispatch_census_begin();
    rejected(encode_kda_replay_materialize_f32(
        &ctx,
        &enc,
        2,
        p,
        &s.h,
        &r.bind(),
        &readonly_out,
    ));
    dispatch_census_begin();
    rejected(encode_kda_replay_f32(
        &ctx,
        &enc,
        2,
        p,
        &i.bind(&s),
        &ReplayRecords {
            k: &r.t[0],
            d: &r.t[0],
            u: &r.t[2],
        },
        LOWER,
        EPS,
    ));
    let alias = s.h.view_subrange(0, vec![128, 8, 2]);
    dispatch_census_begin();
    rejected(encode_kda_replay_f32(
        &ctx,
        &enc,
        2,
        p,
        &i.bind(&s),
        &ReplayRecords {
            k: &alias,
            d: &r.t[1],
            u: &r.t[2],
        },
        LOWER,
        EPS,
    ));
    dispatch_census_begin();
    rejected(encode_kda_replay_materialize_f32(
        &ctx,
        &enc,
        2,
        p,
        &s.h,
        &r.bind(),
        &s.h,
    ));
    let mut b = i.bind(&s);
    b.out = b.q;
    dispatch_census_begin();
    rejected(encode_kda_replay_f32(
        &ctx,
        &enc,
        2,
        p,
        &b,
        &r.bind(),
        LOWER,
        EPS,
    ));
    enc.end();
    let enc = KernelEncoder::begin_concurrent(&c);
    dispatch_census_begin();
    rejected(encode_kda_replay_f32(
        &ctx,
        &enc,
        2,
        p,
        &i.bind(&s),
        &r.bind(),
        LOWER,
        EPS,
    ));
    dispatch_census_begin();
    rejected(encode_kda_replay_materialize_f32(
        &ctx,
        &enc,
        2,
        p,
        &s.h,
        &r.bind(),
        &output,
    ));
    enc.end(); // No commit: every malformed binding failed on the host.
}

mod timing;
