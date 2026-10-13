//! Physical-F16 leaf contracts. No model/runtime integration.
use super::*;
use half::f16;

fn payload(t: &MetalTensor) -> Vec<u8> {
    let bytes = tensor_backing_bytes(t);
    bytes[t.offset as usize..(t.offset + t.n_bytes()) as usize].to_vec()
}
// Only used after command completion, or before first submission. Host writes
// intentionally preserve offset guards and never race in-flight GPU work.
fn write_payload(t: &MetalTensor, bytes: &[u8]) {
    assert_eq!(t.n_bytes() as usize, bytes.len());
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize),
            bytes.len(),
        );
    }
}
fn rounded(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| f16::from_f32(x).to_f32()).collect()
}
fn half_tensor(ctx: &MetalContext, v: &[f32], shape: Vec<u64>) -> MetalTensor {
    let bytes: Vec<_> = v
        .iter()
        .flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes())
        .collect();
    offset_tensor(ctx, 32, &bytes, 28, shape, GgmlType::F16)
}
fn read_half(t: &MetalTensor) -> Vec<f32> {
    assert_eq!(t.dtype, GgmlType::F16);
    payload(t)
        .chunks_exact(2)
        .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect()
}
struct LpRecords {
    t: [MetalTensor; 5],
}
impl LpRecords {
    fn new(ctx: &MetalContext, heads: usize) -> Self {
        Self {
            t: std::array::from_fn(|i| {
                let shape = if i == 4 {
                    vec![8, heads as u64]
                } else {
                    vec![128, 8, heads as u64]
                };
                tensor(
                    ctx,
                    &vec![f32::NAN; if i == 4 { 8 * heads } else { 128 * 8 * heads }],
                    shape,
                )
            }),
        }
    }
    fn bind(&self) -> ReplayLpRecords<'_> {
        ReplayLpRecords {
            k: &self.t[0],
            d: &self.t[1],
            u: &self.t[2],
            v: &self.t[3],
            beta: &self.t[4],
        }
    }
}
struct LpState {
    s: State,
    read: MetalTensor,
    r: LpRecords,
    mode: ReplayLpMode,
}
impl LpState {
    fn new(
        ctx: &MetalContext,
        heads: usize,
        seed: usize,
        silent: bool,
        mode: ReplayLpMode,
    ) -> Self {
        let mut s = State::new(ctx, heads, seed, silent);
        let read = half_tensor(ctx, &tensor_f32_at_offset(&s.h), s.h.shape.clone());
        if mode == ReplayLpMode::SharedF16Anchor {
            s.h = read.clone();
        }
        Self {
            s,
            read,
            r: LpRecords::new(ctx, heads),
            mode,
        }
    }
    fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        heads: usize,
        step: usize,
        b: &KdaDecode<'_>,
        lower: f32,
    ) {
        encode_kda_replay_f16(
            ctx,
            enc,
            heads,
            ReplayPosition {
                window: 8,
                residue: step % 8,
            },
            b,
            self.mode,
            &self.read,
            &self.r.bind(),
            lower,
            EPS,
        )
        .unwrap();
    }
}
const MODES: [ReplayLpMode; 2] = [
    ReplayLpMode::SeparateF32Anchor,
    ReplayLpMode::SharedF16Anchor,
];

#[test]
fn replay_f16_source_handshake() {
    assert_eq!(LP_TGM_BYTES, 14944);
    for mode in MODES {
        let signature = format!("kernel void {}(", mode.kernel());
        let body = SOURCE
            .split(signature.as_str())
            .nth(1)
            .unwrap()
            .split("\n}\n")
            .next()
            .unwrap();
        assert!(body.contains("device half * read_state [[buffer(19)]]"));
        assert!(body.contains("record_v [[buffer(20)]]"));
        assert!(body.contains("record_beta [[buffer(21)]]"));
        assert!(body.contains("flush ? record_v[record_head + i] : record_u[record_head + i]"));
        assert!(body.contains("old_u = (old_u - old_hk) * hist_beta[j]"));
        for (start, end) in [
            (
                "    if (tid < D) {\n        float q_acc",
                "    if (tid == 0u)",
            ),
            ("    float q_sumsq", "    if (!flush)"),
            ("    float o_sumsq", "\n}"),
        ] {
            let section = body.split(start).nth(1).unwrap().split(end).next().unwrap();
            let ordinary = ORDINARY.split(start).nth(1).unwrap();
            // Ordinary normalization is followed directly by k0.
            let ordinary = if start.contains("q_sumsq") {
                ordinary.split("    const uint k0").next().unwrap()
            } else {
                ordinary.split(end).next().unwrap()
            };
            assert_eq!(section.trim(), ordinary.trim());
        }
    }
}

#[test]
fn replay_f16_physical_buffers_residues_flush_repair_and_canaries() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let heads = 3;
    for mode in MODES {
        let lp = LpState::new(&ctx, heads, 0, true, mode);
        let poisoned = LpState::new(&ctx, heads, 0, true, mode);
        let reference = State::new(&ctx, heads, 0, true);
        // Ordinary F32 working state initialized from the physical half read.
        // It independently checks every LP nonflush update/output, then resets
        // to the newly rounded repaired read at the next window boundary.
        let low_reference = State::new(&ctx, heads, 0, true);
        write_payload(&low_reference.h, bytemuck::cast_slice(&read_half(&lp.read)));
        if mode == ReplayLpMode::SharedF16Anchor {
            write_payload(&reference.h, bytemuck::cast_slice(&read_half(&lp.read)));
        }
        let materialized = tensor(
            &ctx,
            &vec![f32::NAN; heads * 128 * 128],
            vec![128, 128, heads as u64],
        );
        for step in 0..25 {
            let input = Inputs::new(&ctx, heads, step, true);
            let before_anchor = payload(&lp.s.h);
            let before_read = payload(&lp.read);
            let before_records: Vec<_> = lp.r.t.iter().map(payload).collect();
            let before_inputs: Vec<_> = input.t.iter().map(payload).collect();
            if step % 8 == 7 {
                // Poison every cached LP delta immediately before repair.
                // Finite huge and NaN patterns would destroy a cached-u flush.
                let corrupt = vec![if step == 7 { f32::NAN } else { 1e30 }; heads * 128 * 8];
                write_payload(&poisoned.r.t[2], bytemuck::cast_slice(&corrupt));
            }
            run(&ctx, |enc| {
                encode_kda_decode(&ctx, enc, heads, &input.bind(&reference), LOWER, EPS).unwrap();
                encode_kda_decode(&ctx, enc, heads, &input.bind(&low_reference), LOWER, EPS)
                    .unwrap();
                lp.encode(&ctx, enc, heads, step, &input.bind(&lp.s), LOWER);
                poisoned.encode(&ctx, enc, heads, step, &input.bind(&poisoned.s), LOWER);
                encode_kda_replay_materialize_f16(
                    &ctx,
                    enc,
                    heads,
                    ReplayPosition {
                        window: 8,
                        residue: (step + 1) % 8,
                    },
                    &lp.read,
                    &lp.r.bind().cached(),
                    &materialized,
                )
                .unwrap();
            });
            assert_bits("clean/poisoned output", &lp.s.out, &poisoned.s.out);
            assert_eq!(payload(&lp.s.h), payload(&poisoned.s.h));
            assert_eq!(payload(&lp.read), payload(&poisoned.read));
            assert_bits("ordinary convolution", &lp.s.conv, &reference.conv);
            if step % 8 == 7 {
                assert_bits("flush full precision readout", &lp.s.out, &reference.out);
                let expected = tensor_f32_at_offset(&reference.h);
                if mode == ReplayLpMode::SeparateF32Anchor {
                    assert_bits("authoritative flush", &lp.s.h, &reference.h);
                }
                assert_eq!(read_half(&lp.read), rounded(&expected));
                write_payload(&low_reference.h, bytemuck::cast_slice(&read_half(&lp.read)));
                assert_eq!(tensor_f32_at_offset(&materialized), rounded(&expected));
                assert_eq!(
                    before_records,
                    lp.r.t.iter().map(payload).collect::<Vec<_>>()
                );
                if mode == ReplayLpMode::SharedF16Anchor {
                    // Independent ordinary control with exactly the shared
                    // anchor rounding cadence; historical u was recomputed
                    // by ordinary KDA from each window's rounded anchor.
                    write_payload(&reference.h, bytemuck::cast_slice(&rounded(&expected)));
                }
            } else {
                assert_bits(
                    "nonflush physical read output",
                    &lp.s.out,
                    &low_reference.out,
                );
                assert_bits(
                    "nonflush physical read state",
                    &materialized,
                    &low_reference.h,
                );
                assert_eq!(before_anchor, payload(&lp.s.h));
                assert_eq!(before_read, payload(&lp.read));
                // Scalar F32 cached replay verifies materialization at every
                // residue. No dot/reduction oracle needed for cached updates.
                let mut expected = read_half(&lp.read);
                let k = tensor_f32_at_offset(&lp.r.t[0]);
                let d = tensor_f32_at_offset(&lp.r.t[1]);
                let u = tensor_f32_at_offset(&lp.r.t[2]);
                for head in 0..heads {
                    for v in 0..128 {
                        for key in 0..128 {
                            let i = (head * 128 + v) * 128 + key;
                            for slot in 0..=step % 8 {
                                let base = (head * 8 + slot) * 128;
                                expected[i] =
                                    k[base + key].mul_add(u[base + v], expected[i] * d[base + key]);
                            }
                        }
                    }
                }
                let actual = tensor_f32_at_offset(&materialized);
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
                for (plane, t) in lp.r.t.iter().enumerate() {
                    let after = payload(t);
                    let channels = if plane == 4 { 1 } else { 128 };
                    let slot_bytes = channels * 4;
                    for head in 0..heads {
                        for slot in 0..8 {
                            if slot != step % 8 {
                                let start = (head * 8 + slot) * slot_bytes;
                                assert_eq!(
                                    &after[start..start + slot_bytes],
                                    &before_records[plane][start..start + slot_bytes]
                                );
                            }
                        }
                    }
                }
            }
            assert!(
                tensor_f32_at_offset(&lp.s.out)[(heads - 1) * 128..]
                    .iter()
                    .all(|x| *x == 0.0)
            );
            assert_eq!(
                before_inputs,
                input.t.iter().map(payload).collect::<Vec<_>>()
            );
            for t in [&lp.s.h, &lp.read, &lp.s.conv, &lp.s.out, &materialized]
                .into_iter()
                .chain(lp.r.t.iter())
                .chain(input.t.iter())
            {
                assert_offset_guards(t, 32, 28);
            }
        }
    }
}

#[test]
fn replay_f16_malformed_bindings_predispatch() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let input = Inputs::new(&ctx, 2, 0, false);
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    for mode in MODES {
        let lp = LpState::new(&ctx, 2, 0, false, mode);
        let rejected = |r: Result<(), MetalError>| {
            assert!(r.is_err());
            assert!(dispatch_census_take().is_empty());
        };
        let call = |heads,
                    p,
                    b: &KdaDecode<'_>,
                    read: &MetalTensor,
                    r: &ReplayLpRecords<'_>,
                    lower,
                    eps| {
            encode_kda_replay_f16(&ctx, &enc, heads, p, b, mode, read, r, lower, eps)
        };
        let p = ReplayPosition {
            window: 8,
            residue: 0,
        };
        for (heads, window, residue, lower, eps) in [
            (0, 8, 0, LOWER, EPS),
            (usize::MAX, 8, 0, LOWER, EPS),
            (2, 1, 0, LOWER, EPS),
            (2, 8, 8, LOWER, EPS),
            (2, 8, 0, f32::NAN, EPS),
            (2, 8, 0, LOWER, 0.),
        ] {
            dispatch_census_begin();
            rejected(call(
                heads,
                ReplayPosition { window, residue },
                &input.bind(&lp.s),
                &lp.read,
                &lp.r.bind(),
                lower,
                eps,
            ));
        }
        for field in 0..7 {
            for defect in 0..6 {
                let original = match field {
                    0 => &lp.read,
                    1 => &lp.s.h,
                    _ => &lp.r.t[field - 2],
                };
                let mut t = original.clone();
                match defect {
                    0 => t.dtype = GgmlType::I32,
                    1 => t.shape = vec![1],
                    2 => t.offset += 4,
                    3 => t.offset = u64::MAX - 15,
                    4 => t.offset = t.buffer.length() as u64,
                    _ => t.provenance = MetalTensorProvenance::OwnedWeightReadOnly,
                }
                let mut b = input.bind(&lp.s);
                let mut r = lp.r.bind();
                let mut read = &lp.read;
                match field {
                    0 => read = &t,
                    1 => b.state = &t,
                    2 => r.k = &t,
                    3 => r.d = &t,
                    4 => r.u = &t,
                    5 => r.v = &t,
                    _ => r.beta = &t,
                }
                dispatch_census_begin();
                rejected(call(2, p, &b, read, &r, LOWER, EPS));
            }
        }
        let mut alias_beta = lp.r.t[0].clone();
        alias_beta.shape = vec![8, 2];
        let mut r = lp.r.bind();
        r.beta = &alias_beta;
        dispatch_census_begin();
        rejected(call(2, p, &input.bind(&lp.s), &lp.read, &r, LOWER, EPS));
        let mut b = input.bind(&lp.s);
        b.out = b.q;
        dispatch_census_begin();
        rejected(call(2, p, &b, &lp.read, &lp.r.bind(), LOWER, EPS));
        let mut partial = lp.read.clone();
        partial.offset += 16;
        // Valid aligned extent (inside the suffix), forbidden partial alias.
        if mode == ReplayLpMode::SharedF16Anchor {
            dispatch_census_begin();
            rejected(call(
                2,
                p,
                &input.bind(&lp.s),
                &partial,
                &lp.r.bind(),
                LOWER,
                EPS,
            ));
        }
        let mut r = lp.r.bind();
        r.v = r.k;
        dispatch_census_begin();
        rejected(call(2, p, &input.bind(&lp.s), &lp.read, &r, LOWER, EPS));
        let other = half_tensor(&ctx, &read_half(&lp.read), lp.read.shape.clone());
        if mode == ReplayLpMode::SharedF16Anchor {
            dispatch_census_begin();
            rejected(call(
                2,
                p,
                &input.bind(&lp.s),
                &other,
                &lp.r.bind(),
                LOWER,
                EPS,
            ));
        } else {
            let mut alias = lp.s.h.clone();
            alias.dtype = GgmlType::F16;
            dispatch_census_begin();
            rejected(call(
                2,
                p,
                &input.bind(&lp.s),
                &alias,
                &lp.r.bind(),
                LOWER,
                EPS,
            ));
        }
        let mut alias = lp.read.clone();
        alias.dtype = GgmlType::F32;
        dispatch_census_begin();
        rejected(encode_kda_replay_materialize_f16(
            &ctx,
            &enc,
            2,
            p,
            &lp.read,
            &lp.r.bind().cached(),
            &alias,
        ));
    }
    enc.end();
    let concurrent = KernelEncoder::begin_concurrent(&command);
    let lp = LpState::new(&ctx, 2, 0, false, MODES[0]);
    dispatch_census_begin();
    let result = encode_kda_replay_f16(
        &ctx,
        &concurrent,
        2,
        ReplayPosition {
            window: 8,
            residue: 0,
        },
        &input.bind(&lp.s),
        lp.mode,
        &lp.read,
        &lp.r.bind(),
        LOWER,
        EPS,
    );
    assert!(result.is_err());
    assert!(dispatch_census_take().is_empty());
    concurrent.end(); // deliberately never commit malformed commands
}

mod precision;
mod timing;
