//! Production Full loop lifetime witness, not a generic autoreleasepool test.
//! Weak probes cover prefill::allocate scratch. Direct uploads, tile/capture/
//! trace buffers and command objects have only sampled device accounting.
use super::*;
use objc2::rc::Weak;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

const CHUNK: usize = 16;
const STEPS: usize = 256;
const CAPACITY: usize = 8 * CHUNK + 1 + STEPS;
const MIB: u64 = 1024 * 1024;
const PAGE_SLACK: u64 = 16384;
const DEVICE_RESIDUAL: u64 = 2 * MIB;

// One real ratio-1 attention/FFN/Engram layer, using the existing toy geometry.
const DENSE: &[(&str, &[u64], f32)] = &[
    ("layers.0.attn.wq_a.weight", &[32, 6], 1.0 / 128.0),
    ("layers.0.attn.wq_b.weight", &[6, 8], 1.0 / 128.0),
    ("layers.0.attn.wkv.weight", &[32, 4], 1.0 / 128.0),
    ("layers.0.attn.wo_a.weight", &[4, 6], 1.0 / 128.0),
    ("layers.0.attn.wo_b.weight", &[6, 32], 1.0 / 128.0),
    ("layers.0.attn.q_norm.weight", &[6], 1.0),
    ("layers.0.attn.kv_norm.weight", &[4], 1.0),
    ("layers.0.attn.attn_sink", &[2], 0.0),
    ("layers.0.attn.compressor.wkv.weight", &[32, 4], 1.0 / 128.0),
    ("layers.0.attn.compressor.norm.weight", &[4], 1.0),
    ("layers.0.attn.indexer.wq_b.weight", &[6, 8], 1.0 / 128.0),
    (
        "layers.0.attn.indexer.weights_proj.weight",
        &[32, 2],
        1.0 / 128.0,
    ),
    ("layers.0.attn.indexer.wk.weight", &[4, 4], 1.0 / 128.0),
    ("layers.0.attn.indexer.k_norm.weight", &[4], 1.0),
    ("layers.0.attn_norm.weight", &[32], 1.0),
    ("layers.0.ffn_norm.weight", &[32], 1.0),
    ("layers.0.hc_attn_fn", &[128, 24], 1.0 / 1024.0),
    ("layers.0.hc_ffn_fn", &[128, 24], 1.0 / 1024.0),
    ("layers.0.hc_attn_scale", &[3], 0.25),
    ("layers.0.hc_ffn_scale", &[3], 0.25),
    ("layers.0.hc_attn_base", &[24], 0.0),
    ("layers.0.hc_ffn_base", &[24], 0.0),
    ("layers.0.ffn.gate.weight", &[32, 8], 1.0 / 64.0),
    (
        "layers.0.ffn.shared_experts.w1.weight",
        &[32, 32],
        1.0 / 64.0,
    ),
    (
        "layers.0.ffn.shared_experts.w3.weight",
        &[32, 32],
        1.0 / 64.0,
    ),
    (
        "layers.0.ffn.shared_experts.w2.weight",
        &[32, 32],
        1.0 / 64.0,
    ),
    ("layers.0.engram.wkv.weight", &[6144, 160], 1.0 / 8192.0),
    ("embed.weight", &[32, 8], 0.0),
    ("head.weight", &[32, 8], 0.0),
];

fn schedule() -> super::super::super::super::gguf::Schedule {
    super::super::super::super::gguf::Schedule {
        layers: 1,
        kv_source_layers: vec![0],
        index_source_layers: vec![0],
        compression_ratios: vec![1],
        candidate_source_layer: Some(0),
    }
}

fn chunk_widths(c: &Config) -> Vec<usize> {
    let mut widths = vec![4 * c.hidden, 4 * c.hidden, c.hidden, 4];
    widths.extend(HcRows::widths(c));
    widths.extend(ffn_gpu::Rows::widths(c, 1));
    widths.extend(attention_gpu::Rows::widths(c, CAPACITY));
    widths.extend(attention_gpu::Rows::select_widths(c, CAPACITY, CHUNK));
    widths.extend(attention_gpu::Rows::tile_widths(c, CHUNK).unwrap());
    widths.extend([6144, 5 * c.hidden]);
    widths
}

// A pre-allocation bound on explicit buffers, including two persistent states,
// one chunk (all optional half scratch), one capture and one token's trace.
// Context/library/PSO/driver allocations are deliberately not called buffers.
fn explicit_buffer_bound(mut price: impl FnMut(u64) -> u64) -> u64 {
    let c = toy::config();
    let weights: u64 = DENSE
        .iter()
        .map(|(_, shape, _)| price(shape.iter().product::<u64>() * 4))
        .sum();
    let banks: u64 = (0..3).map(|_| price(8 * 32 * 17)).sum();
    let states = 2 * gpu_step::bytes_with_schedule(&c, CAPACITY, &schedule(), 1).unwrap();
    let chunk: u64 = chunk_widths(&c)
        .into_iter()
        .map(|w| price((w * CHUNK * 4) as u64))
        .sum();
    let [input, inner] = ffn_half_scratch_elements(&c, CHUNK).unwrap();
    let half: u64 = [
        attention_half_scratch_elements(&c, CHUNK).unwrap(),
        input,
        inner,
    ]
    .into_iter()
    .map(|n| price((2 * n) as u64))
    .sum();
    let diagnostics: u64 = [
        c.hidden,
        c.hidden,
        c.index_topk,
        CAPACITY.div_ceil(c.candidate_block),
        4 * c.hidden,
        4,
    ]
    .into_iter()
    .map(|n| price((4 * n) as u64))
    .sum();
    weights + banks + states + chunk + half + diagnostics
}

#[test]
fn full_lifetime_fixture_explicit_buffers_below_32_mib() {
    let bound = explicit_buffer_bound(|bytes| bytes.max(1) + PAGE_SLACK);
    assert!(bound < 32 * MIB, "explicit buffer bound {bound}");
    assert_eq!(CAPACITY, 385);
}

struct TinyWeights {
    dense: BTreeMap<String, MetalTensor>,
    banks: Vec<MetalTensor>,
}

impl TinyWeights {
    fn new(ctx: &MetalContext) -> Self {
        let dense = DENSE
            .iter()
            .map(|&(name, shape, value)| {
                let values = (0..shape.iter().product::<u64>() as usize)
                    .map(|i| match name {
                        "embed.weight" => (i / 32) as f32 / 8.0 - 0.25,
                        "head.weight" => (i / 32) as f32 / 32.0 - 0.09375,
                        _ => value,
                    })
                    .collect::<Vec<_>>();
                (
                    name.to_owned(),
                    upload(ctx, &values, shape.to_vec()).unwrap(),
                )
            })
            .collect();
        let banks = [1u8, 3, 2]
            .into_iter()
            .map(|shift| {
                let mut bytes = Vec::new();
                for expert in 0..8u8 {
                    let code = (expert + shift) % 8;
                    for _ in 0..32 {
                        bytes.push(122);
                        bytes.extend([code | code << 4; 16]);
                    }
                }
                MetalTensor::from_bytes(ctx, &bytes, vec![32, 32, 8], GgmlType::MXFP4).unwrap()
            })
            .collect();
        Self { dense, banks }
    }

    fn model<'a>(&'a self, ctx: &'a MetalContext) -> Forward<'a> {
        let c = toy::config();
        let schedule = schedule();
        let b = bind::Binder {
            ctx,
            weights: &self.dense,
        };
        let mut hash = engram::HashConfig::default();
        hash.layers.truncate(1);
        hash.layers[0].layer = 0;
        Forward {
            ctx,
            engram_rows: EngramRows::Toy(|_, rows| {
                Ok(rows
                    .iter()
                    .flat_map(|row| vec![(*row % 17) as f32 / 128.0; 256])
                    .collect())
            }),
            engram_hash: hash,
            layers: vec![Layer {
                attention: b.attention_with_schedule(0, &c, &schedule).unwrap(),
                attn_hc: b.hc(0, "attn", &c).unwrap(),
                ffn_hc: b.hc(0, "ffn", &c).unwrap(),
                router: b.linear("layers.0.ffn.gate.weight", 32, 8).unwrap(),
                bias: (0..8).map(|i| i as f32 / 32.0).collect(),
                shared: ["w1", "w3", "w2"].map(|part| {
                    b.linear(
                        &format!("layers.0.ffn.shared_experts.{part}.weight"),
                        32,
                        32,
                    )
                    .unwrap()
                }),
                banks: [&self.banks[0], &self.banks[1], &self.banks[2]],
                engram: Some(Engram {
                    projection: b.linear("layers.0.engram.wkv.weight", 6144, 160).unwrap(),
                    gate: EngramGate::Separate {
                        q: vec![1.0; 128],
                        k: vec![1.0; 128],
                    },
                }),
            }],
            schedule,
            config: c,
            token_map: vec![5, 3, 2, 7, 0, 1, 6, 4],
            pad: 2,
            embedding: &self.dense["embed.weight"],
            norm: vec![1.0; 32],
            head: b.linear("head.weight", 32, 8).unwrap(),
            prefill_rows: crate::deepseek_v41::forward::MAX_PREFILL_CHUNK_ROWS,
            identity: Arc::new(()),
        }
    }
}

#[derive(Default)]
struct Observation {
    pending: Vec<Weak<ProtocolObject<dyn MTLBuffer>>>,
    chunk_bytes: u64,
    resource_high_water: u64,
    device_high_water: u64,
    after_chunks: Vec<(usize, usize, u64)>,
}

thread_local! {
    static OBSERVATION: RefCell<Option<Observation>> = const { RefCell::new(None) };
}

struct Observe;
impl Observe {
    fn begin() -> Self {
        OBSERVATION.with(|slot| assert!(slot.replace(Some(Observation::default())).is_none()));
        Self
    }

    fn finish(self) -> Observation {
        OBSERVATION.with(|slot| slot.take().unwrap())
    }
}
impl Drop for Observe {
    fn drop(&mut self) {
        OBSERVATION.with(|slot| slot.take());
    }
}

// Only weak references: the witness must not extend the lifetime it measures.
pub(in super::super) fn allocated(ctx: &MetalContext, tensor: &MetalTensor) {
    OBSERVATION.with(|slot| {
        if let Some(observation) = slot.borrow_mut().as_mut() {
            let weak = Weak::from_retained(&tensor.buffer);
            assert!(
                weak.load().is_some(),
                "live scratch must support a weak probe"
            );
            observation.pending.push(weak);
            observation.chunk_bytes += tensor.buffer.length() as u64;
            observation.device_high_water = observation
                .device_high_water
                .max(ctx.device.currentAllocatedSize() as u64);
        }
    });
}

pub(in super::super) fn completed_chunk(ctx: &MetalContext, state: &State, rows: usize) {
    OBSERVATION.with(|slot| {
        if let Some(o) = slot.borrow_mut().as_mut() {
            assert!(!state.is_poisoned());
            assert!(
                o.pending.iter().all(|weak| weak.load().is_none()),
                "completed chunk scratch retained at position {}",
                state.position()
            );
            assert_eq!(
                o.pending.is_empty(),
                rows == 1,
                "must observe actual multi-row scratch"
            );
            o.resource_high_water = o.resource_high_water.max(o.chunk_bytes);
            o.pending.clear();
            o.chunk_bytes = 0;
            o.after_chunks.push((
                state.position(),
                rows,
                ctx.device.currentAllocatedSize() as u64,
            ));
        }
    });
}

#[derive(Default)]
struct Captures {
    records: Vec<(usize, Vec<u32>)>,
    fault: Option<(usize, bool)>,
}
impl capture::Capture for Captures {
    fn wants_position(&self, position: usize) -> bool {
        position.is_multiple_of(CHUNK)
    }
    fn is_checkpoint(&self, _: usize) -> bool {
        true
    }
    fn layer(&mut self, capture: capture::AttentionCapture<'_>) -> Result<()> {
        assert_eq!(capture.layer, 0);
        assert!(capture.input.iter().all(|v| v.is_finite()));
        let checkpoint = capture.checkpoint.unwrap();
        assert!(checkpoint.output.iter().all(|v| v.is_finite()));
        let bits = capture
            .input
            .iter()
            .chain(checkpoint.output)
            .map(|v| v.to_bits())
            .chain(checkpoint.compressed_ids.iter().map(|&v| v as u32))
            .chain(checkpoint.candidate_mask.unwrap().iter().map(|&v| v as u32))
            .collect();
        self.records.push((capture.position, bits));
        if let Some((position, unwind)) = self
            .fault
            .filter(|(position, _)| *position == capture.position)
        {
            if unwind {
                panic!("lifetime capture panic at {position}");
            }
            return Err(format!("lifetime capture error at {position}"));
        }
        Ok(())
    }
}

fn same_state(model: &Forward<'_>, a: &State, b: &State) {
    assert!(!a.is_poisoned() && !b.is_poisoned());
    assert_eq!(a.addresses, b.addresses);
    let regions = |state: &State| {
        state
            .gpu_step
            .as_ref()
            .unwrap()
            .attention()
            .causal_regions(state.position())
            .unwrap()
            .into_iter()
            .map(|(tensor, n)| {
                f32_slice(tensor).unwrap()[..n]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(regions(a), regions(b));
    assert_eq!(
        model.snapshot_payload_bytes(a).unwrap(),
        model.snapshot_payload_bytes(b).unwrap()
    );
}

fn sequence(
    model: &Forward<'_>,
    actual: &mut State,
    reference: &mut State,
    chunks: usize,
    observe: bool,
) {
    let tokens = (0..chunks * CHUNK + 1)
        .map(|i| (i % 8) as u32)
        .collect::<Vec<_>>();
    let mut expected = Captures::default();
    let mut expected_logits = Vec::new();
    for chunk in tokens.chunks(CHUNK) {
        expected_logits = run_captured(
            model,
            reference,
            chunk,
            None,
            CHUNK,
            true,
            Some(&mut expected),
        )
        .unwrap();
    }
    let scope = observe.then(Observe::begin);
    let mut captures = Captures::default();
    let profile = profile::Scope::begin().unwrap();
    let logits = run_captured(
        model,
        actual,
        &tokens,
        None,
        CHUNK,
        true,
        Some(&mut captures),
    )
    .unwrap();
    let report = profile.finish();
    assert_eq!(logits, expected_logits);
    assert_eq!(captures.records, expected.records);
    same_state(model, actual, reference);
    for (role, count) in [("prefill_chunk_rows", chunks), ("gpu_step", 1)] {
        assert_eq!(
            report
                .operations
                .iter()
                .filter(|o| o.operation == role)
                .map(|o| o.completed_submits)
                .sum::<usize>(),
            count
        );
    }
    if let Some(scope) = scope {
        let observation = scope.finish();
        assert_eq!(observation.after_chunks.len(), chunks + 1);
        let max = observation
            .after_chunks
            .iter()
            .map(|&(_, _, bytes)| bytes)
            .max()
            .unwrap();
        let min = observation
            .after_chunks
            .iter()
            .map(|&(_, _, bytes)| bytes)
            .min()
            .unwrap();
        assert!(
            max - min <= DEVICE_RESIDUAL,
            "post-chunk device drift: {:?}",
            observation.after_chunks
        );
        eprintln!(
            "Full chunks={chunks} rows={CHUNK}+scalar-tail tracked_scratch_high_water={} sampled_device_high_water={} post_chunk_device={:?}",
            observation.resource_high_water,
            observation.device_high_water,
            observation.after_chunks
        );
    }
    let baseline = model.ctx.device.currentAllocatedSize() as u64;
    let mut peak = baseline;
    for i in 0..STEPS {
        let token = (i % 8) as u32;
        let mut traces = 0;
        let mut trace = |row: LayerTrace<'_>| {
            assert_eq!(row.position, tokens.len() + i);
            assert_eq!(row.layer, 0);
            assert!(
                row.hidden
                    .iter()
                    .chain(row.next_pre_mix)
                    .all(|v| v.is_finite())
            );
            traces += 1;
            Ok(())
        };
        assert_eq!(
            model
                .step_with_capture(actual, token, None, Some(&mut trace), Some(&mut captures))
                .unwrap(),
            model.step(reference, token, None).unwrap()
        );
        assert_eq!(traces, 1);
        peak = peak.max(model.ctx.device.currentAllocatedSize() as u64);
    }
    same_state(model, actual, reference);
    if observe {
        assert!(
            peak <= baseline + DEVICE_RESIDUAL,
            "256 completed steps retain device allocations: baseline={baseline} peak={peak}"
        );
        eprintln!(
            "Full teacher-forced steps={STEPS} post_step_device_baseline={baseline} post_step_device_high_water={peak}"
        );
    }
}

#[test]
#[ignore = "self-leased tiny Metal Full lifetime witness; MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1; no full model"]
fn full_chunks_and_scalar_steps_release_inside_one_outer_pool() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    assert!(!crate::env_flag::read_default_off(
        "QWEN_METAL_LEASE_SKIP_WIRED_GATE"
    ));
    let _lease =
        metal::acquire_metal_benchmark_lease().expect("production lease and ordinary wired gate");
    objc2::rc::autoreleasepool(|_| {
        let ctx = MetalContext::new().unwrap();
        let bound = explicit_buffer_bound(|bytes| {
            let priced = ctx
                .price_shared_buffer_upper(bytes)
                .unwrap()
                .priced_upper_bytes;
            assert!(priced <= bytes.max(1) + PAGE_SLACK);
            priced
        });
        assert!(bound < 32 * MIB, "explicit buffers {bound}");
        let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
            bound,
            16 * MIB,
            0,
            ctx.memory_signals(),
            true,
        );
        assert!(
            admission.is_admitted(),
            "tiny fixture ordinary admission: {admission:?}"
        );
        // Setup drains are outside the witness; no test-owned inner pools below.
        let weights = objc2::rc::autoreleasepool(|_| TinyWeights::new(&ctx));
        let model = weights.model(&ctx);
        assert_eq!(
            gpu_step::chunk_widths(&model, CAPACITY, CHUNK).unwrap(),
            chunk_widths(&model.config)
        );
        let (mut actual, mut reference) = objc2::rc::autoreleasepool(|_| {
            (
                model.new_state(CAPACITY).unwrap(),
                model.new_state(CAPACITY).unwrap(),
            )
        });
        let empty = model.capture_snapshot(&actual).unwrap();
        sequence(&model, &mut actual, &mut reference, 8, false);
        let warmed = ctx.device.currentAllocatedSize() as u64;
        for chunks in [2, 4, 8] {
            model.restore_snapshot(&mut actual, &empty).unwrap();
            model.restore_snapshot(&mut reference, &empty).unwrap();
            sequence(&model, &mut actual, &mut reference, chunks, true);
            let after = ctx.device.currentAllocatedSize() as u64;
            assert!(
                after <= warmed + DEVICE_RESIDUAL,
                "sequence device drift: warmed={warmed} after={after}"
            );
        }
        eprintln!(
            "explicit_buffer_bound={bound} CPU_admission={} warmed_device={warmed}; context/PSO/driver separate",
            16 * MIB
        );
        // Callback failure/unwind after checked completion must not publish the
        // failing chunk's addresses, including the genuine scalar final chunk.
        for (state, position, unwind) in [
            (&mut actual, CHUNK, false),
            (&mut reference, 2 * CHUNK, true),
        ] {
            model.restore_snapshot(state, &empty).unwrap();
            let mut expected_addresses = state.addresses;
            for _ in 0..position {
                expected_addresses = expected_addresses
                    .step_with_config(
                        engram::Token::Compressed(model.token_map[0]),
                        &model.engram_hash,
                    )
                    .unwrap()
                    .1;
            }
            let mut sink = Captures {
                fault: Some((position, unwind)),
                ..Captures::default()
            };
            let observation = Observe::begin();
            let result = catch_unwind(AssertUnwindSafe(|| {
                run_captured(
                    &model,
                    state,
                    &vec![0; 2 * CHUNK + 1],
                    None,
                    CHUNK,
                    true,
                    Some(&mut sink),
                )
            }));
            if unwind {
                assert!(result.is_err());
            } else {
                assert!(result.unwrap().is_err());
            }
            assert!(state.is_poisoned());
            assert_eq!(state.position(), position);
            assert_eq!(state.addresses, expected_addresses);
            assert!(
                observation
                    .finish()
                    .pending
                    .iter()
                    .all(|weak| weak.load().is_none()),
                "failed chunk scratch survives its pool"
            );
            assert!(
                (ctx.device.currentAllocatedSize() as u64) <= warmed + DEVICE_RESIDUAL,
                "failure/unwind retains completed resources"
            );
        }
    });
}
