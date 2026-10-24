use super::*;
use crate::deepseek_v41::forward::ced::{qualification, tests as incumbent};
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;

thread_local! {
    static LAYERS: Cell<Option<usize>> = const { Cell::new(None) };
    static RECORD: Cell<bool> = const { Cell::new(false) };
    static ARITHMETIC: RefCell<Option<Vec<metal::DispatchCensusRow>>> = const { RefCell::new(None) };
    static COPIES: RefCell<Vec<metal::DispatchCensusRow>> = const { RefCell::new(Vec::new()) };
    static BAD_ATTENTION: Cell<bool> = const { Cell::new(false) };
    static STATUSES: RefCell<Vec<(Unit, Vec<u32>, u32)>> = const { RefCell::new(Vec::new()) };
    static CONTROLS: RefCell<Vec<(usize, Range<usize>, Vec<u32>, Vec<u32>)>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn override_layers() -> Option<usize> {
    LAYERS.with(Cell::get)
}

pub(in super::super::super) struct Scope(Option<usize>, PhantomData<Rc<()>>);
impl Scope {
    pub(in super::super::super) fn new(layers: usize) -> Self {
        assert!([0, 1, 2, 4].contains(&layers));
        Self(LAYERS.with(|v| v.replace(Some(layers))), PhantomData)
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        LAYERS.with(|v| v.set(self.0));
    }
}

pub(super) struct Diagnostics(
    Vec<(Unit, attention_gpu::QueryControlView)>,
    Option<MetalTensor>,
);
impl Diagnostics {
    pub(super) fn new(model: &Forward<'_>, storage: &Storage, units: &[Unit]) -> Result<Self> {
        Ok(Self(
            units
                .iter()
                .filter(|u| u.kind != Kind::Seed)
                .map(|unit| {
                    let view = storage
                        .controls
                        .checked_tile_view(unit.rows.start, unit.rows.len())?
                        .diagnostic_copy(model.ctx)?;
                    Ok((unit.clone(), view))
                })
                .collect::<Result<_>>()?,
            if BAD_ATTENTION.with(Cell::get) {
                Some(upload(model.ctx, &[f32::NAN], vec![1])?)
            } else {
                None
            },
        ))
    }
    pub(super) fn fault(
        &mut self,
        attention: &attention_gpu::Workspace,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        _unit: &Unit,
    ) -> Result<()> {
        if BAD_ATTENTION.with(|v| v.replace(false)) {
            attention.encode_ced_test_nonfinite(ctx, enc, self.1.as_ref().unwrap())?;
        }
        Ok(())
    }
    pub(super) fn encode(
        &mut self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        storage: &Storage,
        unit: &Unit,
    ) -> Result<()> {
        if unit.kind != Kind::Seed {
            let (_, snapshot) = self
                .0
                .iter()
                .find(|(u, _)| u == unit)
                .ok_or("CED diagnostic unit missing")?;
            snapshot.encode_diagnostic_copy(
                ctx,
                enc,
                &storage
                    .controls
                    .checked_tile_view(unit.rows.start, unit.rows.len())?,
            )?;
        }
        Ok(())
    }
    pub(super) fn complete(
        &self,
        attention: &attention_gpu::Workspace,
        n: usize,
        units: &[Unit],
        cancel: Option<&AtomicBool>,
    ) -> Result<()> {
        for unit in units {
            if unit.kind == Kind::Seed {
                incumbent::checkpoint("ced_seed_complete", cancel)?;
            } else {
                let (_, snapshot) = self.0.iter().find(|(u, _)| u == unit).unwrap();
                qualification::controls(unit.layer, unit.rows.start, unit.rows.len(), snapshot)?;
                record(unit.layer, unit.rows.clone(), snapshot);
                incumbent::checkpoint("ced_decoder_complete", cancel)?;
                if unit.rows.end == n {
                    qualification::decoder_layer(attention, unit.layer, n)?;
                }
            }
        }
        Ok(())
    }
}

pub(super) fn before_copies() {
    ARITHMETIC.with(|v| {
        if let Some(rows) = v.borrow_mut().as_mut() {
            rows.extend(metal::dispatch_census_take());
            metal::dispatch_census_begin();
        }
    });
}

pub(super) fn after_copies() {
    if ARITHMETIC.with(|v| v.borrow().is_some()) {
        COPIES.with(|v| v.borrow_mut().extend(metal::dispatch_census_take()));
        metal::dispatch_census_begin();
    }
}

pub(super) fn completed_statuses(units: &[Unit], statuses: &[StatusUnit]) {
    if RECORD.with(Cell::get) {
        STATUSES.with(|v| {
            v.borrow_mut()
                .extend(units.iter().zip(statuses).map(|(unit, status)| {
                    (
                        unit.clone(),
                        status.ffn.as_ref().map(bits).unwrap_or_default(),
                        bits(&status.attention)[0],
                    )
                }))
        });
    }
}

struct Capture;
impl Capture {
    fn begin() -> Self {
        RECORD.with(|v| assert!(!v.replace(true)));
        ARITHMETIC.with(|v| assert!(v.replace(Some(Vec::new())).is_none()));
        CONTROLS.with(|v| v.borrow_mut().clear());
        COPIES.with(|v| v.borrow_mut().clear());
        STATUSES.with(|v| v.borrow_mut().clear());
        metal::dispatch_census_begin();
        Self
    }
    fn finish(&self) -> (Vec<metal::DispatchCensusRow>, Vec<metal::DispatchCensusRow>) {
        before_copies();
        (
            ARITHMETIC.with(|v| v.take().unwrap()),
            COPIES.with(|v| v.take()),
        )
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        let _ = metal::dispatch_census_take();
        RECORD.with(|v| v.set(false));
        ARITHMETIC.with(|v| v.take());
        BAD_ATTENTION.with(|v| v.set(false));
    }
}

pub(in super::super::super) fn record(
    layer: usize,
    range: Range<usize>,
    view: &attention_gpu::QueryControlView,
) {
    if RECORD.with(Cell::get) {
        CONTROLS.with(|v| {
            v.borrow_mut()
                .push((layer, range, bits(view.selected()), bits(view.candidates())))
        });
    }
}

fn bits(t: &MetalTensor) -> Vec<u32> {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u32>()
                .add((t.offset / 4) as usize),
            t.n_elements() as usize,
        )
        .to_vec()
    }
}

#[test]
fn batch_override_is_nested_thread_local_and_unwind_safe() {
    assert_eq!(batch_layers(), 2);
    let outer = Scope::new(4);
    {
        let _inner = Scope::new(1);
        assert_eq!(batch_layers(), 1);
    }
    assert_eq!(batch_layers(), 4);
    std::thread::spawn(|| assert_eq!(batch_layers(), 2))
        .join()
        .unwrap();
    assert!(
        std::panic::catch_unwind(|| {
            let _inner = Scope::new(0);
            panic!("restore");
        })
        .is_err()
    );
    drop(outer);
    assert_eq!(batch_layers(), 2);
}

fn configure_four(model: &mut Forward<'_>, window: usize) {
    reuse::configure(model, window);
    model.layers[2].attention = model.layers[3].attention.clone();
    model.layers[3].attention.compressor = None;
    model.layers[3].attention.index = None;
    model.schedule = crate::deepseek_v41::gguf::Schedule {
        layers: 6,
        kv_source_layers: vec![1, 2],
        index_source_layers: vec![1, 2, 4],
        compression_ratios: vec![0, 2, 1, 1, 1, 1],
        candidate_source_layer: Some(2),
    };
    model.schedule.validate().unwrap();
}

fn plan(model: &Forward<'_>, n: usize) -> CedPlan {
    CedPlan::shrinking(
        &model.config,
        &model.schedule,
        &model.engram_hash,
        &crate::deepseek_v41::forward::ced_plan::CedRequest::cold(n, n),
    )
    .unwrap()
}

fn bound(
    c: &Config,
    schedule: &crate::deepseek_v41::gguf::Schedule,
    n: usize,
    price: &impl Fn(u64) -> u64,
) -> u64 {
    let hash = engram::HashConfig {
        layers: Vec::new(),
        ..engram::HashConfig::default()
    };
    let p = CedPlan::shrinking(
        c,
        schedule,
        &hash,
        &crate::deepseek_v41::forward::ced_plan::CedRequest::cold(n, n),
    )
    .unwrap();
    let memory = crate::deepseek_v41::forward::ced::estimate_from_parts(
        c,
        schedule.layers,
        &[],
        0,
        n,
        &p,
        true,
        &|n| Ok(if n == 0 { 0 } else { price(n) }),
    )
    .unwrap();
    reuse::weight_bytes(price)
        + 2 * super::super::super::bytes_with_schedule(c, n, schedule, 0).unwrap()
        + memory.incremental_metal_bytes
}

fn admit(model: &Forward<'_>, n: usize) {
    let _depth = Scope::new(4);
    let bytes = bound(&model.config, &model.schedule, n, &|n| {
        model
            .ctx
            .price_shared_buffer_upper(n)
            .unwrap()
            .priced_upper_bytes
    });
    assert!(
        bytes <= 128 * 1024 * 1024,
        "complete batch fixture ledger {bytes}"
    );
    let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
        bytes,
        32 * 1024 * 1024,
        0,
        model.ctx.memory_signals(),
        true,
    );
    assert!(admission.refusal().is_none(), "{:?}", admission.refusal());
}

type Dispatch = (String, [u64; 6], bool);
fn dispatches(rows: Vec<metal::DispatchCensusRow>) -> Vec<Dispatch> {
    rows.into_iter()
        .map(|r| {
            (
                r.kernel,
                [
                    r.grid_width,
                    r.grid_height,
                    r.grid_depth,
                    r.threads_width,
                    r.threads_height,
                    r.threads_depth,
                ],
                r.encoder_concurrent,
            )
        })
        .collect()
}

struct Arm {
    head: Vec<u32>,
    causal: Vec<Vec<u32>>,
    controls: Vec<(usize, Range<usize>, Vec<u32>, Vec<u32>)>,
    counters: CedCounters,
    arithmetic: Vec<Dispatch>,
    copies: Vec<Dispatch>,
}

fn arm(model: &Forward<'_>, n: usize, depth: usize) -> Arm {
    let _depth = Scope::new(depth);
    let capture = Capture::begin();
    let mut state = model.new_state(n).unwrap();
    let (head, report) = crate::deepseek_v41::forward::ced::run(
        model,
        &mut state,
        &incumbent::tokens(n),
        None,
        Arithmetic::Production,
    )
    .unwrap();
    assert!(state.is_ready() && state.position() == n && report.fallback.is_none());
    assert!(
        report.observed_incremental_metal_bytes.unwrap()
            <= report.memory.as_ref().unwrap().incremental_metal_bytes
    );
    if depth != 0 {
        eprintln!(
            "batch_memory N={n} depth={depth} {}",
            serde_json::to_string(report.memory.as_ref().unwrap()).unwrap()
        );
    }
    let causal = incumbent::causal_bits(&state)
        .into_iter()
        .map(|v| v.into_iter().map(f32::to_bits).collect())
        .collect();
    let (arithmetic, copies) = capture.finish();
    let counters = report.counters.unwrap();
    counters.validate(&plan(model, n)).unwrap();
    if depth != 0 {
        assert_eq!(
            counters.completed_commands,
            n.div_ceil(prefill::DEFAULT_CHUNK_ROWS)
                + model
                    .schedule
                    .layers
                    .saturating_sub(model.schedule.candidate_source_layer.unwrap())
                    .div_ceil(depth)
        );
    }
    Arm {
        head: head.into_iter().map(f32::to_bits).collect(),
        causal,
        controls: CONTROLS.with(|v| v.take()),
        counters,
        arithmetic: dispatches(arithmetic),
        copies: dispatches(copies),
    }
}

fn assert_pair(actual: &Arm, reference: &Arm, model: &Forward<'_>, n: usize) {
    assert_eq!(actual.head, reference.head, "all head bits N={n}");
    assert_eq!(
        actual.causal, reference.causal,
        "all causal state bits N={n}"
    );
    assert_eq!(
        actual.controls, reference.controls,
        "all absolute query controls N={n}"
    );
    assert_eq!(
        actual.arithmetic, reference.arithmetic,
        "ordered arithmetic PSOs/grids/serial encoders N={n}"
    );
    let mut expected = reference.counters.clone();
    expected.completed_commands = actual.counters.completed_commands;
    expected.encoded_commands = actual.counters.encoded_commands;
    assert_eq!(
        actual.counters, expected,
        "only actual command retirements change"
    );
    let p = plan(model, n);
    let mut units = Vec::new();
    for g in p.decoder.chunks(4) {
        units.extend(plan_batch::units(&p, g[0].layer..g.last().unwrap().layer + 1).unwrap());
    }
    let outputs = units.iter().filter(|u| u.kind != Kind::Seed).count();
    assert_eq!(
        actual.copies.len(),
        units.len() + 2 * outputs,
        "one sticky snapshot/unit and two test-only control snapshots/output"
    );
    assert!(
        actual
            .copies
            .iter()
            .all(|r| r.0 == "kernel_deepseek_v41_attention_copy" && !r.2)
    );
    eprintln!(
        "N={n} reference_commands={} batch_commands={} work_rows={}",
        reference.counters.completed_commands,
        actual.counters.completed_commands,
        p.work.total_output_rows
    );
}

#[test]
#[ignore = "self-leased M4/M5 <=128 MiB complete buffers; MTL debug; no real model"]
fn bounded_old_vs_batch_1_2_4_exact_heads_causal_controls_ordered_arithmetic_chunks_and_tails() {
    let _lease = reuse::require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    objc2::rc::autoreleasepool(|_| {
        let weights = crate::deepseek_v41::forward::toy::Weights::new(&ctx);
        let embedding = incumbent::embeddings(&ctx);
        let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
        configure_four(&mut model, 3);
        prefill::with_batched_f32(|| {
            for (window, n) in [
                (3, 1),
                (3, 127),
                (3, 128),
                (3, 129),
                (3, 2047),
                (3, 2048),
                (3, 2049),
                (257, 640),
                (2050, 2050),
            ] {
                reuse::configure(&mut model, window);
                if window == 2050 {
                    model.layers.truncate(4);
                    model.schedule.layers = 4;
                    model.schedule.compression_ratios.truncate(4);
                    model.schedule.index_source_layers.retain(|&id| id < 4);
                    let p = plan(&model, n);
                    assert!(
                        p.decoder
                            .iter()
                            .any(|l| l.seed_rows.len() > prefill::DEFAULT_CHUNK_ROWS)
                    );
                    assert!(
                        p.decoder
                            .iter()
                            .any(|l| l.output_rows.len() > prefill::DEFAULT_CHUNK_ROWS)
                    );
                }
                admit(&model, n);
                let reference = objc2::rc::autoreleasepool(|_| arm(&model, n, 0));
                assert!(reference.copies.is_empty());
                for depth in [1, 2, 4] {
                    let actual = objc2::rc::autoreleasepool(|_| arm(&model, n, depth));
                    assert_pair(&actual, &reference, &model, n);
                }
            }
        });
    });
}

#[test]
fn all_new_fixture_ledgers_include_four_layer_status_and_diagnostic_storage_below_128_mib() {
    let _depth = Scope::new(4);
    let mut schedule = crate::deepseek_v41::gguf::Schedule {
        layers: 6,
        kv_source_layers: vec![1, 2],
        index_source_layers: vec![1, 2, 4],
        compression_ratios: vec![0, 2, 1, 1, 1, 1],
        candidate_source_layer: Some(2),
    };
    for (window, n) in [
        (3, 1),
        (3, 127),
        (3, 128),
        (3, 129),
        (3, 2047),
        (3, 2048),
        (3, 2049),
        (257, 640),
        (2050, 2050),
    ] {
        let c = Config {
            window,
            index_topk: 2,
            ..crate::deepseek_v41::forward::toy::config()
        };
        if window == 2050 {
            schedule.layers = 4;
            schedule.compression_ratios.truncate(4);
            schedule.index_source_layers.retain(|&id| id < 4);
        }
        let bytes = bound(&c, &schedule, n, &|n| n + 16384);
        assert!(
            bytes <= 128 * 1024 * 1024,
            "W={window} N={n} full fixture ledger {bytes}"
        );
    }
    assert!(reuse::wide_bound(&|n| n + 16384) <= 128 * 1024 * 1024);
}

#[test]
#[ignore = "self-leased <=128 MiB complete buffers; MTL debug; no real model"]
fn early_ffn_and_attention_errors_survive_later_valid_units_with_incumbent_precedence() {
    let _lease = reuse::require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    objc2::rc::autoreleasepool(|_| {
        let weights = crate::deepseek_v41::forward::toy::Weights::new(&ctx);
        let embedding = incumbent::embeddings(&ctx);
        let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
        configure_four(&mut model, 65);
        admit(&model, 640);
        prefill::with_batched_f32(|| {
            objc2::rc::autoreleasepool(|_| {
                let _ = arm(&model, 640, 4);
            });
            let baseline = ctx.current_allocated_size();
            for (bad_attention, bad_ffn) in [(false, true), (true, false), (true, true)] {
                let _depth = Scope::new(4);
                model.layers[2].bias[0] = if bad_ffn { f32::NAN } else { 0.0 };
                // A second failure cannot outrank the first logical unit.
                model.layers[4].bias[0] = if bad_ffn { f32::NAN } else { 0.0 };
                objc2::rc::autoreleasepool(|_| {
                    let capture = Capture::begin();
                    BAD_ATTENTION.with(|v| v.set(bad_attention));
                    let mut state = model.new_state(640).unwrap();
                    let addresses = state.addresses;
                    let p = plan(&model, 640);
                    let error = crate::deepseek_v41::forward::ced::run(
                        &model,
                        &mut state,
                        &incumbent::tokens(640),
                        None,
                        Arithmetic::Production,
                    )
                    .unwrap_err();
                    let statuses = STATUSES.with(|v| v.take());
                    assert!(
                        statuses.iter().any(|(u, ffn, _)| u.layer == 3
                            && u.kind == Kind::Rows
                            && ffn.iter().all(|&s| s == 1)),
                        "later valid route writes executed"
                    );
                    assert!(
                        statuses.iter().any(|(u, _, _)| u.layer == 5),
                        "entire submitted batch drained"
                    );
                    if bad_ffn {
                        assert!(
                            error.to_string().contains("layer 2 row 0 route status"),
                            "{error}"
                        );
                        assert!(
                            statuses
                                .iter()
                                .any(|(u, ffn, _)| u.layer == 2 && ffn.iter().any(|&s| s != 1))
                        );
                    } else {
                        assert!(
                            error.to_string().contains("CED attention status"),
                            "{error}"
                        );
                    }
                    if bad_attention {
                        assert!(
                            statuses.iter().all(|(_, _, s)| s & 4 != 0),
                            "global finite error stays sticky through seed/rows/layers"
                        );
                    }
                    assert_eq!(
                        state
                            .gpu_step
                            .as_ref()
                            .unwrap()
                            .attention
                            .ced_public_decoder_progress(),
                        (
                            0,
                            p.decoder.iter().map(|l| l.seed_rows.start).collect(),
                            p.decoder.iter().map(|l| l.output_rows.start).collect()
                        )
                    );
                    incumbent::assert_poisoned_rejects(&model, &mut state, addresses);
                    drop(capture);
                });
                assert!(
                    ctx.current_allocated_size() <= baseline,
                    "all request resources return to the warmed baseline"
                );
            }
        });
    });
}

#[test]
#[ignore = "self-leased <=128 MiB complete buffers; MTL debug; no real model"]
fn batch_encode_completion_cancel_and_unwind_poison_without_publication_or_resource_leaks() {
    use incumbent::Action;
    let _lease = reuse::require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    objc2::rc::autoreleasepool(|_| {
        let weights = crate::deepseek_v41::forward::toy::Weights::new(&ctx);
        let embedding = incumbent::embeddings(&ctx);
        let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
        configure_four(&mut model, 65);
        admit(&model, 640);
        prefill::with_batched_f32(|| {
            objc2::rc::autoreleasepool(|_| {
                let _ = arm(&model, 640, 2);
            });
            let baseline = ctx.current_allocated_size();
            for (role, action) in [
                ("ced_batch_unit_encoded", Action::Error),
                ("ced_batch_unit_encoded", Action::Cancel),
                ("ced_batch_unit_encoded", Action::Panic),
                ("ced_decoder_batch", Action::Completion),
                ("ced_batch_completed", Action::Error),
                ("ced_batch_completed", Action::Cancel),
                ("ced_batch_completed", Action::Panic),
            ] {
                let _depth = Scope::new(2);
                objc2::rc::autoreleasepool(|_| {
                    let mut state = model.new_state(640).unwrap();
                    let addresses = state.addresses;
                    let cancel = AtomicBool::new(false);
                    let fault = incumbent::Scope::new(role, action);
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        crate::deepseek_v41::forward::ced::run(
                            &model,
                            &mut state,
                            &incumbent::tokens(640),
                            Some(&cancel),
                            Arithmetic::Production,
                        )
                    }));
                    assert_eq!(fault.hits(), 1, "{role}");
                    if action == Action::Panic {
                        assert!(result.is_err());
                    } else {
                        assert!(result.unwrap().is_err());
                    }
                    assert_eq!(
                        state.gpu_step.as_ref().unwrap().commands,
                        if role == "ced_batch_unit_encoded" {
                            1
                        } else {
                            2
                        },
                        "no future decoder group is submitted after {role}"
                    );
                    assert_eq!(
                        state
                            .gpu_step
                            .as_ref()
                            .unwrap()
                            .attention
                            .ced_public_decoder_progress()
                            .0,
                        0
                    );
                    incumbent::assert_poisoned_rejects(&model, &mut state, addresses);
                });
                assert!(
                    ctx.current_allocated_size() <= baseline,
                    "{role}: resources return to baseline after draining"
                );
            }
        });
    });
}

#[test]
#[ignore = "self-leased M5 required; <=128 MiB explicit buffers; MTL debug; no real model"]
fn m5_small_and_wide_batch_lineage_preserves_full_tile_packed_singletons() {
    let _planning = Scope::new(4);
    let _lease = reuse::require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    assert!(
        metal::moe_grouped_tensor_selected(&ctx),
        "M5 tensor tier required, not M4 coverage"
    );
    for wide in [false, true] {
        objc2::rc::autoreleasepool(|_| {
            let weights = crate::deepseek_v41::forward::toy::Weights::new(&ctx);
            let embedding = incumbent::embeddings(&ctx);
            let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
            if wide {
                reuse::configure_wide(&mut model);
            } else {
                reuse::configure(&mut model, 65);
            }
            let bytes = if wide {
                reuse::wide_bound(&|n| ctx.price_shared_buffer_upper(n).unwrap().priced_upper_bytes)
            } else {
                reuse::admit(&ctx, &model.config, 129, 129)
            };
            assert!(bytes <= 128 * 1024 * 1024);
            let reference = objc2::rc::autoreleasepool(|_| arm(&model, 129, 0));
            for depth in [1, 2, 4] {
                let actual = objc2::rc::autoreleasepool(|_| arm(&model, 129, depth));
                assert_pair(&actual, &reference, &model, 129);
                if wide {
                    let score =
                        if std::env::var_os("QWEN_V41_SINK_TENSOR32").is_none_or(|v| v != "0") {
                            "score_tensor32"
                        } else {
                            "score_tensor"
                        };
                    for suffix in [score, "pv_tensor"] {
                        assert!(
                            actual
                                .arithmetic
                                .iter()
                                .any(|r| r.0 == format!("kernel_deepseek_v41_attention_{suffix}")
                                    && r.1[1] == 1),
                            "original FullTile tensor singleton lineage"
                        );
                    }
                    assert!(
                        actual.arithmetic.iter().any(|r| r.0.contains("f16act")),
                        "half-stage overwrite exercised"
                    );
                }
            }
        });
    }
}
