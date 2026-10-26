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
    static FFN_FAULT: Cell<Option<FfnFault>> = const { Cell::new(None) };
    static FFN_HITS: Cell<usize> = const { Cell::new(0) };
    static BAD_CANARY: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION: RefCell<Option<(AllocationSite, Rc<Cell<usize>>)>> = const { RefCell::new(None) };
    static CANARY_WORDS_CHECKED: Cell<usize> = const { Cell::new(0) };
    static PUBLIC_PROGRESS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    static RETIRED_PROGRESS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    static STATUSES: RefCell<Vec<(Unit, Vec<u32>, u32)>> = const { RefCell::new(Vec::new()) };
    static CONTROLS: RefCell<Vec<(usize, Range<usize>, Vec<u32>, Vec<u32>)>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super::super) enum AllocationSite {
    Ledger,
    SeedNormalized,
    DecoderPool,
}

pub(in super::super::super) fn allocation_checkpoint(
    site: AllocationSite,
    bytes: usize,
) -> std::result::Result<(), metal::MetalError> {
    let fail = ALLOCATION.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.as_ref().is_some_and(|(expected, _)| *expected == site) {
            let (_, hits) = slot.take().unwrap();
            hits.set(hits.get() + 1);
            true
        } else {
            false
        }
    });
    if fail {
        Err(metal::MetalError::NoBuffer(bytes))
    } else {
        Ok(())
    }
}

struct AllocationFault {
    previous: Option<(AllocationSite, Rc<Cell<usize>>)>,
    hits: Rc<Cell<usize>>,
}
impl AllocationFault {
    fn new(site: AllocationSite) -> Self {
        let hits = Rc::new(Cell::new(0));
        let previous = ALLOCATION.with(|slot| slot.replace(Some((site, hits.clone()))));
        Self { previous, hits }
    }
}
impl Drop for AllocationFault {
    fn drop(&mut self) {
        ALLOCATION.with(|slot| slot.replace(self.previous.take()));
    }
}

#[derive(Clone, Copy, Debug)]
struct FfnFault {
    layer: usize,
    start: usize,
    row: usize,
    word: i32,
}

struct FfnFaultScope(Option<FfnFault>, usize, PhantomData<Rc<()>>);
impl FfnFaultScope {
    fn new(fault: FfnFault) -> Self {
        Self(
            FFN_FAULT.with(|v| v.replace(Some(fault))),
            FFN_HITS.with(|v| v.replace(0)),
            PhantomData,
        )
    }
}
impl Drop for FfnFaultScope {
    fn drop(&mut self) {
        FFN_FAULT.with(|v| v.set(self.0));
        FFN_HITS.with(|v| v.set(self.1));
    }
}

const CANARY_BITS: u32 = 0x4b1af03d;

pub(super) fn initialize_status_canaries(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    ledger: &MetalTensor,
) -> Result<()> {
    before_copies();
    let mut floats = ledger.clone();
    floats.dtype = GgmlType::F32;
    metal::encode_fill_f32(ctx, enc, &floats, f32::from_bits(CANARY_BITS))
        .map_err(|e| e.to_string())?;
    after_copies();
    Ok(())
}

pub(super) fn check_status_canaries(statuses: &[StatusUnit]) -> Result<()> {
    for status in statuses {
        for guard in &status.canaries {
            let words = bits(guard);
            op::require(
                words.iter().all(|&word| word == CANARY_BITS),
                "CED GPU status ledger canary overwritten",
            )?;
            CANARY_WORDS_CHECKED.with(|v| v.set(v.get() + words.len()));
        }
    }
    Ok(())
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
            FFN_FAULT
                .with(|v| v.get().map(|fault| fault.word))
                .or_else(|| BAD_CANARY.with(|v| v.get().then_some(0)))
                .map(|word| {
                    MetalTensor::from_bytes(model.ctx, &word.to_le_bytes(), vec![1], GgmlType::I32)
                        .map_err(|e| e.to_string())
                })
                .transpose()?,
        ))
    }
    pub(super) fn fault(
        &mut self,
        attention: &attention_gpu::Workspace,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        unit: &Unit,
        status: &StatusUnit,
    ) -> Result<()> {
        if BAD_ATTENTION.with(|v| v.replace(false)) {
            attention.encode_ced_test_nonfinite(ctx, enc, self.1.as_ref().unwrap())?;
        }
        if let Some(fault) = FFN_FAULT.with(Cell::get)
            && fault.layer == unit.layer
            && fault.start == unit.rows.start
            && unit.kind == Kind::Rows
        {
            let ffn = status
                .ffn
                .as_ref()
                .ok_or("CED injected FFN status missing")?;
            op::require(
                fault.row < unit.rows.len(),
                "CED injected FFN row outside unit",
            )?;
            let target = ffn.view_subrange(fault.row as u64, vec![1]);
            metal::encode_copy_offset_i32(
                ctx,
                enc,
                self.2.as_ref().ok_or("CED injected FFN source missing")?,
                0,
                &target,
                1,
            )
            .map_err(|e| e.to_string())?;
            FFN_FAULT.with(|v| v.set(None));
            FFN_HITS.with(|v| v.set(v.get() + 1));
        }
        if BAD_CANARY.with(|v| v.replace(false)) {
            let target = status
                .canaries
                .last()
                .ok_or("CED guard fault missing boundary")?
                .view_subrange(0, vec![1]);
            metal::encode_copy_offset_i32(
                ctx,
                enc,
                self.2.as_ref().ok_or("CED guard fault source missing")?,
                0,
                &target,
                1,
            )
            .map_err(|e| e.to_string())?;
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

pub(super) fn completed_statuses(
    attention: &attention_gpu::Workspace,
    units: &[Unit],
    statuses: &[StatusUnit],
) {
    if RECORD.with(Cell::get) {
        PUBLIC_PROGRESS.with(|v| {
            v.borrow_mut()
                .push(attention.ced_public_decoder_progress().unwrap().0)
        });
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

pub(super) fn group_retired(attention: &attention_gpu::Workspace) {
    if RECORD.with(Cell::get) {
        RETIRED_PROGRESS.with(|v| {
            v.borrow_mut()
                .push(attention.ced_public_decoder_progress().unwrap().0)
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
        CANARY_WORDS_CHECKED.with(|v| v.set(0));
        PUBLIC_PROGRESS.with(|v| v.borrow_mut().clear());
        RETIRED_PROGRESS.with(|v| v.borrow_mut().clear());
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
        BAD_CANARY.with(|v| v.set(false));
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

#[test]
fn typed_allocation_failure_sites_are_one_shot_and_do_not_request_real_oom() {
    for site in [
        AllocationSite::Ledger,
        AllocationSite::SeedNormalized,
        AllocationSite::DecoderPool,
    ] {
        let fault = AllocationFault::new(site);
        std::thread::spawn(|| assert!(allocation_checkpoint(AllocationSite::Ledger, 19).is_ok()))
            .join()
            .unwrap();
        assert!(matches!(
            allocation_checkpoint(site, 37),
            Err(metal::MetalError::NoBuffer(37))
        ));
        assert_eq!(fault.hits.get(), 1);
        assert!(allocation_checkpoint(site, 37).is_ok());
    }
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
        &crate::deepseek_v41::forward::ced_plan::CedRequest::cold(n, n)
            .with_full_chunk_rows(model.prefill_rows),
    )
    .unwrap()
}

fn bound(
    c: &Config,
    schedule: &crate::deepseek_v41::gguf::Schedule,
    n: usize,
    width: usize,
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
        &crate::deepseek_v41::forward::ced_plan::CedRequest::cold(n, n).with_full_chunk_rows(width),
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
    let bytes = bound(
        &model.config,
        &model.schedule,
        n,
        model.prefill_rows,
        &|n| {
            model
                .ctx
                .price_shared_buffer_upper(n)
                .unwrap()
                .priced_upper_bytes
        },
    );
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
    groups: usize,
    memory: crate::deepseek_v41::forward::ced::CedMemoryEstimate,
    profile: profile::Report,
}

fn arm(model: &Forward<'_>, n: usize, depth: usize) -> Arm {
    let _depth = Scope::new(depth);
    let capture = Capture::begin();
    let mut state = model.new_state(n).unwrap();
    let profiling = profile::Scope::begin().unwrap();
    let (head, report) = crate::deepseek_v41::forward::ced::run(
        model,
        &mut state,
        &incumbent::tokens(n),
        None,
        Arithmetic::Production,
    )
    .unwrap();
    assert!(state.is_ready() && state.position() == n && report.fallback.is_none());
    assert_eq!(
        state.prefill_origin(),
        crate::deepseek_v41::forward::PrefillExecution::ShrinkingCed
    );
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
    let p = plan(model, n);
    counters.validate(&p).unwrap();
    let selected = report.memory.as_ref().unwrap().decoder_batch_layers;
    if selected != 0 {
        assert_eq!(
            counters.completed_commands,
            n.div_ceil(p.full_chunk_rows)
                + model
                    .schedule
                    .layers
                    .saturating_sub(model.schedule.candidate_source_layer.unwrap())
                    .div_ceil(selected)
        );
    }
    let groups = if selected == 0 {
        0
    } else {
        counters.completed_commands - n.div_ceil(p.full_chunk_rows)
    };
    let profile = profiling.finish();
    assert_eq!(
        profile
            .operations
            .iter()
            .map(|op| op.completed_submits)
            .sum::<usize>(),
        counters.completed_commands
    );
    for op in profile
        .operations
        .iter()
        .filter(|op| op.completed_submits != 0)
    {
        assert!(op.invalid_driver_scheduling_timestamps <= op.completed_submits);
        if let Some(ms) = op.driver_scheduling_ms {
            assert!(ms.is_finite() && ms > 0.0);
            assert!(op.invalid_driver_scheduling_timestamps < op.completed_submits);
        } else {
            assert_eq!(
                op.invalid_driver_scheduling_timestamps, op.completed_submits,
                "every command must sample or explicitly report unavailable driver timestamps"
            );
        }
    }
    Arm {
        head: head.into_iter().map(f32::to_bits).collect(),
        causal,
        controls: CONTROLS.with(|v| v.take()),
        counters,
        arithmetic: dispatches(arithmetic),
        copies: dispatches(copies),
        groups,
        memory: report.memory.unwrap(),
        profile,
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
        units.len() + 2 * outputs + actual.groups,
        "one sticky snapshot/unit, two test-only control snapshots/output, one GPU canary initialization/group"
    );
    assert!(
        actual
            .copies
            .iter()
            .all(
                |r| (r.0 == "kernel_deepseek_v41_attention_copy" || r.0 == "kernel_fill_f32")
                    && !r.2
            )
    );
    assert_eq!(
        actual
            .copies
            .iter()
            .filter(|r| r.0 == "kernel_fill_f32")
            .count(),
        actual.groups
    );
    eprintln!(
        "N={n} reference_commands={} batch_commands={} work_rows={}",
        reference.counters.completed_commands,
        actual.counters.completed_commands,
        p.work.total_output_rows
    );
}

fn command_plan(report: &profile::Report) -> Vec<(&'static str, [usize; 4], usize)> {
    let mut commands = report
        .operations
        .iter()
        .filter(|op| op.completed_submits != 0)
        .map(|op| (op.operation, op.dimensions, op.completed_submits))
        .collect::<Vec<_>>();
    commands.sort_unstable();
    commands
}

#[test]
#[ignore = "self-leased <=128 MiB complete buffers; automatic structural fallback; MTL debug"]
fn automatic_unit_budget_fallback_matches_incumbent_counts_prices_and_all_bits() {
    let _lease = reuse::require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    objc2::rc::autoreleasepool(|_| {
        let weights = crate::deepseek_v41::forward::toy::Weights::new(&ctx);
        let embedding = incumbent::embeddings(&ctx);
        let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
        configure_four(&mut model, 65);
        prefill::with_batched_f32(|| {
            for width in [2048, 4096] {
                model.prefill_rows = width;
                let n = 2049;
                admit(&model, n);
                let p = plan(&model, n);
                let reference = objc2::rc::autoreleasepool(|_| arm(&model, n, 0));
                let _limit = plan_batch::UnitLimitScope::new(1);
                assert_eq!(
                    batch_layers(),
                    2,
                    "the fallback does not force the incumbent override"
                );
                assert!(p.decoder.chunks(2).any(|g| {
                    plan_batch::unit_count(&p, g[0].layer..g.last().unwrap().layer + 1).unwrap() > 1
                }));
                let ledger = AllocationFault::new(AllocationSite::Ledger);
                let actual = objc2::rc::autoreleasepool(|_| arm(&model, n, 2));
                assert_eq!(
                    ledger.hits.get(),
                    0,
                    "an automatically unsubmitted ledger is never allocated"
                );
                assert_eq!(actual.memory.decoder_batch_layers, 0);
                assert!(
                    actual
                        .memory
                        .decoder_schedule_reason
                        .unwrap()
                        .contains("physical_unit_budget")
                );
                assert_eq!(actual.memory.full_chunk_rows, width);
                assert_eq!(actual.memory.priced_batch_status_bytes, 0);
                assert_eq!(actual.memory.batch_cpu_bytes, 0);
                assert_eq!(actual.memory.priced_batch_diagnostic_bytes, 0);
                assert_eq!(actual.memory.batch_encoding_reserve_bytes, 0);
                assert_eq!(actual.memory.command_reserve_bytes, 16 * 1024 * 1024);
                assert_eq!(actual.groups, 0);
                assert!(actual.copies.is_empty());
                assert_eq!(actual.head, reference.head);
                assert_eq!(actual.causal, reference.causal);
                assert_eq!(actual.controls, reference.controls);
                assert_eq!(actual.arithmetic, reference.arithmetic);
                assert_eq!(actual.counters, reference.counters);
                assert_eq!(
                    command_plan(&actual.profile),
                    command_plan(&reference.profile)
                );
                assert!(
                    !actual
                        .profile
                        .operations
                        .iter()
                        .any(|op| op.operation == "ced_decoder_batch")
                );
                let tail = usize::from(p.scalar_tail().is_some());
                let expected = n.div_ceil(width)
                    + p.decoder
                        .iter()
                        .map(|l| {
                            l.seed_rows.len().div_ceil(prefill::DEFAULT_CHUNK_ROWS)
                                + (l.output_rows.len() - tail).div_ceil(prefill::DEFAULT_CHUNK_ROWS)
                                + tail
                                - usize::from(l.fuse_seed_with_packed_output(n, width))
                        })
                        .sum::<usize>();
                assert_eq!(actual.counters.completed_commands, expected);
                assert_eq!(
                    qualification::production_command_count(&p).unwrap(),
                    expected
                );
                let mut actual_memory = actual.memory.clone();
                let mut reference_memory = reference.memory.clone();
                actual_memory.process_ru_maxrss_before_bytes = None;
                actual_memory.process_ru_maxrss_after_bytes = None;
                reference_memory.process_ru_maxrss_before_bytes = None;
                reference_memory.process_ru_maxrss_after_bytes = None;
                actual_memory.decoder_schedule_reason = reference_memory.decoder_schedule_reason;
                assert_eq!(
                    actual_memory, reference_memory,
                    "only truthful schedule reason differs"
                );
            }
        });
    });
}

#[test]
#[ignore = "self-leased <=128 MiB complete buffers; selected Full 2K/4K and decoder 2K; MTL debug"]
fn selected_full_geometry_edges_preserve_batch_1_2_4_bits_and_driver_attribution() {
    let _lease = reuse::require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    objc2::rc::autoreleasepool(|_| {
        let weights = crate::deepseek_v41::forward::toy::Weights::new(&ctx);
        let embedding = incumbent::embeddings(&ctx);
        let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
        configure_four(&mut model, 3);
        model.config.candidate_block = 8;
        for layer in &mut model.layers {
            layer.attention.candidate_block_size = 8;
        }
        prefill::with_batched_f32(|| {
            for width in [2048, 4096] {
                model.prefill_rows = width;
                for n in [2049, 4095, 4096, 4097] {
                    admit(&model, n);
                    let p = plan(&model, n);
                    assert_eq!(p.scalar_tail(), (n % width == 1).then_some(n - 1));
                    let reference = objc2::rc::autoreleasepool(|_| arm(&model, n, 0));
                    for depth in [1, 2, 4] {
                        let actual = objc2::rc::autoreleasepool(|_| arm(&model, n, depth));
                        assert_pair(&actual, &reference, &model, n);
                        assert_eq!(actual.memory.full_chunk_rows, width);
                    }
                }
            }
        });
    });
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
        let bytes = bound(&c, &schedule, n, prefill::DEFAULT_CHUNK_ROWS, &|n| {
            n + 16384
        });
        assert!(
            bytes <= 128 * 1024 * 1024,
            "W={window} N={n} full fixture ledger {bytes}"
        );
    }
    assert!(reuse::wide_bound(&|n| n + 16384) <= 128 * 1024 * 1024);
    let schedule = crate::deepseek_v41::gguf::Schedule {
        layers: 6,
        kv_source_layers: vec![1, 2],
        index_source_layers: vec![1, 2, 4],
        compression_ratios: vec![0, 2, 1, 1, 1, 1],
        candidate_source_layer: Some(2),
    };
    for width in [2048, 4096] {
        for n in [2049, 4095, 4096, 4097] {
            let c = Config {
                candidate_block: 8,
                index_topk: 2,
                ..crate::deepseek_v41::forward::toy::config()
            };
            assert!(
                bound(&c, &schedule, n, width, &|n| n + 16384) <= 128 * 1024 * 1024,
                "selected Full fixture width={width} N={n}"
            );
        }
    }
}

#[test]
#[ignore = "self-leased <=128 MiB complete buffers; MTL debug; no real model"]
fn attention_error_survives_all_later_healthy_units_and_releases_owners_while_poisoned() {
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
            let _depth = Scope::new(4);
            let mut state = objc2::rc::autoreleasepool(|_| model.new_state(640).unwrap());
            let baseline = ctx.current_allocated_size();
            objc2::rc::autoreleasepool(|_| {
                let _capture = Capture::begin();
                BAD_ATTENTION.with(|v| v.set(true));
                let addresses = state.addresses;
                let error = crate::deepseek_v41::forward::ced::run(
                    &model,
                    &mut state,
                    &incumbent::tokens(640),
                    None,
                    Arithmetic::Production,
                )
                .unwrap_err();
                assert!(
                    error.to_string().contains("CED attention status 4"),
                    "{error}"
                );
                let statuses = STATUSES.with(|v| v.take());
                assert_eq!(
                    statuses.len(),
                    plan_batch::units(&plan(&model, 640), 2..6).unwrap().len()
                );
                for (unit, ffn, attention) in &statuses {
                    assert_eq!(
                        ffn.len(),
                        if unit.kind == Kind::Rows {
                            unit.rows.len()
                        } else {
                            0
                        }
                    );
                    assert!(
                        ffn.iter().all(|&word| word == 1),
                        "every nonfaulted FFN word is READY: {unit:?}"
                    );
                    assert_eq!(
                        *attention, 4,
                        "one-shot attention error remains sticky: {unit:?}"
                    );
                }
                assert!(CANARY_WORDS_CHECKED.with(Cell::get) > 0);
                assert_eq!(PUBLIC_PROGRESS.with(|v| v.take()), [0]);
                assert!(
                    state
                        .gpu_step
                        .as_ref()
                        .unwrap()
                        .attention
                        .ced_public_decoder_progress()
                        .is_none()
                );
                incumbent::assert_poisoned_rejects(&model, &mut state, addresses);
            });
            assert!(
                ctx.current_allocated_size() <= baseline,
                "transient owners released while poisoned State remains alive"
            );
            assert!(state.is_poisoned());
        });
    });
}

fn two_decoder_multichunk(model: &mut Forward<'_>) {
    configure_four(model, 2050);
    model.layers.truncate(4);
    model.schedule.layers = 4;
    model.schedule.compression_ratios.truncate(4);
    model.schedule.index_source_layers.retain(|&id| id < 4);
}

fn assert_one_ffn_fault(statuses: &[(Unit, Vec<u32>, u32)], fault: FfnFault, attention: u32) {
    let mut failures = Vec::new();
    for (unit, words, status) in statuses {
        assert_eq!(
            words.len(),
            if unit.kind == Kind::Rows {
                unit.rows.len()
            } else {
                0
            }
        );
        assert_eq!(*status, attention, "{unit:?}");
        for (row, &word) in words.iter().enumerate() {
            if unit.layer == fault.layer && unit.rows.start == fault.start && row == fault.row {
                assert_eq!(word, fault.word as u32);
                failures.push((unit.layer, unit.rows.start, row));
            } else {
                assert_eq!(
                    word,
                    metal::ROUTE_STATUS_READY as u32,
                    "{unit:?} row {row}: all other ledger words must remain READY"
                );
            }
        }
    }
    assert_eq!(failures, [(fault.layer, fault.start, fault.row)]);
    assert!(
        statuses
            .iter()
            .any(|(u, words, _)| u.layer == 2 && u.rows == (2048..2050) && words.len() == 2)
    );
    assert!(
        statuses
            .iter()
            .any(|(u, words, _)| u.layer == 3 && u.kind == Kind::Rows && words.len() == 1)
    );
    assert!(CANARY_WORDS_CHECKED.with(Cell::get) > 0);
}

#[test]
#[ignore = "self-leased <=128 MiB complete buffers; one GPU status fault after healthy math; MTL debug"]
fn single_ffn_word_survives_healthy_same_layer_second_chunk_and_every_later_word() {
    let _lease = reuse::require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    objc2::rc::autoreleasepool(|_| {
        let weights = crate::deepseek_v41::forward::toy::Weights::new(&ctx);
        let embedding = incumbent::embeddings(&ctx);
        let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
        two_decoder_multichunk(&mut model);
        admit(&model, 2050);
        prefill::with_batched_f32(|| {
            objc2::rc::autoreleasepool(|_| {
                let _ = arm(&model, 2050, 2);
            });
            ctx.pipeline("kernel_copy_offset_i32").unwrap();
            let _depth = Scope::new(2);
            let mut state = objc2::rc::autoreleasepool(|_| model.new_state(2050).unwrap());
            let baseline = ctx.current_allocated_size();
            objc2::rc::autoreleasepool(|_| {
                let _capture = Capture::begin();
                let spec = FfnFault {
                    layer: 2,
                    start: 0,
                    row: 2047,
                    word: -7,
                };
                let _fault = FfnFaultScope::new(spec);
                let addresses = state.addresses;
                let error = crate::deepseek_v41::forward::ced::run(
                    &model,
                    &mut state,
                    &incumbent::tokens(2050),
                    None,
                    Arithmetic::Production,
                )
                .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .ends_with("prefill FFN layer 2 row 2047 route status -7"),
                    "{error}"
                );
                assert_eq!(FFN_HITS.with(Cell::get), 1);
                let statuses = STATUSES.with(|v| v.take());
                assert_eq!(
                    statuses
                        .iter()
                        .map(|(u, _, _)| u.clone())
                        .collect::<Vec<_>>(),
                    plan_batch::units(&plan(&model, 2050), 2..4).unwrap()
                );
                assert_one_ffn_fault(&statuses, spec, 0);
                assert_eq!(PUBLIC_PROGRESS.with(|v| v.take()), [0]);
                assert!(
                    state
                        .gpu_step
                        .as_ref()
                        .unwrap()
                        .attention
                        .ced_public_decoder_progress()
                        .is_none()
                );
                incumbent::assert_poisoned_rejects(&model, &mut state, addresses);
            });
            assert!(
                ctx.current_allocated_size() <= baseline,
                "all transient status/seed/control/decoder owners drop with poisoned State still alive"
            );
        });
    });
}

#[test]
#[ignore = "self-leased <=128 MiB complete buffers; competing one-shot errors; MTL debug"]
fn competing_attention_and_single_ffn_word_obey_logical_unit_precedence() {
    let _lease = reuse::require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    objc2::rc::autoreleasepool(|_| {
        let weights = crate::deepseek_v41::forward::toy::Weights::new(&ctx);
        let embedding = incumbent::embeddings(&ctx);
        let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
        two_decoder_multichunk(&mut model);
        admit(&model, 2050);
        prefill::with_batched_f32(|| {
            for start in [0, 2048] {
                let _depth = Scope::new(2);
                objc2::rc::autoreleasepool(|_| {
                    let _capture = Capture::begin();
                    BAD_ATTENTION.with(|v| v.set(true));
                    let spec = FfnFault {
                        layer: 2,
                        start,
                        row: 1,
                        word: -7,
                    };
                    let _fault = FfnFaultScope::new(spec);
                    let mut state = model.new_state(2050).unwrap();
                    let addresses = state.addresses;
                    let error = crate::deepseek_v41::forward::ced::run(
                        &model,
                        &mut state,
                        &incumbent::tokens(2050),
                        None,
                        Arithmetic::Production,
                    )
                    .unwrap_err();
                    assert_eq!(FFN_HITS.with(Cell::get), 1);
                    if start == 0 {
                        assert!(
                            error
                                .to_string()
                                .ends_with("prefill FFN layer 2 row 1 route status -7"),
                            "{error}"
                        );
                    } else {
                        assert!(
                            error.to_string().ends_with("CED attention status 4"),
                            "{error}"
                        );
                    }
                    assert_one_ffn_fault(&STATUSES.with(|v| v.take()), spec, 4);
                    incumbent::assert_poisoned_rejects(&model, &mut state, addresses);
                });
            }
        });
    });
}

#[test]
#[ignore = "self-leased <=128 MiB complete buffers; GPU ledger boundary clobber detection; MTL debug"]
fn gpu_written_status_boundary_canary_detects_a_gpu_clobber_after_completion() {
    let _lease = reuse::require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    objc2::rc::autoreleasepool(|_| {
        let weights = crate::deepseek_v41::forward::toy::Weights::new(&ctx);
        let embedding = incumbent::embeddings(&ctx);
        let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
        configure_four(&mut model, 65);
        admit(&model, 640);
        prefill::with_batched_f32(|| {
            let _depth = Scope::new(2);
            let _capture = Capture::begin();
            BAD_CANARY.with(|v| v.set(true));
            let mut state = model.new_state(640).unwrap();
            let addresses = state.addresses;
            let error = crate::deepseek_v41::forward::ced::run(
                &model,
                &mut state,
                &incumbent::tokens(640),
                None,
                Arithmetic::Production,
            )
            .unwrap_err();
            assert!(
                error
                    .to_string()
                    .ends_with("CED GPU status ledger canary overwritten"),
                "{error}"
            );
            assert_eq!(
                state.gpu_step.as_ref().unwrap().commands,
                2,
                "GPU completion precedes boundary inspection"
            );
            assert!(!BAD_CANARY.with(Cell::get));
            incumbent::assert_poisoned_rejects(&model, &mut state, addresses);
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
                let _ = arm(&model, 640, 0);
            });
            for (depth, role, action) in [
                (2, "ced_batch_unit_encoded", Action::Error),
                (2, "ced_batch_unit_encoded", Action::Cancel),
                (2, "ced_batch_unit_encoded", Action::Panic),
                (2, "ced_decoder_batch", Action::Completion),
                (2, "ced_batch_completed", Action::Error),
                (2, "ced_batch_completed", Action::Cancel),
                (2, "ced_batch_completed", Action::Panic),
                (2, "ced_batch_retired", Action::Error),
                (2, "ced_batch_retired", Action::Cancel),
                (2, "ced_batch_retired", Action::Panic),
                (0, "ced_seed_decoder_rows", Action::Completion),
            ] {
                let _depth = Scope::new(depth);
                let mut state = objc2::rc::autoreleasepool(|_| model.new_state(640).unwrap());
                let baseline = ctx.current_allocated_size();
                objc2::rc::autoreleasepool(|_| {
                    let _capture = Capture::begin();
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
                    assert!(
                        state
                            .gpu_step
                            .as_ref()
                            .unwrap()
                            .attention
                            .ced_public_decoder_progress()
                            .is_none()
                    );
                    if role == "ced_batch_retired" {
                        assert_eq!(
                            RETIRED_PROGRESS.with(|v| v.take()),
                            [2],
                            "one complete group retired before injected failure"
                        );
                    } else {
                        assert!(RETIRED_PROGRESS.with(|v| v.borrow().is_empty()));
                    }
                    incumbent::assert_poisoned_rejects(&model, &mut state, addresses);
                });
                assert!(
                    ctx.current_allocated_size() <= baseline,
                    "{role}: resources return to pre-existing causal-state baseline while poisoned State remains alive"
                );
                assert!(state.is_poisoned());
            }
            for (depth, site) in [
                (2, AllocationSite::Ledger),
                (2, AllocationSite::SeedNormalized),
                (2, AllocationSite::DecoderPool),
                (0, AllocationSite::SeedNormalized),
                (0, AllocationSite::DecoderPool),
            ] {
                let _depth = Scope::new(depth);
                let mut state = objc2::rc::autoreleasepool(|_| model.new_state(640).unwrap());
                let baseline = ctx.current_allocated_size();
                objc2::rc::autoreleasepool(|_| {
                    let fault = AllocationFault::new(site);
                    let addresses = state.addresses;
                    let error = crate::deepseek_v41::forward::ced::run(
                        &model,
                        &mut state,
                        &incumbent::tokens(640),
                        None,
                        Arithmetic::Production,
                    )
                    .unwrap_err();
                    assert_eq!(fault.hits.get(), 1, "typed allocation site {site:?}");
                    assert!(
                        error.to_string().contains("could not create buffer of"),
                        "{site:?}: {error}"
                    );
                    assert_eq!(
                        state.gpu_step.as_ref().unwrap().commands,
                        1,
                        "allocation failure never submits a decoder command"
                    );
                    assert!(
                        state
                            .gpu_step
                            .as_ref()
                            .unwrap()
                            .attention
                            .ced_public_decoder_progress()
                            .is_none()
                    );
                    incumbent::assert_poisoned_rejects(&model, &mut state, addresses);
                });
                assert!(
                    ctx.current_allocated_size() <= baseline,
                    "{site:?}: partial allocations released with poisoned State alive"
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
