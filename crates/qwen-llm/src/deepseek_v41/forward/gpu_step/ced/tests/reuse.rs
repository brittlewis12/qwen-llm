use super::*;
use crate::deepseek_v41::forward;
use forward::ced::tests::{self as incumbent, Action, Scope as Fault};
use forward::ced_plan::CedRequest;
use objc2::rc::Weak;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

const LIMIT: u64 = 128 * 1024 * 1024;
const MAX: usize = 513;
const SEQUENCE: [usize; 7] = [513, 1, 512, 33, 127, 129, 513];
const POISON: u32 = 0x7fc0beef;

thread_local! {
    static FRESH: Cell<bool> = const { Cell::new(false) };
    static OBSERVE: RefCell<Option<Rc<RefCell<Observation>>>> = const { RefCell::new(None) };
}

pub(super) fn fresh_decoder_rows() -> bool {
    FRESH.with(Cell::get)
}

struct Fresh {
    previous: bool,
    _thread: PhantomData<Rc<()>>,
}

impl Fresh {
    fn new(value: bool) -> Self {
        Self {
            previous: FRESH.with(|slot| slot.replace(value)),
            _thread: PhantomData,
        }
    }
}

impl Drop for Fresh {
    fn drop(&mut self) {
        FRESH.with(|slot| slot.set(self.previous));
    }
}

#[test]
fn fresh_incumbent_override_is_nested_thread_local_and_unwind_safe() {
    assert!(!fresh_decoder_rows());
    let outer = Fresh::new(true);
    {
        let _inner = Fresh::new(false);
        assert!(!fresh_decoder_rows());
    }
    assert!(fresh_decoder_rows());
    std::thread::spawn(|| assert!(!fresh_decoder_rows()))
        .join()
        .unwrap();
    assert!(
        catch_unwind(|| {
            let _inner = Fresh::new(false);
            panic!("restore test override");
        })
        .is_err()
    );
    assert!(fresh_decoder_rows());
    drop(outer);
    assert!(!fresh_decoder_rows());
}

fn dependency_plan(c: &Config, n: usize) -> CedPlan {
    dependency_plan_at_capacity(c, n, n)
}

fn dependency_plan_at_capacity(c: &Config, n: usize, capacity: usize) -> CedPlan {
    let schedule = crate::deepseek_v41::gguf::Schedule {
        layers: 6,
        kv_source_layers: vec![1, 3],
        index_source_layers: vec![1, 3, 4],
        compression_ratios: vec![0, 2, 2, 1, 1, 1],
        candidate_source_layer: Some(3),
    };
    CedPlan::shrinking(
        c,
        &schedule,
        &engram::HashConfig {
            layers: Vec::new(),
            ..engram::HashConfig::default()
        },
        &CedRequest::cold(n, capacity),
    )
    .unwrap()
}

#[test]
fn derives_actual_maximum_excluding_native_and_genuine_scalar_tails() {
    for n in [1, 2, 128, 513, 2048, 2049, 4096, 4097] {
        for window in [1, 3, 257, 2048] {
            let c = Config {
                window,
                ..forward::toy::config()
            };
            let plan = dependency_plan(&c, n);
            assert_eq!(
                maximum_decoder_rows(&plan, Arithmetic::ScalarNative).unwrap(),
                None
            );
            let maximum = maximum_decoder_rows(&plan, Arithmetic::Production).unwrap();
            let mut actual = Vec::new();
            for layer in &plan.decoder {
                let tail = prefill::original_scalar_tail(n);
                for chunk in chunk_views(layer.output_rows.start..tail.unwrap_or(n), 2048).unwrap()
                {
                    actual.push(chunk.absolute_rows);
                }
            }
            let expected = actual.iter().map(Range::len).max();
            assert_eq!(maximum.as_ref().map(Range::len), expected);
            if let Some(maximum) = maximum {
                assert!((1..=2048).contains(&maximum.len()));
                assert_eq!(
                    maximum,
                    *actual
                        .iter()
                        .find(|r| r.len() == expected.unwrap())
                        .unwrap()
                );
            }
        }
    }
    let c = Config {
        window: 257,
        ..forward::toy::config()
    };
    assert_eq!(
        maximum_decoder_rows(&dependency_plan(&c, 128), Arithmetic::Production).unwrap(),
        Some(0..128)
    );
    assert_eq!(
        maximum_decoder_rows(&dependency_plan(&c, 1), Arithmetic::Production).unwrap(),
        None
    );
    let c = Config { window: 1, ..c };
    assert_eq!(
        maximum_decoder_rows(&dependency_plan(&c, 128), Arithmetic::Production).unwrap(),
        Some(127..128)
    );
}

#[test]
fn outer_prefix_rejects_zero_oversize_and_overflow_before_views() {
    for (hidden, rows) in [
        (0, 1),
        (32, 0),
        (32, 2049),
        (usize::MAX, 2),
        (usize::MAX / 8, 1),
    ] {
        assert!(checked_outer_elements(hidden, rows).is_err());
    }
    assert_eq!(checked_outer_elements(32, 1).unwrap(), (128, 32));
    assert_eq!(checked_outer_elements(32, 513).unwrap(), (65664, 16416));
    let mut plan = dependency_plan(&forward::toy::config(), 128);
    plan.decoder[0].output_rows = 129..128;
    assert!(maximum_decoder_rows(&plan, Arithmetic::Production).is_err());
}

type WeakBuffer = Weak<ProtocolObject<dyn MTLBuffer>>;

#[derive(Debug, PartialEq, Eq)]
struct Unit {
    layer: usize,
    range: Range<usize>,
    tensors: Vec<Vec<u32>>,
}

#[derive(Default)]
struct Observation {
    encoder: Vec<WeakBuffer>,
    backing: Vec<WeakBuffer>,
    seeds: Vec<WeakBuffer>,
    poison_seed: bool,
    compounds: Vec<(u32, bool)>,
    births: usize,
    units: Vec<Unit>,
    plateau: Vec<u64>,
}

pub(super) fn seed_allocated(seed: &LocalSeed) {
    OBSERVE.with(|slot| {
        if let Some(o) = slot.borrow().as_ref() {
            let mut o = o.borrow_mut();
            assert!(
                o.seeds.iter().all(|w| w.load().is_none()),
                "only one co-live seed scratch"
            );
            o.seeds.extend(
                [&seed.collapsed, &seed.normalized, seed.attention.value()]
                    .into_iter()
                    .map(weak),
            );
            if o.poison_seed {
                fill(&seed.x, POISON);
                o.poison_seed = false;
            }
        }
    });
}

pub(super) fn compound_completed(
    attention: &attention_gpu::Workspace,
    scratch: &BlockRows,
    layer: usize,
) {
    OBSERVE.with(|slot| {
        if let Some(o) = slot.borrow().as_ref() {
            o.borrow_mut().compounds.push((
                bits(attention.status())[0],
                scratch.ffn.validate_status_layers(layer..layer + 1).is_ok(),
            ));
        }
    });
}

struct Observe {
    current: Rc<RefCell<Observation>>,
    previous: Option<Rc<RefCell<Observation>>>,
}

impl Observe {
    fn begin() -> Self {
        let current = Rc::new(RefCell::new(Observation::default()));
        let previous = OBSERVE.with(|slot| slot.replace(Some(Rc::clone(&current))));
        Self { current, previous }
    }
}

impl Drop for Observe {
    fn drop(&mut self) {
        OBSERVE.with(|slot| slot.replace(self.previous.take()));
    }
}

fn weak(t: &MetalTensor) -> WeakBuffer {
    let weak = Weak::from_retained(&t.buffer);
    assert!(weak.load().is_some());
    weak
}

fn visible_scratch(rows: &BlockRows) -> [&MetalTensor; 10] {
    [
        &rows.half,
        &rows.output,
        &rows.hc.pre,
        &rows.hc.post,
        &rows.hc.comb,
        &rows.hc.normalized,
        &rows.ffn.hc.pre,
        &rows.ffn.hc.post,
        &rows.ffn.hc.comb,
        &rows.ffn.hc.normalized,
    ]
}

pub(super) fn encoder_allocated(x: &MetalTensor, incoming: &MetalTensor, rows: &BlockRows) {
    OBSERVE.with(|slot| {
        if let Some(o) = slot.borrow().as_ref() {
            let mut o = o.borrow_mut();
            o.encoder.extend(
                [x, incoming]
                    .into_iter()
                    .chain(visible_scratch(rows))
                    .map(weak),
            );
        }
    });
}

pub(super) fn pool_born(ctx: &MetalContext, rows: &BlockRows) {
    OBSERVE.with(|slot| {
        if let Some(o) = slot.borrow().as_ref() {
            let mut o = o.borrow_mut();
            assert!(
                o.encoder.iter().all(|w| w.load().is_none()),
                "encoder witnesses live at pool birth"
            );
            o.births += 1;
            assert_eq!(o.births, 1, "one request-local maximum backing");
            o.backing
                .extend(visible_scratch(rows).into_iter().map(weak));
            o.plateau.push(ctx.current_allocated_size());
        }
    });
}

fn bits(t: &MetalTensor) -> Vec<u32> {
    assert!(matches!(t.dtype, GgmlType::F32 | GgmlType::I32));
    assert_eq!(t.offset % 4, 0);
    assert!(t.offset + t.n_elements() * 4 <= t.buffer.length() as u64);
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

fn fill(t: &MetalTensor, value: u32) {
    assert!(t.offset + t.n_elements() * 4 <= t.buffer.length() as u64);
    unsafe {
        std::slice::from_raw_parts_mut(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u32>()
                .add((t.offset / 4) as usize),
            t.n_elements() as usize,
        )
        .fill(value);
    }
}

pub(super) fn decoder_completed(
    ctx: &MetalContext,
    scratch: &BlockRows,
    controls: &attention_gpu::QueryControlView,
    layer: usize,
    x: &MetalTensor,
    incoming: &MetalTensor,
) {
    OBSERVE.with(|slot| {
        if let Some(o) = slot.borrow().as_ref() {
            let mut o = o.borrow_mut();
            for unexecuted in (0..6).filter(|&id| id != layer) {
                let status = scratch
                    .ffn
                    .validate_status_layers(unexecuted..unexecuted + 1)
                    .unwrap_err();
                assert_eq!(
                    status,
                    format!("prefill FFN layer {unexecuted} row 0 route status 0"),
                    "unexecuted-layer sentinel must be reset at the current n stride"
                );
            }
            o.units.push(Unit {
                layer,
                range: controls.absolute_rows(),
                tensors: visible_scratch(scratch)
                    .into_iter()
                    .chain([x, incoming, controls.selected(), controls.candidates()])
                    .map(bits)
                    .collect(),
            });
            o.plateau.push(ctx.current_allocated_size());
        }
    });
}

// Covers all 40-layer toy allocations (including both Engram matrices), then
// every replacement made by nonuniform_model. Charge originals too: an outer
// autorelease pool may retain them until the entire fixture has drained.
fn weight_bytes(price: &impl Fn(u64) -> u64) -> u64 {
    let mut bytes = Vec::new();
    for layer in 0..40 {
        bytes.extend([32 * 6, 6 * 8, 32 * 4, 4 * 6, 6 * 32, 6, 4, 2].map(|n| n * 4));
        bytes.extend([32, 32, 128 * 24, 128 * 24].map(|n| n * 4));
        if [2, 8, 14, 20].contains(&layer) {
            bytes.extend([32 * 4, 4, 4 * 4, 4].map(|n| n * 4));
            if layer < 20 {
                bytes.push(32 * 4 * 4);
            }
        }
        if [2, 8, 14, 20, 24, 28, 32, 36].contains(&layer) {
            bytes.extend([6 * 8 * 4, 32 * 2 * 4]);
        }
        bytes.extend([32 * 32 * 4; 3]);
        if [1, 14].contains(&layer) {
            bytes.extend([6144 * 160 * 4, 32 * 4 * 4, 32 * 4 * 4]);
        }
        bytes.extend([24 * 4, 3 * 4, 24 * 4, 3 * 4, 32 * 8 * 4, 8 * 4]);
    }
    bytes.extend([32 * 8 * 4; 2]);
    bytes.extend([8 * 32 * 17; 3]);
    // Replacement embeddings/head plus a superset of every varied projection
    // for six layers, including a compressor, gate, and full index projection.
    bytes.extend([32 * 8 * 4; 2]);
    for _ in 0..6 {
        bytes.extend(
            [
                128 * 24,
                128 * 24,
                32 * 8,
                32 * 32,
                32 * 32,
                32 * 32,
                32 * 6,
                6 * 8,
                32 * 4,
                4 * 3,
                4 * 3,
                6 * 32,
                32 * 4,
                32 * 4,
                6 * 8,
                32 * 2,
                4 * 4,
            ]
            .map(|n| n * 4),
        );
    }
    bytes.into_iter().map(price).sum()
}

fn fixture_config(window: usize) -> Config {
    Config {
        window,
        index_topk: 2,
        ..forward::toy::config()
    }
}

fn explicit_bound(c: &Config, n: usize, capacity: usize, price: &impl Fn(u64) -> u64) -> u64 {
    let plan = dependency_plan_at_capacity(c, n, capacity);
    let priced = |bytes| Ok(if bytes == 0 { 0 } else { price(bytes) });
    let estimate =
        forward::ced::estimate_from_parts(c, 6, &[], 0, capacity, &plan, true, &priced).unwrap();
    let schedule = crate::deepseek_v41::gguf::Schedule {
        layers: 6,
        kv_source_layers: vec![1, 3],
        index_source_layers: vec![1, 3, 4],
        compression_ratios: vec![0, 2, 2, 1, 1, 1],
        candidate_source_layer: Some(3),
    };
    // Arms run sequentially and retain CPU bits only. Two state budgets also
    // cover a poisoned state during rejection checks, never a second row pool.
    let states = 2 * super::super::bytes_with_schedule(c, capacity, &schedule, 0).unwrap();
    // Guard replacement and original are both charged; half staging is already
    // included in estimate, even on devices that do not select it.
    let guards =
        price((4 * c.hidden * MAX + 32) as u64 * 4) + price((c.hidden * MAX + 32) as u64 * 4);
    weight_bytes(price) + states + estimate.incremental_metal_bytes + guards
}

#[test]
fn complete_fixture_ledgers_stay_below_128_mib() {
    let price = |n| if n == 0 { 0 } else { n + 16384 };
    for (window, n, capacity) in [
        (915, 1900, 1900),
        (257, 640, 640),
        (257, 2049, 2049),
        (3, 33, 33),
        (1, 128, 128),
        (1025, 2542, 2542),
        (2050, 2542, 2542),
    ] {
        let bound = explicit_bound(&fixture_config(window), n, capacity, &price);
        assert!(bound <= LIMIT, "W={window}, N={n}: {bound} > {LIMIT}");
    }
    let c = Config {
        index_topk: 512,
        candidate_block: 128,
        candidate_topk: 128,
        ..fixture_config(257)
    };
    let bound = explicit_bound(&c, MAX, 16384, &price);
    assert!(
        bound <= LIMIT,
        "separate P4 slab fixture: {bound} > {LIMIT}"
    );
}

fn admit(ctx: &MetalContext, c: &Config, n: usize, capacity: usize) -> u64 {
    let price = |n| ctx.price_shared_buffer_upper(n).unwrap().priced_upper_bytes;
    let bytes = explicit_bound(c, n, capacity, &price);
    assert!(
        bytes <= LIMIT,
        "complete explicit buffer ledger: {bytes} > {LIMIT}"
    );
    let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
        bytes,
        32 * 1024 * 1024,
        0,
        ctx.memory_signals(),
        true,
    );
    assert!(
        admission.refusal().is_none(),
        "bounded fixture admission: {:?}",
        admission.refusal()
    );
    bytes
}

fn require_debug_lease() -> metal::MetalProcessLease {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    assert!(!crate::env_flag::read_default_off(
        "QWEN_METAL_LEASE_SKIP_WIRED_GATE"
    ));
    metal::acquire_metal_benchmark_lease().expect("bounded CED reuse production lease")
}

fn configure(model: &mut Forward<'_>, window: usize) {
    model.config.window = window;
    model.engram_hash.layers.clear();
    for layer in &mut model.layers {
        layer.engram = None;
        layer.attention.window = window;
    }
}

struct Census;
impl Census {
    fn begin() -> Self {
        metal::dispatch_census_begin();
        Self
    }
    fn finish(self) -> Vec<metal::DispatchCensusRow> {
        metal::dispatch_census_take()
    }
}
impl Drop for Census {
    fn drop(&mut self) {
        let _ = metal::dispatch_census_take();
    }
}

fn dispatches(rows: &[metal::DispatchCensusRow]) -> Vec<(String, [u64; 6], bool)> {
    rows.iter()
        .map(|r| {
            (
                r.kernel.clone(),
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

fn commands(report: &profile::Report) -> Vec<(&'static str, [usize; 4], usize)> {
    let mut rows = report
        .operations
        .iter()
        .filter(|r| r.completed_submits != 0)
        .map(|r| (r.operation, r.dimensions, r.completed_submits))
        .collect::<Vec<_>>();
    rows.sort_unstable();
    rows
}

fn causal(workspace: &Workspace, n: usize) -> Vec<Vec<u32>> {
    workspace.attention.validate_snapshot_position(n).unwrap();
    workspace
        .attention
        .causal_regions(n)
        .unwrap()
        .into_iter()
        .map(|(t, count)| bits(t)[..count].to_vec())
        .collect()
}

fn check_canaries(before: &[u32], after: &[u32], offset: usize, count: usize) {
    assert_eq!(
        &before[..offset],
        &after[..offset],
        "control prefix overwritten"
    );
    assert_eq!(
        &before[offset + count..],
        &after[offset + count..],
        "control suffix overwritten"
    );
}

// The first decoder layer has 1829 outputs: exercise the requested dirty
// sequence, then one last packed row. Later layers use 513/402 and packed 1.
fn dirty_arm(
    model: &Forward<'_>,
    fresh: bool,
) -> (
    Vec<Unit>,
    Vec<Vec<u32>>,
    CedCounters,
    Vec<metal::DispatchCensusRow>,
    profile::Report,
) {
    let _fresh = Fresh::new(fresh);
    let n = 1900;
    let plan = dependency_plan(&model.config, n);
    assert_eq!(
        plan.decoder[0].output_rows.len(),
        SEQUENCE.iter().sum::<usize>() + 1
    );
    let observe = Observe::begin();
    let mut workspace = Workspace::new(model, n).unwrap();
    let mut storage = Storage::new(model, n, &plan).unwrap();
    let mut counters = CedCounters::default();
    let input = incumbent::tokens(n);
    let hashes = vec![engram::LayerRows::default(); n];
    let profile = profile::Scope::begin().unwrap();
    let census = Census::begin();
    workspace.attention.begin_ced(&plan).unwrap();
    objc2::rc::autoreleasepool(|_| {
        workspace.ced_encoder_rows(
            model,
            &plan,
            0,
            &input,
            &hashes,
            None,
            &mut storage,
            &mut counters,
        )
    })
    .unwrap();
    let whole = storage
        .controls
        .checked_tile_view(plan.candidate_rows.start, plan.candidate_rows.len())
        .unwrap();
    fill(whole.selected(), POISON);
    fill(whole.candidates(), POISON);
    let mut pool = DecoderRowsPool {
        maximum: Some(plan.candidate_rows.start..plan.candidate_rows.start + MAX),
        backing: None,
    };
    for (index, layer) in plan.decoder.iter().enumerate() {
        for chunk in chunk_views(layer.seed_rows.clone(), 2048).unwrap() {
            objc2::rc::autoreleasepool(|_| {
                workspace.ced_seed(
                    model,
                    layer.layer,
                    chunk.absolute_rows,
                    Arithmetic::Production,
                    None,
                    &mut storage,
                    &mut counters,
                )
            })
            .unwrap();
        }
        let widths = if index == 0 {
            SEQUENCE.into_iter().chain([1]).collect::<Vec<_>>()
        } else {
            chunk_views(layer.output_rows.clone(), MAX)
                .unwrap()
                .map(|r| r.absolute_rows.len())
                .collect()
        };
        let mut start = layer.output_rows.start;
        for width in widths {
            let end = start + width;
            let selected_before = bits(whole.selected());
            let candidates_before = bits(whole.candidates());
            if let Some(backing) = &pool.backing {
                for t in visible_scratch(backing) {
                    fill(t, POISON);
                }
            }
            objc2::rc::autoreleasepool(|_| {
                workspace.ced_decoder_rows(
                    model,
                    layer.layer,
                    start..end,
                    None,
                    index + 1 == plan.decoder.len() && end == n,
                    None,
                    &mut storage,
                    &mut counters,
                    &mut pool,
                )
            })
            .unwrap();
            let row_offset = start - plan.candidate_rows.start;
            check_canaries(
                &selected_before,
                &bits(whole.selected()),
                row_offset * model.config.index_topk,
                width * model.config.index_topk,
            );
            let cw = n.div_ceil(model.config.candidate_block);
            check_canaries(
                &candidates_before,
                &bits(whole.candidates()),
                row_offset * cw,
                width * cw,
            );
            if let Some(backing) = &pool.backing {
                assert_eq!(backing.hc.rows, MAX);
                assert!(
                    observe
                        .current
                        .borrow()
                        .backing
                        .iter()
                        .all(|w| w.load().is_some())
                );
            }
            start = end;
        }
        assert_eq!(start, n);
    }
    drop(pool);
    assert!(
        observe
            .current
            .borrow()
            .backing
            .iter()
            .all(|w| w.load().is_none())
    );
    counters.validate(&plan).unwrap();
    workspace.attention.commit_ced(n).unwrap();
    let mut causal = causal(&workspace, n);
    causal.push(bits(&workspace.logits));
    let census = census.finish();
    let profile = profile.finish();
    let mut observed = observe.current.borrow_mut();
    assert_eq!(observed.births, usize::from(!fresh));
    (
        std::mem::take(&mut observed.units),
        causal,
        counters,
        census,
        profile,
    )
}

#[test]
#[ignore = "self-leased <=128 MiB complete buffers; MTL debug validation; no real model"]
fn dirty_combined_rows_match_fresh_bitwise_and_preserve_dispatch_commands() {
    let _lease = require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    admit(&ctx, &fixture_config(915), 1900, 1900);
    objc2::rc::autoreleasepool(|_| {
        let weights = forward::toy::Weights::new(&ctx);
        let embedding = incumbent::embeddings(&ctx);
        let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
        configure(&mut model, 915);
        prefill::with_batched_f32(|| {
            let fresh = objc2::rc::autoreleasepool(|_| dirty_arm(&model, true));
            let reuse = objc2::rc::autoreleasepool(|_| dirty_arm(&model, false));
            assert_eq!(
                reuse.0, fresh.0,
                "active outputs/mHC/control bits after every checked completion"
            );
            assert_eq!(reuse.1, fresh.1, "all causal state bits");
            assert_eq!(reuse.2, fresh.2, "work/completion counters");
            assert_eq!(
                dispatches(&reuse.3),
                dispatches(&fresh.3),
                "captured pipelines and grids"
            );
            assert_eq!(
                commands(&reuse.4),
                commands(&fresh.4),
                "command roles and dimensions"
            );
        });
    });
}

struct Allocations;

impl Allocations {
    fn begin() -> Self {
        metal::allocation_census_begin();
        Self
    }
    fn finish(self) -> Vec<metal::MetalAllocationCensusRow> {
        metal::allocation_census_take()
    }
}

impl Drop for Allocations {
    fn drop(&mut self) {
        let _ = metal::allocation_census_take();
    }
}

#[test]
#[ignore = "self-leased <=128 MiB complete buffers; MTL debug validation; no real model"]
fn one_maximum_backing_and_zero_metal_allocations_for_every_checked_view() {
    let _lease = require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    admit(&ctx, &fixture_config(257), 640, 640);
    objc2::rc::autoreleasepool(|_| {
        let weights = forward::toy::Weights::new(&ctx);
        let embedding = incumbent::embeddings(&ctx);
        let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
        configure(&mut model, 257);
        let plan = dependency_plan(&model.config, 640);
        let storage = Storage::new(&model, 640, &plan).unwrap();
        let controls = storage
            .controls
            .checked_tile_view(plan.candidate_rows.start, MAX)
            .unwrap();
        let expected = objc2::rc::autoreleasepool(|_| {
            let census = Allocations::begin();
            let rows = BlockRows::new(&model, 640, MAX, Some(&controls)).unwrap();
            let allocations = census.finish();
            assert!(!allocations.is_empty());
            drop(rows);
            allocations
        });
        objc2::rc::autoreleasepool(|_| {
            let mut pool = DecoderRowsPool {
                maximum: Some(controls.absolute_rows()),
                backing: None,
            };
            let census = Allocations::begin();
            let view = pool.checked_rows(&model, 640, &storage, &controls).unwrap();
            assert_eq!(
                census.finish(),
                expected,
                "exactly one incumbent maximum backing allocation set"
            );
            drop(view);
            let backing = pool.backing.as_mut().unwrap();
            let mut guards = Vec::new();
            for (target, shape) in [
                (
                    &mut backing.half,
                    vec![model.config.hidden as u64, 4, MAX as u64],
                ),
                (
                    &mut backing.output,
                    vec![model.config.hidden as u64, MAX as u64],
                ),
            ] {
                let count = shape.iter().product::<u64>();
                let guard = prefill::allocate(&ctx, count as usize + 32, 1, GgmlType::F32).unwrap();
                fill(&guard, POISON);
                *target = guard.view_subrange(16, shape);
                guards.push(guard);
            }
            let weak_backing = weak(&backing.half);
            let fixed = ctx.current_allocated_size();
            for (index, n) in SEQUENCE.into_iter().enumerate() {
                // Move the retained control offset without changing the max
                // backing or chaining a view from the previous smaller view.
                let start = plan.candidate_rows.start + usize::from(index % 2 != 0);
                let controls = storage.controls.checked_tile_view(start, n).unwrap();
                let census = Allocations::begin();
                objc2::rc::autoreleasepool(|_| {
                    let view = pool.checked_rows(&model, 640, &storage, &controls).unwrap();
                    assert_eq!(view.hc.rows, n);
                    assert_eq!(view.half.n_elements(), (model.config.hidden * 4 * n) as u64);
                    assert_eq!(view.output.n_elements(), (model.config.hidden * n) as u64);
                    assert_eq!(view.arithmetic, prefill::RowsArithmetic::Packed);
                    assert!(
                        view.checked_prefix(&model.config, 640, 6, &controls)
                            .is_err(),
                        "never derive from a previous view"
                    );
                    assert!(weak_backing.load().is_some());
                });
                assert!(
                    census.finish().is_empty(),
                    "checked view allocated new Metal storage"
                );
                assert_eq!(
                    ctx.current_allocated_size(),
                    fixed,
                    "fixed backing plateau after view drain"
                );
            }
            let too_large = storage
                .controls
                .checked_tile_view(plan.candidate_rows.start, MAX + 1);
            assert!(
                too_large.is_err(),
                "retained controls reject oversized tile"
            );
            assert!(
                storage
                    .controls
                    .checked_tile_view(plan.candidate_rows.start, 0)
                    .is_err()
            );
            for guard in &guards {
                let words = bits(guard);
                assert!(
                    words[..16]
                        .iter()
                        .chain(&words[words.len() - 16..])
                        .all(|&v| v == POISON)
                );
            }
            drop(pool);
            // Guard owners intentionally retain these two replaced resources.
            drop(guards);
        });
    });
}

type DriverArm = (
    Vec<u32>,
    Vec<Vec<u32>>,
    CedCounters,
    Vec<metal::DispatchCensusRow>,
    profile::Report,
    Vec<Unit>,
);

fn driver_arm(model: &Forward<'_>, n: usize, fresh: bool, arithmetic: Arithmetic) -> DriverArm {
    let _fresh = Fresh::new(fresh);
    let observe = Observe::begin();
    let mut state = model.new_state(n).unwrap();
    let profile = profile::Scope::begin().unwrap();
    let census = Census::begin();
    let (logits, report) =
        forward::ced::run(model, &mut state, &incumbent::tokens(n), None, arithmetic).unwrap();
    let census = census.finish();
    let profile = profile.finish();
    assert!(state.is_ready());
    assert_eq!(state.position(), n);
    assert_eq!(
        state.prefill_origin(),
        forward::PrefillExecution::ShrinkingCed
    );
    assert!(report.fallback.is_none());
    report
        .counters
        .as_ref()
        .unwrap()
        .validate(&dependency_plan(&model.config, n))
        .unwrap();
    assert!(
        report.observed_incremental_metal_bytes.unwrap()
            <= report.memory.as_ref().unwrap().incremental_metal_bytes
    );
    let mut observation = observe.current.borrow_mut();
    let needs_pool = !fresh
        && maximum_decoder_rows(&dependency_plan(&model.config, n), arithmetic)
            .unwrap()
            .is_some();
    assert_eq!(observation.births, usize::from(needs_pool));
    assert!(
        observation
            .backing
            .iter()
            .chain(&observation.encoder)
            .chain(&observation.seeds)
            .all(|w| w.load().is_none()),
        "request scratch must drop before return"
    );
    let causal = incumbent::causal_bits(&state)
        .into_iter()
        .map(|row| row.into_iter().map(f32::to_bits).collect())
        .collect();
    (
        logits.into_iter().map(f32::to_bits).collect(),
        causal,
        report.counters.unwrap(),
        census,
        profile,
        std::mem::take(&mut observation.units),
    )
}

fn assert_fusion_pair(fused: &DriverArm, reference: &DriverArm, saved: usize) {
    assert_eq!(fused.0, reference.0, "complete head bits");
    assert_eq!(fused.1, reference.1, "all causal state bits");
    assert_eq!(
        fused.5, reference.5,
        "every packed output, current-n scratch and retained control word"
    );
    assert_eq!(
        dispatches(&fused.3),
        dispatches(&reference.3),
        "ordered arithmetic pipelines/grids"
    );
    let mut expected = reference.2.clone();
    expected.encoded_commands -= saved;
    expected.completed_commands -= saved;
    assert_eq!(fused.2, expected, "only submission count changes");
    let mut expanded = Vec::new();
    let mut compounds = 0;
    for (role, dims, submits) in commands(&fused.4) {
        if role == "ced_seed_decoder_rows" {
            let [layer, start, rows, seeds] = dims;
            assert!(seeds > 0);
            expanded.push(("ced_local_seed", [layer, start - seeds, seeds, 0], submits));
            expanded.push(("ced_decoder_rows", [layer, start, rows, 0], submits));
            compounds += submits;
        } else {
            expanded.push((role, dims, submits));
        }
    }
    expanded.sort_unstable();
    assert_eq!(
        expanded,
        commands(&reference.4),
        "exact roles and absolute ranges after splitting compounds"
    );
    assert_eq!(compounds, saved);
    for arm in [fused, reference] {
        assert_eq!(
            commands(&arm.4).iter().map(|r| r.2).sum::<usize>(),
            arm.2.completed_commands
        );
    }
}

#[test]
#[ignore = "self-leased <=128 MiB complete buffers; MTL debug validation; no real model"]
fn seed_fusion_matches_unfused_empty_partial_full_wrap_packed_one_second_chunk_and_scalar() {
    let _lease = require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    for (window, n, arithmetic, saved) in [
        (257, 128, Arithmetic::Production, 1),
        (257, 640, Arithmetic::Production, 3),
        (1025, 2542, Arithmetic::Production, 3),
        (2050, 2542, Arithmetic::Production, 1),
        (257, 2049, Arithmetic::Production, 2),
        (1, 128, Arithmetic::Production, 0),
        (3, 33, Arithmetic::ScalarNative, 0),
    ] {
        admit(&ctx, &fixture_config(window), n, n);
        objc2::rc::autoreleasepool(|_| {
            let weights = forward::toy::Weights::new(&ctx);
            let embedding = incumbent::embeddings(&ctx);
            let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
            configure(&mut model, window);
            let compare = || {
                let reference = objc2::rc::autoreleasepool(|_| {
                    let _unfused = incumbent::Unfused::begin();
                    driver_arm(&model, n, false, arithmetic)
                });
                let fused =
                    objc2::rc::autoreleasepool(|_| driver_arm(&model, n, false, arithmetic));
                assert_fusion_pair(&fused, &reference, saved);
            };
            if arithmetic.scalar() {
                compare();
            } else {
                prefill::with_batched_f32(compare);
            }
        });
    }
}

#[test]
#[ignore = "self-leased <=128 MiB complete buffers; MTL debug validation; no real model"]
fn compound_seed_nonfinite_is_sticky_after_decoder_and_route_failure_poison_and_drain() {
    let _lease = require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    admit(&ctx, &fixture_config(257), 640, 640);
    objc2::rc::autoreleasepool(|_| {
        let weights = forward::toy::Weights::new(&ctx);
        let embedding = incumbent::embeddings(&ctx);
        let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
        configure(&mut model, 257);
        prefill::with_batched_f32(|| {
            objc2::rc::autoreleasepool(|_| {
                let _ = driver_arm(&model, 640, false, Arithmetic::Production);
            });
            let baseline = ctx.current_allocated_size();
            for bad_seed in [true, false] {
                model.layers[3].bias[0] = if bad_seed { 0.0 } else { f32::NAN };
                let observe = Observe::begin();
                observe.current.borrow_mut().poison_seed = bad_seed;
                objc2::rc::autoreleasepool(|_| {
                    let mut state = model.new_state(640).unwrap();
                    let addresses = state.addresses;
                    let profile = profile::Scope::begin().unwrap();
                    let error = forward::ced::run(
                        &model,
                        &mut state,
                        &incumbent::tokens(640),
                        None,
                        Arithmetic::Production,
                    )
                    .unwrap_err();
                    let profile = profile.finish();
                    assert_eq!(
                        commands(&profile)
                            .iter()
                            .filter(|r| r.0 == "ced_seed_decoder_rows")
                            .map(|r| r.2)
                            .sum::<usize>(),
                        1
                    );
                    let o = observe.current.borrow();
                    assert_eq!(
                        o.compounds.len(),
                        1,
                        "checked GPU completion reached after full decoder encoding"
                    );
                    if bad_seed {
                        assert!(!o.poison_seed, "seed-only corruption hook was exercised");
                        assert_ne!(
                            o.compounds[0].0 & 4,
                            0,
                            "seed finite-status bit survives decoder dispatches"
                        );
                    } else {
                        assert_eq!(o.compounds[0], (0, false));
                        assert!(
                            error.to_string().contains("layer 3 row 0 route status"),
                            "{error}"
                        );
                    }
                    drop(o);
                    incumbent::assert_poisoned_rejects(&model, &mut state, addresses);
                });
                let o = observe.current.borrow();
                assert!(
                    o.encoder
                        .iter()
                        .chain(&o.backing)
                        .chain(&o.seeds)
                        .all(|w| w.load().is_none())
                );
                assert_eq!(o.births, 1);
                assert!(ctx.current_allocated_size() <= baseline + 2 * 1024 * 1024);
            }
        });
    });
}

#[test]
#[ignore = "self-leased <=128 MiB complete buffers; MTL debug validation; no real model"]
fn request_driver_matches_fresh_short_maximum_packed_singleton_and_scalar_tail() {
    let _lease = require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    admit(&ctx, &fixture_config(257), 2049, 2049);
    objc2::rc::autoreleasepool(|_| {
        let weights = forward::toy::Weights::new(&ctx);
        let embedding = incumbent::embeddings(&ctx);
        let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
        configure(&mut model, 257);
        prefill::with_batched_f32(|| {
            for (n, arithmetic) in [
                (1, Arithmetic::Production),
                (128, Arithmetic::Production),
                (640, Arithmetic::Production),
                (2049, Arithmetic::Production),
                (33, Arithmetic::ScalarNative),
            ] {
                let fresh = objc2::rc::autoreleasepool(|_| driver_arm(&model, n, true, arithmetic));
                let reuse =
                    objc2::rc::autoreleasepool(|_| driver_arm(&model, n, false, arithmetic));
                assert_eq!(reuse.0, fresh.0, "N={n} head bits");
                assert_eq!(reuse.1, fresh.1, "N={n} causal state bits");
                assert_eq!(reuse.2, fresh.2, "N={n} completion/work counters");
                assert_eq!(
                    dispatches(&reuse.3),
                    dispatches(&fresh.3),
                    "N={n} pipelines/grids"
                );
                assert_eq!(
                    commands(&reuse.4),
                    commands(&fresh.4),
                    "N={n} command roles"
                );
            }
        });
    });
}

#[test]
#[ignore = "self-leased <=128 MiB complete buffers; MTL debug validation; no real model"]
fn reuse_allocation_view_completion_error_cancel_and_unwind_drop_and_poison() {
    let _lease = require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    admit(&ctx, &fixture_config(257), 640, 640);
    objc2::rc::autoreleasepool(|_| {
        let weights = forward::toy::Weights::new(&ctx);
        let embedding = incumbent::embeddings(&ctx);
        let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
        configure(&mut model, 257);
        prefill::with_batched_f32(|| {
            // Warm pipeline caches before taking the request low-water sample.
            objc2::rc::autoreleasepool(|_| {
                let _ = driver_arm(&model, 640, false, Arithmetic::Production);
            });
            let baseline = ctx.current_allocated_size();
            for (role, action) in [
                ("ced_pool_allocate", Action::Error),
                ("ced_pool_allocate", Action::Panic),
                ("ced_pool_view", Action::Error),
                ("ced_pool_view", Action::Panic),
                ("ced_decoder_rows", Action::Completion),
                ("ced_seed_decoder_rows", Action::Completion),
                ("ced_seed_encoded", Action::Error),
                ("ced_seed_encoded", Action::Cancel),
                ("ced_seed_encoded", Action::Panic),
                ("ced_seed_complete", Action::Error),
                ("ced_seed_complete", Action::Cancel),
                ("ced_seed_complete", Action::Panic),
                ("ced_decoder_complete", Action::Error),
                ("ced_decoder_complete", Action::Cancel),
                ("ced_decoder_complete", Action::Panic),
                ("ced_commit", Action::Error),
            ] {
                let _unfused = (role == "ced_decoder_rows").then(incumbent::Unfused::begin);
                let observe = Observe::begin();
                let profile = profile::Scope::begin().unwrap();
                objc2::rc::autoreleasepool(|_| {
                    let mut state = model.new_state(640).unwrap();
                    let addresses = state.addresses;
                    let cancel = AtomicBool::new(false);
                    let fault = Fault::new(role, action);
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        forward::ced::run(
                            &model,
                            &mut state,
                            &incumbent::tokens(640),
                            Some(&cancel),
                            Arithmetic::Production,
                        )
                    }));
                    assert_eq!(fault.hits(), 1, "missing fault hook {role}");
                    if action == Action::Panic {
                        assert!(result.is_err());
                    } else {
                        assert!(result.unwrap().is_err());
                    }
                    drop(fault);
                    incumbent::assert_poisoned_rejects(&model, &mut state, addresses);
                });
                let profile = profile.finish();
                let observation = observe.current.borrow();
                if role == "ced_seed_encoded" {
                    assert!(
                        observation.compounds.is_empty(),
                        "encode-only fault must discard the unsubmitted compound"
                    );
                    assert!(
                        !commands(&profile)
                            .iter()
                            .any(|r| r.0 == "ced_seed_decoder_rows")
                    );
                } else if role == "ced_seed_complete" {
                    assert_eq!(
                        observation.compounds,
                        [(0, true)],
                        "seed-complete hook runs only after checked joint completion and FFN status"
                    );
                }
                assert!(
                    observation
                        .backing
                        .iter()
                        .chain(&observation.encoder)
                        .chain(&observation.seeds)
                        .all(|w| w.load().is_none()),
                    "{role}: resources survived request drain"
                );
                assert_eq!(observation.births, usize::from(role != "ced_pool_allocate"));
                assert!(
                    ctx.current_allocated_size() <= baseline + 2 * 1024 * 1024,
                    "{role}: request low water failed to recover"
                );
            }
        });
    });
}

#[test]
#[ignore = "self-leased <=128 MiB complete buffers; separate P4 slab; no real model"]
fn p4_selection_slabs_rebind_without_another_score_or_pair_allocation() {
    let _lease = require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    let c = Config {
        index_topk: 512,
        candidate_block: 128,
        candidate_topk: 128,
        ..fixture_config(257)
    };
    admit(&ctx, &c, MAX, 16384);
    objc2::rc::autoreleasepool(|_| {
        let weights = forward::toy::Weights::new(&ctx);
        let embedding = incumbent::embeddings(&ctx);
        let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
        configure(&mut model, c.window);
        model.config = c;
        let plan = dependency_plan_at_capacity(&model.config, MAX, 16384);
        let storage = Storage::new(&model, 16384, &plan).unwrap();
        let controls = storage.controls.checked_tile_view(0, MAX).unwrap();
        let mut pool = DecoderRowsPool {
            maximum: Some(0..MAX),
            backing: None,
        };
        let census = Allocations::begin();
        let view = pool
            .checked_rows(&model, 16384, &storage, &controls)
            .unwrap();
        let allocations = census.finish();
        for bytes in [512 * 16384 * 4, 512 * 4 * 512 * 2 * 4] {
            assert_eq!(
                allocations
                    .iter()
                    .filter(|r| r.requested_bytes == bytes)
                    .count(),
                1,
                "one maximum P4 score/pair slab"
            );
        }
        drop(view);
        for (index, n) in SEQUENCE.into_iter().enumerate() {
            let start = usize::from(index % 2 != 0);
            let controls = storage.controls.checked_tile_view(start, n).unwrap();
            let census = Allocations::begin();
            objc2::rc::autoreleasepool(|_| {
                let view = pool
                    .checked_rows(&model, 16384, &storage, &controls)
                    .unwrap();
                assert_eq!(view.hc.rows, n);
                assert_eq!(view.arithmetic, prefill::RowsArithmetic::Packed);
            });
            assert!(
                census.finish().is_empty(),
                "P4 checked view allocated Metal storage"
            );
        }
    });
}

#[test]
#[ignore = "self-leased M5 tensor tier; <=4 MiB explicit buffers; MTL debug validation"]
fn half_prefix_staged_subgroups_use_current_n_strides_and_match_fresh_bits() {
    let _lease = require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    assert!(
        metal::moe_grouped_tensor_selected(&ctx),
        "M5 tensor tier required"
    );
    let (groups, input, output) = (3, 64, 65);
    let elements = groups * input * MAX;
    let output_elements = groups * output * MAX;
    // Three BF16 projections, guarded F32 input and two guarded outputs,
    // one reused half maximum and one co-live fresh half maximum.
    let sizes = [
        (input * output * 2) as u64,
        (input * output * 2) as u64,
        (input * output * 2) as u64,
        ((elements + 32) * 4) as u64,
        ((output_elements + 32) * 4) as u64,
        ((output_elements + 32) * 4) as u64,
        (elements * 2) as u64,
        (elements * 2) as u64,
    ];
    let priced: u64 = sizes
        .into_iter()
        .map(|n| ctx.price_shared_buffer_upper(n).unwrap().priced_upper_bytes)
        .sum();
    assert!(priced <= 4 * 1024 * 1024 && priced <= LIMIT);
    let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
        priced,
        4 * 1024 * 1024,
        0,
        ctx.memory_signals(),
        true,
    );
    assert!(admission.refusal().is_none());
    objc2::rc::autoreleasepool(|_| {
        let projections = (0..groups)
            .map(|group| {
                let bytes = (0..input * output)
                    .flat_map(|i| {
                        half::bf16::from_f32(((i * 19 + group * 7) % 101) as f32 / 8192.0 - 0.006)
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>();
                forward::Linear {
                    ctx: &ctx,
                    tensor: MetalTensor::from_bytes(
                        &ctx,
                        &bytes,
                        vec![input as u64, output as u64],
                        GgmlType::BF16,
                    )
                    .unwrap(),
                    native_control: None,
                    input,
                    output,
                }
            })
            .collect::<Vec<_>>();
        let source = prefill::allocate(&ctx, elements + 32, 1, GgmlType::F32).unwrap();
        let expected = prefill::allocate(&ctx, output_elements + 32, 1, GgmlType::F32).unwrap();
        let actual = prefill::allocate(&ctx, output_elements + 32, 1, GgmlType::F32).unwrap();
        let backing = prefill::HalfScratch::new(&ctx, elements).unwrap().unwrap();
        let execute = |f: &mut dyn FnMut(&KernelEncoder)| {
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            f(&enc);
            enc.end();
            metal::commit_and_wait(&command).unwrap();
        };
        fill(&source, POISON);
        let mut maximum_half = None;
        execute(&mut |enc| {
            maximum_half = Some(
                backing
                    .convert(
                        &ctx,
                        enc,
                        &source.view_subrange(16, vec![elements as u64]),
                        elements,
                    )
                    .unwrap(),
            );
        });
        let maximum_half = maximum_half.unwrap();
        for n in SEQUENCE {
            objc2::rc::autoreleasepool(|_| {
                for t in [&source, &expected, &actual] {
                    fill(t, POISON);
                }
                unsafe {
                    std::slice::from_raw_parts_mut(
                        maximum_half
                            .buffer
                            .contents()
                            .as_ptr()
                            .cast::<u16>()
                            .add((maximum_half.offset / 2) as usize),
                        elements,
                    )
                    .fill(0x7eaf);
                }
                let count = groups * input * n;
                let values = (0..count)
                    .map(|i| {
                        let group = i / (input * n);
                        ((i * 31 + group * 17) % 113) as f32 / 71.0 - 0.75
                    })
                    .collect::<Vec<_>>();
                let x = source.view_subrange(16, vec![count as u64]);
                write(&x, &values).unwrap();
                let census = Allocations::begin();
                let prefix = backing.checked_prefix(count).unwrap();
                assert!(census.finish().is_empty());
                let fresh = prefill::HalfScratch::new(&ctx, count).unwrap().unwrap();
                let census = Census::begin();
                let mut reused_half = None;
                execute(&mut |enc| {
                    let a = prefix.convert(&ctx, enc, &x, count).unwrap();
                    let b = fresh.convert(&ctx, enc, &x, count).unwrap();
                    for (group, projection) in projections.iter().enumerate() {
                        let x = x.view_subrange(
                            (group * input * n) as u64,
                            vec![input as u64, n as u64],
                        );
                        let a =
                            a.view_subrange((group * input * n) as u64, vec![(input * n) as u64]);
                        let b =
                            b.view_subrange((group * input * n) as u64, vec![(input * n) as u64]);
                        let offset = (16 + group * output * n) as u64;
                        let expected =
                            expected.view_subrange(offset, vec![output as u64, n as u64]);
                        let actual = actual.view_subrange(offset, vec![output as u64, n as u64]);
                        projection
                            .encode_rows_with_arithmetic_and_half(
                                enc,
                                &x,
                                Some(&b),
                                &expected,
                                n,
                                prefill::RowsArithmetic::Packed,
                            )
                            .unwrap();
                        projection
                            .encode_rows_with_arithmetic_and_half(
                                enc,
                                &x,
                                Some(&a),
                                &actual,
                                n,
                                prefill::RowsArithmetic::Packed,
                            )
                            .unwrap();
                    }
                    reused_half = Some(a);
                });
                let dispatch = census.finish();
                assert_eq!(
                    dispatch
                        .iter()
                        .filter(|r| r.kernel.contains("f16act"))
                        .count(),
                    2 * groups
                );
                assert_eq!(
                    bits(&actual),
                    bits(&expected),
                    "n={n}: subgroup output and all canaries"
                );
                let result = bits(&actual);
                assert!(
                    result[..16]
                        .iter()
                        .chain(&result[16 + groups * output * n..])
                        .all(|&v| v == POISON)
                );
                let a = reused_half.unwrap();
                let half_bits = unsafe {
                    std::slice::from_raw_parts(
                        a.buffer
                            .contents()
                            .as_ptr()
                            .cast::<u16>()
                            .add((a.offset / 2) as usize),
                        count,
                    )
                };
                assert_eq!(
                    half_bits,
                    values
                        .iter()
                        .map(|&v| half::f16::from_f32(v).to_bits())
                        .collect::<Vec<_>>()
                );
            });
        }
    });
}

fn wide_config() -> Config {
    Config {
        heads: 16,
        head_dim: 512,
        query_rank: 64,
        groups: 4,
        output_rank: 32,
        ..fixture_config(65)
    }
}

fn wide_projection_shapes(c: &Config) -> Vec<(usize, usize)> {
    let mut shapes = vec![
        (c.hidden, c.query_rank),
        (c.query_rank, c.heads * c.head_dim),
        (c.hidden, c.head_dim),
        (c.groups * c.output_rank, c.hidden),
        (c.hidden, c.head_dim),
        (c.hidden, c.head_dim),
        (c.query_rank, c.index_heads * c.index_dim),
        (c.head_dim, c.index_dim),
    ];
    shapes.extend((0..c.groups).map(|_| (c.heads * c.head_dim / c.groups, c.output_rank)));
    shapes
}

fn wide_bound(price: &impl Fn(u64) -> u64) -> u64 {
    let c = wide_config();
    explicit_bound(&c, 129, 129, price)
        + 6 * wide_projection_shapes(&c)
            .into_iter()
            .map(|(input, output)| price((input * output * 2) as u64))
            .sum::<u64>()
}

#[test]
fn wide_lineage_fixture_is_separate_from_p4_and_below_128_mib() {
    let price = |n| if n == 0 { 0 } else { n + 16384 };
    assert!(wide_bound(&price) <= LIMIT);
}

fn bf16_projection<'a>(
    ctx: &'a MetalContext,
    input: usize,
    output: usize,
    seed: usize,
) -> forward::Linear<'a> {
    let bytes = (0..input * output)
        .flat_map(|i| {
            half::bf16::from_f32(
                (((i * 19 + seed * 7) % 61) as f32 - 30.0) / (64.0 * (input as f32).sqrt()),
            )
            .to_le_bytes()
        })
        .collect::<Vec<_>>();
    forward::Linear {
        ctx,
        tensor: MetalTensor::from_bytes(
            ctx,
            &bytes,
            vec![input as u64, output as u64],
            GgmlType::BF16,
        )
        .unwrap(),
        native_control: None,
        input,
        output,
    }
}

fn configure_wide(model: &mut Forward<'_>) {
    let c = wide_config();
    configure(model, c.window);
    for (id, layer) in model.layers.iter_mut().enumerate() {
        let p = |input, output, salt| bf16_projection(model.ctx, input, output, id * 31 + salt);
        let a = &mut layer.attention;
        a.query_a = p(c.hidden, c.query_rank, 1);
        a.query_b = p(c.query_rank, c.heads * c.head_dim, 2);
        a.query_norm = vec![1.0; c.query_rank];
        a.kv = p(c.hidden, c.head_dim, 3);
        a.kv_norm = vec![1.0; c.head_dim];
        a.sink = vec![0.0; c.heads];
        a.output_a = (0..c.groups)
            .map(|group| p(c.heads * c.head_dim / c.groups, c.output_rank, group + 4))
            .collect();
        a.output_b = p(c.groups * c.output_rank, c.hidden, 9);
        a.head_dim = c.head_dim;
        if let Some(compressor) = &mut a.compressor {
            compressor.kv = p(c.hidden, c.head_dim, 10);
            if compressor.gate.is_some() {
                compressor.gate = Some(p(c.hidden, c.head_dim, 11));
            }
            compressor.norm = vec![1.0; c.head_dim];
        }
        if let Some(index) = &mut a.index {
            index.query = p(c.query_rank, c.index_heads * c.index_dim, 12);
            if index.key.is_some() {
                index.key = Some(p(c.head_dim, c.index_dim, 13));
            }
        }
    }
    model.config = c;
}

#[test]
#[ignore = "self-leased M5 wide FullTile lineage; <=128 MiB explicit buffers; no real model"]
fn wide_full_tile_lineage_and_staged_groups_match_fresh_through_packed_singleton() {
    let _lease = require_debug_lease();
    let ctx = MetalContext::new().unwrap();
    assert!(
        metal::moe_grouped_tensor_selected(&ctx),
        "M5 tensor tier required"
    );
    let price = |n| ctx.price_shared_buffer_upper(n).unwrap().priced_upper_bytes;
    let bytes = wide_bound(&price);
    assert!(bytes <= LIMIT);
    let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
        bytes,
        32 * 1024 * 1024,
        0,
        ctx.memory_signals(),
        true,
    );
    assert!(admission.refusal().is_none());
    objc2::rc::autoreleasepool(|_| {
        let weights = forward::toy::Weights::new(&ctx);
        let embedding = incumbent::embeddings(&ctx);
        let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
        configure_wide(&mut model);
        let fresh =
            objc2::rc::autoreleasepool(|_| driver_arm(&model, 129, true, Arithmetic::Production));
        let reuse =
            objc2::rc::autoreleasepool(|_| driver_arm(&model, 129, false, Arithmetic::Production));
        assert_eq!(reuse.0, fresh.0, "wide production head bits");
        assert_eq!(reuse.1, fresh.1, "wide production causal state bits");
        assert_eq!(reuse.2, fresh.2);
        assert_eq!(dispatches(&reuse.3), dispatches(&fresh.3));
        assert_eq!(commands(&reuse.4), commands(&fresh.4));
        let unfused = objc2::rc::autoreleasepool(|_| {
            let _unfused = incumbent::Unfused::begin();
            driver_arm(&model, 129, false, Arithmetic::Production)
        });
        assert_fusion_pair(&reuse, &unfused, 2);
        for suffix in ["score_tensor", "pv_tensor"] {
            let name = format!("kernel_deepseek_v41_attention_{suffix}");
            assert!(
                reuse
                    .3
                    .iter()
                    .any(|r| r.kernel == name && r.grid_height == 1),
                "compact singleton must retain the original FullTile Tensor route: {name}"
            );
        }
        assert!(
            reuse.3.iter().any(|r| r.kernel.contains("f16act")),
            "staged subgroup projections not exercised"
        );
    });
}
