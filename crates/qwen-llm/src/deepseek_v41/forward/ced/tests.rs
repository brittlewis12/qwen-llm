use super::*;
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

pub(in super::super) mod chunked;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Error,
    Panic,
    Cancel,
    Completion,
}

struct Injection {
    role: &'static str,
    action: Action,
    hits: Rc<Cell<usize>>,
}

thread_local! {
    static INJECTION: RefCell<Option<Injection>> = const { RefCell::new(None) };
}

struct Scope {
    previous: Option<Injection>,
    hits: Rc<Cell<usize>>,
    _thread: PhantomData<Rc<()>>,
}

impl Scope {
    fn new(role: &'static str, action: Action) -> Self {
        let hits = Rc::new(Cell::new(0));
        let previous = INJECTION.with(|slot| {
            slot.replace(Some(Injection {
                role,
                action,
                hits: Rc::clone(&hits),
            }))
        });
        Self {
            previous,
            hits,
            _thread: PhantomData,
        }
    }

    fn hits(&self) -> usize {
        self.hits.get()
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        INJECTION.with(|slot| slot.replace(self.previous.take()));
    }
}

fn take_action(role: &'static str, completion: bool) -> Option<Action> {
    INJECTION.with(|slot| {
        let mut slot = slot.borrow_mut();
        let injection = slot.as_ref()?;
        if injection.role != role || (injection.action == Action::Completion) != completion {
            return None;
        }
        let injection = slot.take().unwrap();
        injection.hits.set(injection.hits.get() + 1);
        Some(injection.action)
    })
}

// The caller must keep this guard alive through commit_and_wait.
pub(in super::super) fn completion_fault(
    role: &'static str,
) -> Option<crate::metal::FailNextCompletionCheck> {
    take_action(role, true).map(|_| crate::metal::fail_next_completion_check())
}

pub(in super::super) fn checkpoint(role: &'static str, cancel: Option<&AtomicBool>) -> Result<()> {
    match take_action(role, false) {
        Some(Action::Error) => Err(format!("injected CED checkpoint: {role}")),
        Some(Action::Panic) => panic!("injected CED checkpoint panic: {role}"),
        Some(Action::Cancel) => {
            cancel
                .expect("cancellation injection requires the caller's cancel flag")
                .store(true, Ordering::Relaxed);
            cancelled(cancel)
        }
        Some(Action::Completion) => unreachable!(),
        None => Ok(()),
    }
}

#[test]
fn scoped_hooks_are_one_shot_nested_thread_local_and_unwind_safe() {
    assert!(checkpoint("ced_begin", None).is_ok());
    assert!(completion_fault("ced_encoder_rows").is_none());
    let outer = Scope::new("ced_begin", Action::Error);
    {
        let inner = Scope::new("ced_commit", Action::Cancel);
        assert!(checkpoint("ced_begin", None).is_ok());
        let cancel = AtomicBool::new(false);
        assert!(
            checkpoint("ced_commit", Some(&cancel))
                .unwrap_err()
                .contains("cancelled")
        );
        assert!(cancel.load(Ordering::Relaxed));
        assert_eq!(inner.hits(), 1);
        assert!(checkpoint("ced_commit", None).is_ok());
    }
    std::thread::spawn(|| {
        assert!(checkpoint("ced_begin", None).is_ok());
        assert!(completion_fault("ced_encoder_rows").is_none());
    })
    .join()
    .unwrap();
    assert!(
        catch_unwind(|| {
            let inner = Scope::new("ced_publication", Action::Panic);
            assert!(completion_fault("ced_publication").is_none());
            let _ = checkpoint("ced_publication", None);
            drop(inner);
        })
        .is_err()
    );
    assert!(
        checkpoint("ced_begin", None)
            .unwrap_err()
            .contains("ced_begin")
    );
    assert_eq!(outer.hits(), 1);
    assert!(checkpoint("ced_begin", None).is_ok());
    drop(outer);
    {
        let scope = Scope::new("ced_encoder_scalar", Action::Completion);
        assert!(checkpoint("ced_encoder_scalar", None).is_ok());
        assert!(completion_fault("ced_decoder_scalar").is_none());
        let guard = completion_fault("ced_encoder_scalar").expect("selected completion role");
        assert_eq!(scope.hits(), 1);
        assert!(completion_fault("ced_encoder_scalar").is_none());
        drop(guard);
    }
    assert!(checkpoint("ced_begin", None).is_ok());
    assert!(completion_fault("ced_encoder_scalar").is_none());
}

fn cpu_state() -> State {
    State {
        gpu_step: None,
        attention: attention::Session::default(),
        addresses: engram::AddressState::new(2).unwrap(),
        max_context: 33,
        phase: StatePhase::Ready,
        prefill_origin: PrefillExecution::FullForward,
        identity: Arc::new(()),
    }
}

#[test]
fn provisional_guard_commits_addresses_and_origin_only_on_success() {
    let mut state = cpu_state();
    let before = state.addresses;
    let (_, next) = before.step(engram::Token::Compressed(5)).unwrap();
    assert!(state.is_ready());
    assert!(!state.is_poisoned());
    let guard = Provisional::begin(&mut state);
    assert_eq!(guard.state.phase, StatePhase::Provisional);
    assert!(!guard.state.is_ready());
    assert!(!guard.state.is_poisoned());
    assert_eq!(guard.state.addresses, before);
    assert_eq!(guard.state.position(), 0);
    guard.commit(next);
    assert!(state.is_ready());
    assert_eq!(state.addresses, next);
    assert_eq!(state.position(), 1);
    assert_eq!(state.prefill_origin(), PrefillExecution::ShrinkingCed);
}

#[test]
fn provisional_guard_error_and_unwind_poison_without_committing_addresses() {
    for unwind in [false, true] {
        let mut state = cpu_state();
        state.addresses = state
            .addresses
            .step(engram::Token::Compressed(3))
            .unwrap()
            .1;
        let before = state.addresses;
        let mut operation = || -> Result<()> {
            let guard = Provisional::begin(&mut state);
            let (_, staged) = guard
                .state
                .addresses
                .step(engram::Token::Compressed(5))
                .unwrap();
            assert_ne!(staged, before);
            assert_eq!(guard.state.addresses, before);
            if unwind {
                panic!("provisional unwind");
            }
            Err("provisional error".into())
        };
        if unwind {
            assert!(catch_unwind(AssertUnwindSafe(operation)).is_err());
        } else {
            assert_eq!(operation().unwrap_err(), "provisional error");
        }
        assert!(state.is_poisoned());
        assert!(!state.is_ready());
        assert_eq!(state.position(), 1);
        assert_eq!(state.addresses, before);
        assert_eq!(state.prefill_origin(), PrefillExecution::FullForward);
    }
}

fn schedule() -> super::super::super::gguf::Schedule {
    super::super::super::gguf::Schedule {
        layers: 6,
        kv_source_layers: vec![1, 3],
        index_source_layers: vec![1, 3, 4],
        compression_ratios: vec![0, 2, 2, 1, 1, 1],
        candidate_source_layer: Some(3),
    }
}

fn hash_config() -> engram::HashConfig {
    let mut hash = engram::HashConfig::default();
    hash.layers.truncate(1);
    hash.layers[0].layer = 0;
    hash
}

fn plan(n: usize, capacity: usize) -> CedPlan {
    let mut config = toy::config();
    config.window = 3;
    config.index_topk = 2;
    CedPlan::shrinking(
        &config,
        &schedule(),
        &hash_config(),
        &CedRequest::cold(n, capacity),
    )
    .unwrap()
}

fn complete_counters(plan: &CedPlan) -> CedCounters {
    let mut counters = CedCounters::default();
    let schedule = schedule();
    for layer in plan.encoder_layers.clone() {
        counters
            .encoder(layer, plan.prompt_rows, schedule.produces_index(layer))
            .unwrap();
    }
    counters
        .publication(plan.publication_layer, plan.prompt_rows)
        .unwrap();
    for layer in &plan.decoder {
        counters.seed(layer.layer, layer.seed_rows.len()).unwrap();
        counters
            .decoder(
                layer.layer,
                layer.output_rows.len(),
                schedule.produces_index(layer.layer),
            )
            .unwrap();
    }
    counters.encoded_commands = 7;
    counters.completed_commands = 7;
    counters
}

#[test]
fn counters_require_completed_commands_and_exact_layer_coverage() {
    let p = plan(17, 33);
    let counters = complete_counters(&p);
    counters.validate(&p).unwrap();
    let mut incomplete = counters.clone();
    incomplete.completed_commands -= 1;
    assert!(incomplete.validate(&p).is_err());
    for layer in 0..6 {
        let mut missing = counters.clone();
        missing.layers.retain(|count| count.layer != layer);
        assert!(missing.validate(&p).is_err(), "missing layer {layer}");
        let mut wrong = counters.clone();
        let count = wrong
            .layers
            .iter_mut()
            .find(|count| count.layer == layer)
            .unwrap();
        if layer < p.publication_layer {
            count.encoder_output_rows -= 1;
        } else {
            count.decoder_output_rows -= 1;
        }
        assert!(wrong.validate(&p).is_err(), "incomplete layer {layer}");
    }
    let mut duplicate_work = counters.clone();
    duplicate_work.decoder(3, 1, true).unwrap();
    assert!(duplicate_work.validate(&p).is_err());
}

#[test]
fn counters_reject_duplicate_and_extraneous_layer_records() {
    let p = plan(17, 33);
    for id in [0, 3, 6] {
        let mut counters = complete_counters(&p);
        let mut duplicate = counters.layers[0].clone();
        duplicate.layer = id;
        counters.layers.push(duplicate);
        assert!(
            counters.validate(&p).is_err(),
            "duplicate/extraneous layer record {id}"
        );
    }
}

fn vary(linear: &mut Linear<'_>, seed: usize) {
    assert_eq!(linear.tensor.dtype, GgmlType::F32);
    let values = (0..linear.input * linear.output)
        .map(|i| {
            (((i * 17 + seed * 13 + i / linear.input * 7) % 61) as f32 - 30.0)
                / (20.0 * (linear.input as f32).sqrt())
        })
        .collect::<Vec<_>>();
    linear.tensor = upload(
        linear.ctx,
        &values,
        vec![linear.input as u64, linear.output as u64],
    )
    .unwrap();
}

fn embeddings(ctx: &MetalContext) -> MetalTensor {
    upload(
        ctx,
        &(0..256)
            .map(|i| ((i * 7 % 37) as f32 - 18.0) / 19.001)
            .collect::<Vec<_>>(),
        vec![32, 8],
    )
    .unwrap()
}

// Same real operators and nonuniform recipe as prefill's dynamic-schedule witness.
// No rowwise/grouped/F32-linear override is installed: ScalarNative uses the
// scalar attention, scalar native expert operands, and complete scalar mHC path.
fn nonuniform_model<'a>(
    weights: &'a toy::Weights,
    ctx: &'a MetalContext,
    embeddings: &'a MetalTensor,
) -> Forward<'a> {
    let mut model = weights.model(ctx);
    let engram = model.layers[1].engram.take();
    model.layers = model
        .layers
        .into_iter()
        .enumerate()
        .filter_map(|(id, layer)| [0, 2, 3, 20, 24, 25].contains(&id).then_some(layer))
        .collect();
    model.layers[0].engram = engram;
    model.schedule = schedule();
    model.engram_hash = hash_config();
    model.config.window = 3;
    model.config.index_topk = 2;
    vary_model(&mut model, embeddings);
    model.schedule.validate().unwrap();
    model.engram_hash.validate().unwrap();
    assert_eq!(model.layers.len(), 6);
    assert!(model.layers[0].engram.is_some());
    assert!(model.layers[1..].iter().all(|layer| layer.engram.is_none()));
    model
}

fn vary_model<'a>(model: &mut Forward<'a>, embeddings: &'a MetalTensor) {
    model.embedding = embeddings;
    vary(&mut model.head, 91);
    for (id, layer) in model.layers.iter_mut().enumerate() {
        vary(&mut layer.attn_hc.projection, id + 1);
        vary(&mut layer.ffn_hc.projection, id + 3);
        vary(&mut layer.router, id + 7);
        for (j, projection) in layer.shared.iter_mut().enumerate() {
            vary(projection, id + j + 11);
        }
        let a = &mut layer.attention;
        vary(&mut a.query_a, id + 19);
        vary(&mut a.query_b, id + 23);
        vary(&mut a.kv, id + 29);
        for (j, projection) in a.output_a.iter_mut().enumerate() {
            vary(projection, id + j + 31);
        }
        vary(&mut a.output_b, id + 37);
        if let Some(compressor) = a.compressor.as_mut() {
            vary(&mut compressor.kv, id + 41);
            if let Some(gate) = compressor.gate.as_mut() {
                vary(gate, id + 43);
            }
        }
        if let Some(index) = a.index.as_mut() {
            index.topk = model.config.index_topk;
            vary(&mut index.query, id + 47);
            vary(&mut index.weights, id + 53);
            if let Some(key) = index.key.as_mut() {
                vary(key, id + 59);
            }
        }
    }
}

fn tokens(n: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 3 + i / 5 + 1) % 8) as u32).collect()
}

fn assert_bits(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: extent");
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "{label}[{i}]: non-finite {a} / {b}"
        );
        assert_eq!(a.to_bits(), b.to_bits(), "{label}[{i}]: {a} != {b}");
    }
}

fn causal_bits(state: &State) -> Vec<Vec<f32>> {
    let attention = state.gpu_step.as_ref().unwrap().attention();
    attention
        .validate_snapshot_position(state.position())
        .unwrap();
    attention
        .causal_regions(state.position())
        .unwrap()
        .into_iter()
        .map(|(tensor, count)| f32_slice(tensor).unwrap()[..count].to_vec())
        .collect()
}

fn assert_causal(a: &State, b: &State, label: &str) {
    assert!(a.is_ready() && b.is_ready(), "{label}: non-ready state");
    assert_eq!(a.position(), b.position(), "{label}: position");
    assert_eq!(
        a.addresses, b.addresses,
        "{label}: Engram addresses/history"
    );
    let a = causal_bits(a);
    let b = causal_bits(b);
    assert_eq!(
        a.len(),
        9,
        "all rings plus two owners' KV/keys/pending operands"
    );
    assert_eq!(a.len(), b.len(), "{label}: region coverage");
    for (region, (a, b)) in a.iter().zip(&b).enumerate() {
        assert_bits(a, b, &format!("{label}, causal region {region}"));
    }
}

fn assert_report(report: &PrefillReport, n: usize, capacity: usize, scalar: bool) {
    let p = plan(n, capacity);
    assert_eq!(report.execution, PrefillExecution::ShrinkingCed);
    assert!(
        report.fallback.is_none(),
        "unexpected fallback: {:?}",
        report.fallback
    );
    let counters = report.counters.as_ref().expect("executed CED counters");
    counters.validate(&p).unwrap();
    let mut expected = complete_counters(&p);
    expected.encoded_commands = if scalar {
        n + p.work.decoder_seed_rows + p.work.decoder_output_rows
    } else {
        n.div_ceil(prefill::DEFAULT_CHUNK_ROWS)
            + p.decoder
                .iter()
                .map(|l| {
                    l.seed_rows.len().div_ceil(prefill::DEFAULT_CHUNK_ROWS)
                        + l.output_rows.len().div_ceil(prefill::DEFAULT_CHUNK_ROWS)
                })
                .sum::<usize>()
    };
    expected.completed_commands = expected.encoded_commands;
    let mut actual = counters.clone();
    actual.layers.sort_by_key(|l| l.layer);
    expected.layers.sort_by_key(|l| l.layer);
    assert_eq!(
        actual, expected,
        "actual executed work, including index producers"
    );
    assert!(
        p.boundary_rows.start > 0,
        "fixture must actually truncate dependencies"
    );
    assert!(p.work.total_output_rows < p.work.full_forward_output_rows);
    let memory = report
        .memory
        .as_ref()
        .expect("normal CED memory admission estimate");
    assert_eq!(
        memory.retained_residual_bytes,
        p.memory.boundary_residual_bytes
    );
    assert_eq!(memory.retained_mix_bytes, p.memory.boundary_mix_bytes);
    assert_eq!(memory.retained_candidate_bytes, p.memory.candidate_bytes);
    assert_eq!(memory.retained_selected_bytes, p.memory.selected_bytes);
    assert!(memory.incremental_metal_bytes > 0);
    assert!(memory.incremental_cpu_bytes > 0);
    let observed = report
        .observed_incremental_metal_bytes
        .expect("observed CED memory");
    assert!(
        observed <= memory.incremental_metal_bytes,
        "unpriced CED allocation: {observed} > {}",
        memory.incremental_metal_bytes
    );
}

fn assert_native_census(rows: &[metal::DispatchCensusRow], output_rows: usize) {
    assert_eq!(
        rows.iter()
            .filter(|r| r.kernel == "kernel_expert_native_mxfp4_f32")
            .count(),
        3 * output_rows
    );
    assert!(
        rows.iter()
            .any(|r| r.kernel == "kernel_deepseek_v41_attention_score")
    );
    for row in rows {
        assert!(
            !row.kernel.contains("grouped")
                && !row.kernel.contains("_f16")
                && !row.kernel.contains("_half")
                && !row.kernel.contains("_mma"),
            "scalar NativeF32 unexpectedly dispatches {}",
            row.kernel
        );
    }
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn scalar_native_full_vs_ced_bitwise_state_logits_restore_growth_and_256_steps() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("CED GPU lease");
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let embedding = embeddings(&ctx);
    let model = nonuniform_model(&weights, &ctx, &embedding);
    for n in [17, 33] {
        let input = tokens(n);
        let mut full = model.new_state(n + 256).unwrap();
        let mut ced = model.new_state(n).unwrap();
        metal::dispatch_census_begin();
        for &token in &input[..n - 1] {
            model.advance(&mut full, token, None).unwrap();
        }
        let expected = model.step(&mut full, input[n - 1], None).unwrap();
        let census = metal::dispatch_census_take();
        assert_native_census(&census, n * model.layers.len());
        metal::dispatch_census_begin();
        let result = run(&model, &mut ced, &input, None, Arithmetic::ScalarNative);
        let census = metal::dispatch_census_take();
        let (actual, report) = result.unwrap();
        assert_native_census(&census, plan(n, n).work.total_output_rows);
        assert_report(&report, n, n, true);
        assert_bits(&actual, &expected, &format!("N={n} final full logits"));
        assert_causal(&full, &ced, &format!("N={n} final state"));
        assert_eq!(full.prefill_origin(), PrefillExecution::FullForward);
        assert_eq!(ced.prefill_origin(), PrefillExecution::ShrinkingCed);
        let snapshot = model.capture_snapshot(&ced).unwrap();
        assert_eq!(snapshot.position(), n);
        assert_eq!(snapshot.prefill_origin(), PrefillExecution::ShrinkingCed);
        let mut restored = model.new_state(n + 256).unwrap();
        let mut grown = model.new_state(n).unwrap();
        for state in [&mut restored, &mut grown] {
            for token in [7, 2, 0, 6] {
                model.advance(state, token, None).unwrap();
            }
            model.restore_snapshot(state, &snapshot).unwrap();
            assert_causal(&ced, state, "restored CED snapshot into dirty state");
            assert_eq!(state.prefill_origin(), PrefillExecution::ShrinkingCed);
        }
        for state in [&mut ced, &mut grown] {
            model.ensure_capacity(state, n + 256).unwrap();
            assert!(state.capacity() >= n + 256);
            assert_eq!(state.prefill_origin(), PrefillExecution::ShrinkingCed);
            assert_causal(&full, state, "CED capacity growth");
        }
        for (step, token) in tokens(256).into_iter().enumerate() {
            let expected = model.step(&mut full, token, None).unwrap();
            for (name, state) in [
                ("ced", &mut ced),
                ("restored", &mut restored),
                ("grown", &mut grown),
            ] {
                let actual = model.step(state, token, None).unwrap();
                let label = format!("N={n} teacher-forced step {step} {name}");
                assert_bits(&actual, &expected, &label);
                assert_causal(&full, state, &label);
                assert_eq!(state.prefill_origin(), PrefillExecution::ShrinkingCed);
            }
        }
        assert_eq!(
            model.capture_snapshot(&grown).unwrap().prefill_origin(),
            PrefillExecution::ShrinkingCed
        );
    }
}

fn mismatches(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn production_rows_preserve_encoder_and_owner_globals_report_decoder_differences() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("CED GPU lease");
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let embedding = embeddings(&ctx);
    let model = nonuniform_model(&weights, &ctx, &embedding);
    let n = 33;
    let mut full = model.new_state(n).unwrap();
    let mut ced = model.new_state(n).unwrap();
    let input = tokens(n);
    let expected = model.prefill(&mut full, &input, None).unwrap();
    let full_before = causal_bits(&full);
    let (actual, report) = model
        .prefill_with_execution(&mut ced, &input, None, PrefillExecution::ShrinkingCed)
        .unwrap();
    assert_report(&report, n, n, false);
    assert_eq!(actual.len(), model.config.vocab);
    assert!(actual.iter().chain(&expected).all(|v| v.is_finite()));
    assert_eq!(full.addresses, ced.addresses);
    let a = causal_bits(&full);
    let b = causal_bits(&ced);
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(&b).flatten().all(|v| v.is_finite()));
    let stride = model.config.window * model.config.head_dim;
    eprintln!(
        "production N=33 six-layer CED vs untouched full: logit bit mismatches={}/{}; decoder ring mismatches={:?}. All rings are required bitwise for this fixture; head differences remain characterization, not qualified parity.",
        mismatches(&actual, &expected),
        actual.len(),
        (3..6)
            .map(|layer| mismatches(
                &a[0][layer * stride..(layer + 1) * stride],
                &b[0][layer * stride..(layer + 1) * stride]
            ))
            .collect::<Vec<_>>()
    );
    assert_bits(
        &a[0][..3 * stride],
        &b[0][..3 * stride],
        "production ALL encoder local rings",
    );
    for layer in 3..6 {
        assert_bits(
            &a[0][layer * stride..(layer + 1) * stride],
            &b[0][layer * stride..(layer + 1) * stride],
            &format!("production N=33 six-layer decoder ring {layer}"),
        );
    }
    for region in 1..a.len() {
        assert_bits(
            &a[region],
            &b[region],
            &format!("production owner KV/key/pending region {region}"),
        );
    }
    for (before, after) in full_before.iter().zip(&a) {
        assert_bits(before, after, "untouched full reference");
    }
}

fn assert_poisoned_rejects(
    model: &Forward<'_>,
    state: &mut State,
    addresses: engram::AddressState,
) {
    assert_eq!(state.phase, StatePhase::Poisoned);
    assert_eq!(state.position(), 0);
    assert_eq!(state.addresses, addresses);
    assert_eq!(state.prefill_origin(), PrefillExecution::FullForward);
    assert!(model.step(state, 0, None).is_err());
    assert!(model.advance(state, 0, None).is_err());
    assert!(model.prefill(state, &[0], None).is_err());
    assert!(
        model
            .prefill_with_execution(state, &[0], None, PrefillExecution::ShrinkingCed)
            .is_err()
    );
    assert!(model.capture_snapshot(state).is_err());
    assert!(model.snapshot_payload_bytes(state).is_err());
    let required = state.capacity() + 1;
    assert!(model.ensure_capacity(state, required).is_err());
    assert_eq!(state.phase, StatePhase::Poisoned);
    assert_eq!(state.addresses, addresses);
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn checkpoints_error_cancel_and_unwind_poison_every_ced_phase() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("CED GPU lease");
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let embedding = embeddings(&ctx);
    let model = nonuniform_model(&weights, &ctx, &embedding);
    let input = tokens(9);
    for arithmetic in [Arithmetic::ScalarNative, Arithmetic::Production] {
        for role in [
            "ced_begin",
            "ced_publication",
            "ced_encoder_complete",
            "ced_seed_complete",
            "ced_decoder_complete",
            "ced_commit",
        ] {
            for action in [Action::Error, Action::Cancel, Action::Panic] {
                let mut state = model.new_state(input.len()).unwrap();
                let addresses = state.addresses;
                let cancel = AtomicBool::new(false);
                let scope = Scope::new(role, action);
                let result = catch_unwind(AssertUnwindSafe(|| {
                    run(&model, &mut state, &input, Some(&cancel), arithmetic)
                }));
                assert_eq!(
                    scope.hits(),
                    1,
                    "missing runtime hook {role}, {arithmetic:?}, {action:?}"
                );
                match action {
                    Action::Panic => assert!(result.is_err(), "checkpoint {role} did not unwind"),
                    Action::Error => {
                        assert!(result.unwrap().unwrap_err().to_string().contains(role))
                    }
                    Action::Cancel => {
                        assert!(cancel.load(Ordering::Relaxed));
                        assert!(
                            result
                                .unwrap()
                                .unwrap_err()
                                .to_string()
                                .contains("cancelled")
                        );
                    }
                    Action::Completion => unreachable!(),
                }
                drop(scope);
                assert_poisoned_rejects(&model, &mut state, addresses);
            }
        }
    }
    let mut healthy = model.new_state(input.len()).unwrap();
    let (_, report) = run(&model, &mut healthy, &input, None, Arithmetic::ScalarNative).unwrap();
    assert_report(&report, input.len(), input.len(), true);
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn discarded_completion_poisoning_is_scoped_to_each_command_role() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("CED GPU lease");
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let embedding = embeddings(&ctx);
    let model = nonuniform_model(&weights, &ctx, &embedding);
    let input = tokens(9);
    for (role, arithmetic) in [
        ("ced_encoder_rows", Arithmetic::Production),
        ("ced_encoder_scalar", Arithmetic::ScalarNative),
        ("ced_local_seed", Arithmetic::Production),
        ("ced_decoder_rows", Arithmetic::Production),
        ("ced_decoder_scalar", Arithmetic::ScalarNative),
    ] {
        let mut state = model.new_state(input.len()).unwrap();
        let addresses = state.addresses;
        let scope = Scope::new(role, Action::Completion);
        let error = run(&model, &mut state, &input, None, arithmetic).unwrap_err();
        assert_eq!(scope.hits(), 1, "missing completion hook {role}");
        assert!(
            error
                .to_string()
                .contains("injected discarded command buffer"),
            "{role}: {error}"
        );
        drop(scope);
        assert_poisoned_rejects(&model, &mut state, addresses);
        let mut healthy = model.new_state(input.len()).unwrap();
        let (_, report) = run(&model, &mut healthy, &input, None, arithmetic).unwrap();
        assert_report(&report, input.len(), input.len(), arithmetic.scalar());
    }
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn preflight_rejections_keep_ready_and_nonzero_start_reports_full_fallback() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("CED GPU lease");
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let embedding = embeddings(&ctx);
    let model = nonuniform_model(&weights, &ctx, &embedding);
    let other = nonuniform_model(&weights, &ctx, &embedding);
    let mut state = model.new_state(17).unwrap();
    let addresses = state.addresses;
    let before = causal_bits(&state);
    let cancel = AtomicBool::new(true);
    let scope = Scope::new("ced_begin", Action::Panic);
    for (tokens, flag, target, message) in [
        (vec![0, 1], Some(&cancel), &model, "cancelled"),
        (vec![0, u32::MAX], None, &model, "token outside vocabulary"),
        (vec![0; 18], None, &model, "capacity"),
        (vec![], None, &model, "empty"),
        (vec![0, 1], None, &other, "another model"),
    ] {
        let error = run(target, &mut state, &tokens, flag, Arithmetic::ScalarNative).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
        assert!(state.is_ready());
        assert_eq!(state.addresses, addresses);
        assert_eq!(state.prefill_origin(), PrefillExecution::FullForward);
        for (a, b) in before.iter().zip(causal_bits(&state)) {
            assert_bits(a, &b, "preflight must not mutate causal bytes");
        }
    }
    assert_eq!(scope.hits(), 0);
    drop(scope);
    model.advance(&mut state, 3, None).unwrap();
    let snapshot = model.capture_snapshot(&state).unwrap();
    let mut full = model.new_state(17).unwrap();
    model.restore_snapshot(&mut full, &snapshot).unwrap();
    let scope = Scope::new("ced_begin", Action::Panic);
    let expected = model.prefill(&mut full, &[0, 4, 2, 7], None).unwrap();
    let (actual, report) = model
        .prefill_with_execution(
            &mut state,
            &[0, 4, 2, 7],
            None,
            PrefillExecution::ShrinkingCed,
        )
        .unwrap();
    assert_eq!(
        scope.hits(),
        0,
        "unsupported request must fall back before beginning CED"
    );
    assert_eq!(report.execution, PrefillExecution::FullForward);
    assert!(
        report
            .fallback
            .as_deref()
            .unwrap()
            .contains("cold final-logits/final-state requests only")
    );
    assert!(report.counters.is_none());
    assert!(report.memory.is_none());
    assert!(report.observed_incremental_metal_bytes.is_none());
    assert_bits(
        &actual,
        &expected,
        "nonzero-start actual full-forward fallback",
    );
    assert_causal(&full, &state, "nonzero-start fallback");
    assert_eq!(state.prefill_origin(), PrefillExecution::FullForward);
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn encoder_route_status_failure_cannot_commit_ced() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("CED GPU lease");
    let ctx = MetalContext::new().unwrap();
    let weights = toy::Weights::new(&ctx);
    let embedding = embeddings(&ctx);
    let mut model = nonuniform_model(&weights, &ctx, &embedding);
    model.layers[2].bias[0] = f32::NAN;
    for arithmetic in [Arithmetic::Production, Arithmetic::ScalarNative] {
        let mut state = model.new_state(9).unwrap();
        let addresses = state.addresses;
        let error = run(&model, &mut state, &tokens(9), None, arithmetic).unwrap_err();
        let expected = if arithmetic.scalar() {
            "CED scalar encoder FFN status"
        } else {
            "layer 2 row 0 route status"
        };
        assert!(error.to_string().contains(expected), "{error}");
        assert_poisoned_rejects(&model, &mut state, addresses);
    }
}
