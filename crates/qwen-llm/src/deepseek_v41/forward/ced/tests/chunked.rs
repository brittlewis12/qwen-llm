use super::*;
use std::collections::BTreeSet;
use std::ops::Range;

const CHUNK: usize = prefill::DEFAULT_CHUNK_ROWS;

fn set_window(model: &mut Forward<'_>, window: usize) {
    model.prefill_rows = CHUNK;
    model.config.window = window;
    for layer in &mut model.layers {
        layer.attention.window = window;
    }
}

fn dependency_plan(model: &Forward<'_>, n: usize, capacity: usize) -> CedPlan {
    CedPlan::shrinking(
        &model.config,
        &model.schedule,
        &model.engram_hash,
        &CedRequest::cold(n, capacity),
    )
    .unwrap()
}

fn chunks(range: Range<usize>) -> Vec<Range<usize>> {
    super::super::super::ced_plan::chunk_views(range, CHUNK)
        .unwrap()
        .map(|view| view.absolute_rows)
        .collect()
}

// Derive arithmetic from the ORIGINAL full request, not the shrinking tile size.
fn decoder_chunks(plan: &CedPlan, outputs: Range<usize>) -> Vec<(Range<usize>, bool)> {
    let scalar_tail = plan.prompt_rows % CHUNK == 1;
    let packed_end = plan.prompt_rows - usize::from(scalar_tail);
    let mut tiles = chunks(outputs.start..packed_end.max(outputs.start))
        .into_iter()
        .map(|range| (range, false))
        .collect::<Vec<_>>();
    if scalar_tail {
        tiles.push((plan.prompt_rows - 1..plan.prompt_rows, true));
    }
    tiles
}

fn probe_positions(plan: &CedPlan) -> Vec<(usize, usize)> {
    let mut probes = BTreeSet::new();
    for layer in &plan.decoder {
        let output = &layer.output_rows;
        for (tile, _) in decoder_chunks(plan, output.clone()) {
            probes.insert((layer.layer, tile.start));
            probes.insert((layer.layer, tile.end - 1));
        }
        for position in [0, 1, 2046, 2047, 2048, 2049, 4094, 4095, 4096] {
            if output.contains(&position) {
                probes.insert((layer.layer, position));
            }
        }
        probes.insert((layer.layer, output.end - 1));
    }
    probes.into_iter().collect()
}

struct Probe {
    layer: usize,
    position: usize,
    selected: MetalTensor,
    candidates: MetalTensor,
    encoded: bool,
}

struct Probes {
    selected_width: usize,
    candidate_width: usize,
    probes: Vec<Probe>,
}

thread_local! {
    static CONTROL_PROBES: RefCell<Option<Rc<RefCell<Probes>>>> = const { RefCell::new(None) };
}

struct ControlScope {
    probes: Rc<RefCell<Probes>>,
}

#[derive(Debug, PartialEq, Eq)]
struct Control {
    layer: usize,
    position: usize,
    selected: Vec<i32>,
    candidates: Vec<i32>,
}

impl ControlScope {
    fn begin(model: &Forward<'_>, plan: &CedPlan, capacity: usize) -> Result<Self> {
        op::require(
            CONTROL_PROBES.with(|slot| slot.borrow().is_none()),
            "nested chunked control capture",
        )?;
        let positions = probe_positions(plan);
        let counts = positions
            .iter()
            .map(|&(_, position)| {
                (
                    (position + 1).min(model.config.index_topk),
                    (position + 1).div_ceil(model.config.candidate_block),
                )
            })
            .collect::<Vec<_>>();
        let mut metal_bytes = 0u64;
        let mut words = 0usize;
        for &(selected, candidates) in &counts {
            for count in [selected, candidates] {
                metal_bytes = metal_bytes
                    .checked_add(
                        model
                            .ctx
                            .price_shared_buffer_upper((count * 4) as u64)
                            .map_err(|e| e.to_string())?
                            .priced_upper_bytes,
                    )
                    .ok_or("control probe price overflow")?;
                words = words
                    .checked_add(count)
                    .ok_or("control probe CPU overflow")?;
            }
        }
        let cpu_bytes = words
            .checked_mul(4)
            .and_then(|n| {
                n.checked_add(
                    positions.len()
                        * (std::mem::size_of::<Probe>()
                            + std::mem::size_of::<Control>()
                            + 4 * std::mem::size_of::<usize>()),
                )
            })
            .ok_or("control probe CPU overflow")?;
        let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
            metal_bytes,
            cpu_bytes as u64,
            0,
            model.ctx.memory_signals(),
            true,
        );
        if let Some(reason) = admission.refusal() {
            return Err(format!("chunked control probe admission: {reason}"));
        }
        let probes = positions
            .into_iter()
            .zip(counts)
            .map(|((layer, position), (selected, candidates))| {
                Ok(Probe {
                    layer,
                    position,
                    selected: prefill::allocate(model.ctx, selected, 1, GgmlType::I32)?,
                    candidates: prefill::allocate(model.ctx, candidates, 1, GgmlType::I32)?,
                    encoded: false,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let probes = Rc::new(RefCell::new(Probes {
            selected_width: model.config.index_topk,
            candidate_width: capacity.div_ceil(model.config.candidate_block),
            probes,
        }));
        CONTROL_PROBES.with(|slot| slot.replace(Some(Rc::clone(&probes))));
        Ok(Self { probes })
    }

    fn finish(self) -> Vec<Control> {
        self.probes
            .borrow()
            .probes
            .iter()
            .map(|probe| {
                assert!(
                    probe.encoded,
                    "missing encode_control_probes runtime hook: layer {} absolute row {}",
                    probe.layer, probe.position
                );
                Control {
                    layer: probe.layer,
                    position: probe.position,
                    selected: read_i32(&probe.selected),
                    candidates: read_i32(&probe.candidates),
                }
            })
            .collect()
    }
}

impl Drop for ControlScope {
    fn drop(&mut self) {
        CONTROL_PROBES.with(|slot| slot.replace(None));
    }
}

fn read_i32(tensor: &MetalTensor) -> Vec<i32> {
    assert_eq!(tensor.dtype, GgmlType::I32);
    assert_eq!(tensor.offset % 4, 0);
    assert!(tensor.offset + tensor.n_elements() * 4 <= tensor.buffer.length() as u64);
    unsafe {
        std::slice::from_raw_parts(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<i32>()
                .add((tensor.offset / 4) as usize),
            tensor.n_elements() as usize,
        )
        .to_vec()
    }
}

// Parent calls this for FULL and CED complete outputs, after selection and before
// scratch/control rebinding. Scalar calls use rows=1. Copies finish in the same
// checked command; no active scope means no allocation, dispatch or readback.
pub(in super::super::super) fn encode_control_probes(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    layer: usize,
    start: usize,
    rows: usize,
    selected: &MetalTensor,
    candidates: &MetalTensor,
) -> Result<()> {
    let active = CONTROL_PROBES.with(|slot| slot.borrow().clone());
    let Some(active) = active else { return Ok(()) };
    let mut active = active.borrow_mut();
    let end = start
        .checked_add(rows)
        .ok_or("control probe range overflow")?;
    if !active
        .probes
        .iter()
        .any(|p| p.layer == layer && (start..end).contains(&p.position))
    {
        return Ok(());
    }
    let sw = active.selected_width;
    let cw = active.candidate_width;
    op::require(
        rows > 0
            && rows <= CHUNK
            && selected.dtype == GgmlType::I32
            && candidates.dtype == GgmlType::I32
            && selected.n_elements() == (rows * sw) as u64
            && candidates.n_elements() == (rows * cw) as u64,
        "chunked control probe source extents",
    )?;
    for probe in &mut active.probes {
        if probe.layer != layer || !(start..end).contains(&probe.position) {
            continue;
        }
        op::require(!probe.encoded, "duplicate chunked control probe")?;
        let row = probe.position - start;
        for (source, width, destination) in [
            (selected, sw, &probe.selected),
            (candidates, cw, &probe.candidates),
        ] {
            let source = source.view_subrange((row * width) as u64, vec![destination.n_elements()]);
            metal::encode_copy_offset_i32(
                ctx,
                enc,
                &source,
                0,
                destination,
                destination.n_elements() as usize,
            )
            .map_err(|e| e.to_string())?;
        }
        probe.encoded = true;
    }
    Ok(())
}

struct Census;

impl Census {
    fn begin() -> Self {
        assert!(!metal::dispatch_census_is_active());
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

type Command = (&'static str, [usize; 4]);

fn full_commands(model: &Forward<'_>, n: usize) -> Vec<Command> {
    chunks(0..n)
        .into_iter()
        .map(|range| {
            if range.len() == 1 {
                ("gpu_step", [range.start, 0, model.layers.len(), 0])
            } else {
                (
                    "prefill_chunk_rows",
                    [range.start, model.layers.len(), range.len(), 0],
                )
            }
        })
        .collect()
}

fn ced_commands(plan: &CedPlan) -> Vec<Command> {
    let mut commands = chunks(0..plan.prompt_rows)
        .into_iter()
        .map(|range| {
            (
                if range.len() == 1 {
                    "ced_encoder_scalar"
                } else {
                    "ced_encoder_rows"
                },
                [range.start, range.len(), plan.publication_layer, 0],
            )
        })
        .collect::<Vec<_>>();
    for layer in &plan.decoder {
        let mut fuse = !unfused()
            && layer.fuse_seed_with_packed_output(plan.prompt_rows, plan.full_chunk_rows);
        if !fuse {
            for range in chunks(layer.seed_rows.clone()) {
                commands.push(("ced_local_seed", [layer.layer, range.start, range.len(), 0]));
            }
        }
        for (range, scalar) in decoder_chunks(plan, layer.output_rows.clone()) {
            commands.push((
                if scalar {
                    "ced_decoder_scalar"
                } else if fuse {
                    "ced_seed_decoder_rows"
                } else {
                    "ced_decoder_rows"
                },
                [
                    layer.layer,
                    range.start,
                    range.len(),
                    if fuse { layer.seed_rows.len() } else { 0 },
                ],
            ));
            fuse = false;
        }
    }
    commands
}

fn full_blocks(model: &Forward<'_>, n: usize) -> Vec<(usize, bool)> {
    chunks(0..n)
        .into_iter()
        .flat_map(|range| std::iter::repeat_n((range.len(), range.len() == 1), model.layers.len()))
        .collect()
}

fn ced_blocks(plan: &CedPlan) -> Vec<(usize, bool)> {
    let mut blocks = chunks(0..plan.prompt_rows)
        .into_iter()
        .flat_map(|range| {
            std::iter::repeat_n((range.len(), range.len() == 1), plan.encoder_layers.len())
        })
        .collect::<Vec<_>>();
    for layer in &plan.decoder {
        blocks.extend(
            decoder_chunks(plan, layer.output_rows.clone())
                .into_iter()
                .map(|(range, scalar)| (range.len(), scalar)),
        );
    }
    blocks
}

fn assert_commands(report: &profile::Report, expected: &[Command], label: &str) {
    let mut actual = Vec::new();
    for op in &report.operations {
        for _ in 0..op.completed_submits {
            actual.push((op.operation, op.dimensions));
        }
    }
    let mut expected = expected.to_vec();
    actual.sort_unstable();
    expected.sort_unstable();
    assert_eq!(
        actual, expected,
        "{label}: completed command roles and absolute ranges"
    );
}

fn assert_rows_census(
    rows: &[metal::DispatchCensusRow],
    c: &Config,
    blocks: &[(usize, bool)],
    label: &str,
) {
    const ROUTE: &str = "kernel_deepseek_v41_route";
    const BUCKET: &str = "kernel_moe_route_bucket_slots_f32";
    const GATE_UP: &str = "kernel_moe_swiglu_clamped_mxfp4_f32x_grouped_slots";
    const DOWN: &str = "kernel_moe_down_mxfp4_f32x_grouped_slots";
    const NATIVE: &str = "kernel_expert_native_mxfp4_f32";

    let packed_output_rows = blocks
        .iter()
        .filter(|&&(_, scalar)| !scalar)
        .map(|&(n, _)| n)
        .sum::<usize>();
    let scalar_blocks = blocks.iter().filter(|&&(_, scalar)| scalar).count();
    let mut expected = Vec::new();
    for &(n, scalar) in blocks {
        assert!((1..=CHUNK).contains(&n), "{label}: complete block width");
        expected.push((ROUTE, [n as u64, 1, 1], [256, 1, 1]));
        if scalar {
            assert_eq!(n, 1, "{label}: original scalar block width");
            for output in [c.intermediate, c.intermediate, c.hidden] {
                expected.push((
                    NATIVE,
                    [output.div_ceil(4) as u64, c.topk as u64, 1],
                    [128, 1, 1],
                ));
            }
        } else {
            expected.push((BUCKET, [c.experts.div_ceil(256) as u64, 1, 1], [256, 1, 1]));
            // encode_routed_rows passes n tokens, not n * topk slots. The
            // grouped grids tile tokens on x and outputs on y; z is experts.
            for (kernel, tile, output) in [(GATE_UP, 16, c.intermediate), (DOWN, 32, c.hidden)] {
                expected.push((
                    kernel,
                    [
                        n.div_ceil(tile) as u64,
                        output.div_ceil(64) as u64,
                        c.experts as u64,
                    ],
                    [128, 1, 1],
                ));
            }
        }
    }
    let actual = rows
        .iter()
        .filter(|d| {
            d.kernel == ROUTE
                || d.kernel.starts_with("kernel_moe_")
                || d.kernel.starts_with("kernel_expert_")
        })
        .map(|d| {
            (
                d.kernel.as_str(),
                [d.grid_width, d.grid_height, d.grid_depth],
                [d.threads_width, d.threads_height, d.threads_depth],
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual, expected,
        "{label}: exact ordered route/bucket/expert grids per complete block, including packed singletons"
    );
    assert_eq!(
        rows.iter().filter(|d| d.kernel == NATIVE).count(),
        3 * scalar_blocks,
        "{label}: native expert GEMVs only for original genuine scalar tails"
    );
    assert_eq!(
        rows.iter()
            .filter(|d| d.kernel == ROUTE)
            .map(|d| d.grid_width as usize)
            .sum::<usize>(),
        packed_output_rows + scalar_blocks,
        "{label}: exact routed operand-row coverage"
    );
    assert_eq!(
        rows.iter()
            .filter(|d| d.kernel == GATE_UP || d.kernel == DOWN)
            .count(),
        2 * (blocks.len() - scalar_blocks),
        "{label}: two grouped F32x expert kernels per packed complete block, never per-row fallback"
    );
    let qk_rows = rows
        .iter()
        .filter(|d| d.kernel == "kernel_deepseek_v41_attention_score_rows_lane8")
        .map(|d| d.grid_depth as usize)
        .sum::<usize>();
    let softmax_rows = rows
        .iter()
        .filter(|d| d.kernel == "kernel_deepseek_v41_attention_softmax_rows")
        .map(|d| d.grid_height as usize)
        .sum::<usize>();
    assert_eq!(
        qk_rows, packed_output_rows,
        "{label}: F32 batched QK query coverage"
    );
    assert_eq!(
        softmax_rows, packed_output_rows,
        "{label}: F32 batched softmax query coverage"
    );
    let dense = metal::grouped_projection_pipeline_name(GgmlType::F32).unwrap();
    if packed_output_rows > 0 {
        assert!(
            rows.iter().any(|d| d.kernel == dense),
            "{label}: grouped F32 dense path missing"
        );
        assert!(
            rows.iter()
                .any(|d| d.kernel == "kernel_deepseek_v41_attention_output_rows"),
            "{label}: toy geometry must use the F32 PV path"
        );
    }
    for row in rows {
        assert!(
            !row.encoder_concurrent,
            "{label}: concurrent command encoder"
        );
        assert!(
            !row.kernel.starts_with("kernel_mat_mat_")
                && !row.kernel.contains("_half")
                && !row.kernel.contains("_f16")
                && !row.kernel.contains("_mma")
                && !row.kernel.contains("_tensor")
                && (!row.kernel.starts_with("kernel_latent_grouped_") || row.kernel == dense),
            "{label}: unexpected Half/tensor/MMA or nonbaseline dense dispatch {}",
            row.kernel
        );
    }
}

fn with_controlled_rows<T>(f: impl FnOnce() -> T) -> T {
    // Toy MXFP4 gate/up does not support native rows. Keep the F32x grouped
    // expert path for every packed block, including shrink-created singletons.
    prefill::with_batched_f32(f)
}

fn assert_chunked_report(
    model: &Forward<'_>,
    report: &PrefillReport,
    plan: &CedPlan,
    commands: &[Command],
) {
    assert_eq!(report.execution, PrefillExecution::ShrinkingCed);
    assert!(
        report.fallback.is_none(),
        "unexpected fallback: {:?}",
        report.fallback
    );
    let counters = report.counters.as_ref().expect("actual CED counters");
    counters.validate(plan).unwrap();
    let mut expected = CedCounters::default();
    for layer in plan.encoder_layers.clone() {
        expected
            .encoder(
                layer,
                plan.prompt_rows,
                model.schedule.produces_index(layer),
            )
            .unwrap();
    }
    expected
        .publication(plan.publication_layer, plan.prompt_rows)
        .unwrap();
    for layer in &plan.decoder {
        expected.seed(layer.layer, layer.seed_rows.len()).unwrap();
        expected
            .decoder(
                layer.layer,
                layer.output_rows.len(),
                model.schedule.produces_index(layer.layer),
            )
            .unwrap();
    }
    expected.encoded_commands = commands.len();
    expected.completed_commands = commands.len();
    let mut actual = counters.clone();
    actual.layers.sort_by_key(|l| l.layer);
    expected.layers.sort_by_key(|l| l.layer);
    assert_eq!(
        actual, expected,
        "chunked CED actual work must match the whole-request dependency plan"
    );
    let memory = report.memory.as_ref().expect("normal CED memory admission");
    assert_eq!(
        memory.retained_residual_bytes,
        plan.memory.boundary_residual_bytes
    );
    assert_eq!(memory.retained_mix_bytes, plan.memory.boundary_mix_bytes);
    assert_eq!(memory.retained_candidate_bytes, plan.memory.candidate_bytes);
    assert_eq!(memory.retained_selected_bytes, plan.memory.selected_bytes);
    assert!(memory.incremental_cpu_bytes > 0 && memory.incremental_metal_bytes > 0);
    assert!(report.observed_incremental_metal_bytes.unwrap() <= memory.incremental_metal_bytes);
}

fn assert_state(model: &Forward<'_>, full: &State, ced: &State, label: &str) {
    assert!(
        full.is_ready() && ced.is_ready(),
        "{label}: non-ready state"
    );
    assert_eq!(full.position(), ced.position(), "{label}: position");
    assert_eq!(
        full.addresses, ced.addresses,
        "{label}: Engram addresses/history"
    );
    let a = causal_bits(full);
    let b = causal_bits(ced);
    assert_eq!(a.len(), 1 + 4 * model.schedule.kv_source_layers.len());
    assert_eq!(a.len(), b.len(), "{label}: owner inventory");
    let stride = model.config.window * model.config.head_dim;
    assert_eq!(a[0].len(), model.layers.len() * stride);
    assert_eq!(a[0].len(), b[0].len());
    for layer in 0..model.layers.len() {
        assert_bits(
            &a[0][layer * stride..(layer + 1) * stride],
            &b[0][layer * stride..(layer + 1) * stride],
            &format!("{label}: physical local ring layer {layer}"),
        );
    }
    for (slot, &owner) in model.schedule.kv_source_layers.iter().enumerate() {
        for (part, name) in [
            "completed KV",
            "index keys",
            "pending value",
            "pending gate",
        ]
        .into_iter()
        .enumerate()
        {
            let region = 1 + slot * 4 + part;
            assert_bits(
                &a[region],
                &b[region],
                &format!("{label}: owner {owner} {name}"),
            );
        }
    }
}

fn assert_controls(full: &[Control], ced: &[Control], label: &str) {
    assert!(!full.is_empty(), "{label}: no control probes");
    assert_eq!(full.len(), ced.len(), "{label}: probe coverage");
    for (a, b) in full.iter().zip(ced) {
        assert_eq!(
            (a.layer, a.position),
            (b.layer, b.position),
            "{label}: absolute control identity"
        );
        for (role, a_values, b_values) in [
            ("selected IDs", &a.selected, &b.selected),
            ("candidate mask", &a.candidates, &b.candidates),
        ] {
            assert_eq!(a_values.len(), b_values.len(), "{label}: {role} extent");
            for (index, (&expected, &actual)) in a_values.iter().zip(b_values).enumerate() {
                assert_eq!(
                    actual, expected,
                    "{label}: {role} layer {} absolute row {} element {index}",
                    a.layer, a.position
                );
            }
        }
        assert!(
            a.selected
                .iter()
                .all(|&id| id >= 0 && id as usize <= a.position)
        );
        assert!(a.selected.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(a.candidates.iter().all(|&value| value == 0 || value == 1));
    }
}

fn run_case(model: &Forward<'_>, n: usize, continuation: usize) {
    let label = format!(
        "complete F32Rows layers={} W={} N={n}",
        model.layers.len(),
        model.config.window
    );
    let plan = dependency_plan(model, n, n);
    let full_commands = full_commands(model, n);
    let ced_commands = ced_commands(&plan);
    let input = tokens(n);
    let mut full = model.new_state(n).unwrap();
    let mut ced = model.new_state(n).unwrap();
    let capture = ControlScope::begin(model, &plan, n).unwrap();
    let profile = profile::Scope::begin().unwrap();
    let census = Census::begin();
    let result = model.prefill(&mut full, &input, None);
    let full_census = census.finish();
    let full_profile = profile.finish();
    let expected = result.unwrap();
    let full_controls = capture.finish();
    let capture = ControlScope::begin(model, &plan, n).unwrap();
    let profile = profile::Scope::begin().unwrap();
    let census = Census::begin();
    let result =
        model.prefill_with_execution(&mut ced, &input, None, PrefillExecution::ShrinkingCed);
    let ced_census = census.finish();
    let ced_profile = profile.finish();
    let (actual, report) = result.unwrap();
    let ced_controls = capture.finish();
    assert_commands(&full_profile, &full_commands, &format!("{label} full"));
    assert_commands(&ced_profile, &ced_commands, &format!("{label} CED"));
    assert_chunked_report(model, &report, &plan, &ced_commands);
    let full_blocks = full_blocks(model, n);
    let ced_blocks = ced_blocks(&plan);
    assert_eq!(
        full_blocks.iter().map(|&(n, _)| n).sum::<usize>(),
        n * model.layers.len(),
        "{label}: full complete-block row coverage"
    );
    assert_eq!(
        ced_blocks.iter().map(|&(n, _)| n).sum::<usize>(),
        plan.work.total_output_rows,
        "{label}: CED complete-block row coverage"
    );
    assert_rows_census(
        &full_census,
        &model.config,
        &full_blocks,
        &format!("{label} full"),
    );
    assert_rows_census(
        &ced_census,
        &model.config,
        &ced_blocks,
        &format!("{label} CED"),
    );
    assert_eq!(
        actual.len(),
        model.config.vocab,
        "{label}: complete vocabulary logits"
    );
    assert_controls(&full_controls, &ced_controls, &label);
    assert_state(model, &full, &ced, &label);
    assert_bits(&actual, &expected, &format!("{label}: final full logits"));
    assert_eq!(full.prefill_origin(), PrefillExecution::FullForward);
    assert_eq!(ced.prefill_origin(), PrefillExecution::ShrinkingCed);
    if continuation == 0 {
        return;
    }
    let snapshot = model.capture_snapshot(&ced).unwrap();
    assert_eq!(snapshot.position(), n);
    assert_eq!(snapshot.prefill_origin(), PrefillExecution::ShrinkingCed);
    let mut restored = model.new_state(n + continuation).unwrap();
    let mut grown = model.new_state(n).unwrap();
    for state in [&mut restored, &mut grown] {
        for token in [7, 2, 0, 6] {
            model.advance(state, token, None).unwrap();
        }
        model.restore_snapshot(state, &snapshot).unwrap();
        assert_state(
            model,
            &ced,
            state,
            &format!("{label}: snapshot restored into dirty state"),
        );
        assert_eq!(state.prefill_origin(), PrefillExecution::ShrinkingCed);
    }
    for state in [&mut full, &mut ced, &mut grown] {
        model.ensure_capacity(state, n + continuation).unwrap();
        assert!(state.capacity() >= n + continuation);
    }
    for state in [&ced, &restored, &grown] {
        assert_state(
            model,
            &full,
            state,
            &format!("{label}: restored/grown initial state"),
        );
        assert_eq!(state.prefill_origin(), PrefillExecution::ShrinkingCed);
    }
    for (step, token) in tokens(continuation).into_iter().enumerate() {
        let expected = model.step(&mut full, token, None).unwrap();
        for (name, state) in [
            ("CED", &mut ced),
            ("restored", &mut restored),
            ("grown", &mut grown),
        ] {
            let actual = model.step(state, token, None).unwrap();
            let label = format!("{label}: {name} teacher-forced scalar step {step}");
            assert_bits(&actual, &expected, &label);
            assert_state(model, &full, state, &label);
            assert_eq!(state.prefill_origin(), PrefillExecution::ShrinkingCed);
        }
    }
    assert_eq!(
        model.capture_snapshot(&grown).unwrap().prefill_origin(),
        PrefillExecution::ShrinkingCed
    );
}

#[test]
fn multichunk_graph_geometry_really_crosses_encoder_and_decoder_boundaries() {
    assert_eq!(CHUNK, 2048);
    let mut config = toy::config();
    config.window = 1025;
    config.index_topk = 2;
    let make = |n| {
        CedPlan::shrinking(
            &config,
            &schedule(),
            &hash_config(),
            &CedRequest::cold(n, n),
        )
        .unwrap()
    };
    let p = make(4096);
    assert_eq!(p.boundary_rows, 1023..4096);
    assert_eq!(p.decoder[0].seed_rows, 1023..2047);
    assert_eq!(p.decoder[0].output_rows, 2047..4096);
    assert_eq!(chunks(0..4096), [0..2048, 2048..4096]);
    assert_eq!(
        decoder_chunks(&p, p.decoder[0].output_rows.clone()),
        [(2047..4095, false), (4095..4096, false)]
    );
    assert_eq!(p.work.decoder_output_rows, 3075);
    assert_eq!(p.work.decoder_seed_rows, 3072);
    let p = make(4097);
    assert_eq!(chunks(0..4097), [0..2048, 2048..4096, 4096..4097]);
    assert_eq!(
        decoder_chunks(&p, p.decoder[0].output_rows.clone()),
        [(2048..4096, false), (4096..4097, true)]
    );
    let p = CedPlan::shrinking(
        &Config {
            window: 128,
            ..toy::config()
        },
        &super::super::super::super::gguf::Schedule::default(),
        &engram::HashConfig::default(),
        &CedRequest::cold(2542, 2542),
    )
    .unwrap();
    assert_eq!(p.boundary_rows, 1..2542);
    assert_eq!(p.decoder[0].seed_rows, 1..128);
    assert_eq!(
        chunks(p.decoder[0].output_rows.clone()),
        [128..2176, 2176..2542]
    );
    assert_eq!(p.work.encoder_output_rows, 20 * 2542);
    assert_eq!(p.work.decoder_output_rows, 24_150);
    assert_eq!(p.work.decoder_seed_rows, 2540);
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn batched_f32_multichunk_4096_bitwise_restore_growth_and_256_scalar_steps() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("chunked CED GPU lease");
    with_controlled_rows(|| {
        let ctx = MetalContext::new().unwrap();
        let weights = toy::Weights::new(&ctx);
        let embedding = embeddings(&ctx);
        let mut model = nonuniform_model(&weights, &ctx, &embedding);
        set_window(&mut model, 1025);
        let p = dependency_plan(&model, 4096, 4096);
        assert_eq!(p.decoder[0].output_rows.len(), 2049);
        assert_eq!(p.decoder[0].seed_rows.len(), 1024);
        run_case(&model, 4096, 256);
    });
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn batched_f32_chunk_edges_and_genuine_scalar_tails_bitwise() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("chunked CED GPU lease");
    with_controlled_rows(|| {
        let ctx = MetalContext::new().unwrap();
        let weights = toy::Weights::new(&ctx);
        let embedding = embeddings(&ctx);
        let mut model = nonuniform_model(&weights, &ctx, &embedding);
        set_window(&mut model, 1025);
        for n in [1, 2047, 2048, 2049, 2540, 2541, 2542, 4097] {
            run_case(&model, n, 0);
        }
    });
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn batched_f32_default_d20_w128_2542_bitwise_all_layers() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("chunked CED GPU lease");
    with_controlled_rows(|| {
        let ctx = MetalContext::new().unwrap();
        let weights = toy::Weights::new(&ctx);
        let embedding = embeddings(&ctx);
        let mut model = weights.model(&ctx);
        model.config.index_topk = 2;
        vary_model(&mut model, &embedding);
        set_window(&mut model, 128);
        assert_eq!(model.layers.len(), 40);
        let p = dependency_plan(&model, 2542, 2542);
        assert_eq!(p.encoder_layers, 0..20);
        assert_eq!(p.boundary_rows.len(), 2541);
        assert_eq!(p.work.encoder_output_rows, 20 * 2542);
        assert_eq!(p.work.decoder_output_rows, 24_150);
        assert_eq!(p.work.decoder_seed_rows, 2540);
        run_case(&model, 2542, 0);
    });
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn batched_f32_window_one_packed_singletons_bitwise() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("chunked CED GPU lease");
    with_controlled_rows(|| {
        let ctx = MetalContext::new().unwrap();
        let weights = toy::Weights::new(&ctx);
        let embedding = embeddings(&ctx);
        let mut model = nonuniform_model(&weights, &ctx, &embedding);
        set_window(&mut model, 1);
        let p = dependency_plan(&model, 17, 17);
        assert_eq!(p.work.decoder_seed_rows, 0);
        assert_eq!(p.work.decoder_output_rows, 3);
        run_case(&model, 17, 16);
    });
}
