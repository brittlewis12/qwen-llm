use super::super::query_batch::{bound, config, values};
use super::*;
use crate::deepseek_v41::forward::attention_gpu::query_batch::{Scratch, selection_shapes};

const REAL_PROTOCOL: &str = "v41-mask-real-pipeline-v1";
const START: usize = 16383;
const ROWS: usize = 2;
const CAPACITY: usize = START + ROWS;
const BLOCKS: usize = 2049;
const KEPT: usize = 2048;
const REAL_LIMIT: u64 = 128 << 20;

fn geometry() -> (Config, Schedule) {
    (
        Config {
            index_heads: 1,
            index_topk: 512,
            candidate_block: 8,
            candidate_topk: KEPT,
            rope_dim: 0,
            ..config()
        },
        Schedule {
            layers: 4,
            kv_source_layers: vec![0, 2],
            index_source_layers: vec![0, 2, 3],
            compression_ratios: vec![1; 4],
            candidate_source_layer: Some(0),
        },
    )
}

fn producer_oracle(input: &[u32]) -> Option<Vec<u32>> {
    let mut set = input.to_vec();
    set.sort_unstable();
    if set.len() != KEPT
        || set.last() != Some(&((BLOCKS - 1) as u32))
        || set.windows(2).any(|w| w[0] >= w[1])
    {
        return None;
    }
    Some(reference(BLOCKS, &set))
}

fn expected_mask() -> Vec<u32> {
    std::iter::once(0)
        .chain(std::iter::repeat_n(1, KEPT))
        .collect()
}

fn input_data(c: &Config) -> Vec<f32> {
    let mut data = vec![0.0; ROWS * c.hidden];
    for row in data.chunks_exact_mut(c.hidden) {
        row[0] = (c.hidden as f32).sqrt();
    }
    data
}

fn key_data(c: &Config) -> Vec<f32> {
    let mut keys = vec![0.0; CAPACITY * c.index_dim];
    for id in 0..START {
        keys[id * c.index_dim] = if id < c.candidate_block { 0.0 } else { 1.0 };
        keys[id * c.index_dim + 1] = (CAPACITY - id) as f32;
    }
    keys
}

fn write_f32(t: &MetalTensor, data: &[f32]) {
    assert_eq!(t.dtype, GgmlType::F32);
    assert_eq!(t.n_elements() as usize, data.len());
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<f32>(),
            data.len(),
        );
    }
}

fn weight_tensors<'a>(weights: &'a [Weights<'_>]) -> Vec<&'a MetalTensor> {
    let mut tensors = Vec::new();
    for w in weights {
        tensors.extend([
            &w.query_a.tensor,
            &w.query_b.tensor,
            &w.kv.tensor,
            &w.output_b.tensor,
        ]);
        tensors.extend(w.output_a.iter().map(|w| &w.tensor));
        if let Some(w) = &w.compressor {
            tensors.push(&w.kv.tensor);
        }
        if let Some(w) = &w.index {
            tensors.extend([&w.query.tensor, &w.weights.tensor]);
            tensors.extend(w.key.as_ref().map(|w| &w.tensor));
        }
    }
    tensors
}

fn weight_digest(weights: &[Weights<'_>]) -> String {
    let mut hash = Sha256::new();
    for t in weight_tensors(weights) {
        for v in f32_slice(t).unwrap() {
            hash.update(v.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

fn seed_real(w: &mut Workspace, c: &Config) {
    write_f32(&w.raw, &values(w.raw.n_elements() as usize, 17));
    let keys = key_data(c);
    for (i, owner) in w.owners.iter().enumerate() {
        for (t, salt) in [
            (&owner.compressed, 19 + i * 2),
            (&owner.pending_value, 43 + i * 2),
            (&owner.pending_gate, 53 + i * 2),
        ] {
            write_f32(t, &values(t.n_elements() as usize, salt));
        }
        write_f32(&owner.keys, &keys);
    }
    w.next_position = START;
}

fn design_weights(weights: &mut [Weights<'_>]) {
    let channel = |w: &Linear<'_>, output: Option<usize>| {
        let mut data = vec![0.0; w.input * w.output];
        if let Some(output) = output {
            data[output * w.input] = 1.0;
        }
        write_f32(&w.tensor, &data);
    };
    for (id, w) in weights.iter().enumerate() {
        channel(&w.query_a, Some(0));
        if let Some(index) = &w.index {
            channel(&index.query, Some(usize::from(id != 0)));
            channel(&index.weights, Some(0));
        }
        if let Some(compressor) = &w.compressor {
            channel(&compressor.kv, None);
        }
    }
}

// Original row controls remain in the case pool when replaced by guarded views.
// Price them too, plus the optional half scratch even on a non-tensor device.
fn inventory(c: &Config, schedule: &Schedule) -> Vec<usize> {
    let mut sizes = shapes(c, CAPACITY, schedule).unwrap();
    sizes.extend(Rows::widths(c, CAPACITY).into_iter().map(|w| w * ROWS));
    sizes.extend(Scratch::shapes(c, ROWS).unwrap());
    sizes.extend(selection_shapes(c, CAPACITY, ROWS, 512).unwrap());
    sizes.push(
        prefill::attention_half_scratch_elements(c, ROWS)
            .unwrap()
            .div_ceil(2),
    );
    for id in 0..schedule.layers {
        sizes.extend([
            c.hidden * c.query_rank,
            c.query_rank * c.heads * c.head_dim,
            c.hidden * c.head_dim,
            c.heads * c.head_dim * c.output_rank,
            c.groups * c.output_rank * c.hidden,
        ]);
        if schedule.owns_kv(id) {
            sizes.push(c.hidden * c.head_dim);
        }
        if schedule.produces_index(id) {
            sizes.extend([
                c.query_rank * c.index_heads * c.index_dim,
                c.hidden * c.index_heads,
            ]);
            if schedule.owns_kv(id) {
                sizes.push(c.head_dim * c.index_dim);
            }
        }
    }
    for logical in [
        ROWS * c.hidden,
        ROWS * c.hidden,
        ROWS * c.index_topk,
        ROWS * BLOCKS,
        KEPT,
    ] {
        sizes.push(32 + logical + TAIL);
    }
    sizes
}

#[derive(serde::Serialize)]
struct RealCpuInventory {
    seed_keys_values_and_upload_staging: u64,
    full_readbacks_and_integrity_oracles: u64,
    retained_ids_masks_and_both_arm_outputs: u64,
    metadata_hashes_reports_pools_and_runtime_reserve: u64,
}

impl RealCpuInventory {
    fn new(sizes: &[usize]) -> Self {
        let largest = *sizes.iter().max().unwrap() as u64;
        Self {
            seed_keys_values_and_upload_staging: largest * 4 * 4,
            full_readbacks_and_integrity_oracles: largest * 8 * 4,
            retained_ids_masks_and_both_arm_outputs: 1 << 20,
            metadata_hashes_reports_pools_and_runtime_reserve: 8 << 20,
        }
    }

    fn total(&self) -> u64 {
        self.seed_keys_values_and_upload_staging
            + self.full_readbacks_and_integrity_oracles
            + self.retained_ids_masks_and_both_arm_outputs
            + self.metadata_hashes_reports_pools_and_runtime_reserve
    }
}

fn all_tensors(w: &Workspace, r: &Rows) -> Vec<MetalTensor> {
    let mut tensors = vec![
        &w.raw,
        &w.query_rank,
        &w.query,
        &w.latent,
        &w.gate,
        &w.key,
        &w.index_query,
        &w.index_weights,
        &w.index_scores,
        &w.block_scores,
        &w.candidates,
        &w.sort_scores,
        &w.sort_ids,
        &w.selected,
        &w.attention_scores,
        &w.attended,
        &w.low,
        &w.output,
        &w.status,
        &w.group_ids,
        &w.ready,
        &r.rank,
        &r.query,
        &r.latent,
        &r.low,
        &r.selected,
        &r.candidates,
        &r.grouped_input,
        &r.grouped_low,
        &r.compressor_value,
        &r.compressor_gate,
        &r.index_query,
        &r.index_weights,
        &r.select_scores,
        &r.tile.history,
        &r.tile.scores,
        &r.tile.attended,
    ];
    tensors.extend(r.select_pairs.as_ref());
    if let Some(half) = &r.half {
        tensors.push(half.backing_for_test());
    }
    for matrix in [w.value_matrix.as_ref(), r.tile.matrix.as_ref()]
        .into_iter()
        .flatten()
    {
        tensors.extend([&matrix.values, &matrix.probabilities]);
    }
    for owner in &w.owners {
        tensors.extend([
            &owner.compressed,
            &owner.keys,
            &owner.pending_value,
            &owner.pending_gate,
        ]);
    }
    for constants in &w.constants {
        tensors.extend([
            &constants.query_norm,
            &constants.kv_norm,
            &constants.sink,
            &constants.frequencies,
        ]);
        tensors.extend(constants.compressor_norm.as_ref());
        tensors.extend(constants.key_norm.as_ref());
    }
    tensors.into_iter().cloned().collect()
}

struct RealFixture<'a> {
    weights: Vec<Weights<'a>>,
    workspace: Workspace,
    rows: Rows,
    input: Guarded,
    output: Guarded,
    selected: Guarded,
    mask: Guarded,
    retained: Guarded,
    inventory: Vec<MetalTensor>,
    weight_hash: String,
}

impl<'a> RealFixture<'a> {
    fn new(ctx: &'a MetalContext, arm: Arm, capture: bool) -> Self {
        let (c, schedule) = geometry();
        let mut weights = bound(ctx, &c, &schedule);
        design_weights(&mut weights);
        let weight_hash = weight_digest(&weights);
        let mut workspace = Workspace::new_with_schedule(
            ctx,
            &c,
            CAPACITY,
            &weights.iter().collect::<Vec<_>>(),
            &schedule,
        )
        .unwrap();
        let mut rows = Rows::new(ctx, &c, CAPACITY, ROWS).unwrap();
        let mut inventory = all_tensors(&workspace, &rows);
        inventory.extend(weight_tensors(&weights).into_iter().cloned());
        let mut guarded = |logical, dtype| {
            let mut guard = Guarded::new(ctx, 32, logical, GUARD);
            guard.view.dtype = dtype;
            inventory.push(guard.backing.clone());
            guard
        };
        let input = guarded(ROWS * c.hidden, GgmlType::F32);
        let output = guarded(ROWS * c.hidden, GgmlType::F32);
        let selected = guarded(ROWS * c.index_topk, GgmlType::I32);
        let mask = guarded(ROWS * BLOCKS, GgmlType::I32);
        let retained = guarded(KEPT, GgmlType::I32);
        write_f32(&input.view, &input_data(&c));
        mask.write(&vec![POISON; ROWS * BLOCKS]);
        selected.write(&vec![POISON; ROWS * c.index_topk]);
        seed_real(&mut workspace, &c);
        rows.selected = selected.view.clone();
        rows.candidates = mask.view.clone();
        workspace.mask_publication_experiment = Some(MaskPublicationExperiment {
            clear_scatter: matches!(arm, Arm::ClearScatter),
            retained_ids: capture.then(|| retained.view.clone()),
        });
        workspace.begin_batch(START, ROWS).unwrap();
        Self {
            weights,
            workspace,
            rows,
            input,
            output,
            selected,
            mask,
            retained,
            inventory,
            weight_hash,
        }
    }

    fn encode_layer(&mut self, ctx: &MetalContext, enc: &KernelEncoder, id: usize) {
        self.workspace
            .encode_rows(
                ctx,
                enc,
                id,
                START,
                &self.weights[id],
                &self.input.view,
                &self.rows,
            )
            .unwrap();
        self.workspace
            .finish_rows(ctx, enc, &self.weights[id], &self.rows, &self.output.view)
            .unwrap();
    }

    fn check(&self, id: usize, capture: bool) -> Vec<f32> {
        let c = &self.workspace.config;
        let mask = self.mask.read();
        assert_eq!(&mask[..KEPT], vec![1; KEPT]);
        assert_eq!(
            mask[KEPT], POISON,
            "invisible row-0 word must not be published"
        );
        assert_eq!(&mask[BLOCKS..], expected_mask());
        if capture {
            let retained = self.retained.read();
            assert_eq!(
                retained[0],
                (BLOCKS - 1) as u32,
                "forced +Inf newest ranks first"
            );
            assert_eq!(producer_oracle(&retained).unwrap(), mask[BLOCKS..]);
        }
        if id >= 2 {
            assert_eq!(
                ids(
                    &prefill::row(&self.rows.selected, c.index_topk, 0),
                    c.index_topk
                ),
                (0..512).collect::<Vec<_>>()
            );
            assert_eq!(
                ids(
                    &prefill::row(&self.rows.selected, c.index_topk, 1),
                    c.index_topk
                ),
                (8..520).collect::<Vec<_>>()
            );
        }
        assert_eq!(ids(self.workspace.status(), 1), [0]);
        assert_eq!(f32_slice(&self.input.view).unwrap(), input_data(c));
        assert_eq!(weight_digest(&self.weights), self.weight_hash);
        for guard in [
            &self.input,
            &self.output,
            &self.selected,
            &self.mask,
            &self.retained,
        ] {
            guard.check_guards();
        }
        let expected_keys = key_data(c);
        for (i, owner) in self.workspace.owners.iter().enumerate() {
            assert_eq!(
                &f32_slice(&owner.keys).unwrap()[..START * c.index_dim],
                &expected_keys[..START * c.index_dim]
            );
            let expected = values(owner.compressed.n_elements() as usize, 19 + i * 2);
            assert_eq!(
                &f32_slice(&owner.compressed).unwrap()[..START * c.head_dim],
                &expected[..START * c.head_dim]
            );
        }
        let output = f32_slice(&self.output.view).unwrap().to_vec();
        assert!(output.iter().all(|v| v.is_finite()));
        output
    }

    fn wrong_control(&mut self, ctx: &MetalContext) {
        execute(ctx, |enc| {
            self.workspace
                .sort(ctx, enc, &self.workspace.index_scores, CAPACITY, false)
                .unwrap()
        });
        let unmasked = ids(&self.workspace.sort_ids, 512);
        assert_eq!(unmasked, (0..512).collect::<Vec<_>>());
        let mask = self.mask.read();
        assert!(unmasked.iter().any(|&id| mask[BLOCKS + id / 8] == 0));
        assert_eq!(ids(self.workspace.status(), 1), [0]);
    }
}

fn with_real_fixture<T>(
    ctx: &MetalContext,
    arm: Arm,
    capture: bool,
    work: impl FnOnce(&mut RealFixture<'_>) -> T,
) -> T {
    let mut result = None;
    drained_case_pool(
        ctx.begin_allocation_transaction(),
        |_| {
            let (c, schedule) = geometry();
            let sizes = inventory(&c, &schedule);
            let cpu = RealCpuInventory::new(&sizes);
            let priced: u64 = sizes
                .iter()
                .filter(|&&n| n > 0)
                .map(|&n| {
                    ctx.price_shared_buffer_upper(n as u64 * 4)
                        .unwrap()
                        .priced_upper_bytes
                })
                .sum();
            assert!(priced + cpu.total() <= REAL_LIMIT);
            let admission = evaluate_metal_memory_admission_with_cpu_bytes(
                priced,
                cpu.total(),
                0,
                ctx.memory_signals(),
                true,
            );
            assert!(
                admission.is_admitted(),
                "real pipeline refused: {admission:?}"
            );
            eprintln!(
                "{}",
                json!({"protocol": REAL_PROTOCOL, "event": "admission", "arm": arm.name(), "capture": capture, "allocation_upper_count": sizes.len(), "priced_metal_bytes": priced, "cpu_inventory": cpu, "cpu_upper_bytes": cpu.total(), "limit_bytes": REAL_LIMIT})
            );
            metal::allocation_census_begin();
            let mut f = RealFixture::new(ctx, arm, capture);
            let allocations = metal::allocation_census_take();
            let realized: u64 = f.inventory.iter().map(|t| t.buffer.length() as u64).sum();
            assert_eq!(
                allocations.len(),
                f.inventory.len(),
                "complete allocation inventory"
            );
            assert_eq!(
                allocations.iter().map(|a| a.buffer_length).sum::<u64>(),
                realized
            );
            assert!(
                realized <= priced,
                "whole live inventory exceeds admitted price"
            );
            let probes: Vec<_> = f
                .inventory
                .iter()
                .map(|t| Weak::from_retained(&t.buffer))
                .collect();
            assert!(probes.iter().all(|weak| weak.load().is_some()));
            result = Some(work(&mut f));
            (probes, realized)
        },
        |(probes, realized)| {
            assert!(
                probes.iter().all(|weak| weak.load().is_none()),
                "real fixture buffer survived case drain"
            );
            eprintln!(
                "{}",
                json!({"protocol": REAL_PROTOCOL, "event": "case_release", "arm": arm.name(), "capture": capture, "buffers": probes.len(), "realized_metal_bytes": realized, "all_released": true})
            );
        },
    );
    result.unwrap()
}

fn census_check(census: &[metal::DispatchCensusRow], arm: Arm, id: usize) {
    let count = |suffix: &str| {
        census
            .iter()
            .filter(|d| d.kernel == format!("kernel_deepseek_v41_attention_{suffix}"))
            .count()
    };
    assert!(census.iter().all(|d| !d.encoder_concurrent));
    let source = usize::from(id == 0);
    assert_eq!(count("blocks"), source);
    assert_eq!(
        count("keep"),
        source * usize::from(matches!(arm, Arm::Incumbent))
    );
    assert_eq!(
        count("keep_clear"),
        source * usize::from(matches!(arm, Arm::ClearScatter))
    );
    assert_eq!(
        count("keep_scatter"),
        source * usize::from(matches!(arm, Arm::ClearScatter))
    );
    assert_eq!(count("index_heads"), if id == 1 { 0 } else { ROWS });
    assert_eq!(
        count("index_heads_rows"),
        0,
        "pruned batch must use normal per-row fallback"
    );
    assert_eq!(
        count("sort_init"),
        if id == 0 {
            3
        } else if id == 1 {
            0
        } else {
            2
        }
    );
    assert_eq!(count("select_hier_local"), 0);
    assert!(count("score_rows_lane8") > 0, "actual attention consumer");
    assert!(count("softmax_rows") > 0);
}

fn correctness(ctx: &MetalContext) -> (Vec<Vec<f32>>, [usize; 2]) {
    let mut baseline: Option<Vec<Vec<f32>>> = None;
    let mut producer = None;
    let mut counts = [0; 2];
    for (index, arm) in [Arm::Incumbent, Arm::ClearScatter].into_iter().enumerate() {
        let output = with_real_fixture(ctx, arm, true, |f| {
            let mut outputs = Vec::new();
            for id in 0..4 {
                execute(ctx, |enc| {
                    metal::dispatch_census_begin();
                    f.encode_layer(ctx, enc, id);
                    let census = metal::dispatch_census_take();
                    census_check(&census, arm, id);
                    // Exactly one retained-ID copy is omitted from native samples.
                    counts[index] += census.len() - usize::from(id == 0);
                    let rows: Vec<_> = census.iter().map(|d| json!({"kernel": d.kernel, "grid": [d.grid_width, d.grid_height, d.grid_depth], "threads": [d.threads_width, d.threads_height, d.threads_depth], "serial": !d.encoder_concurrent})).collect();
                    eprintln!(
                        "{}",
                        json!({"protocol": REAL_PROTOCOL, "event": "dispatch_census", "arm": arm.name(), "layer": id, "rows": rows})
                    );
                });
                outputs.push(f.check(id, true));
            }
            f.workspace.validate_snapshot_position(CAPACITY).unwrap();
            let retained = f.retained.read();
            eprintln!(
                "{}",
                json!({"protocol": REAL_PROTOCOL, "event": "actual_producer_ids", "arm": arm.name(), "ids": retained, "sha256_le": valid_ids::id_digest(&retained)})
            );
            if let Some(expected) = &producer {
                assert_eq!(
                    &retained, expected,
                    "same actual score-ranked producer IDs in both arms"
                );
            } else {
                producer = Some(retained);
            }
            f.wrong_control(ctx);
            outputs
        });
        if let Some(expected) = &baseline {
            for (actual, expected) in output.iter().zip(expected) {
                close(actual, expected);
            }
        } else {
            baseline = Some(output);
        }
    }
    assert_eq!(
        counts[1],
        counts[0] + 1,
        "extra scatter dispatch is counted"
    );
    eprintln!(
        "{}",
        json!({"protocol": REAL_PROTOCOL, "event": "correctness_complete", "arms": 2, "layers_per_arm": 4, "native_span_dispatches": counts, "actual_retained_ids_per_arm": KEPT, "every_mask_word": true, "unmasked_wrong_control": true, "output_bound": "attention_gpu::tests::close: 2e-5 + 4e-6*abs(reference)"})
    );
    (baseline.unwrap(), counts)
}

fn real_context(kind: &str) -> MetalContext {
    for (name, _) in std::env::vars_os() {
        let name = name.to_string_lossy();
        assert!(
            !name.starts_with("QWEN_V41_") && name != "QWEN_METAL_LEASE_SKIP_WIRED_GATE",
            "unset {name} for production-default dispatch"
        );
    }
    let ctx = MetalContext::new().expect("explicit real pipeline requires Metal");
    assert!(
        ctx.device.name().to_string().contains("M5"),
        "parent-only M5 experiment"
    );
    assert!(!metal::dispatch_census_is_active());
    eprintln!(
        "{}",
        json!({
            "protocol": REAL_PROTOCOL, "event": "context", "run_kind": kind, "device": ctx.describe(),
            "fixture_sha256": format!("{:x}", Sha256::digest(include_bytes!("real_pipeline.rs"))),
            "host_sha256": format!("{:x}", Sha256::digest(include_bytes!("../../../attention_gpu.rs"))),
            "query_batch_sha256": format!("{:x}", Sha256::digest(include_bytes!("../../query_batch.rs"))),
            "msl_sha256": format!("{:x}", Sha256::digest(include_bytes!("../../../../../../../../kernels/deepseek_v41_attention.metal"))),
            "start": START, "rows": ROWS, "capacity": CAPACITY, "visible_lengths": [16384, 16385],
            "candidate_block": 8, "candidate_topk": KEPT, "index_topk": 512,
            "kv_owners": [0, 2], "index_sources": [0, 2, 3], "candidate_source": 0, "ratios": [1, 1, 1, 1],
            "production_dispatch_changed": false, "scope": "reduced scheduled host path; NOT model/prompt-threshold evidence",
            "reset": "fresh independently seeded workspace/weights/rows and metadata for every arm/sample; serial complete-case pools",
        })
    );
    ctx
}

#[test]
#[ignore = "parent-only self-leased M5 real producer/publication/follower; MTL_DEBUG_LAYER=1; <=128 MiB CPU+Metal"]
fn mask_real_pipeline_m5_correctness() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("production lease and wired guard");
    let ctx = real_context("correctness");
    correctness(&ctx);
}

#[test]
#[ignore = "parent-only self-leased M5 full real span ABBA/BAAB timing; native validation flags absent; <=128 MiB"]
fn mask_real_pipeline_m5_native_timing() {
    for (name, _) in std::env::vars_os() {
        assert!(
            !forbidden_native_variable(&name.to_string_lossy()),
            "native launch must omit {name:?}"
        );
    }
    let _lease = metal::acquire_metal_benchmark_lease().expect("production lease and wired guard");
    let ctx = real_context("native_timing");
    let (baseline, dispatches) = correctness(&ctx);
    eprintln!(
        "{}",
        json!({"protocol": REAL_PROTOCOL, "event": "timing_begin", "rounds": ROUNDS, "orders": ["ABBA", "BAAB"], "total_samples": 24, "warmup_samples": 0, "retries": 0, "spread_or_win_gate": false, "spans_per_sample": 1, "includes": "command creation + serial encode_rows/finish_rows all four layers + real blocks/sort/publication/follower selection/attention + commit/wait", "excludes": "allocation/seed/admission/compilation/census/probe copies/readback/oracles/release", "subspans_summed": false})
    );
    use Arm::{ClearScatter as B, Incumbent as A};
    let mut sample = 0;
    for round in 0..ROUNDS {
        for (order, arms) in [("ABBA", [A, B, B, A]), ("BAAB", [B, A, A, B])] {
            for (slot, arm) in arms.into_iter().enumerate() {
                with_real_fixture(&ctx, arm, false, |f| {
                    let wall = std::time::Instant::now();
                    let command = ctx.queue.commandBuffer().unwrap();
                    let enc = KernelEncoder::begin(&command);
                    for id in 0..4 {
                        f.encode_layer(&ctx, &enc, id);
                    }
                    enc.end();
                    metal::commit_and_wait(&command).unwrap();
                    let wall_ns = wall.elapsed().as_nanos() as u64;
                    let start = command.GPUStartTime();
                    let end = command.GPUEndTime();
                    let valid = start.is_finite() && end.is_finite() && start > 0.0 && end > start;
                    eprintln!(
                        "{}",
                        json!({"protocol": REAL_PROTOCOL, "event": "sample", "sample": sample, "round": round, "order": order, "slot": slot, "arm": arm.name(), "cpu_wall_ns": wall_ns, "gpu_start_s": start, "gpu_end_s": end, "gpu_timestamps_valid": valid, "gpu_span_ns": if valid { Some((end-start)*1e9) } else { None }, "dispatches": dispatches[usize::from(matches!(arm, B))], "weight_sha256": f.weight_hash})
                    );
                    assert!(valid, "raw invalid sample retained; no retries");
                    close(&f.check(3, false), &baseline[3]);
                    f.workspace.validate_snapshot_position(CAPACITY).unwrap();
                    eprintln!(
                        "{}",
                        json!({"protocol": REAL_PROTOCOL, "event": "sample_check", "sample": sample, "status": 0, "guards_inputs_outputs_masks_and_follower_ids": true})
                    );
                });
                sample += 1;
            }
        }
    }
    assert_eq!(sample, 24);
    eprintln!(
        "{}",
        json!({"protocol": REAL_PROTOCOL, "event": "timing_complete", "samples": sample})
    );
}

#[test]
fn mask_real_pipeline_cpu_geometry_and_normal_fallback() {
    let (c, s) = geometry();
    s.validate().unwrap();
    assert_eq!((START, ROWS, CAPACITY), (16383, 2, 16385));
    assert_eq!(
        (c.candidate_block, c.candidate_topk, c.index_topk),
        (8, KEPT, 512)
    );
    assert_eq!((START + 1).div_ceil(8), KEPT);
    assert_eq!(CAPACITY.div_ceil(8), BLOCKS);
    assert!(
        CAPACITY.div_ceil(8) > c.candidate_topk,
        "entire batch follows real per-row fallback"
    );
    assert_eq!(s.kv_source_layers, [0, 2]);
    assert_eq!(s.index_source_layers, [0, 2, 3]);
    assert_eq!(s.candidate_source_layer, Some(0));
    assert_eq!(s.compression_ratios, [1; 4]);
    assert_eq!(
        selection_shapes(&c, CAPACITY, ROWS, 512).unwrap(),
        [32768, 8192]
    );
}

#[test]
fn mask_real_pipeline_cpu_bounds() {
    let (c, s) = geometry();
    let sizes = inventory(&c, &s);
    let cpu = RealCpuInventory::new(&sizes);
    let conservative_metal: u64 = sizes.iter().map(|&n| n as u64 * 4 + 65536).sum();
    assert!(conservative_metal + cpu.total() < REAL_LIMIT);
    assert!(
        cpu.seed_keys_values_and_upload_staging >= 4 * CAPACITY as u64 * c.index_dim as u64 * 4
    );
    assert!(
        REAL_LIMIT > super::LIMIT,
        "whole pipeline has its own bound, not the 64 MiB leaf envelope"
    );
}

#[test]
fn mask_real_pipeline_cpu_protocol() {
    assert_ne!(REAL_PROTOCOL, super::PROTOCOL);
    assert_eq!(ROUNDS, 3);
    let mut counts = [0; 2];
    for _ in 0..ROUNDS {
        for order in [[0, 1, 1, 0], [1, 0, 0, 1]] {
            for arm in order {
                counts[arm] += 1;
            }
        }
    }
    assert_eq!(counts, [12, 12]);
    assert_eq!(
        BATCH, 4,
        "leaf batch is deliberately NOT used by real span timing"
    );
}

#[test]
fn mask_real_pipeline_cpu_oracle_and_nonvacuous_controls() {
    let retained: Vec<u32> = (1..BLOCKS as u32).rev().collect();
    let mask = producer_oracle(&retained).unwrap();
    assert_eq!(mask, expected_mask());
    assert!(!matches(&vec![POISON; BLOCKS], &mask));
    assert_ne!(vec![1; BLOCKS], mask, "no-mask control cannot pass");
    let mut wrong = mask.clone();
    wrong[BLOCKS - 1] = 0;
    assert!(!matches(&wrong, &mask));
    let mut invalid = retained.clone();
    invalid[1] = BLOCKS as u32;
    assert!(producer_oracle(&invalid).is_none());
    invalid[1] = invalid[0];
    assert!(producer_oracle(&invalid).is_none());
    assert!(producer_oracle(&retained[..KEPT - 1]).is_none());
    let unmasked: Vec<_> = (0..512).collect();
    let masked: Vec<_> = (0..CAPACITY)
        .filter(|&id| mask[id / 8] == 1)
        .take(512)
        .collect();
    assert_eq!(masked, (8..520).collect::<Vec<_>>());
    assert_ne!(unmasked, masked);
    let keys = key_data(&geometry().0);
    assert!(keys.iter().all(|v| v.is_finite()));
    assert_eq!(keys[1], CAPACITY as f32);
}
