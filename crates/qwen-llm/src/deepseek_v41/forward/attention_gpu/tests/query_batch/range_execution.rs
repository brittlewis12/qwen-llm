use super::*;
use crate::deepseek_v41::forward::attention_gpu::query_batch::{
    selection_shapes, with_old_host_loop,
};
use serde_json::json;
use std::time::Instant;

fn bindings(w: &Workspace) -> Vec<MetalTensor> {
    [
        &w.selected,
        &w.candidates,
        &w.latent,
        &w.gate,
        &w.index_query,
        &w.index_weights,
    ]
    .into_iter()
    .cloned()
    .collect()
}

fn assert_bindings(w: &Workspace, before: &[MetalTensor]) {
    for (a, b) in bindings(w).iter().zip(before) {
        assert!(std::ptr::eq(&*a.buffer, &*b.buffer));
        assert_eq!((a.offset, &a.shape, a.dtype), (b.offset, &b.shape, b.dtype));
    }
}

fn no_owner() -> Schedule {
    Schedule {
        layers: 1,
        kv_source_layers: vec![],
        index_source_layers: vec![],
        compression_ratios: vec![0],
        candidate_source_layer: None,
    }
}

fn parity_schedule() -> Schedule {
    Schedule {
        layers: 6,
        kv_source_layers: vec![1, 3],
        index_source_layers: vec![1, 3, 5],
        compression_ratios: vec![0, 2, 2, 1, 1, 1],
        candidate_source_layer: Some(3),
    }
}

#[test]
fn range_parity_schedule_keeps_ratio_two_and_ratio_one_candidate_ownership() {
    let schedule = parity_schedule();
    schedule.validate().unwrap();
    assert!(schedule.owns_kv(1) && schedule.ratio(1) == 2);
    assert_eq!(schedule.candidate_source_layer, Some(3));
    assert!(schedule.owns_kv(3) && schedule.ratio(3) == 1);
    assert!(schedule.produces_index(5) && !schedule.owns_kv(5));
}

fn host_geometry() -> Config {
    Config {
        hidden: 32,
        output_rank: 32,
        groups: 1,
        query_rank: 1280,
        heads: 64,
        head_dim: 512,
        rope_dim: 64,
        index_dim: 128,
        ..config()
    }
}

fn host_buffer_elements(c: &Config, n: usize) -> Vec<usize> {
    let mut sizes = shapes(c, 4096, &no_owner()).unwrap();
    sizes.extend(Rows::widths(c, 4096).into_iter().map(|width| width * n));
    sizes.extend(Scratch::shapes(c, n).unwrap());
    sizes.extend(
        selection_shapes(c, 4096, n, super::super::super::query_batch::tile_rows()).unwrap(),
    );
    sizes.extend([
        prefill::attention_half_scratch_elements(c, n).unwrap(),
        c.hidden * n,
        c.hidden * c.query_rank,
        c.query_rank * c.heads * c.head_dim,
        c.hidden * c.head_dim,
        c.heads * c.head_dim * c.output_rank,
        c.output_rank * c.hidden,
    ]);
    sizes.retain(|&n| n > 0);
    sizes
}

#[test]
fn host_discovery_geometry_and_conservative_peak_fit_four_gib() {
    let c = host_geometry();
    assert_eq!(
        (c.hidden, c.query_rank, c.heads, c.head_dim, c.rope_dim),
        (32, 1280, 64, 512, 64)
    );
    no_owner().validate().unwrap();
    for n in [128, 4096] {
        let sizes = host_buffer_elements(&c, n);
        assert!(sizes.iter().all(|&elements| elements > 0));
        let metal: u64 = sizes.into_iter().map(|n| n as u64 * 4 + 65536).sum();
        assert!(metal + (512 << 20) < 4 << 30);
    }
}

#[test]
#[ignore = "self-leased opt3 test-profile HOST discovery; no command submission/model/M5; <=4 GiB CPU+Metal; native validation flags absent"]
fn prepared_serial_host_encoding_discovery() {
    assert!(
        cfg!(debug_assertions),
        "use the repository opt3 test profile, not --release"
    );
    for key in [
        "MTL_DEBUG_LAYER",
        "MTL_SHADER_VALIDATION",
        "METAL_DEVICE_WRAPPER_TYPE",
        "MTL_CAPTURE_ENABLED",
        "MTL_SHADER_VALIDATION_ENABLE_PIPELINES",
        "METAL_FORCE_SHADER_VALIDATION",
        "DYLD_INSERT_LIBRARIES",
    ] {
        assert!(
            std::env::var_os(key).is_none(),
            "native host discovery requires {key} absent"
        );
    }
    let _lease = metal::acquire_metal_benchmark_lease().expect("production lease and wired guard");
    let ctx = MetalContext::new().unwrap();
    let c = host_geometry();
    let s = no_owner();
    let sizes = host_buffer_elements(&c, 4096);
    let priced: u64 = sizes
        .iter()
        .map(|&n| {
            ctx.price_shared_buffer_upper(n as u64 * 4)
                .unwrap()
                .priced_upper_bytes
        })
        .sum();
    let cpu = 512u64 << 20;
    assert!(priced + cpu <= 4 << 30);
    let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
        priced,
        cpu,
        0,
        ctx.memory_signals(),
        true,
    );
    assert!(
        admission.is_admitted(),
        "host discovery refused: {admission:?}"
    );
    let _transaction = ctx.begin_allocation_transaction();
    let mut weight = scheduled_weights(&ctx, &c, &s, 0, GgmlType::BF16);
    weight.frequencies =
        op::rotary_frequencies(c.rope_dim, 160000.0, 65536, 16.0, 32.0, 1.0).unwrap();
    let mut w = Workspace::new_with_schedule(&ctx, &c, 4096, &[&weight], &s).unwrap();
    let original = bindings(&w);
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    eprintln!(
        "{}",
        json!({"protocol":"v41-range-host-discovery-v1", "event":"begin", "profile":"opt3 test-profile HOST discovery, not canonical release/GPU/model timing", "debug_assertions":cfg!(debug_assertions), "optimization":"parent must build with repository profile.test opt-level=3", "priced_metal_bytes":priced, "cpu_upper_bytes":cpu, "limit_bytes":4u64<<30, "cases":2, "rows":[128,4096], "hidden":32, "query_rank":1280, "head_dim":512, "heads":64, "rope_dim":64, "output_rank":32, "groups":1, "dtype":"BF16", "schedule":"one ownerless/no-index layer", "gpu_preparation":false, "prepared_payload":"stable synthetic zero buffers, not consumed by command-free serial API", "submissions":0, "dispatches":0, "includes":"complete encode_rows_serial API including all arm-specific validation", "excludes":"context/weights/buffers/upload/begin_batch/census/assertions/reporting", "orders":["warm AB","ABBA","BAAB","ABBA","BAAB"], "samples_per_case":18, "speed_gate":false})
    );
    for n in [128, 4096] {
        let rows = Rows::new(&ctx, &c, 4096, n).unwrap();
        let input = prefill::allocate(&ctx, c.hidden, n, GgmlType::F32).unwrap();
        w.next_position = 0;
        w.begin_batch(0, n).unwrap();
        let initial = w.batch;
        let mut bad_shape = input.clone();
        bad_shape.shape = vec![c.hidden as u64];
        let mut bad_dtype = input.clone();
        bad_dtype.dtype = GgmlType::I32;
        let mut bad_bounds = input.clone();
        bad_bounds.offset = input.buffer.length() as u64;
        let mut bad_overflow = input.clone();
        bad_overflow.offset = u64::MAX - 3;
        metal::dispatch_census_begin();
        for bad in [&bad_shape, &bad_dtype, &bad_bounds, &bad_overflow] {
            assert!(
                w.encode_rows_serial(&ctx, &enc, 0, 0, &weight, bad, &rows)
                    .is_err()
            );
            assert_eq!(w.batch, initial);
            assert_eq!((w.next_layer, w.next_position), (0, 0));
            assert_bindings(&w, &original);
        }
        for (id, start) in [(1, 0), (0, 1)] {
            assert!(
                w.encode_rows_serial(&ctx, &enc, id, start, &weight, &input, &rows)
                    .is_err()
            );
            assert_eq!(w.batch, initial);
        }
        assert!(metal::dispatch_census_take().is_empty());
        w.batch = None;
        let mut sample = 0;
        for (pass, order) in ["AB", "ABBA", "BAAB", "ABBA", "BAAB"]
            .into_iter()
            .enumerate()
        {
            for arm in order.chars() {
                w.next_position = 0;
                w.begin_batch(0, n).unwrap();
                metal::dispatch_census_begin();
                metal::allocation_census_begin();
                let mut run = || {
                    let clock = Instant::now();
                    let result = w.encode_rows_serial(&ctx, &enc, 0, 0, &weight, &input, &rows);
                    let ns = clock.elapsed().as_nanos();
                    result.unwrap();
                    ns
                };
                let ns = if arm == 'A' {
                    with_old_host_loop(run)
                } else {
                    run()
                };
                assert!(metal::dispatch_census_take().is_empty());
                assert!(metal::allocation_census_take().is_empty());
                assert_bindings(&w, &original);
                assert!(w.batch.is_none() && w.ced.is_none());
                assert_eq!((w.next_layer, w.next_position), (0, n));
                eprintln!(
                    "{}",
                    json!({"protocol":"v41-range-host-discovery-v1", "event":"sample", "rows":n, "chronological_sample":sample, "pass":pass, "order":order, "arm":if arm=='A' {"old-host-loop"} else {"range"}, "warmup":pass==0, "host_ns":ns, "dispatches":0, "bindings_unchanged":true, "cursor":{"batch":false,"next_layer":0,"next_position":n}, "gpu_completed":false})
                );
                sample += 1;
            }
        }
        assert_eq!(sample, 18);
    }
    enc.end();
    // Deliberately drop the sole uncommitted command; no fake GPU completion.
}

fn payload(t: &MetalTensor) -> Vec<u32> {
    assert_eq!(t.offset % 4, 0);
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
#[ignore = "self-leased bounded BF16 head512/index128 range-vs-old host GPU parity; MTL_DEBUG_LAYER=1; <=256 MiB"]
fn prepared_serial_range_old_host_bitexact_outputs_controls_pending_and_census() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert!(std::env::var_os("QWEN_V41_SELECT_PER_ROW").is_none());
    assert!(crate::env_flag::read_default_on("QWEN_V41_COMPRESS_ROWS"));
    let _lease = metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let c = Config {
        heads: 16,
        head_dim: 512,
        index_dim: 128,
        rope_dim: 64,
        ..config()
    };
    let s = parity_schedule();
    s.validate().unwrap();
    // All-selected, scored top-k, pruned scalar selection, odd chunk head/body/tail.
    for (start, n) in [(0, 3), (3, 4), (7, 4), (17, 4)] {
        let capacity = start + n;
        let sizes = shapes(&c, capacity, &s).unwrap();
        let conservative =
            sizes.into_iter().map(|n| n as u64 * 4 + 65536).sum::<u64>() * 2 + (128 << 20);
        assert!(conservative < 256 << 20);
        assert!(
            metal::evaluate_metal_memory_admission_with_cpu_bytes(
                conservative,
                32 << 20,
                0,
                ctx.memory_signals(),
                true
            )
            .is_admitted()
        );
        let weights: Vec<_> = (0..s.layers)
            .map(|id| {
                let mut w = scheduled_weights(&ctx, &c, &s, id, GgmlType::BF16);
                w.frequencies =
                    op::rotary_frequencies(c.rope_dim, 160000.0, 65536, 16.0, 32.0, 1.0).unwrap();
                w
            })
            .collect();
        let refs = weights.iter().collect::<Vec<_>>();
        let mut old = Workspace::new_with_schedule(&ctx, &c, capacity, &refs, &s).unwrap();
        let mut new = Workspace::new_with_schedule(&ctx, &c, capacity, &refs, &s).unwrap();
        seed(&ctx, &mut old, start);
        seed(&ctx, &mut new, start);
        let old_rows = Rows::new(&ctx, &c, capacity, n).unwrap();
        let new_rows = Rows::new(&ctx, &c, capacity, n).unwrap();
        let input = upload(
            &ctx,
            &values(c.hidden * n, 11),
            vec![c.hidden as u64, n as u64],
        )
        .unwrap();
        let output = prefill::allocate(&ctx, c.hidden, n, GgmlType::F32).unwrap();
        old.begin_batch(start, n).unwrap();
        new.begin_batch(start, n).unwrap();
        for (id, weight) in weights.iter().enumerate() {
            let mut snapshots = Vec::new();
            let mut launches = Vec::new();
            for (old_loop, w, rows) in [(true, &mut old, &old_rows), (false, &mut new, &new_rows)] {
                let before = bindings(w);
                metal::dispatch_census_begin();
                execute(&ctx, |enc| {
                    let mut run = || {
                        w.encode_rows(&ctx, enc, id, start, weight, &input, rows)
                            .unwrap();
                        w.finish_rows(&ctx, enc, weight, rows, &output).unwrap();
                    };
                    if old_loop {
                        with_old_host_loop(run);
                    } else {
                        run();
                    }
                });
                launches.push(format!("{:?}", metal::dispatch_census_take()));
                assert_bindings(w, &before);
                assert_eq!(ids(w.status(), 1), [0]);
                let mut snapshot = vec![
                    payload(&output),
                    payload(&rows.query),
                    payload(&rows.selected),
                    payload(&rows.candidates),
                    payload(&w.raw),
                    payload(&w.status),
                ];
                for owner in &w.owners {
                    snapshot.extend(
                        [
                            &owner.compressed,
                            &owner.keys,
                            &owner.pending_value,
                            &owner.pending_gate,
                        ]
                        .into_iter()
                        .map(payload),
                    );
                }
                snapshots.push(snapshot);
            }
            assert_eq!(snapshots[0], snapshots[1], "start={start} n={n} layer={id}");
            assert_eq!(launches[0], launches[1], "ordered full launch census");
            assert_eq!(old.batch, new.batch);
            assert_eq!(
                (old.next_layer, old.next_position),
                (new.next_layer, new.next_position)
            );
        }
        assert!(new.batch.is_none());
        assert_eq!((new.next_layer, new.next_position), (0, start + n));
    }
}
