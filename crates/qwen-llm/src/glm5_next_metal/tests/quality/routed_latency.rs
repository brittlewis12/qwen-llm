//! DEBUG-OFF one-load heated routed gate/up timing, separate from quality.
use super::*;

struct Sample {
    logits: Vec<f32>,
    wall_ms: f64,
    ledger: Glm5NextMemoryLedger,
    allocated: u64,
    position: usize,
    poisoned: bool,
    lineage: Option<PackedLineage>,
    precision: Option<FastPrecision>,
}

fn run(loaded: &Loaded, tokens: &[u32], candidate: bool) -> Sample {
    let mut session = loaded.fresh();
    let (logits, wall_ms) = selected(candidate, || {
        let start = Instant::now();
        let logits = session.prefill_packed(&loaded.ctx, tokens).unwrap();
        (logits, start.elapsed().as_secs_f64() * 1e3)
    });
    Sample {
        logits,
        wall_ms,
        ledger: *session.ledger(),
        allocated: session.observed_allocation_delta(),
        position: session.position(),
        poisoned: session.is_poisoned(),
        lineage: session.packed_lineage(),
        precision: session.fast_precision(),
    }
}

fn validate(loaded: &Loaded, sample: &Sample, n: usize) {
    assert_eq!(sample.ledger, loaded.ledger);
    assert_eq!(sample.position, n);
    assert!(!sample.poisoned);
    assert_eq!(sample.lineage, Some(PackedLineage::Fast));
    assert_eq!(sample.precision, Some(FastPrecision::Half));
    assert!(sample.wall_ms.is_finite() && sample.wall_ms > 0.0);
    check_logits(&sample.logits, loaded.weights.config.vocab_size as usize);
}

fn contrasts(walls: &[f64; 8]) -> Value {
    assert!(walls.iter().all(|n| n.is_finite() && *n > 0.0));
    let current = (walls[0] + walls[3] + walls[5] + walls[6]) / 4.0;
    let candidate = (walls[1] + walls[2] + walls[4] + walls[7]) / 4.0;
    let abba_current = (walls[0] + walls[3]) * 0.5;
    let abba_candidate = (walls[1] + walls[2]) * 0.5;
    let baab_current = (walls[5] + walls[6]) * 0.5;
    let baab_candidate = (walls[4] + walls[7]) * 0.5;
    json!({"current_mean_ms":current,"candidate_mean_ms":candidate,"candidate_over_current":candidate/current,
        "latency_change_fraction":candidate/current-1.0,"abba_candidate_over_current":abba_candidate/abba_current,
        "baab_candidate_over_current":baab_candidate/baab_current,"current_second_over_first":baab_current/abba_current,
        "candidate_second_over_first":baab_candidate/abba_candidate,"policy":"descriptive paired blocks and drift only; no pass threshold"})
}

#[test]
#[ignore = "release M5 DEBUG-OFF, production lease, fresh GLM53_ROUTED_HEATED_REPORT; external fresh 240s wall limit"]
fn m5_routed_heated_plain_latency() {
    environment_for(false);
    let _lease = crate::metal::acquire_metal_benchmark_lease()
        .expect("production benchmark lease, validation OFF");
    let mut out = Report::new("GLM53_ROUTED_HEATED_REPORT", true);
    let docs = documents();
    let long = &docs.iter().find(|d| d.stratum == "long").unwrap().tokens;
    let loaded = Loaded::load(false, &mut out);
    assert_eq!(loaded.capacity, LATENCY_CAPACITY);
    assert!(long.iter().all(|&id| id < loaded.weights.config.vocab_size));
    out.emit(json!({"event":"protocol","shapes":[512,1024,2048,4096],"samples_per_shape":8,"sample_order":ORDER,
        "warm_prefills_per_shape":2,"one_weight_load":true,"retained_logits_rows":8,"workspace_rows":2048,"capacity":4416,
        "scope":"ordinary prefill full wait/readback only; fresh admitted session allocation/drop outside each timer; no observers/capture/interventions; checks, hashes and JSON after each eight-sample block; 512 is unchanged dispatch control"}));
    let mut completed = 0;
    for n in [512, 1024, 2048, 4096] {
        assert_plain();
        let warm = [
            run(&loaded, &long[..n], false),
            run(&loaded, &long[..n], true),
        ];
        for sample in &warm {
            validate(&loaded, sample, n);
        }
        assert_eq!(warm[0].allocated, warm[1].allocated);
        out.emit(json!({"event":"warm_complete","rows":n,"arms":2,"same_allocated_bytes":warm[0].allocated,"logits_finite":true}));
        drop(warm);
        assert_plain();
        // No diagnostics, validation, hashing or report work within the block.
        let samples: [Sample; 8] = ORDER.map(|candidate| run(&loaded, &long[..n], candidate));
        assert_plain();
        for sample in &samples {
            validate(&loaded, sample, n);
            assert_eq!(sample.allocated, samples[0].allocated);
        }
        if n == 512 {
            for sample in &samples[1..] {
                assert!(
                    sample
                        .logits
                        .iter()
                        .zip(&samples[0].logits)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "512-row dispatch control must remain unchanged"
                );
            }
        }
        let walls = samples.each_ref().map(|sample| sample.wall_ms);
        for (index, sample) in samples.iter().enumerate() {
            out.emit(json!({"event":"plain_sample","rows":n,"sample":index,"arm":LABELS[usize::from(ORDER[index])],
                "block":if index<4 {"ABBA"} else {"BAAB"},"wall_ms":sample.wall_ms,"logits_sha256":sha256_hex(bytemuck::cast_slice(&sample.logits)),
                "logits_finite":true,"position":sample.position,"allocated_bytes":sample.allocated}));
        }
        out.emit(json!({"event":"shape_complete","rows":n,"samples":8,"dispatch_control":n==512,"contrasts":contrasts(&walls)}));
        completed += 8;
    }
    assert_eq!(completed, 32);
    loaded.finish();
    out.emit(json!({"event":"complete","execution_complete":true,"samples":32,"warm_prefills":8,"sessions":40,
        "weight_loads":1,"retained_stamps_unchanged":true,"observers_clear":true,"performance_verdict":null}));
    out.sync();
}

#[test]
fn cpu_routed_heated_order_contrasts_and_storage_contract() {
    let body = include_str!("routed_latency.rs")
        .split("\nfn m5_routed_heated_plain_latency() {")
        .nth(1)
        .unwrap()
        .split("#[test]")
        .next()
        .unwrap();
    assert!(body.contains("crate::metal::acquire_metal_benchmark_lease()"));
    assert!(!body.contains("production_lease()"));
    assert!(body.contains("environment_for(false)"));
    assert_eq!(ORDER, [false, true, true, false, true, false, false, true]);
    assert_eq!(LATENCY_CAPACITY, 4416);
    let value = contrasts(&[4.0, 2.0, 2.0, 4.0, 2.0, 4.0, 4.0, 2.0]);
    assert_eq!(value["candidate_over_current"], 0.5);
    assert_eq!(value["current_second_over_first"], 1.0);
    assert_eq!(value["candidate_second_over_first"], 1.0);
    assert!(std::panic::catch_unwind(|| contrasts(&[f64::NAN; 8])).is_err());
    assert!(std::panic::catch_unwind(|| contrasts(&[0.0; 8])).is_err());
    let terms = reserve(154880, false);
    assert_eq!(terms[0].1, 10 * 154880 * 4);
}
