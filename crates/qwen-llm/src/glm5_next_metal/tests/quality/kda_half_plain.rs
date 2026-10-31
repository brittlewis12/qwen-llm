//! One-load ordinary latency only; external runner owns the real 240s deadline.
use super::*;

const LENGTHS: [usize; 4] = [512, 1024, 2048, 4096];
const ORDER: [bool; 8] = [false, true, true, false, true, false, false, true];
const WARM: [bool; 2] = [false, true];

struct Sample {
    wall_ms: f64,
    logits: Vec<f32>,
    position: usize,
    poisoned: bool,
    lineage: Option<PackedLineage>,
    precision: Option<FastPrecision>,
    ledger: Glm5NextMemoryLedger,
    allocation_delta: u64,
}

fn plain_reserve(vocab: u32) -> (Vec<(&'static str, u64)>, u64) {
    let row = u64::from(vocab) * 4;
    let terms = vec![
        ("eight_returned_logits_rows", 8 * row),
        ("two_live_logits_rows", 2 * row),
        (
            "eight_row_headers_and_session_records",
            8 * size_of::<Sample>() as u64,
        ),
        ("report_hashes_and_scalar_records", REPORT_BYTES),
        (
            "frozen_fixture_parse_peak_and_retained_tokens",
            FROZEN.len() as u64 * 64 + CAPACITY as u64 * 4,
        ),
    ];
    let cpu = cpu_reserve_total(&terms);
    (terms, cpu)
}

fn sample(
    ctx: &MetalContext,
    weights: &Glm5NextWeights,
    tokens: &[u32],
    cpu: u64,
    enabled: bool,
) -> Sample {
    let mut s =
        Glm5NextSession::with_prefill_rows_and_cpu_reserve(ctx, weights, CAPACITY, ROWS, cpu)
            .unwrap();
    let (result, wall_ms) = {
        let _stage = KdaHalfInputsScope::new(enabled);
        let started = Instant::now();
        let result = s.prefill_packed(ctx, tokens);
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        (result, wall_ms)
    };
    Sample {
        wall_ms,
        logits: result.unwrap(),
        position: s.position(),
        poisoned: s.is_poisoned(),
        lineage: s.packed_lineage(),
        precision: s.fast_precision(),
        ledger: *s.ledger(),
        allocation_delta: s.observed_allocation_delta(),
    }
}

fn check_samples(samples: &[Sample], tokens: usize, vocab: usize, ledger: Glm5NextMemoryLedger) {
    assert_plain();
    assert!(memory::kda_half_inputs_enabled());
    for sample in samples {
        assert!(sample.wall_ms.is_finite() && sample.wall_ms > 0.0);
        assert_eq!(sample.position, tokens);
        assert!(!sample.poisoned);
        assert_eq!(sample.lineage, Some(PackedLineage::Fast));
        assert_eq!(sample.precision, Some(FastPrecision::Half));
        assert_eq!(sample.ledger, ledger);
        check_logits(&sample.logits, vocab);
    }
}

fn contrasts(walls: &[f64; 8]) -> Value {
    assert!(walls.iter().all(|n| n.is_finite() && *n > 0.0));
    let (a0, b0) = ((walls[0] + walls[3]) / 2.0, (walls[1] + walls[2]) / 2.0);
    let (a1, b1) = ((walls[5] + walls[6]) / 2.0, (walls[4] + walls[7]) / 2.0);
    json!({"ABBA":{"a_mean_ms":a0,"b_mean_ms":b0,"b_over_a":b0/a0,"a_minus_b_ms":a0-b0},
        "BAAB":{"a_mean_ms":a1,"b_mean_ms":b1,"b_over_a":b1/a1,"a_minus_b_ms":a1-b1},
        "a_control_block_mean_drift_percent":(a1/a0-1.0)*100.0,
        "a_control_first_to_last_drift_percent":(walls[6]/walls[0]-1.0)*100.0,
        "descriptive_only":true,"performance_verdict":null})
}

#[test]
fn cpu_kda_half_plain_protocol_balances_eight_samples_and_prices_returned_rows() {
    let body = include_str!("kda_half_plain.rs")
        .split("\nfn m5_kda_half_one_load_plain_abba_baab() {")
        .nth(1)
        .unwrap();
    assert!(body.contains("crate::metal::acquire_metal_benchmark_lease()"));
    assert!(!body.contains("production_lease()"));
    assert!(body.contains("(\"MTL_DEBUG_LAYER\", \"0\")"));
    assert!(body.contains("(\"QWEN_METAL_LEASE_WAIT\", \"1\")"));
    assert_eq!(ORDER, [false, true, true, false, true, false, false, true]);
    for block in ORDER.chunks_exact(4) {
        assert_eq!(block.iter().filter(|&&b| b).count(), 2);
    }
    let positions = |enabled| {
        ORDER
            .iter()
            .enumerate()
            .filter(|(_, b)| **b == enabled)
            .map(|(i, _)| i)
            .sum::<usize>()
    };
    assert_eq!(positions(false), positions(true));
    assert_eq!(WARM, [false, true]);
    assert_eq!(
        (LENGTHS.len() * ORDER.len(), LENGTHS.len() * WARM.len()),
        (32, 8)
    );
    assert_eq!(LENGTHS, [512, 1024, 2048, 4096]);
    assert!(LENGTHS.iter().all(|&n| n <= CAPACITY));
    assert_eq!(chunks(4096), [2048, 2048]);
    let (terms, cpu) = plain_reserve(154880);
    assert_eq!(terms[0].1, 8 * 154880 * 4);
    assert_eq!(terms[1].1, 2 * 154880 * 4);
    assert_eq!(terms[2].1, 8 * size_of::<Sample>() as u64);
    assert_eq!(cpu, cpu_reserve_total(&terms));
    assert!(cpu < 64 << 20);
}

#[test]
fn cpu_kda_half_plain_contrasts_keep_both_blocks_and_control_drift() {
    let value = contrasts(&[10.0, 9.0, 9.0, 10.0, 9.9, 11.0, 11.0, 9.9]);
    for block in ["ABBA", "BAAB"] {
        assert!((value[block]["b_over_a"].as_f64().unwrap() - 0.9).abs() < 1e-12);
    }
    assert!(
        (value["a_control_block_mean_drift_percent"]
            .as_f64()
            .unwrap()
            - 10.0)
            .abs()
            < 1e-12
    );
    assert!(value["performance_verdict"].is_null());
    assert_eq!(value["descriptive_only"], true);
    assert!(std::panic::catch_unwind(|| contrasts(&[f64::NAN; 8])).is_err());
}

#[test]
#[ignore = "release actual Q4/M5 one-load ordinary latency; DEBUG=0, global KDA=1, production lease, fresh GLM53_KDA_HALF_PLAIN_REPORT; external <=240s deadline required"]
fn m5_kda_half_one_load_plain_abba_baab() {
    let invocation = Instant::now();
    assert!(!cfg!(debug_assertions));
    for (key, value) in [
        ("MTL_DEBUG_LAYER", "0"),
        ("QWEN_METAL_LEASE_WAIT", "1"),
        ("QWEN_GLM53_KDA_HALF_INPUTS", "1"),
        ("QWEN_MATMAT_Q6_K_N64_MIN_N", "1024"),
    ] {
        assert_eq!(
            std::env::var(key).as_deref(),
            Ok(value),
            "requires {key}={value}"
        );
    }
    assert_eq!(crate::metal::mat_mat_q6_k_n64_min_query(), 1024);
    assert!(crate::env_flag::read_default_on("QWEN_MATMAT_Q6_K_N64"));
    assert!(memory::kda_half_inputs_enabled());
    assert_plain();
    let _routed_off = TensorGateUpScope::new(false);
    let _lease = crate::metal::acquire_metal_benchmark_lease()
        .expect("production benchmark lease, validation OFF");
    let _dense = tensor_prefill::Scope::new(true);
    let _down = tensor_down::Scope::new(true);
    let _wide = tensor_prefill::Q8WideScope::new(true);
    let _q5 = tensor_q5_down::Scope::new(true);
    let mut out = Report::new("GLM53_KDA_HALF_PLAIN_REPORT", true);
    let docs = documents();
    let tokens = &docs[2].tokens[..4096];
    let gguf = GgufFile::open(std::env::var_os("GLM53_GGUF").unwrap()).expect("open pinned Q4");
    let header_sha = pin_q4_shards(&gguf);
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let artifact = Glm5NextArtifactLayout::inspect(&gguf).unwrap();
    assert!(artifact.packed_prefill());
    let c = artifact.config();
    assert_eq!(c.blocks.len(), 45);
    assert_eq!(c.leading_dense_block_count, 3);
    assert_metadata_pins(&gguf.tensors);
    assert!(tokens.iter().all(|&t| t < c.vocab_size));
    let (terms, cpu) = plain_reserve(c.vocab_size);
    let ctx = MetalContext::new().unwrap();
    assert!(ctx.device.name().to_string().contains("M5"));
    assert!(ctx.tensor_ops_status().is_available());
    assert!(crate::metal::moe_grouped_tensor_selected(&ctx));
    out.emit(json!({"event":"header","schema":"glm53.kda_half_one_load_plain.v1",
        "binary_sha256":binary_hash(),"test_source_sha256":sha256_hex(include_bytes!("kda_half_plain.rs")),
        "pin_helper_source_sha256":sha256_hex(include_bytes!("kda_half_inputs.rs")),
        "packed_source_sha256":sha256_hex(include_bytes!("../../packed.rs")),
        "stage_source_sha256":sha256_hex(include_bytes!("../../kda_half_inputs.rs")),
        "fixture_sha256":sha256_hex(FROZEN),"artifact_pin_sha256":sha256_hex(Q4_PIN),
        "metadata_shard_sha256":header_sha,"artifact_layout":artifact_layout(&gguf),
        "identity_scope":"pinned names/sizes and metadata SHA; tensor-table digest/stamps, not full payload hashes",
        "capacity":CAPACITY,"workspace_rows":ROWS,"half_backing_bytes_both_arms":HALF_BYTES,
        "prompt_lengths":LENGTHS,"order":ORDER,"warm_order":WARM,"arms":["CURRENT","KDA_HALF"],
        "qualified_q8_wide_and_q5_down":true,"routed_tensor_gate_up":false,"cpu_terms":terms,"cpu_reserve_bytes":cpu,
        "tokens_sha256":token_hash(tokens),"weight_loads":1,"plain_samples":32,"warm_samples":8,
        "external_deadline_seconds":240,"deadline_enforcement":"external runner; no in-flight interruption claimed",
        "timing":"ordinary prefill_packed including normal wait/readback only; fresh allocation/drop outside clock",
        "observers":false,"census":false,"profile":false,"captures":false,"performance_verdict":null}));
    let weights = load_admitted(&ctx, &gguf, artifact.model(), cpu, &mut out);
    let ledger = Glm5NextSession::plan_memory(&ctx, &weights, CAPACITY, ROWS).unwrap();
    assert!(ledger.scratch_options().kda_half_inputs);
    let specs = memory::packed_normalized_half_specs(c, ROWS as u64, ledger.scratch_options());
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].name, "normalized_half");
    assert_eq!(specs[0].shape, [4096, 2048]);
    assert_eq!(specs[0].bytes(), Some(HALF_BYTES));
    out.emit(json!({"event":"session_plan","ledger":ledger.terms(),"cpu_reserve_bytes":cpu,
        "half_backing_priced_upper_bytes":ctx.price_shared_buffer_upper(HALF_BYTES).unwrap().priced_upper_bytes}));
    let mut records = Vec::with_capacity(LENGTHS.len());
    for length in LENGTHS {
        assert_plain();
        let warm: Vec<_> = WARM
            .into_iter()
            .map(|enabled| sample(&ctx, &weights, &tokens[..length], cpu, enabled))
            .collect();
        check_samples(&warm, length, c.vocab_size as usize, ledger);
        let warm_walls = warm.iter().map(|s| s.wall_ms).collect::<Vec<_>>();
        drop(warm);
        let mut samples = Vec::with_capacity(ORDER.len());
        for enabled in ORDER {
            samples.push(sample(&ctx, &weights, &tokens[..length], cpu, enabled));
        }
        check_samples(&samples, length, c.vocab_size as usize, ledger);
        let walls: [f64; 8] = std::array::from_fn(|i| samples[i].wall_ms);
        let deltas = samples
            .iter()
            .map(|s| s.allocation_delta)
            .collect::<Vec<_>>();
        records.push(
            json!({"event":"shape","prompt_tokens":length,"prefill_chunks":chunks(length),
            "unchanged_short_control":length==512,"warm_wall_ms":warm_walls,"warm_order":WARM,
            "plain_wall_ms":walls,"order":ORDER,"allocation_deltas":deltas,"all_logits_finite":true,
            "contrasts":contrasts(&walls),"performance_verdict":null}),
        );
    }
    assert_plain();
    assert!(
        gguf.revalidate_retained_shard_stamps().unwrap() == stamps,
        "retained shard stamps changed"
    );
    for record in records {
        out.emit(record);
    }
    out.emit(
        json!({"event":"complete","execution_complete":true,"weight_loads":1,"warm_samples":8,
        "plain_samples":32,"sessions":40,"same_16mib_ledger":true,"observers_clear":true,
        "retained_shard_stamps_unchanged":true,"elapsed_seconds":invocation.elapsed().as_secs_f64(),
        "performance_verdict":null,"promotion_threshold":null}),
    );
    out.sync();
}
